"""Numerical and standalone regression tests for exact pair velocity inference."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5netcdf
import h5py
import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad

from glacier_offsets.velocity_basis import (PERIOD, EPOCH_SECONDS, utc_days, harmonic, definition,
                                            background_columns, design, evaluate)
from glacier_offsets.velocity_fit import (fit, diagnostics, whiten, calibrate_noise, covariance_ra, _solve,
                                          blocked_folds, false_discovery, quality_rule)
from glacier_offsets.velocity_io import (Collection, save_fit, evaluate_file, variable_hashes,
                                         write_json, writer_lock, digest, export_daily, fingerprint)
from glacier_offsets.velocity_inspect import alias_diagnostics, sample_points
from glacier_offsets.velocity_pilot import (candidate_bank, fit_point, learn_noise, select_models,
                                           paired_bootstrap, acceptance, _noise_draw, horizontal_azimuth)
from glacier_offsets.velocity_recovery import (load_settings, reviewed_regions, map_stage, calibration_gate, pilot, run_recipe)
from glacier_offsets.velocity_recovery import PilotProgress
from glacier_offsets.velocity_calibration import compare_calibrations
from tools.prepare_velocity_experiment import freeze, extended_config, estimate_extension_capacity
from tools.validate_existing_velocity_calibration import validate_fixed


def make_input(path, early=None, late=None, offset=0, shape=(3,4)):
    early = np.arange(0.,720,6)+6209.25 if early is None else np.asarray(early)
    late = early+6 if late is None else np.asarray(late)
    count = len(early)
    basis = definition(early,late,'constant')
    matrix = design(early,late,basis,'free_en',[1,0])[0]
    coefficients = np.array([1.,.2,.08,.06,.016,.012])
    data = np.einsum('ijk,k->ij',matrix,coefficients)
    with h5netcdf.File(path,'w') as h:
        h.dimensions = {'time':count,'y':shape[0],'x':shape[1],'enu':3,'burst':2}
        h.attrs.update(schema_version='glacier-offsets-displacements-v1',package_status='complete',EPSG=3031,
                       package_identity='fixture',
                       geometry_identity='reference',layout_identity='layout',geometry_reference_date='2017-01-01',
                       geocoding_resampling='nearest',horizontal_vector_basis='local geographic east/north; physical metres')
        def var(name,dims,data,**attrs):
            data = np.asarray(data)
            string = data.dtype.kind in 'USO'
            v = h.create_variable(name,dims,dtype=h5py.string_dtype() if string else data.dtype)
            v[...] = data
            v.attrs.update(attrs)
            return v
        var('x',('x',),np.arange(shape[1])*4000.+100000)
        var('y',('y',),200000-np.arange(shape[0])*4000.)
        var('crs',(),np.int32(0),GeoTransform='98000 4000 0 202000 0 -4000')
        var('pair_id',('time',),[f'pair_{offset+i:03}' for i in range(count)])
        var('burst_id',('burst',),['burst_iw1','burst_iw2'])
        var('enu',('enu',),['east','north','up'])
        owner = np.ones(shape,dtype='i4');owner[:,-1] = 2
        association = np.arange(np.prod(shape),dtype='i8').reshape(shape)
        association[0,1] = association[0,0] # copied measurement
        var('source_burst_index',('y','x'),owner)
        var('source_observation_index',('y','x'),association)
        for name,vector in [('range_sensitivity_enu',[-.5,.05,-.86]),('azimuth_sensitivity_enu',[.1,.99,.003])]:
            var(name,('enu','y','x'),np.broadcast_to(np.array(vector)[:,None,None],(3,*shape)))
        for name in ('reference_center_line','reference_center_sample'):
            var(name,('y','x'),np.ones(shape))
        for name,values in [('earlier_burst_time',early),('later_burst_time',late)]:
            times = np.tile((values*86400+EPOCH_SECONDS)[:,None],(1,2))
            times[:,1] += .7
            var(name,('time','burst'),times,units='seconds since 1970-01-01 00:00:00',calendar='proleptic_gregorian')
        for axis,name in enumerate(('east_displacement_m','north_displacement_m')):
            var(name,('time','y','x'),np.broadcast_to(data[:,axis,None,None],(count,*shape)),units='m')
        var('horizontal_valid_mask',('time','y','x'),np.ones((count,*shape),dtype='u1'))
        var('horizontal_geometry_valid_mask',('y','x'),np.ones(shape,dtype='u1'))
        var('snr',('time','y','x'),np.full((count,*shape),25.))
        var('correlation_peak',('time','y','x'),np.full((count,*shape),.5))
        h.attrs['content_hashes_json'] = json.dumps(variable_hashes(h))
    return basis,coefficients


def rehash(path):
    with h5netcdf.File(path,'a') as h:
        h.attrs['content_hashes_json'] = json.dumps(variable_hashes(h))


class BasisTests(unittest.TestCase):
    def test_controlled_flow_angles_match_reference_north_clockwise_convention(self):
        assert_allclose(horizontal_azimuth(0),[0,1],atol=1e-15)
        assert_allclose(horizontal_azimuth(90),[1,0],atol=1e-15)
        assert_allclose(horizontal_azimuth(105),[np.cos(np.deg2rad(15)),-np.sin(np.deg2rad(15))],atol=1e-15)

    @unittest.skipUnless(importlib.util.find_spec('cvxopt'),'ice convex solver unavailable')
    def test_l1_dual_solver_recovers_when_the_primal_gram_solve_fails(self):
        from cvxopt import solvers
        rng=np.random.default_rng(93)
        early=np.arange(60)*6.+6209.3;late=early+6
        a,_,indices=design(early,late,definition(early,late,'spline_l1'),'free_en',[1,0])
        a=a.reshape(-1,a.shape[-1]);y=rng.normal(size=len(a))+a[:,0]
        original=solvers.qp
        def fail_primal(p,*args,**kwargs):
            if p.size[0]==2*len(indices):
                raise ValueError('Simulated singular Gram solve')
            return original(p,*args,**kwargs)
        with patch('cvxopt.solvers.qp',side_effect=fail_primal):
            coefficients,scale=_solve(a,y,indices,.1,True)
        residual=y-a@coefficients;x=a/scale;beta=coefficients*scale
        dual=residual*min(1,.1/max(abs(x[:,indices].T@residual)))
        primal=.5*residual@residual+.1*np.sum(abs(beta[indices]))
        self.assertLess(primal-(y@dual-.5*dual@dual),1e-7*max(1,primal))

    @unittest.skipUnless(importlib.util.find_spec('cvxopt'),'ice convex solver unavailable')
    def test_weak_l1_penalties_have_a_feasible_dual_certificate(self):
        rng=np.random.default_rng(55)
        early=np.arange(60)*6.+6209.3;late=early+6
        basis=definition(early,late,'spline_l1')
        a,_,indices=design(early,late,basis,'free_en',[1,0]);a=a.reshape(-1,a.shape[-1])
        y=rng.normal(size=len(a))+a[:,0]+a[:,1]
        for penalty in (1e-4,.01,1.,100.):
            coefficients,scale=_solve(a,y,indices,penalty,True)
            x=a/scale;beta=coefficients*scale;residual=y-a @ coefficients
            unpenalized=np.setdiff1d(np.arange(len(beta)),indices)
            assert_allclose(x[:,unpenalized].T @ residual,0,atol=1e-8)
            dual=residual*min(1,penalty/max(np.max(abs(x[:,indices].T @ residual)),1e-30))
            self.assertLessEqual(np.max(abs(x[:,indices].T @ dual)),penalty*(1+1e-9))
            primal=.5*residual @ residual+penalty*np.sum(abs(beta[indices]))
            gap=primal-(y @ dual-.5*dual @ dual)
            self.assertLess(gap,1e-7*max(1,primal))

    def test_exact_harmonic_quadrature_and_attenuation(self):
        for duration in (6,12,.001):
            early,late = 6251.341,6251.341+duration
            actual = harmonic([early],[late])[0]
            expected = [quad(lambda t: fun(2*np.pi*t/PERIOD),early,late,epsabs=1e-10)[0] for fun in (np.cos,np.sin)]
            assert_allclose(actual,expected,atol=1e-11)
            self.assertAlmostEqual(np.linalg.norm(actual)/duration,np.sinc(duration/PERIOD),places=9)

    def test_utc_fixed_origin_offsets_and_leap_day(self):
        assert_allclose(utc_days(['2000-01-01Z','2000-01-01T08:00:00+08:00','2000-03-01T00:00:00Z']),[0,0,60])
        self.assertAlmostEqual(float(utc_days('2017-01-01T00:00:00.5Z')-utc_days('2017-01-01T00:00:00Z')),.5/86400,places=11)

    def test_spline_primitive_and_velocity_derivative(self):
        early,late = np.arange(0,720,6.),np.arange(0,720,6.)+6
        l2,l1 = (definition(early,late,key) for key in ('spline_l2','spline_l1'))
        self.assertEqual(l1['splines'],l2['splines'])
        days = np.linspace(1,719,70)
        eps = 1e-3
        derivative = (background_columns(days+eps,l2,displacement=True)-background_columns(days-eps,l2,displacement=True))/(2*eps)
        assert_allclose(derivative,background_columns(days,l2),atol=1e-7)
        for j in (2,5,12):
            integrated = background_columns([360.],l2,displacement=True)[0,j]-background_columns([180.],l2,displacement=True)[0,j]
            expected = quad(lambda t:background_columns([t],l2)[0,j],180,360,epsabs=1e-8)[0]
            self.assertAlmostEqual(integrated,expected,places=6)

    def test_known_vector_signal_and_evaluation(self):
        early = np.arange(0,720,6.)+6209.7
        late = early+6
        basis = definition(early,late,'linear')
        m = np.array([1.,.2,.0001,-.0002,.08,.06,-.025,.01])
        g = design(early,late,basis,'free_en',[1,0])[0]
        d = np.einsum('ijk,k->ij',g,m)
        result = fit(early,late,d,np.tile(np.eye(2)*.01,(len(early),1,1)),basis,direction=[1,0])
        assert_allclose(result['coefficients'],m,atol=1e-12)
        diag = diagnostics(result)
        self.assertAlmostEqual(diag['amplitude'],.1)
        self.assertAlmostEqual(diag['phase'],np.arctan2(.06,.08))
        for j in (0,27):
            expected = [quad(lambda t:evaluate(m,basis,'free_en',[1,0],np.array([t]))[0,axis],early[j],late[j])[0] for axis in range(2)]
            assert_allclose(expected,d[j],atol=1e-9)

    def test_l1_l2_leave_target_and_secular_unpenalized(self):
        early=np.arange(0,720,6.);late=early+6
        for background in ('spline_l1','spline_l2'):
            basis=definition(early,late,background)
            g,_,indices=design(early,late,basis,'along_flow',[1,0])
            self.assertFalse(np.isin([0,1,2,3,g.shape[-1]-2,g.shape[-1]-1],indices).any())
            m=np.zeros(g.shape[-1]);m[:4]=[1,.2,.0001,0];m[-2:]=[.08,.06]
            result=fit(early,late,np.einsum('ijk,k->ij',g,m),np.tile(np.eye(2),(len(early),1,1)),basis,
                       'along_flow',[1,0],penalty=1e4)
            assert_allclose(result['coefficients'][-2:],m[-2:],atol=1e-7)
            self.assertEqual(result['target_rank'],2)


class StatisticsTests(unittest.TestCase):
    def test_resampled_noise_is_standardized_at_control_quality(self):
        rng=np.random.default_rng(654)
        n=600
        noise={'covariance_ra_m2':np.eye(2).tolist(),'floors_m':[.01,.01],'correlation':0.,
               'quality_models':{'snr':{'coefficients':[[0.,-1.],[0.,-1.]],'domain':[[0.],[np.log(100.)]]}}}
        control={'snr':np.full(n,100.),'peak':np.full(n,np.nan),'early':np.arange(n)*6.,'late':np.arange(n)*6.+6}
        residual={'point':control,'valid':np.ones(n,bool),'residual_ra':rng.normal(size=(n,2))*.1}
        target=control|{'snr':np.ones(n),'inverse':np.eye(2)}
        draws=np.concatenate([_noise_draw(rng,target,noise,[residual]) for _ in range(20)])
        assert_allclose(np.std(draws,axis=0),[1.,1.],atol=.1)
        plain=np.random.default_rng(83);cached=np.random.default_rng(83);cache={}
        for _ in range(10):
            assert_allclose(_noise_draw(plain,target,noise,[residual]),
                            _noise_draw(cached,target,noise,[residual],cache=cache),rtol=0,atol=0)

    def test_fixed_calibration_is_checked_on_unseen_dates_and_purged_bursts(self):
        rng=np.random.default_rng(25)
        dates=[f'2017-01-{i:02d}' for i in range(1,32)]
        missions={d:'S1B' if i%2 else 'S1A' for i,d in enumerate(dates)}
        rows=[]
        for a,b in zip(dates,dates[1:]):
            x=int(missions[b]=='S1B')-int(missions[a]=='S1B')
            for g in range(4):
                rows.append({'pair':a+'_'+b,'earlier':a,'later':b,'burst1':f'a{g}','burst2':f'b{g}',
                             'mean':.03*x+rng.normal(0,.001)})
        cal={'beta_px':.03,'provenance':{'training_dates':dates[:10]}}
        report=validate_fixed(rows,dates,missions,cal)
        self.assertTrue(report['passed'])
        training_dates=set(dates[:10])
        self.assertTrue(all(not training_dates.intersection((rows[i]['earlier'],rows[i]['later'])) for i in report['temporal_test_rows']))
        for fold in report['held_out_burst_groups']['folds']:
            self.assertFalse(set(fold['test_group'])&set(fold['training_bursts']))
        self.assertFalse(validate_fixed(rows,dates,missions,cal|{'beta_px':.2})['passed'])

    def test_split_year_calibration_and_heldout_burst_purging(self):
        rng=np.random.default_rng(1)
        dates=[f'{year}-01-{i:02d}' for year in (2017,2018) for i in range(1,17)]
        missions={d:'S1B' if i%2 else 'S1A' for i,d in enumerate(dates)}
        def rows_for(second_beta):
            rows=[]
            for year in (2017,2018):
                ds=[d for d in dates if d.startswith(str(year))]
                for a,b in zip(ds,ds[1:]):
                    delta=int(missions[b]=='S1B')-int(missions[a]=='S1B')
                    for i in range(4):
                        rows.append(dict(pair=a+'_'+b,earlier=a,later=b,burst1=f'a{i}',burst2=f'b{i}',
                                         mean=(.03 if year==2017 else second_beta)*delta+rng.normal(0,.001)))
            return rows
        rows=rows_for(.03)
        result=compare_calibrations(rows,dates,missions)
        self.assertTrue(result['shared_calibration_supported'])
        for fold in result['folds']:
            if fold['kind']=='leave_burst_group_out':
                training={rows[i][k] for i in fold['train_rows'] for k in ('burst1','burst2')}
                testing={rows[i][k] for i in fold['test_rows'] for k in ('burst1','burst2')}
                self.assertFalse(training&testing)
        changed=compare_calibrations(rows_for(.08),dates,missions)
        self.assertFalse(changed['shared_calibration_supported'])
        self.assertTrue(changed['split_year_estimates']['2018']['independently_supported'])

    def test_full_covariance_whitening(self):
        rng=np.random.default_rng(4)
        a=rng.normal(size=(20,2,6));y=rng.normal(size=(20,2))
        covariance=np.tile(np.array([[4,1.2],[1.2,1.]]),(20,1,1))
        aw,yw=whiten(a,y,covariance)
        assert_allclose(np.sum(yw*yw),np.einsum('ni,nij,nj->',y,np.linalg.inv(covariance),y))
        assert_allclose(np.einsum('nip,niq->pq',aw,aw),np.einsum('nip,nij,njq->pq',a,np.linalg.inv(covariance),a))

    def test_noise_quality_monotonic_fallback_and_weight_cap(self):
        rng=np.random.default_rng(6);n=3000
        snr=rng.uniform(5,50,n);peak=rng.uniform(.1,.9,n)
        sigma=.2+2/np.sqrt(snr*peak)
        r=rng.normal(size=(n,2))*sigma[:,None]
        noise=calibrate_noise(r,snr,peak,np.arange(n)%20)
        self.assertLessEqual(np.max(np.array(noise['quality_models']['snr_peak']['coefficients'])[:,1:]),0)
        cov=covariance_ra(noise,np.array([10.,40.,-1.,200.,20.]),np.array([.5,.5,.5,.5,np.nan]),True)
        self.assertTrue((np.diag(cov[1])<=np.diag(cov[0])).all())
        assert_allclose(cov[2],noise['covariance_ra_m2']);assert_allclose(cov[3],noise['covariance_ra_m2'])
        self.assertTrue(np.isfinite(cov[4]).all())
        baseline=np.array(noise['covariance_ra_m2'])
        for candidate in cov:
            self.assertGreaterEqual(np.linalg.eigvalsh(candidate-.01*baseline).min(),-1e-10)
        self.assertTrue(quality_rule(np.array([25.,9.,9.]),np.array([np.nan,.4,.1])).tolist()==[True,True,False])

    def test_huber_pair_weight_has_square_root_semantics(self):
        early=np.arange(0,360,6.);late=early+6;basis=definition(early,late)
        data=(late-early)[:,None]*np.array([1.,.2])+harmonic(early,late)@np.array([[.1,0],[0,0]])
        data[4]+=100
        covariance=np.tile(np.eye(2)*.1,(len(early),1,1))
        plain=fit(early,late,data,covariance,basis,direction=[1,0])
        robust=fit(early,late,data,covariance,basis,direction=[1,0],robust=True)
        self.assertLess(robust['robust_factors'][4],.02)
        self.assertLess(abs(diagnostics(robust)['amplitude']-.1),abs(diagnostics(plain)['amplitude']-.1))
        matrix=design(early,late,basis,'free_en',[1,0])[0]
        aw,yw=whiten(matrix,data,covariance)
        f=np.sqrt(robust['robust_factors'])
        expected=np.linalg.lstsq((aw*f[:,None,None]).reshape(-1,matrix.shape[-1]),(yw*f[:,None]).ravel(),rcond=None)[0]
        assert_allclose(robust['coefficients'],expected,atol=1e-7)

    def test_free_en_detects_cross_flow_without_claiming_along_phase(self):
        early=np.arange(0,360,6.);late=early+6;basis=definition(early,late)
        matrix=design(early,late,basis,'free_en',[1,0])[0]
        coefficients=np.array([1.,.2,0,0,.1,.03])
        data=np.einsum('ijk,k->ij',matrix,coefficients)
        result=fit(early,late,data,np.tile(np.eye(2)*.001,(len(early),1,1)),basis,direction=[1,0])
        diag=diagnostics(result)
        self.assertLess(diag['p_value'],1e-20)
        self.assertGreater(diag['along_p_value'],.99)
        self.assertTrue(np.isinf(diag['phase_std']))
        self.assertAlmostEqual(diag['ellipse_major'],np.hypot(.1,.03))


    def test_acquisition_purged_folds(self):
        early=np.arange(0,720,6.);late=early+6
        for train,test in blocked_folds(early,late):
            held=np.r_[early[test],late[test]]
            self.assertFalse(np.isin(early[train],held).any())
            self.assertFalse(np.isin(late[train],held).any())
            self.assertTrue(test.any())

    def test_alias_information_and_target_rank(self):
        early=np.arange(0,720,6.)+.25;late=early+6
        d=alias_diagnostics(early,late)
        self.assertGreater(d['Msf_M2_condition'],1e4)
        self.assertEqual(d['viewing_geometries_at_this_point'],1)
        early=np.arange(12.)*PERIOD;late=early+PERIOD
        basis=definition(early,late)
        result=fit(early,late,np.ones((len(early),2)),np.tile(np.eye(2),(len(early),1,1)),basis,direction=[1,0])
        self.assertEqual(result['target_rank'],0)
        self.assertFalse(diagnostics(result)['identifiable'])

    def test_null_false_positives_and_95_interval_coverage(self):
        rng=np.random.default_rng(12345)
        early=np.arange(0,360,6.)+.333;late=early+6;basis=definition(early,late)
        cov=np.tile(np.array([[.04,.01],[.01,.09]]),(len(early),1,1))
        detections=0;covered=0
        for _ in range(1000):
            noise=np.einsum('nij,nj->ni',np.linalg.cholesky(cov),rng.normal(size=(len(early),2)))
            result=fit(early,late,6*np.array([1.,.2])+noise,cov,basis,'along_flow',[1,0])
            d=diagnostics(result)
            detections+=d['p_value']<.05
            covered+=d['p_value']>=.05
        self.assertLess(detections/1000,.075)
        self.assertGreater(covered/1000,.925)
        self.assertLess(covered/1000,.975)

    def test_false_discovery_and_acceptance(self):
        assert_allclose(false_discovery([1e-8,.01,.2,.8,np.nan]),[1,1,0,0,0])
        self.assertFalse(acceptance({'rows':[]})['passed'])
        report={'rows':[{'case':'baseline','period':'joint','region':'r','source_id':1,'model':'along_flow','amplitude':.1,
                 'draws':200,'attempted_draws':200,'median_amplitude_error':.1,'median_phase_error_degrees':10,'coverage_95':.95}]}
        # A positive-amplitude result alone must never authorize mapping.
        self.assertFalse(acceptance(report)['passed'])


class IOTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name);self.path=self.folder/'input.nc'
        self.basis,self.coefficients=make_input(self.path)

    def test_standalone_roundtrip_masks_and_evaluation(self):
        noise={'covariance_ra_m2':[[.01,0],[0,.01]],'floors_m':[.01,.01],'correlation':0,'quality_models':{}}
        with Collection([self.path]) as collection:
            point=collection.point(0,0)
            candidate={'basis':self.basis,'weighting':'constant','penalty':0}
            result=fit_point(point,noise,candidate,'free_en',direction=[1,0])
            result['fit_valid']=True
            save_fit(self.folder/'fit.nc',collection,{(0,0):result},'free_en',{'input':collection.sources})
        self.path.unlink()
        times=['2017-01-01','2017-01-02']
        actual=evaluate_file(self.folder/'fit.nc',times,0,0)
        expected=evaluate(result['coefficients'],self.basis,'free_en',[1,0],utc_days(times))
        assert_allclose(actual,expected)
        export_daily(self.folder/'fit.nc',self.folder/'daily.nc','2017-01-01','2017-01-03')
        with h5netcdf.File(self.folder/'daily.nc') as h:
            assert_allclose(h['east_velocity'][:,0,0],expected[:,0],rtol=1e-6)
            self.assertEqual(len(h.dimensions['time']),2)
        with h5netcdf.File(self.folder/'fit.nc') as h:
            self.assertNotIn('_FillValue',h['fit_valid'].attrs)
            self.assertEqual(h['detection_valid'][0,0],0)
        with self.assertRaisesRegex(ValueError,'not estimable'):
            evaluate_file(self.folder/'fit.nc',times,1,1)

    def test_duplicate_midpoints_missing_bursts_and_multiple_inputs(self):
        e=np.array([6209.,6212.,6221.]);l=np.array([6215.,6212.5,6227.])
        l[1]=6218.;e[0]=6206.;l[0]=6224. # same midpoint 6215
        make_input(self.path,e,l)
        with h5netcdf.File(self.path,'a') as h:
            h['earlier_burst_time'][1,1]=np.nan
            h['later_burst_time'][1,1]=np.nan
        rehash(self.path)
        other=self.folder/'other.nc';make_input(other,e+24,l+24,offset=3)
        with Collection([self.path,other]) as c:
            self.assertEqual(len(c.pairs),6)
            point=c.point(0,3)
            self.assertEqual(point['valid'].sum(),5)
            self.assertEqual(c.point(0,0)['valid'].sum(),6)
        with self.assertRaisesRegex(ValueError,'Duplicate pair'):
            with Collection([self.path,self.path]):pass
        with h5netcdf.File(other,'a') as h:h['source_observation_index'][0,0]=999
        rehash(other)
        with self.assertRaisesRegex(ValueError,'source associations'):
            with Collection([self.path,other]):pass

    def test_corruption_and_revision_checks(self):
        with h5netcdf.File(self.path,'a') as h:h['east_displacement_m'][0,0,0]=999
        with self.assertRaisesRegex(ValueError,'checksum'):
            with Collection([self.path]):pass
        rehash(self.path)
        with Collection([self.path]) as c:
            os.utime(self.path,None)
            with self.assertRaisesRegex(ValueError,'changed'):c.check_revisions()

    def test_uncalibrated_swath_observations_are_unavailable(self):
        with h5netcdf.File(self.path,'a') as h:
            h.attrs['source']='swath_corrected'
            v=h.create_variable('correction_available',('time','y','x'),dtype='u1')
            v[:]=1;v[0,0,0]=0;v[0,0,1]=0
        rehash(self.path)
        with Collection([self.path]) as c:
            self.assertFalse(c.point(0,0)['valid'][0])
            self.assertTrue(c.point(1,0)['valid'][0])

    def test_no_native_modules_or_processing_runtime(self):
        root=Path(__file__).resolve().parents[1]
        code='import sys; from glacier_offsets.cli import main; main(["fit-velocity","--help"]); print([k for k in sys.modules if k.startswith(("isce3","osgeo","glacier_offsets.packaging"))])'
        run=subprocess.run([sys.executable,'-c',code],env=os.environ|{'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        run=subprocess.run([str(root/'bin'/'glacier-offsets'),'fit-velocity','--help'],cwd=root,
            env=os.environ|{'GLACIER_OFFSETS_PYTHON':sys.executable,'PYTHONPATH':'src:.runtime/deps'},capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        script='import sys; from pathlib import Path; from glacier_offsets.cli import main\ntry: main(["fit-velocity","--help"])\nexcept SystemExit: pass\nassert not any(Path(p or ".").resolve()==Path(".runtime/deps").resolve() for p in sys.path)'
        run=subprocess.run([sys.executable,'-c',script],cwd=root,
            env=os.environ|{'PYTHONPATH':'src:.runtime/deps'},capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        # --help exits before print; explicitly inspect loaded module graph.
        run=subprocess.run([sys.executable,'-c','import glacier_offsets.velocity_recovery,sys; assert not any(k.startswith(("osgeo","isce3","glacier_offsets.packaging")) for k in sys.modules)'],env=os.environ|{'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)
        run=subprocess.run([str(root/'bin'/'glacier-offsets'),'fit-velocity','--help'],env=os.environ|{'GLACIER_OFFSETS_PYTHON':sys.executable,'PYTHONPATH':str(root/'.runtime/deps')},capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr)

    def test_settings_review_and_map_gates(self):
        path=self.folder/'config.yaml';path.write_text('inputs: [input.nc]\noutput: output\n')
        cfg=load_settings(path)
        self.assertEqual(cfg['inputs'],[str(self.path.resolve())])
        with Collection([self.path]) as c:
            with self.assertRaisesRegex(ValueError,'reviewed regions'):reviewed_regions(cfg,c)
            with self.assertRaisesRegex(ValueError,'reviewed regions'):map_stage(cfg,c)
            self.assertTrue(calibration_gate(cfg,c))
        path.write_text('inputs: [input.nc]\noutput: output\nmodels: {bogus: true}\n')
        with self.assertRaisesRegex(ValueError,'Unknown'):load_settings(path)

    def test_pilot_checkpoint_requires_matching_recipe_and_content(self):
        cache=PilotProgress(self.folder/'cache',{'source':'a'})
        cache.save('stage.json',{'model':'constant'})
        self.assertEqual(PilotProgress(self.folder/'cache',{'source':'a'}).read('stage.json'),{'model':'constant'})
        self.assertIsNone(PilotProgress(self.folder/'cache',{'source':'b'}).read('stage.json'))
        (self.folder/'cache/stage.json').write_text('{"model":"changed"}')
        self.assertIsNone(cache.read('stage.json'))

    def test_cv_checkpoint_survives_only_changes_outside_its_numerical_dependencies(self):
        directory=self.folder/'checkpoint'
        cache=PilotProgress(directory,{'simulation':'a'},cv_identity='same numerical problem')
        cache.save('cross_validation.json',{'models':[]})
        cache.save('validation_recovery.json',{'rows':[]})
        changed=PilotProgress(directory,{'simulation':'b'},cv_identity='same numerical problem')
        self.assertEqual(changed.read('cross_validation.json'),{'models':[]})
        self.assertIsNone(changed.read('validation_recovery.json'))
        different=PilotProgress(directory,{'simulation':'b'},cv_identity='changed fitting problem')
        self.assertIsNone(different.read('cross_validation.json'))

    def test_sampling_deduplicates_map_copies(self):
        with Collection([self.path]) as c:
            regions=[{'id':'r','role':'target','bounds':[90000,180000,120000,210000]}]
            points=sample_points(c,regions,per_region=100)
            self.assertEqual(len({p['source_id'] for p in points}),len(points))
            self.assertLess(len(points),12)

    def test_sampling_separates_sources_before_reading_time_series(self):
        make_input(self.path,shape=(8,8))
        with h5netcdf.File(self.path,'a') as h:
            h['x'][:]=np.arange(8)*1000+100000
            h['y'][:]=200000-np.arange(8)*1000
        rehash(self.path)
        with Collection([self.path]) as c:
            regions=[{'id':'r','role':'target','bounds':[99000,192000,108000,201000]}]
            with patch.object(c,'point',wraps=c.point) as read:
                points=sample_points(c,regions,per_region=5)
            self.assertEqual(read.call_count,len(points))
            positions=np.array([c.observation_position(p['row'],p['col']) for p in points])
            distances=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)
            self.assertTrue((distances[np.triu_indices(len(points),1)]>=3000).all())

    def test_freeze_is_checksum_verified_and_keeps_source(self):
        sha=digest(self.path)
        report=freeze([self.path],self.folder/'frozen')
        self.assertEqual(report['inputs'][0]['sha256'],sha)
        self.assertEqual(digest(self.path),sha)
        self.assertEqual(digest(self.folder/'frozen'/'input.nc'),sha)
        freeze([self.path],self.folder/'frozen')
        with h5netcdf.File(self.path,'a') as h:h['east_displacement_m'][0,0,0]=999
        rehash(self.path)
        with self.assertRaisesRegex(ValueError,'Frozen baseline differs'):
            freeze([self.path],self.folder/'frozen')
        parent=self.folder/'production.yaml';parent.write_text('stack: {}\n')
        extended_config(parent,self.folder/'extended.yaml')
        import yaml
        settings=yaml.safe_load((self.folder/'extended.yaml').read_text())
        self.assertEqual(settings['stack']['reference_date'],'2017-01-01')
        self.assertFalse(settings['stack']['swath_correction']['enabled'])

    def test_capacity_estimate_excludes_smaller_reference_alignment(self):
        root=self.folder/'stack'
        for day,size in [('2017-01-01',10),('2017-01-07',30)]:
            burst=root/'acquisitions'/day/'bursts'/'burst_iw1'/'slc.tif'
            burst.parent.mkdir(parents=True);burst.write_bytes(b'x'*size)
            write_json(root/'acquisitions'/day/'manifest.json',{'alignment_products':{str(burst):'test'}})
        plan={'reference_date':'2017-01-01','selected_burst_ids':['burst_iw1'],
              'inventory':[{'eligible':True,'date':'2018-01-01','available_burst_ids':['burst_iw1']}]}
        write_json(self.folder/'plan.json',plan)
        with patch('tools.prepare_velocity_experiment.shutil.disk_usage',return_value=SimpleNamespace(free=20)):
            report=estimate_extension_capacity(root,self.folder/'plan.json')
        self.assertEqual(report['estimated_new_alignment_bytes'],30)
        self.assertEqual(report['minimum_extra_alignment_bytes'],10)
        self.assertFalse(report['enough_for_alignments_alone'])

    def test_pilot_then_gated_tile_map_resume(self):
        # Small but complete workflow with real held-out noise estimation, CV,
        # injections, bootstrap, fit products and standalone daily evaluation.
        rng=np.random.default_rng(4)
        with h5netcdf.File(self.path,'a') as h:
            for axis,name in enumerate(('east_displacement_m','north_displacement_m')):
                values=h[name][:]
                values+=rng.normal(0,.015,values.shape)
                values[:,2,:]=rng.normal(0,.015,values[:,2,:].shape)
                values[:,0,1]=values[:,0,0]
                h[name][:]=values
        rehash(self.path)
        regions=[]
        for i,(row,col) in enumerate(((0,0),(0,2),(1,0),(1,2),(2,0),(2,2))):
            x=100000+4000*col;y=200000-4000*row
            regions.append({'id':f'r{i}','role':'target' if i<4 else 'control','reviewed':True,
                            'review_notes':'Synthetic known truth','bounds':[x-500,y-500,x+500,y+500]})
        write_json(self.folder/'regions.json',{'EPSG':3031,'regions':regions})
        config=self.folder/'config.yaml'
        config.write_text('inputs: [input.nc]\noutput: output\nregions: regions.json\n'
            'sampling: {points_per_region: 1}\nmodels: {backgrounds: [constant], weights: [constant], penalties: [0.0001]}\n'
            'pilot: {bootstrap_draws: 3, validation_draws: 2, tuning_draws: 2}\nmap: {tile_size: 2}\n')
        cfg=load_settings(config)
        with Collection([self.path]) as c,writer_lock(cfg['output']):
            report=pilot(cfg,c)
            self.assertFalse(report['map_allowed'])
            with patch('glacier_offsets.velocity_recovery.select_models',side_effect=AssertionError('verified CV must be reused')):
                repeated=pilot(cfg,c)
            self.assertEqual(report['selected'],repeated['selected'])
            self.assertTrue((Path(cfg['output'])/'pilot_free_en.nc').exists())
            with self.assertRaisesRegex(ValueError,'Mapping blocked'):map_stage(cfg,c)
            # Exercise tile mechanics independently of the scientific gate,
            # which this deliberately undersampled synthetic pilot cannot pass.
            report['map_allowed']=True
            write_json(Path(cfg['output'])/'pilot.json',report)
            with patch('glacier_offsets.velocity_recovery.calibration_gate',return_value=[]):
                result=map_stage(cfg,c)
                self.assertEqual(result['status'],'complete')
                tile=next((Path(cfg['output'])/'tiles').glob('*free_en.nc'))
                before=tile.stat().st_mtime_ns
                map_stage(cfg,c)
                self.assertEqual(tile.stat().st_mtime_ns,before)
            with h5netcdf.File(Path(cfg['output'])/'map_free_en.nc') as h:
                self.assertEqual(h['coefficients'].shape[-2:],c.shape)
                self.assertEqual(h['detection_valid'][0,0],h['detection_valid'][0,1])


if __name__=='__main__':
    unittest.main()

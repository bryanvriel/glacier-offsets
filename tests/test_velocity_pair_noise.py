"""Control calibration, held-out isolation, and pair-weighted robust fits."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from glacier_offsets.velocity_pair_noise import SCHEMA, calibrate_pairs, pair_covariances
from glacier_offsets.velocity_fit import covariance_ra
from glacier_offsets.velocity_basis import definition, design
from glacier_offsets.velocity_pilot import learn_noise, fit_point
from glacier_offsets.velocity_annual_noise import read_pair_noise
from glacier_offsets.velocity_io import digest
from glacier_offsets.velocity_pixel_general import Layout, solve
from glacier_offsets.velocity_workflow import run_workflow
from test_velocity_pixel import fixture
from velocity_synthetic import Truth, write_package, pair_list, config_text


def pooled():
    return {'covariance_ra_m2':[[.01,0],[0,.04]],'floors_m':[.01,.02],
            'correlation':0.,'quality_models':{},'control_bias_m':[0.,0.]}


def controls(n=40, nt=120):
    rng=np.random.default_rng(941)
    early=6209.25+np.arange(nt)*6.
    ids=np.array([f'p{i}' for i in range(nt)])
    records=[]
    scales=np.tile([.1,.2],(nt,1));scales[::3]*=6
    for k in range(n):
        data=rng.normal(size=(nt,2))*scales
        point={'pair_id':ids,'region':f'r{k%2}','role':'control','spatial_block':str(k),
               'early':early,'late':early+6,'data':data,'valid':np.ones(nt,bool),
               'snr':np.full(nt,30.),'peak':np.full(nt,.5),'inverse':np.eye(2)}
        records.append({'point':point,'residual_ra':data,'valid':point['valid']})
    return records


class PairNoiseTests(unittest.TestCase):
    def test_pair_scatter_support_bias_and_duplicate_blocks(self):
        records=controls(nt=3)
        for r in records:r['residual_ra']=r['residual_ra']+np.array([4.,-2.])
        calibration=calibrate_pairs(records,pooled(),min_blocks=8,shrinkage_blocks=4)
        duplicated=calibrate_pairs(records+records,pooled(),min_blocks=8,shrinkage_blocks=4)
        self.assertEqual(calibration,duplicated)
        loud=calibration['pairs']['p0'];quiet=calibration['pairs']['p1']
        self.assertEqual(loud['independent_blocks'],40)
        self.assertGreater(np.trace(loud['covariance_ra_m2']),8*np.trace(quiet['covariance_ra_m2']))
        assert_allclose(loud['bias_ra_m'],[4.,-2.],atol=.5)
        self.assertFalse(calibration['bias_applied'])
        self.assertGreater(np.linalg.eigvalsh(loud['covariance_ra_m2']).min(),0.)

    def test_sparse_and_unseen_pairs_fall_back_and_lookup_uses_ids(self):
        noise=pooled();noise['pair_calibration']=calibrate_pairs(controls(n=2,nt=3),noise)
        self.assertTrue(all(r['pooled_fallback'] for r in noise['pair_calibration']['pairs'].values()))
        assert_array_equal(pair_covariances(noise,['p2','unknown','p0']),np.tile(noise['covariance_ra_m2'],(3,1,1)))
        noise['pair_calibration']['pairs']['p2']['covariance_ra_m2']=[[1.,.2],[.2,2.]]
        cov=covariance_ra(noise,np.ones(2)*30,np.ones(2)*.5,pair_ids=['p2','p0'])
        assert_array_equal(cov[0],[[1.,.2],[.2,2.]])
        assert_allclose(covariance_ra(noise,np.ones(2)*30,np.ones(2)*.5,True,['p2','p0']),cov)
        with self.assertRaisesRegex(ValueError,'pair_id'):covariance_ra(noise,[30],[.5])

    def test_training_calibration_ignores_heldout_acquisitions_and_excluded_region(self):
        records=controls(n=12)
        points=[r['point'] for r in records]
        take=np.arange(120)<80
        options={'mode':'pair','min_pair_blocks':3,'shrinkage_blocks':10.}
        original,_=learn_noise(points,train=take,excluded_region='r0',options=options)
        altered=[]
        for p in points:
            data=p['data'].copy();data[~take]=1e6
            if p['region']=='r0':data[:]=1e7
            altered.append(p|{'data':data})
        repeated,_=learn_noise(altered,train=take,excluded_region='r0',options=options)
        self.assertEqual(original,repeated)
        self.assertTrue(original['pair_calibration']['pairs']['p100']['pooled_fallback'])

    def test_pair_weighting_improves_horizontal_recovery(self):
        rng=np.random.default_rng(133)
        early=6209.25+np.arange(120)*6.;late=early+6
        basis=definition(early,late)
        g=design(early,late,basis,'along_flow',[1.,0.])[0]
        truth=np.array([1.,.2,.08,.06]);data=np.einsum('ijk,k->ij',g,truth)
        sigma=np.full(120,.02);sigma[::3]=3.
        data+=rng.normal(size=data.shape)*sigma[:,None]
        ids=np.array([f'p{i}' for i in range(120)])
        point={'pair_id':ids,'early':early,'late':late,'data':data,'valid':np.ones(120,bool),
               'snr':np.full(120,30.),'peak':np.full(120,.5),'inverse':np.eye(2)}
        candidate={'basis':basis,'weighting':'constant','penalty':0.}
        baseline=pooled()
        calibrated=baseline|{'pair_calibration':{'pairs':{p:{'covariance_ra_m2':(np.eye(2)*s*s).tolist()}
                                                        for p,s in zip(ids,sigma)}}}
        ordinary=fit_point(point,baseline,candidate,'along_flow',direction=[1.,0.])
        weighted=fit_point(point,calibrated,candidate,'along_flow',direction=[1.,0.])
        self.assertLess(np.linalg.norm(weighted['coefficients']-truth),.2*np.linalg.norm(ordinary['coefficients']-truth))

    def test_pixel_robust_fit_rejects_gross_error_without_clipping_real_harmonic(self):
        field,axis,truth,*_=fixture();layout=Layout([2017,2018])
        cov=np.tile(np.diag([.01,.04]),(2,1,1))
        pair={'by_year':cov,'by_pair':cov[field['year_index']]}
        field['data']=field['data'].copy();field['data'][:,13]+=100.
        ordinary=solve(field,layout,field['valid'],axis,cov)
        robust=solve(field,layout,field['valid'],axis,pair)
        self.assertLess(np.linalg.norm(robust['cs']-truth),.05*np.linalg.norm(ordinary['cs']-truth))
        self.assertTrue((robust['downweighted_pairs']>0).all())
        clean,axis,truth,*_=fixture()
        original=solve(clean,layout,clean['valid'],axis,pair)
        # Add a large, exactly modeled harmonic: no secular-only metre cuts.
        from glacier_offsets.velocity_basis import harmonic
        from glacier_offsets.velocity_pixel_general import pair_geometry
        delta=truth*50
        projection=np.einsum('stci,si->stc',pair_geometry(clean)[...,:2],axis)
        clean['data']+=np.einsum('stj,sj->st',harmonic(clean['early'],clean['late']),delta)[...,None]*projection
        recovered=solve(clean,layout,clean['valid'],axis,pair)
        assert_allclose(recovered['cs'],original['cs']+delta,atol=1e-8)
        assert_array_equal(recovered['downweighted_pairs'],0)

    def test_annual_huber_matches_pair_huber_and_auto_keeps_single_gls(self):
        field,axis,truth,*_=fixture();layout=Layout([2017,2018])
        cov=np.tile(np.diag([.01,.04]),(2,1,1))
        pair={'by_year':cov,'by_pair':cov[field['year_index']]}
        rng=np.random.default_rng(7);field['data']=field['data']+rng.normal(size=field['data'].shape)*[.1,.2]
        field['data'][:,13]+=100.
        annual=solve(field,layout,field['valid'],axis,cov,robust='huber')
        paired=solve(field,layout,field['valid'],axis,pair)
        assert_allclose(annual['coefficients'],paired['coefficients'],atol=1e-10)
        assert_array_equal(annual['downweighted_pairs'],paired['downweighted_pairs'])
        legacy=solve(field,layout,field['valid'],axis,cov)
        single=solve(field,layout,field['valid'],axis,pair,robust='none')
        assert_allclose(single['coefficients'],legacy['coefficients'],atol=1e-10)
        assert_array_equal(legacy['robust_iterations'],1)
        self.assertLess(np.linalg.norm(annual['cs']-truth),.05*np.linalg.norm(legacy['cs']-truth))
        with self.assertRaisesRegex(ValueError,'robust'):solve(field,layout,field['valid'],axis,cov,robust='tukey')

    def test_source_scale_protects_uniformly_noisy_source_and_is_never_tighter(self):
        field,axis,truth,*_=fixture();layout=Layout([2017,2018])
        cov=np.tile(np.diag([.01,.04]),(2,1,1))
        rng=np.random.default_rng(19);noise=rng.normal(size=field['data'].shape)*[.1,.2]
        noise[:2]*=3.  # first two sources three times noisier than the covariance
        field['data']=field['data']+noise
        fixed=solve(field,layout,field['valid'],axis,cov,robust='huber')
        scaled=solve(field,layout,field['valid'],axis,cov,robust='huber',robust_scale='source')
        self.assertTrue((scaled['robust_channel_scale'][:2]>2.).all())
        assert_array_equal(scaled['robust_channel_scale'][2:]>=1.,True)
        self.assertTrue((scaled['robust_channel_scale'][2:]<1.3).all())
        self.assertGreater(fixed['downweighted_pairs'][:2].min(),3*scaled['downweighted_pairs'][:2].max())

    def test_robust_fit_does_not_depend_on_batch_composition(self):
        field,axis,truth,*_=fixture();layout=Layout([2017,2018])
        cov=np.tile(np.diag([.01,.04]),(2,1,1))
        rng=np.random.default_rng(23);field['data']=field['data']+rng.standard_t(2,size=field['data'].shape)*[.1,.2]
        field['data'][3,20:24]+=40.  # one source needs many more iterations
        whole=solve(field,layout,field['valid'],axis,cov,robust='huber',robust_scale='source')
        self.assertGreater(whole['robust_iterations'].max(),whole['robust_iterations'].min())
        for subset in ([0,1],[2,4],[3]):
            part={k:(v[subset] if k in ('early','late','data','peak','valid') else v) for k,v in field.items()}
            part['geometry_by_year']=field['geometry_by_year'][:,subset]
            alone=solve(part,layout,part['valid'],axis[subset],cov,robust='huber',robust_scale='source')
            assert_array_equal(alone['coefficients'],whole['coefficients'][subset])

    def test_robust_choices_are_validated(self):
        from glacier_offsets.velocity_pixel_config import validate_parameters
        self.assertEqual(validate_parameters({'screening':{'robust':'huber','robust_scale':'source'}})['screening']['robust'],'huber')
        for bad in ({'robust':'tukey'},{'robust_scale':'mad'},{'robust':1}):
            with self.assertRaisesRegex(ValueError,'screening.robust'):validate_parameters({'screening':bad})

    def test_imported_artifact_pixel_fit_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'stack.nc'
            write_package(path,Truth([2017,2018]),pair_list('2017-07-01',60))
            with h5py.File(path) as h:ids=h['pair_id'][:].astype(str).tolist()
            noise=pooled();noise['pair_calibration']={'pairs':{p:{'covariance_ra_m2':(np.diag([.01,.04])*(1+i%3)).tolist()}
                                                             for i,p in enumerate(ids)}}
            artifact={'schema':SCHEMA,'source_sha256':[digest(path)],'noise':noise}
            calibration=root/'pair-noise.json';calibration.write_text(json.dumps(artifact))
            read_pair_noise(calibration,list(reversed(ids)),[digest(path)])
            with self.assertRaisesRegex(ValueError,'SHA256'):read_pair_noise(calibration,ids,['wrong'])
            with self.assertRaisesRegex(ValueError,'missing pair'):read_pair_noise(calibration,['missing'],[digest(path)])
            config=root/'pixel.yaml'
            config.write_text(config_text(['stack.nc'],'cache','out','mode: pair\nfile: pair-noise.json',min_span=300))
            run_workflow(config,'run')
            with h5py.File(root/'out/source_coefficients.h5') as h:
                self.assertIn('covariance_by_pair',h)
                self.assertTrue(h['residual_vertical/estimable'][:].any())
                self.assertIn('downweighted_pairs',h['residual_vertical'])
            self.assertTrue((root/'out/verification.json').is_file())
            before=digest(root/'out/source_coefficients.h5')
            run_workflow(config,'fit')
            self.assertEqual(before,digest(root/'out/source_coefficients.h5'))


if __name__=='__main__':unittest.main()

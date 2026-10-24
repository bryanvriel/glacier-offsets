"""Fixed-support deramp comparisons, attrition, and native CLI integration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.config import DEFAULT_MATCHING
from glacier_offsets.matching_grid import build_grid
from tools.compare_swath_overlap import (attach_results, cohort_summary, compare_deramp_modes,
                                         comparison_reference, main, run_job, select_views)


def write_raster(path, array, dtype):
    ds=gdal.GetDriverByName('GTiff').Create(str(path),array.shape[1],array.shape[0],1,dtype,
                                           options=['TILED=NO'])
    ds.GetRasterBand(1).WriteArray(array);ds=None


def make_pair(root):
    """Two swaths with identical, known one-pixel range motion."""
    root=root.resolve();gdal.UseExceptions()
    stack=root/'stack';pair=stack/'pairs/2020-01-01_2020-01-07'
    (pair/'radar').mkdir(parents=True);(stack/'geometry').mkdir()
    (stack/'geometry/manifest.json').write_text('{"geometry_identity":"fixed"}');(stack/'stack_manifest.json').write_text('{}')
    params=deepcopy(DEFAULT_MATCHING)
    params.update(engine='isce3',deramp=0,window=[32,64],search=[4,8],stride=[48,96],
                  batch=[1,2],first_center=[23.5,47.5],zoom=8,snr_threshold=0.,correlation_threshold=0.)
    shape=(160,320);grid=build_grid(shape,shape,params['window'],params['search'],params['stride'],
                                  first_center=params['first_center'])
    yy,xx=np.indices(shape)
    geometry={}
    for key,array in [('x',-2000000.+xx*5),('y',-1000000.-yy*20)]:
        path=stack/'geometry'/f'{key}.tif';write_raster(path,array,gdal.GDT_Float64);geometry[key]=str(path)
    rng=np.random.default_rng(231)
    ref=(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype('complex64')
    sources=[];dates=['2020-01-01','2020-01-07']
    keys=['t038_080975_iw1','t038_080975_iw2']
    for key in keys:
        sources.append({'burst_id':key,'offset':[0,0],'valid_window':[0,shape[0],0,shape[1]],
                        'coreg':{'geometry_crs':3031,'geometry':geometry}})
    for i,day in enumerate(dates):
        folder=stack/'acquisitions'/day;folder.mkdir(parents=True)
        slcs={};products={}
        for key in keys:
            path=folder/f'{key}.tif';write_raster(path,np.roll(ref,i,axis=1),gdal.GDT_CFloat32)
            slcs[key]=str(path);products[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        (folder/'manifest.json').write_text(json.dumps({'slcs':slcs,'alignment_products':products,
            'specification':{'geometry':'fixed'}}))
    manifest={'matching':{'matching':params,'gross_offsets':{'mode':'zero'}},
              'stack_pair':{'acquisition_dates':dates,'geometry_identity':'fixed'},'assembly':{'sources':sources}}
    (pair/'manifest.json').write_text(json.dumps(manifest))
    line,sample=grid.centers
    with h5py.File(pair/'radar/offsets.h5','w') as h:
        for key,data in [('center_line',line),('center_sample',sample),('valid',np.ones(grid.shape,bool)),
                         ('input_valid',np.ones(grid.shape,bool)),('source_burst_index',np.ones(grid.shape,int)),
                         ('total/azimuth_px',np.zeros(grid.shape)),('total/range_px',np.ones(grid.shape)),
                         ('correlation_peak',np.ones(grid.shape))]:h[key]=data
    return pair,grid


class SwathSelectionTests(unittest.TestCase):
    def test_new_dates_keep_grid_and_clear_unrelated_merged_reference(self):
        saved={'valid':np.ones((1,2),bool),'center_line':np.array([[1.,2.]]),
               'total/azimuth_px':np.ones((1,2)),'total/range_px':np.ones((1,2))}
        params={'deramp':1,'zoom':16}
        p,s,days=comparison_reference(params,saved,['2020-01-01','2020-01-07'],
                                     ['2020-02-01','2020-02-13'],True)
        self.assertEqual(days,['2020-02-01','2020-02-13']);self.assertEqual(p,{'deramp':0,'zoom':16})
        self.assertFalse(s['valid'].any());self.assertTrue(np.isnan(s['total/range_px']).all())
        np.testing.assert_array_equal(s['center_line'],saved['center_line'])
        self.assertTrue(saved['valid'].all());self.assertEqual(params['deramp'],1)
        with self.assertRaisesRegex(ValueError,'requires --compare-deramp'):
            comparison_reference(params,saved,[],['2020-02-01','2020-02-13'])
        with self.assertRaisesRegex(ValueError,'increasing acquisition dates'):
            comparison_reference(params,saved,[],['2020-02-13','2020-02-01'],True)

    def test_failed_later_owner_is_preserved_in_both_modes_without_quality_fallback(self):
        sources=[{'burst_id':f't038_{i:06d}_{swath}','offset':[i,0]}
                 for i,swath in [(1,'iw1'),(2,'iw1'),(3,'iw2')]]
        records={}
        for e in sources:
            support=np.ones((1,2),bool)
            result={'valid':support.copy(),'finite':support.copy(),'search_boundary':~support,
                    'total':np.ones((1,2,2)),'correlation_peak':np.ones((1,2)),'snr':np.ones((1,2))}
            records[e['burst_id']]={'rows':np.array([0]),'cols':np.array([0,1]),'support':support,
                                   'geometry':np.zeros((1,2,5,2)),'results':{0:result,1:result}}
        selection,numbers=select_views(sources,records,(1,2))
        later=records[sources[1]['burst_id']]
        later['results'].pop(1)
        later['results'][0]=deepcopy(later['results'][0]);later['results'][0]['valid'][:]=False
        for mode in (0,1):
            v=attach_results(selection,records,numbers,mode)['iw1']
            np.testing.assert_array_equal(v['owner'],2)
            self.assertTrue(v['support'].all());self.assertFalse(v['valid'].any())
            self.assertEqual(bool(v['native_completed'].all()),mode==0)
        self.assertTrue(np.isnan(attach_results(selection,records,numbers,1)['iw1']['total']).all())

    def views(self):
        views={}
        for mode,valid in [(0,[True,True,False,False]),(1,[True,False,True,False])]:
            views[mode]={}
            for swath in ('iw1','iw2'):
                mask=np.array([valid]);shape=mask.shape
                views[mode][swath]={'support':np.ones(shape,bool),'owner':np.ones(shape,int),
                    'geometry':np.zeros((*shape,5,2)),'valid':mask,'finite':np.ones(shape,bool),
                    'search_boundary':~mask,'native_completed':np.ones(shape,bool),
                    'total':np.full((*shape,2),mode+(swath=='iw2')*.125),
                    'peak':np.full(shape,.8),'snr':np.full(shape,4.)}
        return views

    def test_attrition_is_reported_and_comparison_uses_identical_surviving_points(self):
        summary=cohort_summary(self.views(),np.ones((1,4),bool))
        self.assertEqual(summary['input_supported_n'],4)
        self.assertEqual(summary['validity_transitions'],{'valid_both':1,'only_deramp0':1,'only_deramp1':1,'neither':1})
        for mode in ('0','1'):
            stats=summary['modes'][mode]['difference_on_common_valid_px']
            self.assertEqual(stats['range']['n'],1)
            self.assertEqual(stats['azimuth']['mean'],.125)
        self.assertEqual(summary['deramp1_minus_deramp0_on_common_valid_px']['iw1']['range']['mean'],1.)

    def test_parameter_or_owner_changes_are_rejected_before_comparison(self):
        views=self.views();saved={'center_line':np.zeros((1,4)),'center_sample':np.ones((1,4))}
        args=(Path('/unused'),Path('/pair'),{0:{'deramp':0,'zoom':8},1:{'deramp':1,'zoom':16}},
              views,saved,np.ones((1,4),bool),[])
        with self.assertRaisesRegex(ValueError,'differ only'):compare_deramp_modes(*args)
        args[2][1]['zoom']=8;views[1]['iw1']['owner'][0,0]=2
        with self.assertRaises(AssertionError):compare_deramp_modes(*args)

    def test_native_job_error_is_returned_without_discarding_other_jobs(self):
        with patch('tools.compare_swath_overlap._native_worker',side_effect=RuntimeError('test native failure')):
            name,result,error=run_job(((1,'iw1'),(),{}))
        self.assertEqual(name,(1,'iw1'));self.assertIsNone(result)
        self.assertIn('test native failure',error)


class SwathNativeIntegrationTests(unittest.TestCase):
    def test_failed_jobs_keep_fixed_support_and_write_incomplete_results(self):
        class FailedPool:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def imap_unordered(self,worker,jobs):
                return iter((name,None,'RuntimeError: injected failure') for name,_,_ in jobs)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pair,grid=make_pair(root);out=root/'failed'
            with patch('tools.compare_swath_overlap._native_executor',return_value=FailedPool()):
                self.assertEqual(main([str(pair),'--output',str(out),'--compare-deramp']),1)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'incomplete');self.assertEqual(len(report['job_errors']),4)
            group=report['groups']['all_input_supported']
            self.assertEqual(group['input_supported_n'],np.prod(grid.shape))
            self.assertEqual(group['common_native_valid_n'],0)
            for mode in (0,1):
                with np.load(out/f'deramp-{mode}/observations.npz') as obs:
                    self.assertTrue(obs['dual_support'].all());self.assertFalse(obs['dual_valid'].any())
                    self.assertTrue(np.isnan(obs['iw1_total']).all())
                    self.assertFalse(obs['iw1_native_completed'].any())

    def test_date_override_matches_known_motion_without_reusing_merged_offsets(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pair,grid=make_pair(root);out=root/'new-dates'
            manifest_path=pair/'manifest.json';manifest=json.loads(manifest_path.read_text())
            manifest['matching']['matching']['deramp']=1;manifest_path.write_text(json.dumps(manifest))
            # Re-key the aligned observations to different dates from the grid template.
            days=['2020-02-01','2020-02-13']
            for old,new in zip(('2020-01-01','2020-01-07'),days):
                source=root/'stack/acquisitions'/old;target=root/'stack/acquisitions'/new
                target.mkdir();record=json.loads((source/'manifest.json').read_text())
                record['status']='complete';record['burst_times']={}
                (target/'manifest.json').write_text(json.dumps(record))
            self.assertEqual(main([str(pair),'--output',str(out),'--workers','2','--compare-deramp',
                                   '--acquisition-dates',*days]),0)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['pair'],'_'.join(days));self.assertEqual(report['acquisition_dates'],days)
            self.assertFalse(report['merged_reference_available'])
            self.assertTrue(report['production_sizes_mtimes_unchanged'])
            for mode in (0,1):
                with np.load(out/f'deramp-{mode}/observations.npz') as obs:
                    self.assertTrue(np.isnan(obs['merged_total']).all())
                    np.testing.assert_array_equal(obs['center_line'],grid.centers[0])
                    for swath in ('iw1','iw2'):
                        np.testing.assert_allclose(obs[f'{swath}_total'],np.broadcast_to([0,1],(*grid.shape,2)),atol=.05)

    def test_paired_cli_preserves_sources_centers_support_and_known_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);pair,grid=make_pair(root);out=root/'experiment'
            before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in (root/'stack').rglob('*') if p.is_file()}
            self.assertEqual(main([str(pair),'--output',str(out),'--workers','2','--compare-deramp']),0)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'complete');self.assertTrue(report['production_sizes_mtimes_unchanged'])
            self.assertEqual(report['groups']['all_input_supported']['input_supported_n'],np.prod(grid.shape))
            for mode in (0,1):
                with np.load(out/f'deramp-{mode}/observations.npz') as obs:
                    np.testing.assert_array_equal(obs['center_line'],grid.centers[0])
                    np.testing.assert_array_equal(obs['center_sample'],grid.centers[1])
                    self.assertTrue(obs['dual_support'].all());self.assertTrue(obs['dual_valid'].all())
                    for swath in ('iw1','iw2'):
                        np.testing.assert_allclose(obs[f'{swath}_total'],np.broadcast_to([0,1],(*grid.shape,2)),atol=.05)
                        self.assertTrue(obs[f'{swath}_native_completed'].all())
                raw=out/f'deramp-{mode}/t038_080975_iw1.npz'
                with np.load(raw) as obs:
                    self.assertEqual(json.loads(str(obs['effective_json']))['derampMethod'],mode)
                    np.testing.assert_array_equal(obs['total'],obs['gross']+obs['residual'])
                    self.assertIn('search_boundary',obs.files)
            self.assertGreater((out/'deramp_comparison.png').stat().st_size,1000)
            self.assertTrue((out/'comparison_masks.npz').is_file())
            for path,(data,mtime) in before.items():
                self.assertEqual(path.read_bytes(),data);self.assertEqual(path.stat().st_mtime_ns,mtime)
            with self.assertRaisesRegex(ValueError,'new output directory'):
                main([str(pair),'--output',str(out),'--compare-deramp'])


if __name__=='__main__':unittest.main()

"""Common-grid SLC ownership, native science, exact intervals and restart controls."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal
from pyproj import CRS
from scipy.ndimage import gaussian_filter, shift

from glacier_offsets.assembly import assemble, annotate_result, plan_sources, settings, source_indices
from glacier_offsets.ampcor import match
from glacier_offsets.config import load_defaults
from glacier_offsets.geocode import center_geometry, geocode, physical_fields, formal_uncertainties, ConservativeMapper
from glacier_offsets.products import checksum, identity, input_provenance, write_json
from glacier_offsets.matching_grid import MatchingGrid
from test_synthetic_offsets import write_slc


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        source=self.root/'input';source.write_text('shared pair')
        self.cfg=load_defaults()
        self.cfg['inputs'].update({key:str(source) for key in ('reference_safe','secondary_safe','reference_orbit','secondary_orbit','dem')})
        self.cfg['inputs'].update(polarization='HH',burst_id=None)
        self.cfg['output']={'directory':str(self.root/'output')}
        self.cfg['matching'].update(backend='cpu',window=[32,64],search=[8,12],stride=[16,32],zoom=8,batch=[1,4])
        self.cfg['geocoding'].update(epsg=3031,spacing_m=50)
        self.cfg['aoi']={'bbox':[-110,-80,-90,-70]}
        self.sources=[];self.pairs={}
        shape=(320,384);rng=np.random.default_rng(41)
        amp=gaussian_filter(rng.uniform(.1,1,shape),.6)
        reference=(amp*np.exp(1j*rng.normal(0,.2,shape))).astype('complex64')
        secondary=shift(reference,(2,-3),order=0,mode='constant',cval=0)
        for index,offset in enumerate((0,128)):
            key=f't001_00000{index}_iw2';root=self.root/key;root.mkdir()
            grid={'length':192,'width':384,'prf':500.,'range_pixel_spacing':2.,'wavelength':.05,
                  'lookside':'right','ref_epoch':'2020-01-01T00:00:00','sensing_start':offset/500,
                  'starting_range':800000.}
            slcs={}
            for side,array in [('reference',reference),('secondary',secondary)]:
                crop=array[offset:offset+192].copy()
                crop[:4]=0;crop[-4:]=0;crop[:,:4]=0;crop[:,-4:]=0
                crop*=np.exp(index*.3j)
                path=root/f'{side}.tif';write_slc(path,crop);slcs[side]=str(path)
            yy,xx=np.indices((192,384));geometry={}
            for name,data in {'x':-1e6+xx*10.,'y':-1e6-(yy+offset)*10.,'z':np.zeros(xx.shape),
                              'layover_shadow_mask':np.zeros(xx.shape)}.items():
                path=root/f'{name}.tif';ds=gdal.GetDriverByName('GTiff').Create(str(path),384,192,1,gdal.GDT_Float64)
                ds.GetRasterBand(1).WriteArray(data);ds=None;geometry[name]=str(path)
            spec={'inputs':input_provenance(self.cfg),'polarization':'HH','burst_id':key}
            time=datetime(2020,1,1)+timedelta(seconds=offset/500)
            m={'coreg_identity':identity(spec),'coreg_specification':spec,'software':{'packages':{}},
               'coreg':{'reference_grid':grid,'slcs':slcs,'geometry':geometry,'geometry_crs':CRS.from_epsg(3031).to_wkt()},
               'interval_seconds':86400.+index,'acquisition_times':[time.isoformat(),(time+timedelta(seconds=86400+index)).isoformat()],
               'stages':{'coregister':'complete'}}
            write_json(root/'manifest.json',m);self.sources.append((root,m))
            native=SimpleNamespace(as_isce3_radargrid=lambda grid=grid:SimpleNamespace(**grid),orbit=object())
            self.pairs[key]=(SimpleNamespace(id=key,native=native),SimpleNamespace(id=key,native=native))

    def workflow_patches(self, stack):
        from contextlib import ExitStack
        self.cfg['output']['directory']=str(self.root)
        rows=tuple(self.pairs.values())
        selection={'selected_ids':list(self.pairs),'selection_identity':'test-selection','common_count':len(rows)}
        stack.enter_context(patch('glacier_offsets.batch.select_aoi',return_value=(rows,selection)))
        stack.enter_context(patch('glacier_offsets.batch.inspect_environment',return_value={'packages':{}}))
        stack.enter_context(patch('glacier_offsets.batch.dem_extent',return_value=object()))
        stack.enter_context(patch('glacier_offsets.batch.validate_burst'))
        stack.enter_context(patch('glacier_offsets.assembly.validate_native_layout'))
        return rows

    def test_batch_dispatch_assembles_then_runs_one_match_and_geocode(self):
        from contextlib import ExitStack
        from glacier_offsets.batch import execute_batch
        with ExitStack() as stack:
            self.workflow_patches(stack)
            coreg=stack.enter_context(patch('glacier_offsets.pipeline.execute'))
            native=stack.enter_context(patch('glacier_offsets.ampcor.match'))
            maps=stack.enter_context(patch('glacier_offsets.geocode.geocode'))
            stack.enter_context(patch('glacier_offsets.diagnostics.diagnostics'))
            self.assertEqual(execute_batch(self.cfg,'run'),0)
            self.assertEqual([c.args[1] for c in coreg.call_args_list],['coregister','coregister'])
            native.assert_called_once();maps.assert_called_once()
            self.assertEqual(native.call_args.args[1],self.root/'merged')
            self.assertEqual(maps.call_args.args[1],self.root/'merged')
            self.assertEqual(set(native.call_args.kwargs['native_pair']),set(self.pairs))
            report=json.loads((self.root/'batch_manifest.json').read_text())
            self.assertEqual(report['merged']['status'],'complete')
            self.assertEqual(report['merged']['source_burst_ids'],list(self.pairs))

    def test_batch_missing_match_failure_and_empty_selection_do_not_publish_stale_merge(self):
        from contextlib import ExitStack
        from glacier_offsets.batch import execute_batch
        with ExitStack() as stack:
            self.workflow_patches(stack)
            root,m=assemble(self.cfg,self.sources)
            self.assertEqual(execute_batch(self.cfg,'geocode'),1)
            report=json.loads((self.root/'batch_manifest.json').read_text())
            self.assertIn('run match',report['merged']['error'])
            with patch('glacier_offsets.pipeline.execute',side_effect=[RuntimeError('burst failed'),None]):
                self.assertEqual(execute_batch(self.cfg,'run'),1)
            self.assertNotIn('coreg_identity',json.loads((root/'manifest.json').read_text()))
            self.assertEqual(json.loads((self.root/'batch_manifest.json').read_text())['merged']['status'],'blocked_incomplete_bursts')
            with patch('glacier_offsets.batch.select_aoi',return_value=((),{'selected_ids':[],'selection_identity':'empty'})):
                self.assertEqual(execute_batch(self.cfg,'run'),2)

    def test_batch_rejects_changed_fractional_assembly_and_invalidates_old_products(self):
        from contextlib import ExitStack
        from glacier_offsets.batch import execute_batch
        with ExitStack() as stack:
            self.workflow_patches(stack)
            root,m=assemble(self.cfg,self.sources)
            path=self.sources[1][0]/'manifest.json'
            changed=json.loads(path.read_text());changed['coreg']['reference_grid']['sensing_start']+=.001
            write_json(path,changed)
            self.assertEqual(execute_batch(self.cfg,'assemble'),1)
            stale=json.loads((root/'manifest.json').read_text())
            self.assertNotIn('coreg_identity',stale)
            self.assertEqual(stale['assembly_status'],'required_assembly_failed')

    def test_source_ownership_geometry_cache_and_selection_change(self):
        root,m=assemble(self.cfg,self.sources)
        self.assertEqual(m['coreg']['reference_grid']['length'],320)
        self.assertEqual(m['assembly']['identity'],identity(m['assembly']['specification']))
        for field in ('reference','secondary'):
            expected=np.zeros((320,384),'complex64')
            for entry in m['assembly']['sources']:
                y0,y1,x0,x1=entry['valid_window'];dy,dx=entry['offset']
                source=gdal.Open(entry['coreg']['slcs'][field]).ReadAsArray()
                expected[y0+dy:y1+dy,x0+dx:x1+dx]=source[y0:y1,x0:x1]
            np.testing.assert_array_equal(gdal.Open(m['coreg']['slcs'][field]).ReadAsArray(),expected)
        line=np.array([[50.,140.,200.]]);sample=np.full(line.shape,100.)
        np.testing.assert_array_equal(source_indices(m['coreg'],line,sample),[[1,2,2]])
        np.testing.assert_array_equal(center_geometry(m['coreg'],line,sample)[2],0.)
        paths=[Path(p) for p in m['assembly']['products']]
        before={p:(checksum(p),p.stat().st_mtime_ns) for p in paths}
        _,again=assemble(self.cfg,self.sources[::-1])
        self.assertEqual(again['coreg_identity'],m['coreg_identity'])
        self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in paths})
        pairs=deepcopy(self.pairs)
        for pair in pairs.values():
            pair[0].native.first_valid_line=4;pair[0].native.last_valid_line=187
            pair[0].native.first_valid_sample=4;pair[0].native.last_valid_sample=379
        _,annotated=assemble(self.cfg,self.sources,native_pairs=pairs)
        self.assertEqual(annotated['coreg_identity'],m['coreg_identity'])
        self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in paths})
        _,changed=assemble(self.cfg,self.sources[:1])
        self.assertNotEqual(changed['coreg_identity'],m['coreg_identity'])
        self.assertEqual(changed['coreg']['reference_grid']['length'],192)
        self.assertNotIn('match_identity',changed)

    def test_incompatible_grids_and_invalid_config_rejected(self):
        for value in [None,{'enabled':'yes'},{'alignment_tolerance_px':.6},{'unknown':True}]:
            with self.assertRaises(ValueError):settings({'burst_merge':value})
        for change in [{'prf':501.},{'starting_range':800000.5},{'sensing_start':.257}]:
            sources=deepcopy(self.sources);sources[1][1]['coreg']['reference_grid'].update(change)
            with self.assertRaisesRegex(ValueError,'incompatible|fractional'):
                plan_sources(sources)

    def test_missing_prior_does_not_invent_a_secondary_search_seam(self):
        _,m=assemble(self.cfg,self.sources)
        grid=MatchingGrid(np.array([120,140]),np.array([80]),(32,64),(8,12),(20,32))
        result={'grid':grid,'gross':np.full((*grid.shape,2),np.nan),
                'input_valid':np.ones(grid.shape,bool),'valid':np.ones(grid.shape,bool)}
        annotate_result(m,result)
        np.testing.assert_array_equal(result['seam_search_known'],False)
        np.testing.assert_array_equal(result['seam_crossing'],[[True],[False]])

    def test_native_matching_merged_geocoding_and_exact_per_source_intervals(self):
        root,m=assemble(self.cfg,self.sources)
        cfg=deepcopy(self.cfg);cfg.pop('aoi');cfg['inputs']['burst_id']='merged'
        match(cfg,root,m)
        with h5py.File(root/'radar/offsets.h5') as h:
            valid=h['valid'][:];seam=h['seam_crossing'][:];owners=h['source_burst_index'][:];dt=h['interval_seconds'][:]
            self.assertTrue((valid&seam).any())
            self.assertTrue((valid&~seam).any())
            for k,axis in enumerate(('azimuth','range')):
                np.testing.assert_allclose(h[f'total/{axis}_px'][:][valid],(2,-3)[k],atol=.15)
            np.testing.assert_array_equal(dt[owners==1],86400.)
            np.testing.assert_array_equal(dt[owners==2],86401.)
        geocode(cfg,root,m)
        directory=root/'geo'
        source=gdal.Open(str(directory/'source_burst_index.tif')).ReadAsArray()
        intervals=gdal.Open(str(directory/'pair_interval_seconds.tif')).ReadAsArray()
        np.testing.assert_array_equal(intervals[source==1],86400.)
        np.testing.assert_array_equal(intervals[source==2],86401.)
        for axis in ('range','azimuth'):
            disp=gdal.Open(str(directory/f'{axis}_displacement_m.tif')).ReadAsArray()
            vel=gdal.Open(str(directory/f'{axis}_velocity_m_per_day.tif')).ReadAsArray()
            np.testing.assert_allclose(vel,disp/(intervals/86400),rtol=2e-6,equal_nan=True)
        self.assertTrue((directory/'seam_crossing_mask.tif').is_file())
        paths=[root/'radar/offsets.h5',*directory.iterdir()]
        before={p:(checksum(p),p.stat().st_mtime_ns) for p in paths}
        with patch('glacier_offsets.ampcor.native_match',side_effect=AssertionError('cache rerun')):
            _,again=assemble(cfg,self.sources);match(cfg,root,again);geocode(cfg,root,again)
        self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in paths})

    def test_variable_interval_velocity_variance_scaled_before_interpolation(self):
        x,y=np.meshgrid(np.arange(3)*100.,np.arange(3)*100.)
        mapper=ConservativeMapper(x,y,np.ones(x.shape,bool),25)
        intervals=86400*(1+np.indices(x.shape)[0])
        covariance=np.ones((*x.shape,3));scale=np.full(x.shape,3.)
        result=formal_uncertainties(covariance,2.,scale,intervals,mapper)
        expected=np.sqrt(mapper.map(9./(intervals/86400)**2,variance=True))
        np.testing.assert_array_equal(result['azimuth_velocity_sigma_m_per_day'],expected)
        fields=physical_fields(np.ones(x.shape),np.ones(x.shape),2.,scale,intervals)
        np.testing.assert_array_equal(fields['range_velocity_m_per_day'],2/(intervals/86400))

    def test_velocity_prior_uses_each_original_burst_grid_and_interval(self):
        root,m=assemble(self.cfg,self.sources)
        path=self.root/'velocity.tif'
        ds=gdal.GetDriverByName('GTiff').Create(str(path),100,100,1,gdal.GDT_Float32)
        ds.SetProjection(CRS.from_epsg(3031).to_wkt());ds.SetGeoTransform((-1010000,500,0,-990000,0,-500))
        ds.GetRasterBand(1).Fill(0);ds.GetRasterBand(1).SetUnitType('m/day');ds=None
        cfg=deepcopy(self.cfg);cfg['gross_offsets'].update(mode='velocity',velocity={
            'vx':{'path':str(path),'band':1},'vy':{'path':str(path),'band':1},'basis':'east_north',
            'units':'m/day','interpolation':'bilinear','product':'synthetic','version':'1'})
        cfg['inputs']['burst_id']='merged';cfg.pop('aoi')
        seen=[]
        def prediction(lon,lat,height,east,north,grid,orbit,valid):
            self.assertEqual(grid.length,192)
            seen.append(grid.sensing_start)
            return np.zeros((*lon.shape,2)),valid
        with patch('glacier_offsets.velocity_prior.radar_prediction',side_effect=prediction):
            match(cfg,root,m,native_pair=self.pairs)
        self.assertEqual(seen,[0.,128/500])
        with h5py.File(root/'radar/offsets.h5') as h:
            self.assertTrue(h['prior/valid'][:].any())
            metadata=json.loads(h['prior'].attrs['metadata_json'])
            self.assertEqual([s['metadata']['interval_seconds'] for s in metadata['sources']],[86400.,86401.])


if __name__=='__main__':unittest.main()

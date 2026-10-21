"""Native CPU motion/masks agree exactly with independent stage runs.

Coregistered synthetic SLC/geometry fixtures replace COMPASS; all matching,
M5 preparation (except orbit projection), geocoding and caching code runs.
Named ancillary products use the documented numerical-comparison policy.
"""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal
from pyproj import CRS
from scipy.ndimage import gaussian_filter, shift
from shapely.geometry import box

from glacier_offsets.batch import execute_batch, effective_config
from glacier_offsets.comparison import ANCILLARY_MAP, compare_array
from glacier_offsets.config import load_defaults
from glacier_offsets.inputs import Burst, select_pair
from glacier_offsets.pipeline import execute
from glacier_offsets.products import identity, input_provenance, checksum, write_json
from test_aoi import pair
from test_synthetic_offsets import write_slc


class BatchScienceTests(unittest.TestCase):
    def test_native_zero_velocity_standalone_and_restart_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'input';source.write_text('synthetic acquisition')
            cfg=load_defaults();cfg['inputs'].update({k:str(source) for k in (
                'reference_safe','secondary_safe','reference_orbit','secondary_orbit','dem')})
            cfg['burst_merge']['enabled']=False  # Standalone equivalence is the legacy independent mode.
            cfg['inputs']['polarization']='HH'
            cfg['matching'].update(backend='cpu',window=[32,32],search=[8,8],stride=[32,32],zoom=8,deramp=0)
            cfg['geocoding'].update(epsg=3031,spacing_m=100)
            cfg['aoi']={'bbox':[-1,-1,6,3]}
            rows=[pair('a'),pair('b',box(3,0,5,2))]
            # Different intervals, radar sizes, start times, range spacing and PRF.
            b=rows[1]
            rows[1]=(b[0],Burst(b[1].id,b[1].swath,b[0].start+timedelta(days=6,microseconds=250000),b[1].native))
            fixtures={}
            for i,pair_ in enumerate(rows):
                shape=(160+32*i,192+32*i);grid=dict(length=shape[0],width=shape[1],prf=1000+i,
                    range_pixel_spacing=2.3+i,sensing_start=100+i,starting_range=800000+i)
                native=pair_[0].native
                native.as_isce3_radargrid=lambda grid=grid:type('Grid',(),grid)()
                pair_[1].native.as_isce3_radargrid=native.as_isce3_radargrid
                native.orbit=object()
                folder=d/pair_[0].id;folder.mkdir()
                rng=np.random.default_rng(85+i)
                amp=gaussian_filter(rng.uniform(.1,1,shape),.6)
                ref=amp*np.exp(1j*rng.normal(0,.2,shape))
                write_slc(folder/'ref.tif',ref)
                write_slc(folder/'sec.tif',shift(ref,(2,-3),order=3,mode='constant',cval=0))
                line,sample=np.indices(shape)
                geometry={}
                for name,data in {'x':sample*10.,'y':-2e6-line*10.,'z':np.zeros(shape),
                                  'layover_shadow_mask':np.zeros(shape)}.items():
                    path=folder/(name+'.tif')
                    ds=gdal.GetDriverByName('GTiff').Create(str(path),shape[1],shape[0],1,gdal.GDT_Float64)
                    ds.GetRasterBand(1).WriteArray(data);ds=None;geometry[name]=str(path)
                fixtures[pair_[0].id]={'reference_grid':grid,'geometry':geometry,
                    'geometry_crs':CRS.from_epsg(3031).to_wkt(),
                    'slcs':{'reference':str(folder/'ref.tif'),'secondary':str(folder/'sec.tif')}}
            velocity=d/'velocity.tif'
            ds=gdal.GetDriverByName('GTiff').Create(str(velocity),100,100,1,gdal.GDT_Float32)
            ds.SetProjection(CRS.from_epsg(3031).to_wkt());ds.SetGeoTransform((-1000,100,0,-1999000,0,-100))
            ds.GetRasterBand(1).Fill(0);ds.GetRasterBand(1).SetUnitType('m/day');ds=None
            def coreg(effective,force=False,context=None):
                p=context.pair(effective['inputs']['burst_id']) if context else select_pair(effective,rows)
                root=Path(effective['output']['directory'])/p[0].id;root.mkdir(parents=True,exist_ok=True)
                path=root/'manifest.json'
                if path.exists() and not force:return root,json.loads(path.read_text())
                provenance=context.provenance if context else input_provenance(effective)
                spec={'inputs':provenance,'polarization':'HH','burst_id':p[0].id}
                manifest={'coreg_identity':identity(spec),'coreg_specification':spec,'software':{'packages':{}},
                          'coreg':fixtures[p[0].id],'stages':{'coregister':'complete'},
                          'interval_seconds':(p[1].start-p[0].start).total_seconds(),
                          'acquisition_times':[b.start.isoformat() for b in p]}
                write_json(path,manifest);return root,manifest
            def projection(lon,lat,height,east,north,grid,orbit,valid):
                expected=rows[0 if grid.length==160 else 1][0].native
                self.assertIs(orbit,expected.orbit)
                return np.zeros((*lon.shape,2)),valid
            with patch('glacier_offsets.inputs.discover_validated',return_value=rows), \
                    patch('glacier_offsets.inputs.discover',return_value=rows), \
                    patch('glacier_offsets.batch.dem_extent',return_value=(box(-10,-10,10,10),CRS.from_epsg(4326))), \
                    patch('glacier_offsets.batch.inspect_environment',return_value={'packages':{}}), \
                    patch('glacier_offsets.pipeline.coregister',side_effect=coreg), \
                    patch('glacier_offsets.velocity_prior.radar_prediction',side_effect=projection) as projected:
                for mode in ('zero','velocity'):
                    cfg['gross_offsets']['mode']=mode
                    if mode=='velocity':
                        cfg['gross_offsets']['velocity']={'vx':{'path':str(velocity),'band':1},
                            'vy':{'path':str(velocity),'band':1},'basis':'east_north','units':'m/day',
                            'interpolation':'bilinear','product':'synthetic zero','version':'1'}
                    output=d/('batch-'+mode);cfg['output']={'directory':str(output)}
                    self.assertEqual(execute_batch(cfg,'coregister'),0)
                    self.assertEqual(execute_batch(cfg,'match'),0)
                    self.assertEqual(execute_batch(cfg,'geocode'),0)
                    for pair_ in rows:
                        key=pair_[0].id
                        single=effective_config(cfg,key);single['output']['directory']=str(d/('single-'+mode))
                        execute(single,'coregister');execute(single,'match');execute(single,'geocode')
                        a=output/key;b=Path(single['output']['directory'])/key
                        with h5py.File(a/'radar/offsets.h5') as x,h5py.File(b/'radar/offsets.h5') as y:
                            for name in ('center_line','center_sample','valid','input_valid','total/azimuth_px','total/range_px'):
                                np.testing.assert_array_equal(x[name][:],y[name][:])
                            self.assertTrue(x['valid'][:].any())
                        for product in (a/'geo').glob('*.tif'):
                            x=gdal.Open(str(product));y=gdal.Open(str(b/'geo'/product.name))
                            result=compare_array(x.ReadAsArray(),y.ReadAsArray(),ancillary=product.name in ANCILLARY_MAP)
                            self.assertNotEqual(result['status'],'failed',(product.name,result))
                            self.assertEqual(x.GetGeoTransform(),y.GetGeoTransform())
                        manifest=json.loads((a/'manifest.json').read_text())
                        self.assertEqual(manifest['interval_seconds'],(pair_[1].start-pair_[0].start).total_seconds())
                        self.assertEqual(manifest['coreg']['reference_grid']['length'],fixtures[key]['reference_grid']['length'])
                    paths=[p for key in ('a','b') for p in [output/key/'radar/offsets.h5', *sorted((output/key/'geo').glob('*.tif'))]]
                    before={str(p):(checksum(p),p.stat().st_mtime_ns) for p in paths}
                    cfg['aoi']={'bbox':[3.1,.1,4,1]}
                    self.assertEqual(execute_batch(cfg,'match'),0)
                    self.assertEqual(execute_batch(cfg,'geocode'),0)
                    self.assertEqual(json.loads((output/'batch_manifest.json').read_text())['selected_ids'],['b'])
                    self.assertEqual(before,{str(p):(checksum(p),p.stat().st_mtime_ns) for p in paths})
                    cfg['aoi']={'bbox':[-1,-1,6,3]}
                    self.assertEqual(execute_batch(cfg,'match'),0)
                self.assertEqual(projected.call_count,4)  # two batch and two standalone velocity preparations

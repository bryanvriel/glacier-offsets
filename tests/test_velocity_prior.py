"""M5 sampling, units, projection basis, rounding and native varying-gross tests."""
import json
import tempfile
from pathlib import Path
import unittest
import h5py
import numpy as np
from pyproj import CRS, Transformer
from osgeo import gdal,osr
from glacier_offsets.velocity_prior import (sample_component,displacement_enu,round_predictions,validate_velocity,radar_prediction)
from glacier_offsets.ampcor import native_match,write_offsets,matching_specification
from glacier_offsets.config import DEFAULT_MATCHING
from glacier_offsets.matching_grid import build_grid
from test_synthetic_offsets import write_slc
from scipy.ndimage import gaussian_filter,shift


class VelocityTests(unittest.TestCase):
    def test_cf_sampling_nodata_coordinate_centers_and_dimension_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'velocity.nc'
            with h5py.File(p,'w') as h:
                h['x']=[0.,100.,200.];h['y']=[-2000000.,-2000100.,-2000200.]
                c=h.create_dataset('crs',data=0)
                for key,value in CRS.from_epsg(3031).to_cf().items():c.attrs[key]=value
                h['VX']=np.array([[1,2,3],[4,-999,6],[7,8,9]],float)
                h['VX'].attrs.update(grid_mapping='crs',units='meter/year',_FillValue=-999.)
            xx=np.array([[0.,50.,100.,-10.,200.]])
            yy=np.array([[-2000000.,-2000000.,-2000100.,-2000000.,-2000200.]])
            lon,lat=Transformer.from_crs(3031,4326,always_xy=True).transform(xx,yy)
            spec={'path':str(p),'variable':'VX'}
            a,_,_=sample_component(spec,lon,lat,'bilinear','m/year')
            np.testing.assert_allclose(a[0,:2],[1.,1.5],atol=1e-6)
            self.assertTrue(np.isnan(a[0,2:4]).all());self.assertAlmostEqual(a[0,4],9.)
            with self.assertRaisesRegex(ValueError,'units metadata'):sample_component(spec,lon,lat,'nearest','m/day')
            with h5py.File(p,'a') as h:h['VX'].attrs['_FillValue']=0.;h['VX'][1,1]=0.
            a,_,_=sample_component(spec,lon,lat,'nearest','m/year')
            self.assertTrue(np.isnan(a[0,2]))

    def test_gdal_sampling_half_pixel_and_mask(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'vx.tif';ds=gdal.GetDriverByName('GTiff').Create(str(p),3,3,1,gdal.GDT_Float32)
            s=osr.SpatialReference();s.ImportFromEPSG(3031);ds.SetProjection(s.ExportToWkt())
            ds.SetGeoTransform((-50,100,0,-1999950,0,-100));b=ds.GetRasterBand(1)
            b.WriteArray(np.arange(1,10).reshape(3,3));b.SetNoDataValue(5);b.SetUnitType('m/day');ds=None
            lon,lat=Transformer.from_crs(3031,4326,always_xy=True).transform(np.array([[50.,100.]]),np.array([[-2000000.,-2000100.]]))
            a,_,_=sample_component({'path':str(p),'band':1},lon,lat,'bilinear','m/day')
            self.assertAlmostEqual(a[0,0],1.5,places=5);self.assertTrue(np.isnan(a[0,1]))

    def test_basis_interval_signed_zero_and_projected_scale(self):
        lon=np.array([[-100.,30.]]);lat=np.array([[-75.,-80.]])
        vx=np.array([[365.25,-730.5]]);vy=np.array([[0.,365.25]])
        c=CRS.from_epsg(3031);s={'basis':'east_north','units':'m/year','year_days':365.25}
        e,n=displacement_enu(vx,vy,lon,lat,c,s,2*86400.)
        np.testing.assert_allclose(e,[[2.,-4.]]);np.testing.assert_allclose(n,[[0.,2.]])
        ground=s|{'basis':'map_xy','map_units':'ground'}
        eg,ng=displacement_enu(vx,vy,lon,lat,c,ground,2*86400.)
        np.testing.assert_allclose(np.hypot(eg,ng),np.hypot(e,n),rtol=1e-7)
        # EPSG3031 positive x at longitude -100 points mostly south;
        # positive y points mostly west. This checks rotation and signed axes.
        self.assertLess(ng[0,0],-1.9);self.assertLess(eg[0,0],0.)
        doubled=displacement_enu(vx,vy,lon,lat,c,ground,4*86400.)
        np.testing.assert_allclose(doubled[0],2*eg);np.testing.assert_allclose(doubled[1],2*ng)
        ep,np_=displacement_enu(vx,vy,lon,lat,c,ground|{'map_units':'projected'},2*86400.)
        self.assertGreater(np.max(np.abs(np.hypot(ep,np_)-np.hypot(eg,ng))),.01)
        z=displacement_enu(vx*0,vy*0,lon,lat,c,ground,86400.)
        np.testing.assert_array_equal(z[0],np.zeros_like(vx));np.testing.assert_array_equal(z[1],np.zeros_like(vy))
        # Independent forward projection of EN steps, normalize physical map axes.
        from glacier_offsets.velocity_prior import GEOD
        x,y=Transformer.from_crs(4326,c,always_xy=True).transform(lon,lat)
        le,la,_=GEOD.fwd(lon,lat,np.full(lon.shape,90.),np.ones(lon.shape))
        xe,ye=Transformer.from_crs(4326,c,always_xy=True).transform(le,la)
        ln,la,_=GEOD.fwd(lon,lat,np.zeros(lon.shape),np.ones(lon.shape))
        xn,yn=Transformer.from_crs(4326,c,always_xy=True).transform(ln,la)
        scale=np.hypot(xe-x,ye-y)
        np.testing.assert_allclose(((xe-x)*eg+(xn-x)*ng)/scale,e,atol=1e-5)
        np.testing.assert_allclose(((ye-y)*eg+(yn-y)*ng)/scale,n,atol=1e-5)

    def test_rounding_signed_halves_nan_overflow(self):
        p=np.array([[[.5,-.5],[1.49,-1.51],[np.nan,2.],[2**31,0.],[0.,0.]]])
        integer,valid=round_predictions(p,np.ones((1,5),bool))
        np.testing.assert_array_equal(integer[0,[0,1,4]],[[1,-1],[1,-2],[0,0]])
        np.testing.assert_array_equal(valid,[[True,True,False,False,True]])
        self.assertTrue((integer[~valid]==np.iinfo(np.int32).min).all())

    def test_ambiguous_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file';p.touch()
            s={'vx':{'path':str(p),'band':1},'vy':{'path':str(p),'band':1},'basis':'map_xy',
               'units':'m/year','product':'test','version':'v1','interpolation':'bilinear'}
            with self.assertRaisesRegex(ValueError,'year_days'):validate_velocity(s)
            with self.assertRaisesRegex(ValueError,'map_units'):validate_velocity(s|{'year_days':365.25})
            validate_velocity(s|{'year_days':365.25,'map_units':'ground'})
            with self.assertRaisesRegex(ValueError,'units must'):validate_velocity(s|{'units':'km/year'})

    def measure(self,translation,prior_values,invalid=False,backend='cpu'):
        rng=np.random.default_rng(95);amp=gaussian_filter(rng.uniform(.1,1,(256,256)),.6)
        ref=amp*np.exp(1j*rng.normal(0,.2,amp.shape));sec=shift(ref,translation,order=3,mode='constant',cval=0)
        p=DEFAULT_MATCHING|{'window':[48,48],'search':[12,12],'stride':[64,64],'batch':[1,2],
                           'first_center':[71.5,71.5],'zoom':8,'deramp':0}
        grid=build_grid(ref.shape,sec.shape,p['window'],p['search'],p['stride'],first_center=p['first_center'])
        values=np.broadcast_to(prior_values,(*grid.shape,2)).copy().astype('int32')
        valid=np.ones(grid.shape,bool);flags=np.zeros(grid.shape,'uint8')
        if invalid:
            valid[0,0]=False;values[0,0]=np.iinfo(np.int32).min;flags[0,0]=2
            values[0,1]=[10000,-10000]  # must be masked before native C++
        prior={'grid':grid,'integer_px':values,'valid':valid,'rejection_flags':flags}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);write_slc(d/'ref.tif',ref);write_slc(d/'sec.tif',sec)
            r=native_match(d/'ref.tif',d/'sec.tif',d/'varying',p,{'gross_offsets':{'mode':'velocity'}},backend,prior=prior)
            zero=native_match(d/'ref.tif',d/'sec.tif',d/'zero',p,{'gross_offsets':{'mode':'zero'}},'cpu')
        return r,zero

    def test_native_spatially_varying_and_biased_priors_recover_total(self):
        prior=np.array([[[2,-3],[0,0],[-2,3]],[[1,2],[-1,-2],[4,-5]],[[0,0],[3,-4],[-3,4]]])
        for translation in [(3,-4),(-3,4),(0,0)]:
            r,z=self.measure(translation,prior)
            np.testing.assert_array_equal(r['gross'],prior)
            np.testing.assert_allclose(r['total'],np.broadcast_to(translation,r['total'].shape),atol=.15)
            np.testing.assert_allclose(r['total'],z['total'],atol=.15)
            np.testing.assert_array_equal(r['total'],r['gross']+r['residual'])
            self.assertTrue(r['valid'].all())

    def test_missing_and_outside_priors_are_not_zero_fallback(self):
        r,_=self.measure((3,-4),(2,-3),invalid=True)
        self.assertFalse(r['input_valid'][0,:2].any());self.assertTrue(np.isnan(r['gross'][0,:2]).all())
        self.assertEqual(r['prior']['rejection_flags'][0,0],2);self.assertEqual(r['prior']['rejection_flags'][0,1],16)
        np.testing.assert_allclose(r['total'][r['valid']],np.broadcast_to([3,-4],r['total'][r['valid']].shape),atol=.15)

    def test_large_translation_velocity_search_recovery(self):
        r,z=self.measure((18,-17),(17,-16))
        np.testing.assert_allclose(r['total'][r['valid']],np.broadcast_to([18,-17],r['total'][r['valid']].shape),atol=.15)
        self.assertTrue(r['valid'].any())
        self.assertGreater(np.max(np.abs(np.median(z['total'],axis=(0,1))-[18,-17])),3.)

    def test_prior_product_metadata_and_source_identity(self):
        r,_=self.measure((3,-4),(2,-3),invalid=True)
        prior=r['prior'];prior['metadata']={'settings':{'units':'m/day'},'missing_policy':'mask'}
        for key in ['vx','vy','east_displacement_m','north_displacement_m','longitude_deg','latitude_deg','height_m']:
            prior[key]=np.ones(r['grid'].shape)
        prior['predicted_px']=prior['integer_px'].astype(float)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);write_offsets(d/'offsets.h5',r,{'matching':DEFAULT_MATCHING})
            with h5py.File(d/'offsets.h5') as h:
                self.assertEqual(h.attrs['mergeGrossOffset'],0)
                self.assertIn('exactly once',h.attrs['total_definition'])
                self.assertEqual(h['prior/vx'].attrs['units'],'m/day')
                self.assertEqual(h['prior/integer_px/azimuth'].dtype,np.dtype('int32'))
                np.testing.assert_array_equal(h['prior/rejection_flags'][:],prior['rejection_flags'])
                self.assertTrue(np.isnan(h['gross/azimuth_px'][0,0]))
            p=d/'source';p.write_text('one')
            s={'vx':{'path':str(p),'band':1},'vy':{'path':str(p),'band':1}}
            cfg={'matching':DEFAULT_MATCHING,'gross_offsets':{'mode':'velocity','velocity':s}}
            a=matching_specification(cfg,{'coreg_identity':'same'})
            p.write_text('two');b=matching_specification(cfg,{'coreg_identity':'same'})
            self.assertNotEqual(a['velocity_provenance'],b['velocity_provenance'])

    def test_native_gpu_consistency(self):
        from glacier_offsets.environment import select_backend
        try:select_backend('gpu')
        except RuntimeError as exc:self.skipTest(str(exc))
        cpu,_=self.measure((3,-4),(2,-3));gpu,_=self.measure((3,-4),(2,-3),backend='gpu')
        mask=cpu['valid']&gpu['valid'];self.assertTrue(mask.any())
        np.testing.assert_allclose(cpu['total'][mask],gpu['total'][mask],atol=.25)

    def test_all_missing_prior_runs_no_native_windows(self):
        grid=build_grid((128,128),(128,128),(32,32),(8,8),(32,32))
        prior={'grid':grid,'integer_px':np.full((*grid.shape,2),np.iinfo(np.int32).min,dtype='int32'),
               'valid':np.zeros(grid.shape,bool),'rejection_flags':np.full(grid.shape,2,'uint8')}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);write_slc(d/'ref.tif',np.ones((128,128),complex));write_slc(d/'sec.tif',np.ones((128,128),complex))
            params=DEFAULT_MATCHING|{'window':[32,32],'search':[8,8],'stride':[32,32],'zoom':8}
            # select_backend may instantiate for compatibility checks elsewhere;
            # no runAmpcor call or segment directory is allowed here.
            r=native_match(d/'ref.tif',d/'sec.tif',d/'out',params,{'gross_offsets':{'mode':'velocity'}},'cpu',prior=prior)
            self.assertEqual(r['effective_parameters']['segments'],[])
            self.assertFalse(r['valid'].any());self.assertTrue(np.isnan(r['total']).all())
            self.assertFalse(any((d/'out').glob('row-*')))

    def test_auto_runtime_fallback_reruns_whole_varying_field(self):
        from unittest.mock import patch
        from glacier_offsets.ampcor import varying_match
        from glacier_offsets.environment import select_backend
        cpu_class=select_backend('cpu')[1]
        grid=build_grid((128,128),(128,128),(32,32),(8,8),(32,32))
        prior={'grid':grid,'integer_px':np.zeros((*grid.shape,2),'int32'),
               'valid':np.ones(grid.shape,bool),'rejection_flags':np.zeros(grid.shape,'uint8')}
        params=DEFAULT_MATCHING|{'window':[32,32],'search':[8,8],'stride':[32,32],'zoom':8}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);rng=np.random.default_rng(45);a=rng.uniform(.1,1,(128,128)).astype(complex)
            write_slc(d/'ref.tif',a);write_slc(d/'sec.tif',a)
            ref,sec=gdal.Open(str(d/'ref.tif')),gdal.Open(str(d/'sec.tif'))
            def fake_backend(requested):return ('gpu' if requested=='auto' else 'cpu'),cpu_class
            def cpu_fallback(reference,secondary,output,p,cfg,backend,**kwargs):
                return native_match(reference,secondary,output,p,cfg,'cpu',**kwargs)
            with patch('glacier_offsets.ampcor.select_backend',side_effect=fake_backend),patch('glacier_offsets.ampcor.native_match',side_effect=cpu_fallback) as runner:
                r=varying_match(d/'ref.tif',d/'sec.tif',d/'out',params,{'gross_offsets':{'mode':'velocity'}},'auto',prior,ref,sec)
                self.assertEqual(runner.call_count,grid.shape[0]+1)
            self.assertEqual(r['backend'],'cpu')
            expected=varying_match(d/'ref.tif',d/'sec.tif',d/'cpu',params,{'gross_offsets':{'mode':'velocity'}},'cpu',prior,ref,sec)
            np.testing.assert_array_equal(r['total'],expected['total'])
            np.testing.assert_allclose(r['total'],0.,atol=.15)

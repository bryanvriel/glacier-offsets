import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from osgeo import gdal
from glacier_offsets.geocode import ConservativeMapper, RadarNearestMapper, formal_uncertainties, sample_geometry, write_geotiff

class GeocodeTests(unittest.TestCase):
    def test_nearest_preserves_isolated_sample_without_borrowing_from_good_samples(self):
        x,y=np.meshgrid(np.arange(5)*100.,np.arange(5)*100.)
        good=np.zeros(x.shape,bool);good[2,2]=True
        nearest=RadarNearestMapper(x,y,good,25)
        linear=ConservativeMapper(x,y,good,25)
        self.assertFalse(linear.valid.any())
        gt=nearest.transform
        xx,yy=np.meshgrid(gt[0]+(np.arange(nearest.shape[1])+.5)*25,
                          gt[3]-(np.arange(nearest.shape[0])+.5)*25)
        expected=(np.abs(xx-200)<50)&(np.abs(yy-200)<50)
        np.testing.assert_array_equal(nearest.valid,expected)
        out=nearest.map(np.where(good,-7.,99.))
        np.testing.assert_array_equal(out[expected],-7.)
        self.assertTrue(np.isnan(out[~expected]).all())
        zero=nearest.map(np.zeros(x.shape))
        np.testing.assert_array_equal(zero[expected],0)
        # No averaging reduces covariance when just one observation is copied.
        covariance=np.ones((*x.shape,3));covariance[...,0]=4;covariance[...,1]=9
        sigma=formal_uncertainties(covariance,2.,np.full(x.shape,3.),86400,nearest)
        np.testing.assert_array_equal(sigma['azimuth_displacement_sigma_m'][expected],6.)
        np.testing.assert_array_equal(sigma['range_velocity_sigma_m_per_day'][expected],6.)

    def test_nearest_uses_radar_coordinates_on_rotated_sheared_geometry(self):
        row,col=np.indices((5,6));x=100*col+70*row;y=30*col+200*row
        good=np.ones(x.shape,bool);good[2,3]=False
        mapper=RadarNearestMapper(x,y,good,25)
        gt=mapper.transform
        xx,yy=np.meshgrid(gt[0]+(np.arange(mapper.shape[1])+.5)*25,
                          gt[3]-(np.arange(mapper.shape[0])+.5)*25)
        inverse=np.linalg.inv([[100,70],[30,200]])
        fractional_col=inverse[0,0]*xx+inverse[0,1]*yy
        fractional_row=inverse[1,0]*xx+inverse[1,1]*yy
        rr=np.floor(fractional_row+.5).astype(int);cc=np.floor(fractional_col+.5).astype(int)
        inside=(fractional_row>=0)&(fractional_row<=4)&(fractional_col>=0)&(fractional_col<=5)
        expected=inside&~((rr==2)&(cc==3))
        np.testing.assert_array_equal(mapper.valid,expected)
        values=1000*row+col
        actual=mapper.map(values)
        np.testing.assert_array_equal(actual[mapper.valid],(1000*rr+cc)[mapper.valid])
        self.assertTrue(np.isnan(actual[~expected]).all())

    def test_geocode_mask_policy_units_quality_and_restart(self):
        from glacier_offsets.geocode import geocode
        from glacier_offsets.products import checksum
        row,col=np.indices((5,5));quality=np.zeros((5,5),bool);quality[2,2]=True
        flags=np.full((5,5),16,dtype='uint8');flags[2,2]=0
        flags[2,3]=4;flags[0,0]=1;flags[4,4]=2
        input_valid=np.ones((5,5),bool);input_valid[0,0]=False
        geometry=np.ones((5,5),bool);geometry[1,1]=False
        lon=-100+col*.003;lat=-75+row*.003
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'radar').mkdir()
            offsets=root/'radar/offsets.h5'
            with h5py.File(offsets,'w') as h:
                for name,array in [('center_line',row*32+31.5),('center_sample',col*128+127.5),
                                   ('valid',quality),('input_valid',input_valid),('rejection_flags',flags),
                                   ('total/range_px',np.full((5,5),2.)),('total/azimuth_px',np.full((5,5),-3.)),
                                   ('covariance',np.ones((5,5,3))),('snr',np.full((5,5),8.)),
                                   ('correlation_peak',np.where(quality,.8,.1))]:h.create_dataset(name,data=array)
            digest=checksum(offsets)
            manifest={'match_identity':'fixture','stages':{},'interval_seconds':2*86400,
                      'matching':{'matching':{'snr_threshold':2.,'correlation_threshold':.4}},
                      'coreg':{'reference_grid':{'range_pixel_spacing':2.}},
                      'coreg_specification':{'inputs':{}},'software':{'packages':{}},
                      'acquisition_times':['2020-01-01','2020-01-03']}
            cfg={'inputs':{'burst_id':'synthetic','polarization':'HH'},
                 'geocoding':{'epsg':3031,'spacing_m':25,'resampling':'nearest','mask':'quality'}}
            masks={}
            for policy in ['quality','input','default']:
                if policy=='default':
                    cfg['geocoding'].pop('mask');cfg['geocoding'].pop('resampling')
                else:
                    cfg['geocoding']['mask']=policy
                with patch('glacier_offsets.geocode.center_geometry',return_value=(lon,lat,np.zeros((5,5)),np.full((5,5),3.),geometry)):
                    geocode(cfg,root,manifest)
                masks[policy]=gdal.Open(str(root/'geo/valid_mask.tif')).ReadAsArray().astype(bool)
                q=gdal.Open(str(root/'geo/quality_valid_mask.tif')).ReadAsArray().astype(bool)
                np.testing.assert_array_equal(q,masks.get('quality',q))
                self.assertTrue(q.any());self.assertTrue(np.all(~q|masks[policy]))
                for name,value in [('range_displacement_m',4.),('azimuth_displacement_m',-9.),
                                   ('range_velocity_m_per_day',2.),('azimuth_velocity_m_per_day',-4.5),
                                   ('range_displacement_sigma_m',2.),('azimuth_displacement_sigma_m',3.)]:
                    ds=gdal.Open(str(root/'geo'/f'{name}.tif'));a=ds.ReadAsArray()
                    np.testing.assert_array_equal(a[masks[policy]],value)
                    self.assertTrue(np.isnan(a[~masks[policy]]).all())
                    self.assertEqual(ds.GetMetadataItem('source_mask'),'input' if policy=='default' else policy)
                    self.assertEqual(json.loads(ds.GetMetadataItem('matching_quality_thresholds_json')),
                                     {'snr_threshold':2.,'correlation_threshold':.4})
                self.assertEqual(checksum(offsets),digest)
                peaks=gdal.Open(str(root/'geo/correlation_peak.tif')).ReadAsArray()
                snr=gdal.Open(str(root/'geo/snr.tif')).ReadAsArray()
                np.testing.assert_array_equal(np.isfinite(peaks),masks[policy])
                np.testing.assert_array_equal(np.isfinite(snr),masks[policy])
                if policy!='quality':self.assertTrue(np.any(np.isclose(peaks,.1)))
            self.assertGreater(masks['input'].sum(),masks['quality'].sum())
            np.testing.assert_array_equal(masks['default'],masks['input'])
            # A geocoding-only change/restart never needs matching or recomputes geometry.
            with patch('glacier_offsets.geocode.center_geometry',side_effect=RuntimeError('cached')) as geometry_call:
                geocode(cfg,root,manifest);geometry_call.assert_not_called()
            # Explicit regeneration of legacy stack maps upgrades only the
            # inventory; physical displacements and their identity stay intact.
            old_identity = manifest['geocode_identity']
            old_displacement = gdal.Open(str(root/'geo/range_displacement_m.tif')).ReadAsArray()
            manifest['stack_pair'] = {'geometry_reference_date': '2020-01-01',
                'geometry_reference_burst_times': {}, 'geometry_reference_inputs': {},
                'acquisition_dates': ['2020-01-01', '2020-01-03']}
            with patch('glacier_offsets.geocode.center_geometry',return_value=(lon,lat,np.zeros((5,5)),np.full((5,5),3.),geometry)):
                geocode(cfg,root,manifest,force=True)
            self.assertEqual(old_identity, manifest['geocode_identity'])
            self.assertEqual(manifest['geocoding']['product_profile'], 'displacement-v1')
            self.assertFalse(list((root/'geo').glob('*velocity*.tif')))
            np.testing.assert_array_equal(gdal.Open(str(root/'geo/range_displacement_m.tif')).ReadAsArray(), old_displacement)

    def test_nearest_missing_geometry_column_remains_nodata(self):
        x,y=np.meshgrid(np.arange(5)*100.,np.arange(4)*100.)
        x[:,2]=np.nan;y[:,2]=np.nan
        mapper=RadarNearestMapper(x,y,np.ones(x.shape,bool),25)
        out=mapper.map(np.ones(x.shape))
        xx=mapper.transform[0]+(np.arange(out.shape[1])+.5)*25
        missing=(xx>150)&(xx<250)
        self.assertTrue(np.isnan(out[:,missing]).all())
        self.assertTrue(np.isfinite(out[:,~missing]).all())

    def test_formal_uncertainty_units_components_and_squared_weights(self):
        # One right triangle permits an independent analytical barycentric solution.
        x=np.array([[0.,4.],[0.,np.nan]]);y=np.array([[0.,0.],[4.,np.nan]])
        good=np.array([[True,True],[True,False]])
        mapper=ConservativeMapper(x,y,good,1)
        cov=np.zeros((2,2,3))
        cov[...,0]=[[1.,4.],[9.,99.]]  # azimuth variance
        cov[...,1]=[[4.,9.],[16.,99.]] # range variance, deliberately different
        cov[...,2]=0.5               # cross term does not enter marginal variances
        scale=np.array([[2.,3.],[4.,5.]])
        interval=6*86400+1
        fields=formal_uncertainties(cov,2.3,scale,interval,mapper)
        xx,yy=np.meshgrid(np.arange(4)+.5,4-(np.arange(4)+.5))
        w0=1-(xx+yy)/4;w1=xx/4;w2=yy/4
        expected_az=np.sqrt(w0**2*1*2**2+w1**2*4*3**2+w2**2*9*4**2)
        expected_rg=np.sqrt((w0**2*4+w1**2*9+w2**2*16)*2.3**2)
        for axis,expected in [('azimuth',expected_az),('range',expected_rg)]:
            actual=fields[f'{axis}_displacement_sigma_m']
            np.testing.assert_allclose(actual[mapper.valid],expected[mapper.valid],rtol=1e-6)
            self.assertTrue(np.isnan(actual[~mapper.valid]).all())
            np.testing.assert_allclose(fields[f'{axis}_velocity_sigma_m_per_day'][mapper.valid],
                                       expected[mapper.valid]/(interval/86400),rtol=1e-6)
        zero=formal_uncertainties(np.zeros_like(cov),2.3,scale,interval,mapper)
        for array in zero.values():np.testing.assert_array_equal(array[mapper.valid],0)
        empty=ConservativeMapper(x,y,np.zeros_like(good),1)
        self.assertTrue(np.isnan(formal_uncertainties(cov,2.3,scale,interval,empty)['range_displacement_sigma_m']).all())

    def test_linear_geometry_positions_and_holes(self):
        x,y=np.meshgrid(np.arange(5)*100.,np.arange(4)*100.)
        good=np.ones(x.shape,bool)
        mapper=ConservativeMapper(x,y,good,50)
        out=mapper.map(2*x+3*y)
        gt=mapper.transform
        xx,yy=np.meshgrid(gt[0]+(np.arange(out.shape[1])+.5)*gt[1],gt[3]+(np.arange(out.shape[0])+.5)*gt[5])
        np.testing.assert_allclose(out[mapper.valid],(2*xx+3*yy)[mapper.valid],atol=1e-4)
        good[:,2]=False
        broken=ConservativeMapper(x,y,good,50)
        # No valid interpolation can cross missing center column x=200.
        self.assertFalse(broken.valid[:,(xx[0]>100)&(xx[0]<300)].any())
        np.testing.assert_array_equal(mapper.map(np.zeros(x.shape))[mapper.valid],0)
    def test_full_resolution_half_pixel_sampling_and_tiff_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'geometry.tif';x,y=np.meshgrid(np.arange(12),np.arange(8))
            data=(2*x+3*y).astype(float)
            write_geotiff(path,data,(0,2,0,8,0,-2),3031,{'units':'m'})
            line=np.array([[1.5,3.5]]);sample=np.array([[2.5,7.5]])
            np.testing.assert_allclose(sample_geometry(path,line,sample),2*sample+3*line)
            ds=gdal.Open(str(path));np.testing.assert_array_equal(ds.ReadAsArray(),data)
            self.assertIn('3031',ds.GetProjection());self.assertEqual(ds.GetGeoTransform(),(0,2,0,8,0,-2))
    def test_output_crs_and_spacing_validation(self):
        from glacier_offsets.geocode import validate_geocoding
        self.assertEqual(validate_geocoding({'epsg':3031,'spacing_m':200}),(3031,200.0))
        self.assertEqual(validate_geocoding({'epsg':3413,'spacing_m':100}),(3413,100.0))
        for settings in [{},{'epsg':4326,'spacing_m':200},{'epsg':3031,'spacing_m':float('nan')},{'epsg':3031,'spacing_m':0},
                         {'epsg':3031,'spacing_m':200,'resampling':'cubic'},
                         {'epsg':3031,'spacing_m':200,'mask':'all'},
                         {'epsg':3031,'spacing_m':200,'unknown':True}]:
            with self.assertRaises(ValueError):validate_geocoding(settings)

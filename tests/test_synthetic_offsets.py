import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy.ndimage import shift, gaussian_filter
from osgeo import gdal
from glacier_offsets.ampcor import native_match
from glacier_offsets.config import DEFAULT_MATCHING


def write_slc(path,array):
    ds=gdal.GetDriverByName('GTiff').Create(str(path),array.shape[1],array.shape[0],1,gdal.GDT_CFloat32)
    ds.GetRasterBand(1).WriteArray(array.astype('complex64'));ds=None


class SyntheticTests(unittest.TestCase):
    def test_glacier_defaults_rectangular_windows_and_gross_modes(self):
        rng=np.random.default_rng(23)
        amp=gaussian_filter(rng.uniform(0.1,1,(256,640)),0.6)
        ref=(amp*np.exp(1j*rng.normal(0,0.2,amp.shape))).astype('complex64')
        params=DEFAULT_MATCHING|{'first_center':[63.5,191.5]}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);write_slc(d/'ref.tif',ref)
            for translation,gross in [((-4,18),(-3,16)),((4,-18),(3,-16))]:
                write_slc(d/'sec.tif',shift(ref,translation,order=3,mode='constant',cval=0))
                results=[]
                for mode,prior in [('zero',(0,0)),('constant',gross)]:
                    cfg={'gross_offsets':{'mode':mode,'constant':{'azimuth_px':prior[0],'range_px':prior[1]}}}
                    r=native_match(d/'ref.tif',d/'sec.tif',d/mode,params,cfg,'cpu')
                    self.assertTrue(r['valid'].all())
                    np.testing.assert_allclose(r['total'],np.broadcast_to(translation,r['total'].shape),atol=0.15)
                    np.testing.assert_array_equal(r['total'],r['gross']+r['residual'])
                    self.assertEqual(r['effective_parameters']['corrSurfaceOverSamplingFactor'],32)
                    np.testing.assert_array_equal(np.diff(r['grid'].starts_line),32)
                    np.testing.assert_array_equal(np.diff(r['grid'].starts_sample),128)
                    results.append(r)
                np.testing.assert_allclose(results[0]['total'],results[1]['total'],atol=0.15)

    def measure(self,translation,gross=(0,0),search=(8,8),backend='cpu',mode='constant',deramp=0):
        rng=np.random.default_rng(4)
        amp=gaussian_filter(rng.uniform(0.1,1,(256,256)),0.6)
        ref=(amp*np.exp(1j*rng.normal(0,0.2,amp.shape))).astype('complex64')
        sec=shift(ref,translation,order=3,mode='constant',cval=0,prefilter=True)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);write_slc(d/'ref.tif',ref);write_slc(d/'sec.tif',sec)
            cfg={'gross_offsets':{'mode':mode,'constant':{'azimuth_px':gross[0],'range_px':gross[1]}}}
            p=DEFAULT_MATCHING|{'window':[48,48],'search':list(search),'stride':[64,64],'batch':[1,2],
                                'first_center':[71.5,71.5],'deramp':deramp,'zoom':8,'correlation_threshold':0.4,'snr_threshold':2}
            result=native_match(d/'ref.tif',d/'sec.tif',d/'out',p,cfg,backend)
        return result
    def test_integer_axis_sign_gross_and_no_double_counting(self):
        for translation in [(3,-4),(-3,4)]:
            for gross in [(0,0),(2,-3),(-2,3)]:
                with self.subTest(translation=translation,gross=gross):
                    r=self.measure(translation,gross,search=(12,12))
                    np.testing.assert_allclose(np.median(r['total'],axis=(0,1)),translation,atol=0.15)
                    np.testing.assert_allclose(r['total'],r['gross']+r['residual'],atol=1e-6)
                    self.assertTrue(np.all(r['gross']==gross))
    def test_large_translation_requires_gross(self):
        r=self.measure((18,-17),(17,-16),(4,4))
        np.testing.assert_allclose(np.median(r['total'],axis=(0,1)),(18,-17),atol=0.15)
        zero=self.measure((18,-17),(0,0),(4,4))
        self.assertGreater(np.max(np.abs(np.median(zero['total'],axis=(0,1))-[18,-17])),5)
    def test_fractional_and_exact_zero(self):
        for shift_px in [(0,0),(2.5,-1.25),(-1.5,2.25)]:
            r=self.measure(shift_px)
            np.testing.assert_allclose(np.median(r['total'],axis=(0,1)),shift_px,atol=0.25)

    def test_zero_mode_equals_constant_zero_and_deramp(self):
        a=self.measure((3,-4),mode='zero')
        b=self.measure((3,-4),mode='constant')
        np.testing.assert_array_equal(a['total'],b['total'])
        r=self.measure((3,-4),deramp=1)
        np.testing.assert_allclose(np.median(r['total'],axis=(0,1)),(3,-4),atol=0.25)
        self.assertTrue(r['input_valid'].all())
        self.assertTrue(r['valid'].any())
    def test_same_total_when_both_searches_contain_match_and_bad_prior(self):
        zero=self.measure((3,-4),mode='zero',search=(12,12))
        const=self.measure((3,-4),gross=(2,-3),search=(12,12))
        np.testing.assert_allclose(zero['total'],const['total'],atol=0.25)
        bad=self.measure((18,-17),gross=(-2,2),search=(4,4))
        self.assertFalse(np.allclose(np.median(bad['total'],axis=(0,1)),(18,-17),atol=0.25))
        no_motion=self.measure((0,0),mode='zero')
        np.testing.assert_array_equal(no_motion['total'],np.zeros_like(no_motion['total']))

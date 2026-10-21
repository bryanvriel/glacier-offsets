import unittest
import numpy as np
from glacier_offsets.matching_grid import build_grid

class GridTests(unittest.TestCase):
    def test_centers_bounds_stride(self):
        for gross in [(0,0),(9,-7),(-9,7)]:
            g=build_grid((256,300),(256,300),(32,40),(8,10),(25,30),gross)
            for axis,starts in enumerate([g.starts_line,g.starts_sample]):
                self.assertTrue(np.all(starts>=0));self.assertTrue(np.all(starts+g.window[axis]<= [256,300][axis]))
                self.assertTrue(np.all(starts+gross[axis]-g.search[axis]>=0))
                self.assertTrue(np.all(starts+gross[axis]+g.search[axis]+g.window[axis]<= [256,300][axis]))
            self.assertEqual(g.centers[0][0,0],g.starts_line[0]+15.5)
            self.assertEqual(g.centers[1][0,0],g.starts_sample[0]+19.5)
    def test_no_bounds_and_explicit_center(self):
        with self.assertRaises(ValueError):build_grid((32,32),(32,32),(32,32),(8,8),(8,8),(90,90))
        g=build_grid((128,128),(128,128),(32,32),(8,8),(16,16),first_center=[47.5,47.5])
        self.assertEqual(g.starts_line[0],32)
        with self.assertRaises(ValueError):build_grid((128,128),(128,128),(32,32),(8,8),(16,16),first_center=[47,47])
    def test_connected_zero_fill_and_isolated_quantized_zero(self):
        from test_synthetic_offsets import write_slc
        from osgeo import gdal
        from glacier_offsets.matching_grid import valid_windows
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=np.ones((128,128),dtype='complex64')
            a[20,20]=0
            write_slc(p/'a.tif',a)
            g=build_grid(a.shape,a.shape,(32,32),(8,8),(32,32))
            ds=gdal.Open(str(p/'a.tif'));self.assertTrue(valid_windows(ds,ds,g,(0,0))[0,0]);ds=None
            a[20:24,20:24]=0;write_slc(p/'a.tif',a)
            ds=gdal.Open(str(p/'a.tif'));self.assertFalse(valid_windows(ds,ds,g,(0,0))[0,0])

    def test_compass_complex_nodata_zero_does_not_mask_zero_real_component(self):
        from test_synthetic_offsets import write_slc
        from osgeo import gdal
        from glacier_offsets.matching_grid import valid_windows
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'slc.tif';a=np.full((128,128),1j,dtype='complex64')
            write_slc(p,a)
            ds=gdal.Open(str(p),gdal.GA_Update);ds.GetRasterBand(1).SetNoDataValue(0);ds=None
            ds=gdal.Open(str(p));g=build_grid(a.shape,a.shape,(32,32),(8,8),(32,32))
            self.assertTrue(valid_windows(ds,ds,g,(0,0)).all())

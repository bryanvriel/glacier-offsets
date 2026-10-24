"""Physical-coordinate and native checks for the isolated range control."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from osgeo import gdal

from glacier_offsets.ampcor import _native_executor
from glacier_offsets.matching_grid import MatchingGrid
from tools.compare_swath_range_oversampling import prepare_strip, prepared_grid, resample_range, run_range_job
from test_swath_overlap import make_pair, write_raster


class RangeCoordinatesTests(unittest.TestCase):
    def test_complex_fourier_values_and_no_azimuth_mixing(self):
        n=128;x=np.arange(n);x2=(np.arange(2*n)+.5)/2-.5
        def signal(x):return np.exp(2j*np.pi*17*x/n)+.4j*np.exp(-2j*np.pi*25*x/n)
        scales=np.array([1,2j,-3])[:,None]
        actual=resample_range(scales*signal(x)[None,:])
        np.testing.assert_allclose(actual,scales*signal(x2)[None,:],atol=1.e-6)
        # Even-length Nyquist input is split symmetrically, as in scipy resample.
        nyquist=resample_range(((-1.)**x)[None,:].astype('complex64'))
        np.testing.assert_allclose(nyquist[0],np.cos(np.pi*x2),atol=1.e-6)

    def test_exact_chip_centers_pixel_cell_boundaries_and_search_distances(self):
        grid=MatchingGrid(np.array([100,132]),np.array([200,328]),(64,256),(10,40),(32,128))
        y0,x0=90,160;prepared=prepared_grid(grid,(y0,x0))
        line,sample=prepared.centers
        np.testing.assert_array_equal(line+y0,grid.centers[0])
        np.testing.assert_array_equal(sample/2+x0-.25,grid.centers[1])
        left=(prepared.starts_sample-.5)/2+x0-.25
        right=(prepared.starts_sample+prepared.window[1]-.5)/2+x0-.25
        np.testing.assert_array_equal(left,grid.starts_sample-.5)
        np.testing.assert_array_equal(right,grid.starts_sample+grid.window[1]-.5)
        self.assertEqual(prepared.search[1]/2,grid.search[1])
        self.assertEqual(prepared.stride[1]/2,grid.stride[1])

    def test_nonfinite_or_real_input_cannot_be_silently_filled(self):
        for data in (np.ones((2,10)),np.full((2,10),complex(np.nan,1))):
            with self.assertRaisesRegex(ValueError,'finite complex'):resample_range(data)


class NativeRangeControlTests(unittest.TestCase):
    def test_native_original_units_and_source_preservation(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();pair,grid=make_pair(root);stack=pair.parent.parent
            paths=[stack/'acquisitions'/day/'t038_080975_iw1.tif' for day in ('2020-01-01','2020-01-07')]
            params=json.loads((pair/'manifest.json').read_text())['matching']['matching']
            with _native_executor(1,single_threaded=True) as pool:
                for shift in (1.,.3):
                    # Range-only known translation of a complex periodic texture.
                    ref=gdal.Open(str(paths[0])).ReadAsArray()
                    phase=np.exp(-2j*np.pi*np.fft.fftfreq(ref.shape[1])*shift)
                    secondary=np.fft.ifft(np.fft.fft(ref,axis=-1)*phase,axis=-1)
                    write_raster(paths[1],secondary.astype('complex64'),gdal.GDT_CFloat32)
                    original=[p.read_bytes() for p in paths]
                    out=root/f'control-{shift}';out.mkdir()
                    job=('t038_080975_iw1',paths,out,params,grid,np.ones(grid.shape,bool))
                    key,result,error=pool.apply(run_range_job,(job,))
                    self.assertIsNone(error);self.assertEqual(key,'t038_080975_iw1')
                    self.assertTrue(result['valid'].all())
                    np.testing.assert_allclose(result['total'],np.broadcast_to([0,shift],(*grid.shape,2)),atol=.05)
                    np.testing.assert_array_equal(result['grid'].centers,grid.centers)
                    np.testing.assert_array_equal(result['total'],result['gross']+result['residual'])
                    for path,data in zip(paths,original):self.assertEqual(path.read_bytes(),data)
                    self.assertEqual(result['effective_parameters']['derampMethod'],0)
                    self.assertEqual(result['effective_parameters']['windowSizeWidth'],params['window'][1]*2)


if __name__=='__main__':unittest.main()

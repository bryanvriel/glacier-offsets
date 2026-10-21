"""Scientific controls for the experimental common-grid boundary comparison."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('boundary_validation', Path(__file__).parents[1] / 'tools/validate_burst_boundary.py')
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


class BoundaryTests(unittest.TestCase):
    def test_epoch_aware_placement_and_fractional_sampling_rejection(self):
        base = dict(prf=500., range_pixel_spacing=2., wavelength=.05, lookside='right',
                    length=1500, width=2000, sensing_start=172800., starting_range=800000.,
                    ref_epoch='2020-01-08T00:00:00')
        other = base | {'ref_epoch': '2020-01-08T00:00:02.600001'}
        offset, residual = boundary.placement(base, other)
        self.assertEqual(offset, 1300)
        self.assertAlmostEqual(residual[0], .0005, places=6)
        for change in ({'ref_epoch': '2020-01-08T00:00:02.601'}, {'prf': 501.}, {'starting_range': 800000.5}):
            with self.assertRaises(ValueError):
                boundary.placement(base, other | change)

    def test_windowed_placement_and_common_ownership_preserve_complex_pixels(self):
        data = (np.arange(90).reshape(9, 10) + 1j).astype('complex64')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'slc.tif'
            boundary.write_slc(path, data)
            actual = boundary.read_crop(path, (3, 2), (10, 5), burst_line=6)
            expected = np.zeros((10, 5), 'complex64')
            expected[3:] = data[:7, 2:7]
            np.testing.assert_array_equal(actual, expected)
            merged = boundary.splice(actual, 2*actual, 6)
            np.testing.assert_array_equal(merged[:6], actual[:6])
            np.testing.assert_array_equal(merged[6:], 2*actual[6:])

    def test_support_proxy_allows_quantized_zeros_but_rejects_internal_gaps(self):
        data = np.ones((100, 200), 'complex64')
        data[:5] = 0
        data[-7:] = 0
        data[50, 20] = 0
        self.assertEqual(boundary.support_rows(data, data), (5, 93))
        data[40] = 0
        with self.assertRaises(ValueError):
            boundary.support_rows(data, data)


if __name__ == '__main__':
    unittest.main()

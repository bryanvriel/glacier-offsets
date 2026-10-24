"""Range-2x complex interpolation before native matching: units, settings and bias."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from glacier_offsets import range_oversampling as r2
from glacier_offsets.ampcor import _native_executor, _native_worker, matching_specification
from glacier_offsets.config import DEFAULT_MATCHING, load_defaults, validate_common
from glacier_offsets.matching_grid import build_grid
from glacier_offsets.products import identity
from test_synthetic_offsets import write_slc
from tools.compare_swath_range_oversampling import normalize_range_result, resample_range

PARAMS = DEFAULT_MATCHING | {'window': [32, 64], 'search': [6, 8], 'stride': [32, 64], 'zoom': 8,
                             'deramp': 0, 'backend': 'cpu', 'batch': [1, 8]}


def speckle(shape, shift, bandwidth=.88, seed=4):
    """Band-limited complex speckle and a copy shifted exactly in range."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft(rng.standard_normal(shape) + 1j*rng.standard_normal(shape), axis=1)
    frequency = np.fft.fftfreq(shape[1])
    spectrum *= np.abs(frequency) < bandwidth/2
    return [np.fft.ifft(spectrum*np.exp(-2j*np.pi*frequency*s), axis=1).astype('complex64') for s in (0, shift)]


class RangeOversamplingUnitTests(unittest.TestCase):
    def test_interpolation_matches_validated_prototype_and_preserves_centers(self):
        rows = (np.random.default_rng(1).standard_normal((5, 301)) + 1j).astype('complex64')
        np.testing.assert_array_equal(r2.resample_range(rows), resample_range(rows))
        with self.assertRaisesRegex(ValueError, 'finite'):
            r2.resample_range(np.where(rows.real > 1, np.nan, rows))
        grid = build_grid((400, 900), (400, 900), (32, 64), (6, 8), (32, 64))
        native = r2.native_grid(grid)
        self.assertEqual((native.window, native.search, native.stride), ((32, 128), (6, 16), (32, 128)))
        np.testing.assert_array_equal(native.centers[0], grid.centers[0])
        np.testing.assert_array_equal(native.centers[1]/2 - .25, grid.centers[1])

    def test_settings_and_gross_offsets_scale_only_range(self):
        native = r2.native_parameters(PARAMS | {'range_oversampling': 2, 'first_center': [10, 20]})
        self.assertEqual((native['window'], native['search'], native['stride']), ([32, 128], [6, 16], [32, 128]))
        self.assertEqual((native['deramp'], native['range_oversampling']), (0, 1))
        self.assertNotIn('first_center', native)
        for key in ('raw_oversampling', 'oversampling', 'zoom', 'batch', 'snr_threshold'):
            self.assertEqual(native[key], PARAMS[key])
        cfg = {'gross_offsets': {'mode': 'constant', 'constant': {'azimuth_px': 3, 'range_px': -2}}}
        scaled, gross = r2.native_gross(cfg, None)
        self.assertIsNone(gross)
        self.assertEqual(scaled['gross_offsets']['constant'], {'azimuth_px': 3, 'range_px': -4})
        self.assertEqual(cfg['gross_offsets']['constant']['range_px'], -2)
        array = np.array([[[1, 2], [-3, -5]]], 'int32')
        _, gross = r2.native_gross({'gross_offsets': {'mode': 'velocity'}}, array)
        np.testing.assert_array_equal(gross, [[[1, 4], [-3, -10]]])
        self.assertEqual(gross.dtype, np.int32)

    def test_normalization_matches_validated_prototype(self):
        rng = np.random.default_rng(2)
        grid = build_grid((400, 900), (400, 900), (32, 64), (6, 8), (32, 64))
        shape = grid.shape
        gross = np.broadcast_to(np.array([1, 4], 'float32'), (*shape, 2)).copy()
        residual = rng.uniform(-6, 16, (*shape, 2)).astype('float32')
        covariance = rng.uniform(0, 1, (*shape, 3)).astype('float32')
        covariance[0, 0] = [99, 99, 0]
        native = {'gross': gross, 'residual': residual, 'total': gross+residual, 'covariance': covariance,
                  'snr': rng.uniform(0, 20, shape).astype('float32'),
                  'correlation_peak': rng.uniform(0, 1, shape).astype('float32'),
                  'input_valid': rng.random(shape) > .2, 'finite': np.ones(shape, bool),
                  'valid': np.zeros(shape, bool), 'search_boundary': np.zeros(shape, bool)}
        params = PARAMS | {'snr_threshold': 5, 'correlation_threshold': .2}
        ours = r2.normalize(deepcopy(native), grid, params)
        expected = normalize_range_result(deepcopy(native), grid, params)
        for name in ('gross', 'residual', 'total', 'covariance', 'search_boundary', 'valid'):
            np.testing.assert_array_equal(ours[name], expected[name], err_msg=name)
        np.testing.assert_array_equal(ours['gross'][..., 1], 2)
        np.testing.assert_array_equal(ours['covariance'][0, 0], [99, 99, 0])

    def test_option_is_validated_and_absent_from_existing_identities(self):
        dem = Path(self.enterContext(tempfile.TemporaryDirectory()))/'dem.tif'
        dem.touch()
        def configured(**matching):
            cfg = load_defaults()
            cfg['inputs'].update(burst_id='t001_000001_iw1', dem=str(dem))
            cfg['output'] = {'directory': str(dem.parent)}
            cfg['geocoding']['epsg'] = 3031
            cfg['matching'].update({'backend': 'cpu', 'deramp': 0} | matching)
            return cfg
        validate_common(configured(range_oversampling=2))
        scratch = configured(range_oversampling=2)
        scratch['storage']['scratch_directory'] = str(dem.parent)
        validate_common(scratch)
        scratch['storage']['scratch_directory'] = 'relative'
        with self.assertRaisesRegex(ValueError, 'scratch_directory'):
            validate_common(scratch)
        for changes, message in [({'range_oversampling': 3}, 'must be 1 or 2'),
                                 ({'range_oversampling': 2, 'deramp': 1}, 'requires engine isce3'),
                                 ({'range_oversampling': 2, 'backend': 'auto'}, 'requires engine isce3')]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                validate_common(configured(**changes))
        manifest = {'coreg_identity': 'fixed', 'stack_pair': {}}
        cfg = configured()
        legacy = deepcopy(cfg)
        legacy['matching'].pop('range_oversampling')
        self.assertEqual(identity(matching_specification(cfg, manifest)),
                         identity(matching_specification(legacy, manifest)))
        spec = matching_specification(configured(range_oversampling=2), manifest)
        self.assertEqual(spec['matching']['range_oversampling'], 2)
        self.assertEqual(spec['range_preparation'], r2.METHOD)


class RangeOversamplingNativeTests(unittest.TestCase):
    def test_range2x_removes_magnitude_first_range_bias(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        shape, shift = (192, 1536), .3
        reference, secondary = speckle(shape, shift)
        write_slc(root/'ref.tif', reference)
        write_slc(root/'sec.tif', secondary)
        with r2.prepared_inputs({'reference': root/'ref.tif', 'secondary': root/'sec.tif'}, root/'range2x') as (paths, record):
            self.assertEqual(record['method'], r2.METHOD)
            self.assertEqual(r2.prepare(root/'ref.tif', root/'check.tif')['original_shape_yx'], list(shape))
            grid = build_grid(shape, shape, PARAMS['window'], PARAMS['search'], PARAMS['stride'])
            kwargs = {'_grid': grid, '_valid_input': np.ones(grid.shape, bool)}
            cfg = {'gross_offsets': {'mode': 'zero'}}
            with _native_executor(1, single_threaded=True) as pool:
                native = pool.apply(_native_worker, (((str(root/'ref.tif'), str(root/'sec.tif'), root/'native',
                                                       PARAMS, cfg, 'cpu'), kwargs),))
                range2x = pool.apply(r2.range_worker, (((str(paths['reference']), str(paths['secondary']), root/'r2',
                                                         PARAMS | {'range_oversampling': 2}, cfg, 'cpu'), kwargs),))
        self.assertFalse((root/'range2x').exists())
        np.testing.assert_array_equal(range2x['grid'].starts_sample, grid.starts_sample)
        self.assertEqual((range2x['grid'].window, range2x['grid'].search), (grid.window, grid.search))
        self.assertGreater(int(range2x['valid'].sum()), 100)
        error = range2x['total'][..., 1][range2x['valid']] - shift
        bias = native['total'][..., 1][native['valid']] - shift
        self.assertLess(abs(error.mean()), .02)
        self.assertGreater(abs(bias.mean()), .1)  # Plain magnitude matching is pulled toward whole pixels.
        self.assertLess(abs(range2x['total'][..., 0][range2x['valid']]).max(), .02)
        self.assertEqual(range2x['effective_parameters']['range_oversampling']['method'], r2.METHOD)


if __name__ == '__main__':
    unittest.main()

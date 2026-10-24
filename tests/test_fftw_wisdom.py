"""Pinned FFTW wisdom for stack CPU matching: keys, reuse and worker import."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from glacier_offsets import fftw_cache
from glacier_offsets.ampcor import _native_executor, _native_worker
from glacier_offsets.config import DEFAULT_MATCHING
from glacier_offsets.matching_grid import build_grid
from test_synthetic_offsets import write_slc

PARAMS = DEFAULT_MATCHING | {'window': [32, 32], 'search': [8, 8], 'stride': [32, 32], 'zoom': 8, 'batch': [1, 3]}


def requires_fftw(test):
    return unittest.skipIf(fftw_cache.library() is None, 'No shared FFTW library in this native environment')(test)


class PinnedWisdomTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    @requires_fftw
    def test_key_tracks_host_build_and_plan_shaping_settings_only(self):
        key = fftw_cache.wisdom_key(PARAMS)
        self.assertTrue(key['fftw_version'].startswith('fftw-'))
        self.assertTrue(key['cpu'] and key['machine'] and key['isce3_version'])
        self.assertEqual(fftw_cache.wisdom_key(PARAMS | {'snr_threshold': 99, 'stride': [16, 16]}), key)
        for name, value in [('window', [32, 64]), ('search', [8, 10]), ('batch', [1, 4]),
                            ('raw_oversampling', 1), ('oversampling', 16), ('zoom', 16), ('deramp', 0)]:
            with self.subTest(name=name):
                self.assertNotEqual(fftw_cache.wisdom_key(PARAMS | {name: value}), key)

    @requires_fftw
    def test_measured_once_reused_unchanged_and_off_outside_stack_matching(self):
        self.assertIsNone(fftw_cache.pinned_wisdom(PARAMS))
        with patch.object(fftw_cache, 'measure', return_value='(fftw-3.3 fftwf_wisdom)\n') as measure, \
                fftw_cache.pinned(self.root):
            first = fftw_cache.pinned_wisdom(PARAMS)
            again = fftw_cache.pinned_wisdom(PARAMS)
            other = fftw_cache.pinned_wisdom(PARAMS | {'zoom': 16})
        self.assertEqual(measure.call_count, 2)
        self.assertTrue(first['measured_now'])
        self.assertFalse(again['measured_now'])
        self.assertEqual({k: v for k, v in again.items() if k != 'measured_now'},
                         {k: v for k, v in first.items() if k != 'measured_now'})
        self.assertNotEqual(other['path'], first['path'])
        self.assertEqual(Path(first['path']).parent, self.root)
        self.assertTrue(Path(first['path']).with_suffix('.json').is_file())
        self.assertFalse(list(self.root.glob('.*.tmp')))
        self.assertIsNone(fftw_cache.pinned_wisdom(PARAMS))

    @requires_fftw
    def test_workers_import_pinned_plans_without_measuring_or_republishing(self):
        rng = np.random.default_rng(3)
        image = (rng.standard_normal((160, 224)) + 1j*rng.standard_normal((160, 224))).astype('complex64')
        write_slc(self.root/'ref.tif', image)
        write_slc(self.root/'sec.tif', np.roll(image, (1, 2), (0, 1)))
        with fftw_cache.pinned(self.root/'cache'):
            wisdom = fftw_cache.pinned_wisdom(PARAMS)
        self.assertTrue(wisdom['measured_now'])
        stamp = Path(wisdom['path']).stat().st_mtime_ns
        grid = build_grid(image.shape, image.shape, PARAMS['window'], PARAMS['search'], PARAMS['stride'])
        jobs = []
        # Full chunks, a narrower remainder and a narrow segment, as stack lanes produce.
        for name, columns in (('full', slice(None)), ('narrow', slice(0, 2)), ('single', slice(3, 4))):
            local = type(grid)(grid.starts_line, grid.starts_sample[columns], grid.window, grid.search, grid.stride)
            jobs.append(((str(self.root/'ref.tif'), str(self.root/'sec.tif'), self.root/name, PARAMS,
                          {'gross_offsets': {'mode': 'zero'}}, 'cpu'),
                         {'_grid': local, '_valid_input': np.ones(local.shape, bool)}))
        with _native_executor(2, single_threaded=True, wisdom=wisdom) as pool:
            results = pool.map(_native_worker, jobs)
        for result in results:
            parameters = result['effective_parameters']
            self.assertEqual(parameters['fftw_wisdom_sha256'], wisdom['sha256'])
            self.assertFalse(parameters['fftw_wisdom_extended'])
        self.assertEqual(Path(wisdom['path']).stat().st_mtime_ns, stamp)
        self.assertEqual(fftw_cache._sha256(wisdom['path']), wisdom['sha256'])

    def test_missing_pinned_file_fails_instead_of_measuring(self):
        missing = {'path': str(self.root/'absent.wisdom'), 'sha256': '0'*64}
        with ExitStack() as stack:
            pool = stack.enter_context(_native_executor(1, single_threaded=True, wisdom=missing))
            with self.assertRaisesRegex(RuntimeError, 'Pinned FFTW wisdom is missing'):
                pool.apply(_native_worker, (((), {}),))


if __name__ == '__main__':
    unittest.main()

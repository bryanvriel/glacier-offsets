"""Native CPU burst scheduling preserves the assembled stack's science."""
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets import stack_matching
from glacier_offsets.ampcor import (NATIVE_CALLS_PER_WORKER, _native_executor, match,
                                    _empty_match, matching_specification, native_match, varying_match)
from glacier_offsets.assembly import annotate_result
from glacier_offsets.matching_grid import valid_windows
from glacier_offsets.matching_grid import MatchingGrid
from glacier_offsets.products import checksum
from glacier_offsets.stack_pairs import common_map_grid, match_prepared, matching_template, prepared_pair
import test_stack_science


class SchedulingTests(unittest.TestCase):
    def test_budget_uses_first_omp_level_and_defaults_to_one(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(stack_matching.cpu_worker_budget(), 1)
        for value, expected in [('8', 8), (' 4, 2 ', 4)]:
            with self.subTest(value=value), patch.dict(os.environ, OMP_NUM_THREADS=value):
                self.assertEqual(stack_matching.cpu_worker_budget(), expected)
        for value in ('0', '-1', '', '2,', 'garbage', '2.0', '4,0'):
            with self.subTest(value=value), patch.dict(os.environ, OMP_NUM_THREADS=value):
                with self.assertRaisesRegex(ValueError, 'OMP_NUM_THREADS'):
                    stack_matching.cpu_worker_budget()

    def test_ragged_partitions_cover_each_selected_window_exactly_once(self):
        owner = np.array([[1, 1, 2, 2, 2], [1, 1, 2, 2, 2], [1, 2, 2, 2, 2],
                          [1, 1, 2, 2, 2], [1, 1, 2, 2, 2]])
        selected = np.ones(owner.shape, bool)
        selected[2, 2] = False
        counts = np.zeros(owner.shape, int)
        rectangles = stack_matching.burst_rectangles(owner, selected)
        self.assertIn((1, 0, 2, 0, 2), rectangles)
        for index, y0, y1, x0, x1 in rectangles:
            counts[y0:y1, x0:x1] += 1
            self.assertTrue((owner[y0:y1, x0:x1] == index).all())
        np.testing.assert_array_equal(counts, selected)
        self.assertEqual(stack_matching.burst_rectangles(owner, np.zeros(owner.shape, bool)), [])

    def test_shared_rectangles_are_bounded_without_reassigning_burst_ownership(self):
        owner = np.zeros((21, 29), 'uint16')
        owner[2:8, 3:15] = 1
        owner[8:14, 3:15] = 2
        owner[14:20, 3:15] = 3  # Burst absent from one matched date.
        for workers in (1, 2, 40):
            selected = np.ones(owner.shape, bool)
            selected[3:5, 5:7] = False
            tasks = stack_matching.matching_rectangles(owner, selected, [1, 2], workers)
            count = np.zeros(owner.shape, int)
            target = max(1, (int(selected.sum())+2*workers-1)//(2*workers))
            for lane, a, b, c, d in tasks:
                count[a:b,c:d] += 1
                if lane is None:
                    self.assertFalse(np.isin(owner[a:b,c:d], [1, 2]).any())
                    self.assertLessEqual((b-a)*(d-c), target)
                else:
                    self.assertTrue((owner[a:b,c:d] == lane).all())
            np.testing.assert_array_equal(count, selected)
        tasks = stack_matching.matching_rectangles(owner, np.isin(owner, [1, 2]), [1, 2], 2)
        self.assertTrue(all(t[0] is not None for t in tasks))
        self.assertEqual(stack_matching.matching_rectangles(owner, selected, [], 0), [])

    def test_idle_workers_share_padding_and_avoid_the_serial_lane_tail(self):
        # A deterministic event-based pool models cost as window count. This
        # verifies actual submission/completion logic without timing assertions
        # depending on machine load, FFT plans or process startup overhead.
        owner = np.zeros((100, 80), int)
        for index in (1, 2, 3):
            owner[(index-1)*10:index*10] = index
        grid = MatchingGrid(np.arange(100), np.arange(80), (1,1), (0,0), (1,1))
        prior = {'grid':grid, 'valid':owner>0, 'integer_px':np.zeros((100,80,2), 'int32')}
        manifest = {'coreg':{'slcs':{'reference':'ref','secondary':'sec'}},
                    'assembly':{'sources':[{'burst_id':f'b{i}', 'interval_seconds':1} for i in (1,2,3)]}}

        class Pending:
            def __init__(self, pool, args, kwargs):
                self.pool, self.kwargs = pool, kwargs
                self.start = pool.now
                self.duration = int(np.prod(kwargs['_grid'].shape))
                self.end = self.start+self.duration
                self.delivered = False
                self.name = Path(args[2]).name
            def ready(self):
                return self.pool.now >= self.end
            def get(self, timeout=None):
                if not self.ready():
                    self.pool.now = min(t.end for t in self.pool.tasks if not t.delivered)
                    if not self.ready():
                        raise multiprocessing.TimeoutError
                self.delivered = True
                result = _empty_match(self.kwargs['_grid'], self.kwargs['_valid_input'], 'cpu')
                result.update(elapsed_seconds=self.duration, effective_parameters={})
                return result
        class Pool:
            now = 0
            def __init__(self):
                self.tasks = []
            def apply_async(self, function, arguments):
                args, kwargs = arguments[0]
                task = Pending(self, args, kwargs)
                running = [t for t in self.tasks if t.end > self.now]
                self.assertions.assertLess(len(running), 3)
                if task.name.startswith('burst-'):
                    lane = task.name.split('-part-')[0]
                    self.assertions.assertFalse(any(t.name.startswith(lane+'-part-') for t in running))
                self.tasks.append(task)
                return task
        pool = Pool(); pool.assertions = self
        @contextmanager
        def executor(workers, **kwargs):
            self.assertEqual(workers, 3)
            yield pool
        with patch.dict(os.environ, OMP_NUM_THREADS='3'), \
                patch.object(stack_matching, 'source_indices', return_value=owner), \
                patch.object(stack_matching, '_native_executor', executor), \
                patch.object(stack_matching, '_write_native_layout'):
            result = stack_matching.match_bursts({'gross_offsets':{'mode':'zero'},'matching':{}},
                                                 manifest, self.enterContext(tempfile.TemporaryDirectory()),
                                                 prior, full_grid=True)
        shared = [t for t in pool.tasks if t.name.startswith('shared-')]
        self.assertGreater(len(shared), 3)
        self.assertTrue(any(a.start <= b.start < a.end for a in shared for b in shared if a is not b))
        old_tail = np.count_nonzero((owner == 0) | (owner == 1))
        self.assertLess(pool.now, .65*old_tail)
        self.assertEqual(sum(t.duration for t in pool.tasks), owner.size)
        self.assertEqual(result['effective_parameters']['workers'], 3)


class NativeBurstMatchingTests(unittest.TestCase):
    def setUp(self):
        # Reuse the independent two-burst/three-date native science fixture
        # without inheriting and rerunning its test methods.
        self.f = test_stack_science.StackScienceTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.cfg = self.f.effective()
        self.cfg['matching']['batch'] = [2, 4]
        self.frame = self.f.geometry()
        self.dates = self.f.dates(self.frame)
        self.a, self.b = self.f.acquisitions[0], self.f.acquisitions[-1]
        self.directory, self.manifest = self.pair()

    def pair(self):
        return prepared_pair(self.cfg, self.f.plan, self.frame, self.a, self.b,
                             self.dates, common_map_grid(self.cfg, self.frame))

    def run_workers(self, workers, name, prior=None):
        with patch.dict(os.environ, OMP_NUM_THREADS=str(workers), OPENBLAS_NUM_THREADS='7'):
            result = match_prepared(self.cfg, self.manifest, self.directory/name, deepcopy(prior))
            self.assertEqual(os.environ['OMP_NUM_THREADS'], str(workers))
            self.assertEqual(os.environ['OPENBLAS_NUM_THREADS'], '7')
        annotate_result(self.manifest, result)
        return result

    def assert_science_equal(self, expected, actual):
        np.testing.assert_array_equal(expected['grid'].starts_line, actual['grid'].starts_line)
        np.testing.assert_array_equal(expected['grid'].starts_sample, actual['grid'].starts_sample)
        for field in ('input_valid', 'finite', 'search_boundary', 'valid', 'source_burst_index',
                      'seam_crossing', 'seam_search_known'):
            np.testing.assert_array_equal(expected[field], actual[field], err_msg=field)
        for field in ('gross', 'residual', 'total', 'snr', 'correlation_peak', 'covariance', 'interval_seconds'):
            np.testing.assert_allclose(expected[field], actual[field], rtol=1e-5, atol=1e-5,
                                       equal_nan=True, err_msg=field)

    def test_native_zero_and_constant_match_legacy_grid_with_one_and_two_workers(self):
        # Leave full-grid margins for both signs of constant gross offsets.
        self.cfg['matching']['first_center'] = [31.5, 55.5]
        grid = matching_template(self.cfg, self.frame)
        slcs = self.manifest['coreg']['slcs']
        for mode, gross in [('zero', (0, 0)), ('constant', (2, -2))]:
            self.cfg['gross_offsets'] = {'mode': mode, 'constant':dict(zip(('azimuth_px', 'range_px'), gross))}
            valid = valid_windows(gdal.Open(slcs['reference']), gdal.Open(slcs['secondary']), grid, gross)
            legacy = native_match(slcs['reference'], slcs['secondary'], self.directory/f'legacy-{mode}',
                self.cfg['matching'], self.cfg, 'cpu', _grid=grid, _valid_input=valid)
            annotate_result(self.manifest, legacy)
            serial = self.run_workers(1, f'serial-{mode}')
            parallel = self.run_workers(2, f'parallel-{mode}')
            self.assert_science_equal(legacy, serial)
            self.assert_science_equal(serial, parallel)
            effective = parallel['effective_parameters']
            self.assertEqual(effective['workers'], 2)
            self.assertEqual(effective['threads_per_worker'], 1)
            segments = effective['segments']
            self.assertEqual({s['worker_omp_threads'] for s in segments}, {'1'})
            self.assertEqual({s['worker_blas_threads'] for s in segments}, {'1'})
            self.assertEqual(len({s['worker_pid'] for s in segments}), 2)
            self.assertTrue(parallel['seam_crossing'].any())
            measured = parallel['total'][parallel['valid']]
            np.testing.assert_allclose(measured, np.broadcast_to([3, -3], measured.shape), atol=.18)
        capped = self.run_workers(12, 'capped')
        self.assertEqual(capped['effective_parameters']['workers'], 2)

    def test_constant_gross_near_image_edges_retains_legacy_segment_masks(self):
        self.cfg['gross_offsets'] = {'mode':'constant', 'constant':{'azimuth_px':2, 'range_px':-2}}
        grid = matching_template(self.cfg, self.frame)
        slcs = self.manifest['coreg']['slcs']
        ref, sec = (gdal.Open(slcs[key]) for key in ('reference', 'secondary'))
        gross = np.broadcast_to(np.array([2, -2], 'int32'), (*grid.shape, 2)).copy()
        prior = {'grid':grid, 'integer_px':gross, 'valid':valid_windows(ref, sec, grid, gross),
                 'rejection_flags':np.zeros(grid.shape, 'uint8')}
        legacy = varying_match(slcs['reference'], slcs['secondary'], self.directory/'legacy-edges',
                               self.cfg['matching'], self.cfg, 'cpu', prior, ref, sec)
        legacy.pop('prior')
        annotate_result(self.manifest, legacy)
        parallel = self.run_workers(2, 'parallel-edges')
        self.assert_science_equal(legacy, parallel)
        self.assertNotIn('prior', parallel)

    def test_missing_regions_share_workers_and_preserve_all_raw_and_masked_results(self):
        self.cfg['matching']['first_center'] = [31.5, 55.5]
        slcs = self.manifest['coreg']['slcs']
        original = gdal.Open(slcs['secondary']).ReadAsArray()
        original[:, 200:] = 0
        from test_synthetic_offsets import write_slc
        missing = self.f.root/'partial-secondary.tif'
        write_slc(missing, original)
        slcs['secondary'] = str(missing)
        source = gdal.Open(self.manifest['coreg']['source_index']).ReadAsArray()
        source[:, 200:] = 3
        index_path = self.f.root/'partial-source.tif'
        ds = gdal.GetDriverByName('GTiff').Create(str(index_path), source.shape[1], source.shape[0], 1, gdal.GDT_UInt16)
        ds.GetRasterBand(1).WriteArray(source); ds = None
        self.manifest['coreg']['source_index'] = str(index_path)
        self.manifest['assembly']['sources'].append(self.manifest['assembly']['sources'][0] |
                                                   {'burst_id':'missing', 'interval_seconds':None})
        grid = matching_template(self.cfg, self.frame)
        valid = valid_windows(gdal.Open(slcs['reference']), gdal.Open(slcs['secondary']), grid, (0,0))
        legacy = native_match(slcs['reference'], slcs['secondary'], self.directory/'legacy-partial',
                              self.cfg['matching'], self.cfg, 'cpu', _grid=grid, _valid_input=valid)
        annotate_result(self.manifest, legacy)
        serial = self.run_workers(1, 'serial-partial')
        parallel = self.run_workers(2, 'parallel-partial')
        self.assert_science_equal(legacy, serial)
        self.assert_science_equal(serial, parallel)
        self.assertEqual(parallel['effective_parameters']['workers'], 2)
        shared = [s for s in parallel['effective_parameters']['segments']
                  if s['work_kind'] == 'unowned_or_inactive']
        self.assertTrue(shared)
        self.assertTrue(all(s['scheduling_burst_id'] is None for s in shared))
        self.assertFalse(parallel['input_valid'][parallel['source_burst_index'] == 3].any())

    def test_fragmented_varying_priors_equal_legacy_and_recycle_workers(self):
        self.cfg['gross_offsets']['mode'] = 'velocity'
        grid = matching_template(self.cfg, self.frame)
        gross = np.broadcast_to(np.array([3, -3], 'int32'), (*grid.shape, 2)).copy()
        gross[:, grid.shape[1]//2:] += [1, -1]
        eligible = np.ones(grid.shape, bool)
        eligible[1::2, 1::2] = False
        eligible[0] = False
        gross[~eligible] = np.iinfo(np.int32).min
        prior = {'grid':grid, 'integer_px':gross, 'valid':eligible,
                 'rejection_flags':np.where(eligible, 0, 2).astype('uint8'), 'metadata':{}}
        slcs = self.manifest['coreg']['slcs']
        ref, sec = (gdal.Open(slcs[key]) for key in ('reference', 'secondary'))
        previous = deepcopy(prior)
        previous['valid'] &= valid_windows(ref, sec, grid, gross)
        legacy = varying_match(slcs['reference'], slcs['secondary'], self.directory/'legacy-varying',
                               self.cfg['matching'], self.cfg, 'cpu', previous, ref, sec)
        annotate_result(self.manifest, legacy)
        serial = self.run_workers(1, 'serial-varying', prior)
        parallel = self.run_workers(2, 'parallel-varying', prior)
        self.assert_science_equal(legacy, serial)
        self.assert_science_equal(serial, parallel)
        np.testing.assert_array_equal(serial['prior']['valid'], parallel['prior']['valid'])
        np.testing.assert_array_equal(serial['prior']['rejection_flags'], parallel['prior']['rejection_flags'])
        counts = Counter(s['worker_pid'] for s in parallel['effective_parameters']['segments'])
        self.assertGreater(len(counts), 2)
        self.assertLessEqual(max(counts.values()), NATIVE_CALLS_PER_WORKER)
        self.assertTrue(np.isnan(parallel['total'][~parallel['prior']['valid']]).all())

    def test_partial_date_caps_workers_and_empty_coverage_starts_no_pool(self):
        self.b.bursts.pop(list(self.b.bursts)[-1])
        self.dates = self.f.dates(self.frame)
        self.directory, self.manifest = self.pair()
        result = self.run_workers(8, 'partial')
        self.assertEqual(result['effective_parameters']['workers'], 1)
        self.assertFalse(result['valid'][result['source_burst_index'] == 2].any())
        self.assertTrue(np.isnan(result['interval_seconds'][result['source_burst_index'] == 2]).all())
        self.b.bursts.clear()
        self.dates = self.f.dates(self.frame)
        self.directory, self.manifest = self.pair()
        with patch.object(stack_matching, '_native_executor', side_effect=AssertionError('empty work spawned')):
            empty = self.run_workers(8, 'empty')
        self.assertEqual(empty['effective_parameters']['workers'], 0)
        self.assertFalse(empty['input_valid'].any())
        self.assertTrue(np.isnan(empty['total']).all())

    def test_gpu_and_auto_gpu_keep_existing_single_worker_path(self):
        for backend in ('gpu', 'auto'):
            self.cfg['matching']['backend'] = backend
            with patch('glacier_offsets.environment.select_backend', return_value=('gpu', None)), \
                    patch('glacier_offsets.ampcor.native_match', return_value={'backend':'gpu'}) as native, \
                    patch.object(stack_matching, 'match_bursts', side_effect=AssertionError('GPU burst pool')):
                result = match_prepared(self.cfg, self.manifest, self.directory/backend, None)
            self.assertEqual(result['backend'], 'gpu')
            self.assertEqual(native.call_args.args[5], backend)

    def test_worker_failure_and_interrupt_clean_up_and_retry_without_stale_completion(self):
        # Use an actual invalid native input to propagate a child exception.
        before = {p.pid for p in multiprocessing.active_children()}
        original = self.manifest['coreg']['slcs']['secondary']
        invalid = self.f.root/'real-only.tif'
        ds = gdal.GetDriverByName('GTiff').Create(str(invalid), 384, 320, 1, gdal.GDT_Float32)
        ds.GetRasterBand(1).Fill(1); ds = None
        self.manifest['coreg']['slcs']['secondary'] = str(invalid)
        with patch.dict(os.environ, OMP_NUM_THREADS='2'):
            with self.assertRaisesRegex(ValueError, 'complex inputs'):
                match(self.cfg, self.directory, self.manifest)
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '2')
        self.assertNotIn('match_identity', self.manifest)
        self.assertFalse((self.directory/'radar/offsets.h5').exists())
        self.manifest['coreg']['slcs']['secondary'] = original

        class InterruptResult:
            def ready(self):return False
            def get(self, **kwargs):raise KeyboardInterrupt
        class InterruptPool:
            def __init__(self, pool):self.pool = pool
            def apply_async(self, *args):
                self.pool.apply_async(*args)
                return InterruptResult()
        @contextmanager
        def interrupting_pool(*args, **kwargs):
            with _native_executor(*args, **kwargs) as pool:
                yield InterruptPool(pool)
        with patch.dict(os.environ, OMP_NUM_THREADS='2'), \
                patch.object(stack_matching, '_native_executor', interrupting_pool):
            with self.assertRaises(KeyboardInterrupt):
                match(self.cfg, self.directory, self.manifest)
            self.assertEqual(os.environ['OMP_NUM_THREADS'], '2')
        self.assertNotIn('match_identity', self.manifest)
        self.assertFalse((self.directory/'radar/offsets.h5').exists())
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)
        with patch.dict(os.environ, OMP_NUM_THREADS='2'):
            match(self.cfg, self.directory, self.manifest)
        self.assertEqual(self.manifest['stages']['match'], 'complete')
        path = self.directory/'radar/offsets.h5'
        previous = (checksum(path), path.stat().st_mtime_ns)
        fingerprint = matching_specification(self.cfg, self.manifest)
        with patch.dict(os.environ, OMP_NUM_THREADS='1'), \
                patch('glacier_offsets.stack_pairs.match_prepared', side_effect=AssertionError('cache missed')):
            match(self.cfg, self.directory, self.manifest)
            self.assertEqual(matching_specification(self.cfg, self.manifest), fingerprint)
        self.assertEqual((checksum(path), path.stat().st_mtime_ns), previous)
        with h5py.File(path) as h:
            self.assertTrue(h['valid'][:].any())


if __name__ == '__main__':
    unittest.main()

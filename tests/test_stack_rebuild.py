"""Ownership migration preserves native products and resumes interrupted commits."""
from contextlib import contextmanager, ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import h5py

import test_stack_science
from glacier_offsets.cli import main
from glacier_offsets.inputs import Burst
from glacier_offsets.products import checksum, write_json
from glacier_offsets.stack_lock import stack_lock
from glacier_offsets.stack_rebuild import PREVIOUS_POLICY, rebuild_layout
from glacier_offsets.stack_storage import read_manifest
from glacier_offsets.stack_workflow import execute_stack


class StackRebuildTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_stack_science.StackScienceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.cfg = f.cfg
        self.root = Path(self.cfg['output']['directory']).resolve()
        # Turn the native fixture into two staggered swaths. The prior layout
        # cuts them in azimuth; the migration must really change ownership.
        directory, manifest = f.sources[0]
        old_key, new_key = directory.name, directory.name.replace('_iw2', '_iw1')
        f.sources[0] = (directory.with_name(new_key), manifest)
        f.sources[1][1]['coreg']['reference_grid']['starting_range'] += 128*2
        for a in f.acquisitions:
            a.bursts = {new_key if k == old_key else k:
                        Burst(new_key, 1, b.start, b.native) if k == old_key else b for k, b in a.bursts.items()}
        f.aligned_slcs = {(day, new_key if key == old_key else key): path
                          for (day, key), path in f.aligned_slcs.items()}
        f.plan = f.make_plan(f.acquisitions)
        f.plan.anchor['selection'] = {'selected_ids': [b.id for b in f.plan.bursts]}
        with patch('glacier_offsets.stack_storage.POLICY', PREVIOUS_POLICY), \
             patch('glacier_offsets.stack_ownership.POLICY', PREVIOUS_POLICY), \
             patch('glacier_offsets.stack_ownership._swath', return_value='unlabelled'):
            self.frame = f.geometry()
            self.dates = f.dates(self.frame)
        catalog = deepcopy(f.plan.report)
        catalog.update(anchor=f.plan.anchor, reference_specification_identity=self.frame['geometry_identity'],
                       geometry_identity=self.frame['geometry_identity'], layout_identity=self.frame['layout_identity'],
                       acquisitions={d: {'active': True, 'status': 'complete', 'assembly_identity': m['assembly_identity']}
                                     for d, m in self.dates.items()})
        for key in catalog['pairs']:
            root = self.root/'pairs'/key
            for name in ('radar/offsets.h5', 'geo/velocity.tif', 'diagnostics/check.png', 'native/raw.bip'):
                p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'old result')
            write_json(root/'manifest.json', {'stages': {'match': 'complete'}, 'match_identity': 'old'})
        write_json(self.root/'stack_manifest.json', catalog)
        self.calls = f.reference_call.call_count, f.align_call.call_count
        self.native = {p: (checksum(p), p.stat().st_mtime_ns) for e in self.frame['sources']
                       for p in map(Path, read_manifest(Path(e['manifest']).parent)['products'])}
        for date in self.dates.values():
            for p in map(Path, date['slcs'].values()):
                self.native[p] = checksum(p), p.stat().st_mtime_ns
            # Unused coregistration offset rasters must remain too, even when
            # they are not registered as alignment inputs for later matching.
            p = self.root/'acquisitions'/date['date']/'bursts'/f.plan.bursts[0].id/'coreg/range.off'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'retained geometric offsets')
            self.native[p] = checksum(p), p.stat().st_mtime_ns

    @contextmanager
    def inputs(self):
        with ExitStack() as stack:
            for module in ('stack_rebuild', 'stack_workflow'):
                for name, value in [('plan_stack', self.fixture.plan), ('validate_stack', 'cpu'),
                                     ('inspect_environment', self.fixture.software)]:
                    stack.enter_context(patch(f'glacier_offsets.{module}.{name}', return_value=value))
            yield

    def snapshot(self):
        return {str(p): (checksum(p), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}

    def assert_native_unchanged(self):
        self.assertEqual(self.native, {p: (checksum(p), p.stat().st_mtime_ns) for p in self.native})
        f = self.fixture
        self.assertEqual(self.calls, (f.reference_call.call_count, f.align_call.call_count))

    def test_dry_run_then_rebuild_and_normal_run_never_calls_compass(self):
        before = self.snapshot()
        with self.inputs(), patch('glacier_offsets.stack_cli.load_stack_config', return_value=self.cfg), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['stack', 'rebuild-layout', 'stack.yaml', '--dry-run']), 0)
            self.assertEqual(json.loads(output.getvalue())['aligned_dates'], 3)
        self.assertEqual(before, self.snapshot())
        with self.inputs():
            result = rebuild_layout(self.cfg)
            self.assertEqual(result['pair_directories'], 3)
            self.assertEqual(result['status'], 'complete')
            frame = read_manifest(self.root/'geometry')
            self.assertNotEqual(frame['layout_identity'], self.frame['layout_identity'])
            self.assertTrue(any('cut_global_sample' in cut for cut in frame['ownership']['cuts']))
            self.assertEqual(frame['coreg']['reference_grid'], self.frame['coreg']['reference_grid'])
            self.assertFalse(list((self.root/'pairs').iterdir()))
            for day in self.dates:
                self.assertFalse((self.root/f'acquisitions/{day}/slc.tif').exists())
                data = read_manifest(self.root/'acquisitions'/day)
                self.assertNotIn('assembly_identity', data)
                self.assertNotIn('slc_cleanup', data)
                self.assertEqual(data['status'], 'aligned')
            self.assertEqual(execute_stack(self.cfg, 'run'), 0)
            self.assertEqual(rebuild_layout(self.cfg)['status'], 'already_current')
        for key in self.fixture.plan.report['pairs']:
            with h5py.File(self.root/'pairs'/key/'radar/offsets.h5') as h:
                self.assertGreater(h['valid'][:].sum(), 0)
        self.assert_native_unchanged()

    def test_rejects_changed_reference_and_damaged_alignment_before_any_mutation(self):
        before = self.snapshot()
        with self.inputs(), patch('glacier_offsets.stack_rebuild.inspect_environment', return_value={'packages': {'changed': 'version'}}):
            with self.assertRaisesRegex(ValueError, 'unchanged reference inputs'):
                rebuild_layout(self.cfg)
        self.assertEqual(before, self.snapshot())
        path = Path(self.dates[self.fixture.acquisitions[0].date]['slcs'][self.fixture.plan.bursts[0].id])
        path.write_bytes(b'damaged')
        before = self.snapshot()
        with self.inputs(), self.assertRaisesRegex(ValueError, 'incomplete or damaged'):
            rebuild_layout(self.cfg)
        self.assertEqual(before, self.snapshot())

    def test_macos_sidecars_are_ignored_but_unregistered_pair_directories_still_fail(self):
        pairs = self.root/'pairs'
        key = next(iter(self.fixture.plan.report['pairs']))
        sidecars = [pairs/f'._{key}', pairs/'.DS_Store']
        for path in sidecars:
            path.write_bytes(b'macOS metadata')
        before = self.snapshot()
        with self.inputs():
            self.assertEqual(rebuild_layout(self.cfg, dry_run=True)['pair_directories'], 3)
        self.assertEqual(before, self.snapshot())
        unregistered = pairs/'2021-01-01_2021-01-07'
        unregistered.mkdir()
        with self.inputs(), self.assertRaisesRegex(ValueError, 'Unregistered pair output'):
            rebuild_layout(self.cfg)
        self.assertEqual(before, self.snapshot())
        unregistered.rmdir()
        with self.inputs():
            self.assertEqual(rebuild_layout(self.cfg)['pair_directories'], 3)
        self.assertEqual(set(pairs.iterdir()), set(sidecars))
        for path in sidecars:
            self.assertEqual((checksum(path), path.stat().st_mtime_ns), before[str(path)])
        self.assert_native_unchanged()

    def test_interrupted_commit_blocks_processing_and_repeated_resume_preserves_native(self):
        import glacier_offsets.stack_rebuild as module
        write = module.write_json
        counter = 0

        def interrupt(path, data):
            nonlocal counter
            if str(path).endswith('/acquisitions/2020-01-04/manifest.json'):
                counter += 1
                if counter <= 2:
                    raise KeyboardInterrupt()
            return write(path, data)

        with self.inputs(), patch('glacier_offsets.stack_rebuild.write_json', side_effect=interrupt):
            for _ in range(2):
                with self.assertRaises(KeyboardInterrupt):
                    rebuild_layout(self.cfg)
                with self.assertRaisesRegex(ValueError, 'rebuild is incomplete'):
                    execute_stack(self.cfg, 'run')
        with self.inputs():
            self.assertEqual(rebuild_layout(self.cfg, dry_run=True)['status'], 'resume_ready')
            self.assertEqual(rebuild_layout(self.cfg)['status'], 'complete')
            self.assertEqual(execute_stack(self.cfg, 'coregister'), 0)
        self.assert_native_unchanged()

    def test_cleanup_continues_when_cached_macos_sidecars_disappear(self):
        import os
        unlink = os.unlink
        disappeared = []
        for key in self.fixture.plan.report['pairs']:
            (self.root/'pairs'/key/'diagnostics/._check.png').write_bytes(b'AppleDouble')

        def remove_before_unlink(path, *args, **kwargs):
            if Path(path).name == '._check.png':
                # The traversal already cached this entry, but macOS removed
                # it as a side effect of deleting its associated PNG.
                unlink(path, *args, **kwargs)
                disappeared.append(str(path))
            return unlink(path, *args, **kwargs)

        with self.inputs(), patch('os.unlink', side_effect=remove_before_unlink):
            self.assertEqual(rebuild_layout(self.cfg)['status'], 'complete')
        self.assertEqual(len(disappeared), 3)
        self.assertFalse(list((self.root/'pairs').iterdir()))
        self.assert_native_unchanged()

    def test_cleanup_permission_failure_keeps_journal_resumable(self):
        import errno
        import os
        unlink = os.unlink

        def denied(path, *args, **kwargs):
            if Path(path).name == 'check.png':
                raise PermissionError(errno.EACCES, 'Permission denied', str(path))
            return unlink(path, *args, **kwargs)

        with self.inputs(), patch('os.unlink', side_effect=denied):
            with self.assertRaises(PermissionError):
                rebuild_layout(self.cfg)
        journal = json.loads((self.root/'.layout-rebuild/journal.json').read_text())
        self.assertEqual(journal['status'], 'applying')
        with self.inputs():
            with self.assertRaisesRegex(ValueError, 'rebuild is incomplete'):
                execute_stack(self.cfg, 'run')
            self.assertEqual(rebuild_layout(self.cfg)['status'], 'complete')
            self.assertEqual(execute_stack(self.cfg, 'coregister'), 0)
        self.assertFalse(list((self.root/'pairs').iterdir()))
        self.assert_native_unchanged()

    def test_inactive_dates_rebased_and_partial_alignment_reuses_completed_bursts(self):
        day = self.fixture.acquisitions[0].date
        path = self.root/'acquisitions'/day/'manifest.json'
        data = read_manifest(path.parent)
        data.pop('alignment_identity')
        data['status'] = 'aligning'
        write_json(path, data)
        # The current plan excludes another date, whose caches still need to be
        # rekeyed so extending the filters later does not redo its alignment.
        original = self.fixture.plan
        limited = deepcopy(original)
        limited.acquisitions = original.acquisitions[:2]
        self.fixture.plan = limited
        with self.inputs():
            self.assertEqual(rebuild_layout(self.cfg)['aligned_dates'], 2)
            self.assertEqual(execute_stack(self.cfg, 'coregister'), 0)
        self.fixture.plan = original
        with self.inputs():
            self.assertEqual(execute_stack(self.cfg, 'coregister'), 0)
        self.assert_native_unchanged()

    def test_refuses_another_writer_and_symlinked_derived_directories(self):
        with stack_lock(self.root), self.inputs():
            with self.assertRaisesRegex(ValueError, 'Another stack writer'):
                rebuild_layout(self.cfg)
            with self.assertRaisesRegex(ValueError, 'Another stack writer'):
                execute_stack(self.cfg, 'run')
        pair = next((self.root/'pairs').iterdir())
        import shutil
        shutil.rmtree(pair)
        pair.symlink_to(self.fixture.root, target_is_directory=True)
        before = self.snapshot()
        with self.inputs(), self.assertRaisesRegex(ValueError, 'symlink'):
            rebuild_layout(self.cfg)
        self.assertEqual(before, self.snapshot())


if __name__ == '__main__':
    unittest.main()

"""Merged SLC lifecycle: last consumers, restart, recovery and retained coreg."""
from contextlib import ExitStack
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from glacier_offsets.ampcor import matching_specification
from glacier_offsets.assembly import assemble, release_merged
from glacier_offsets.geocode import geocoding_specification
from glacier_offsets.products import checksum, identity, write_json
from glacier_offsets.stack_storage import assemble_date, read_manifest
from glacier_offsets.stack_workflow import execute_stack
from glacier_offsets.storage import keep_merged_slcs, preview_array, products_valid, release_slcs
import test_assembly
import test_stack_science


class StorageJournalTests(unittest.TestCase):
    def test_settings_and_interrupted_cleanup_never_accept_unrecorded_loss(self):
        self.assertFalse(keep_merged_slcs({}))
        self.assertTrue(keep_merged_slcs({'storage':{'keep_merged_slcs':True}}))
        for value in ('false', None, 1):
            with self.assertRaises(ValueError):keep_merged_slcs({'storage':{'keep_merged_slcs':value}})
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            paths = [root/'slc.tif', root/'slc.tif.msk', root/'valid.tif']
            for p in paths:p.write_bytes(p.name.encode())
            products = {str(p):checksum(p) for p in paths}
            m = {'assembly_identity':'assembly', 'products':products}
            original = Path.unlink
            count = 0
            def interrupted(path, *args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:raise KeyboardInterrupt
                return original(path, *args, **kwargs)
            with patch.object(Path, 'unlink', interrupted), self.assertRaises(KeyboardInterrupt):
                release_slcs(root, m, 'assembly_identity', [paths[0]], products)
            m = read_manifest(root)
            self.assertEqual(m['slc_cleanup']['status'], 'releasing')
            self.assertTrue(products_valid(products, m, 'assembly_identity', 'assembly', allow_released=True))
            self.assertFalse(products_valid(products, m, 'assembly_identity', 'assembly'))
            release_slcs(root, m, 'assembly_identity', [paths[0]], products)
            self.assertEqual(m['slc_cleanup']['status'], 'released')
            self.assertTrue(paths[2].is_file())
            paths[2].unlink()
            self.assertFalse(products_valid(products, m, 'assembly_identity', 'assembly', allow_released=True))
            self.assertFalse(products_valid(products, m, 'assembly_identity', 'changed', allow_released=True))

    def test_cleanup_refuses_external_or_unregistered_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); owned = root/'owned'; owned.mkdir()
            source = root/'original.slc'; source.write_bytes(b'original')
            products = {str(source):checksum(source)}
            m = {'assembly_identity':'a'}
            with self.assertRaisesRegex(ValueError, 'owned file'):
                release_slcs(owned, m, 'assembly_identity', [source], products)
            self.assertEqual(source.read_bytes(), b'original')
            with self.assertRaisesRegex(ValueError, 'Unregistered'):
                release_slcs(root, m, 'assembly_identity', [source], {})

    def test_pair_cache_release_preserves_previews_geometry_and_matching(self):
        f = test_assembly.AssemblyTests(); f.setUp(); self.addCleanup(f.doCleanups)
        root, m = assemble(f.cfg, f.sources)
        before = {side:preview_array(m, side) for side in m['coreg']['slcs']}
        sources = {p:(checksum(p), p.stat().st_mtime_ns) for folder, _ in f.sources for p in folder.glob('*.tif')}
        m['match_identity'] = 'completed-offsets'
        write_json(root/'manifest.json', m)
        release_merged(f.cfg, root, m)
        self.assertTrue(all(not Path(p).exists() for p in m['coreg']['slcs'].values()))
        self.assertTrue(all(Path(p).is_file() for p in m['coreg']['geometry'].values()))
        for side in before:np.testing.assert_array_equal(before[side], preview_array(m, side))
        _, reused = assemble(f.cfg, f.sources, allow_released=True)
        self.assertEqual(reused['match_identity'], 'completed-offsets')
        self.assertTrue(all(not Path(p).exists() for p in reused['coreg']['slcs'].values()))
        _, restored = assemble(f.cfg, f.sources)
        self.assertEqual(restored['match_identity'], 'completed-offsets')
        self.assertNotIn('slc_cleanup', restored)
        self.assertTrue(all(Path(p).is_file() for p in restored['coreg']['slcs'].values()))
        self.assertEqual(sources, {p:(checksum(p), p.stat().st_mtime_ns) for p in sources})

    def test_real_pair_workflow_resumes_and_retains_on_request_without_rematching(self):
        from glacier_offsets.batch import execute_batch
        f = test_assembly.AssemblyTests(); f.setUp(); self.addCleanup(f.doCleanups)
        with ExitStack() as stack:
            f.workflow_patches(stack)
            stack.enter_context(patch('glacier_offsets.pipeline.execute'))  # Existing synthetic COMPASS products.
            self.assertEqual(execute_batch(f.cfg, 'run'), 0)
            root = f.root/'merged'; m = read_manifest(root)
            self.assertTrue(all(not Path(p).exists() for p in m['coreg']['slcs'].values()))
            self.assertTrue((root/'diagnostics/overview.png').is_file())
            path = root/'radar/offsets.h5'; before = (checksum(path), path.stat().st_mtime_ns)
            with patch('glacier_offsets.ampcor.native_match', side_effect=AssertionError('rematched released cache')):
                self.assertEqual(execute_batch(f.cfg, 'geocode'), 0)
                self.assertEqual(execute_batch(f.cfg, 'run'), 0)
                f.cfg['storage']['keep_merged_slcs'] = True
                self.assertEqual(execute_batch(f.cfg, 'run'), 0)
            self.assertTrue(all(Path(p).is_file() for p in read_manifest(root)['coreg']['slcs'].values()))
            self.assertEqual((checksum(path), path.stat().st_mtime_ns), before)


class StackStorageTests(unittest.TestCase):
    def setUp(self):
        self.f = test_stack_science.StackScienceTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.cfg = self.f.cfg
        self.root = Path(self.cfg['output']['directory'])
        self.calls = []
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('glacier_offsets.stack_workflow.plan_stack', side_effect=lambda cfg,previous:self.f.plan))
        self.stack.enter_context(patch('glacier_offsets.stack_workflow.validate_stack', return_value='cpu'))
        self.stack.enter_context(patch('glacier_offsets.stack_workflow.inspect_environment', return_value=self.f.software))

    def fake_match(self, cfg, root, m, force=False, **kwargs):
        fingerprint = identity(matching_specification(cfg, m))
        path = root/'radar/offsets.h5'
        if not force and m.get('match_identity') == fingerprint and path.is_file():return
        # Every input must exist until its matching consumer finishes.
        self.assertTrue(all(Path(p).is_file() for p in m['coreg']['slcs'].values()))
        if not self.calls and getattr(self, 'check_first_working_set', False):
            self.assertEqual(len(list(self.root.glob('acquisitions/*/slc.tif'))), 2)
        self.calls.append(root.name)
        path.parent.mkdir(exist_ok=True); path.write_bytes(root.name.encode())
        m.update(match_identity=fingerprint, matching_summary={'quality_valid':1})
        m['stages']['match'] = 'complete'; write_json(root/'manifest.json', m)

    def fake_geocode(self, cfg, root, m, **kwargs):
        m['geocode_identity'] = identity(geocoding_specification(cfg, m))
        m['stages']['geocode'] = 'complete'; write_json(root/'manifest.json', m)

    def fake_diagnostics(self, cfg, root, m):
        for side in ('reference', 'secondary'):
            self.assertEqual(preview_array(m, side).shape, (300, 1200))
        write_json(root/'diagnostics/report.json', {})
        m['stages']['diagnostics'] = 'complete'

    def mocked_science(self):
        stack = ExitStack()
        stack.enter_context(patch('glacier_offsets.stack_workflow.match', side_effect=self.fake_match))
        stack.enter_context(patch('glacier_offsets.stack_workflow.geocode', side_effect=self.fake_geocode))
        stack.enter_context(patch('glacier_offsets.stack_workflow.diagnostics', side_effect=self.fake_diagnostics))
        return stack

    def test_default_lazy_assembly_last_use_cleanup_and_unchanged_restart(self):
        self.check_first_working_set = True
        with self.mocked_science(), patch('glacier_offsets.stack_workflow.assemble_date', wraps=assemble_date) as merge:
            self.assertEqual(execute_stack(self.cfg, 'run'), 0)
            self.assertEqual(merge.call_count, 3)
        self.assertEqual(len(self.calls), 3)
        for a in self.f.acquisitions:
            m = read_manifest(self.root/'acquisitions'/a.date)
            self.assertEqual(m['slc_cleanup']['status'], 'released')
            self.assertFalse(Path(m['slc']).exists())
            self.assertFalse(Path(m['slc'] + '.msk').exists())
            self.assertTrue(Path(m['validity_mask']).is_file())
            self.assertTrue(all(Path(p).is_file() for p in m['alignment_products']))
        paths = list(self.root.glob('pairs/*/radar/offsets.h5'))
        before = {p:(checksum(p), p.stat().st_mtime_ns) for p in paths}
        with self.mocked_science(), patch('glacier_offsets.stack_workflow.assemble_date', side_effect=AssertionError('rebuilt cache')):
            self.assertEqual(execute_stack(self.cfg, 'run'), 0)
            self.assertEqual(execute_stack(self.cfg, 'geocode'), 0)
            self.assertEqual(execute_stack(self.cfg, 'diagnostics'), 0)
        self.assertEqual(before, {p:(checksum(p), p.stat().st_mtime_ns) for p in before})
        self.assertEqual(len(self.calls), 3)

    def test_stack_restart_hashes_each_unchanged_product_once_per_invocation(self):
        with self.mocked_science():
            self.assertEqual(execute_stack(self.cfg, 'run'), 0)
        expected = set(read_manifest(self.root/'geometry')['products'])
        for acquisition in self.f.acquisitions:
            manifest = read_manifest(self.root/'acquisitions'/acquisition.date)
            expected.update(name for name in manifest['products'] if Path(name).is_file())
        for _ in range(2):
            with self.mocked_science(), patch('glacier_offsets.products.checksum', wraps=checksum) as reads:
                self.assertEqual(execute_stack(self.cfg, 'run'), 0)
            counts = Counter(str(Path(c.args[0]).resolve()) for c in reads.call_args_list)
            self.assertEqual(set(counts), expected)
            self.assertTrue(all(count == 1 for count in counts.values()), counts)
            report = json.loads((self.root/'stack_manifest.json').read_text())['metrics']
            self.assertEqual(report['product_checksum_files_hashed'], len(expected))
            self.assertEqual(report['product_checksum_bytes_hashed'], sum(Path(p).stat().st_size for p in expected))
            self.assertGreater(report['product_checksum_cache_hits'], 0)
        self.assertEqual(len(self.calls), 3)

    def test_failed_pair_keeps_required_slcs_and_retry_reuses_other_offsets(self):
        first = '_'.join(self.f.plan.pairs[0])
        original = self.fake_match
        def failing(cfg, root, m, **kwargs):
            if root.name == first:raise RuntimeError('matching failed')
            original(cfg, root, m, **kwargs)
        with self.mocked_science(), patch('glacier_offsets.stack_workflow.match', side_effect=failing):
            self.assertEqual(execute_stack(self.cfg, 'run'), 1)
        for day in self.f.plan.pairs[0]:
            self.assertTrue((self.root/'acquisitions'/day/'slc.tif').is_file())
        last = self.f.acquisitions[-1].date
        self.assertFalse((self.root/'acquisitions'/last/'slc.tif').exists())
        previous = {p:(checksum(p), p.stat().st_mtime_ns) for p in self.root.glob('pairs/*/radar/offsets.h5')}
        with self.mocked_science():self.assertEqual(execute_stack(self.cfg, 'run'), 0)
        self.assertEqual(previous, {p:(checksum(p), p.stat().st_mtime_ns) for p in previous})
        self.assertFalse(any(self.root.glob('acquisitions/*/slc.tif')))

    def test_cleanup_interruption_preserves_committed_offsets_and_resumes(self):
        from glacier_offsets.stack_storage import release_date
        self.f.plan = self.f.make_plan(self.f.acquisitions[:2])
        first = True
        def interrupted(*args):
            nonlocal first
            release_date(*args)
            if first:
                first = False
                raise KeyboardInterrupt
        with self.mocked_science(), patch('glacier_offsets.stack_workflow.release_date', side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):execute_stack(self.cfg, 'run')
        pair = next(self.root.glob('pairs/*/manifest.json')).parent
        m = read_manifest(pair)
        self.assertIn('match_identity', m)
        output = pair/'radar/offsets.h5'; before = (checksum(output), output.stat().st_mtime_ns)
        with self.mocked_science():self.assertEqual(execute_stack(self.cfg, 'run'), 0)
        self.assertEqual((checksum(output), output.stat().st_mtime_ns), before)
        self.assertEqual(len(self.calls), 1)

    def test_append_reassembles_only_needed_dates_and_retain_option_restores_caches(self):
        self.f.plan = self.f.make_plan(self.f.acquisitions[:2])
        with self.mocked_science():self.assertEqual(execute_stack(self.cfg, 'run'), 0)
        old = next(self.root.glob('pairs/*/radar/offsets.h5')); before = (checksum(old), old.stat().st_mtime_ns)
        links = [(a.date,b.date) for a,b in zip(self.f.acquisitions, self.f.acquisitions[1:])]
        self.f.plan = self.f.make_plan(self.f.acquisitions, links=links)
        with self.mocked_science(), patch('glacier_offsets.stack_workflow.assemble_date', wraps=assemble_date) as merge:
            self.assertEqual(execute_stack(self.cfg, 'run'), 0)
            self.assertEqual({c.args[1].date for c in merge.call_args_list},
                             {self.f.acquisitions[1].date, self.f.acquisitions[2].date})
        self.assertEqual((checksum(old), old.stat().st_mtime_ns), before)
        self.cfg['storage']['keep_merged_slcs'] = True
        calls = len(self.calls)
        with self.mocked_science():self.assertEqual(execute_stack(self.cfg, 'run'), 0)
        self.assertEqual(len(self.calls), calls)
        self.assertEqual(len(list(self.root.glob('acquisitions/*/slc.tif'))), 3)
        self.assertEqual(self.f.reference_call.call_count, 2)
        self.assertEqual(self.f.align_call.call_count, 4)

    def test_real_native_geocode_and_diagnostics_after_slc_release(self):
        self.f.plan = self.f.make_plan(self.f.acquisitions[:2])
        self.assertEqual(execute_stack(self.cfg, 'run'), 0)
        pair = next(self.root.glob('pairs/*/manifest.json')).parent
        self.assertTrue((pair/'geo/range_displacement_m.tif').is_file())
        self.assertFalse((pair/'geo/range_velocity_m_per_day.tif').exists())
        self.assertTrue((pair/'diagnostics/overview.png').is_file())
        self.assertTrue((pair/'diagnostics/window_locations.png').is_file())
        self.assertFalse(any(self.root.glob('acquisitions/*/slc.tif')))
        with patch('glacier_offsets.stack_workflow.match', side_effect=AssertionError('rematched')):
            self.assertEqual(execute_stack(self.cfg, 'geocode'), 0)
            self.assertEqual(execute_stack(self.cfg, 'diagnostics'), 0)


if __name__ == '__main__':unittest.main()

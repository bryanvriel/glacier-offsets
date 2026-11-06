"""Calibration restart checks parallelize reads without changing stage semantics."""
from pathlib import Path
import tempfile
from threading import Barrier, Lock, current_thread, main_thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from glacier_offsets.products import checksum, write_json
from glacier_offsets.swath_experiment_io import Experiment, Inputs
from glacier_offsets.swath_verification import prefetch_verification, verification_paths


class SwathVerificationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.exp = SimpleNamespace(cfg={'stack': {'verification_workers': 3}},
            stack=self.root/'stack', output=self.root/'calibration', inputs=Inputs(),
            dates=[f'2020-01-{n:02}' for n in range(1, 6)], frame={'products': {}})
        self.exp.links = [{'pair': f'{a}_{b}', 'earlier': a, 'later': b}
                          for a, b in zip(self.exp.dates, self.exp.dates[1:])]
        self.files = []
        for day in self.exp.dates:
            file = self.root/f'{day}.slc'
            file.write_bytes(day.encode())
            self.files.append(file)
            write_json(self.exp.stack/'acquisitions'/day/'manifest.json',
                       {'alignment_products': {str(file): checksum(file)}})

    def test_parallel_checks_are_bounded_read_only_and_cached(self):
        barrier, guard, started = Barrier(3), Lock(), []
        original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}

        def read(path):
            self.assertIsNot(current_thread(), main_thread())
            with guard:
                started.append(path)
            if path in self.files[:3]:
                barrier.wait(timeout=5)
            return checksum(path)

        with patch('glacier_offsets.products.checksum', side_effect=read) as hashed:
            prefetch_verification(self.exp, ['measure'])
            self.assertEqual(hashed.call_count, 5)
            for p in self.files:
                self.exp.inputs.file(p, checksum(p))
            self.assertEqual(hashed.call_count, 5)
        self.assertEqual(set(started), set(self.files))
        self.assertEqual(original, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in original})

    def test_cached_reports_reused_but_corruption_and_changes_detected(self):
        file = self.root/'observations.npz'; file.write_bytes(b'original')
        spec = {'version': 1}
        from glacier_offsets.products import identity
        report = self.exp.output/'measurements'/self.exp.links[0]['pair']/'report.json'
        write_json(report, {'status': 'complete', 'identity': identity(spec),
                            'products': {str(file): checksum(file)}})
        old = checksum(self.files[0])
        self.files[0].write_bytes(b'pre-existing corruption')
        prefetch_verification(self.exp, ['measure'])
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.exp.inputs.file(self.files[0], old)
        with patch('glacier_offsets.products.checksum', side_effect=AssertionError('read again')):
            self.assertTrue(Experiment.current(self.exp, report, spec))
        file.write_bytes(b'changed after prefetch')
        self.assertFalse(Experiment.current(self.exp, report, spec))
        self.exp.inputs.file(file)
        file.write_bytes(b'changed after use')
        with self.assertRaisesRegex(ValueError, 'Input changed'):
            self.exp.inputs.unchanged()

    def test_pilot_scope_alias_dedup_and_worker_override(self):
        alias = self.root/'alias'; alias.symlink_to(self.files[0])
        self.exp.frame['products'] = {str(alias): checksum(alias)}
        paths = list(verification_paths(self.exp, ['measure'], pilot_only=True))
        self.assertNotIn(str(self.files[3]), paths)
        threads = set()
        def read(path):
            threads.add(current_thread().name)
            return checksum(path)
        with patch('glacier_offsets.products.checksum', side_effect=read) as hashed:
            prefetch_verification(self.exp, ['measure'], workers=1, pilot_only=True)
        self.assertEqual(hashed.call_count, 3)
        self.assertEqual(len(threads), 1)

    def test_missing_reads_deferred_to_owning_stage(self):
        self.files[0].unlink()
        prefetch_verification(self.exp, ['measure'])
        self.assertEqual(self.exp.inputs.hashes.files_hashed, 4)
        with self.assertRaises(FileNotFoundError):
            self.exp.inputs.file(self.files[0])

    def test_interrupt_stops_submissions_and_joins_active_readers(self):
        started = []
        def read(path):
            started.append(path)
            raise KeyboardInterrupt()
        with patch('glacier_offsets.products.checksum', side_effect=read):
            with self.assertRaises(KeyboardInterrupt):
                prefetch_verification(self.exp, ['measure'], workers=2)
        self.assertLessEqual(len(started), 2)
        # All submitted reads have terminated when the caller releases its lock.
        self.assertTrue(set(started) <= set(self.files[:2]))

    def test_no_checks_for_metadata_only_stages_and_invalid_workers_rejected(self):
        for stage in ('plan', 'audit', 'report', 'rescore', 'export'):
            self.assertEqual(list(verification_paths(self.exp, [stage])), [])
        for value in (0, -1, True):
            with self.assertRaises(ValueError):
                prefetch_verification(self.exp, ['measure'], workers=value)


if __name__ == '__main__':
    unittest.main()

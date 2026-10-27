"""Concurrent restart checks remain bounded, read-only and revision-aware."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from threading import Barrier, Lock, current_thread, main_thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from glacier_offsets.products import ChecksumCache, checksum, identity, write_json
from glacier_offsets.stack_storage import SourceHashes, alignment_verified, read_manifest
from glacier_offsets.stack_verification import acquisition_checks


class StackVerificationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.cfg = {'stack': {'verification_workers': 3}, 'output': {'directory': str(self.root)}}
        self.acquisitions = [SimpleNamespace(date=f'2020-01-{n+1:02}') for n in range(8)]

    def test_checks_overlap_with_bounded_prefetch_and_date_specific_failures(self):
        barrier, guard = Barrier(3), Lock()
        started = []

        def check(cfg, acquisition, *args):
            self.assertIsNot(current_thread(), main_thread())
            with guard:
                started.append(acquisition.date)
            if acquisition in self.acquisitions[:3]:
                barrier.wait(timeout=5)
            if acquisition is self.acquisitions[1]:
                raise ValueError('bad acquisition')
            return acquisition.date

        with patch('glacier_offsets.stack_verification.check_acquisition', side_effect=check), \
                acquisition_checks(self.cfg, self.acquisitions, {}, SourceHashes(), ChecksumCache()) as checks:
            iterator = iter(checks)
            first, result = next(iterator)
            self.assertEqual(result.result(timeout=5), first.date)
            self.assertEqual(set(started), {a.date for a in self.acquisitions[:3]})
            dates = [first.date]
            for acquisition, result in iterator:
                dates.append(acquisition.date)
                if acquisition is self.acquisitions[1]:
                    with self.assertRaisesRegex(ValueError, 'bad acquisition'):
                        result.result(timeout=5)
                else:
                    self.assertEqual(result.result(timeout=5), acquisition.date)
        self.assertEqual(dates, [a.date for a in self.acquisitions])

    def test_interruption_does_not_submit_remaining_dates(self):
        checked = []
        def check(cfg, acquisition, *args):
            checked.append(acquisition.date)
        with patch('glacier_offsets.stack_verification.check_acquisition', side_effect=check):
            with self.assertRaises(KeyboardInterrupt):
                with acquisition_checks(self.cfg, self.acquisitions, {}, SourceHashes(), ChecksumCache()) as checks:
                    _, future = next(iter(checks))
                    future.result(timeout=5)
                    raise KeyboardInterrupt
        self.assertLessEqual(len(checked), 3)
        self.assertTrue(set(checked) <= {a.date for a in self.acquisitions[:3]})

    def test_warming_checks_changes_no_files_and_reuse_still_checks_revisions(self):
        frame = {'geometry_identity': 'geometry', 'sources': []}
        orbit = self.root/'orbit'; orbit.write_bytes(b'orbit')
        acquisitions = self.acquisitions[:3]
        for a in acquisitions:
            safe = self.root/f'{a.date}.SAFE'; safe.mkdir()
            (safe/'measurement').write_bytes(a.date.encode())
            a.scene, a.orbit, a.error = SimpleNamespace(path=safe), SimpleNamespace(path=orbit), None
            from glacier_offsets.stack_storage import date_specification
            spec = date_specification(a, frame, SourceHashes())
            product = self.root/f'{a.date}.tif'; product.write_bytes(b'product')
            write_json(self.root/'acquisitions'/a.date/'manifest.json', {
                'alignment_identity': identity(spec), 'products': {str(product): checksum(product)}})
        before = {p: (checksum(p), p.stat().st_mtime_ns) for p in self.root.rglob('*') if p.is_file()}
        digests = ChecksumCache()
        with acquisition_checks(self.cfg, acquisitions, frame, SourceHashes(), digests) as checks:
            specifications = {a.date: f.result(timeout=5) for a, f in checks}
        self.assertEqual(digests.files_hashed, 3)
        self.assertEqual(before, {p: (checksum(p), p.stat().st_mtime_ns) for p in before})
        first = acquisitions[0]
        manifest = read_manifest(self.root/'acquisitions'/first.date)
        self.assertTrue(alignment_verified(manifest, identity(specifications[first.date]), checksums=digests))
        (self.root/f'{first.date}.tif').write_bytes(b'damaged')
        self.assertFalse(alignment_verified(manifest, identity(specifications[first.date]), checksums=digests))

    def test_raw_source_hashes_share_only_identical_paths_without_serializing_dates(self):
        paths = [self.root/'one', self.root/'two']
        for p in paths:
            p.write_bytes(p.name.encode())
        hashes, barrier = SourceHashes(), Barrier(2)
        def read(path):
            barrier.wait(timeout=5)
            return checksum(path)
        with patch('glacier_offsets.stack_storage.checksum', side_effect=read) as digest, \
                ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(hashes, paths*2))
        self.assertEqual(digest.call_count, 2)
        self.assertEqual(results[:2], results[2:])

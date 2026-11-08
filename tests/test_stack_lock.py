"""Cross-process shared-reader and exclusive-writer stack locking."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from glacier_offsets.stack_lock import stack_lock


PROBE = '''
from glacier_offsets.stack_lock import stack_lock
import sys
try:
    with stack_lock(sys.argv[1], shared=sys.argv[2] == 'shared'):
        print('acquired')
except ValueError as error:
    print(str(error))
    raise SystemExit(2)
'''


class StackLockTests(unittest.TestCase):
    def probe(self, root, shared=False):
        env = dict(os.environ)
        source = str(Path(__file__).resolve().parents[1] / 'src')
        env['PYTHONPATH'] = source + os.pathsep + env.get('PYTHONPATH', '')
        return subprocess.run(
            [sys.executable, '-c', PROBE, str(root), 'shared' if shared else 'exclusive'],
            env=env, capture_output=True, text=True, timeout=10)

    def test_readers_share_source_without_creating_lock_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with stack_lock(root, shared=True):
                result = self.probe(root, shared=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), 'acquired')
                self.assertEqual(list(root.iterdir()), [])

    def test_reader_blocks_production_writer(self):
        with tempfile.TemporaryDirectory() as temp:
            with stack_lock(temp, shared=True):
                result = self.probe(temp)
                self.assertEqual(result.returncode, 2)
                self.assertIn('writer or reader', result.stdout)
                self.assertIn(temp, result.stdout)

    def test_writer_blocks_reader_and_other_writer(self):
        with tempfile.TemporaryDirectory() as temp:
            with stack_lock(temp):
                reader = self.probe(temp, shared=True)
                writer = self.probe(temp)
                self.assertEqual(reader.returncode, 2)
                self.assertIn('source stack lock', reader.stdout)
                self.assertIn(temp, reader.stdout)
                self.assertEqual(writer.returncode, 2)
                self.assertIn('Another stack writer', writer.stdout)

    def test_exception_releases_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                with stack_lock(temp, shared=True):
                    raise RuntimeError('failure')
            self.assertEqual(self.probe(temp).returncode, 0)

    def test_unfinished_layout_blocks_shared_readers(self):
        with tempfile.TemporaryDirectory() as temp:
            journal = Path(temp) / '.layout-rebuild/journal.json'
            journal.parent.mkdir()
            journal.write_text(json.dumps({'status': 'running'}))
            with self.assertRaisesRegex(ValueError, 'Layout rebuild is incomplete'):
                with stack_lock(temp, shared=True):
                    pass
            with self.assertRaisesRegex(ValueError, 'exclusive'):
                with stack_lock(temp, shared=True, rebuilding=True):
                    pass


if __name__ == '__main__':
    unittest.main()

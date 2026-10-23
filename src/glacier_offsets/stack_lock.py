"""Exclusive stack writers; inspection and dry runs do not create lock files."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path


@contextmanager
def stack_lock(root, *, create=False, rebuilding=False):
    root = Path(root)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    # Lock the directory inode so read-only preflight does not create a file.
    descriptor = os.open(root, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another stack writer is running in this output directory') from exc
        journal = root/'.layout-rebuild/journal.json'
        if not rebuilding and journal.is_file() and json.loads(journal.read_text()).get('status') != 'complete':
            raise ValueError('Layout rebuild is incomplete; rerun stack rebuild-layout before processing')
        yield
    finally:
        os.close(descriptor)

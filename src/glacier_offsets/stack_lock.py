"""Exclusive stack writers and shared readers; no lock files are created."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path


@contextmanager
def stack_lock(root, *, create=False, rebuilding=False, shared=False):
    """Protect a stack directory against overlapping readers and writers.

    Shared locks allow concurrent read-only experiments. All production writes
    retain the default exclusive lock, including layout rebuilds.
    """
    root = Path(root)
    if shared and rebuilding:
        raise ValueError('Layout rebuilding requires an exclusive stack lock')
    if create:
        root.mkdir(parents=True, exist_ok=True)
    # Lock the directory inode so read-only preflight does not create a file.
    descriptor = os.open(root, os.O_RDONLY)
    try:
        try:
            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            if shared:
                raise ValueError(f'A stack writer holds the source stack lock: {root}') from exc
            raise ValueError(f'Another stack writer or reader is running in this output directory: {root}') from exc
        journal = root/'.layout-rebuild/journal.json'
        if not rebuilding and journal.is_file() and json.loads(journal.read_text()).get('status') != 'complete':
            raise ValueError('Layout rebuild is incomplete; rerun stack rebuild-layout before processing')
        yield
    finally:
        os.close(descriptor)

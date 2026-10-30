"""Immutable, content-identified cache artifacts with resumable partial builds.

An artifact is identified by the SHA-256 of its canonical identity record
(inputs, settings and implementation). It is built in a private partial
directory under an exclusive lock, then published by one atomic rename.
Published files are read-only and checksum-verified on every reuse.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from .velocity_io import digest, serializable, write_json


RECORD = 'record.json'
IDENTITY = 'identity.json'


def canonical(value):
    return json.dumps(serializable(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def identity_digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@contextmanager
def exclusive_lock(path, description):
    """Non-blocking writer lock; a concurrent writer fails instead of waiting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'Another process is writing {description}') from error
        yield


class Artifact:
    """One cache entry: ``<cache>/<kind>/<id>/`` with ``record.json``."""

    def __init__(self, root, kind, identity):
        self.root = Path(root)
        self.kind = kind
        self.identity = serializable(identity)
        self.id = identity_digest(identity)[:32]
        self.directory = self.root / kind / self.id
        self.partial = self.root / kind / f'.{self.id}.partial'
        self.lockfile = self.root / kind / f'.{self.id}.lock'

    def path(self, name):
        return self.directory / name

    def lock(self):
        return exclusive_lock(self.lockfile, f'{self.kind} cache artifact {self.id}')

    def record(self):
        return json.loads((self.directory / RECORD).read_text())

    def complete(self):
        """Return the verified record, or None when the artifact is not published."""
        if not self.directory.is_dir():
            return None
        record_file = self.directory / RECORD
        if not record_file.is_file():
            raise ValueError(f'Cache artifact {self.directory} has no record; remove it after inspection')
        record = json.loads(record_file.read_text())
        if record.get('identity') != json.loads(canonical(self.identity)):
            raise ValueError(f'Cache artifact {self.directory} identity does not match its directory')
        for name, expected in record['files'].items():
            path = self.directory / name
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f'Cache artifact file changed: {path}')
        return record

    def begin(self):
        """Return the partial directory, resuming only an identical identity."""
        identity_file = self.partial / IDENTITY
        if self.partial.exists():
            same = identity_file.is_file() and json.loads(identity_file.read_text()) == json.loads(canonical(self.identity))
            if not same:
                shutil.rmtree(self.partial)
        if not self.partial.exists():
            self.partial.mkdir(parents=True)
            write_json(identity_file, self.identity)
            return self.partial, False
        return self.partial, True

    def publish(self, files, metadata=None):
        """Checksum the listed partial files, make them read-only, then rename."""
        hashes = {}
        for name in files:
            path = self.partial / name
            if not path.is_file():
                raise ValueError(f'Missing artifact file before publication: {path}')
            hashes[name] = digest(path)
        for leftover in self.partial.iterdir():
            if leftover.name not in hashes and leftover.name != IDENTITY:
                if leftover.is_dir():
                    shutil.rmtree(leftover)
                else:
                    leftover.unlink()
        record = {'schema': 'glacier-offsets-velocity-cache-v1', 'kind': self.kind, 'id': self.id,
                  'identity': self.identity, 'files': hashes, **(metadata or {})}
        write_json(self.partial / RECORD, record)
        (self.partial / IDENTITY).unlink()
        for name in (*hashes, RECORD):
            os.chmod(self.partial / name, 0o444)
        os.replace(self.partial, self.directory)
        return self.complete()

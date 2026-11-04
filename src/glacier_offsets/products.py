"""Atomic metadata, checksums and stage identities."""
import hashlib
import json
import os
from pathlib import Path
import stat
from threading import Lock


def checksum(path):
    path = Path(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for file in sorted(p for p in path.rglob('*') if p.is_file()):
            digest.update(str(file.relative_to(path)).encode())
            digest.update(checksum(file).encode())
        return digest.hexdigest()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class ChecksumCache:
    """Reuse file digests only within one invocation and one file revision.

    Every lookup stats the file. Rewrites, replacements and deletions cannot
    reuse a digest merely because the manifest or pathname is unchanged.
    Directories are deliberately not cached: their own timestamps do not
    describe changes to descendants. Explicit trusted-restart seeding is the
    exception; no cache is persisted between runs.
    """
    def __init__(self):
        self._files = {}
        self._trusted = {}
        self._locks = {}
        self._guard = Lock()
        self.hits = 0
        self.files_hashed = 0
        self.bytes_hashed = 0
        self.trusted_hits = 0
        self.trusted_mode = False

    def trust_existing(self, products):
        """Explicitly trust recorded digests at the current filesystem revision.

        This does not establish content integrity. Directories are trusted as a
        unit, without inspecting their members. Changed revisions fall back to
        hashing; new outputs must never inherit an old digest after a rewrite.
        Seed before starting concurrent readers.
        """
        self.trusted_mode = True
        for name, digest in products.items():
            path = Path(name).resolve()
            old = self._trusted.get(path)
            if old is not None and old[1] != digest:
                raise ValueError(f'Conflicting recorded checksums for {path}; rerun without --no-check')
            try:
                revision = self._revision(path.stat())
            except FileNotFoundError:
                continue
            self._trusted[path] = (revision, digest)

    @staticmethod
    def _revision(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)

    def __call__(self, path):
        path = Path(path).resolve()
        with self._guard:
            lock = self._locks.setdefault(path, Lock())
        # Serialize aliases of one file, while hashing different files concurrently.
        with lock:
            return self._digest(path)

    def _digest(self, path):
        before = path.stat()
        trusted = self._trusted.get(path)
        if trusted is not None and trusted[0] == self._revision(before):
            with self._guard:
                self.trusted_hits += 1
            return trusted[1]
        if not stat.S_ISREG(before.st_mode):
            return checksum(path)
        revision = self._revision(before)
        saved = self._files.get(path)
        if saved is not None and saved[0] == revision:
            with self._guard:
                self.hits += 1
            return saved[1]
        digest = checksum(path)
        if self._revision(path.stat()) != revision:
            self._files.pop(path, None)
            raise ValueError(f'File changed while computing checksum: {path}')
        self._files[path] = (revision, digest)
        with self._guard:
            self.files_hashed += 1
            self.bytes_hashed += before.st_size
        return digest


def identity(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, path)


def input_provenance(cfg):
    keys = ('reference_safe', 'secondary_safe', 'reference_orbit', 'secondary_orbit', 'dem')
    hashes = {p: checksum(p) for p in dict.fromkeys(cfg['inputs'][k] for k in keys)}
    return {k: {'path': cfg['inputs'][k], 'sha256': hashes[cfg['inputs'][k]]} for k in keys}

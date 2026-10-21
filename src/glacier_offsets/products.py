"""Atomic metadata, checksums and stage identities."""
import hashlib
import json
import os
from pathlib import Path


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

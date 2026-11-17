"""Reuse full GOFF validation only for verified file bytes and validator code."""
import json
import logging
from pathlib import Path
from time import perf_counter

from .displacement import validate_nisar
from .velocity_cache import exclusive_lock
from .velocity_io import digest, fingerprint, revision, write_json


def verified_nisar(path, cache, expected=None):
    path = Path(path)
    before = revision(path)
    start = perf_counter()
    if expected is not None:
        if list(before) != expected['revision']:
            raise ValueError('Input package changed since collection')
        sha = expected['sha256']
    else:
        logging.info('Checksumming NISAR package: %s', path)
        sha = digest(path)
    implementation = {k: digest(Path(__file__).with_name(k)) for k in ('displacement.py', 'nisar_grid.py', 'velocity_io.py', 'nisar_validation.py')}
    key = {'schema': 'nisar-validation-v1', 'sha256': sha, 'implementation': implementation}
    record = Path(cache)/'verified-nisar'/(fingerprint(key)+'.json')
    def read():
        try:
            value = json.loads(record.read_text())
            return value if value['identity'] == key else None
        except (OSError, ValueError, KeyError):
            return None
    value = read()
    if value is None:
        with exclusive_lock(record.with_suffix('.lock'), 'NISAR validation'):
            value = read()
            if value is None:
                logging.info('Fully validating NISAR arrays: %s', path)
                result = validate_nisar(path)
                if revision(path) != before:
                    raise ValueError('Input changed during validation')
                value = {'identity': key, 'result': result}
                write_json(record, value)
    if revision(path) != before:
        raise ValueError('Input changed during validation')
    logging.info('NISAR validation verified in %.1fs: %s', perf_counter()-start, path)
    return sha

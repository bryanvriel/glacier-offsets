"""Bounded native geometry jobs and verified, atomically published tile caches.

Only the parent reads measurement arrays and writes the final package. Spawned
workers open their own source handles; HDF5 handles are never shared or forked.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import logging
import multiprocessing
import os
from pathlib import Path
import tempfile
from time import perf_counter
import zipfile
import zlib

import h5py
import numpy as np

from .displacement import valid_nisar_mask
from .products import identity
from .velocity_io import revision


def default_workers():
    """Conservative automatic parallelism, respecting Linux CPU affinity."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    return min(4, cpus or 1)


def geometry_worker(path, expected_revision, points, velocity):
    from .nisar_packaging import native_geometry
    if list(revision(path)) != expected_revision:
        raise ValueError('Source subset changed during geometry calculation')
    with h5py.File(path, 'r') as h:
        result = native_geometry(h, points, {'groundTrackVelocity': velocity})
    if list(revision(path)) != expected_revision:
        raise ValueError('Source subset changed during geometry calculation')
    return result


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def read_tile(path, key, shape):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as h:
            native = h['native']
            if (str(h['key']) != key or native.shape != shape or native.dtype != np.dtype('float64')
                    or str(h['sha256']) != array_digest(native)):
                raise ValueError('geometry identity, shape or checksum mismatch')
        return native
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
        logging.warning('Rebuilding damaged NISAR geometry cache %s: %s', path, exc)
        return None


def write_tile(path, key, native):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent packages may calculate the same tile. Each publishes a complete
    # verified entry by rename, so neither readers nor writers see partial data.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.tile-', suffix='.tmp', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, key=key, native=native, sha256=array_digest(native))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class GeometryRunner:
    """Pipeline at most two tiles per worker, with deterministic output order."""

    def __init__(self, workers, cache, fingerprint):
        self.workers = default_workers() if workers is None else workers
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError('NISAR workers must be a positive integer')
        self.cache = Path(cache) if cache is not None else None
        self.fingerprint = fingerprint
        self.executor = None
        self.stats = dict(evaluated_cells=0, skipped_cells=0, cached_cells=0, cache_hits=0, computed_tiles=0)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def tiles(self, h, record, layer):
        from .nisar_packaging import VARIABLES, interpolators, native_geometry
        from .packaging import tiles
        interps = interpolators(h)
        pending = deque()
        start = last_log = perf_counter()
        total = ((len(record['y'])+255)//256)*((len(record['x'])+255)//256)
        before = self.stats.copy()
        completed = 0
        source_selection = record.get('source_selection', (slice(0, None, 1), slice(0, None, 1)))
        prefix = identity({'source_sha256': record['sha256'], 'layer': layer,
                           'sampling': record.get('sampling'),
                           'source_selection': [[s.start, s.stop, s.step] for s in source_selection], **self.fingerprint})

        def finish(item):
            nonlocal completed, last_log
            sec, arrays, mask, height, cube, eligible, native, future, path, key, hit = item
            if future is not None:
                native.reshape(-1, 3, 3)[eligible.ravel()] = future.result()
            if path is not None:
                if list(revision(record['path'])) != record['revision']:
                    raise ValueError('Source subset changed during geometry calculation')
                write_tile(path, key, native)
            count = int(eligible.sum())
            self.stats['skipped_cells'] += height.size-count
            if hit:
                self.stats['cache_hits'] += 1
                self.stats['cached_cells'] += count
            else:
                self.stats['evaluated_cells'] += count
                self.stats['computed_tiles'] += 1
            completed += 1
            now = perf_counter()
            if completed == 1 or completed == total or now-last_log >= 5:
                elapsed = now-start
                logging.info('NISAR tiles %d/%d (%.0f%%), %.1fs elapsed, %.1fs remaining; '
                             '%d cells computed, %d skipped, %d cached: %s',
                             completed, total, 100*completed/total, elapsed,
                             elapsed*(total-completed)/completed,
                             self.stats['evaluated_cells']-before['evaluated_cells'],
                             self.stats['skipped_cells']-before['skipped_cells'],
                             self.stats['cached_cells']-before['cached_cells'], record['granuleId'])
                last_log = now
            return sec, arrays, mask, height, cube, native

        for sec in tiles((len(record['y']), len(record['x']))):
            source = tuple(slice(s.start+t.start*s.step, s.start+(t.stop-1)*s.step+1, s.step)
                           for s, t in zip(source_selection, sec))
            arrays = {k: h[f'layer{layer}/{k}'][source] for k in VARIABLES}
            mask, height = h['mask'][source], h['surfaceHeight'][source]
            px, py = np.meshgrid(record['x'][sec[1]], record['y'][sec[0]])
            points = np.stack([height.ravel(), py.ravel(), px.ravel()], -1)
            cube = {k: f(points) for k, f in interps.items()}
            cube_good = np.isfinite(np.stack(list(cube.values()), -1)).all(-1).reshape(height.shape)
            eligible = (valid_nisar_mask(mask) & np.isfinite(arrays['alongTrackOffset'])
                        & np.isfinite(arrays['slantRangeOffset']) & np.isfinite(height) & cube_good)
            count = int(eligible.sum())
            key = identity({'product': prefix, 'tile': [[s.start, s.stop] for s in sec]})
            path = self.cache / prefix / (key+'.npz') if self.cache is not None else None
            shape = (*height.shape, 3, 3)
            native = read_tile(path, key, shape) if path is not None else None
            hit = native is not None
            future = None
            if native is not None:
                path = None
            else:
                native = np.full(shape, np.nan, dtype='float64')
                if count:
                    selected = eligible.ravel()
                    velocity = cube['groundTrackVelocity'][selected]
                    if self.workers == 1:
                        native.reshape(-1, 3, 3)[selected] = native_geometry(h, points[selected], {'groundTrackVelocity': velocity})
                    else:
                        if self.executor is None:
                            self.executor = ProcessPoolExecutor(max_workers=self.workers,
                                                               mp_context=multiprocessing.get_context('spawn'))
                        future = self.executor.submit(geometry_worker, record['path'], record['revision'], points[selected], velocity)
            pending.append((sec, arrays, mask, height, cube, eligible, native, future, path, key, hit))
            if len(pending) >= (2*self.workers if self.workers > 1 else 1):
                yield finish(pending.popleft())
        while pending:
            yield finish(pending.popleft())

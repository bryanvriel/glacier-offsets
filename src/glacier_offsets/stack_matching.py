"""CPU burst scheduling on the unchanged assembled stack matching grid.

Only native calls run in children. The parent owns masks, sparse output arrays,
provenance and publication; no GDAL objects or native bursts cross processes.
"""
from collections import deque
import logging
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from .ampcor import _empty_match, _native_executor, _native_worker, _write_native_layout
from .assembly import source_indices
from .matching_grid import MatchingGrid


def cpu_worker_budget():
    """Use the first OpenMP level as a process budget; unset means sequential."""
    value = os.environ.get('OMP_NUM_THREADS', '1')
    try:
        levels = [int(part.strip()) for part in value.split(',')]
        if any(level <= 0 for level in levels):
            raise ValueError
    except ValueError as exc:
        raise ValueError('OMP_NUM_THREADS must contain positive integers, e.g. 4 or 4,1') from exc
    return levels[0]


def burst_rectangles(owner, selected):
    """Coalesce equal row runs into rectangles, never changing chip coordinates.

    Each selected window appears exactly once, including ragged/holed ownership.
    Inputs remain the original full SLCs, so chip/search halos cross scheduling
    boundaries without introducing artificial image edges.
    """
    active = {}
    rectangles = []
    for row in range(owner.shape[0]):
        labels = np.where(selected[row], owner[row], 0)
        edges = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
        following = {}
        for lo, hi in zip(edges, edges[1:]):
            label = int(labels[lo])
            if not label:
                continue
            key = (label, int(lo), int(hi))
            first = active.pop(key, row)
            following[key] = first
        rectangles.extend((label, first, row, lo, hi) for (label, lo, hi), first in active.items())
        active = following
    rectangles.extend((label, first, owner.shape[0], lo, hi)
                      for (label, lo, hi), first in active.items())
    return sorted(rectangles)


def match_bursts(cfg, manifest, directory, prior, *, full_grid=False):
    """One in-flight native call per burst, up to the OpenMP CPU budget.

    ``prior.valid`` already includes full-SLC support/bounds checks. Constant
    matching's full-grid path also retains its raw invalid-window measurements;
    unowned/missing regions share the first runnable burst's scheduling lane.
    Varying/edge-bounded matching continues to call native Ampcor only where
    eligible, leaving the other raw windows as NaN.
    """
    budget = cpu_worker_budget()
    grid = prior['grid']
    result = _empty_match(grid, prior['valid'].copy(), 'cpu')
    velocity = cfg['gross_offsets']['mode'] == 'velocity'
    if velocity:
        result['prior'] = prior
    owner = source_indices(manifest['coreg'], *grid.centers)
    sources = manifest['assembly']['sources']
    runnable = [index for index, source in enumerate(sources, 1)
                if source['interval_seconds'] is not None and source['interval_seconds'] > 0
                and np.any(prior['valid'] & (owner == index))]
    workers = min(budget, len(runnable))
    effective = {'strategy': 'stack-burst-cpu-v1', 'mergeGrossOffset': 0,
                 'requested_workers': budget, 'workers': workers, 'threads_per_worker': 1,
                 'partition_policy': 'center ownership; unowned/inactive windows share first runnable lane',
                 'segments': []}
    result['effective_parameters'] = effective
    output = Path(directory) / 'native'
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if workers:
        # These are scheduling labels only; scientific ownership is unchanged
        # and annotate_result still applies the original seam/interval masks.
        labels = np.where(np.isin(owner, runnable), owner, runnable[0])
        selected = np.ones(grid.shape, bool) if full_grid else prior['valid']
        lanes = {index: deque() for index in runnable}
        for number, (index, y0, y1, x0, x1) in enumerate(burst_rectangles(labels, selected)):
            lanes[index].append((number, y0, y1, x0, x1))
        ready = deque(index for index in runnable if lanes[index])
        active = {}
        records = {}
        logging.info('CPU stack matching: %d workers for %d runnable bursts (OMP_NUM_THREADS=%d)',
                     workers, len(runnable), budget)
        with _native_executor(workers, single_threaded=True) as pool:
            while ready or active:
                while ready and len(active) < workers:
                    index = ready.popleft()
                    number, y0, y1, x0, x1 = lanes[index].popleft()
                    section = (slice(y0, y1), slice(x0, x1))
                    local = MatchingGrid(grid.starts_line[y0:y1], grid.starts_sample[x0:x1],
                                         grid.window, grid.search, grid.stride)
                    args = (manifest['coreg']['slcs']['reference'], manifest['coreg']['slcs']['secondary'],
                            output / f'burst-{index:04d}-part-{number:06d}', cfg['matching'], cfg, 'cpu')
                    kwargs = {'_grid': local, '_valid_input': prior['valid'][section],
                              '_gross': None if full_grid else prior['integer_px'][section]}
                    pending = pool.apply_async(_native_worker, ((args, kwargs),))
                    active[index] = (number, section, pending)
                completed = [index for index, (_, _, pending) in active.items() if pending.ready()]
                if not completed:
                    # Timed get keeps Ctrl-C responsive even on native failures.
                    try:
                        next(iter(active.values()))[2].get(timeout=.05)
                    except multiprocessing.TimeoutError:
                        pass
                    continue
                for index in completed:
                    number, section, pending = active.pop(index)
                    native = pending.get()  # Failure terminates/joins the pool; no product is published.
                    for name in ('gross', 'residual', 'total', 'snr', 'correlation_peak', 'covariance',
                                 'finite', 'search_boundary', 'valid'):
                        result[name][section] = native[name]
                    result['elapsed_seconds'] += native['elapsed_seconds']
                    records[number] = native['effective_parameters'] | {
                        'scheduling_burst_id': sources[index-1]['burst_id'],
                        'grid_slice': [[part.start, part.stop] for part in section]}
                    if lanes[index]:
                        ready.append(index)
        effective['segments'] = [records[number] for number in sorted(records)]
    effective['wall_seconds'] = time.perf_counter() - started
    _write_native_layout(output, result)
    return result

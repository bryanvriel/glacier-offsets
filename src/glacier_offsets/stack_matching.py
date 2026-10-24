"""CPU burst scheduling on the unchanged assembled stack matching grid.

Only native calls run in children. The parent owns masks, sparse output arrays,
provenance and publication; no GDAL objects or native bursts cross processes.
"""
from collections import deque
from contextlib import ExitStack
import logging
import multiprocessing
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from .ampcor import _empty_match, _native_executor, _native_worker, _write_native_layout
from .assembly import source_indices
from .fftw_cache import pinned_wisdom
from . import range_oversampling
from .storage import scratch_directory
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


def matching_rectangles(owner, selected, runnable, workers):
    """Keep burst lanes and split other work into bounded, shareable tasks.

    None denotes scheduling work without a runnable source, not scientific
    ownership. Full-grid matching retains these raw invalid-window results.
    Bounding their size to half an average worker's workload avoids a single
    large padding rectangle becoming the serial tail of a partial acquisition.
    """
    if not workers:
        return []
    labels = np.where(np.isin(owner, runnable), owner, -1)
    target = max(1, (int(selected.sum()) + 2*workers-1)//(2*workers))
    tasks = []
    for index, y0, y1, x0, x1 in burst_rectangles(labels, selected):
        if index != -1:
            tasks.append((index, y0, y1, x0, x1))
            continue
        width = min(x1-x0, target)
        height = max(1, target//width)
        for row in range(y0, y1, height):
            for col in range(x0, x1, width):
                tasks.append((None, row, min(row+height, y1), col, min(col+width, x1)))
    return tasks


def match_bursts(cfg, manifest, directory, prior, *, full_grid=False):
    """One in-flight native call per burst, up to the OpenMP CPU budget.

    ``prior.valid`` already includes full-SLC support/bounds checks. Constant
    matching's full-grid path also retains its raw invalid-window measurements;
    unowned/missing regions use a shared queue available to any idle worker.
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
    effective = {'strategy': 'stack-burst-cpu-v2', 'mergeGrossOffset': 0,
                 'requested_workers': budget, 'workers': workers, 'threads_per_worker': 1,
                 'partition_policy': 'center ownership; bounded unowned/inactive tasks shared by idle workers',
                 'segments': []}
    result['effective_parameters'] = effective
    output = Path(directory) / 'native'
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if workers:
        selected = np.ones(grid.shape, bool) if full_grid else prior['valid']
        lanes = {index: deque() for index in runnable}
        shared = deque()
        for number, (index, y0, y1, x0, x1) in enumerate(matching_rectangles(owner, selected, runnable, workers)):
            queue = shared if index is None else lanes[index]
            queue.append((number, y0, y1, x0, x1))
        ready = deque(index for index in runnable if lanes[index])
        active = {}
        records = {}
        logging.info('CPU stack matching: %d workers for %d runnable bursts (OMP_NUM_THREADS=%d)',
                     workers, len(runnable), budget)
        inputs = manifest['coreg']['slcs']
        worker, native = _native_worker, cfg['matching']
        range2x = range_oversampling.factor(cfg['matching']) != 1
        if range2x:
            # Workers convert each segment between original and range-2x native coordinates.
            worker, native = range_oversampling.range_worker, range_oversampling.native_parameters(cfg['matching'])
        wisdom = pinned_wisdom(native)
        if wisdom is not None:
            if wisdom['measured_now']:
                logging.info('Measured pinned FFTW wisdom %s', wisdom['path'])
            effective['fftw_wisdom'] = {k: wisdom[k] for k in ('method', 'path', 'key_identity', 'sha256')}
        context = ExitStack()
        if range2x:
            scratch = scratch_directory(cfg)
            temporary = (output/'range2x' if scratch is None else
                         Path(context.enter_context(tempfile.TemporaryDirectory(prefix='range2x-', dir=scratch)))/'inputs')
            inputs, effective['range_oversampling'] = context.enter_context(
                range_oversampling.prepared_inputs(inputs, temporary, workers=budget))
        with context, _native_executor(workers, single_threaded=True, wisdom=wisdom) as pool:
            while ready or shared or active:
                while (ready or shared) and len(active) < workers:
                    index = ready.popleft() if ready else None
                    queue = shared if index is None else lanes[index]
                    number, y0, y1, x0, x1 = queue.popleft()
                    section = (slice(y0, y1), slice(x0, x1))
                    local = MatchingGrid(grid.starts_line[y0:y1], grid.starts_sample[x0:x1],
                                         grid.window, grid.search, grid.stride)
                    args = (inputs['reference'], inputs['secondary'],
                            output / (f'shared-part-{number:06d}' if index is None else
                                      f'burst-{index:04d}-part-{number:06d}'), cfg['matching'], cfg, 'cpu')
                    kwargs = {'_grid': local, '_valid_input': prior['valid'][section],
                              '_gross': None if full_grid else prior['integer_px'][section]}
                    pending = pool.apply_async(worker, ((args, kwargs),))
                    active[number] = (index, section, pending)
                completed = [number for number, (_, _, pending) in active.items() if pending.ready()]
                if not completed:
                    # Timed get keeps Ctrl-C responsive even on native failures.
                    try:
                        next(iter(active.values()))[2].get(timeout=.05)
                    except multiprocessing.TimeoutError:
                        pass
                    continue
                for number in completed:
                    index, section, pending = active.pop(number)
                    native = pending.get()  # Failure terminates/joins the pool; no product is published.
                    for name in ('gross', 'residual', 'total', 'snr', 'correlation_peak', 'covariance',
                                 'finite', 'search_boundary', 'valid'):
                        result[name][section] = native[name]
                    result['elapsed_seconds'] += native['elapsed_seconds']
                    records[number] = native['effective_parameters'] | {
                        'scheduling_burst_id': sources[index-1]['burst_id'] if index is not None else None,
                        'work_kind': 'burst' if index is not None else 'unowned_or_inactive',
                        'elapsed_seconds': native['elapsed_seconds'],
                        'grid_slice': [[part.start, part.stop] for part in section]}
                    if index is not None and lanes[index]:
                        ready.append(index)
        effective['segments'] = [records[number] for number in sorted(records)]
    effective['wall_seconds'] = time.perf_counter() - started
    _write_native_layout(output, result)
    return result

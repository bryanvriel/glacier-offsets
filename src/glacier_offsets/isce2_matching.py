"""Actual ISCE2 CPU magnitude matching through an isolated interpreter."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from osgeo import gdal


def validate_settings(cfg):
    p = cfg['matching']
    if p['backend'] == 'gpu':
        raise ValueError('ISCE2 magnitude Ampcor is a CPU engine; select backend cpu or auto')
    if p['raw_oversampling'] != 2 or any(v % 2 for v in p['window']):
        raise ValueError('ISCE2 Ampcor requires raw_oversampling=2 and even chip dimensions')
    if p['correlation_threshold'] != 0:
        raise ValueError('ISCE2 Ampcor does not expose refined correlation peaks; use snr_threshold and correlation_threshold=0')
    settings = p.get('isce2', {})
    if not isinstance(settings, dict) or set(settings)-{'python', 'packages', 'library_path'}:
        raise ValueError('matching.isce2 accepts python, packages, library_path')
    for key, value in settings.items():
        if value is not None and (not isinstance(value, str) or not Path(value).exists()):
            raise ValueError(f'matching.isce2.{key} must be an existing local path or null')


def runtime(cfg):
    settings = cfg['matching'].get('isce2', {})
    return tuple(settings.get(key) for key in ('python', 'packages', 'library_path'))


def child_environment(settings):
    _, packages, libraries = settings
    env = os.environ.copy()
    if packages:
        env['PYTHONPATH'] = packages
    elif settings[0] and Path(settings[0]).resolve() != Path(sys.executable).resolve():
        env.pop('PYTHONPATH', None)  # Avoid importing another Python ABI's project dependencies.
    if libraries:
        key = 'DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'
        env[key] = libraries + (os.pathsep + env[key] if env.get(key) else '')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    return env


def invoke(settings, arguments, log):
    with Path(log).open('w') as stream:
        result = subprocess.run([settings[0] or sys.executable, str(Path(__file__).with_name('isce2_worker.py')),
                                 *map(str, arguments)], env=child_environment(settings),
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        detail = Path(log).read_text()[-4000:]
        raise RuntimeError(f'ISCE2 magnitude worker failed ({result.returncode}); see {log}\n{detail}')


@lru_cache(maxsize=8)
def _probe(settings):
    with tempfile.TemporaryDirectory(prefix='glacier-isce2-probe-') as scratch:
        path = Path(scratch)
        invoke(settings, ['--probe', path/'provenance.json'], path/'probe.log')
        return json.loads((path/'provenance.json').read_text())


def provenance(cfg):
    validate_settings(cfg)
    return _probe(runtime(cfg))


def match(reference, secondary, output, params, cfg, *, prior=None, grid=None, valid_input=None):
    from .ampcor import _empty_match, _write_native_layout
    from .config import gross_values
    from .matching_grid import build_grid, valid_windows
    from .stack_matching import burst_rectangles, cpu_worker_budget
    validate_settings(cfg)
    native = provenance(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = [gdal.Open(str(p)) for p in (reference, secondary)]
    if any(ds is None or ds.RasterCount != 1 or not gdal.DataTypeIsComplex(ds.GetRasterBand(1).DataType) for ds in datasets):
        raise ValueError('ISCE2 Ampcor requires single-band complex SLCs')
    shape = datasets[0].RasterYSize, datasets[0].RasterXSize
    if any((ds.RasterYSize, ds.RasterXSize) != shape for ds in datasets):
        raise ValueError('ISCE2 adapter requires aligned equal-size SLCs')
    grid = prior['grid'] if prior is not None else grid
    if grid is None:
        grid = build_grid(shape, shape, params['window'], params['search'], params['stride'], gross_values(cfg), params.get('first_center'))
    gross = prior['integer_px'].copy() if prior is not None else np.broadcast_to(
        np.array(gross_values(cfg), 'int32'), (*grid.shape, 2)).copy()
    eligible = prior['valid'].copy() if prior is not None else np.ones(grid.shape, bool)
    eligible &= valid_windows(*datasets, grid, gross) if valid_input is None else valid_input
    eligible &= (grid.starts_line[:, None] >= grid.search[0]) & (grid.starts_sample[None, :] >= grid.search[1])
    result = _empty_match(grid, eligible, 'cpu')
    result['gross'][:] = gross
    varying = cfg['gross_offsets']['mode'] == 'velocity' or np.unique(gross.reshape(-1, 2), axis=0).shape[0] > 1
    if varying and prior is not None:
        result['gross'][~prior['valid']] = np.nan
    images = []
    for label, source in zip(('reference', 'secondary'), (reference, secondary)):
        filename = output/label
        ds = gdal.Translate(str(filename)+'.vrt', str(source), format='VRT')
        ds = None
        images.append(str(filename.resolve()))
    # Group constant priors, retaining original chip coordinates and reading
    # the complete MERGED images across every worker/scheduling boundary.
    y, x = np.meshgrid(grid.starts_line, grid.starts_sample, indexing='ij')
    runnable = ((y >= grid.search[0]) & (x >= grid.search[1]) &
                (y+grid.window[0] <= shape[0]) & (x+grid.window[1] <= shape[1]))
    for axis, start in enumerate((y, x)):
        low = start+gross[..., axis].astype('int64')-grid.search[axis]
        runnable &= (low >= 0) & (low+grid.window[axis]+2*grid.search[axis] <= shape[axis])
    if varying and prior is not None:
        runnable &= prior['valid']
    if not eligible.any():
        runnable[:] = False
    # Like DenseAmpcor, static-prior stripes can include zero-fill/masked chips;
    # preserve their raw results, then apply the complete-support mask. Starting
    # one interpreter per tiny valid rectangle would dominate real-data runs.
    labels = np.zeros(grid.shape, 'int32')
    values = np.unique(gross[runnable], axis=0)
    for index, value in enumerate(values, 1):
        labels[runnable & np.all(gross == value, axis=-1)] = index
    tasks = burst_rectangles(labels, runnable)
    workers = min(cpu_worker_budget(), len(tasks))
    # Share a full-width static-prior field across row stripes as DenseAmpcor
    # does, without inheriting its automatic grid or fork state.
    if tasks and len(tasks) < cpu_worker_budget():
        split = []
        budget = cpu_worker_budget()
        for label, y0, y1, x0, x1 in tasks:
            for a, b in zip(np.linspace(y0, y1, min(budget, y1-y0)+1, dtype=int)[:-1],
                            np.linspace(y0, y1, min(budget, y1-y0)+1, dtype=int)[1:]):
                split.append((label, int(a), int(b), x0, x1))
        tasks = split
        workers = min(budget, len(tasks))
    started = time.perf_counter()

    def run(task):
        index, (label, y0, y1, x0, x1) = task
        directory = output/f'segment-{index:05d}'
        directory.mkdir(exist_ok=True)
        request = {'images': images, 'shape': shape, 'matching': params, 'gross': values[label-1].tolist(),
                   'rows': grid.starts_line[y0:y1].tolist(), 'cols': grid.starts_sample[x0:x1].tolist(),
                   'result': str((directory/'result.npz').resolve())}
        path = directory/'request.json'
        path.write_text(json.dumps(request))
        invoke(runtime(cfg), [path], directory/'native.log')
        with np.load(request['result']) as saved:
            return {key: saved[key] for key in saved.files}

    if workers:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for saved in pool.map(run, enumerate(tasks)):
                rows = np.searchsorted(grid.starts_line, saved['rows'])
                cols = np.searchsorted(grid.starts_sample, saved['cols'])
                if (np.any(rows >= grid.shape[0]) or np.any(cols >= grid.shape[1]) or
                        not np.array_equal(grid.starts_line[rows], saved['rows']) or
                        not np.array_equal(grid.starts_sample[cols], saved['cols'])):
                    raise RuntimeError('ISCE2 returned observations off the requested chip grid')
                take = (rows, cols)
                for key in ('total', 'snr', 'covariance'):
                    result[key][take] = saved[key]
    result['residual'] = result['total']-result['gross']
    result['finite'] = np.isfinite(result['total']).all(axis=-1) & np.isfinite(result['snr'])
    result['search_boundary'] = np.any(abs(result['residual']) >= np.array(grid.search)-1/params['oversampling'], axis=-1)
    result['valid'] = eligible & result['finite'] & ~result['search_boundary'] & (result['snr'] >= params['snr_threshold'])
    result['elapsed_seconds'] = time.perf_counter()-started
    result['effective_parameters'] = {'engine': native, 'imageDataType1': 'mag', 'imageDataType2': 'mag',
        'workers': workers, 'segments': len(tasks), 'raw_oversampling': 2,
        'zoomWindowSize': params['zoom'], 'oversamplingFactor': params['oversampling'],
        'native_threshold_snr': 1.e-6, 'native_threshold_covariance': 1000.,
        'acrossLooks': 1, 'downLooks': 1, 'winsizeFilt': 1, 'oversamplingFactorFilt': 64,
        'correlation_peak': 'unavailable from ISCE2; retained as NaN, not used for validity',
        'input_strategy': 'merged SLCs; no burst input substitution',
        'coordinate_adapter': 'Fortran firstSample = zero-based chip start + 1 - search; integer reported centers mapped back to exact geometric centers'}
    if prior is not None and 'metadata' in prior:
        result['prior'] = prior
    _write_native_layout(output, result)
    return result

"""Fixed-support unspliced overlap observations using production range-2x matching."""
from pathlib import Path
import json
import logging
import shutil

import numpy as np
from osgeo import gdal
from pyproj import Geod, Transformer
import scipy.fft

from .ampcor import _native_executor
from .fftw_cache import pinned, pinned_wisdom, wisdom_key
from .geocode import sample_geometry
from .matching_grid import MatchingGrid, invalid_samples
from .products import identity, write_json
from .range_oversampling import ROW_BLOCK, native_parameters, range_worker, resample_range
from .stack_pairs import matching_template


def prepare_strip(source, output, bounds, workers=1):
    """Crop AFTER interpolating complete original rows, with the production kernel.

    bounds are original pixel-cell bounds [y0,y1,x0,x1]. Output sample zero
    represents (y0,x0-.25). No new Fourier boundary is introduced near a chip.
    """
    y0, y1, x0, x1 = map(int, bounds)
    ds = gdal.Open(str(source))
    if not (0 <= y0 < y1 <= ds.RasterYSize and 0 <= x0 < x1 <= ds.RasterXSize):
        raise ValueError('Prepared strip exceeds original raster')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    target = gdal.GetDriverByName('GTiff').Create(str(output), 2*(x1-x0), y1-y0, 1,
        gdal.GDT_CFloat32, options=['TILED=NO', 'COMPRESS=NONE', 'BIGTIFF=IF_SAFER'])
    with scipy.fft.set_workers(workers):
        for row in range(y0, y1, ROW_BLOCK):
            count = min(ROW_BLOCK, y1-row)
            full = resample_range(ds.ReadAsArray(0, row, ds.RasterXSize, count))
            target.GetRasterBand(1).WriteArray(full[:, 2*x0:2*x1], 0, row-y0)
    target = ds = None


def support_arrays(path, grid):
    """Separate reference-chip and secondary-search support; zeros follow production."""
    ds = gdal.Open(str(path))
    bad = invalid_samples(ds)
    summed = np.pad(bad.astype('uint32'), ((1, 0), (1, 0)))
    np.cumsum(summed, axis=0, out=summed)
    np.cumsum(summed, axis=1, out=summed)
    result = []
    for halo in ((0, 0), grid.search):
        y = grid.starts_line[:, None]-halo[0]
        x = grid.starts_sample[None, :]-halo[1]
        yy, xx = y+grid.window[0]+2*halo[0], x+grid.window[1]+2*halo[1]
        good = (y >= 0) & (x >= 0) & (yy <= ds.RasterYSize) & (xx <= ds.RasterXSize)
        y, yy = np.clip(y, 0, ds.RasterYSize), np.clip(yy, 0, ds.RasterYSize)
        x, xx = np.clip(x, 0, ds.RasterXSize), np.clip(xx, 0, ds.RasterXSize)
        count = (summed[yy, xx].astype('int64')-summed[y, xx]-summed[yy, x]+summed[y, x])
        result.append(good & (count == 0))
    return result


def overlap_grid(cfg, frame):
    """Consecutive subset of production centers with nominal dual-swath support."""
    full = matching_template(cfg, frame)
    sources = [e for e in frame['sources'] if e['burst_id'].endswith(('_iw1', '_iw2'))]
    masks = {}
    for e in sources:
        dy, dx = e['offset']
        a, b, c, d = e['valid_window']
        masks[e['burst_id']] = ((full.starts_line[:, None]-dy-full.search[0] >= a) &
            (full.starts_line[:, None]-dy+full.window[0]+full.search[0] <= b) &
            (full.starts_sample[None, :]-dx-full.search[1] >= c) &
            (full.starts_sample[None, :]-dx+full.window[1]+full.search[1] <= d))
    union = {s: np.logical_or.reduce([masks[e['burst_id']] for e in sources if e['burst_id'].endswith(s)])
             for s in ('iw1', 'iw2')}
    dual = union['iw1'] & union['iw2']
    rows, cols = np.flatnonzero(dual.any(axis=1)), np.flatnonzero(dual.any(axis=0))
    if not rows.size or not cols.size:
        raise ValueError('No complete dual-swath chip/search footprint on production centers')
    rows, cols = np.arange(rows[0], rows[-1]+1), np.arange(cols[0], cols[-1]+1)
    grid = MatchingGrid(full.starts_line[rows], full.starts_sample[cols], full.window, full.search, full.stride)
    return grid, sources, {k: v[np.ix_(rows, cols)] & dual[np.ix_(rows, cols)] for k, v in masks.items()}


def run_job(job):
    key, args, kwargs = job
    try:
        return key, range_worker((args, kwargs)), None
    except Exception as exc:
        return key, None, f'{type(exc).__name__}: {exc}'


def experiment_wisdom(experiment):
    native = native_parameters(experiment.cfg['matching'])
    digest = identity(wisdom_key(native))
    target = experiment.output/'.cache/fftw'
    source = experiment.stack/'.cache/fftw'/f'{digest}.wisdom'
    if source.exists() and not (target/source.name).exists():
        target.mkdir(parents=True, exist_ok=True)
        experiment.inputs.file(source)
        shutil.copyfile(source, target/source.name)
        metadata = source.with_suffix('.json')
        if metadata.exists():
            shutil.copyfile(metadata, target/metadata.name)
    with pinned(target):
        return pinned_wisdom(native)


def measure_link(experiment, link, prepared_cache, wisdom):
    """Match one link, retaining each native job so interrupted links can resume."""
    from .swath_overlap_support import select_views, attach_results, burst_interior_mask
    from glacier_offsets.swath_experiment_io import read_json, save_npz, load_npz, cohorts
    out = experiment.output/'measurements'/link['pair']
    days = [link['earlier'], link['later']]
    acquisitions = [experiment.verify_acquisition(day) for day in days]
    grid, sources, masks = overlap_grid(experiment.cfg, experiment.frame)
    specification = {'experiment': experiment.key, 'link': link,
                     'alignments': [a['alignment_identity'] for a in acquisitions],
                     'wisdom_sha256': wisdom['sha256'],
                     'implementation': experiment.implementation('swath_observations.py', 'swath_overlap_support.py', 'range_oversampling.py',
                        'ampcor.py', 'fftw_cache.py', 'matching_grid.py', 'swath_experiment_io.py')}
    if experiment.current(out/'report.json', specification):
        return read_json(out/'report.json')
    out.mkdir(parents=True, exist_ok=True)
    yy, xx = grid.centers
    records, jobs, products = {}, [], []
    for e in sources:
        key = e['burst_id']
        selected = masks[key]
        if not selected.any() or any(key not in a['slcs'] for a in acquisitions):
            continue
        rows, cols = np.flatnonzero(selected.any(axis=1)), np.flatnonzero(selected.any(axis=0))
        rows, cols = np.arange(rows[0], rows[-1]+1), np.arange(cols[0], cols[-1]+1)
        dy, dx = e['offset']
        local = MatchingGrid(grid.starts_line[rows]-dy, grid.starts_sample[cols]-dx,
                             grid.window, grid.search, grid.stride)
        bounds = [int(local.starts_line[0]-local.search[0]),
                  int(local.starts_line[-1]+local.window[0]+local.search[0]),
                  int(local.starts_sample[0]-local.search[1]),
                  int(local.starts_sample[-1]+local.window[1]+local.search[1])]
        paths, support = [], []
        for day, acq in zip(days, acquisitions):
            source = acq['slcs'][key]
            digest = experiment.inputs.file(source, acq['alignment_products'][str(Path(source).resolve())])
            cache_key = identity({'source': digest, 'bounds': bounds, 'window': grid.window,
                                  'search': grid.search, 'stride': grid.stride, 'experiment': experiment.key})
            directory = experiment.output/'.cache/strips'/cache_key
            directory.mkdir(parents=True, exist_ok=True)
            support_file = directory/'support.npz'
            if support_file.exists():
                valid = load_npz(support_file)
            else:
                chip, search = support_arrays(source, local)
                save_npz(support_file, chip=chip, search=search)
                valid = {'chip': chip, 'search': search}
            support.append(valid)
            prepared = directory/'range2x.tif'
            if not prepared.exists():
                temporary = directory/'preparing.tif'
                prepare_strip(source, temporary, bounds, experiment.workers)
                temporary.replace(prepared)
            prepared_cache.setdefault(day, set()).add(prepared)
            paths.append(str(prepared))
        accepted = support[0]['chip'] & support[1]['search'] & selected[np.ix_(rows, cols)]
        transformer = Transformer.from_crs(e['coreg']['geometry_crs'], 4326, always_xy=True)
        geometries = []
        line, sample = local.centers
        hy, hx = [(v-1)/2 for v in grid.window]
        for a, b in ((0, 0), (-hy, -hx), (-hy, hx), (hy, -hx), (hy, hx)):
            gx = sample_geometry(e['coreg']['geometry']['x'], line+a, sample+b)
            gy = sample_geometry(e['coreg']['geometry']['y'], line+a, sample+b)
            lon, lat = transformer.transform(gx, gy)
            height = sample_geometry(e['coreg']['geometry']['z'], line+a, sample+b)
            geometries.append(np.stack([lon, lat, height], axis=-1))
        geometry = np.stack(geometries, axis=-2)
        records[key] = {'entry': e, 'rows': rows, 'cols': cols, 'support': accepted,
                        'geometry': geometry[..., :2], 'llh': geometry, 'results': {}, 'bounds': bounds}
        job_spec = dict(specification, burst=key)
        job_report = out/key/'report.json'
        job_result = out/key/'observations.npz'
        if experiment.current(job_report, job_spec):
            records[key]['results'][0] = load_npz(job_result)
            continue
        cropped = MatchingGrid(local.starts_line-bounds[0], local.starts_sample-bounds[2],
                               grid.window, grid.search, grid.stride)
        args = (*paths, out/key/'native', experiment.cfg['matching'], experiment.cfg, 'cpu')
        jobs.append((key, args, {'_grid': cropped, '_valid_input': accepted}))
    selection, numbers = select_views(sources, records, grid.shape)
    errors = []
    if jobs:
        with _native_executor(min(experiment.workers, len(jobs)), single_threaded=True, wisdom=wisdom) as pool:
            for key, result, error in pool.imap_unordered(run_job, jobs):
                if error:
                    errors.append({'burst': key, 'error': error})
                    write_json(out/key/'report.json', {'status': 'failed', 'error': error})
                    continue
                record = records[key]
                np.testing.assert_array_equal(result['input_valid'], record['support'])
                fields = ('total', 'gross', 'residual', 'valid', 'input_valid', 'finite',
                          'search_boundary', 'snr', 'covariance', 'correlation_peak')
                arrays = {k: result[k] for k in fields}
                arrays.update(rows=record['rows'], cols=record['cols'], geometry_llh=record['llh'],
                              effective_json=np.array(json.dumps(result['effective_parameters'])))
                result_file = out/key/'observations.npz'
                save_npz(result_file, **arrays)
                records[key]['results'][0] = arrays
                experiment.finish(out/key/'report.json', dict(specification, burst=key),
                    {'status': 'complete', 'effective_parameters': result['effective_parameters']}, [result_file])
                logging.info('%s %s: %d native-valid', link['pair'], key, int(result['valid'].sum()))
    views = attach_results(selection, records, numbers, 0)
    dual = selection['iw1']['support'] & selection['iw2']['support']
    interior = burst_interior_mask(sources, yy, xx, {s: v['owner'] for s, v in selection.items()},
                                   dual, grid.window, grid.search)
    a, b = selection['iw1']['geometry'], selection['iw2']['geometry']
    _, _, distance = Geod(ellps='WGS84').inv(a[..., 0], a[..., 1], b[..., 0], b[..., 1])
    geometry_eligible = np.isfinite(distance).all(axis=-1) & (distance.max(axis=-1) <= .1)
    arrays = {'center_line': yy, 'center_sample': xx, 'dual_support': dual,
              'burst_interior': interior, 'geometry_eligible': geometry_eligible,
              'geometry_distance_m': distance, 'source_ids': np.array([numbers[i] for i in sorted(numbers)]),
              'dual_valid': dual & views['iw1']['valid'] & views['iw2']['valid']}
    for swath, view in views.items():
        for field in ('owner', 'support', 'valid', 'finite', 'search_boundary', 'native_completed', 'total', 'peak', 'snr'):
            arrays[f'{swath}_{field}'] = view[field]
        # Additional fields are published at the frozen owners, including failed jobs.
        for name, width in (('gross', 2), ('residual', 2), ('covariance', 3), ('geometry_llh', (5, 3))):
            shape = grid.shape + ((width,) if isinstance(width, int) else width)
            value = np.full(shape, np.nan)
            for number, key in numbers.items():
                if key not in records:
                    continue
                record = records[key]
                source = record['llh'] if name == 'geometry_llh' else record['results'].get(0, {}).get(name)
                if source is None:
                    continue
                section = np.ix_(record['rows'], record['cols'])
                take = view['owner'][section] == number
                block = value[section]
                block[take] = source[take]
                value[section] = block
            arrays[f'{swath}_{name}'] = value
    observation_file = out/'observations.npz'
    save_npz(observation_file, **arrays)
    products.append(observation_file)
    products.extend(out.glob('*/observations.npz'))
    report = {'status': 'incomplete' if errors else 'complete', 'errors': errors,
              'counts': {name: int(mask.sum()) for name, mask in cohorts(arrays).items()},
              'dual_support_n': int(dual.sum()), 'source_ids': numbers,
              'geometry_gate_m': .1, 'geometry_max_m': float(np.max(distance[dual])) if dual.any() else None,
              'acquisition_burst_times': {d: a.get('burst_times', {}) for d, a in zip(days, acquisitions)},
              'verified_inputs': dict(experiment.inputs.products), 'wisdom': wisdom,
              'selection': 'latest fully supported source, fixed before matching; no failure fallback'}
    return experiment.finish(out/'report.json', specification, report, products)


def measure(experiment, pilot_only=False):
    from glacier_offsets.swath_experiment_io import read_json
    experiment.verify_geometry()
    wisdom = experiment_wisdom(experiment)
    cache, records = {}, {}
    pilot_days = set(experiment.dates[:3])
    pilot = [x for x in experiment.links if {x['earlier'], x['later']} <= pilot_days]
    links = pilot if pilot_only else pilot + [x for x in experiment.links if x not in pilot]
    for i, link in enumerate(links):
        logging.info('Overlap link %d/%d: %s', i+1, len(links), link['pair'])
        try:
            records[link['pair']] = measure_link(experiment, link, cache, wisdom)
        except (ValueError, OSError, KeyError) as exc:
            records[link['pair']] = {'status': 'unavailable', 'error': str(exc)}
            logging.error('%s unavailable: %s', link['pair'], exc)
        write_json(experiment.output/'measurement_status.json', records)
        if i == len(pilot)-1:
            passed = all(records[x['pair']]['status'] == 'complete' and
                         records[x['pair']].get('counts', {}).get('native', 0) > 0 for x in pilot)
            write_json(experiment.output/'pilot.json', {'status': 'complete' if passed else 'incomplete',
                'links': [x['pair'] for x in pilot], 'passed': passed,
                'gate': 'all native jobs complete; each link has geometry-eligible native-valid burst interiors'})
            if not passed:
                logging.error('Pilot is incomplete; full measurement expansion withheld')
                break
        future = {x[k] for x in links[i+1:] for k in ('earlier', 'later')}
        for day in list(cache):
            if day not in future:
                for path in cache.pop(day):
                    path.unlink(missing_ok=True)
    experiment.inputs.unchanged()
    return {'status': 'complete' if len(records) == len(links) and all(r['status'] == 'complete' for r in records.values())
            else 'incomplete', 'links': {k: {'status': v['status'], 'counts': v.get('counts'), 'error': v.get('error')}
                                        for k, v in records.items()}, 'pilot_only': pilot_only}

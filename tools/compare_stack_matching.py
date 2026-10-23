#!/usr/bin/env python3
"""Isolated full-pair comparison of merged/unspliced and complex/magnitude inputs.

Uses native ISCE3 Ampcor on the saved global matching grid and fixed ownership.
Retains extra overlap observations from original bursts as diagnostic controls.
No COMPASS, changed ownership, smoothing, gap filling or production writes.
"""
import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.ampcor import (_empty_match, _native_executor, _native_worker,
                                   _write_native_layout, match)
from glacier_offsets.config import load_defaults
from glacier_offsets.geocode import geocode
from glacier_offsets.matching_grid import MatchingGrid, valid_windows
from glacier_offsets.products import ChecksumCache, identity, write_json
from glacier_offsets.raster_io import require_native_mmap
from glacier_offsets.stack_pairs import matching_template
from glacier_offsets.stack_storage import assemble_date, read_manifest, verified

FIELDS = ('gross', 'residual', 'total', 'snr', 'correlation_peak', 'covariance',
          'input_valid', 'finite', 'search_boundary', 'valid')
VARIANTS = {'merged_magnitude': ('merged', 0), 'unspliced_complex': ('unspliced', 1),
            'unspliced_magnitude': ('unspliced', 0)}


def native_job(job):
    index, args, kwargs = job
    return index, _native_worker((args, kwargs))


def save_native(path, result, fingerprint):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **{k: result[k] for k in FIELDS},
                            effective_json=json.dumps(result['effective_parameters']),
                            elapsed_seconds=result['elapsed_seconds'], fingerprint=fingerprint)
    temporary.replace(path)


def load_native(path, fingerprint):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['fingerprint']) != fingerprint:
            raise ValueError(f'Native checkpoint settings changed: {path}')
        return {k: saved[k] for k in FIELDS} | {
            'effective_parameters': json.loads(str(saved['effective_json'])),
            'elapsed_seconds': float(saved['elapsed_seconds'])}


def source_jobs(manifest, dates, grid, original_valid):
    """Entire supported burst footprints, including both views of overlaps."""
    jobs = []
    for index, source in enumerate(manifest['assembly']['sources'], 1):
        key = source['burst_id']
        if source['interval_seconds'] is None or any(key not in date['slcs'] for date in dates):
            continue
        paths = [date['slcs'][key] for date in dates]
        datasets = [gdal.Open(path) for path in paths]
        for ds in datasets:
            require_native_mmap(ds)
        dy, dx = source['offset']
        y0, y1, x0, x1 = source['valid_window']
        rows = np.flatnonzero((grid.starts_line-dy-grid.search[0] >= y0) &
                             (grid.starts_line-dy+grid.window[0]+grid.search[0] <= y1))
        cols = np.flatnonzero((grid.starts_sample-dx-grid.search[1] >= x0) &
                             (grid.starts_sample-dx+grid.window[1]+grid.search[1] <= x1))
        if not rows.size or not cols.size:
            continue
        local = MatchingGrid(grid.starts_line[rows]-dy, grid.starts_sample[cols]-dx,
                             grid.window, grid.search, grid.stride)
        # Keep the old full-chip support mask in every comparison. Nothing is
        # recovered by substituting a different source or ignoring a real hole.
        support = valid_windows(*datasets, local, (0, 0)) & original_valid[np.ix_(rows, cols)]
        jobs.append({'index': index, 'burst_id': key, 'paths': paths, 'rows': rows, 'cols': cols,
                     'grid': local, 'support': support})
        logging.info('Burst %s: %d x %d native windows, %d supported', key, *local.shape, support.sum())
    return jobs


def unspliced_match(cfg, manifest, directory, prior, *, jobs, owner, workers, fingerprint):
    if prior is not None:
        raise ValueError('This controlled comparison requires zero gross offsets')
    grid = matching_template(cfg, {'coreg': manifest['coreg']})
    result = _empty_match(grid, np.zeros(grid.shape, bool), 'cpu')
    result['gross'].fill(0)  # Known zero prior, even for unobserved/invalid slots.
    pending = []
    mapping = {job['index']: job for job in jobs}
    records = []
    directory.mkdir(parents=True, exist_ok=True)

    def collect(index, native):
        job = mapping[index]
        section = np.ix_(job['rows'], job['cols'])
        take = owner[section] == index
        for key in FIELDS:
            block = result[key][section]
            block[take] = native[key][take]
            result[key][section] = block
        records.append(native['effective_parameters'] | {'burst_id': job['burst_id'],
                       'native_input_paths': job['paths'], 'assigned_centers': int(take.sum())})
        result['elapsed_seconds'] += native['elapsed_seconds']

    for job in jobs:
        index = job['index']
        path = directory/'native'/job['burst_id']/'result.npz'
        cached = load_native(path, fingerprint)
        if cached is not None:
            collect(index, cached)
            continue
        args = (*job['paths'], path.parent, cfg['matching'], cfg, 'cpu')
        pending.append((index, args, {'_grid': job['grid'], '_valid_input': job['support']}))
    started = time.perf_counter()
    if pending:
        with _native_executor(min(workers, len(pending)), single_threaded=True) as pool:
            for index, native in pool.imap_unordered(native_job, pending, chunksize=1):
                job = mapping[index]
                save_native(directory/'native'/job['burst_id']/'result.npz', native, fingerprint)
                collect(index, native)
                logging.info('Completed original burst %s', job['burst_id'])
    result['effective_parameters'] = {'strategy': 'validation-unspliced-fixed-center-owner-v1',
        'workers': workers, 'wall_seconds': time.perf_counter()-started, 'segments': records,
        'support_policy': 'original pair chip/search mask intersect both native SLC supports and reference annotation rectangle',
        'overlap_policy': 'retain all burst observations; publish only the fixed owning burst; no averaging or substitution'}
    _write_native_layout(directory/'native', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.workers < 1:
        raise ValueError('workers must be positive')
    pair, out = args.pair.resolve(), args.output.resolve()
    stack = pair.parent.parent
    if out == stack or stack in out.parents or out in stack.parents:
        raise ValueError('Choose a new validation directory outside the production stack')
    if out.exists() and not args.resume:
        raise ValueError('Output exists; use --resume for a matching experiment')
    gdal.UseExceptions()
    original = json.loads((pair/'manifest.json').read_text())
    if original['matching']['gross_offsets']['mode'] != 'zero':
        raise ValueError('This experiment requires zero gross offsets')
    if original['matching']['matching']['deramp'] != 1:
        raise ValueError('Expected a complex-deramped saved baseline (deramp=1)')
    days = original['stack_pair']['acquisition_dates']
    dates = [read_manifest(stack/'acquisitions'/day) for day in days]
    frame = read_manifest(stack/'geometry')
    inputs = {pair/'manifest.json', pair/'radar/offsets.h5', pair/'geo/center_geometry.h5', stack/'geometry/manifest.json',
              Path(original['coreg']['source_index'])}
    inputs.update(Path(p) for date in dates for p in date['slcs'].values())
    inputs.update(stack/'acquisitions'/day/'manifest.json' for day in days)
    before = {str(path): [path.stat().st_size, path.stat().st_mtime_ns] for path in inputs}
    specification = {'pair': str(pair), 'match_identity': original['match_identity'],
        'geometry_identity': original['stack_pair']['geometry_identity'], 'variants': VARIANTS,
        'settings': original['matching']['matching'], 'inputs': before,
        'implementation': 'full-pair-amplitude-comparison-v1'}
    fingerprint = identity(specification)
    previous = out/'experiment.json'
    if previous.is_file() and json.loads(previous.read_text())['identity'] != fingerprint:
        raise ValueError('Experiment inputs/settings changed; choose another output directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(previous, {'identity': fingerprint, 'specification': specification, 'status': 'running'})
    cfg = load_defaults()
    cfg['output'] = {'directory': str(out/'input_assemblies')}
    cfg['inputs'].update(burst_id='stack', polarization=original['coreg_specification']['polarization'])
    cfg['matching'] = deepcopy(original['matching']['matching'])
    cfg['gross_offsets'] = deepcopy(original['matching']['gross_offsets'])
    cfg['geocoding'] = {k: original['geocoding'][k] for k in ('epsg', 'spacing_m', 'resampling')}
    cfg['geocoding']['mask'] = original['geocoding']['source_mask']
    checksums = ChecksumCache()
    for date in dates:
        products = date.get('alignment_products', date['products'])
        for name in date['slcs'].values():
            if checksums(name) != products[str(Path(name).resolve())]:
                raise ValueError(f'Aligned source checksum mismatch: {name}')
    assemblies = []
    for day, date in zip(days, dates):
        cached = read_manifest(Path(cfg['output']['directory'])/'acquisitions'/day)
        if cached and verified(cached, 'assembly_identity', cached.get('assembly_identity'), checksums=checksums):
            assemblies.append(cached)
        else:
            logging.info('Assembling isolated input %s', day)
            assemblies.append(assemble_date(cfg, SimpleNamespace(date=day), frame, date, force=True, checksums=checksums))
    with h5py.File(pair/'radar/offsets.h5') as h:
        original_valid, owner = h['input_valid'][:], h['source_burst_index'][:]
    grid = matching_template(cfg, {'coreg': original['coreg']})
    jobs = source_jobs(original, dates, grid, original_valid)
    write_json(out/'burst_grids.json', [{'burst_id': job['burst_id'], 'index': job['index'],
               'global_rows': job['rows'].tolist(), 'global_columns': job['cols'].tolist(),
               'input_paths': job['paths'], 'supported_windows': int(job['support'].sum())} for job in jobs])
    geometry_names = ('longitude_deg', 'latitude_deg', 'height_m', 'azimuth_ground_spacing_m_per_slc_line', 'geometry_valid')
    with h5py.File(pair/'geo/center_geometry.h5') as h:
        geometry = tuple(h[key][:] for key in geometry_names)
    variants = {}
    import os
    os.environ['OMP_NUM_THREADS'] = str(args.workers)
    for name, (kind, deramp) in VARIANTS.items():
        directory = out/name
        directory.mkdir(exist_ok=True)
        variant = deepcopy(cfg)
        variant['matching']['deramp'] = deramp
        manifest = read_manifest(directory)
        if not manifest:
            manifest = deepcopy(original)
            for key in ('match_identity', 'geocode_identity', 'diagnostics_identity', 'matching', 'matching_summary', 'geocoding'):
                manifest.pop(key, None)
            manifest['stages'] = {'coregister': 'retained aligned source bursts; no COMPASS'}
            manifest['coreg']['slcs'] = dict(zip(('reference', 'secondary'), (p['slc'] for p in assemblies)))
            manifest['coreg_identity'] = identity({'experiment': fingerprint, 'variant': name})
            manifest['validation'] = {'experiment_identity': fingerprint, 'source_pair': str(pair),
                'input_mode': kind, 'deramp': deramp, 'fixed_grid_and_ownership': True,
                'geometry_cache': str(pair/'geo/center_geometry.h5'), 'no_filtering': True}
        logging.info('Matching full pair: %s', name)
        if kind == 'unspliced':
            def experimental_match(c, m, d, prior):
                return unspliced_match(c, m, d, prior, jobs=jobs, owner=owner,
                                       workers=args.workers, fingerprint=identity({'experiment': fingerprint, 'variant': name}))
            with patch('glacier_offsets.stack_pairs.match_prepared', side_effect=experimental_match):
                match(variant, directory, manifest)
        else:
            match(variant, directory, manifest)
        logging.info('Geocoding full pair: %s', name)
        # Reuse the saved center geometry for all variants, isolating the matcher.
        # The earlier map audit established exact reproduction of the baseline.
        with patch('glacier_offsets.geocode.center_geometry', return_value=geometry):
            geocode(variant, directory, manifest)
        variants[name] = {'directory': str(directory), 'matching_summary': manifest['matching_summary'],
                          'geocoding': manifest['geocoding']}
        write_json(out/'progress.json', variants)
    unchanged = all([Path(path).stat().st_size, Path(path).stat().st_mtime_ns] == saved for path, saved in before.items())
    if not unchanged:
        raise ValueError('Production input sizes/mtimes changed during the experiment')
    write_json(previous, {'identity': fingerprint, 'specification': specification, 'status': 'complete',
                         'variants': variants, 'production_sizes_mtimes_unchanged': unchanged,
                         'limitations': 'Native ISCE3 factorial comparison, not an end-to-end ISCE2 test or independent displacement truth.'})
    logging.info('Full-pair comparison completed: %s', out)


if __name__ == '__main__':
    main()

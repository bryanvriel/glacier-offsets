"""Grounded, slow control boxes for velocity-pixels controls-mode noise, from a package alone.

This is for stacks without a reviewed anchor set. The rules match the Evans IW3
anchor export (tools/export_evans_anchor_regions.py), minus its anchor-review
acceptance. A square cell of --cell metres is kept when:
- every unique source in it has CATS ocean mask and tidal response <= 0.02;
- at least --min-supported sources have >= 20 peak-supported pairs;
- the median EN speed of those sources is <= 0.05 m/day, and at least 90% of
  them are <= 0.2 m/day.

Speeds are per-source medians of east/north displacement rates over pairs with
horizontal validity and correlation peak >= 0.1. The output is a regions JSON
with role: control boxes only. The controls-mode estimator uses every source
inside each box. NISAR uses geocoded map locations, exact per-cell intervals and
the fitting collection's frame-overlap selection. These cells are spatially
correlated; their count is not an independent-sample count. Run in ice with
PYTHONPATH=src.
"""
import argparse
import hashlib
import json
import logging
from time import perf_counter
from pathlib import Path

import h5py
import numpy as np

from glacier_offsets.displacement import is_nisar
from glacier_offsets.velocity_io import digest, revision, fingerprint


def package_digest(path):
    size = Path(path).stat().st_size
    logging.info('Computing package SHA-256 (%0.2f GB); sequential file read', size/1e9)
    value = hashlib.sha256()
    count = 0
    started = last = perf_counter()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
            count += len(chunk)
            now = perf_counter()
            if now-last >= 5:
                logging.info('Package checksum %.1f%%, %.2f/%.2f GB (%.1fs)',
                             100*count/max(size, 1), count/1e9, size/1e9, now-started)
                last = now
    logging.info('Package checksum complete (%.1fs)', perf_counter()-started)
    return value.hexdigest()


def nisar_samples(h, cell=3000., min_supported=10):
    """Read horizontal data once per HDF5 chunk, only in plausible controls.

    This proposal builder checks package metadata and fingerprints the whole file
    at publication. Full array verification is optional; fitting still validates
    its inputs independently.
    """
    from glacier_offsets.nisar_collection import frame_candidates
    if h.attrs.get('package_status') != 'complete':
        raise ValueError('Incomplete NISAR package')
    spec = json.loads(h['provenance_json'][()].decode())
    if h.attrs['package_identity'] != fingerprint(spec):
        raise ValueError('Package provenance fingerprint mismatch')
    if h.attrs.get('source', 'raw') != 'raw':
        raise ValueError('Sentinel-1 swath corrections are not supported on NISAR packages')
    if any('solid_earth' in k.lower() or 'body_tide' in k.lower() for k in (*h.attrs, *h.keys())):
        raise ValueError('Use a package before fitting-time solid-Earth correction')
    if 'tidal_correction_provenance_json' not in h or 'horizontal_conversion_provenance_json' not in h:
        raise ValueError('Use a tide-corrected horizontal displacement package')
    ids = h['pair_id'][:].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate pair IDs')
    members = [dict(pair_id=gid, frame=int(h['frame_number'][t]),
                    acquisition_pair=tuple(int(h[k][t]) for k in
                        ('track_number', 'reference_absolute_orbit', 'secondary_absolute_orbit')))
               for t, gid in enumerate(ids)]
    keys = [(r['acquisition_pair'], r['frame']) for r in members]
    if len(set(keys)) != len(keys):
        raise ValueError('Conflicting reprocessed copies of the same acquisition pair/frame')
    priority = sorted(range(len(members)), key=lambda t: (members[t]['frame'], members[t]['pair_id']))
    shape = (len(h['y']), len(h['x']))
    fields = ('valid_mask', 'horizontal_valid_mask', 'tidal_correction_available', 'correlation_peak',
              'east_displacement_m', 'north_displacement_m', 'earlier_observation_time', 'later_observation_time')
    for key in fields:
        if h[key].shape != (len(ids), *shape):
            raise ValueError('Invalid dimensions: '+key)
    for key in ('earlier_observation_time', 'later_observation_time'):
        if h[key].attrs.get('units') != 'seconds since 1970-01-01 00:00:00':
            raise ValueError('Observation endpoints must use UTC epoch seconds')
    chunks = h['valid_mask'].chunks
    dy, dx = chunks[-2:] if chunks else (256, 256)
    sections = [(slice(r, min(r+dy, shape[0])), slice(c, min(c+dx, shape[1])))
                for r in range(0, shape[0], dy) for c in range(0, shape[1], dx)]
    present = np.zeros(shape, bool)
    started = last = perf_counter()
    logging.info('Scanning NISAR coverage: %d products, %d chunk-aligned tiles', len(ids), len(sections))
    for i, sec in enumerate(sections):
        present[sec] = np.any(h['valid_mask'][(slice(None), *sec)] == 1, axis=0)
        if i == 0 or i+1 == len(sections) or perf_counter()-last >= 5:
            logging.info('Coverage tiles %d/%d (%.1fs)', i+1, len(sections), perf_counter()-started)
            last = perf_counter()
    row, col = np.nonzero(present)
    x, y = h['x'][:][col], h['y'][:][row]
    boxes = np.stack([np.floor(x/cell), np.floor(y/cell)], -1).astype('i8')
    _, group = np.unique(boxes, axis=0, return_inverse=True)
    response = h['tidal_response_factor'][:][row, col]
    ocean = h['tidal_model_ocean_mask'][:][row, col]
    bad = ~np.isfinite(response) | ~np.isfinite(ocean) | (response > .02) | (ocean > .02)
    counts = np.bincount(group)
    grounded = (np.bincount(group, weights=bad, minlength=len(counts)) == 0) & (counts >= min_supported)
    eligible = np.zeros(shape, bool)
    eligible[row, col] = grounded[group]
    speed = np.full(shape, np.nan)
    support = np.zeros(shape, int)
    selected = [sec for sec in sections if eligible[sec].any()]
    # No location can meet the existing 20-observation rule with fewer pairs.
    if len({r['acquisition_pair'] for r in members}) < 20:
        selected = []
    logging.info('Grounded candidates: %d locations; reading %d/%d time-series tiles',
                 eligible.sum(), len(selected), len(sections))
    started = last = perf_counter()
    for i, sec in enumerate(selected):
        take = eligible[sec]
        def read(key):
            return h[key][(slice(None), *sec)][:, take]
        valid = read('valid_mask') == 1
        candidates = frame_candidates(members, priority, lambda t: valid[t])
        good = np.stack([candidates[t] for t in range(len(ids))])
        good &= (read('horizontal_valid_mask') == 1) & (read('tidal_correction_available') == 1)
        good &= read('correlation_peak') >= .1
        dt = (read('later_observation_time')-read('earlier_observation_time'))/86400.
        good &= np.isfinite(dt) & (dt > 0)
        east, north = read('east_displacement_m'), read('north_displacement_m')
        good &= np.isfinite(east) & np.isfinite(north)
        support[sec][take] = good.sum(axis=0)
        usable = good.any(axis=0)
        medians = []
        for component in (east, north):
            rates = np.full(component.shape, np.nan)
            np.divide(component, dt, out=rates, where=good)
            median = np.full(len(usable), np.nan)
            median[usable] = np.nanmedian(rates[:, usable], axis=0)
            medians.append(median)
        speed[sec][take] = np.hypot(*medians)
        if i == 0 or i+1 == len(selected) or perf_counter()-last >= 5:
            logging.info('Control time-series tiles %d/%d (%.1fs)', i+1, len(selected), perf_counter()-started)
            last = perf_counter()
    return row, col, support[row, col], speed[row, col]


def main(a):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    before = revision(Path(a.package))
    if getattr(a, 'verify_package', False):
        logging.info('Fully verifying package arrays (may take time for large files)')
        from glacier_offsets.displacement import validate_nisar
        with h5py.File(a.package) as profile:
            if is_nisar(profile): validate_nisar(a.package)
    with h5py.File(a.package) as h:
        nisar = is_nisar(h)
        required = ('east_displacement_m', 'north_displacement_m', 'tidal_response_factor',
                    'tidal_model_ocean_mask', 'correlation_peak')
        missing = [key for key in required if key not in h]
        if nisar and 'horizontal_valid_mask' not in h: missing.append('horizontal_valid_mask')
        if missing:
            raise ValueError('Use a tide-corrected horizontal displacement package; missing '+', '.join(missing))
        if nisar:
            row, col, support, speed = nisar_samples(h, a.cell, a.min_supported)
        else:
            so = h['source_observation_index'][:]
            unique, first = np.unique(so, return_index=True)
            first = first[unique >= 0]
            row, col = np.unravel_index(first, so.shape)
            dt = (h['later_time'][:]-h['earlier_time'][:])/86400.
            valid = 'horizontal_valid_mask' if 'horizontal_valid_mask' in h else 'valid_mask'
            east = np.full((len(row), len(dt)), np.nan)
            north = np.full_like(east, np.nan)
            support = np.zeros(len(row), int)
            for t in range(len(dt)):
                ok = (h[valid][t][row, col] > 0) & (h['correlation_peak'][t][row, col] >= .1)
                east[ok, t] = h['east_displacement_m'][t][row, col][ok]/dt[t]
                north[ok, t] = h['north_displacement_m'][t][row, col][ok]/dt[t]
                support += ok
            with np.errstate(all='ignore'):
                speed = np.hypot(np.nanmedian(east, 1), np.nanmedian(north, 1))
        x, y = h['x'][:][col], h['y'][:][row]
        response = h['tidal_response_factor'][:][row, col]
        ocean = h['tidal_model_ocean_mask'][:][row, col]
        epsg = int(np.ravel(h.attrs['EPSG'])[0]) if 'EPSG' in h.attrs else 3031
    cx, cy = np.floor(x/a.cell).astype(int), np.floor(y/a.cell).astype(int)
    regions, rejected = [], 0
    boxes, group = np.unique(np.stack([cx, cy], -1), axis=0, return_inverse=True)
    order = np.argsort(group, kind='stable')
    groups = np.split(order, np.cumsum(np.bincount(group))[:-1])
    for (i, j), m in zip(boxes, groups):
        grounded = bool((np.nan_to_num(response[m], nan=1.) <= .02).all() and (np.nan_to_num(ocean[m], nan=1.) <= .02).all())
        sup = m[support[m] >= 20]
        if not grounded or len(sup) < a.min_supported:
            rejected += 1
            continue
        med = float(np.nanmedian(speed[sup]))
        slow = float(np.mean(speed[sup] <= .2))
        if med > .05 or slow < .9:
            rejected += 1
            continue
        regions.append({'id': f'grounded_{i}_{j}', 'role': 'control', 'reviewed': True,
                        'bounds': [i*a.cell, j*a.cell, (i+1)*a.cell, (j+1)*a.cell],
                        'review': 'data-based grounded/slow cell (CATS mask and response <= 0.02, speed); no imagery review',
                        'sources': len(m), 'supported_sources': len(sup), 'median_speed_m_day': med,
                        'not_fast_fraction': slow})
    checksum = package_digest(a.package)
    if revision(Path(a.package)) != before:
        raise ValueError('Source package changed during control selection')
    out = {'EPSG': epsg, 'regions': regions,
           'provenance': {'package': str(a.package), 'package_sha256': checksum,
                          'array_verification': 'full' if nisar and getattr(a, 'verify_package', False) else 'not_requested', 'cell_m': a.cell,
                          'observation_support': 'geocoded_cell' if nisar else 'native_source',
                          'min_supported_sources': a.min_supported, 'rejected_cells': rejected,
                          'rules': 'all sources CATS mask/response <= 0.02; median speed <= 0.05 and >= 90% <= 0.2 m/day',
                          'tool_sha256': digest(Path(__file__))}}
    Path(a.output).write_text(json.dumps(out, indent=1)+'\n')
    print(f'{len(regions)} control boxes, {sum(r["sources"] for r in regions)} sources; {rejected} cells rejected')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cell', type=float, default=3000.)
    p.add_argument('--min-supported', type=int, default=10)
    p.add_argument('--verify-package', action='store_true', help='Fully verify NISAR array hashes before screening (slow; default: metadata checks and final file SHA-256)')
    main(p.parse_args())

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
inside each box. Run in ice with PYTHONPATH=src.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from glacier_offsets.velocity_io import digest


def main(a):
    with h5py.File(a.package) as h:
        so = h['source_observation_index'][:]
        unique, first = np.unique(so, return_index=True)
        first = first[unique >= 0]
        row, col = np.unravel_index(first, so.shape)
        x, y = h['x'][:][col], h['y'][:][row]
        response = h['tidal_response_factor'][:][row, col]
        ocean = h['tidal_model_ocean_mask'][:][row, col]
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
        epsg = int(np.ravel(h.attrs['EPSG'])[0]) if 'EPSG' in h.attrs else 3031
    cx, cy = np.floor(x/a.cell).astype(int), np.floor(y/a.cell).astype(int)
    regions, rejected = [], 0
    for i, j in sorted(set(zip(cx.tolist(), cy.tolist()))):
        m = (cx == i) & (cy == j)
        grounded = bool((np.nan_to_num(response[m], nan=1.) <= .02).all() and (np.nan_to_num(ocean[m], nan=1.) <= .02).all())
        sup = m & (support >= 20)
        if not grounded or sup.sum() < a.min_supported:
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
                        'sources': int(m.sum()), 'supported_sources': int(sup.sum()), 'median_speed_m_day': med,
                        'not_fast_fraction': slow})
    out = {'EPSG': epsg, 'regions': regions,
           'provenance': {'package': str(a.package), 'package_sha256': digest(Path(a.package)), 'cell_m': a.cell,
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
    main(p.parse_args())

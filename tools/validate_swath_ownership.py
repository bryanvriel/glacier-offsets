#!/usr/bin/env python3
"""Validate a swath ownership change using retained, aligned stack SLCs.

Writes isolated reference/date assemblies, checks full chip/search support on
both dates, and optionally matches a bounded range strip using native Ampcor.
No COMPASS, production writes, gap filling or geocoding is performed.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.ampcor import native_match
from glacier_offsets.config import load_defaults
from glacier_offsets.matching_grid import MatchingGrid
from glacier_offsets.products import ChecksumCache, identity, write_json
from glacier_offsets.stack_ownership import POLICY
from glacier_offsets.stack_storage import assemble_date, write_reference_layout


def supported(valid, grid, side, *, origin_sample=0):
    """The production full chip/search test, applied to saved validity masks."""
    integral = np.pad(~valid.astype(bool), ((1, 0), (1, 0))).astype('uint32')
    np.cumsum(integral, axis=0, out=integral)
    np.cumsum(integral, axis=1, out=integral)
    halo = grid.search if side else (0, 0)
    y = grid.starts_line[:, None]-halo[0]
    x = grid.starts_sample[None, :]-halo[1]-origin_sample
    h, w = (grid.window[i]+2*halo[i] for i in range(2))
    return (integral[y+h, x+w]+integral[y, x]-integral[y+h, x]-integral[y, x+w]) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stack', type=Path)
    parser.add_argument('--pair', required=True, help='YYYY-MM-DD_YYYY-MM-DD')
    parser.add_argument('--output', type=Path, required=True, help='New directory outside the source stack')
    parser.add_argument('--native-samples', nargs=2, type=int, metavar=('START', 'STOP'),
                        help='Also match centers in this global range-sample interval')
    args = parser.parse_args()
    source, output = args.stack.resolve(), args.output.resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError('Output must be a new directory outside the source stack')
    gdal.UseExceptions()
    old = json.loads((source/'geometry/manifest.json').read_text())
    pair = source/'pairs'/args.pair
    previous = json.loads((pair/'manifest.json').read_text())
    if previous['matching']['gross_offsets']['mode'] != 'zero':
        raise ValueError('This validation requires zero gross offsets')
    days = previous['stack_pair']['acquisition_dates']
    aligned = [json.loads((source/'acquisitions'/day/'manifest.json').read_text()) for day in days]
    originals = {Path(p) for date in aligned for p in date['slcs'].values()}
    originals.update(Path(p) for entry in old['sources']
                     for p in [entry['coreg']['slcs']['reference'], *entry['coreg']['geometry'].values()])
    originals.update([Path(old['coreg']['source_index']), pair/'radar/offsets.h5'])
    originals.update(Path(date['validity_mask']) for date in aligned)
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in originals}
    checksums = ChecksumCache()
    print('Building isolated reference ownership', flush=True)
    frame = write_reference_layout(output/'geometry', old['sources'], checksums=checksums)
    frame['geometry_identity'] = identity({'previous': old['geometry_identity'], 'ownership': POLICY})
    write_json(output/'geometry/manifest.json', frame)
    cfg = load_defaults()
    cfg['output'] = {'directory': str(output)}
    cfg['matching'] = deepcopy(previous['matching']['matching'])
    cfg['gross_offsets'] = deepcopy(previous['matching']['gross_offsets'])
    dates = []
    for day, data in zip(days, aligned):
        print(f'Assembling {day} from existing aligned bursts', flush=True)
        dates.append(assemble_date(cfg, SimpleNamespace(date=day), frame, data, checksums=checksums))
    p = cfg['matching']
    with h5py.File(pair/'radar/offsets.h5') as h:
        grid = MatchingGrid((h['center_line'][:, 0]-(p['window'][0]-1)/2).astype('int32'),
                            (h['center_sample'][0, :]-(p['window'][1]-1)/2).astype('int32'),
                            tuple(p['window']), tuple(p['search']), tuple(p['stride']))
        old_valid = h['input_valid'][:]
        new_valid = np.ones(grid.shape, bool)
        per_date = []
        for side, (day, data) in enumerate(zip(days, dates)):
            mask = gdal.Open(data['validity_mask']).ReadAsArray()
            support = supported(mask, grid, side)
            new_valid &= support
            per_date.append({'date': day, 'supported_windows': int(support.sum())})
        report = {'pair': args.pair, 'source': str(source), 'output': str(output),
                  'ownership': frame['ownership'], 'windows': int(old_valid.size),
                  'old_input_valid': int(old_valid.sum()), 'new_input_valid': int(new_valid.sum()),
                  'recovered_windows': int((new_valid & ~old_valid).sum()),
                  'lost_windows': int((old_valid & ~new_valid).sum()), 'dates': per_date,
                  'scope': 'Retained aligned SLCs; support validation is not displacement accuracy validation.'}
        if args.native_samples:
            columns = np.flatnonzero((grid.centers[1][0] >= args.native_samples[0]) &
                                     (grid.centers[1][0] < args.native_samples[1]))
            if not columns.size:
                raise ValueError('No matching centers in the requested interval')
            local = MatchingGrid(grid.starts_line, grid.starts_sample[columns], grid.window, grid.search, grid.stride)
            print(f'Native matching {local.shape[0]} x {local.shape[1]} windows', flush=True)
            result = native_match(dates[0]['slc'], dates[1]['slc'], output/'native', p, cfg,
                                  _grid=local, _valid_input=new_valid[:, columns])
            # A control is unchanged only when neither chip nor secondary search
            # footprint intersects any changed source ownership.
            x0 = int(local.starts_sample[0]-local.search[1])
            x1 = int(local.starts_sample[-1]+local.window[1]+local.search[1])
            owner_paths = [old['coreg']['source_index'], frame['coreg']['source_index']]
            owners = [gdal.Open(path).ReadAsArray(x0, 0, x1-x0, frame['coreg']['reference_grid']['length'])
                      for path in owner_paths]
            same = owners[0] == owners[1]
            controls = supported(same, local, 0, origin_sample=x0) & supported(same, local, 1, origin_sample=x0)
            controls &= h['valid'][:][:, columns] & result['valid']
            old_offsets = np.stack([h[f'total/{axis}_px'][:][:, columns] for axis in ('azimuth', 'range')], axis=-1)
            delta = abs(result['total']-old_offsets)[controls]
            report['native'] = {'samples': args.native_samples, 'windows': int(np.prod(local.shape)),
                                'unchanged_controls': int(controls.sum()),
                                'max_control_difference_px': float(delta.max()) if delta.size else None,
                                'new_valid': int(result['valid'].sum()),
                                'recovered_valid': int((result['valid'] & ~old_valid[:, columns]).sum())}
            if not delta.size or delta.max() > 1/p['oversampling']:
                raise AssertionError(f'Unchanged-source native controls failed: {report["native"]}')
    report['source_sizes_mtimes_unchanged'] = all((p.stat().st_size, p.stat().st_mtime_ns) == before[p] for p in originals)
    if not report['source_sizes_mtimes_unchanged']:
        raise AssertionError('Source products changed during validation')
    report['status'] = 'complete'
    write_json(output/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Benchmark verified NISAR extraction and purged screening without fitting a map.

Run in ice with PYTHONPATH=src. Inputs and region files are read-only. Validation
receipts are reusable by velocity-pixels when the same cache directory is used.
"""
import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from glacier_offsets.nisar_collection import NisarCollection
from glacier_offsets.nisar_velocity import purged_predictions, _purged_predictions_reference
from glacier_offsets.velocity_io import write_json
from glacier_offsets.velocity_pixel_general import Layout


def benchmark(package, regions, cache, samples=128):
    record = json.loads(Path(regions).read_text())
    if int(record['EPSG']) != 3031:
        raise ValueError('Benchmark control bounds must use EPSG:3031')
    controls = [r for r in record['regions'] if r.get('role') == 'control' and r.get('reviewed')]
    if not controls or samples < 1:
        raise ValueError('Need reviewed controls and a positive sample count')
    report = {'package': str(Path(package).resolve()), 'regions': str(Path(regions).resolve())}
    start = perf_counter()
    with NisarCollection([package], cache_directory=cache) as c:
        report.update(collection_seconds=perf_counter()-start, grid_shape=list(c.shape), products=len(c.members))
        years = sorted({r['year'] for r in c.members})
        pieces = []
        for j in np.unique(np.linspace(0, len(controls)-1, min(3, len(controls)), dtype=int)):
            west, south, east, north = controls[j]['bounds']
            rows = np.flatnonzero((c.y >= south) & (c.y <= north))
            cols = np.flatnonzero((c.x >= west) & (c.x <= east))
            if not len(rows) or not len(cols): continue
            section = (slice(int(rows.min()), int(rows.max())+1), slice(int(cols.min()), int(cols.max())+1))
            tile = c.tile(section, radar_only=True)
            count = tile['valid'].reshape(len(c.members), -1).sum(axis=0)
            eligible = np.flatnonzero(count >= 3)
            if not len(eligible): continue
            selected = eligible[np.linspace(0, len(eligible)-1, min(samples, len(eligible)), dtype=int)]
            field = {k: tile[k].reshape(len(c.members), -1).T[selected] for k in ('early', 'late', 'valid', 'peak')}
            field['data'] = tile['data_ra'].reshape(len(c.members), -1, 2).swapaxes(0, 1)[selected]
            field['geometry_by_pair'] = tile['geometry'].reshape(len(c.members), -1, 2, 3).swapaxes(0, 1)[selected]
            pieces.append(field)
            sample_section = section
        if not pieces: raise ValueError('No benchmark locations in these controls')
        f = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
        f['year_index'] = np.array([years.index(r['year']) for r in c.members])
        for key in ('reference_acquisition', 'secondary_acquisition'): f[key] = np.array([r[key] for r in c.members])
        f['overlap'] = Layout(years, []).overlaps(f['early'], f['late'])
        start = perf_counter(); reference = _purged_predictions_reference(f)
        report['scalar_screen_seconds'] = perf_counter()-start
        start = perf_counter(); current = purged_predictions(f, valid_only=True)
        report['optimized_screen_seconds'] = perf_counter()-start
        eligible = f['valid'] & np.isfinite(f['peak']) & (f['peak'] >= .08)
        np.testing.assert_allclose(current[eligible], reference[eligible], atol=1e-10, rtol=1e-10)
        difference = abs(current[eligible]-reference[eligible])
        report.update(screen_locations=len(f['early']), control_regions_sampled=len(pieces),
                      max_prediction_difference_m=float(np.nanmax(difference)))
        # Compare an entire native 256-cell tile, including masks and geometry.
        row, col = int(sample_section[0].start//256)*256, int(sample_section[1].start//256)*256
        section = (slice(row, min(row+256, c.shape[0])), slice(col, min(col+256, c.shape[1])))
        indices = range(min(8, len(c.members)))
        start = perf_counter()
        old = {}
        for y in range(section[0].start, section[0].stop, 32):
            for x in range(section[1].start, section[1].stop, 32):
                sec = (slice(y, min(y+32, section[0].stop)), slice(x, min(x+32, section[1].stop)))
                tile = c.tile(sec, indices)
                dest = (slice(y-row, sec[0].stop-row), slice(x-col, sec[1].stop-col))
                for key in ('early', 'late', 'data_ra', 'valid', 'geometry'):
                    if key not in old:
                        old[key] = np.empty((len(indices), section[0].stop-row, section[1].stop-col, *tile[key].shape[3:]), dtype=tile[key].dtype)
                    old[key][(slice(None), *dest)] = tile[key]
        report['previous_extraction_seconds'] = perf_counter()-start
        start = perf_counter(); current = c.tile(section, indices, radar_only=True)
        report['aligned_extraction_seconds'] = perf_counter()-start
        for key in old: np.testing.assert_array_equal(current[key], old[key])
        c.check_revisions()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--regions', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=128, help='Maximum locations per sampled control region')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    result = benchmark(args.package, args.regions, args.cache, args.samples)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))

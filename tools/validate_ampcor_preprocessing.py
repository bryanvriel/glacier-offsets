#!/usr/bin/env python3
"""Known fractional range-shift controls on real complex burst textures.

Only the isolated secondary tile is synthetic: Fourier-shift the original complex
SLC in range, then compare amplitude matching with deramp 0 and 1. Azimuth
translation is held at zero: a plain Fourier shift across the TOPS azimuth
carrier is not established here as a physical motion simulation.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from osgeo import gdal
from scipy.ndimage import fourier_shift

from glacier_offsets.ampcor import _native_executor, _native_worker
from glacier_offsets.matching_grid import build_grid
from glacier_offsets.products import write_json


def run_job(job):
    label, args, kwargs = job
    return label, _native_worker((args, kwargs))


def write_tile(path, array):
    ds = gdal.GetDriverByName('GTiff').Create(str(path), array.shape[1], array.shape[0], 1,
                                            gdal.GDT_CFloat32, options=['TILED=NO'])
    ds.GetRasterBand(1).WriteArray(array.astype('complex64'))
    ds = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pair, out = args.pair.resolve(), args.output.resolve()
    stack = pair.parent.parent
    if out.exists() or out == stack or stack in out.parents or out in stack.parents:
        raise ValueError('Use a new directory outside the production stack')
    out.mkdir(parents=True)
    gdal.UseExceptions()
    manifest = json.loads((pair/'manifest.json').read_text())
    date = manifest['stack_pair']['acquisition_dates'][0]
    slcs = json.loads((stack/'acquisitions'/date/'manifest.json').read_text())['slcs']
    ids = [next(k for k in slcs if k.endswith(f'_iw{swath}')) for swath in (1, 2)]
    params = manifest['matching']['matching']
    shifts = [(0., -0.375), (0., 0.3), (0., -3.375)]
    jobs = []
    for key in ids:
        # A single complete complex tile supplies known band-limited translations.
        # Native chip/search footprints remain within the tile boundaries.
        source = gdal.Open(slcs[key])
        tile = source.ReadAsArray(5000, 500, 4096, 384)
        if not np.isfinite(tile).all() or (abs(tile) == 0).mean() > .001:
            raise ValueError(f'{key}: choose a fully supported control tile')
        directory = out/key
        directory.mkdir()
        reference = directory/'reference.tif'
        write_tile(reference, tile)
        grid = build_grid(tile.shape, tile.shape, params['window'], params['search'], params['stride'])
        for index, translation in enumerate(shifts):
            secondary = directory/f'secondary-{index}.tif'
            shifted = np.fft.ifftn(fourier_shift(np.fft.fftn(tile), translation))
            write_tile(secondary, shifted)
            for mode in (0, 1):
                matching = deepcopy(params)
                matching['deramp'] = mode
                label = {'burst': key, 'translation_px': translation, 'deramp': mode}
                native = directory/f'shift-{index}-deramp-{mode}'
                jobs.append((label, (reference, secondary, native, matching, {'gross_offsets': {'mode': 'zero'}}, 'cpu'),
                             {'_grid': grid, '_valid_input': np.ones(grid.shape, bool)}))
    records = []
    with _native_executor(4, single_threaded=True) as pool:
        for label, result in pool.imap_unordered(run_job, jobs, chunksize=1):
            good = result['valid']
            delta = result['total'][good]-np.array(label['translation_px'])
            label.update(windows=int(good.size), accepted=int(good.sum()), errors_px={
                axis: {'median_signed': float(np.median(delta[:, k])),
                       'median_absolute': float(np.median(abs(delta[:, k]))),
                       'p90_absolute': float(np.quantile(abs(delta[:, k]), .9))}
                for k, axis in enumerate(('azimuth', 'range'))})
            records.append(label)
    report = {'status': 'complete', 'source_pair': str(pair), 'tile_origin_yx': [500, 5000],
              'tile_shape': [384, 4096], 'records': records,
              'scope': 'Known range translations of real complex SLC textures with zero azimuth shift; no temporal decorrelation, independent glacier truth or smoothing.'}
    write_json(out/'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

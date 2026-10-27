#!/usr/bin/env python
"""Plot geocoded range and azimuth displacements (m) for up to six stack pairs.

Reads <output.directory>/pairs/<pair>/geo/{range,azimuth}_displacement_m.tif for
pairs with both products, picks up to --count at random, and saves one 2x3
figure per component. Products are read only; figures go to --output-dir.
"""
import argparse
from pathlib import Path
import random

import numpy as np
from osgeo import gdal

from glacier_offsets.stack_config import load_status_config

COMPONENTS = {'range': (-3, 3), 'azimuth': (-5, 5)}
ROWS, COLUMNS = 2, 3


def completed_pairs(root):
    """Pair directories with both geocoded displacements, skipping AppleDouble '._' entries."""
    return sorted(p for p in (root/'pairs').glob('[!.]*')
                  if all((p/'geo'/f'{c}_displacement_m.tif').is_file() for c in COMPONENTS))


def read(path, quality=False):
    ds = gdal.Open(str(path))
    data = ds.GetRasterBand(1).ReadAsArray().astype('float32')
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None and not np.isnan(nodata):
        data[data == nodata] = np.nan
    if quality:
        mask = gdal.Open(str(path.with_name('quality_valid_mask.tif'))).ReadAsArray()
        data[mask != 1] = np.nan
    x0, dx, _, y0, _, dy = ds.GetGeoTransform()
    extent = np.array([x0, x0 + dx*ds.RasterXSize, y0 + dy*ds.RasterYSize, y0])/1e3
    return data, extent, ds.GetSpatialRef().GetAuthorityCode(None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Stack YAML; its output.directory holds pairs/')
    parser.add_argument('--count', type=int, default=ROWS*COLUMNS, help='Pairs to plot, at most 6 (default: 6)')
    parser.add_argument('--seed', type=int, help='Random seed for a reproducible pair choice')
    parser.add_argument('--quality', action='store_true', help='Mask pixels outside quality_valid_mask.tif')
    parser.add_argument('--output-dir', type=Path, default=Path('.'), help='Figure directory (default: .)')
    parser.add_argument('--show', action='store_true', help='Also open interactive Matplotlib windows')
    args = parser.parse_args(argv)
    if not 1 <= args.count <= ROWS*COLUMNS:
        parser.error(f'--count must be between 1 and {ROWS*COLUMNS}')
    gdal.UseExceptions()
    root = Path(load_status_config(args.config)['output']['directory'])
    pairs = completed_pairs(root)
    if not pairs:
        parser.error(f'No pairs with geocoded displacements under {root/"pairs"}')
    chosen = sorted(random.Random(args.seed).sample(pairs, min(args.count, len(pairs))))

    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for component, (vmin, vmax) in COMPONENTS.items():
        fig, axes = plt.subplots(ROWS, COLUMNS, figsize=(15, 9), sharex=True, sharey=True, layout='constrained')
        cmap = plt.get_cmap('RdBu_r').copy()
        cmap.set_bad('#e6e6e6')
        for ax, pair in zip(axes.flat, chosen):
            data, extent, epsg = read(pair/'geo'/f'{component}_displacement_m.tif', args.quality)
            image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
            ax.set_title(f'{pair.name} · {int(np.isfinite(data).sum())} px')
            ax.set_aspect('equal')
        for ax in axes.flat[len(chosen):]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel(f'Easting (km, EPSG:{epsg})')
        for ax in axes[:, 0]:
            ax.set_ylabel(f'Northing (km, EPSG:{epsg})')
        fig.colorbar(image, ax=axes, shrink=.8, extend='both', label=f'{component.capitalize()} displacement (m)')
        fig.suptitle(f'{root.name} · geocoded {component} displacement'
                     f'{" · quality-valid" if args.quality else ""} · {len(chosen)} of {len(pairs)} pairs')
        output = args.output_dir/f'{root.name}_{component}_displacement_m.png'
        fig.savefig(output, dpi=150)
        print(f'Saved {output.resolve()}')
    print('Pairs: ' + ', '.join(p.name for p in chosen))
    if args.show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()

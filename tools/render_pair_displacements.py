#!/usr/bin/env python
"""Render dated PNG previews of every geocoded range/azimuth displacement.

Supply a directory containing pair directories, a stack root containing pairs/,
or one pair directory. Each pair's geo/{azimuth,range}_displacement_m.tif is
rendered into <output-dir>/{azimuth,range}/<pair>.png. Missing components are
reported and skipped. Inputs are read only; existing preview PNGs are replaced.

Requires NumPy, Matplotlib, and rasterio; runs without importing glacier_offsets.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import rasterio


COMPONENTS = ('azimuth', 'range')


def discover(directory):
    """Collect available components without requiring both for every pair."""
    if not directory.is_dir():
        raise ValueError(f'Not a directory: {directory}')
    if (directory/'geo').is_dir():
        pairs = [directory]
    else:
        root = directory/'pairs' if (directory/'pairs').is_dir() else directory
        pairs = sorted(p for p in root.iterdir()
                       if p.is_dir() and not p.name.startswith('.') and (p/'geo').is_dir())
    sources = {component: [] for component in COMPONENTS}
    for pair in pairs:
        found = False
        for component in COMPONENTS:
            source = pair/'geo'/f'{component}_displacement_m.tif'
            if source.is_file():
                sources[component].append((pair.name, source))
                found = True
        if found:
            for component in COMPONENTS:
                if not (pair/'geo'/f'{component}_displacement_m.tif').is_file():
                    print(f'Skipping missing {component} raster: {pair}', flush=True)
    if not any(sources.values()):
        raise ValueError(f'No geo/{{azimuth,range}}_displacement_m.tif rasters under {directory}')
    return sources


def map_frame(sources):
    """Check CRS/orientation and fix one map frame for all components and dates."""
    crs = None
    bounds = []
    for entries in sources.values():
        for _, source in entries:
            with rasterio.open(source) as dataset:
                if dataset.crs is None:
                    raise ValueError(f'Missing CRS: {source}')
                if crs is None:
                    crs = dataset.crs
                elif dataset.crs != crs:
                    raise ValueError(f'CRS differs from other inputs: {source}; render different CRSs separately')
                t = dataset.transform
                if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
                    raise ValueError(f'Expected a north-up, unrotated geocoded raster: {source}')
                bounds.append(dataset.bounds)
    if crs.is_geographic:
        scale = 1.
        labels = ('Longitude (degrees)', 'Latitude (degrees)')
    elif crs.is_projected:
        scale = crs.linear_units_factor[1]/1000.
        authority = crs.to_authority()
        name = ':'.join(authority) if authority else 'Projected'
        labels = (f'{name} x (km)', f'{name} y (km)')
    else:
        raise ValueError(f'Expected a geographic or projected map CRS, got {crs}')
    frame = np.array([min(b.left for b in bounds), max(b.right for b in bounds),
                      min(b.bottom for b in bounds), max(b.top for b in bounds)])*scale
    return frame, scale, labels


def render(sources, output, limits, cmap_name='RdBu_r', dpi=150):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('0.9')
    frame, scale, labels = map_frame(sources)
    total = sum(len(entries) for entries in sources.values())
    count = 0
    for component, entries in sources.items():
        folder = output/component
        folder.mkdir(parents=True, exist_ok=True)
        vmin, vmax = limits[component]
        for pair, source in entries:
            with rasterio.open(source) as dataset:
                data = np.ma.masked_invalid(dataset.read(1, masked=True))
                b = dataset.bounds
                extent = np.array([b.left, b.right, b.bottom, b.top])*scale
            dates = pair.split('_')
            title = f'{dates[0]} to {dates[1]}' if len(dates) == 2 else pair
            fig, ax = plt.subplots(figsize=(12, 11))
            try:
                fig.subplots_adjust(left=.10, right=.87, bottom=.09, top=.90)
                image = ax.imshow(data, extent=extent, origin='upper', cmap=cmap,
                                  vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.set(xlim=frame[:2], ylim=frame[2:], xlabel=labels[0], ylabel=labels[1],
                       title=f'{title}\n{component.capitalize()} displacement')
                ax.set_aspect('equal')
                colorbar_ax = fig.add_axes([.90, .20, .025, .58])
                fig.colorbar(image, cax=colorbar_ax, extend='both',
                             label=f'{component.capitalize()} displacement (m)')
                fig.savefig(folder/f'{pair}.png', dpi=dpi, facecolor='white')
            finally:
                plt.close(fig)
            count += 1
            if count == 1 or count % 20 == 0 or count == total:
                print(f'Rendered {count}/{total}: {component}/{pair}.png', flush=True)
        print(f'{component.capitalize()}: {len(entries)} previews in {folder.absolute()}', flush=True)
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path, help='Pairs directory, stack root, or one pair directory')
    parser.add_argument('--output-dir', type=Path,
                        help='Preview root (default: <directory>/displacement_previews)')
    parser.add_argument('--cmap', default='RdBu_r', help='Matplotlib color map (default: RdBu_r)')
    parser.add_argument('--clim', type=float, nargs=2, default=(-12., 12.), metavar=('MIN', 'MAX'),
                        help='Shared color limits in metres (default: -12 12)')
    parser.add_argument('--azimuth-clim', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='Override azimuth color limits in metres')
    parser.add_argument('--range-clim', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='Override range color limits in metres')
    parser.add_argument('--dpi', type=int, default=150, help='PNG resolution (default: 150 dpi)')
    args = parser.parse_args(argv)
    limits = {c: getattr(args, f'{c}_clim') or args.clim for c in COMPONENTS}
    for name, values in [('clim', args.clim), *[(f'{c}-clim', limits[c]) for c in COMPONENTS]]:
        if not all(math.isfinite(v) for v in values) or values[0] >= values[1]:
            parser.error(f'--{name} requires finite MIN < MAX')
    if args.dpi < 1:
        parser.error('--dpi must be positive')
    directory = args.directory.expanduser()
    output = args.output_dir.expanduser() if args.output_dir is not None else directory/'displacement_previews'
    try:
        sources = discover(directory)
        render(sources, output, limits, args.cmap, args.dpi)
    except (ValueError, OSError, rasterio.errors.RasterioError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

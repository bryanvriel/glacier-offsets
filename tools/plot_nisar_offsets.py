#!/usr/bin/env python
"""Save side-by-side range and azimuth quicklooks for completed GOFF subsets."""
import argparse
from pathlib import Path
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from glacier_offsets.displacement import valid_nisar_mask


def plot_subset(source, output, layer, fixed_limit):
    with h5py.File(source, 'r') as h:
        x, y = h['x'][:], h['y'][:]
        epsg = int(h.attrs.get('epsg', 3031))
        valid = valid_nisar_mask(h['mask'][:])
        offsets = [h[f'layer{layer}/{name}'][:] for name in ('slantRangeOffset', 'alongTrackOffset')]
    if valid.shape != (len(y), len(x)) or any(offset.shape != valid.shape for offset in offsets):
        raise ValueError('Offsets, mask and map coordinates have inconsistent shapes')

    dx = (x[-1]-x[0])/(len(x)-1) if len(x) > 1 else 80
    dy = (y[-1]-y[0])/(len(y)-1) if len(y) > 1 else -80
    extent = (x[0]-dx/2, x[-1]+dx/2, min(y[0]-dy/2, y[-1]+dy/2), max(y[0]-dy/2, y[-1]+dy/2))
    cmap = plt.get_cmap('RdBu_r').with_extremes(bad='#e6e6e6')
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, layout='constrained')
    try:
        for ax, offset, label in zip(axes, offsets, ('Slant range offset', 'Along-track (azimuth) offset')):
            mask = valid & np.isfinite(offset)
            values = offset[mask]
            limit = fixed_limit or (float(np.percentile(abs(values), 98)) if values.size else 0) or 1
            image = ax.imshow(np.ma.masked_where(~mask, offset), extent=extent,
                              origin='upper' if dy < 0 else 'lower', interpolation='nearest',
                              cmap=cmap, vmin=-limit, vmax=limit)
            below, above = np.any(values < -limit), np.any(values > limit)
            extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
            fig.colorbar(image, ax=ax, label='Displacement (m)', shrink=.83, extend=extend)
            ax.set(title=f'{label} · {mask.mean():.0%} valid', xlabel=f'EPSG:{epsg} x (m)')
        axes[0].set_ylabel(f'EPSG:{epsg} y (m)')
        axes[0].locator_params(nbins=5)
        fig.suptitle(f'{source.stem}\nLayer {layer}', fontsize=9)
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subsets', type=Path, help='Directory of NISAR subset .h5 files')
    parser.add_argument('output', type=Path, help='Directory for the generated PNGs')
    parser.add_argument('--layer', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--limit', type=float, help='Fixed symmetric color limit in metres (default: per-panel 98th percentile of absolute offsets)')
    args = parser.parse_args(argv)
    if args.limit is not None and not (np.isfinite(args.limit) and args.limit > 0):
        parser.error('--limit must be a positive finite number')
    files = sorted(path for path in args.subsets.glob('*.h5') if path.is_file())
    if not files:
        parser.error(f'No completed .h5 subsets found in {args.subsets}')
    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for source in files:
        output = args.output / f'{source.stem}_layer{args.layer}.png'
        try:
            plot_subset(source, output, args.layer, args.limit)
        except (KeyError, OSError, ValueError) as exc:
            print(f'{source.name}: {exc}', file=sys.stderr)
            failures += 1
        else:
            print(output)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())

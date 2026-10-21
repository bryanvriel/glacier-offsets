#!/usr/bin/env python
"""Plot workflow offsets.h5 at full-resolution reference radar-window centers."""
import argparse
import json
import os
from pathlib import Path
import sys

# Support this workspace's plotting dependencies without changing the conda env.
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / '.runtime' / 'deps'))
os.environ.setdefault('MPLCONFIGDIR', str(WORKSPACE / '.runtime' / 'matplotlib'))

import h5py
import numpy as np


def locate_offsets(path):
    path = Path(path).expanduser()
    candidates = [path] if path.is_file() else [path / 'offsets.h5', path / 'radar' / 'offsets.h5']
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f'No offsets.h5 found in {path}; supply a burst output directory or its radar directory')


def cell_edges(centers, stride):
    """Display bins centered on measurements; these are not matching chip edges."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([centers[0] - stride / 2, centers[0] + stride / 2])
    if not np.all(np.diff(centers) > 0):
        raise ValueError('Radar-window centers must increase along each axis')
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.r_[centers[0] - (midpoints[0] - centers[0]), midpoints,
                 centers[-1] + (centers[-1] - midpoints[-1])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='Burst output directory, radar directory, or offsets.h5')
    parser.add_argument('--field', choices=['residual', 'total'], default='total')
    parser.add_argument('--mask', choices=['all', 'input', 'valid'], default='all',
                        help='all: unfiltered; input: input-valid; valid: quality-valid (default: all)')
    parser.add_argument('--percentile', type=float, default=98,
                        help='Upper percentile for color limits; 100 shows the full range (default: 98)')
    parser.add_argument('--output', type=Path, help='Figure path; defaults beside offsets.h5')
    parser.add_argument('--show', action='store_true', help='Also open an interactive Matplotlib window')
    args = parser.parse_args(argv)
    if not 0 < args.percentile <= 100:
        parser.error('--percentile must be in (0, 100]')
    try:
        source = locate_offsets(args.directory)
        with h5py.File(source) as h:
            azimuth = h[f'{args.field}/azimuth_px'][:]
            range_offset = h[f'{args.field}/range_px'][:]
            snr = h['snr'][:]
            lines, samples = h['center_line'][:], h['center_sample'][:]
            stride = h.attrs['stride']  # [azimuth, range] in full-resolution SLC pixels
            selected = np.ones(azimuth.shape, bool) if args.mask == 'all' else h[{'input': 'input_valid', 'valid': 'valid'}[args.mask]][:].astype(bool)
            metadata = json.loads(h.attrs.get('metadata_json', '{}'))
            prior_flags = h['prior/rejection_flags'][:] if 'prior/rejection_flags' in h else None
        if azimuth.ndim != 2 or any(a.shape != azimuth.shape for a in [range_offset, snr, lines, samples, selected]):
            raise ValueError('Offset, SNR, mask, and center arrays must share one 2D shape')
        if not np.isfinite(lines).all() or not np.isfinite(samples).all():
            raise ValueError('Radar-window centers must be finite')
        if not np.allclose(lines, lines[:, :1]) or not np.allclose(samples, samples[:1, :]):
            raise ValueError('Expected a rectangular radar-window center grid')
        x_edges = cell_edges(samples[0], stride[1])
        y_edges = cell_edges(lines[:, 0], stride[0])
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))

    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, sharey=True, layout='constrained')
    panel_counts = []
    for ax, data, title, is_offset in zip(
        axes, [azimuth, range_offset, snr],
        [f'{args.field.capitalize()} azimuth offset', f'{args.field.capitalize()} range offset', 'Amplitude-correlation SNR'],
        [True, True, False],
    ):
        display = np.where(selected & np.isfinite(data), data, np.nan)
        finite = display[np.isfinite(display)]
        panel_counts.append((title, int(finite.size), int(display.size-finite.size)))
        if is_offset:
            limit = float(np.percentile(np.abs(finite), args.percentile)) if finite.size else 1.0
            limit = limit if limit > 0 else 1.0
            vmin, vmax, cmap = -limit, limit, 'RdBu_r'
        else:
            limit = float(np.percentile(finite, args.percentile)) if finite.size else 1.0
            vmin, vmax, cmap = 0.0, max(limit, 1e-12), 'viridis'
        colors = plt.get_cmap(cmap).copy()
        colors.set_bad('#e6e6e6')
        artist = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(display),
                               shading='flat', cmap=colors, vmin=vmin, vmax=vmax, rasterized=True)
        extend = 'neither'
        if finite.size:
            below, above = bool(np.any(finite < vmin)), bool(np.any(finite > vmax))
            extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
        colorbar = fig.colorbar(artist, ax=ax, shrink=.85, extend=extend)
        colorbar.set_label('Reference SLC pixels' if is_offset else 'SNR (dimensionless)')
        ax.set_title(f'{title} · {finite.size}/{display.size} finite windows')
        ax.set_ylabel('Reference azimuth line (zero-based)')
        if not finite.size:
            ax.text(.5, .5, 'No finite selected observations', ha='center', va='center', transform=ax.transAxes)
    axes[-1].set_xlabel('Reference range sample (zero-based)')
    axes[0].set_ylim(y_edges[-1], y_edges[0])
    selection = {'all': 'unfiltered observations', 'input': 'input-valid observations', 'valid': 'quality-valid observations'}[args.mask]
    burst = metadata.get('burst_id', source.parent.parent.name)
    axes[0].legend(handles=[Patch(facecolor='#e6e6e6', edgecolor='#999999',
                                 label='Gray = missing or masked observation')],
                   loc='upper right', fontsize=8)
    fig.suptitle(f'{burst} · {selection} · {int(selected.sum())}/{selected.size} grid slots selected\n'
                 'Each displayed bin is one matching-window observation at its radar center')
    output = args.output or source.parent / f'offsets_{args.field}_{args.mask}.png'
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f'Saved {output.resolve()} (color limits: {args.percentile:g}th percentile; source: {source.resolve()})')
    for title, finite, missing in panel_counts:
        print(f'{title}: {finite} finite, {missing} missing/masked windows (gray)')
    if prior_flags is not None:
        reasons = [(1, 'invalid reference geometry'), (2, 'missing velocity prior'),
                   (4, 'radar projection failure'), (8, 'integer overflow'), (16, 'search outside secondary image')]
        print('Velocity-prior rejection counts (flags may overlap): ' + ', '.join(
            f'{name}={int(np.count_nonzero(prior_flags & bit))}' for bit, name in reasons))
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()

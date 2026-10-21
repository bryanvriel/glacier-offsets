#!/usr/bin/env python3
"""Inspect azimuth derivatives of existing assembled offsets; no processing rerun."""
import argparse
import json
from pathlib import Path
import warnings

import h5py
import numpy as np


def derivative(field, line, valid):
    """Forward differences at half-row positions, never filling invalid data."""
    spacing = np.diff(line)
    if np.any(spacing <= 0):
        raise ValueError('Azimuth centers must increase strictly')
    supported = valid[:-1] & valid[1:] & np.isfinite(field[:-1]) & np.isfinite(field[1:])
    return np.where(supported, np.diff(field, axis=0) / spacing[:, None], np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('merged_directory', type=Path)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--snr', type=float, default=5.)
    parser.add_argument('--correlation', type=float, default=.2)
    args = parser.parse_args()
    root = args.merged_directory.resolve()
    out = args.output_directory.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'manifest.json').read_text())
    with h5py.File(root/'radar/offsets.h5') as h:
        line, sample = h['center_line'][:, 0], h['center_sample'][0]
        owner, seam = h['source_burst_index'][:], h['seam_crossing'][:]
        valid = h['valid'][:] & (h['snr'][:] >= args.snr) & (h['correlation_peak'][:] >= args.correlation)
        fields = {axis: h[f'total/{axis}_px'][:] for axis in ('azimuth', 'range')}
    gradients = {axis: derivative(field, line, valid) for axis, field in fields.items()}
    joins = np.flatnonzero(((owner[:-1] != owner[1:]) & (owner[:-1] > 0) & (owner[1:] > 0)).any(axis=1))
    report = {
        'source': str(root/'radar/offsets.h5'),
        'match_identity': manifest['match_identity'],
        'thresholds': {'snr_min': args.snr, 'correlation_peak_min': args.correlation},
        'derivative': 'forward difference / actual full-resolution azimuth line spacing; halfway between matching rows',
        'units': 'offset SLC pixels per reference azimuth SLC line',
        'background': 'same range columns; edges 3 through 8 matching steps away, excluding any seam-crossing endpoints; at least 4 supported background edges',
        'limitations': 'Descriptive, spatially correlated comparisons; no significance test, absolute accuracy claim, smoothing, inpainting or map resampling',
        'joins': [],
    }
    controls = {}
    for j in joins:
        nearby = np.array([k for k in range(max(0, j-8), min(len(line)-1, j+9)) if 3 <= abs(k-j) <= 8])
        entry = {'row_edge': int(j), 'bracketing_center_lines': [float(line[j]), float(line[j+1])], 'fields': {}}
        for axis, gradient in gradients.items():
            background = gradient[nearby].copy()
            background[seam[nearby] | seam[nearby+1]] = np.nan
            selected = np.isfinite(gradient[j]) & (np.isfinite(background).sum(axis=0) >= 4)
            values, baseline = gradient[j, selected], background[:, selected]
            if not values.size:
                entry['fields'][axis] = {'columns': 0}
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                expected = np.nanmedian(baseline, axis=0)
                band = np.nanquantile(baseline, [.1, .9], axis=0)
            delta_line = float(line[j+1]-line[j])
            entry['fields'][axis] = {
                'columns': int(selected.sum()),
                'seam_median_abs_derivative': float(np.median(np.abs(values))),
                'background_median_abs_derivative': float(np.nanmedian(np.abs(baseline))),
                'seam_p90_abs_derivative': float(np.quantile(np.abs(values), .9)),
                'seam_median_signed_derivative': float(np.median(values)),
                'median_local_background_residual_px_per_step': float(np.median(values-expected)*delta_line),
                'seam_median_abs_change_px_per_step': float(np.median(np.abs(values))*delta_line),
                'background_median_abs_change_px_per_step': float(np.nanmedian(np.abs(baseline))*delta_line),
            }
            controls[j, axis] = sample[selected], values, expected, band
        report['joins'].append(entry)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    cuts = [s['offset'][0]+s['valid_window'][0] for s in manifest['assembly']['sources'][1:]]
    midline = (line[:-1]+line[1:])/2
    dx, dy = np.median(np.diff(sample)), np.median(np.diff(line))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    for col, axis in enumerate(fields):
        for row, (data, y, label) in enumerate([
            (np.where(valid, fields[axis], np.nan), line, 'Offset [SLC pixels]'),
            (gradients[axis], midline, 'Azimuth derivative [offset px / SLC line]'),
        ]):
            finite = data[np.isfinite(data)]
            limit = max(float(np.quantile(np.abs(finite), .98)), 1e-6) if finite.size else 1
            im = axes[row, col].imshow(data, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(0, -limit, limit),
                extent=[sample[0]-dx/2, sample[-1]+dx/2, y[-1]+dy/2, y[0]-dy/2])
            for cut in cuts:
                axes[row, col].axhline(cut, color='black', ls='--', lw=.9)
            axes[row, col].set(title=f'{axis.capitalize()}: {label}', xlabel='Reference range sample', ylabel='Reference azimuth line')
            fig.colorbar(im, ax=axes[row, col], shrink=.8, extend='both')
    fig.suptitle(f'PIG offsets and azimuth gradients; SNR ≥ {args.snr:g}, peak ≥ {args.correlation:g}\n'
                 'Dashed lines: SLC splices. Gradients require both adjacent observations; no filling or smoothing.')
    fig.savefig(out/'azimuth_gradients.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(max(1, len(joins)), 2, figsize=(13, max(3, len(joins)*3.2)), squeeze=False, layout='constrained')
    for row, j in enumerate(joins):
        for col, axis in enumerate(fields):
            ax = axes[row, col]
            if (j, axis) not in controls:
                ax.text(.5, .5, 'Insufficient shared support', ha='center', transform=ax.transAxes)
                continue
            x, observed, expected, band = controls[j, axis]
            # Leave unsupported range columns blank in the profile as well.
            index = np.searchsorted(sample, x)
            full_expected = np.full(sample.shape, np.nan)
            full_band = np.full((2, len(sample)), np.nan)
            full_expected[index], full_band[:, index] = expected, band
            ax.fill_between(sample, full_band[0], full_band[1], color='.8', label='Nearby gradients: 10–90%')
            ax.plot(sample, full_expected, color='.3', lw=1, label='Nearby median')
            ax.scatter(x, observed, s=14, color='tab:blue', label='Across join')
            ax.axhline(0, color='.5', lw=.5)
            ax.set(title=f'{axis.capitalize()}; join between lines {line[j]:g}–{line[j+1]:g}; {len(x)} columns',
                   xlabel='Reference range sample', ylabel='Offset px / azimuth SLC line')
            ax.legend(fontsize=8)
    fig.suptitle('Join gradients versus nearby gradients in the same range columns\n'
                 'Background excludes seam-crossing windows; descriptive comparison, correlated samples.')
    fig.savefig(out/'join_gradient_profiles.png', dpi=150)
    plt.close(fig)
    np.savez_compressed(out/'gradients.npz', center_sample=sample, derivative_center_line=midline,
                        strong_valid=valid, **{f'{axis}_d_offset_px_d_line': array for axis, array in gradients.items()})
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

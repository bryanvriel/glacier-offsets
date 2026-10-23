#!/usr/bin/env python3
"""Compare stack splices with original-burst tracking at saved production centers.

Read-only experiment: original SLCs/products are never modified. Native Ampcor
runs on temporary stripped crops, with the pair's saved matching settings.
"""
import argparse
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.ampcor import native_match, _native_executor
from glacier_offsets.matching_grid import MatchingGrid, valid_windows
from glacier_offsets.products import write_json
from validate_burst_boundary import read_crop, splice, write_slc, statistics


def analyse(out):
    """Report saved controls without requiring two valid bursts at an edge."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    report = json.loads((out/'report.json').read_text())
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('.8')
    for tile in report['tiles']:
        with np.load(out/tile['arrays']) as z:
            yy = z['center_line'][:, 0]-(report['matching']['window'][0]-1)/2
            half_search = report['matching']['search'][0]
            window = report['matching']['window'][0]
            for name, entry in tile['variants'].items():
                cut = tile['midpoint_global'] if name == 'midpoint' else tile['cut_global']
                cross = ((yy-half_search < cut) & (yy+window+half_search > cut))[:, None]
                supported = cross & z['upper_valid'] & z[f'{name}_valid'] & (z['upper_snr'] >= 5)
                delta = z[f'{name}_total']-z['upper_total']
                entry['upper_burst_comparison'] = {
                    'count': int(supported.sum()),
                    'changed_over_quarter_pixel': int((supported & (np.max(abs(delta), axis=-1) > .25)).sum()),
                    'azimuth_difference_px': statistics(delta[..., 0][supported]),
                    'range_difference_px': statistics(delta[..., 1][supported])}
            # Preserve negative results: whether the band goes away at the old
            # cut, and whether a new band appears at the shifted cut.
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
            for ax, name, title in zip(axes[0], ['upper', 'production', 'midpoint'],
                                       ['Upper burst alone', 'Current stack splice', 'Splice in middle of overlap']):
                field = np.where(z[f'{name}_valid'], z[f'{name}_total'][..., 0], np.nan)
                im = ax.pcolormesh(z['center_sample'], z['center_line'], field,
                                   cmap=cmap, vmin=-6, vmax=6, shading='nearest')
                ax.axhline(tile['cut_global'], color='black', ls='--', lw=1)
                ax.axhline(tile['midpoint_global'], color='green', ls=':', lw=1)
                ax.invert_yaxis()
                ax.set(title=title, xlabel='Range sample', ylabel='Azimuth line')
            fig.colorbar(im, ax=list(axes[0]), label='Azimuth offset [SLC px]', shrink=.8)
            for side, ax in enumerate(axes[1, :2]):
                for name, color in [('upper', '.3'), ('lower', 'tab:orange'), ('production', 'tab:red'), ('midpoint', 'tab:green')]:
                    ax.plot(np.arange(384)+tile['origin'][0], z[f'{name}_amplitude_profile'][side],
                            color=color, label=name, alpha=.8)
                ax.set(xlim=(tile['cut_global']-15, tile['cut_global']+30),
                       title=f'{report["dates"][side]}: SLC amplitude at edge',
                       xlabel='Azimuth line', ylabel='Median amplitude across strip')
                ax.axvline(tile['cut_global'], color='black', ls='--', lw=1)
                ax.legend(fontsize=8)
            ax = axes[1, 2]
            # A fixed center chosen by range proximity, never by outcome.
            col = np.argmin(abs(z['center_sample'][0]-(tile['origin'][1]+1024)))
            for name in ['upper', 'production', 'midpoint', 'amplitude']:
                ax.plot(z['center_line'][:, col], np.where(z[f'{name}_valid'][:, col],
                    z[f'{name}_total'][:, col, 0], np.nan), '.-', label=name)
            ax.axvline(tile['cut_global'], color='black', ls='--', lw=1)
            ax.axvline(tile['midpoint_global'], color='green', ls=':', lw=1)
            ax.set(title=f'Range sample {z["center_sample"][0, col]:g}',
                   xlabel='Azimuth line', ylabel='Azimuth offset [SLC px]')
            ax.legend(fontsize=8)
            fig.suptitle(f'{tile["name"]}: identical production centers and native settings\n'
                         'Dashed: current cut. Dotted: midpoint cut. Grey: invalid. No correlation/SNR display cutoff.')
            fig.savefig(out/f'{tile["name"]}.png', dpi=150)
            plt.close(fig)
    write_json(out/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--range-centers', type=int, nargs='+', default=[8000, 14000, 20000])
    parser.add_argument('--analyse-only', action='store_true')
    args = parser.parse_args()
    gdal.UseExceptions()
    pair, out = args.pair.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.analyse_only:
        analyse(out)
        return
    manifest = json.loads((pair/'manifest.json').read_text())
    stack = pair.parent.parent
    frame = json.loads((stack/'geometry/manifest.json').read_text())
    dates = manifest['stack_pair']['acquisition_dates']
    acquisitions = [json.loads((stack/'acquisitions'/d/'manifest.json').read_text()) for d in dates]
    params = manifest['matching']['matching'].copy()
    if manifest['matching']['gross_offsets']['mode'] != 'zero':
        raise ValueError('This experiment requires saved zero-prior production offsets')
    with h5py.File(pair/'radar/offsets.h5') as h:
        lines, samples = h['center_line'][:, 0], h['center_sample'][0]
        saved = {k: h[k][:] for k in ('valid', 'input_valid', 'snr', 'correlation_peak')}
        saved['total'] = np.stack([h[f'total/{a}_px'][:] for a in ('azimuth', 'range')], axis=-1)
    inputs = {pair/'manifest.json', pair/'radar/offsets.h5', stack/'geometry/manifest.json'}
    for a in acquisitions:
        inputs.update(Path(p) for p in a['slcs'].values())
        inputs.add(Path(a['slc']))
    before = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in inputs}
    report = {'pair': str(pair), 'match_identity': manifest['match_identity'],
              'geometry_identity': frame['geometry_identity'], 'dates': dates, 'matching': params,
              'limitations': ['Controlled strip comparisons, not independent displacement truth.',
                              'Independent agreement selection is diagnostic, not a product mask.',
                              'All native variants retain production centers, window, search and stride.',
                              'No production products, masks or ownership policy changed.'], 'tiles': []}
    starts = [lines-(params['window'][0]-1)/2, samples-(params['window'][1]-1)/2]
    with _native_executor() as worker, tempfile.TemporaryDirectory(prefix='stack-seams-') as tmp:
        scratch = Path(tmp)
        for join in range(1, len(frame['sources'])):
            upper, lower = frame['sources'][join-1:join+1]
            first_valid = lower['offset'][0]+lower['valid_window'][0]
            cuts = (frame.get('ownership') or {}).get('cuts', [])
            cut = next((c['cut_global_line'] for c in cuts
                        if c.get('upper_burst_id')==upper['burst_id'] and c.get('lower_burst_id')==lower['burst_id']),
                       first_valid)
            top = upper['offset'][0]+upper['valid_window'][1]
            midpoint = (first_valid+top)//2
            for range_center in args.range_centers:
                origin = (cut-192, range_center-1024)
                shape = (384, 2048)
                indexes = [np.flatnonzero((s >= o+search) & (s+window+search <= o+size))
                           for s, o, size, search, window in
                           zip(starts, origin, shape, params['search'], params['window'])]
                iy, ix = indexes
                grid = MatchingGrid(*[(s[ind]-o).astype('int32') for s, ind, o in zip(starts, indexes, origin)],
                                    tuple(params['window']), tuple(params['search']), tuple(params['stride']))
                sources = [[read_crop(a['slcs'][entry['burst_id']], origin, shape, entry['offset'][0])
                            for a in acquisitions] for entry in (upper, lower)]
                production = [read_crop(a['slc'], origin, shape, 0) for a in acquisitions]
                mismatch = [int(np.count_nonzero(p != splice(u, l, cut-origin[0])))
                            for p, u, l in zip(production, *sources)]
                variants = {'upper': sources[0], 'lower': sources[1], 'production': production,
                            'midpoint': [splice(u, l, midpoint-origin[0]) for u, l in zip(*sources)]}
                arrays = {'center_line': grid.centers[0]+origin[0],
                          'center_sample': grid.centers[1]+origin[1]}
                results = {}
                tile_id = f'join-{join}-range-{range_center}'
                profiles = {}
                for name, data in variants.items():
                    profiles[name] = np.stack([np.median(np.abs(x), axis=1) for x in data])
                    paths = [scratch/f'{name}-{side}.tif' for side in range(2)]
                    for path, array in zip(paths, data):
                        write_slc(path, array)
                    ds = [gdal.Open(str(p)) for p in paths]
                    eligible = valid_windows(*ds, grid, (0, 0))
                    ds = None
                    print(f'{tile_id} {name}: {eligible.sum()} supported', flush=True)
                    results[name] = native_match(*paths, scratch/'native', params,
                        {'gross_offsets': {'mode': 'zero'}}, 'cpu', _grid=grid,
                        _gross=np.zeros((*grid.shape, 2), dtype='int32'),
                        _valid_input=eligible, _worker=worker)
                # Ampcor's deramp=0 converts complex data to amplitude before
                # interpolation. This isolates complex-carrier seam sensitivity.
                paths = [scratch/f'production-{side}.tif' for side in range(2)]
                for path, array in zip(paths, production):
                    write_slc(path, array)
                results['amplitude'] = native_match(*paths, scratch/'native', params | {'deramp': 0},
                    {'gross_offsets': {'mode': 'zero'}}, 'cpu', _grid=grid,
                    _gross=np.zeros((*grid.shape, 2), dtype='int32'),
                    _valid_input=results['production']['input_valid'], _worker=worker)
                old = {k: v[np.ix_(iy, ix)] for k, v in saved.items()}
                replay = results['production']['total']-old['total']
                if np.max(np.abs(replay)) > 1/32:
                    raise AssertionError('Cropped native replay differs from saved production by >1/32 px')
                a, b = results['upper'], results['lower']
                agreement = (a['valid'] & b['valid'] & (a['snr'] >= 5) & (b['snr'] >= 5) &
                             (np.max(np.abs(a['total']-b['total']), axis=-1) <= .25))
                baseline = (a['total']+b['total'])/2
                tile = {'name': tile_id, 'origin': list(origin), 'cut_global': cut,
                        'midpoint_global': midpoint, 'rectangular_cut_mismatched_samples': mismatch,
                        'native_replay_difference_px': statistics(replay),
                        'independent_agreement_count': int(agreement.sum()), 'variants': {}}
                arrays.update(independent_agreement=agreement, independent_mean=baseline)
                for name, result in results.items():
                    for field in ('total', 'valid', 'input_valid', 'snr', 'correlation_peak'):
                        arrays[f'{name}_{field}'] = result[field]
                    if name in profiles:
                        arrays[f'{name}_amplitude_profile'] = profiles[name]
                    join_line = midpoint if name == 'midpoint' else cut
                    yy = starts[0][iy, None]
                    crossing = np.broadcast_to((yy-params['search'][0] < join_line) &
                        (yy+params['window'][0]+params['search'][0] > join_line), grid.shape)
                    comparable = agreement & crossing & result['valid']
                    # Moving targets with consistent independent measurements
                    # expose a spurious pull toward the stationary splice.
                    moving = comparable & (np.abs(baseline[..., 0]) > 1)
                    delta = result['total']-baseline
                    tile['variants'][name] = {
                        'crossing_agreement_count': int(comparable.sum()),
                        'moving_crossing_agreement_count': int(moving.sum()),
                        'moving_azimuth_difference_px': statistics(delta[..., 0][moving]),
                        'crossing_difference_px': {axis: statistics(delta[..., k][comparable])
                                                   for k, axis in enumerate(('azimuth', 'range'))},
                        'crossing_changed_over_quarter_pixel': int((comparable &
                            (np.max(np.abs(delta), axis=-1) > .25)).sum())}
                    if name in ('production', 'midpoint'):
                        lower_side = yy-params['search'][0] >= join_line
                        pure = ~crossing
                        expected = np.where(lower_side[..., None], b['total'], a['total'])
                        expected_valid = np.where(lower_side, b['valid'], a['valid'])
                        controls = pure & expected_valid & result['valid']
                        tile['variants'][name]['same_source_control_difference_px'] = statistics(
                            (result['total']-expected)[controls])
                        if not controls.any() or np.max(np.abs((result['total']-expected)[controls])) > 1/32:
                            raise AssertionError('Same-source controls failed')
                archive = f'{tile_id}.npz'
                np.savez_compressed(out/archive, **arrays)
                tile['arrays'] = archive
                report['tiles'].append(tile)
                write_json(out/'report.json', report)
    report['source_sizes_mtimes_unchanged'] = all(
        [p.stat().st_size, p.stat().st_mtime_ns] == before[str(p)] for p in inputs)
    if not report['source_sizes_mtimes_unchanged']:
        raise AssertionError('Source changed during experiment')
    report['source_sizes_mtimes'] = before
    report['status'] = 'completed'
    write_json(out/'report.json', report)
    analyse(out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Summarize a compare_stack_matching experiment without altering its products.

Run in the iceutils environment. Agreement between overlapping burst views is
an internal control, not independent displacement truth or a new quality mask.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path

import h5py
import iceutils as ice
import numpy as np


LABELS = {'baseline': 'Merged complex (saved)', 'merged_magnitude': 'Merged magnitude',
          'unspliced_complex': 'Unspliced complex', 'unspliced_magnitude': 'Unspliced magnitude'}


def offsets(path):
    with h5py.File(path) as h:
        return {k: h[k][:] for k in ('total/azimuth_px', 'total/range_px', 'valid', 'input_valid',
            'correlation_peak', 'snr', 'source_burst_index', 'seam_crossing', 'center_line', 'center_sample')}


def statistics(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return {'n': int(a.size), 'median': float(np.median(a)) if a.size else None,
            'p90': float(np.quantile(a, .9)) if a.size else None,
            'p99': float(np.quantile(a, .99)) if a.size else None,
            'max': float(a.max()) if a.size else None}


def overlap_controls(out, original, arrays, reference='unspliced_magnitude'):
    """Common-center observations from both members of each overlap."""
    baseline = arrays['baseline']
    params = original['matching']['matching']
    global_y = baseline['center_line'][:, 0]-(params['window'][0]-1)/2
    global_x = baseline['center_sample'][0]-(params['window'][1]-1)/2
    sources = []
    for index, entry in enumerate(original['assembly']['sources'], 1):
        path = out/reference/'radar/native'/entry['burst_id']/'result.npz'
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as h:
            source = {k: h[k] for k in ('total', 'valid', 'correlation_peak')}
            effective = json.loads(str(h['effective_json']))
        dy, dx = entry['offset']
        ny, nx = source['valid'].shape
        rows = (effective['referenceStartPixelDownStatic']+dy+np.arange(ny)*params['stride'][0]-global_y[0])/params['stride'][0]
        cols = (effective['referenceStartPixelAcrossStatic']+dx+np.arange(nx)*params['stride'][1]-global_x[0])/params['stride'][1]
        if not np.allclose(rows, np.round(rows)) or not np.allclose(cols, np.round(cols)):
            raise AssertionError('Original-burst controls are off the saved global grid')
        source.update(rows=np.round(rows).astype(int), cols=np.round(cols).astype(int), index=index, id=entry['burst_id'])
        sources.append(source)
    records, all_ids, all_consensus = [], [], []
    for first, second in combinations(sources, 2):
        rows = np.intersect1d(first['rows'], second['rows'])
        cols = np.intersect1d(first['cols'], second['cols'])
        if not rows.size or not cols.size:
            continue
        sections = [np.ix_(np.searchsorted(s['rows'], rows), np.searchsorted(s['cols'], cols)) for s in (first, second)]
        a, b = [s['total'][section] for s, section in zip((first, second), sections)]
        selected = np.ones(a.shape[:2], bool)
        for source, section in zip((first, second), sections):
            selected &= source['valid'][section] & (source['correlation_peak'][section] >= .5)
        selected &= np.all(abs(a-b) <= np.array([1/16, 1/8]), axis=-1)
        global_section = np.ix_(rows, cols)
        selected &= baseline['seam_crossing'][global_section]
        selected &= np.isin(baseline['source_burst_index'][global_section], [first['index'], second['index']])
        for variant in arrays.values():
            selected &= variant['valid'][global_section]
        if not selected.any():
            continue
        yy, xx = np.meshgrid(rows, cols, indexing='ij')
        identifiers = np.ravel_multi_index((yy[selected], xx[selected]), baseline['valid'].shape)
        consensus = ((a+b)/2)[selected]
        all_ids.extend(identifiers)
        all_consensus.extend(consensus)
        entry = {'sources': [first['id'], second['id']], 'n': int(selected.sum()), 'errors_px': {}}
        for name, variant in arrays.items():
            value = np.stack([variant[f'total/{axis}_px'][global_section][selected] for axis in ('azimuth', 'range')], axis=-1)
            entry['errors_px'][name] = {axis: statistics(abs(value-consensus)[:, k]) for k, axis in enumerate(('azimuth', 'range'))}
        records.append(entry)
    identifiers = np.asarray(all_ids, int)
    consensus = np.asarray(all_consensus)
    unique = np.unique(identifiers)
    # A few corners can have more than two valid burst views. Count each global
    # location once, using the median of its independently agreeing pair views.
    target = np.array([np.median(consensus[identifiers == index], axis=0) for index in unique])
    errors = {}
    for name, variant in arrays.items():
        value = np.stack([variant[f'total/{axis}_px'].ravel()[unique] for axis in ('azimuth', 'range')], axis=-1)
        errors[name] = {axis: statistics(abs(value-target)[:, k]) for k, axis in enumerate(('azimuth', 'range'))}
    np.savez_compressed(out/f'overlap_controls_{reference}.npz', indices=unique, consensus_px=target)
    return {'reference_preprocessing': reference, 'unique_centers': int(unique.size), 'burst_pair_comparisons': len(identifiers),
            'selection': f'Both {reference} burst views peak >= 0.5, agree within 1/16 azimuth and 1/8 range pixel; seam-crossing fixed owner in that pair; all variants quality-valid',
            'errors_to_view_consensus_px': errors, 'overlaps': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    out = args.experiment.resolve()
    experiment = json.loads((out/'experiment.json').read_text())
    if experiment['status'] != 'complete':
        raise ValueError('Wait for the full experiment to complete')
    pair = Path(experiment['specification']['pair'])
    original = json.loads((pair/'manifest.json').read_text())
    arrays = {'baseline': offsets(pair/'radar/offsets.h5')}
    arrays.update({name: offsets(Path(data['directory'])/'radar/offsets.h5') for name, data in experiment['variants'].items()})
    baseline = arrays['baseline']
    report = {'pair': pair.name, 'windows': int(baseline['valid'].size), 'variants': {}, 'comparisons': {},
              'production_sizes_mtimes_unchanged': experiment['production_sizes_mtimes_unchanged'],
              'limitations': 'Internal burst-view agreement, no independent displacement truth. Same geometry, grid, ownership, chip/search, zero prior and quality thresholds. No smoothing or gap filling.'}
    for name, variant in arrays.items():
        for key in ('center_line', 'center_sample', 'source_burst_index'):
            np.testing.assert_array_equal(variant[key], baseline[key])
        report['variants'][name] = {'input_valid': int(variant['input_valid'].sum()), 'quality_valid': int(variant['valid'].sum()),
            'lost_input_support': int((baseline['input_valid'] & ~variant['input_valid']).sum()),
            'gained_input_support': int((~baseline['input_valid'] & variant['input_valid']).sum())}
    for merged, unspliced in [('baseline', 'unspliced_complex'), ('merged_magnitude', 'unspliced_magnitude')]:
        a, b = arrays[merged], arrays[unspliced]
        common = a['valid'] & b['valid'] & (a['correlation_peak'] >= .3) & (b['correlation_peak'] >= .3)
        entry = {}
        for label, selected in [('seam', common & baseline['seam_crossing']), ('interior', common & ~baseline['seam_crossing'])]:
            entry[label] = {axis: statistics(abs(a[f'total/{axis}_px'][selected]-b[f'total/{axis}_px'][selected])) for axis in ('azimuth', 'range')}
            entry[label]['within_1_32_pixel_fraction'] = {axis: float(np.mean(abs(a[f'total/{axis}_px'][selected]-b[f'total/{axis}_px'][selected]) <= 1/32)) for axis in ('azimuth', 'range')}
        report['comparisons'][f'{merged}_versus_{unspliced}'] = entry
    report['overlap_controls'] = overlap_controls(out, original, arrays)
    report['complex_overlap_controls'] = overlap_controls(out, original, arrays, reference='unspliced_complex')
    references = {name: np.load(out/f'overlap_controls_unspliced_{name}.npz') for name in ('complex', 'magnitude')}
    common = np.intersect1d(*(a['indices'] for a in references.values()))
    common_errors = {}
    for reference, values in references.items():
        target = values['consensus_px'][np.searchsorted(values['indices'], common)]
        common_errors[reference] = {}
        for name, variant in arrays.items():
            value = np.stack([variant[f'total/{axis}_px'].ravel()[common] for axis in ('azimuth', 'range')], axis=-1)
            common_errors[reference][name] = {axis: statistics(abs(value-target)[:, k]) for k, axis in enumerate(('azimuth', 'range'))}
    report['shared_overlap_controls'] = {'centers': int(common.size), 'errors_by_reference_preprocessing_px': common_errors,
        'interpretation': 'Same centers selected by agreeing independent burst views under BOTH preprocessing modes. Neither consensus is independent displacement truth.'}
    write = out/'comparison.json'
    write.write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('white')
    line, sample = baseline['center_line'][:, 0], baseline['center_sample'][0]
    extent = [sample[0]-64, sample[-1]+64, line[-1]+16, line[0]-16]
    for axis, bound, speedbound in [('azimuth', .4, 1.), ('range', 1.2, .5)]:
        fig, axes = plt.subplots(2, 4, figsize=(19, 10), layout='constrained')
        for col, (name, data) in enumerate(arrays.items()):
            im = axes[0, col].imshow(np.where(data['valid'], data[f'total/{axis}_px'], np.nan), extent=extent,
                                     aspect='auto', cmap=cmap, vmin=-bound, vmax=bound, interpolation='nearest')
            axes[0, col].set(title=LABELS[name], xlabel='Range sample', ylabel='Azimuth line')
            directory = pair if name == 'baseline' else out/name
            raster = ice.Raster(rasterfile=str(directory/'geo'/f'{axis}_velocity_m_per_day.tif'))
            geo_im = axes[1, col].imshow(raster.data, extent=np.array(raster.hdr.extent)/1000,
                                         cmap=cmap, vmin=-speedbound, vmax=speedbound, interpolation='nearest')
            axes[1, col].set(xlabel='EPSG:3031 x [km]', ylabel='EPSG:3031 y [km]')
        fig.colorbar(im, ax=axes[0, :], label=f'{axis.capitalize()} offset [SLC pixels]', shrink=.8, extend='both')
        fig.colorbar(geo_im, ax=axes[1, :], label=f'{axis.capitalize()} velocity [m/day]', shrink=.8, extend='both')
        fig.suptitle(f'{pair.name}: {axis} — full radar grid and saved common map grid\nRaw native observations with unchanged quality checks; no smoothing or gap filling.')
        fig.savefig(out/f'{axis}_comparison.png', dpi=150)
        plt.close(fig)
    controls = report['overlap_controls']
    print(json.dumps({'variants': report['variants'], 'overlap_control_centers': controls['unique_centers'],
                      'overlap_errors': controls['errors_to_view_consensus_px'], 'shared_controls': report['shared_overlap_controls'],
                      'comparisons': report['comparisons']}, indent=2))


if __name__ == '__main__':
    main()

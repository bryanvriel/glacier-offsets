#!/usr/bin/env python3
"""Controlled, read-only-source experiment for two adjacent coregistered bursts.

This does not change production assembly or caches. It compares hard SLC splices
against independent tracking at identical global centers, using native Ampcor.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.ampcor import native_match, _native_executor
from glacier_offsets.matching_grid import MatchingGrid, valid_windows
from glacier_offsets.products import write_json
from glacier_offsets.raster_io import bounded_raster_cache


def placement(first, second, tolerance=0.001):
    """ISCE2-style rounded time/range placement, with explicit residual checks."""
    import isce3
    for key in ('prf', 'range_pixel_spacing', 'wavelength'):
        if not math.isclose(first[key], second[key], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'Incompatible radar sampling: {key}')
    if first['lookside'] != second['lookside']:
        raise ValueError('Incompatible look directions')
    elapsed = (isce3.core.DateTime(second['ref_epoch']) -
               isce3.core.DateTime(first['ref_epoch'])).total_seconds()
    dy = (elapsed + second['sensing_start'] - first['sensing_start']) * first['prf']
    dx = (second['starting_range'] - first['starting_range']) / first['range_pixel_spacing']
    rounded = np.rint([dy, dx]).astype(int)
    residual = np.array([dy, dx]) - rounded
    if np.max(np.abs(residual)) > tolerance:
        raise ValueError(f'Fractional placement {residual}; native SLC resampling required')
    if rounded[0] <= 0 or rounded[0] >= first['length']:
        raise ValueError('Expected time-ordered, overlapping bursts')
    if rounded[1] != 0 or first['width'] != second['width']:
        raise ValueError('This boundary experiment requires equal-range same-swath grids')
    return int(rounded[0]), residual.tolist()


def read_crop(path, origin, shape, burst_line):
    """Place a bounded source window into a common-grid crop; pad outside with zero."""
    y0, x0 = origin
    height, width = shape
    ds = gdal.Open(str(path))
    if x0 < 0 or x0 + width > ds.RasterXSize:
        raise ValueError('Range crop outside source image')
    lo, hi = max(y0, burst_line), min(y0 + height, burst_line + ds.RasterYSize)
    result = np.zeros(shape, dtype='complex64')
    with bounded_raster_cache():
        for row in range(lo, hi, 64):
            size = min(64, hi - row)
            result[row-y0:row-y0+size] = ds.ReadAsArray(x0, row-burst_line, width, size)
    return result


def support_rows(reference, secondary):
    """Estimate rectangular crop support; exact chip masks are checked separately.

    Quantized isolated complex zeros are allowed. This is a measured support
    proxy, not Sentinel-1 annotation firstValidLine/firstValidSample metadata.
    """
    good = np.ones(reference.shape[0], dtype=bool)
    for array in (reference, secondary):
        good &= np.mean(np.isfinite(array) & (array != 0), axis=1) > 0.95
    indexes = np.flatnonzero(good)
    if not len(indexes) or not good[indexes[0]:indexes[-1]+1].all():
        raise ValueError('No contiguous rectangular support in crop')
    return int(indexes[0]), int(indexes[-1] + 1)


def splice(upper, lower, cut):
    """Hard ownership cut, applied identically to both acquisitions; no averaging."""
    result = upper.copy()
    result[cut:] = lower[cut:]
    return result


def write_slc(path, array):
    # The installed CPU Ampcor uses GDAL virtual memory; tiled TIFFs fail that
    # interface on this macOS build. Match the existing stripped SLC products.
    ds = gdal.GetDriverByName('GTiff').Create(str(path), array.shape[1], array.shape[0],
                                           1, gdal.GDT_CFloat32)
    ds.GetRasterBand(1).WriteArray(array)
    ds = None


def audit_isce2_vrt(module, upper, lower, cut, expected, destination):
    """Execute ISCE2's unmodified VRT constructor with explicit ownership boxes."""
    height, width = expected.shape
    for path in (upper, lower):
        ds = gdal.Translate(str(path) + '.vrt', str(path), format='VRT')
        ds = None
    builder = module.VRTConstructor(height, width)
    builder.setDataType('CFLOAT')
    builder.initVRT()
    builder.initBand(band=1)
    for path, first, last in ((upper, 0, cut), (lower, cut, height)):
        burst = SimpleNamespace(numberOfLines=height, numberOfSamples=width,
                                firstValidLine=first, numValidLines=last-first,
                                firstValidSample=0, numValidSamples=width)
        builder.addBurst(burst, str(path), 0, 0, band=1, validOnly=True)
    builder.finishBand()
    builder.finishVRT()
    destination.write_text(builder.vrt)
    actual = gdal.Open(str(destination)).ReadAsArray()
    if not np.array_equal(actual, expected, equal_nan=True):
        raise AssertionError('Hard-cut assembly differs from ISCE2 VRT constructor')
    return True


def nearest_saved_prior(root, lines, samples):
    """Reuse saved integer search guidance at nearest centers; no new projection."""
    with h5py.File(root / 'radar/offsets.h5') as h:
        yy, xx = h['center_line'][:, 0], h['center_sample'][0, :]
        iy = np.abs(lines[..., None] - yy).argmin(axis=-1)
        ix = np.abs(samples[..., None] - xx).argmin(axis=-1)
        gross = np.stack([h[f'prior/integer_px/{axis}'][:][iy, ix]
                          for axis in ('azimuth', 'range')], axis=-1)
        valid = h['prior/valid'][:][iy, ix]
        valid &= (lines >= yy[0]) & (lines <= yy[-1]) & (samples >= xx[0]) & (samples <= xx[-1])
    gross[~valid] = 0  # Invalid cells are excluded, never treated as observations.
    return gross, valid


def selected_field(results, choice, name):
    a, b = results['upper'][name], results['lower'][name]
    return np.where(choice[..., None], b, a) if a.ndim == 3 else np.where(choice, b, a)


def statistics(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return {'count': 0}
    return {'count': int(values.size), 'median': float(np.median(values)),
            'p95_abs': float(np.quantile(np.abs(values), .95)),
            'max_abs': float(np.max(np.abs(values)))}


def compare(results, grid, gross, cuts):
    """Separate splice influence, source disagreement, and unchanged-pixel controls."""
    a, b = results['upper'], results['lower']
    # A map-level highest-correlation selection is only a comparison baseline.
    choice = b['valid'] & (~a['valid'] | (b['correlation_peak'] > a['correlation_peak']))
    baseline = selected_field(results, choice, 'total')
    baseline_valid = a['valid'] | b['valid']
    overlap = a['valid'] & b['valid']
    # Historical inspection thresholds, used for diagnostics only. These do not
    # redefine the workflow's zero-cutoff retention policy or its valid masks.
    qa = {name: r['valid'] & (r['snr'] >= 5) & (r['correlation_peak'] >= .2)
          for name, r in results.items()}
    baseline_qa = np.where(choice, qa['lower'], qa['upper'])
    strong_overlap = qa['upper'] & qa['lower']
    report = {'independent_bursts': {
        'upper_available': int(a['valid'].sum()), 'lower_available': int(b['valid'].sum()),
        'union_available': int(baseline_valid.sum()), 'overlap_available': int(overlap.sum()),
        'overlap_difference_px': {axis: statistics((b['total']-a['total'])[..., k][overlap])
                                  for k, axis in enumerate(('azimuth', 'range'))},
        'diagnostic_quality_upper': int(qa['upper'].sum()),
        'diagnostic_quality_lower': int(qa['lower'].sum()),
        'diagnostic_quality_union': int((qa['upper'] | qa['lower']).sum()),
        'diagnostic_quality_overlap': int(strong_overlap.sum()),
        'diagnostic_quality_overlap_difference_px': {
            axis: statistics((b['total']-a['total'])[..., k][strong_overlap])
            for k, axis in enumerate(('azimuth', 'range'))}}, 'splices': {}}
    arrays = {'baseline_total': baseline, 'baseline_valid': baseline_valid,
              'baseline_lower_source': choice, 'baseline_diagnostic_quality': baseline_qa}
    yy = np.broadcast_to(grid.starts_line[:, None], grid.shape)
    for name, cut in cuts.items():
        r = results[name]
        ref_cross = (yy < cut) & (yy + grid.window[0] > cut)
        secondary_start = yy + gross[..., 0] - grid.search[0]
        sec_cross = (secondary_start < cut) & (secondary_start + grid.window[0] + 2*grid.search[0] > cut)
        crossing = ref_cross | sec_cross
        # Both complete footprints must belong to the SAME burst, not merely
        # avoid individually crossing the cut (large gross shifts can separate them).
        lower = (yy >= cut) & (secondary_start >= cut)
        upper = (yy + grid.window[0] <= cut) & (secondary_start + grid.window[0] + 2*grid.search[0] <= cut)
        pure = lower | upper
        pure_valid = selected_field(results, lower, 'valid')
        control = pure & pure_valid & r['valid']
        expected = selected_field(results, lower, 'total')
        control_difference = r['total'] - expected
        control_exact = bool(np.array_equal(r['total'][control], expected[control], equal_nan=True))
        both = baseline_valid & r['valid']
        difference = r['total'] - baseline
        crossed = crossing & both
        strong = crossed & baseline_qa & qa[name]
        entry = {'cut_local_line': int(cut), 'available': int(r['valid'].sum()),
                 'new_available': int((r['valid'] & ~baseline_valid).sum()),
                 'lost_available': int((~r['valid'] & baseline_valid).sum()),
                 'crossing_comparable': int(crossed.sum()), 'unchanged_source_controls': int(control.sum()),
                 'unchanged_source_offsets_exact': control_exact,
                 'unchanged_source_difference_px': statistics(control_difference[control]),
                 'crossing_difference_px': {axis: statistics(difference[..., k][crossed])
                                           for k, axis in enumerate(('azimuth', 'range'))},
                 'crossing_changed_over_quarter_pixel': int((crossed & (np.max(np.abs(difference), axis=-1) > .25)).sum()),
                 'crossing_correlation': statistics(r['correlation_peak'][crossing & r['valid']])}
        entry['diagnostic_quality'] = {
            'available': int(qa[name].sum()),
            'new': int((qa[name] & ~baseline_qa).sum()),
            'lost': int((~qa[name] & baseline_qa).sum()),
            'crossing_comparable': int(strong.sum()),
            'crossing_changed_over_quarter_pixel': int((strong & (np.max(np.abs(difference), axis=-1) > .25)).sum()),
            'crossing_difference_px': {axis: statistics(difference[..., k][strong])
                                       for k, axis in enumerate(('azimuth', 'range'))}}
        report['splices'][name] = entry
        arrays[f'{name}_crossing'] = crossing
        arrays[f'{name}_difference'] = np.where(both[..., None], difference, np.nan)
        arrays[f'{name}_diagnostic_quality_comparable'] = both & baseline_qa & qa[name]
        # Exceeding one native subpixel in a same-pixels control invalidates the experiment.
        if control.sum() == 0 or np.max(np.abs(control_difference[control])) > 1 / 32:
            raise AssertionError(f'{name}: missing/failed unchanged-source control: {entry}')
    return report, arrays


def plot(output, tiles, mode, quality=False, filename=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['top', 'middle', 'bottom']
    titles = ['Upper owns overlap', 'Midpoint cut', 'Lower owns overlap']
    fig, axes = plt.subplots(len(tiles), 3, figsize=(13, 4.5 if len(tiles) == 1 else 3.3 * len(tiles)), squeeze=False)
    for row, tile in enumerate(tiles):
        with np.load(output / tile['arrays'][mode]) as data:
            for col, name in enumerate(names):
                ax = axes[row, col]
                difference = data[f'{name}_difference'][..., 0].copy()
                if quality:
                    difference[~data[f'{name}_diagnostic_quality_comparable']] = np.nan
                image = ax.pcolormesh(data['center_sample'], data['center_line'], difference,
                                      cmap='RdBu_r', vmin=-.5, vmax=.5,
                                      shading='nearest', rasterized=True)
                ax.axhline(tile['cuts_global'][name], color='black', linewidth=.8)
                ax.invert_yaxis()
                ax.set_facecolor('#e8e8e8')
                if quality and not np.isfinite(difference).any():
                    ax.text(.5, .5, 'No comparable quality matches', transform=ax.transAxes,
                            ha='center', va='center', fontsize=9)
                ax.set(title=f"{titles[col]}; range crop {tile['origin'][1]}",
                       xlabel='Global range sample', ylabel='Global azimuth line')
                ax.title.set_fontsize(10)
    fig.subplots_adjust(top=.74 if len(tiles) == 1 else .87, bottom=.19 if len(tiles) == 1 else .1,
                        hspace=.4, wspace=.3, right=.84)
    colorbar = fig.add_axes([.88, .22 if len(tiles) == 1 else .13, .018, .55 if len(tiles) == 1 else .7])
    fig.colorbar(image, cax=colorbar, label='Azimuth difference (px)')
    label = 'Both SNR >= 5 and correlation >= 0.2; ' if quality else ''
    fig.suptitle(f'{mode}: assembled SLC tracking minus independent tracking\n'
                 f'{label}black = SLC cut; grey = no comparable data\n'
                 'Differences are not ground-truth errors', fontsize=11)
    fig.savefig(output / (filename or f'{mode}-{"quality-" if quality else ""}boundary-differences.png'), dpi=150)
    plt.close(fig)


def analyse_saved(output):
    """Refresh diagnostics from saved native arrays without repeating matching."""
    report = json.loads((output / 'report.json').read_text())
    params = report['matching']
    modes = list(report['tiles'][0]['arrays'])
    for tile in report['tiles']:
        for mode in modes:
            path = output / tile['arrays'][mode]
            with np.load(path) as archive:
                arrays = {name: archive[name] for name in archive.files}
            y0, x0 = tile['origin']
            grid = MatchingGrid((arrays['center_line'][:, 0]-y0-(params['window'][0]-1)/2).astype('int32'),
                                (arrays['center_sample'][0]-x0-(params['window'][1]-1)/2).astype('int32'),
                                tuple(params['window']), tuple(params['search']), tuple(params['stride']))
            results = {name: {field: arrays[f'{name}_{field}'] for field in
                             ('total', 'valid', 'input_valid', 'snr', 'correlation_peak', 'covariance')}
                       for name in ('upper', 'lower', 'top', 'middle', 'bottom')}
            cuts = {name: value-y0 for name, value in tile['cuts_global'].items()}
            tile['results'][mode], updated = compare(results, grid, arrays['gross'], cuts)
            arrays.update(updated)
            np.savez_compressed(path, **arrays)
    report['diagnostic_quality_thresholds'] = {'snr': 5., 'correlation_peak': .2,
                                               'policy': 'Inspection only; production masks unchanged'}
    write_json(output / 'report.json', report)
    for mode in modes:
        plot(output, report['tiles'], mode)
        plot(output, report['tiles'], mode, quality=True)
        focus = [tile for tile in report['tiles'] if tile['results'][mode]['independent_bursts']['diagnostic_quality_union']]
        if focus:
            plot(output, focus, mode, quality=True, filename=f'{mode}-quality-focus.png')


def synthetic_control(output):
    """Known translation across a burst seam, including different complex phases."""
    from scipy.ndimage import gaussian_filter, shift
    from glacier_offsets.config import DEFAULT_MATCHING
    params = DEFAULT_MATCHING | {'window': [32, 64], 'search': [8, 12], 'stride': [16, 64],
                                 'batch': [1, 4], 'zoom': 8, 'backend': 'cpu'}
    shape = 512, 768
    rng = np.random.default_rng(2106)
    amplitude = gaussian_filter(rng.uniform(.1, 1, shape), .6)
    reference = (amplitude * np.exp(1j*rng.normal(0, .2, shape))).astype('complex64')
    translation = np.array([2, -3], dtype='float32')
    secondary = shift(reference, translation, order=0, mode='constant', cval=0)
    upper = [x.copy() for x in (reference, secondary)]
    lower = [reference*np.exp(.7j), secondary*np.exp(1.2j)]
    for array in upper:
        array[320:] = 0
    for array in lower:
        array[:192] = 0
    cuts = {'top': 320, 'middle': 256, 'bottom': 192}
    grid = MatchingGrid(np.arange(16, 464, 16, dtype='int32'), np.arange(16, 688, 64, dtype='int32'),
                        (32, 64), (8, 12), (16, 64))
    gross = np.zeros((*grid.shape, 2), dtype='int32')
    results = {}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='boundary-truth-') as scratch, _native_executor() as worker:
        scratch = Path(scratch)
        for name in ['upper', 'lower', *cuts]:
            pair = upper if name == 'upper' else lower if name == 'lower' else [
                splice(upper[side], lower[side], cuts[name]) for side in (0, 1)]
            paths = [scratch / f'{name}-{side}.tif' for side in (0, 1)]
            for path, array in zip(paths, pair):
                write_slc(path, array)
            ds = [gdal.Open(str(path)) for path in paths]
            valid = valid_windows(*ds, grid, gross)
            ds = None
            results[name] = native_match(*paths, scratch / name, params, {'gross_offsets': {'mode': 'zero'}},
                                         'cpu', _grid=grid, _gross=gross, _valid_input=valid, _worker=worker)
    report, arrays = compare(results, grid, gross, cuts)
    report.update(translation_px=translation.tolist(), matching=params, shape=list(shape),
                  elapsed_wall_seconds=time.perf_counter()-started, truth={})
    passed = True
    for name, result in results.items():
        error = result['total'] - translation
        valid = result['valid']
        crossing = arrays.get(f'{name}_crossing', np.zeros(grid.shape, dtype=bool))
        control = valid & ~crossing
        entry = {'available': int(valid.sum()), 'crossing_available': int((valid & crossing).sum()),
                 'away_error_px': statistics(error[control]), 'crossing_error_px': statistics(error[valid & crossing]),
                 'all_error_px': statistics(error[valid])}
        passed &= bool(control.any() and np.max(np.abs(error[control])) <= .15)
        report['truth'][name] = entry
    report['known_translation_controls_passed'] = passed
    write_json(output / 'synthetic-control.json', report)
    if not passed:
        raise AssertionError('Known translation failed outside the seam influence region')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--bursts', nargs=2, default=['t038_080974_iw2', 't038_080975_iw2'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--range-centers', nargs='+', type=int, default=[4000, 12000, 20000])
    parser.add_argument('--modes', nargs='+', choices=['zero', 'velocity'], default=['zero', 'velocity'])
    parser.add_argument('--isce2-source', type=Path,
                        help='Optional ISCE2 checkout; execute its TopsProc/VRTManager.py for assembly equivalence')
    parser.add_argument('--analyse-only', action='store_true', help='Regenerate diagnostics from saved NPZ arrays')
    parser.add_argument('--synthetic-only', action='store_true', help='Run the known-translation control only')
    args = parser.parse_args()
    gdal.UseExceptions()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.synthetic_only:
        synthetic_control(output)
        return
    if args.analyse_only:
        analyse_saved(output)
        return
    roots = [args.batch.resolve() / key for key in args.bursts]
    manifests = [json.loads((root / 'manifest.json').read_text()) for root in roots]
    if args.bursts[0].rsplit('_', 1)[-1] != args.bursts[1].rsplit('_', 1)[-1]:
        raise ValueError('This experiment is limited to one subswath')
    grids = [m['coreg']['reference_grid'] for m in manifests]
    offset, residual = placement(*grids)
    params = manifests[0]['matching']['matching'].copy()
    if params != manifests[1]['matching']['matching']:
        raise ValueError('Matching settings differ between source bursts')
    params['stride'] = [16, params['stride'][1]]  # Densify only boundary-test centers.
    # Leave room for complete chip/search support and interior controls on both sides.
    height, width = 512, 2048
    origin_y = int((offset + grids[0]['length']) / 2) - height // 2
    grid = MatchingGrid(np.arange(16, height-params['window'][0]-16, 16, dtype='int32'),
                        np.arange(48, width-params['window'][1]-48, params['stride'][1], dtype='int32'),
                        tuple(params['window']), tuple(params['search']), tuple(params['stride']))
    if params['window'] != [64, 256] or params['search'] != [10, 40]:
        raise ValueError('This bounded experiment currently requires the validated 64x256 / 10x40 settings')
    inputs = [root / 'manifest.json' for root in roots] + [root / 'radar/offsets.h5' for root in roots]
    inputs += [Path(m['coreg']['slcs'][s]) for m in manifests for s in ('reference', 'secondary')]
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}
    report = {'status': 'running', 'burst_ids': args.bursts, 'line_offset': offset,
              'placement_residual_px': residual, 'rounding_tolerance_px': .001,
              'nominal_overlap_lines': grids[0]['length'] - offset,
              'matching': params, 'crop_shape': [height, width],
              'provenance': [{'manifest': str(root / 'manifest.json'), 'coreg_identity': m['coreg_identity'],
                              'match_identity': m['match_identity']} for root, m in zip(roots, manifests)],
              'limitations': ['Same IW2 boundary and selected range strips only; not full-AOI validation.',
                              'Hard cuts use measured rectangular crop support, not original annotation valid boxes.',
                              'Velocity mode uses nearest saved integer priors at new centers; no fresh NetCDF sampling/projection.',
                              'No independent displacement truth; differences do not establish accuracy.',
                              'Native settings retained except azimuth test stride 16 instead of 32.',
                              'No production workflow, SLC phase averaging, geocoding or bias correction added.'],
              'tiles': []}
    isce2_vrt = None
    if args.isce2_source:
        source = args.isce2_source.resolve() / 'components/isceobj/TopsProc/VRTManager.py'
        spec = importlib.util.spec_from_file_location('isce2_vrt_audit', source)
        isce2_vrt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(isce2_vrt)
        report['isce2_vrt_source'] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                     'scope': 'Unmodified VRTConstructor with measured crop ownership boxes; not full topsApp'}
    write_json(output / 'report.json', report)
    started = time.perf_counter()
    with _native_executor() as worker, tempfile.TemporaryDirectory(prefix='boundary-') as scratch:
        scratch = Path(scratch)
        for tile_index, center in enumerate(args.range_centers):
            origin_x = ((center - width // 2) // 128) * 128
            origin = origin_y, origin_x
            sources = [[read_crop(m['coreg']['slcs'][side], origin, (height, width), dy)
                        for side in ('reference', 'secondary')]
                       for m, dy in zip(manifests, (0, offset))]
            supports = [support_rows(*pair) for pair in sources]
            lo, hi = supports[1][0], supports[0][1]
            if lo >= hi:
                raise ValueError('No common rectangular valid support for an SLC cut')
            cuts = {'top': hi, 'middle': (lo + hi) // 2, 'bottom': lo}
            tile = {'origin': list(origin), 'support_local_lines': supports,
                    'cuts_global': {k: v + origin_y for k, v in cuts.items()}, 'results': {}, 'arrays': {},
                    'isce2_vrt_exact': {}}
            lines, samples = grid.centers
            lines, samples = lines + origin_y, samples + origin_x
            if 'velocity' in args.modes:
                priors = [nearest_saved_prior(root, lines-dy, samples) for root, dy in zip(roots, (0, offset))]
                lower = lines >= tile['cuts_global']['middle']
                velocity_gross = np.where(lower[..., None], priors[1][0], priors[0][0])
                velocity_valid = np.where(lower, priors[1][1], priors[0][1])
            files = {}
            for name in ['upper', 'lower', *cuts]:
                pair = sources[0] if name == 'upper' else sources[1] if name == 'lower' else [
                    splice(sources[0][side], sources[1][side], cuts[name]) for side in (0, 1)]
                files[name] = [scratch / f'{tile_index}-{name}-{side}.tif' for side in (0, 1)]
                for path, array in zip(files[name], pair):
                    write_slc(path, array)
                if isce2_vrt is not None and name in cuts:
                    for side in (0, 1):
                        tile['isce2_vrt_exact'][f'{name}-{side}'] = audit_isce2_vrt(
                            isce2_vrt, files['upper'][side], files['lower'][side], cuts[name], pair[side],
                            scratch / f'isce2-{tile_index}-{name}-{side}.vrt')
            del sources, pair
            for mode in args.modes:
                gross = velocity_gross if mode == 'velocity' else np.zeros((*grid.shape, 2), dtype='int32')
                eligible = velocity_valid if mode == 'velocity' else np.ones(grid.shape, dtype=bool)
                results = {}
                for name, paths in files.items():
                    ds = [gdal.Open(str(p)) for p in paths]
                    valid = valid_windows(*ds, grid, gross) & eligible
                    ds = None
                    print(f'tile {tile_index+1}/{len(args.range_centers)} {mode} {name}: {valid.sum()} input-valid', flush=True)
                    results[name] = native_match(*paths, scratch / f'native-{tile_index}-{mode}-{name}',
                                                params, {'gross_offsets': {'mode': 'zero'}}, 'cpu',
                                                _grid=grid, _gross=gross, _valid_input=valid, _worker=worker)
                comparison, arrays = compare(results, grid, gross, cuts)
                tile['results'][mode] = comparison
                arrays.update(center_line=lines, center_sample=samples, gross=gross, prior_valid=eligible)
                for name, result in results.items():
                    for field in ('total', 'valid', 'input_valid', 'snr', 'correlation_peak', 'covariance'):
                        arrays[f'{name}_{field}'] = result[field]
                name = f'tile-{tile_index}-{mode}.npz'
                np.savez_compressed(output / name, **arrays)
                tile['arrays'][mode] = name
            report['tiles'].append(tile)
            write_json(output / 'report.json', report)
            for pair in files.values():
                for path in pair:
                    path.unlink()
    report['sources_unchanged'] = all((p.stat().st_size, p.stat().st_mtime_ns) == before[str(p)] for p in inputs)
    if not report['sources_unchanged']:
        raise AssertionError('Source products changed during experiment')
    report['elapsed_wall_seconds'] = time.perf_counter() - started
    report['status'] = 'experiment_completed; scientific adoption not established'
    write_json(output / 'report.json', report)
    analyse_saved(output)
    print(json.dumps({'report': str(output / 'report.json'), 'seconds': report['elapsed_wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()

"""Derived comparison maps; all writes are confined to the experiment directory."""
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal
from pyproj import Transformer

from glacier_offsets.geocode import center_geometry, make_mapper, write_geotiff, geocoding_specification
from glacier_offsets.products import write_json, identity
from glacier_offsets.stack_status import matching_status
from glacier_offsets.swath_ownership import pure_swath_footprints
from glacier_offsets.swath_experiment_io import Inputs, read_json, load_npz, save_npz

MAPPING_POLICY = 'production-nearest-retain-raw-unsupported-v1'
DISPLAY_LIMITS = {'displacement': 5., 'velocity': 5.}


def corrected_preview(total, bias, available):
    """Retain original measurements wherever the correction is unsupported."""
    applied = np.where(available, bias, 0.)
    return total-applied, applied


def verify_raw_map(inputs, root, mapper, total, scale):
    path = root/'geo/azimuth_displacement_m.tif'
    inputs.file(path)
    ds = gdal.Open(str(path))
    np.testing.assert_array_equal(mapper.map(total*scale), ds.ReadAsArray())


def write_previews(out, arrays, predictions, uncertainty, metadata, mapper, grid,
                   owner_swath, availability, pair, *, save_radar=True):
    """One native-valid production mapping for raw and all corrected products."""
    total, scale, interval = (arrays[k] for k in
        ('raw_total_azimuth_px', 'azimuth_scale_m_per_px', 'interval_seconds'))
    metadata = metadata | {'mapping_policy': MAPPING_POLICY,
        'unsupported_policy': 'retain raw observations; correction_available=0',
        'display': 'no overlap outlines, smoothing, filling, or artificial masked strip'}
    products, mapped = [], {}
    for quantity, conversion, units in (('displacement', scale, 'm'), ('velocity', scale/(interval/86400), 'm/day')):
        mapped[f'raw_{quantity}'] = mapper.map(total*conversion)
        path = out/f'raw_azimuth_{quantity}.tif'
        write_geotiff(path, mapped[f'raw_{quantity}'], mapper.transform, grid['epsg'], metadata | {'units': units})
        products.append(path)
        for name, delta in predictions.items():
            bias = symmetric_bias(delta, owner_swath)
            corrected, applied = corrected_preview(total, bias, availability[name])
            # Radar correction-only fields remain masked by scientific availability.
            arrays[f'{name}_bias_to_subtract_px'] = np.where(availability[name], bias, np.nan)
            arrays[f'{name}_corrected_total_azimuth_px'] = np.where(availability[name], corrected, np.nan)
            arrays[f'{name}_correction_available'] = availability[name]
            for label, value in (('corrected', corrected), ('bias_to_subtract', applied)):
                field = mapper.map(value*conversion)
                path = out/f'{name}_{label}_azimuth_{quantity}.tif'
                write_geotiff(path, field, mapper.transform, grid['epsg'], metadata | {
                    'units': units, 'quantity': label, 'model': name,
                    'prediction_available': bool(np.isfinite(delta)), 'relative_contrast_px': delta})
                products.append(path)
                if label == 'corrected':
                    mapped[f'{name}_{quantity}'] = field
            if uncertainty[name] is not None:
                arrays[f'{name}_correction_std_px'] = np.where(availability[name], uncertainty[name]/2, np.nan)
    for name, available in availability.items():
        path = out/f'{name}_correction_available.tif'
        write_geotiff(path, mapper.map(available.astype('float32')), mapper.transform,
                      grid['epsg'], metadata | {'units': '0/1', 'model': name})
        products.append(path)
    for name in ('comparison_available', 'cross_swath_mixed', 'same_swath_burst_join', 'measured_overlap'):
        path = out/f'{name}.tif'
        write_geotiff(path, mapper.map(arrays[name].astype('float32')), mapper.transform,
                      grid['epsg'], metadata | {'units': '0/1'})
        products.append(path)
    radar = out/'derived_radar.npz'
    if save_radar:
        save_npz(radar, **arrays)
    products.append(radar)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x0, dx, _, y0, _, dy = mapper.transform
    shape = mapped['raw_displacement'].shape
    extent = [x0, x0+shape[1]*dx, y0+shape[0]*dy, y0]
    footer = 'Mixed-swath and unsupported windows retain raw values; correction availability is stored separately.'
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), layout='constrained')
    for row, quantity in enumerate(('displacement', 'velocity')):
        for ax, name in zip(axes[row], ('raw', *predictions)):
            limit = DISPLAY_LIMITS[quantity]
            im = ax.imshow(mapped[f'{name}_{quantity}'], extent=extent, origin='upper',
                           interpolation='nearest', cmap='RdBu_r', vmin=-limit, vmax=limit)
            label = {'raw': 'Raw', 'physical': 'Physical', 'spacecraft': 'Spacecraft',
                     'per_acquisition': 'Per acquisition'}[name]
            status = ('Unavailable; raw retained' if name != 'raw' and not np.isfinite(predictions[name])
                      else 'Unvalidated candidate' if name == 'physical' else '')
            ax.set_title(f'{label}: {quantity}\n{status}', fontsize=10)
            ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
            fig.colorbar(im, ax=ax, shrink=.7, label='m' if quantity == 'displacement' else 'm/day')
    fig.suptitle(pair+' — relative constant-swath previews')
    fig.supxlabel(footer, fontsize=9)
    image = out/'comparison.png'; fig.savefig(image, dpi=140); plt.close(fig); products.append(image)
    for filename, names, titles, size in (
        ('corrected_clean.png', ('spacecraft',), ('Spacecraft-corrected azimuth displacement',), (10, 8)),
        ('comparison_clean.png', ('raw', 'spacecraft'), ('Raw', 'Spacecraft corrected'), (16, 7))):
        fig, axes = plt.subplots(1, len(names), figsize=size, layout='constrained', squeeze=False)
        for ax, name, title in zip(axes[0], names, titles):
            im = ax.imshow(mapped[f'{name}_displacement'], extent=extent, origin='upper',
                           interpolation='nearest', cmap='RdBu_r', vmin=-5, vmax=5)
            ax.set_title(title+(' — unavailable; raw retained' if name != 'raw' and not np.isfinite(predictions[name]) else ''))
            ax.set_xlabel(f"Easting (m, EPSG:{grid['epsg']})")
            ax.set_ylabel(f"Northing (m, EPSG:{grid['epsg']})")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.9, label='Azimuth displacement (m)')
        fig.suptitle(pair.replace('_', ' to ')); fig.supxlabel(footer, fontsize=9)
        image = out/filename; fig.savefig(image, dpi=160); plt.close(fig); products.append(image)
    return products


def symmetric_bias(delta, swath):
    return np.where(np.asarray(swath) == 1, -delta/2, np.where(np.asarray(swath) == 2, delta/2, np.nan))


def preview_predictions(primary, pair, missions=None):
    rows = primary['rows']
    selected = [r for r in rows if r['pair'] == pair]
    physical = [np.mean(r['physical']['combined']) for r in selected]
    predictions = {'physical': float(np.mean(physical)) if physical and np.isfinite(physical).all() else np.nan}
    uncertainty = {'physical': None}
    fold = next((f for f in primary['folds'] if f['kind'] == 'leave_link_out' and f['label'] == pair), None)
    for model in ('spacecraft', 'per_acquisition'):
        values = np.array(fold['models'][model]['prediction_px']) if fold else np.array([])
        std = np.array(fold['models'][model]['prediction_std_px']) if fold else np.array([])
        predictions[model] = float(values.mean()) if values.size and np.isfinite(values).all() else np.nan
        uncertainty[model] = float(std.mean()) if std.size and np.isfinite(std).all() else None
    if fold is None and not selected:
        # An unmeasured link contributes no training rows, so the all-data fit
        # already excludes it. Prediction still requires constrained endpoints.
        a, b = pair.split('_')
        model = primary['models']['per_acquisition']
        dates = model['parameter_names']
        graph = primary['graph']
        if (a in dates and b in dates and a not in graph['isolated_dates'] and
                b not in graph['isolated_dates'] and graph['labels'][a] == graph['labels'][b]):
            i, j = dates.index(a), dates.index(b)
            predictions['per_acquisition'] = model['parameters'][j]-model['parameters'][i]
            if model['covariance'] is not None:
                cov = np.asarray(model['covariance'])
                uncertainty['per_acquisition'] = float(np.sqrt(max(0, cov[j, j]+cov[i, i]-2*cov[i, j])))
        spacecraft = primary['models']['spacecraft']
        if missions and spacecraft['rank'] == 1:
            contrast = int(missions[b] == 'S1B')-int(missions[a] == 'S1B')
            predictions['spacecraft'] = contrast*spacecraft['parameters'][0]
            if spacecraft['covariance'] is not None:
                uncertainty['spacecraft'] = abs(contrast)*float(np.sqrt(max(0, spacecraft['covariance'][0][0])))
    return predictions, uncertainty


def preview_link(experiment, link, primary):
    root = experiment.stack/'pairs'/link['pair']
    experiment.inputs.file(root/'manifest.json')
    manifest = read_json(root/'manifest.json')
    status = matching_status(experiment.cfg, manifest, root)
    if status['status'] != 'complete' or manifest['stages'].get('match') != 'complete':
        raise ValueError(f"Production match unavailable: {status['reason']}")
    if (manifest['stages'].get('geocode') != 'complete' or
            manifest.get('geocode_identity') != identity(geocoding_specification(experiment.cfg, manifest))):
        raise ValueError('Production geocoding is incomplete or stale')
    if manifest.get('stack_pair', {}).get('geometry_identity') != experiment.frame['geometry_identity']:
        raise ValueError('Production pair has a different geometry pin')
    digest = experiment.inputs.file(root/'radar/offsets.h5')
    predictions, uncertainty = preview_predictions(primary, link['pair'],
        {day: record['mission'] for day, record in experiment.inventory.items()})
    implementation = experiment.implementation('swath_experiment_previews.py', 'geocode.py', 'swath_experiment_io.py')
    implementation[str(Path(__file__).resolve())] = experiment.inputs.hashes(__file__)
    spec = {'experiment': experiment.key, 'raw_offsets_sha256': digest, 'match_identity': manifest['match_identity'],
            'fit_sha256': experiment.inputs.hashes(experiment.output/'analysis/primary.json'),
            'implementation': implementation,
            'geocoding': experiment.cfg['geocoding'], 'target_grid': manifest['map_grid'],
            'mapping_policy': MAPPING_POLICY, 'display_limits': DISPLAY_LIMITS}
    out = experiment.output/'previews'/link['pair']
    if experiment.current(out/'report.json', spec):
        return read_json(out/'report.json')
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(root/'radar/offsets.h5', 'r') as h:
        line, sample = h['center_line'][:], h['center_sample'][:]
        total = h['total/azimuth_px'][:]
        total_range = h['total/range_px'][:]
        base = h['input_valid'][:] & ((h['rejection_flags'][:] & 7) == 0) & np.isfinite(total) & np.isfinite(total_range)
        owner, interval, seam = h['source_burst_index'][:], h['interval_seconds'][:], h['seam_crossing'][:]
        native_covariance = h['covariance'][:]
        metadata = json.loads(h.attrs['metadata_json'])
        if metadata['matching'] != manifest['matching']['matching']:
            raise ValueError('HDF5 and manifest matching specifications differ')
    lon, lat, height, scale, geometry_valid = center_geometry(manifest['coreg'], line, sample)
    base &= geometry_valid & np.isfinite(interval) & (interval > 0)
    pure = pure_swath_footprints(manifest['coreg'], line, sample, experiment.cfg['matching'],
                                 manifest['assembly']['source_burst_ids'])
    lookup = np.array([0]+[int(s[-1]) for s in manifest['assembly']['source_burst_ids']], dtype='uint8')
    owner_swath = lookup[owner]
    cross_swath = (pure == 0) & (owner > 0)
    availability = {name: base & (pure == owner_swath) & (pure > 0) & np.isfinite(delta)
                    for name, delta in predictions.items()}
    common = np.logical_and.reduce(list(availability.values()))
    x, y = Transformer.from_crs(4326, experiment.cfg['geocoding']['epsg'], always_xy=True).transform(lon, lat)
    grid = manifest['map_grid']
    raw_mapper = make_mapper(x, y, base, experiment.cfg['geocoding']['spacing_m'], 'nearest', target_grid=grid)
    arrays = {'raw_total_azimuth_px': total, 'source_burst_index': owner, 'interval_seconds': interval,
              'azimuth_scale_m_per_px': scale, 'cross_swath_mixed': cross_swath,
              'same_swath_burst_join': seam & ~cross_swath, 'comparison_available': common,
              'native_covariance': native_covariance}
    overlap = np.zeros(total.shape, dtype=bool)
    observed = load_npz(experiment.output/'measurements'/link['pair']/'observations.npz')
    # Exact production centers, with no densification or nearest-good substitution.
    ri = np.searchsorted(line[:, 0], observed['center_line'][:, 0])
    ci = np.searchsorted(sample[0], observed['center_sample'][0])
    np.testing.assert_array_equal(line[ri, 0], observed['center_line'][:, 0])
    np.testing.assert_array_equal(sample[0, ci], observed['center_sample'][0])
    overlap[np.ix_(ri, ci)] = observed['dual_support']
    arrays['measured_overlap'] = overlap
    metadata = {'experiment_identity': experiment.key, 'base_match_identity': manifest['match_identity'],
                'axis': 'later-minus-earlier azimuth', 'allocation': 'IW1=-delta/2, IW2=+delta/2; subtract',
                'spatial_assumption': 'constant relative swath contrast outside measured overlap',
                'physical_status': 'unvalidated candidate; scalar overlap-mean contrast',
                'prediction': 'leave-one-link-out empirical; unfitted physical',
                'unsupported_policy': 'retain raw observations; correction availability stored separately',
                'display': 'production nearest-source mapping; no overlap outline or artificial strip',
                'uncertainty': 'native covariance and correction dependence sensitivity stored separately; no combined accuracy uncertainty'}
    verify_raw_map(experiment.inputs, root, raw_mapper, total, scale)
    mapper = raw_mapper  # Preserve the production source selection for every panel.
    products = write_previews(out, arrays, predictions, uncertainty, metadata, mapper,
                             grid, owner_swath, availability, link['pair'])
    return experiment.finish(out/'report.json', spec, {'status': 'complete', 'predictions_px': predictions,
        'correction_std_px': uncertainty, 'common_source_centers': int(common.sum()),
        'available_source_centers': {k: int(v.sum()) for k, v in availability.items()},
        'cross_swath_mixed_n': int((base & cross_swath).sum()),
        'metadata': metadata, 'raw_input_sha256': digest}, products)


def refresh_preview_link(experiment, link, primary):
    """Regenerate maps from verified retained radar arrays, without SLC reads."""
    out = experiment.output/'previews'/link['pair']
    old_inputs = Inputs()
    old_inputs.file(out/'report.json')
    previous = read_json(out/'report.json')
    if previous.get('status') != 'complete' or previous.get('identity') != identity(previous['specification']):
        raise ValueError('A complete retained preview is required')
    root = experiment.stack/'pairs'/link['pair']
    experiment.inputs.file(root/'manifest.json')
    manifest = read_json(root/'manifest.json')
    if (matching_status(experiment.cfg, manifest, root)['status'] != 'complete' or
            manifest['stages'].get('geocode') != 'complete' or
            manifest.get('geocode_identity') != identity(geocoding_specification(experiment.cfg, manifest))):
        raise ValueError('Production matching/geocoding is incomplete or stale')
    if manifest['stack_pair']['geometry_identity'] != experiment.frame['geometry_identity']:
        raise ValueError('Production pair has a different geometry pin')
    old = previous['specification']
    expected = {'experiment': experiment.key, 'match_identity': manifest['match_identity'],
        'raw_offsets_sha256': experiment.inputs.file(root/'radar/offsets.h5'),
        'fit_sha256': experiment.inputs.file(experiment.output/'analysis/primary.json'),
        'geocoding': experiment.cfg['geocoding'], 'target_grid': manifest['map_grid']}
    if any(old.get(k) != v for k, v in expected.items()):
        raise ValueError('Retained preview inputs changed; run ordinary previews first')
    implementation = experiment.implementation('swath_experiment_previews.py', 'geocode.py', 'swath_experiment_io.py')
    implementation[str(Path(__file__).resolve())] = experiment.inputs.hashes(__file__)
    spec = expected | {'implementation': implementation, 'mapping_policy': MAPPING_POLICY,
                       'display_limits': DISPLAY_LIMITS}
    if experiment.current(out/'report.json', spec):
        return previous
    radar = out/'derived_radar.npz'
    experiment.inputs.file(radar, previous['products'][str(radar)])
    arrays = load_npz(radar)
    cache = root/'geo/center_geometry.h5'
    experiment.inputs.file(cache)
    with h5py.File(cache, 'r') as h:
        if json.loads(h.attrs['metadata_json'])['match_identity'] != manifest['match_identity']:
            raise ValueError('Retained center geometry belongs to a different match')
        lon, lat, geometry_valid = h['longitude_deg'][:], h['latitude_deg'][:], h['geometry_valid'][:]
        np.testing.assert_array_equal(arrays['azimuth_scale_m_per_px'], h['azimuth_ground_spacing_m_per_slc_line'][:])
        with h5py.File(root/'radar/offsets.h5', 'r') as native:
            for saved, source in (('raw_total_azimuth_px', 'total/azimuth_px'),
                                   ('source_burst_index', 'source_burst_index'),
                                   ('interval_seconds', 'interval_seconds')):
                np.testing.assert_array_equal(arrays[saved], native[source][:])
            for name in ('center_line', 'center_sample'):
                np.testing.assert_array_equal(h[name][:], native[name][:])
            base = native['input_valid'][:] & ((native['rejection_flags'][:] & 7) == 0)
            base &= np.isfinite(native['total/range_px'][:]) & np.isfinite(arrays['raw_total_azimuth_px'])
    interval = arrays['interval_seconds']
    base &= geometry_valid & np.isfinite(interval) & (interval > 0)
    grid = manifest['map_grid']
    x, y = Transformer.from_crs(4326, grid['epsg'], always_xy=True).transform(lon, lat)
    mapper = make_mapper(x, y, base, grid['transform'][1], 'nearest', target_grid=grid)
    verify_raw_map(experiment.inputs, root, mapper, arrays['raw_total_azimuth_px'], arrays['azimuth_scale_m_per_px'])
    predictions, uncertainty = preview_predictions(primary, link['pair'],
        {day: record['mission'] for day, record in experiment.inventory.items()})
    for name, value in predictions.items():
        np.testing.assert_equal(value, previous['predictions_px'][name])
    lookup = np.array([0]+[int(s[-1]) for s in manifest['assembly']['source_burst_ids']], dtype='uint8')
    owner_swath = lookup[arrays['source_burst_index']]
    available = {name: arrays[f'{name}_correction_available'] for name in predictions}
    metadata = previous['metadata'] | {'mapping_policy': MAPPING_POLICY,
        'unsupported_policy': 'retain raw observations; correction availability stored separately',
        'display': 'production nearest-source mapping; no overlap outline or artificial strip'}
    old_inputs.unchanged()  # Only this derived completion record is replaced below.
    products = write_previews(out, arrays, predictions, uncertainty, metadata, mapper,
                             grid, owner_swath, available, link['pair'], save_radar=False)
    report = {key: previous[key] for key in ('predictions_px', 'correction_std_px',
        'common_source_centers', 'available_source_centers', 'cross_swath_mixed_n', 'raw_input_sha256')}
    return experiment.finish(out/'report.json', spec, report | {'status': 'complete', 'metadata': metadata,
        'render_source': 'checksum-verified retained radar arrays and production center geometry',
        'raw_mapping_equals_production': True}, products)


def previews(experiment, retained=False):
    if retained:
        # Plot refresh consumes frozen numerical results, not source SLCs or
        # current fitting/measurement code. Verify exactly the retained inputs.
        experiment.inputs.file(experiment.output/'fit.json')
        fitted = read_json(experiment.output/'fit.json')
        if (fitted.get('status') != 'complete' or fitted['specification']['experiment'] != experiment.key or
                fitted['identity'] != identity(fitted['specification'])):
            raise ValueError('A complete matching frozen fit is required')
        path = experiment.output/'analysis/primary.json'
        experiment.inputs.file(path, fitted['products'][str(path)])
    else:
        experiment.checked_report(experiment.output/'fit.json')
    primary = read_json(experiment.output/'analysis/primary.json')
    if not retained:
        experiment.verify_geometry()
    results = {}
    for link in experiment.links:
        if link['step'] != 1:
            continue
        try:
            logging.info('Preview %s', link['pair'])
            results[link['pair']] = (refresh_preview_link if retained else preview_link)(experiment, link, primary)
        except (ValueError, OSError, KeyError) as exc:
            results[link['pair']] = {'status': 'unavailable', 'error': str(exc)}
            logging.error('Preview %s: %s', link['pair'], exc)
        write_json(experiment.output/'preview_status.json', {'status': 'running', 'links': results})
    result = {'status': 'complete' if all(r['status'] == 'complete' for r in results.values()) else 'incomplete',
              'links': results}
    write_json(experiment.output/'preview_status.json', result)
    return result

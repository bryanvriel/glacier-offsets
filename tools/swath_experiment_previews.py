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
from tools.swath_experiment_io import read_json, load_npz, save_npz


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
            'geocoding': experiment.cfg['geocoding'], 'target_grid': manifest['map_grid']}
    out = experiment.output/'previews'/link['pair']
    if experiment.current(out/'report.json', spec):
        return read_json(out/'report.json')
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(root/'radar/offsets.h5', 'r') as h:
        line, sample = h['center_line'][:], h['center_sample'][:]
        total = h['total/azimuth_px'][:]
        base = h['input_valid'][:] & ((h['rejection_flags'][:] & 7) == 0) & np.isfinite(total)
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
    mapper = make_mapper(x, y, common, experiment.cfg['geocoding']['spacing_m'], 'nearest', target_grid=grid)
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
                'uncertainty': 'native covariance and correction dependence sensitivity stored separately; no combined accuracy uncertainty'}
    products, mapped = [], {}
    for quantity, conversion, units in (('displacement', scale, 'm'), ('velocity', scale/(interval/86400), 'm/day')):
        raw_mapped = raw_mapper.map(total*conversion)
        path = out/f'raw_azimuth_{quantity}.tif'
        write_geotiff(path, raw_mapped, mapper.transform, grid['epsg'], metadata | {'units': units})
        products.append(path)
        mapped[f'raw_{quantity}'] = mapper.map(total*conversion)
        for name, delta in predictions.items():
            bias = symmetric_bias(delta, owner_swath)
            corrected = np.where(availability[name], total-bias, np.nan)
            arrays[f'{name}_bias_to_subtract_px'] = np.where(availability[name], bias, np.nan)
            arrays[f'{name}_corrected_total_azimuth_px'] = corrected
            arrays[f'{name}_correction_available'] = availability[name]
            for label, value in (('corrected', corrected), ('bias_to_subtract', bias)):
                field = mapper.map(value*conversion)
                path = out/f'{name}_{label}_azimuth_{quantity}.tif'
                write_geotiff(path, field, mapper.transform, grid['epsg'], metadata | {'units': units, 'quantity': label})
                products.append(path)
                if label == 'corrected':
                    mapped[f'{name}_{quantity}'] = field
            std = uncertainty[name]
            if std is not None:
                arrays[f'{name}_correction_std_px'] = np.where(availability[name], std/2, np.nan)
    for name, field in (('comparison_available', common), ('cross_swath_mixed', cross_swath),
                        ('same_swath_burst_join', seam & ~cross_swath), ('measured_overlap', overlap)):
        path = out/f'{name}.tif'
        write_geotiff(path, raw_mapper.map(field.astype('float32')), mapper.transform, grid['epsg'], metadata | {'units': '0/1'})
        products.append(path)
    radar = out/'derived_radar.npz'
    save_npz(radar, **arrays)
    products.append(radar)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(17, 9), layout='constrained')
    for row, quantity in enumerate(('displacement', 'velocity')):
        values = [mapped[f'{name}_{quantity}'] for name in ('raw', *predictions)]
        finite = np.concatenate([v[np.isfinite(v)] for v in values])
        limit = max(float(np.quantile(abs(finite), .98)), 1.e-6) if finite.size else 1
        for ax, name, value in zip(axes[row], ('raw', *predictions), values):
            im = ax.imshow(value, cmap='RdBu_r', vmin=-limit, vmax=limit)
            support = raw_mapper.map(overlap.astype(float))
            if np.any(support > 0):
                ax.contour(support > 0, levels=[.5], colors='black', linewidths=.4)
            unavailable = name != 'raw' and not np.isfinite(predictions[name])
            ax.set_title(f'{name}: {quantity}'+(' — unavailable' if unavailable else
                         ' (candidate)' if name == 'physical' else ''))
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, shrink=.7, label='m' if quantity == 'displacement' else 'm/day')
    fig.suptitle(link['pair']+f' — {int(common.sum())} common centers; relative constant-swath previews')
    image = out/'comparison.png'
    fig.savefig(image, dpi=140)
    plt.close(fig)
    products.append(image)
    return experiment.finish(out/'report.json', spec, {'status': 'complete', 'predictions_px': predictions,
        'correction_std_px': uncertainty, 'common_source_centers': int(common.sum()),
        'available_source_centers': {k: int(v.sum()) for k, v in availability.items()},
        'cross_swath_mixed_n': int((base & cross_swath).sum()),
        'metadata': metadata, 'raw_input_sha256': digest}, products)


def previews(experiment):
    experiment.checked_report(experiment.output/'fit.json')
    primary = read_json(experiment.output/'analysis/primary.json')
    experiment.verify_geometry()
    results = {}
    for link in experiment.links:
        if link['step'] != 1:
            continue
        try:
            results[link['pair']] = preview_link(experiment, link, primary)
        except (ValueError, OSError, KeyError) as exc:
            results[link['pair']] = {'status': 'unavailable', 'error': str(exc)}
            logging.error('Preview %s: %s', link['pair'], exc)
        write_json(experiment.output/'preview_status.json', {'status': 'running', 'links': results})
    result = {'status': 'complete' if all(r['status'] == 'complete' for r in results.values()) else 'incomplete',
              'links': results}
    write_json(experiment.output/'preview_status.json', result)
    return result

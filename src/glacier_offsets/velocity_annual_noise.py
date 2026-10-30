"""Annual range/azimuth noise covariances: supplied files or reviewed controls.

Supplied files must identify their calendar years (or receive explicit YAML
labels, for the legacy unlabeled ``covariance_by_year`` list) and contain
finite, symmetric, positive-definite 2x2 covariances. Controls mode selects
unique sources spatially from reviewed control regions, never from fitted
harmonic amplitudes, then applies the existing order-aware, acquisition-purged
residuals and spatially balanced robust estimator to each year's full control
record. Every observed year needs its own covariance; there is no pooled or
previous-year fallback.
"""
import json
import logging
from pathlib import Path

import h5py
import numpy as np

from .velocity_cache import Artifact, identity_digest
from .velocity_fit import calibrate_noise
from .velocity_io import digest
from .velocity_pixel_general import load_field
from .velocity_pixel_runtime import module_fingerprints
from .velocity_systematics import order_residual, order_screen


NOISE = 'noise.json'
SCHEMA = 'glacier-offsets-annual-noise-v1'
ESTIMATOR_MODULES = ('velocity_annual_noise', 'velocity_fit', 'velocity_systematics', 'velocity_quality',
                     'velocity_pixel_general')


def check_covariance(matrix, label):
    matrix = np.asarray(matrix, float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError(f'{label}: covariance must be a finite 2x2 range/azimuth matrix')
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f'{label}: covariance must be symmetric')
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f'{label}: covariance must be positive definite') from error
    return matrix


def diagnostics(matrix):
    sigma = np.sqrt(np.diag(matrix))
    eigen = np.linalg.eigvalsh(matrix)
    return {'sigma_range_m': float(sigma[0]), 'sigma_azimuth_m': float(sigma[1]),
            'correlation': float(matrix[0, 1]/np.prod(sigma)), 'eigenvalues_m2': eigen.tolist(),
            'condition': float(eigen[-1]/eigen[0])}


def read_supplied(entries):
    """Map calendar years to supplied covariances, rejecting ambiguity."""
    by_year = {}
    sources = []
    for entry in entries:
        path = Path(entry['path'])
        if not path.is_file():
            raise ValueError(f'Missing supplied noise file: {path}')
        record = json.loads(path.read_text())
        labels = entry.get('years')
        if 'covariance_ra_m2_by_year' in record:
            labeled = {int(year): value for year, value in record['covariance_ra_m2_by_year'].items()}
            if labels is not None:
                absent = sorted(set(labels)-labeled.keys())
                if absent:
                    raise ValueError(f'{path} has no covariance for YAML-selected year(s) {absent}')
                labeled = {year: labeled[year] for year in labels}
        elif 'covariance_by_year' in record:
            if labels is None:
                raise ValueError(f'{path} has unlabeled covariance_by_year; give its calendar years in noise.files[].years')
            values = record['covariance_by_year']
            if len(values) != len(labels):
                raise ValueError(f'{path} has {len(values)} covariances but {len(labels)} YAML year labels')
            labeled = dict(zip(labels, values))
        else:
            raise ValueError(f'{path}: expected covariance_ra_m2_by_year or a labeled covariance_by_year list')
        for year, value in labeled.items():
            if year in by_year:
                raise ValueError(f'Year {year} has more than one supplied covariance')
            by_year[year] = check_covariance(value, f'{path} year {year}')
        sources.append({'path': str(path), 'sha256': digest(path), 'years': sorted(labeled)})
    return by_year, sources


def control_regions(regions_file):
    record = json.loads(Path(regions_file).read_text())
    regions = [r for r in record.get('regions', []) if r.get('role') == 'control' and r.get('reviewed') is True]
    if not regions:
        raise ValueError(f'{regions_file} has no reviewed control regions (role: control, reviewed: true)')
    for region in regions:
        bounds = region.get('bounds')
        if not isinstance(bounds, list) or len(bounds) != 4 or not (bounds[0] < bounds[2] and bounds[1] < bounds[3]):
            raise ValueError(f'Control region {region.get("id")} needs bounds [west, south, east, north]')
    return int(record.get('EPSG', 0)) or None, regions


def select_controls(observations, regions_file, block_size):
    """Unique sources inside reviewed control bounds, by physical chip position."""
    from pyproj import Transformer
    epsg, regions = control_regions(regions_file)
    with h5py.File(observations, 'r') as h:
        epsg = epsg or int(h.attrs['EPSG'])
        x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(h['longitude'][:], h['latitude'][:])
        source_id = h['source_id'][:]
    taken = np.zeros(len(x), bool)
    selected = []
    for region in regions:
        west, south, east, north = region['bounds']
        index = np.flatnonzero((x >= west) & (x <= east) & (y >= south) & (y <= north) & ~taken)
        taken[index] = True
        groups = np.array([f'{region["id"]}:{int(np.floor(a/block_size))}:{int(np.floor(b/block_size))}'
                           for a, b in zip(x[index], y[index])], dtype=object)
        selected.append({'id': region['id'], 'index': index, 'source_id': source_id[index], 'groups': groups})
    if not taken.any():
        raise ValueError('No observation source lies inside the reviewed control regions')
    return epsg, selected


def estimate_controls(observations, solid_earth, years, selected, screening):
    """Full-year, order-aware held-out residual covariance for each year."""
    by_year = {}
    report = []
    for k, year in enumerate(years):
        residual, snr, peak, groups = [], [], [], []
        training = set()
        per_region = []
        for region in selected:
            index = region['index']
            if not len(index):
                per_region.append({'region': region['id'], 'sources': 0, 'usable_residuals': 0})
                continue
            with h5py.File(observations, 'r') as h:
                body = h5py.File(solid_earth, 'r') if solid_earth is not None else None
                try:
                    field = load_field(h, body, index, screening['remove_mixed_windows'])
                    quality = h['snr'][index]
                finally:
                    if body is not None:
                        body.close()
            take = field['year_index'] == k
            local = {key: field[key][:, take] for key in ('early', 'late', 'data', 'valid', 'peak')}
            local.update(order=np.broadcast_to(field['order'][take], local['valid'].shape),
                         calendar_early=field['calendar_early'][take], calendar_late=field['calendar_late'][take])
            if not take.any():
                per_region.append({'region': region['id'], 'sources': len(index), 'usable_residuals': 0})
                continue
            r = order_residual(local, threshold=screening['min_peak'])
            good = order_screen(local, threshold=screening['min_peak'],
                                max_range_residual_m=screening['max_range_residual_m'],
                                max_azimuth_residual_m=screening['max_azimuth_residual_m'])
            residual.extend(r[good])
            snr.extend(quality[:, take][good])
            peak.extend(local['peak'][good])
            groups.extend(np.broadcast_to(region['groups'][:, None], good.shape)[good])
            used = good.any(0)
            training.update(np.floor(np.r_[local['calendar_early'][used], local['calendar_late'][used]]).tolist())
            per_region.append({'region': region['id'], 'sources': len(index),
                               'sources_with_residuals': int(good.any(1).sum()), 'usable_residuals': int(good.sum())})
        count = len(residual)
        try:
            model = calibrate_noise(np.array(residual).reshape(-1, 2), np.array(snr), np.array(peak), np.array(groups, dtype=object))
        except ValueError as error:
            raise ValueError(f'Year {year} cannot be calibrated from the reviewed controls: {count} usable held-out '
                             f'range/azimuth residuals, fewer than the required 40 ({error})') from error
        matrix = check_covariance(model['covariance_ra_m2'], f'control covariance {year}')
        by_year[year] = matrix
        report.append({'year': year, 'samples': model['samples'], 'spatial_groups': model['spatial_groups'],
                       'regions': per_region, 'training_acquisition_days_since_2000': sorted(training),
                       'model': model, 'diagnostics': diagnostics(matrix)})
        logging.info('Year %d control noise: %d residuals, %d groups, sigma RA %.3f/%.3f m', year,
                     model['samples'], model['spatial_groups'], *np.sqrt(np.diag(matrix)))
    return by_year, report


def noise_artifact(cache_directory, config, observations, solid_earth, years):
    noise = config['noise']
    identity = {'stage': 'noise', 'mode': noise['mode'], 'observations': observations.id, 'years': list(years),
                'code_sha256': module_fingerprints(ESTIMATOR_MODULES)}
    if noise['mode'] == 'supplied':
        identity['files'] = [{'sha256': digest(item['path']), 'years': item['years']} for item in noise['files']]
        return Artifact(cache_directory, 'noise', identity), None
    selection = select_controls(observations.path('observations.h5'), noise['regions'], noise['block_size_m'])
    identity.update(solid_earth=None if solid_earth is None else solid_earth.id,
                    screening=config['screening'], regions_sha256=digest(noise['regions']),
                    block_size_m=noise['block_size_m'], min_samples=noise['min_samples'],
                    estimator='order-aware 60-day acquisition-purged residuals; spatially balanced robust covariance; full-year controls',
                    controls_sha256=identity_digest([[r['id'], r['source_id'].tolist()] for r in selection[1]]))
    return Artifact(cache_directory, 'noise', identity), selection


def prepare_noise(cache_directory, config, observations, solid_earth, years):
    artifact, selection = noise_artifact(cache_directory, config, observations, solid_earth, years)
    record = artifact.complete()
    if record is not None:
        logging.info('Reusing verified noise model %s', artifact.id)
        return artifact, record
    with artifact.lock():
        record = artifact.complete()
        if record is not None:
            return artifact, record
        partial, _ = artifact.begin()
        if config['noise']['mode'] == 'supplied':
            by_year, sources = read_supplied(config['noise']['files'])
            missing = [y for y in years if y not in by_year]
            if missing:
                raise ValueError(f'No supplied noise covariance for observed year(s) {missing}; pooled or other-year noise is never substituted')
            result = {'schema': SCHEMA, 'mode': 'supplied', 'sources': sources,
                      'unused_years': sorted(set(by_year)-set(years))}
        else:
            epsg, regions = selection
            set_file = None if solid_earth is None else solid_earth.path('solid_earth.h5')
            by_year, report = estimate_controls(observations.path('observations.h5'), set_file, years, regions, config['screening'])
            result = {'schema': SCHEMA, 'mode': 'controls', 'regions_file': config['noise']['regions'],
                      'regions_epsg': epsg, 'block_size_m': config['noise']['block_size_m'],
                      'controls': [{'region': r['id'], 'source_id': r['source_id'].tolist()} for r in regions],
                      'calibration': report,
                      'method': 'full-year control record per calendar year; order-aware acquisition-purged residuals; '
                                'spatially balanced robust covariance; no Msf amplitudes used'}
        result.update(years=list(years), group_by='year of calendar midpoint',
                      covariance_ra_m2_by_year={str(y): by_year[y].tolist() for y in years},
                      diagnostics={str(y): diagnostics(by_year[y]) for y in years})
        from .velocity_io import write_json
        write_json(partial/NOISE, result)
        return artifact, artifact.publish([NOISE])


def covariance_array(noise_file, years):
    record = json.loads(Path(noise_file).read_text())
    return np.array([check_covariance(record['covariance_ra_m2_by_year'][str(y)], f'noise {y}') for y in years])

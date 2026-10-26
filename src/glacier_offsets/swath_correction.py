"""Optional production spacecraft correction, isolated from native pair products.

Only a pinned relative IW1/IW2 azimuth coefficient is applied. This module never
fits observations, launches native processing, or changes source pair manifests.
"""
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile

import h5py
import numpy as np
from osgeo import gdal
from pyproj import Transformer

from .geocode import geocoding_specification, make_mapper, write_geotiff
from .products import ChecksumCache, identity, write_json
from .stack_lock import stack_lock
from .stack_status import matching_status
from .swath_correction_config import METHOD, load_calibration, output_root, settings
from .swath_ownership import pure_swath_footprints


def read_json(path):
    return json.loads(Path(path).read_text())


def implementation():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('swath_correction.py', 'swath_correction_config.py',
                         'swath_ownership.py', 'geocode.py')}


def specification(cfg, manifest, calibration_sha256):
    """Cheap metadata signature, also used by read-only stack status."""
    return {'method': METHOD, 'implementation': implementation(),
            'match_identity': manifest['match_identity'],
            'geocode_identity': manifest['geocode_identity'],
            'geometry_identity': manifest['stack_pair']['geometry_identity'],
            'calibration_sha256': calibration_sha256,
            'geocoding': cfg['geocoding'], 'unsupported_policy': 'retain_raw',
            'allocation': 'symmetric', 'spatial_assumption': 'constant relative radar-pixel bias per swath'}


def revision(path):
    s = Path(path).stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def raster_files(directory):
    """AppleDouble files match '*.tif' too; they are never scientific rasters."""
    return sorted(p for p in directory.glob('*.tif') if not p.name.startswith('.'))


def product_files(directory):
    return sorted(p for p in directory.rglob('*') if p.is_file() and
                  all(not part.startswith('.') for part in p.relative_to(directory).parts))


class Inputs:
    def __init__(self, checksums):
        self.checksums, self.files = checksums, {}

    def file(self, path, expected=None):
        path = Path(path).resolve()
        before = revision(path)
        digest = self.checksums(path)
        if revision(path) != before:
            raise ValueError(f'Input changed during read: {path}')
        if expected is not None and digest != expected:
            raise ValueError(f'Input checksum mismatch: {path}')
        record = {'sha256': digest, **before}
        if str(path) in self.files and self.files[str(path)] != record:
            raise ValueError(f'Input changed between reads: {path}')
        self.files[str(path)] = record
        return digest

    def unchanged(self):
        for name in list(self.files):
            self.file(name)


def current(directory, spec, inputs, checksums):
    path = directory / 'manifest.json'
    if not path.is_file():
        return False
    try:
        saved = read_json(path)
        return (saved.get('status') == 'complete' and saved.get('identity') == identity(spec)
                and saved.get('inputs') == inputs and bool(saved.get('products'))
                and all((directory / p).is_file() and checksums(directory / p) == digest
                        for p, digest in saved['products'].items()))
    except (OSError, ValueError):
        return False


def correction_arrays(total, owner_swath, pure_swath, eligible, delta, beta_std):
    """Preserve original units and common motion; unsupported values stay raw."""
    available = eligible & (pure_swath == owner_swath) & np.isin(pure_swath, [1, 2])
    bias = np.where(available, np.where(owner_swath == 1, -delta / 2, delta / 2), 0.)
    corrected = np.asarray(total, dtype='float64') - bias
    sigma = np.where(available, beta_std / 2, 0.)
    return corrected, bias, available, sigma


def context(cfg, frame, manifest, cal, scenes):
    """Validate applicability against retained metadata, never date alternation."""
    scope = cal['scope']
    if frame.get('status') != 'complete' or frame['geometry_identity'] != scope['geometry_identity']:
        raise ValueError('Calibration is pinned to a different reference geometry')
    if identity(frame['specification']) != frame['geometry_identity']:
        raise ValueError('Reference geometry completion identity is inconsistent')
    if manifest['stack_pair']['geometry_identity'] != frame['geometry_identity']:
        raise ValueError('Pair and reference geometry identities disagree')
    prf = manifest['coreg']['reference_grid']['prf']
    if not np.isclose(prf, cal['reference_prf_hz'], rtol=1e-10, atol=0):
        raise ValueError('Calibration original-reference pixel interval differs from this pair')
    if manifest['coreg_specification']['polarization'] != scope['polarization']:
        raise ValueError('Pair polarization differs from calibration')
    dates = manifest['stack_pair']['acquisition_dates']
    if len(dates) != 2 or dates[0] >= dates[1]:
        raise ValueError('Pair must contain chronological earlier/later dates')
    result = []
    for day in dates:
        if not scope['start_date'] <= day <= scope['end_date']:
            raise ValueError(f'{day}: outside calibration application scope')
        source = manifest['coreg_specification']['inputs'][day]['safe']['path']
        scene = scenes.get(day, {})
        mission = re.match(r'^(S1[AB])_', Path(source).name)
        if not mission or scene.get('mission') != mission[1] or scene.get('path') != source:
            raise ValueError(f'{day}: spacecraft/source identity unavailable or inconsistent')
        for key in ('direction', 'relative_orbit'):
            if scene.get(key) != scope[key]:
                raise ValueError(f'{day}: calibration {key} differs')
        result.append({'date': day, 'mission': mission[1], 'safe': source,
                       'relative_orbit': scene['relative_orbit'], 'direction': scene['direction']})
    return result


def correct_pair(cfg, pair, frame, scenes, cal, cal_hash, checksums, force=False):
    raw = Path(cfg['output']['directory']) / 'pairs' / pair
    destination = output_root(cfg) / pair
    if destination.resolve().parent != output_root(cfg).resolve():
        raise ValueError('Corrected pair path escapes the separate pairs_directory')
    if destination.is_symlink() or (destination.exists() and any(p.is_symlink() for p in destination.rglob('*'))):
        raise ValueError('Corrected output must not contain symlinks to other products')
    if destination.exists() and (any(destination.iterdir()) and
            (not (destination / 'manifest.json').is_file() or
             read_json(destination / 'manifest.json').get('method') != METHOD)):
        raise ValueError(f'Refusing to overwrite an unrelated directory: {destination}')
    inputs = Inputs(checksums)
    inputs.file(raw / 'manifest.json')
    manifest = read_json(raw / 'manifest.json')
    if manifest.get('stages', {}).get('match') != 'complete' or matching_status(cfg, manifest, raw)['status'] != 'complete':
        raise ValueError('Production matching is missing, incomplete or stale; run stack match first')
    if (manifest.get('stages', {}).get('geocode') != 'complete' or
            manifest.get('geocode_identity') != identity(geocoding_specification(cfg, manifest))):
        raise ValueError('Production geocoding is missing, incomplete or stale; run stack geocode first')
    inputs.file(Path(cfg['output']['directory']) / 'geometry/manifest.json')
    inputs.file(settings(cfg)['calibration'], cal_hash)
    inputs.file(raw / 'radar/offsets.h5')
    inputs.file(raw / 'geo/center_geometry.h5')
    for axis in ('range', 'azimuth'):
        for quantity in ('displacement_m', 'velocity_m_per_day'):
            inputs.file(raw / 'geo' / f'{axis}_{quantity}.tif')
    owner_path = manifest['coreg']['source_index']
    if owner_path not in frame.get('products', {}):
        raise ValueError('Reference ownership product has no completion checksum')
    inputs.file(owner_path, frame['products'][owner_path])
    # Retain/copy native maps and uncertainty without changing their masks.
    for p in raster_files(raw / 'geo'):
        inputs.file(p)
    scenes_pair = context(cfg, frame, manifest, cal, scenes)
    if pair != '_'.join(s['date'] for s in scenes_pair):
        raise ValueError('Pair directory name differs from retained acquisition dates')
    spec = specification(cfg, manifest, cal_hash) | {'scenes': scenes_pair}
    if not force and current(destination, spec, inputs.files, checksums):
        inputs.unchanged()
        return read_json(destination / 'manifest.json') | {'reused': True}
    contrast = int(scenes_pair[1]['mission'] == 'S1B') - int(scenes_pair[0]['mission'] == 'S1B')
    delta = cal['beta_px'] * contrast
    with h5py.File(raw / 'radar/offsets.h5', 'r') as h:
        radar_meta = json.loads(h.attrs['metadata_json'])
        if (radar_meta['matching'] != manifest['matching']['matching'] or
                radar_meta['coreg_identity'] != manifest['coreg_identity'] or
                radar_meta['input_provenance'] != manifest['coreg_specification']['inputs']):
            raise ValueError('Offsets HDF5 and production manifest metadata disagree')
        line, sample = h['center_line'][:], h['center_sample'][:]
        total, rg = h['total/azimuth_px'][:], h['total/range_px'][:]
        owner, interval = h['source_burst_index'][:], h['interval_seconds'][:]
        quality = h['valid'][:]
        base = h['input_valid'][:] & ((h['rejection_flags'][:] & 7) == 0)
        seam = h['seam_crossing'][:]
    with h5py.File(raw / 'geo/center_geometry.h5', 'r') as h:
        meta = json.loads(h.attrs['metadata_json'])
        if meta['match_identity'] != manifest['match_identity']:
            raise ValueError('Cached center geometry has a different matching identity')
        np.testing.assert_array_equal(h['center_line'][:], line)
        np.testing.assert_array_equal(h['center_sample'][:], sample)
        np.testing.assert_array_equal(h['source_burst_index'][:], owner)
        np.testing.assert_array_equal(h['interval_seconds'][:], interval)
        lon, lat = h['longitude_deg'][:], h['latitude_deg'][:]
        scale = h['azimuth_ground_spacing_m_per_slc_line'][:]
        geometry = h['geometry_valid'][:]
    ids = manifest['assembly']['source_burst_ids']
    if (ids != [s['burst_id'] for s in frame['sources']] or
            Path(owner_path).resolve() != Path(frame['coreg']['source_index']).resolve() or
            radar_meta.get('source_burst_ids') != ids):
        raise ValueError('Pair source ownership differs from the pinned reference layout')
    if any(not re.search(r'_iw[123]$', k) for k in ids) or np.max(owner) > len(ids):
        raise ValueError('Invalid source ownership metadata')
    lookup = np.array([0] + [int(k[-1]) for k in ids], dtype='uint8')
    swath = lookup[owner]
    pure = pure_swath_footprints(manifest['coreg'], line, sample, cfg['matching'], ids)
    base &= geometry & np.isfinite(total) & np.isfinite(rg) & np.isfinite(interval) & (interval > 0)
    if cfg['geocoding']['mask'] == 'quality':
        base &= quality
    corrected, bias, available, sigma = correction_arrays(
        total, swath, pure, base, delta, abs(contrast) * cal['beta_std_px'])
    mixed = base & (pure == 0) & (owner > 0)
    grid = manifest['map_grid']
    x, y = Transformer.from_crs(4326, grid['epsg'], always_xy=True).transform(lon, lat)
    mapper = make_mapper(x, y, base, grid['transform'][1], 'nearest', target_grid=grid)
    mapped_raw, mapped_corrected = mapper.map(total * scale), mapper.map(corrected * scale)
    raw_range_m = rg * manifest['coreg']['reference_grid']['range_pixel_spacing']
    raw_maps = {'azimuth_displacement_m': mapped_raw,
                'azimuth_velocity_m_per_day': mapper.map(total * scale / (interval / 86400)),
                'range_displacement_m': mapper.map(raw_range_m),
                'range_velocity_m_per_day': mapper.map(raw_range_m / (interval / 86400))}
    for name, value in raw_maps.items():
        ds = gdal.Open(str(raw / 'geo' / (name + '.tif')))
        np.testing.assert_array_equal(value, ds.ReadAsArray())
        np.testing.assert_allclose(ds.GetGeoTransform(), mapper.transform, rtol=0, atol=0)
    mapped_available = mapper.map(available.astype('float32'))
    unsupported = np.isfinite(mapped_raw) & (mapped_available == 0)
    np.testing.assert_array_equal(mapped_corrected[unsupported], mapped_raw[unsupported])
    training = cal.get('provenance', {}).get('training_dates', [])
    evaluation = ('within_calibration_period_descriptive' if training and all(
        min(training) <= s['date'] <= max(training) for s in scenes_pair) else 'outside_calibration_period_unvalidated')
    metadata = {'correction_method': METHOD, 'calibration_sha256': cal_hash,
                'beta_px': cal['beta_px'], 'pair_contrast_px': delta,
                'axis_sign': 'later-minus-earlier on original stack-reference radar grid',
                'allocation': 'subtract IW1=-delta/2; IW2=+delta/2',
                'unsupported_policy': 'retain raw; see correction_available mask',
                'spatial_assumption': 'constant relative bias per swath in reference radar pixels',
                'accuracy': 'relative empirical calibration; common azimuth bias unconstrained',
                'uncertainty': 'native formal covariance unchanged; calibration sensitivity separate, shared across samples',
                'evaluation': evaluation, 'raw_match_identity': manifest['match_identity']}
    # Build in a disposable directory, commit outputs and completion manifest last.
    with tempfile.TemporaryDirectory(prefix=f'.{pair}.', dir=destination.parent) as temp:
        stage = Path(temp)
        (stage / 'radar').mkdir()
        (stage / 'geo').mkdir()
        (stage / 'diagnostics').mkdir()
        write_json(stage / 'calibration.json', cal)
        shutil.copyfile(raw / 'radar/offsets.h5', stage / 'radar/offsets.h5')
        with h5py.File(stage / 'radar/offsets.h5', 'a') as h:
            for name, value in {'corrected_total/azimuth_px': corrected,
                                'corrected_total/range_px': rg,
                                'bias_to_subtract/azimuth_px': bias,
                                'correction_std/azimuth_px': sigma,
                                'correction_available': available, 'cross_swath_mixed': mixed,
                                'same_swath_burst_join': seam & ~mixed,
                                'azimuth_scale_m_per_px': scale}.items():
                h.create_dataset(name, data=value)
            h.attrs['swath_correction_json'] = json.dumps(metadata)
            h.attrs['derived_total_definition'] = 'corrected_total = original total - bias_to_subtract; original gross/residual/total retained'
        # The raw field and native formal uncertainty use exactly their original bytes.
        for p in raster_files(raw / 'geo'):
            shutil.copyfile(p, stage / 'geo' / p.name)
        shutil.copyfile(raw / 'geo/azimuth_displacement_m.tif', stage / 'geo/raw_azimuth_displacement_m.tif')
        shutil.copyfile(raw / 'geo/azimuth_velocity_m_per_day.tif', stage / 'geo/raw_azimuth_velocity_m_per_day.tif')
        fields = {'azimuth_displacement_m': mapped_corrected,
                  'azimuth_velocity_m_per_day': mapper.map(corrected * scale / (interval / 86400)),
                  'azimuth_bias_to_subtract_m': mapper.map(bias * scale),
                  'azimuth_bias_to_subtract_m_per_day': mapper.map(bias * scale / (interval / 86400)),
                  'azimuth_correction_std_m': mapper.map(sigma * scale),
                  'azimuth_correction_std_m_per_day': mapper.map(sigma * scale / (interval / 86400)),
                  'correction_available': mapped_available,
                  'cross_swath_mixed': mapper.map(mixed.astype('float32')),
                  'same_swath_burst_join': mapper.map((seam & ~mixed).astype('float32'))}
        for name, value in fields.items():
            units = ('m/day' if name.endswith('m_per_day') else 'm' if name.endswith('_m') else '0/1')
            write_geotiff(stage / 'geo' / (name + '.tif'), value, mapper.transform, grid['epsg'],
                          metadata | {'units': units, 'quantity': name})
        write_json(stage / 'geo/geocoding.json', {'target_grid': grid, 'resampling': 'nearest',
                   'source_mask': cfg['geocoding']['mask'], 'raw_geocode_identity': manifest['geocode_identity'],
                   'correction': metadata})
        counts = {'map_pixels': int(np.isfinite(mapped_raw).sum()),
                  'corrected_map_pixels': int((mapped_available == 1).sum()),
                  'retained_raw_map_pixels': int(unsupported.sum()),
                  'mixed_map_pixels': int((fields['cross_swath_mixed'] == 1).sum()),
                  'eligible_centers': int(base.sum()), 'corrected_centers': int(available.sum())}
        diagnostics = {'pair': pair, 'scenes': scenes_pair, 'metadata': metadata,
                       'counts': counts, 'beta_std_px': cal['beta_std_px'],
                       'pair_contrast_std_px': abs(contrast) * cal['beta_std_px'],
                       'uncertainty_interpretation': cal.get('provenance', {}).get('uncertainty'),
                       'checks': {'raw_maps_reproduced_exactly': True,
                                  'unsupported_observations_unchanged': True,
                                  'range_and_native_uncertainty_unchanged': True},
                       'quality': 'Original matching masks retained; no residual/agreement selection',
                       'limits': ['No new overlap measurements or absolute accuracy validation',
                                  'Full-swath constant relative-bias assumption',
                                  'Mixed footprints retain raw values and have correction_available=0',
                                  'Calibration uncertainty excludes model error and is not independent per window']}
        write_json(stage / 'diagnostics/report.json', diagnostics)
        plot_comparison(stage, pair, mapper.transform, mapped_raw, mapped_corrected,
                        -fields['azimuth_bias_to_subtract_m'], metadata)
        inputs.unchanged()
        staged_files = product_files(stage)
        products = {str(p.relative_to(stage)): checksums(p) for p in staged_files}
        completed = {'method': METHOD, 'status': 'complete', 'identity': identity(spec),
                     'specification': spec, 'inputs': inputs.files, 'products': products,
                     'completed_utc': datetime.now(timezone.utc).isoformat(),
                     'counts': counts, 'metadata': metadata}
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / 'manifest.json', {'method': METHOD, 'status': 'running', 'identity': identity(spec)})
        for p in staged_files:
            target = destination / p.relative_to(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(p, target)
        inputs.unchanged()
        write_json(destination / 'manifest.json', completed)
    return completed | {'reused': False}


def plot_comparison(stage, pair, transform, raw, corrected, applied, metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x0, dx, _, y0, _, dy = transform
    extent = [x0, x0 + raw.shape[1] * dx, y0 + raw.shape[0] * dy, y0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), layout='constrained')
    finite = abs(applied[np.isfinite(applied)])
    correction_limit = max(float(np.quantile(finite, .98)), .01) if finite.size else .01
    for ax, data, title, limit in zip(axes, (raw, corrected, applied),
                                    ('Raw', 'Spacecraft corrected', 'Correction added'), (5, 5, correction_limit)):
        im = ax.imshow(data, extent=extent, origin='upper', interpolation='nearest',
                       cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set_title(title)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        fig.colorbar(im, ax=ax, shrink=.8, label='Azimuth displacement (m)')
    fig.suptitle(f'{pair} · relative spacecraft correction Δ={metadata["pair_contrast_px"]:+.5f} px')
    fig.supxlabel('Mixed/unsupported windows retain raw values. Raw/corrected color limits ±5 m.', fontsize=9)
    fig.savefig(stage / 'diagnostics/comparison.png', dpi=140)
    plt.close(fig)


def execute_corrections(cfg, force=False, pairs=None):
    """Read saved completed pairs; raw writer lock prevents concurrent replacement."""
    cal, cal_hash = load_calibration(cfg)
    raw_root, target = Path(cfg['output']['directory']), output_root(cfg)
    with stack_lock(raw_root), stack_lock(target, create=True):
        catalog = read_json(raw_root / 'stack_manifest.json')
        frame = read_json(raw_root / 'geometry/manifest.json')
        active = sorted(k for k, v in catalog['pairs'].items() if v.get('active', True))
        selected = sorted(set(pairs)) if pairs is not None else active
        if not selected or any(k not in active or not re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}', k) for k in selected):
            raise ValueError('Select one or more active saved date pairs')
        scenes = {s['date']: s for s in catalog['inventory'] if s.get('date')}
        report = {'method': METHOD, 'status': 'running', 'source_stack': str(raw_root),
                  'pairs_directory': str(target), 'calibration_sha256': cal_hash, 'pairs': {}}
        checksums = ChecksumCache()
        try:
            for i, pair in enumerate(selected, 1):
                logging.info('Correcting swaths %s (%d/%d)', pair, i, len(selected))
                try:
                    result = correct_pair(cfg, pair, frame, scenes, cal, cal_hash, checksums, force=force)
                    report['pairs'][pair] = {'status': 'complete', 'identity': result['identity'],
                                             'reused': result['reused'], 'counts': result['counts']}
                except Exception as exc:
                    report['pairs'][pair] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                    logging.error('Swath correction %s: %s', pair, exc)
                write_json(target / 'correction_status.json', report)
        except KeyboardInterrupt:
            report['status'] = 'interrupted'
            write_json(target / 'correction_status.json', report)
            raise
        report['status'] = 'complete' if all(v['status'] == 'complete' for v in report['pairs'].values()) else 'failed'
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(target / 'correction_status.json', report)
        return 0 if report['status'] == 'complete' else 1


def correction_status(cfg, manifest, pair):
    """Inspect pins/identities and file revisions without hashing scientific files."""
    from .stack_status import state
    try:
        cal, digest = load_calibration(cfg)
        root = output_root(cfg) / pair
        if not (root / 'manifest.json').is_file():
            return state('pending', 'No corrected pair completion recorded')
        saved = read_json(root / 'manifest.json')
        if saved.get('status') != 'complete':
            return state('incomplete', 'Correction interrupted or incomplete; rerun correct-swaths')
        expected = specification(cfg, manifest, digest)
        actual = {k: v for k, v in saved['specification'].items() if k != 'scenes'}
        if expected != actual:
            return state('stale', 'Raw dependencies, calibration or correction implementation changed')
        if saved.get('identity') != identity(saved['specification']):
            return state('incomplete', 'Correction completion identity is inconsistent')
        for path, entry in saved['inputs'].items():
            if not Path(path).is_file() or revision(path) != {k: entry[k] for k in ('size', 'mtime_ns', 'ctime_ns')}:
                return state('stale', f'Consumed input changed or unavailable: {path}')
        if not saved.get('products') or any(not (root / p).is_file() for p in saved['products']):
            return state('incomplete', 'Corrected output is missing')
        return state('complete', 'Pinned correction complete; scientific checksums verified on execution, not status')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return state('unknown', f'Cannot inspect correction: {exc}')

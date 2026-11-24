"""Preparation, restart, maps and numerical replay for Sentinel-1 multitrack fits."""
from contextlib import ExitStack, contextmanager
import json
import logging
from pathlib import Path
import time

import h5netcdf
import h5py
import numpy as np
from pyproj import CRS
import yaml

from .velocity_annual_noise import prepare_noise
from .velocity_cache import identity_digest
from .velocity_collection import collect, check_unchanged
from .velocity_io import digest, write_json, writer_lock
from .velocity_multitrack import (
    JointLayout, Track, common_grid, covariances_for, fit_batch, ordered_tracks, overlap_axis, pair_rows,
)
from .velocity_pixel_general import Layout, parameter_units
from .velocity_pixel_runtime import module_fingerprints
from .velocity_prepare import prepare_observations
from .velocity_solid_earth import prepare_solid_earth
from .velocity_workflow import (
    FIT_FILE, FIT_PARTIAL, FIT_MODULES, RUN_MANIFEST, invalidate, manifest_year_minimums, record_stage, stage_current,
)
from .velocity_workflow_config import MULTITRACK, scientific_parameters

SCHEMA = 'glacier-offsets-sentinel1-multitrack-v1'
MODULES = (*FIT_MODULES, 'velocity_multitrack', 'velocity_multitrack_workflow')
JOINT_FIELDS = ('coefficients', 'amplitude', 'phase_deg', 'amplitude_sigma', 'phase_sigma_deg',
                'coefficient_covariance', 'target_condition', 'rank', 'estimable', 'rms_m', 'calendar_clusters',
                'downweighted_pairs', 'annual_en_covariance', 'target_information', 'target_min_information',
                'target_min_information_by_group')
AZIMUTH_FIELDS = ('coefficients', 'amplitude', 'phase_deg', 'amplitude_sigma', 'phase_sigma_deg',
                  'coefficient_covariance', 'rms_m', 'estimable')
SOURCE_FIELDS = ('flow_axis_EN', 'mean_flow_speed', 'year_direction_change_deg', 'clean_pairs_by_track_year',
                 'clean_pairs_by_year', 'phase_bins', 'span_days', 'fit_support', 'standard_coverage',
                 'azimuth_flow_projection_by_track', 'rms_by_track_m')


def is_multitrack(out):
    path = Path(out)/RUN_MANIFEST
    return path.is_file() and json.loads(path.read_text()).get('schema') == SCHEMA


def inspect_grid(tracks):
    with ExitStack() as stack:
        handles = [stack.enter_context(h5py.File(path, 'r')) for _, path, _ in tracks]
        crs = CRS.from_epsg(3031)
        for h in handles:
            if 'crs' not in h or not CRS.from_wkt(h['crs'].attrs.get('crs_wkt', '')).equals(crs, ignore_axis_order=True):
                raise ValueError('Multitrack grids must use EPSG:3031')
        x, _, _ = overlap_axis([h['x'][:] for h in handles])
        y, _, _ = overlap_axis([h['y'][:] for h in handles], -1)
        return x, y


def prepare(config, out, cache, hook=None, solid_stop=None):
    ordered = ordered_tracks(config['inputs'])
    x, y = inspect_grid(ordered)
    manifests = [collect([path], cache) for _, path, _ in ordered]
    years = sorted({y for m in manifests for y in m['years']})
    pairs = pair_rows(ordered, manifests)
    entries, minimums = [], {}
    for (track, path, direction), manifest in zip(ordered, manifests):
        needed = manifest_year_minimums(manifest, config['coverage'])
        missing = [year for year in years if year not in needed
                   or manifest['pairs_by_year'][str(year)] < needed[year]]
        if missing:
            raise ValueError(f'Path {track:03d} cannot meet minimum pair support in year(s) {missing}')
        minimums[str(track)] = needed
        obs, obs_record = prepare_observations(cache, manifest, config['execution']['preparation_chunk_pairs'], hook)
        solid, solid_record = None, None
        if config['corrections']['solid_earth']:
            solid, solid_record = prepare_solid_earth(cache, obs, config['execution'], solid_stop)
        try:
            native = {**config, 'noise': {**config['noise'], 'group_by': 'year'}}
            noise, noise_record = prepare_noise(cache, native, obs, solid, years)
        except ValueError as exc:
            raise ValueError(f'Path {track:03d} control calibration failed: {exc}') from exc
        entry = {'track': track, 'direction': direction, 'input': manifest['inputs'][0]}
        for label, artifact, record, filename in [('observations', obs, obs_record, 'observations.h5'),
                                                  ('solid_earth', solid, solid_record, 'solid_earth.h5'),
                                                  ('noise', noise, noise_record, 'noise.json')]:
            entry[label] = str(artifact.path(filename)) if artifact is not None else None
            entry[f'{label}_sha256'] = record['files'][filename] if record is not None else None
        # Revisions are checked after reading/preparation, before publishing the collection.
        check_unchanged(manifest)
        entries.append(entry)
    collection = {'schema': SCHEMA, 'tracks': [r['track'] for r in entries], 'inputs': [r['input'] for r in entries],
                  'years': years, 'pairs': pairs, 'min_pairs_by_track_year': minimums,
                  'common_grid': {'EPSG': 3031, 'shape': [len(y), len(x)], 'first_x': float(x[0]), 'first_y': float(y[0])}}
    write_json(out/'collection-manifest.json', collection)
    return collection, entries


@contextmanager
def inputs(collection, entries, verify=False):
    if verify:
        for entry in entries:
            for key in ('observations', 'solid_earth', 'noise'):
                path = entry[key]
                if path is not None and digest(path) != entry[f'{key}_sha256']:
                    raise ValueError(f'Multitrack cache changed: {path}')
    rows = collection['pairs']
    layout = JointLayout(collection['years'], rows)
    covariance = covariances_for(rows, {e['track']: json.loads(Path(e['noise']).read_text()) for e in entries})
    with ExitStack() as stack:
        tracks = []
        for entry in entries:
            take = np.array([p['track'] == entry['track'] for p in rows])
            tracks.append(Track(stack, entry, [p for p, selected in zip(rows, take) if selected], layout, covariance[take]))
        yield tracks, layout


def _copy_grid(h, grid, tracks, years):
    h.attrs['EPSG'] = 3031
    h.attrs['years'] = years
    h['x'], h['y'] = grid.x, grid.y
    h['track'] = np.array([track.number for track in tracks], dtype=np.int32)
    h['crs'] = np.int32(0)
    h['crs'].attrs.update(grid.crs)
    for key, data in [('row', grid.row), ('col', grid.col), ('map_source_index', grid.back),
                      ('native_source_index', grid.native_index), ('native_source_id', grid.native_id),
                      ('native_chip_separation_m', grid.chip_separation_m), ('native_longitude', grid.longitude),
                      ('native_latitude', grid.latitude)]:
        h.create_dataset(key, data=data, compression='gzip', compression_opts=2)
    h['source_id'] = np.arange(len(grid.row), dtype=np.int32)


def _fit_group(parent, name, template, included, ns):
    group = parent.create_group(name)
    group.attrs['parameter_names'] = json.dumps(template['names'])
    group.attrs['parameter_units'] = json.dumps(parameter_units(template['names']))
    for key in included:
        v = template[key]
        group.create_dataset(key, shape=(ns, *v.shape[1:]), dtype=v.dtype, compression='gzip', compression_opts=2)
    return group


def fit_overlap(out, grid, tracks, layout, config, recipe_sha, hook=None):
    """Fit in bounded batches, checkpointing only fully populated source ranges."""
    partial = out/FIT_PARTIAL
    ns = len(grid.row)
    if partial.is_file():
        with h5py.File(partial, 'r') as h:
            resume = (h.attrs.get('recipe_sha256') == recipe_sha and h.attrs.get('status') == 'writing'
                      and 'source_id' in h and len(h['source_id']) == ns)
        if not resume:
            partial.unlink()
    if not partial.is_file():
        values, joint, azimuth = fit_batch(tracks, grid.native_index[:1], layout, config)
        with h5py.File(partial, 'w') as h:
            _copy_grid(h, grid, tracks, layout.years)
            for key in SOURCE_FIELDS:
                v = values[key]
                h.create_dataset(key, shape=(ns, *v.shape[1:]), dtype=v.dtype, compression='gzip', compression_opts=2)
            _fit_group(h, 'residual_vertical', joint, JOINT_FIELDS, ns)
            for track, fitted in azimuth.items():
                _fit_group(h, f'azimuth_track_{track:03d}', fitted, AZIMUTH_FIELDS, ns)
            h.attrs.update(schema=SCHEMA, status='writing', recipe_sha256=recipe_sha, completed_sources=0)
    batch = config['execution']['batch_size']
    started = time.monotonic()
    with h5py.File(partial, 'r+') as h:
        for begin in range(int(h.attrs['completed_sources']), ns, batch):
            section = slice(begin, min(begin+batch, ns))
            values, joint, azimuth = fit_batch(tracks, grid.native_index[section], layout, config)
            for key in SOURCE_FIELDS:
                h[key][section] = values[key]
            for name, result in [('residual_vertical', joint), *[(f'azimuth_track_{t:03d}', v) for t, v in azimuth.items()]]:
                for key in h[name]:
                    h[name][key][section] = result[key]
            h.flush()
            h.attrs.modify('completed_sources', section.stop)
            h.flush()
            if begin == 0 or begin//batch % 10 == 0 or section.stop == ns:
                logging.info('Fitted %d/%d multitrack source tuples (%.1fs)', section.stop, ns, time.monotonic()-started)
            if hook:
                hook('fit', section.stop)
        h.attrs.modify('status', 'complete')
    partial.replace(out/FIT_FILE)
    return {'sources': ns, 'output_sha256': digest(out/FIT_FILE), 'recipe_sha256': recipe_sha}


@contextmanager
def saved_fit(out):
    out = Path(out)
    completion = json.loads((out/'fit-completion.json').read_text())
    recipe = json.loads((out/'recipe.json').read_text())
    if digest(out/FIT_FILE) != completion['output_sha256'] or identity_digest(recipe) != completion['recipe_sha256']:
        raise ValueError('Stored multitrack fits or recipe changed')
    with h5py.File(out/FIT_FILE, 'r') as h:
        if h.attrs.get('status') != 'complete':
            raise ValueError('Stored multitrack fits are incomplete')
        yield h


def export(out):
    out = Path(out)
    with saved_fit(out) as h:
        back = h['map_source_index'][:]
        footprint = back >= 0
        safe = np.maximum(back, 0)
        tracks, years = h['track'][:], h.attrs['years']
        joint = h['residual_vertical']
        valid = joint['estimable'][:]
        cfg = json.loads((out/RUN_MANIFEST).read_text())['config']['display']
        standard = h['standard_coverage'][:].astype(bool)
        along = valid & standard & (h['mean_flow_speed'][:] >= cfg['min_speed_m_day'])
        phase_quality = (along & (h['year_direction_change_deg'][:] <= cfg['max_direction_change_deg'])
                         & (joint['phase_sigma_deg'][:] <= cfg['max_phase_sigma_deg'])
                         & (joint['amplitude'][:] >= cfg['min_amplitude_over_sigma']*joint['amplitude_sigma'][:]))
        proxy = phase_quality & (h['mean_flow_speed'][:] >= cfg['proxy_min_speed_m_day'])
        partial = out/'coefficient_maps.partial.nc'
        with h5netcdf.File(partial, 'w') as nc:
            nc.dimensions = {'y': back.shape[0], 'x': back.shape[1], 'track': len(tracks), 'year': len(years),
                             'enu_horizontal': 2, 'enu_horizontal2': 2, 'radar_axis': 2, 'harmonic': 2, 'harmonic2': 2}
            nc.attrs.update(schema=SCHEMA, status='writing', EPSG=3031, Conventions='CF-1.10',
                            phase_origin='2000-01-01 00:00 UTC; phase=atan2(S,C)',
                            spatial_support='shared native chips and matching windows across map cells')
            for label, data in [('x', h['x'][:]), ('y', h['y'][:]), ('track', tracks), ('year', years)]:
                nc.create_variable(label, (label,), data=data)
            for label in ('x', 'y'):
                nc[label].attrs.update(units='m', standard_name=f'projection_{label}_coordinate', axis=label.upper())
            nc.create_variable('crs', (), data=np.int32(0)).attrs.update(dict(h['crs'].attrs))
            nc.create_variable('source_observation_index', ('y', 'x'), data=back, compression='gzip', compression_opts=2)
            for coord, labels in [('enu_horizontal', ('east','north')), ('enu_horizontal2', ('east','north')),
                                  ('radar_axis', ('range','azimuth')), ('harmonic', ('C','S')), ('harmonic2', ('C','S'))]:
                nc.create_variable(coord, (coord,), dtype=h5py.string_dtype(), data=np.array(labels, dtype=object))

            def mapped(name, values, leading=(), units=None):
                values = np.asarray(values)
                flags = values.dtype.kind in 'biu'
                dtype = 'i4' if flags else values.dtype
                target = nc.create_variable(name, leading+('y', 'x'), dtype, fillvalue=-1 if flags else np.nan,
                                            compression='gzip', compression_opts=2)
                target.attrs.update(grid_mapping='crs')
                if units:
                    target.attrs['units'] = units
                for index in np.ndindex(values.shape[1:]):
                    vector = values[(slice(None), *index)]
                    target[index] = np.where(footprint, vector[safe], -1 if flags else np.nan)

            mapped('fit_estimable', valid)
            mapped('standard_coverage_mask', standard)
            mapped('along_display_mask', along)
            mapped('phase_quality_display_mask', phase_quality)
            mapped('along_flow_proxy_mask', proxy)
            mapped('native_chip_separation_m', h['native_chip_separation_m'][:], units='m')
            mapped('native_source_id', h['native_source_id'][:], ('track',))
            mapped('clean_pairs_by_track_year', h['clean_pairs_by_track_year'][:], ('track','year'))
            mapped('radar_rms_by_track_m', h['rms_by_track_m'][:], ('track','radar_axis'), 'm')
            mapped('azimuth_flow_projection_by_track', h['azimuth_flow_projection_by_track'][:], ('track',))
            mapped('flow_axis_EN', h['flow_axis_EN'][:], ('enu_horizontal',))
            mapped('annual_EN_velocity', joint['coefficients'][:, :2*len(years)].reshape(len(valid), len(years), 2),
                   ('year','enu_horizontal'), 'm day-1')
            mapped('annual_EN_covariance', joint['annual_en_covariance'][:], ('year','enu_horizontal','enu_horizontal2'), 'm2 day-2')
            names = json.loads(joint.attrs['parameter_names'])
            nc.dimensions['parameter'] = len(names)
            nc.create_variable('residual_vertical_parameter_name', ('parameter',), dtype=h5py.string_dtype(),
                               data=np.array(names,dtype=object))
            mapped('residual_vertical_coefficients', joint['coefficients'][:], ('parameter',))
            for source, target, unit in [('amplitude','msf_along_amplitude','m day-1'),
                                         ('phase_deg','msf_along_phase_deg','degrees'),
                                         ('amplitude_sigma','msf_along_amplitude_sigma','m day-1'),
                                         ('phase_sigma_deg','msf_along_phase_sigma_deg','degrees'),
                                         ('target_condition','msf_condition',None),
                                         ('target_min_information','msf_min_information',None)]:
                mapped(target, joint[source][:], units=unit)
            mapped('msf_cs_covariance', joint['coefficient_covariance'][:], ('harmonic','harmonic2'), 'm2 day-2')
            mapped('msf_min_information_by_track', joint['target_min_information_by_group'][:], ('track',))
            for constituent in ('M2','O1'):
                j = names.index(f'vertical_{constituent}_C')
                cs = joint['coefficients'][:, j:j+2]
                mapped(f'vertical_{constituent.lower()}_amplitude', np.linalg.norm(cs, axis=1), units='m')
                mapped(f'vertical_{constituent.lower()}_phase_deg', np.degrees(np.arctan2(cs[:, 1], cs[:, 0])), units='degrees')
            for key, unit in [('amplitude','m day-1'), ('phase_deg','degrees'), ('estimable',None)]:
                mapped(f'msf_azimuth_{key}_by_track', np.stack([h[f'azimuth_track_{t:03d}/{key}'][:] for t in tracks], 1),
                       ('track',), unit)
            nc.attrs['status'] = 'complete'
        partial.replace(out/'coefficient_maps.nc')
        summary = spatial_summary(h)
    write_json(out/'map-summary.json', summary)
    return ['coefficient_maps.nc', 'map-summary.json']


def spatial_summary(h):
    """Describe map coverage without counting replicates as independent sources."""
    valid = h['residual_vertical/estimable'][:]
    native = h['native_source_id'][:]
    back = h['map_source_index'][:]
    tracks = h['track'][:].tolist()
    location_x = h['x'][:][h['col'][:]]
    location_y = h['y'][:][h['row'][:]]
    blocks = np.floor(np.stack([location_x, location_y], 1)/3000).astype(int)
    _, group = np.unique(blocks, axis=0, return_inverse=True)
    # One deterministic representative per block, with no reused native chip between representatives.
    chosen, occupied, used = [], set(), [set() for _ in tracks]
    separation = h['native_chip_separation_m'][:]
    for i in np.argsort(separation, kind='stable'):
        if valid[i] and group[i] not in occupied and not any(s in bag for s, bag in zip(native[i], used)):
            chosen.append(i)
            occupied.add(group[i])
            for s, bag in zip(native[i], used):
                bag.add(s)
    largest = h['residual_vertical/target_min_information'][:]
    per_track = h['residual_vertical/target_min_information_by_group'][:]
    improvement = np.divide(largest[:, None], per_track, out=np.full_like(per_track, np.nan), where=per_track > 1e-8)
    result = {'tracks': tracks, 'common_cells': int(np.sum(back >= 0)), 'source_tuples': len(native),
              'estimable_source_tuples': int(valid.sum()),
              'estimable_cells': int(np.sum((back >= 0) & valid[np.maximum(back, 0)])),
              'native_sources_in_estimable_tuples': {str(t): len(set(native[valid, i])) for i, t in enumerate(tracks)},
              'joint_information_gain_vs_track': {},
              'distinct_3km_block_representatives': len(chosen),
              'native_chip_separation_percentiles_m': np.percentile(separation, [5, 50, 95]).tolist()}
    for i, track in enumerate(tracks):
        ratios = improvement[chosen, i]
        ratios = ratios[np.isfinite(ratios)]
        result['joint_information_gain_vs_track'][str(track)] = {
            'blocks': len(ratios), 'median': float(np.median(ratios)) if len(ratios) else None,
            'fraction_above_one': float(np.mean(ratios > 1+1e-8)) if len(ratios) else None}
    return result


def verify(out):
    out = Path(out)
    manifest = json.loads((out/RUN_MANIFEST).read_text())
    recipe = json.loads((out/'recipe.json').read_text())
    if recipe['code_sha256'] != module_fingerprints(MODULES):
        raise ValueError('Multitrack numerical implementation changed since fitting')
    collection_path = out/'collection-manifest.json'
    if digest(collection_path) != manifest['collection_sha256']:
        raise ValueError('Stored multitrack collection changed')
    collection = json.loads(collection_path.read_text())
    check_unchanged(collection)
    with inputs(collection, manifest['native_artifacts'], verify=True) as (tracks, layout), saved_fit(out) as h:
        eligible = np.flatnonzero(h['residual_vertical/estimable'][:])
        if not len(eligible):
            raise ValueError('No estimable multitrack sources to verify')
        selected = np.sort(np.random.default_rng(12345).choice(eligible, min(64, len(eligible)), replace=False))
        grid = common_grid([t.obs for t in tracks])
        for key, expected in [('map_source_index', grid.back), ('native_source_id', grid.native_id),
                              ('native_source_index', grid.native_index), ('x', grid.x), ('y', grid.y)]:
            if not np.array_equal(h[key][:], expected):
                raise ValueError(f'Multitrack source association changed: {key}')
        values, joint, az = fit_batch(tracks, grid.native_index[selected], layout, manifest['config'])
        for key in ('flow_axis_EN', 'clean_pairs_by_track_year', 'rms_by_track_m'):
            np.testing.assert_allclose(h[key][selected], values[key], atol=1e-8, rtol=0, equal_nan=True)
        for group, fitted in [('residual_vertical', joint), *[(f'azimuth_track_{t:03d}', v) for t, v in az.items()]]:
            np.testing.assert_allclose(h[f'{group}/coefficients'][selected], fitted['coefficients'], atol=1e-8, rtol=0, equal_nan=True)
        with h5netcdf.File(out/'coefficient_maps.nc', 'r') as nc:
            if not np.array_equal(nc['source_observation_index'][:], grid.back):
                raise ValueError('Exported multitrack map association changed')
            actual = nc['residual_vertical_coefficients'][:][:, grid.row[selected], grid.col[selected]].T
            np.testing.assert_allclose(actual, joint['coefficients'], atol=1e-8, rtol=0, equal_nan=True)
    record = {'status': 'passed', 'sampled_tuples': len(selected), 'exact_grid_and_source_association': True,
              'replayed_track_counts_rms_and_coefficients': True}
    write_json(out/'verification.json', record)
    return record


def report(out):
    out = Path(out)
    summary = json.loads((out/'map-summary.json').read_text())
    verification = json.loads((out/'verification.json').read_text())
    manifest = json.loads((out/RUN_MANIFEST).read_text())
    lines = ['# Sentinel-1 multitrack velocities', '', 'Exploratory annual EN and along-flow Msf estimates.',
             '', f"Validated {verification['sampled_tuples']} source tuples by replay.",
             f"{summary['estimable_cells']:,} cells and {summary['estimable_source_tuples']:,} source tuples pass the fit gates.",
             f"Block summaries use {summary['distinct_3km_block_representatives']:,} 3 km block representatives with distinct native chips.",
             '', '## Track-specific calibration', '', '| Orbit | Year | Range σ (m) | Azimuth σ (m) | Held-out residuals |',
             '| --- | --- | --- | --- | --- |']
    for entry in manifest['native_artifacts']:
        record = json.loads(Path(entry['noise']).read_text())
        for year, info in record['diagnostics'].items():
            count = next((row['samples'] for row in record['calibration'] if row['year'] == int(year)), 0)
            lines.append(f"| {entry['track']:03d} | {year} | {info['sigma_range_m']:.3f} | {info['sigma_azimuth_m']:.3f} | {count} |")
    lines += ['', '## Msf information', '',
              'Ratios compare the smallest Msf information eigenvalue after eliminating nuisance terms. '
              'They use the same joint robust weights on each track subset.', '',
              '| Track used alone | Median joint / track | Blocks with ratio > 1 |', '| --- | --- | --- |']
    for track, value in summary['joint_information_gain_vs_track'].items():
        if value['median'] is not None:
            lines.append(f"| {track} | {value['median']:.2f} | {value['fraction_above_one']:.1%} of {value['blocks']} |")
    lines += ['', 'Uncertainties are pointwise and conditional on the fitted model and final weights. '
              'The 30-day calendar clusters include both viewing geometries. Neighboring cells share native chips and match windows.',
              '', 'The map uses common support and exact native burst endpoints. The Evans pilot report gives geographically matched held-out scores.', '']
    (out/'report.md').write_text('\n'.join(lines))
    return ['report.md']


def run(config, stage, hook=None, solid_stop=None):
    out, cache = Path(config['output']), Path(config['cache_directory'])
    with writer_lock(out):
        cache.mkdir(parents=True, exist_ok=True)
        previous = json.loads((out/RUN_MANIFEST).read_text()) if (out/RUN_MANIFEST).is_file() else {}
        if previous and previous.get('schema') != SCHEMA:
            raise ValueError('Choose a separate output directory for this multitrack collection')
        collection, entries = prepare(config, out, cache, hook, solid_stop)
        resolved = {k: v for k, v in config.items() if k != 'config_file'}
        (out/'inversion-config.resolved.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
        collection_sha = digest(out/'collection-manifest.json')
        recipe = {'schema': SCHEMA, 'parameters': {k: v for k, v in scientific_parameters(config).items() if k != 'display'},
                  'corrections': config['corrections'], 'collection_sha256': collection_sha,
                  'artifacts': [{k: v for k, v in e.items() if k.endswith('_sha256')} for e in entries],
                  'phase_origin': '2000-01-01 00:00 UTC; phase=atan2(S,C)', 'code_sha256': module_fingerprints(MODULES)}
        recipe_sha = identity_digest(recipe)
        manifest = {'schema': SCHEMA, 'version': 2, 'config': resolved, 'config_file': config['config_file'],
                    'collection_sha256': collection_sha, 'native_artifacts': entries, 'recipe_sha256': recipe_sha,
                    'stage': previous.get('stage', 'prepared') if previous.get('recipe_sha256') == recipe_sha else 'prepared'}
        write_json(out/RUN_MANIFEST, manifest)
        if stage == 'prepare':
            return manifest
        fit_identity = {'recipe_sha256': recipe_sha}
        if not stage_current(out, 'fit', fit_identity):
            invalidate(out, ('fit', 'export', 'maps', 'verify', 'report'))
            write_json(out/'recipe.json', recipe)
            with inputs(collection, entries) as (tracks, layout):
                grid = common_grid([t.obs for t in tracks])
                summary = fit_overlap(out, grid, tracks, layout, config, recipe_sha, hook)
            check_unchanged(collection)
            write_json(out/'fit-completion.json', summary)
            record_stage(out, 'fit', fit_identity, ['recipe.json', FIT_FILE, 'fit-completion.json'])
        manifest['stage'] = 'fitted'
        write_json(out/RUN_MANIFEST, manifest)
        if stage == 'run':
            product_identity = {**fit_identity, 'display': config['display'], 'maps_code': module_fingerprints(('velocity_pixel_maps',))}
            if not stage_current(out, 'export', product_identity):
                invalidate(out, ('maps', 'verify', 'report'))
                record_stage(out, 'export', product_identity, export(out))
            if not stage_current(out, 'verify', product_identity):
                verify(out)
                record_stage(out, 'verify', product_identity, ['verification.json'])
            if not stage_current(out, 'maps', product_identity):
                from .velocity_pixel_maps import plot_maps
                records = plot_maps(out, component='msf-along', regions=config['diagnostics']['regions'])
                files = [str(Path(f).relative_to(out)) for record in records for f in record['files']]
                record_stage(out, 'maps', product_identity, files+['amplitude_phase_maps/plot-settings.json'])
            if not stage_current(out, 'report', product_identity):
                record_stage(out, 'report', product_identity, report(out))
            manifest['stage'] = 'complete'
            write_json(out/RUN_MANIFEST, manifest)
        return manifest

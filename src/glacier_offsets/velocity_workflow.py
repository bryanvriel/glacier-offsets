"""YAML-driven netCDF-to-results workflow (version-2 pixel inversion).

``run`` performs validation, preparation, noise calibration, fitting,
export/maps and verification/report; ``prepare`` stops after noise. Cache
artifacts (observations, solid-Earth corrections, noise) are immutable and
shared between runs; the result directory references them through
``run-manifest.json``. Each result stage records an identity; repeated runs
reuse verified stages, rebuild only stages whose inputs changed, and resume a
checkpointed fit with the same recipe.
"""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

import h5py
import numpy as np
import yaml

from .velocity_annual_noise import covariance_array, prepare_noise, read_supplied, control_regions
from .velocity_cache import identity_digest
from .velocity_collection import collect, check_unchanged
from .velocity_io import digest, write_json, writer_lock
from .velocity_pixel_general import Layout, fit_sources, parameter_units, MODELS
from .velocity_pixel_runtime import module_fingerprints
from .velocity_prepare import prepare_observations
from .velocity_solid_earth import Interrupted, check_worker, interpreter, prepare_solid_earth
from .velocity_workflow_config import impossible_settings, load_workflow_config, scientific_parameters


RUN_MANIFEST = 'run-manifest.json'
FIT_FILE = 'source_coefficients.h5'
FIT_PARTIAL = 'source_coefficients.partial.h5'
FIT_MODULES = ('velocity_pixel_general', 'velocity_basis', 'velocity_systematics', 'velocity_quality',
               'velocity_vertical')
STAGES = 'stages'


def _stage_file(out, name):
    return Path(out)/STAGES/f'{name}.json'


def stage_current(out, name, identity):
    """True when a stage record matches and every recorded product is unchanged."""
    path = _stage_file(out, name)
    if not path.is_file():
        return False
    record = json.loads(path.read_text())
    if record.get('identity_sha256') != identity_digest(identity):
        return False
    return all((Path(out)/f).is_file() and digest(Path(out)/f) == sha for f, sha in record['files'].items())


def record_stage(out, name, identity, files):
    write_json(_stage_file(out, name), {'stage': name, 'identity': identity, 'identity_sha256': identity_digest(identity),
                                        'files': {f: digest(Path(out)/f) for f in files},
                                        'completed_utc': datetime.now(timezone.utc).isoformat()})


def invalidate(out, names):
    for name in names:
        path = _stage_file(out, name)
        if path.exists():
            path.unlink()


def _artifact_entry(artifact, record):
    return {'id': artifact.id, 'directory': str(artifact.directory), 'files': record['files']}


def preflight(config, stages):
    """Report missing dependencies and unusable auxiliary inputs before processing."""
    missing = []
    for module in ('h5netcdf', 'h5py', 'pyproj', 'scipy', 'yaml'):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if stages == 'run':
        for module in ('matplotlib', 'rasterio'):
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
    if missing:
        raise ValueError(f'Missing Python dependencies for this workflow: {missing}')
    try:
        from pyproj import Transformer
        Transformer.from_crs(4326, 3031, always_xy=True).transform(0., -70.)
    except Exception as error:
        raise ValueError(f'pyproj cannot transform coordinates ({error}); activate the environment or set PROJ_DATA') from error
    worker = None
    if config['corrections']['solid_earth']:
        worker = check_worker(interpreter(config['execution']))
    noise = config['noise']
    if noise['mode'] == 'controls':
        if not Path(noise['regions']).is_file():
            raise ValueError(f'Missing control regions file: {noise["regions"]}')
        control_regions(noise['regions'])
    else:
        read_supplied(noise['files'])
    regions = config['diagnostics']['regions']
    if regions is not None and not Path(regions).is_file():
        raise ValueError(f'Missing diagnostics regions file: {regions}')
    return worker


def fit_recipe(config, layout, artifacts):
    parameters = {k: v for k, v in scientific_parameters(config).items() if k != 'display'}
    c, s = parameters['coverage'], parameters['screening']
    return {'schema': 'glacier-offsets-pixel-recipe-v2', 'version': 2, 'parameters': parameters, 'layout': layout.record(),
            'artifacts': {k: {'id': v['id'], 'files': v['files']} if 'id' in v else v for k, v in artifacts.items()},
            'corrections': config['corrections'],
            'model': 'independent local along-flow Msf C/S + shared residual vertical M2/O1 height C/S; '
                     'annual EN secular velocities integrated over exact endpoint overlaps; annual azimuth order offsets',
            'comparator': 'direct satellite-azimuth Msf with annual secular/order terms',
            'screen': f"peak>={s['min_peak']:g} and acquisition-purged order-aware RA residual<="
                      f"{s['max_range_residual_m']:g}/{s['max_azimuth_residual_m']:g} m within each year; cross-year pairs "
                      'predicted from integrated purged annual training backgrounds',
            'mixed_window_policy': {'remove_mixed_windows': s['remove_mixed_windows']},
            'min_fit': f"{c['min_pairs_per_year']} clean pairs in each year, >={c['min_pairs_total']} total, "
                       f">={c['min_span_days']:g} days, >={c['min_phase_bins']}/{c['phase_bins']} Msf phase bins; full model rank",
            'standard_coverage': f"same and >={c['min_pairs_standard']} total clean pairs",
            'flow_axis': 'robust order-removed EN velocity per year, equally averaged over years; fixed during each point fit',
            'direction_change': 'maximum pairwise annual secular direction change; zero for one year',
            'covariance': 'annual RA covariance assigned by pair calendar-midpoint year; constant within a year',
            'diagnostic_uncertainty': f"pointwise {parameters['uncertainty']['cluster_days']:g}-day calendar-cluster sandwich, CR1; "
                                      'no spatial resampling; unvalidated as calibrated pixelwise inference',
            'phase_origin': '2000-01-01 00:00 UTC; v=C*cos(omega*tau)+S*sin(omega*tau); phi=atan2(S,C)',
            'interpretation': 'exploratory coefficients, no detection/FDR or physical-Msf certification',
            'code_sha256': module_fingerprints(FIT_MODULES)}


def _create_fit(path, h, layout, covariances, recipe_sha, ns):
    ny = len(layout.years)
    with h5py.File(path, 'w') as result:
        result.attrs.update(status='writing', completed_sources=0, recipe_sha256=recipe_sha,
                            schema='glacier-offsets-pixel-coefficients-v2', years=list(layout.years),
                            layout_json=json.dumps(layout.record()), EPSG=int(h.attrs['EPSG']), crs_wkt=h.attrs['crs_wkt'])
        for name in ('source_id', 'x', 'y', 'map_source_index', 'row', 'col', 'latitude', 'longitude',
                     'known_mixed_source', 'tidal_response_factor'):
            result.create_dataset(name, data=h[name][:], compression='gzip', compression_opts=2)
        result['crs'] = np.int32(0)
        for key, value in h['crs'].attrs.items():
            result['crs'].attrs[key] = value
        result.create_dataset('covariance_by_year', data=covariances)
        options = {'compression': 'gzip', 'compression_opts': 2}
        specs = {'clean_pairs_by_year': ((ns, ny), 'i2'), 'span_days': ((ns,), 'f4'), 'phase_bins': ((ns,), 'i1'),
                 'fit_support': ((ns,), '?'), 'standard_coverage': ((ns,), '?'), 'flow_axis_EN': ((ns, 2), 'f8'),
                 'robust_velocity_EN_by_year': ((ns, ny, 2), 'f8'), 'mean_flow_speed': ((ns,), 'f4'),
                 'year_direction_change_deg': ((ns,), 'f4'), 'azimuth_flow_projection_by_year': ((ns, ny), 'f4')}
        for name, (shape, dtype) in specs.items():
            result.create_dataset(name, shape, dtype=dtype, **options)
        for model in MODELS:
            names = layout.names(model)
            p = len(names)
            group = result.create_group(model)
            group.attrs['parameter_names'] = json.dumps(list(names))
            group.attrs['parameter_units'] = json.dumps(parameter_units(names))
            for name, shape, dtype in [('coefficients', (ns, p), 'f8'), ('cs', (ns, 2), 'f8'), ('amplitude', (ns,), 'f4'),
                                       ('phase_deg', (ns,), 'f4'), ('amplitude_sigma', (ns,), 'f4'),
                                       ('phase_sigma_deg', (ns,), 'f4'), ('coefficient_covariance', (ns, 2, 2), 'f8'),
                                       ('target_condition', (ns,), 'f4'), ('rank', (ns,), 'i1'), ('estimable', (ns,), '?'),
                                       ('rms_m', (ns, 2 if model == 'residual_vertical' else 1), 'f4'),
                                       ('partial_r2', (ns,), 'f4'), ('calendar_clusters', (ns,), 'i4')]:
                group.create_dataset(name, shape, dtype=dtype, **options)


def run_fit(out, observations, solid_earth, noise_file, layout, parameters, batch, recipe_sha, hook=None):
    covariances = covariance_array(noise_file, layout.years)
    destination = out/FIT_FILE
    partial = out/FIT_PARTIAL
    with h5py.File(observations, 'r') as h:
        body = None if solid_earth is None else h5py.File(solid_earth, 'r')
        try:
            ns = len(h['source_id'])
            resume = False
            if partial.exists():
                try:
                    with h5py.File(partial, 'r') as existing:
                        resume = (existing.attrs.get('recipe_sha256') == recipe_sha and existing.attrs.get('status') == 'writing'
                                  and np.array_equal(existing['source_id'][:], h['source_id'][:]))
                except OSError:
                    resume = False
                if not resume:
                    logging.info('Discarding a fit checkpoint from a different recipe')
                    partial.unlink()
            if not resume:
                _create_fit(partial, h, layout, covariances, recipe_sha, ns)
            start = time.time()
            with h5py.File(partial, 'r+') as result:
                begin = int(result.attrs['completed_sources'])
                if begin:
                    logging.info('Resuming fit after %d/%d sources', begin, ns)
                for begin in range(begin, ns, batch):
                    sl = slice(begin, min(begin+batch, ns))
                    _, _, values, fits = fit_sources(h, body, layout, covariances, parameters, sl)
                    for name, value in values.items():
                        result[name][sl] = value
                    for model, fitted in fits.items():
                        for name in result[model]:
                            result[model][name][sl] = fitted[name]
                    result.attrs['completed_sources'] = sl.stop
                    result.flush()
                    if (begin//batch) % 10 == 0:
                        logging.info('Fitted %d/%d unique sources (%.1fs)', sl.stop, ns, time.time()-start)
                    if hook is not None:
                        hook('fit', sl.stop)
                result.attrs.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        finally:
            if body is not None:
                body.close()
    partial.replace(destination)
    return {'sources': ns, 'output_sha256': digest(destination), 'recipe_sha256': recipe_sha,
            'noise_sha256': digest(noise_file), 'elapsed_seconds': time.time()-start}


def _hook(interrupt):
    if not interrupt:
        return None

    def hook(stage, progress):
        if stage == interrupt.get('stage') and progress >= interrupt['after']:
            raise Interrupted(f'{stage} interrupted after {progress} (test injection)')
    return hook


def run_workflow(config_path, stages='run', interrupt=None, batch_size=None, remove_mixed_windows=None):
    """Execute the version-2 workflow; ``stages`` is 'prepare', 'fit' or 'run'."""
    if stages not in ('prepare', 'fit', 'run'):
        raise ValueError('stages must be prepare, fit or run')
    config = load_workflow_config(config_path)
    if remove_mixed_windows is not None:
        config['screening']['remove_mixed_windows'] = bool(remove_mixed_windows)
    if batch_size is not None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch size must be a positive integer')
        config['execution']['batch_size'] = batch_size
    hook = _hook(interrupt)
    preflight(config, stages)
    out = Path(config['output'])
    cache = Path(config['cache_directory'])
    with writer_lock(out):
        cache.mkdir(parents=True, exist_ok=True)
        manifest = collect(config['inputs'], cache)
        pairs_by_year = {int(k): v for k, v in manifest['pairs_by_year'].items()}
        problems = impossible_settings(config, pairs_by_year, manifest['span_days'])
        if config['noise']['mode'] == 'supplied':
            supplied, _ = read_supplied(config['noise']['files'])
            absent = [y for y in manifest['years'] if y not in supplied]
            if absent:
                problems.append(f'supplied noise has no covariance for observed year(s) {absent}')
        if problems:
            raise ValueError('Settings cannot be satisfied by these inputs: '+'; '.join(problems))
        write_json(out/'collection-manifest.json', manifest)
        observations, obs_record = prepare_observations(cache, manifest, config['execution']['preparation_chunk_pairs'], hook)
        artifacts = {'observations': _artifact_entry(observations, obs_record)}
        solid = None
        if config['corrections']['solid_earth']:
            stop = interrupt['after'] if interrupt and interrupt.get('stage') == 'solid_earth' else None
            solid, solid_record = prepare_solid_earth(cache, observations, config['execution'], stop)
            artifacts['solid_earth'] = _artifact_entry(solid, solid_record)
        else:
            artifacts['solid_earth'] = {'applied': False, 'reason': 'corrections.solid_earth: false'}
        with h5py.File(observations.path('observations.h5'), 'r') as h:
            layout = Layout.from_observations(h)
        noise, noise_record = prepare_noise(cache, config, observations, solid, list(layout.years))
        artifacts['noise'] = _artifact_entry(noise, noise_record)
        resolved = {k: v for k, v in config.items() if k != 'config_file'}
        (out/'inversion-config.resolved.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
        run_manifest = {'schema': 'glacier-offsets-pixel-run-v2', 'version': 2, 'config_file': config['config_file'],
                        'config_sha256': digest(config['config_file']), 'config': resolved,
                        'collection': {'file': 'collection-manifest.json', 'sha256': digest(out/'collection-manifest.json'),
                                       'inputs': [{k: i[k] for k in ('path', 'sha256')} for i in manifest['inputs']],
                                       'years': manifest['years'], 'pairs_by_year': manifest['pairs_by_year'],
                                       'cross_year_pairs': manifest['cross_year_pairs']},
                        'layout': layout.record(), 'artifacts': artifacts, 'stage': 'prepared'}
        write_json(out/RUN_MANIFEST, run_manifest)
        if stages == 'prepare':
            logging.info('Prepared observations %s, solid-Earth %s and noise %s', observations.id,
                         artifacts['solid_earth'].get('id', 'disabled'), noise.id)
            return run_manifest
        recipe = fit_recipe(config, layout, artifacts)
        recipe_sha = identity_digest(recipe)
        fit_identity = {'recipe_sha256': recipe_sha}
        if not stage_current(out, 'fit', fit_identity):
            invalidate(out, ('fit', 'export', 'maps', 'series', 'verify', 'report'))
            write_json(out/'recipe.json', recipe)
            summary = run_fit(out, observations.path('observations.h5'),
                              None if solid is None else solid.path('solid_earth.h5'), noise.path('noise.json'),
                              layout, recipe['parameters'], config['execution']['batch_size'], recipe_sha, hook)
            summary['pairs'] = len(manifest['pairs'])
            write_json(out/'fit-completion.json', summary)
            record_stage(out, 'fit', fit_identity, ['recipe.json', FIT_FILE, 'fit-completion.json'])
            logging.info('Completed %d independent source fits', summary['sources'])
        else:
            logging.info('Reusing verified source fits for recipe %s', recipe_sha[:12])
        run_manifest.update(stage='fitted', recipe_sha256=recipe_sha)
        check_unchanged(manifest)
        write_json(out/RUN_MANIFEST, run_manifest)
        if stages == 'fit':
            return run_manifest
        from .velocity_workflow_products import products
        products(out, config, run_manifest)
        run_manifest['stage'] = 'complete'
        write_json(out/RUN_MANIFEST, run_manifest)
        return run_manifest


def output_from(path):
    """Accept a result directory or a version-2 YAML."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        return path
    return Path(load_workflow_config(path)['output'])


def is_v2_output(path):
    path = Path(path)
    return path.is_dir() and (path/RUN_MANIFEST).is_file()


def load_run(out):
    """Resolve and verify cache artifacts referenced by a result directory."""
    out = Path(out)
    manifest = json.loads((out/RUN_MANIFEST).read_text())
    paths = {}
    for kind, entry in manifest['artifacts'].items():
        if 'id' not in entry:
            paths[kind] = None
            continue
        directory = Path(entry['directory'])
        for name, sha in entry['files'].items():
            if not (directory/name).is_file() or digest(directory/name) != sha:
                raise ValueError(f'Referenced {kind} cache artifact changed or is missing: {directory/name}')
        paths[kind] = directory/next(iter(entry['files']))
    return manifest, paths

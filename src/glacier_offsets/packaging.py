"""Package completed displacement observations without launching upstream stages."""
import argparse
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

import h5netcdf
import h5py
import numpy as np
from osgeo import gdal
from pyproj import CRS

from .geocode import geocoding_specification
from .package_geometry import supplemental_geometry, utc_seconds, METHOD as GEOMETRY_METHOD
from .products import ChecksumCache, checksum, identity
from .stack_config import load_status_config
from .stack_lock import stack_lock
from .stack_status import matching_status
from .swath_correction import Inputs
from .swath_correction_config import METHOD as CORRECTION_METHOD, settings as correction_settings

SCHEMA = 'glacier-offsets-displacements-v1'
PRIMARY = ('range_displacement_m', 'azimuth_displacement_m', 'snr', 'correlation_peak', 'valid_mask')
CORRECTION_FLAGS = ('correction_available', 'cross_swath_mixed', 'same_swath_burst_join')
TIME_UNITS = 'seconds since 1970-01-01 00:00:00'
CALENDAR = 'proleptic_gregorian'
IMPLEMENTATION = {name: checksum(Path(__file__).with_name(name)) for name in ('packaging.py', 'package_geometry.py')}


def settings(cfg):
    value = cfg.get('packaging', {})
    if not isinstance(value, dict) or set(value) - {'source', 'output'}:
        raise ValueError('packaging accepts source and output')
    result = {'source': 'raw', 'output': None} | value
    if result['source'] not in ('raw', 'swath_corrected'):
        raise ValueError('packaging.source must be raw or swath_corrected')
    if result['output'] is not None and (not isinstance(result['output'], str) or not Path(result['output']).is_absolute()):
        raise ValueError('packaging.output must be an absolute path after YAML resolution')
    return result


def read_json(path, inputs):
    inputs.file(path)
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON mapping: {path}')
    return value


def tiles(shape):
    for row in range(0, shape[0], 256):
        for col in range(0, shape[1], 256):
            yield slice(row, min(row+256, shape[0])), slice(col, min(col+256, shape[1]))


def read_tile(ds, section):
    y, x = section
    return ds.ReadAsArray(x.start, y.start, x.stop-x.start, y.stop-y.start)


def open_raster(path, grid):
    ds = gdal.Open(str(path))
    if ds is None or ds.RasterCount != 1 or [ds.RasterYSize, ds.RasterXSize] != grid['shape']:
        raise ValueError(f'Raster shape differs from common grid: {path}')
    if tuple(ds.GetGeoTransform()) != tuple(grid['transform']) or CRS.from_wkt(ds.GetProjection()) != CRS.from_epsg(grid['epsg']):
        raise ValueError(f'Raster georeferencing differs from common grid: {path}')
    return ds


def inspect_pair(cfg, frame, grid, pair, correction, checksums):
    """Validate one pair in isolation so failures can be recorded as omissions."""
    root = Path(cfg['output']['directory'])
    raw = root / 'pairs' / pair
    inputs = Inputs(checksums)
    manifest = read_json(raw / 'manifest.json', inputs)
    effective = deepcopy(cfg)
    # A processing --backend override is not written back into the YAML.
    # Packaging consumes the recorded backend without probing CPU/CUDA or
    # requiring users to repeat an operational choice made by the matcher.
    effective['matching']['backend'] = manifest['matching']['matching']['backend']
    if manifest.get('stages', {}).get('match') != 'complete' or matching_status(effective, manifest, raw)['status'] != 'complete':
        raise ValueError('Matching is missing, incomplete or stale')
    if (manifest.get('stages', {}).get('geocode') != 'complete' or
            manifest.get('geocode_identity') != identity(geocoding_specification(cfg, manifest))):
        raise ValueError('Geocoding is missing, incomplete or stale')
    if manifest['stack_pair']['geometry_identity'] != frame['geometry_identity'] or manifest['map_grid'] != grid:
        raise ValueError('Pair differs from the pinned reference geometry or common grid')
    if manifest['assembly']['source_burst_ids'] != [s['burst_id'] for s in frame['sources']]:
        raise ValueError('Pair source-burst ordering differs from the reference')
    sources = manifest['assembly']['sources']
    if len(sources) != len(frame['sources']):
        raise ValueError('Pair source-burst table differs from the reference')
    for actual, expected in zip(sources, frame['sources']):
        if any(actual.get(k) != expected.get(k) for k in
               ('burst_id', 'offset', 'valid_window', 'reference_valid_window', 'ownership_windows')):
            raise ValueError('Pair ownership differs from the current pinned layout')
    if pair != '_'.join(manifest['stack_pair']['acquisition_dates']):
        raise ValueError('Pair name differs from recorded acquisition dates')
    summary = read_json(raw / 'geo/geocoding.json', inputs)
    if summary != manifest['geocoding'] or summary['resampling'] != 'nearest':
        raise ValueError('Saved geocoding metadata disagree or do not use nearest-source resampling')
    earlier, later = map(utc_seconds, manifest['acquisition_times'])
    if not np.isfinite([earlier, later]).all() or later <= earlier:
        raise ValueError('Invalid scene acquisition endpoints')
    for source in manifest['assembly']['sources']:
        endpoints = source['acquisition_times']
        seconds = source['interval_seconds']
        if all(t is not None for t in endpoints):
            gap = utc_seconds(endpoints[1])-utc_seconds(endpoints[0])
            if seconds is None or gap <= 0 or not np.isclose(gap, seconds, rtol=0, atol=1e-6):
                raise ValueError('Invalid per-burst acquisition interval')
        elif seconds is not None:
            raise ValueError('Missing acquisition times with a recorded interval')
    inputs.file(raw / 'radar/offsets.h5')
    inputs.file(raw / 'geo/center_geometry.h5')
    with h5py.File(raw / 'radar/offsets.h5', 'r') as h:
        metadata = json.loads(h.attrs['metadata_json'])
        if any(metadata[k] != expected for k, expected in [('matching', manifest['matching']['matching']),
                ('coreg_identity', manifest['coreg_identity']), ('input_provenance', manifest['coreg_specification']['inputs'])]):
            raise ValueError('Radar metadata differ from the pair completion manifest')
        owner, intervals = h['source_burst_index'][:], h['interval_seconds'][:]
        if not np.isin(owner, np.arange(len(sources)+1)).all():
            raise ValueError('Radar source-burst index is invalid')
        for index, source in enumerate(sources, 1):
            values = intervals[owner == index]
            expected = source['interval_seconds']
            if not (np.isnan(values).all() if expected is None else np.all(values == expected)):
                raise ValueError('Radar intervals differ from the source-burst timing table')
    with h5py.File(raw / 'geo/center_geometry.h5', 'r') as h:
        metadata = json.loads(h.attrs['metadata_json'])
        if metadata['match_identity'] != manifest['match_identity']:
            raise ValueError('Center geometry matching identity differs from the pair')
        if not np.array_equal(owner, h['source_burst_index'][:]):
            raise ValueError('Center geometry ownership differs from radar observations')
    selected = raw
    saved = None
    calibration = None
    if correction['enabled']:
        selected = Path(correction['pairs_directory']) / pair
        saved = read_json(selected / 'manifest.json', inputs)
        spec = saved.get('specification', {})
        if saved.get('status') != 'complete' or saved.get('method') != CORRECTION_METHOD or saved.get('identity') != identity(spec):
            raise ValueError('Swath correction is missing, incomplete or unsupported')
        for key, expected in [('match_identity', manifest['match_identity']), ('geocode_identity', manifest['geocode_identity']),
                              ('geometry_identity', frame['geometry_identity']), ('calibration_sha256', correction['calibration_sha256']),
                              ('geocoding', cfg['geocoding'])]:
            if spec.get(key) != expected:
                raise ValueError(f'Stale swath correction: {key}')
        # The correction stage serializes parsed calibration JSON. Its bytes
        # can differ from the pinned input (indentation/key order); authenticate
        # the saved copy with its completion checksum, and the original pin
        # with the correction specification checked above.
        calibration_digest = saved.get('products', {}).get('calibration.json')
        if calibration_digest is None:
            raise ValueError('Saved swath calibration has no completion checksum')
        inputs.file(selected / 'calibration.json', calibration_digest)
        calibration = json.loads((selected / 'calibration.json').read_text())
        if calibration.get('method') != CORRECTION_METHOD:
            raise ValueError('Unsupported saved swath calibration')
        # Legacy implementation checksums describe provenance, not current code.
        # Check the consumed scientific raw dependencies, independent of diagnostics.
        for name, record in saved.get('inputs', {}).items():
            p = Path(name)
            if p == raw / 'radar/offsets.h5' or p == raw / 'geo/center_geometry.h5' or (p.parent == raw / 'geo' and p.stem in PRIMARY):
                inputs.file(p, record['sha256'])
    keys = PRIMARY + (CORRECTION_FLAGS if saved else ())
    paths = {}
    for key in keys:
        path = selected / 'geo' / f'{key}.tif'
        expected = None
        if saved:
            expected = saved.get('products', {}).get(f'geo/{key}.tif')
            if expected is None:
                raise ValueError(f'Corrected {key} has no completion checksum')
        inputs.file(path, expected)
        ds = open_raster(path, grid)
        recorded_match = ds.GetMetadataItem('match_identity') or ds.GetMetadataItem('raw_match_identity')
        if recorded_match is not None and recorded_match != manifest['match_identity']:
            raise ValueError(f'{key} has a different matching identity')
        ds = None
        paths[key] = path
    arrays = {key: open_raster(path, grid) for key, path in paths.items()}
    for section in tiles(grid['shape']):
        valid = read_tile(arrays['valid_mask'], section)
        if not np.isin(valid, [0, 1]).all():
            raise ValueError('valid_mask must contain only 0 and 1')
        for key in PRIMARY[:2]:
            if np.any((valid == 1) & ~np.isfinite(read_tile(arrays[key], section))):
                raise ValueError(f'Valid {key} observations must be finite')
        for key in CORRECTION_FLAGS if saved else ():
            a = read_tile(arrays[key], section)
            if not np.isin(a[valid == 1], [0, 1]).all():
                raise ValueError(f'{key} must contain 0 or 1 on valid observations')
    arrays.clear()
    inputs.unchanged()
    return {'pair': pair, 'manifest': manifest, 'raw': raw, 'paths': paths, 'inputs': inputs,
            'earlier': earlier, 'later': later, 'midpoint': earlier+(later-earlier)/2,
            'correction': saved, 'calibration': calibration}


def variable(h, name, dimensions, values=None, dtype=None, **attrs):
    if dtype is None:
        dtype = np.asarray(values).dtype
    options = {}
    if len(dimensions) >= 2:
        options = {'compression': 'gzip', 'compression_opts': 4, 'shuffle': True, 'fletcher32': True,
                   'chunks': tuple(1 if d == 'time' else min(len(h.dimensions[d]), 256) for d in dimensions)}
    if np.dtype(dtype).kind == 'f':
        options['fillvalue'] = np.nan
    v = h.create_variable(name, dimensions, dtype=dtype, **options)
    v.attrs.update(attrs)
    if values is not None:
        v[...] = values
    return v


def strings(h, name, dimensions, values):
    return variable(h, name, dimensions, np.asarray(values, dtype=object), dtype=h5py.string_dtype('utf-8'))


def timestamp(h, name, dimensions, values):
    return variable(h, name, dimensions, values, dtype='f8', units=TIME_UNITS, calendar=CALENDAR)


def content_hashes(h):
    """Hash stored variables in bounded blocks, including auxiliary metadata."""
    result = {}
    for name, v in h.variables.items():
        digest = hashlib.sha256()
        if v.dimensions[-2:] == ('y', 'x'):
            for index in np.ndindex(v.shape[:-2]):
                for section in tiles(v.shape[-2:]):
                    digest.update(np.asarray(v[(*index, *section)]).tobytes())
        elif v.dtype.kind in 'OSU':
            values = np.asarray(v[...]).ravel()
            digest.update(json.dumps([s.decode() if isinstance(s, bytes) else str(s) for s in values]).encode())
        else:
            digest.update(np.asarray(v[...]).tobytes())
        result[name] = digest.hexdigest()
    return result


def write_package(path, cfg, frame, grid, pairs, omitted, center, association, orbit_path, specification):
    height, width = grid['shape']
    gt = grid['transform']
    source_ids = [s['burst_id'] for s in frame['sources']]
    with h5netcdf.File(path, 'w') as h:
        h.dimensions = {'time': len(pairs), 'y': height, 'x': width, 'burst': len(source_ids), 'enu': 3, 'cartesian': 3, 'bound': 2, 'radar_axis': 2}
        h.attrs.update(schema_version=SCHEMA, package_identity=identity(specification), package_status='partial' if omitted else 'complete',
                       source=specification['source'], EPSG=grid['epsg'], Conventions='CF-1.10', format='xarray',
                       geometry_reference_date=pairs[0]['manifest']['stack_pair']['geometry_reference_date'],
                       geometry_identity=frame['geometry_identity'], layout_identity=frame['layout_identity'],
                       polarization=cfg['inputs']['polarization'], geocoding_resampling='nearest', source_mask=cfg['geocoding']['mask'],
                       axis_sign='later-minus-earlier; positive range away from sensor; positive azimuth increasing reference sensing time',
                       timing_policy='Exact per-burst sensing-start intervals; time is a pair midpoint label, not an independent acquisition',
                       geometry_method=GEOMETRY_METHOD)
        variable(h, 'x', ('x',), gt[0]+(np.arange(width)+.5)*gt[1], units='m', standard_name='projection_x_coordinate', axis='X')
        variable(h, 'y', ('y',), gt[3]+(np.arange(height)+.5)*gt[5], units='m', standard_name='projection_y_coordinate', axis='Y')
        crs = CRS.from_epsg(grid['epsg'])
        v = variable(h, 'crs', (), np.int32(0))
        v.attrs.update(crs.to_cf())
        v.attrs.update(spatial_ref=crs.to_wkt(), GeoTransform=' '.join(map(str, gt)))
        v = timestamp(h, 'time', ('time',), [p['midpoint'] for p in pairs])
        v.attrs.update(standard_name='time', axis='T', bounds='time_bounds', long_name='pair acquisition midpoint')
        timestamp(h, 'time_bounds', ('time', 'bound'), [[p['earlier'], p['later']] for p in pairs])
        timestamp(h, 'earlier_time', ('time',), [p['earlier'] for p in pairs])
        timestamp(h, 'later_time', ('time',), [p['later'] for p in pairs])
        for name in ('names', 'pair_id'):
            strings(h, name, ('time',), [p['pair'] for p in pairs])
        strings(h, 'burst_id', ('burst',), source_ids)
        strings(h, 'enu', ('enu',), ['east', 'north', 'up'])
        strings(h, 'cartesian', ('cartesian',), ['x', 'y', 'z'])
        strings(h, 'radar_axis', ('radar_axis',), ['line', 'sample'])
        variable(h, 'source_radar_offset', ('burst', 'radar_axis'), [s['offset'] for s in frame['sources']], dtype='i4',
                 long_name='burst origin on the shared reference radar grid')
        times = np.full((len(pairs), len(source_ids), 2), np.nan)
        intervals = np.full(times.shape[:2], np.nan)
        for i, pair in enumerate(pairs):
            for j, source in enumerate(pair['manifest']['assembly']['sources']):
                times[i, j] = [utc_seconds(t) for t in source['acquisition_times']]
                intervals[i, j] = source['interval_seconds'] if source['interval_seconds'] is not None else np.nan
        timestamp(h, 'earlier_burst_time', ('time', 'burst'), times[..., 0])
        timestamp(h, 'later_burst_time', ('time', 'burst'), times[..., 1])
        variable(h, 'pair_interval_seconds', ('time', 'burst'), intervals, units='s',
                 long_name='exact later-minus-earlier burst-start interval; index by source_burst_index minus one')
        timestamp(h, 'reference_burst_time', ('burst',), [utc_seconds(s['geometry_reference_time']) for s in frame['sources']])
        strings(h, 'reference_radar_grids_json', ('burst',), [json.dumps(s['coreg']['reference_grid'], sort_keys=True) for s in frame['sources']])
        inventory = {s['date']: s for s in specification['inventory']}
        for label, endpoint in [('earlier', 0), ('later', 1)]:
            scenes = [inventory.get(p['manifest']['stack_pair']['acquisition_dates'][endpoint], {}) for p in pairs]
            strings(h, f'{label}_spacecraft', ('time',), [s.get('mission', '') for s in scenes])
            strings(h, f'{label}_orbit_direction', ('time',), [s.get('direction', '') for s in scenes])
            variable(h, f'{label}_relative_orbit', ('time',), [s.get('relative_orbit', -1) for s in scenes], dtype='i4')
        variable(h, 'source_observation_index', ('y', 'x'), association, long_name='zero-based flattened sparse observation; -1 outside geometry')
        supported = association >= 0
        for key, values in center.items():
            if key == 'geometry_valid':
                continue
            units = ('degrees_east' if key == 'longitude_deg' else 'degrees_north' if key == 'latitude_deg' else
                     'm' if key == 'height_m' else 'm' if key == 'azimuth_ground_spacing_m_per_slc_line' else '1')
            dimensions = ('enu', 'y', 'x') if values.ndim == 3 else ('y', 'x')
            name = {'longitude_deg': 'observation_longitude', 'latitude_deg': 'observation_latitude',
                    'center_line': 'reference_center_line', 'center_sample': 'reference_center_sample'}.get(key, key)
            if key == 'source_burst_index':
                mapped = np.zeros(association.shape, dtype='i4')
            else:
                mapped = np.full((*association.shape, 3) if values.ndim == 3 else association.shape, np.nan)
            mapped[supported] = values.reshape((-1, 3) if values.ndim == 3 else (-1,))[association[supported]]
            v = variable(h, name, dimensions, np.moveaxis(mapped, -1, 0) if values.ndim == 3 else mapped, units=units, grid_mapping='crs')
            if key == 'reference_los_enu':
                v.attrs['direction'] = 'unit vector from ground observation to reference sensor in local east/north/up'
            if key.endswith('sensitivity_enu'):
                v.attrs['definition'] = 'observable displacement metres per metre of local ENU motion; reference geo2rdr central differences'
            if key == 'source_burst_index':
                v.attrs['indexing'] = '1-based burst index; 0 outside geometry'
        cube = {}
        for key in PRIMARY + (CORRECTION_FLAGS if pairs[0]['correction'] else ()):
            integer = key == 'valid_mask' or key in CORRECTION_FLAGS
            cube[key] = variable(h, key, ('time', 'y', 'x'), dtype='u1' if integer else 'f4',
                                 units='m' if 'displacement' in key else '1', grid_mapping='crs')
            if integer:
                cube[key].attrs.update(flag_values=np.array([0, 1], dtype='uint8'), flag_meanings='invalid valid' if key == 'valid_mask' else 'unavailable available')
        for i, pair in enumerate(pairs):
            logging.info('Packaging %s (%d/%d)', pair['pair'], i+1, len(pairs))
            rasters = {key: open_raster(p, grid) for key, p in pair['paths'].items()}
            for section in tiles(grid['shape']):
                valid = read_tile(rasters['valid_mask'], section).astype(bool)
                if np.any(valid & ~supported[section]):
                    raise ValueError(f'{pair["pair"]}: valid observation has no reference geometry association')
                for key, target in cube.items():
                    a = read_tile(rasters[key], section)
                    if key == 'valid_mask' or key in CORRECTION_FLAGS:
                        a = np.where(valid, a, 0).astype('uint8')
                    else:
                        a = np.where(valid, a, np.nan).astype('float32')
                    target[(i, *section)] = a
            rasters.clear()
        strings(h, 'omitted_pairs_json', (), json.dumps(omitted, sort_keys=True))
        strings(h, 'provenance_json', (), json.dumps(specification, sort_keys=True))
        strings(h, 'pair_provenance_json', ('time',), [json.dumps({k: p['manifest'].get(k) for k in
                ('matching', 'geocoding', 'software', 'stack_pair', 'coreg_specification')}, sort_keys=True) for p in pairs])
        if pairs[0]['correction']:
            strings(h, 'calibration_json', (), json.dumps(pairs[0]['calibration'], sort_keys=True))
            strings(h, 'correction_provenance_json', ('time',), [json.dumps({k: p['correction'].get(k) for k in
                    ('method', 'identity', 'specification', 'metadata')}, sort_keys=True) for p in pairs])
        with h5py.File(orbit_path, 'r') as orbit:
            h.dimensions['orbit_state'] = len(orbit['time'])
            timestamp(h, 'reference_orbit_time', ('orbit_state',), orbit['time'][:])
            for key, units in [('position', 'm'), ('velocity', 'm s-1')]:
                v = variable(h, f'reference_orbit_{key}', ('orbit_state', 'cartesian'), orbit[key][:], units=units)
                v.attrs['coordinate_frame'] = 'Earth-centered Earth-fixed XYZ'
        h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)


def validate_package(path):
    """Validate the standalone schema and observation consistency, without stack access."""
    with h5netcdf.File(path, 'r') as h:
        if h.attrs.get('schema_version') != SCHEMA:
            raise ValueError('Unsupported displacement package schema')
        spec = json.loads(h['provenance_json'][()].decode() if isinstance(h['provenance_json'][()], bytes) else h['provenance_json'][()])
        if h.attrs.get('package_identity') != identity(spec):
            raise ValueError('Package provenance fingerprint mismatch')
        if json.loads(h.attrs.get('content_hashes_json', '{}')) != content_hashes(h):
            raise ValueError('Package content checksum mismatch')
        shape = tuple(len(h.dimensions[d]) for d in ('time', 'y', 'x'))
        if not all(shape):
            raise ValueError('Package dimensions must be nonempty')
        for key in PRIMARY:
            if h[key].dimensions != ('time', 'y', 'x') or h[key].shape != shape:
                raise ValueError(f'Invalid dimensions for {key}')
            if h[key].dtype != np.dtype('u1' if key == 'valid_mask' else 'f4'):
                raise ValueError(f'Invalid dtype for {key}')
        if '_FillValue' in h['valid_mask'].attrs:
            raise ValueError('Integer valid_mask must not have a fill value')
        ids = h['pair_id'][:].astype(str)
        if len(set(ids)) != len(ids) or not np.array_equal(ids, h['names'][:].astype(str)):
            raise ValueError('Pair IDs must be unique and agree with names')
        if np.any(np.diff(h['time'][:]) < 0) or np.any(h['later_time'][:] <= h['earlier_time'][:]):
            raise ValueError('Invalid time ordering or endpoints')
        if not np.allclose(h['time'][:], (h['earlier_time'][:]+h['later_time'][:])/2, rtol=0, atol=5e-7):
            raise ValueError('Pair midpoint labels disagree with endpoints')
        intervals = h['pair_interval_seconds'][:]
        if not np.allclose(h['later_burst_time'][:]-h['earlier_burst_time'][:], intervals, rtol=0, atol=1e-6, equal_nan=True):
            raise ValueError('Per-burst endpoints disagree with intervals')
        grid = spec['grid']
        gt = grid['transform']
        if h.attrs['EPSG'] != grid['epsg'] or grid['shape'] != list(shape[1:]):
            raise ValueError('Package grid metadata disagree')
        for key, expected in [('x', gt[0]+(np.arange(shape[2])+.5)*gt[1]), ('y', gt[3]+(np.arange(shape[1])+.5)*gt[5])]:
            if not np.array_equal(h[key][:], expected):
                raise ValueError(f'Invalid {key} pixel-center coordinates')
        for i in range(shape[0]):
            for section in tiles(shape[1:]):
                valid = h['valid_mask'][(i, *section)]
                if not np.isin(valid, [0, 1]).all():
                    raise ValueError('Invalid mask values')
                for key in PRIMARY[:4]:
                    a = h[key][(i, *section)]
                    if not np.isnan(a[valid == 0]).all() or (key in PRIMARY[:2] and not np.isfinite(a[valid == 1]).all()):
                        raise ValueError(f'Inconsistent valid support in {key}')
                owner = h['source_burst_index'][section]
                if np.any((valid == 1) & ((owner <= 0) | (owner > len(h.dimensions['burst'])))):
                    raise ValueError('Valid observation has no source burst')
                if not np.all(np.isfinite(intervals[i, owner[valid == 1]-1]) & (intervals[i, owner[valid == 1]-1] > 0)):
                    raise ValueError('Valid observation has no positive pair interval')
                for key in ('reference_los_enu', 'range_sensitivity_enu', 'azimuth_sensitivity_enu'):
                    if not np.isfinite(h[key][(slice(None), *section)][:, valid == 1]).all():
                        raise ValueError(f'Valid observation lacks {key}')
        return {'path': str(path), 'pairs': shape[0], 'shape': list(shape), 'status': h.attrs['package_status'],
                'omitted': json.loads(h['omitted_pairs_json'][()].decode() if isinstance(h['omitted_pairs_json'][()], bytes) else h['omitted_pairs_json'][()])}


def package_stack(cfg, *, output=None, force=False):
    options = settings(cfg)
    root = Path(cfg['output']['directory'])
    destination = Path(output or options['output'] or root / 'packages' / f'displacements_{options["source"]}.nc').expanduser().resolve()
    # A package may never overwrite a scientific input or an unrelated file.
    if destination.suffix != '.nc' or destination.is_relative_to(root / 'pairs') or destination.is_relative_to(root / 'geometry') or destination.is_relative_to(root / 'acquisitions'):
        raise ValueError('Package output must be a separate .nc file outside scientific product directories')
    if cfg['geocoding'].get('resampling') != 'nearest':
        raise ValueError('Packaging requires nearest-source geocoding; set geocoding.resampling: nearest and run stack geocode (no rematching needed)')
    correction = {'enabled': False}
    if options['source'] == 'swath_corrected':
        correction = correction_settings(cfg)
        if not correction['enabled']:
            raise ValueError('Enable stack.swath_correction and complete correct-swaths before packaging corrected products')
        if destination.is_relative_to(Path(correction['pairs_directory']).resolve()):
            raise ValueError('Package output must be outside the corrected pairs directory')
    with stack_lock(root):
        inputs = Inputs(ChecksumCache())
        catalog = read_json(root / 'stack_manifest.json', inputs)
        frame = read_json(root / 'geometry/manifest.json', inputs)
        if frame.get('status') != 'complete' or identity(frame['specification']) != frame.get('geometry_identity'):
            raise ValueError('Pinned reference geometry is incomplete')
        grid = catalog.get('map_grid') or frame.get('map_grid')
        if not grid or cfg['geocoding']['epsg'] != grid['epsg'] or cfg['geocoding']['spacing_m'] != grid['transform'][1]:
            raise ValueError('Configured geocoding differs from the saved common grid')
        if frame.get('map_grid', grid) != grid:
            raise ValueError('Catalog and reference geometry common grids disagree')
        if grid['transform'][2] != 0 or grid['transform'][4] != 0 or grid['transform'][1] <= 0 or grid['transform'][5] >= 0:
            raise ValueError('Package grid must be north-up and unrotated')
        owner = frame['coreg']['source_index']
        inputs.file(owner, frame['products'][owner])
        pairs, omitted = [], []
        for pair, entry in sorted(catalog['pairs'].items()):
            if not entry.get('active', True):
                continue
            if Path(pair).name != pair or pair.startswith('.'):
                raise ValueError('Invalid pair ID in saved active catalog')
            try:
                logging.info('Checking package source %s', pair)
                p = inspect_pair(cfg, frame, grid, pair, correction, inputs.checksums)
                if pairs:
                    with h5py.File(pairs[0]['raw'] / 'geo/center_geometry.h5', 'r') as a, h5py.File(p['raw'] / 'geo/center_geometry.h5', 'r') as b:
                        for key in ('center_line', 'center_sample', 'longitude_deg', 'latitude_deg', 'height_m', 'azimuth_ground_spacing_m_per_slc_line', 'geometry_valid'):
                            if not np.array_equal(a[key][:], b[key][:], equal_nan=True):
                                raise ValueError(f'Pair source-center geometry differs: {key}')
                pairs.append(p)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                omitted.append({'pair': pair, 'reason': str(exc)})
                logging.warning('Omitting %s: %s', pair, exc)
        if not pairs:
            raise ValueError('No eligible completed pairs: ' + json.dumps(omitted))
        pairs.sort(key=lambda p: (p['midpoint'], p['pair']))
        center, association, orbit_path = supplemental_geometry(root, frame, grid, pairs[0]['raw'] / 'geo/center_geometry.h5', inputs)
        all_inputs = dict(inputs.files)
        for p in pairs:
            all_inputs.update(p['inputs'].files)
        specification = {'schema': SCHEMA, 'source': options['source'], 'grid': grid,
                         'implementation': IMPLEMENTATION,
                         'geometry': frame['geometry_identity'], 'layout': frame['layout_identity'], 'geometry_method': GEOMETRY_METHOD,
                         'pairs': [p['pair'] for p in pairs], 'omitted': omitted,
                         'inventory': catalog.get('inventory', []), 'geocoding': cfg['geocoding'],
                         # Verify catalog revisions during this invocation, but
                         # unrelated progress/metrics are not package dependencies.
                         'inputs': {k: v['sha256'] for k, v in all_inputs.items()
                                    if k != str((root / 'stack_manifest.json').resolve())}}
        if destination.exists():
            with h5netcdf.File(destination, 'r') as h:
                if h.attrs.get('schema_version') != SCHEMA:
                    raise ValueError(f'Refusing to replace an unrelated output: {destination}')
                unchanged = h.attrs.get('package_identity') == identity(specification)
            if unchanged and not force:
                try:
                    result = validate_package(destination)
                except (OSError, ValueError, KeyError) as exc:
                    logging.warning('Rebuilding damaged package: %s', exc)
                else:
                    inputs.unchanged()
                    for p in pairs:
                        p['inputs'].unchanged()
                    return result | {'reused': True}
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f'.{destination.stem}-', dir=destination.parent) as folder:
            temp = Path(folder) / destination.name
            write_package(temp, cfg, frame, grid, pairs, omitted, center, association, orbit_path, specification)
            result = validate_package(temp)
            inputs.unchanged()
            for p in pairs:
                p['inputs'].unchanged()
            os.replace(temp, destination)
        return result | {'path': str(destination), 'reused': False}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='glacier-offsets package', description=__doc__)
    parser.add_argument('config')
    parser.add_argument('-o', '--output', help='Override packaging.output (relative to current directory)')
    parser.add_argument('--force', action='store_true', help='Rebuild only this package')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        report = package_stack(load_status_config(args.config), output=args.output, force=args.force)
        print(json.dumps(report, indent=2))
        return 0
    except KeyboardInterrupt:
        logging.error('Interrupted; previous completed package is preserved')
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as exc:
        logging.error('%s', exc)
        return 2

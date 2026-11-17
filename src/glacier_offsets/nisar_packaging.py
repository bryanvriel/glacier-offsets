"""Offline import of enriched NISAR GOFF subsets; native geometry is lazy."""
import json
from importlib.metadata import PackageNotFoundError, version
import logging
import os
from pathlib import Path
import tempfile
from time import perf_counter

import h5netcdf
import h5py
import numpy as np
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

from .displacement import NISAR_SCHEMA, text, valid_nisar_mask, covariance_valid, validate_nisar
from .products import checksum, identity
from .velocity_io import revision, serializable
from .nisar_geometry import GeometryRunner
from .nisar_grid import axis_selection, decimation_factor, sampling_record


METHOD = 'nisar-reference-enu-central-difference-v1'
VARIABLES = ('alongTrackOffset', 'slantRangeOffset', 'alongTrackOffsetVariance',
             'slantRangeOffsetVariance', 'crossOffsetVariance', 'snr', 'correlationSurfacePeak')
CUBES = ('referenceZeroDopplerAzimuthTime', 'secondaryZeroDopplerAzimuthTime',
         'referenceSlantRange', 'secondarySlantRange', 'groundTrackVelocity',
         'losUnitVectorX', 'losUnitVectorY', 'alongTrackUnitVectorX', 'alongTrackUnitVectorY', 'incidenceAngle')


def utc(value):
    return float(np.datetime64(text(value).removeprefix('seconds since '), 'ns').astype('int64'))/1e9


def metadata_json(group):
    values = {}
    def convert(a):
        if isinstance(a, (bytes, np.bytes_)):
            return a.decode()
        if isinstance(a, np.ndarray):
            return [convert(v) for v in a.tolist()] if a.ndim else convert(a.item())
        if isinstance(a, (list, tuple)):
            return [convert(v) for v in a]
        return serializable(a)
    def visit(name, item):
        values[name] = {'attributes': {k: convert(v) for k, v in item.attrs.items()
                                     if k not in ('DIMENSION_LIST', 'REFERENCE_LIST')}}
        if isinstance(item, h5py.Dataset):
            values[name]['value'] = convert(item[()])
    for key in ('metadata', 'identification', 'grid'):
        group[key].visititems(lambda name, item, key=key: visit(key+'/'+name, item))
    values['/'] = {'attributes': {k: convert(v) for k, v in group.attrs.items()}}
    return json.dumps(values, sort_keys=True)


def inspect(path, layer):
    """Fail before publication when old/incomplete subsets are selected."""
    with h5py.File(path, 'r') as h:
        required = ['x', 'y', 'mask', 'surfaceHeight', 'grid/centerFrequency', 'grid/projection',
                    'metadata/processingInformation/parameters/reference/frequencyA/zeroDopplerTimeSpacing',
                    'metadata/processingInformation/parameters/reference/frequencyA/slantRangeSpacing',
                    'metadata/processingInformation/parameters/reference/frequencyA/slantRangeStart',
                    'metadata/processingInformation/parameters/reference/frequencyA/numberOfAzimuthLines',
                    'metadata/processingInformation/parameters/reference/frequencyA/numberOfRangeSamples',
                    'metadata/processingInformation/algorithms/geocoding/floatingGeocodingInterpolation',
                    *[f'layer{layer}/{k}' for k in VARIABLES], *[f'radarGrid/{k}' for k in CUBES],
                    *[f'radarGrid/{k}' for k in ('xCoordinates', 'yCoordinates', 'heightAboveEllipsoid')],
                    *[f'metadata/orbit/{a}/{k}' for a in ('reference', 'secondary') for k in ('time', 'position', 'velocity')]]
        missing = [k for k in required if k not in h]
        if missing:
            raise ValueError(f'{path}: incomplete enriched subset; regenerate it; missing {missing}')
        ident = h['identification']
        keys = ('granuleId', 'productType', 'platformName', 'productVersion', 'productSpecificationVersion',
                'compositeReleaseId', 'trackNumber', 'frameNumber', 'lookDirection', 'orbitPassDirection',
                'referenceAbsoluteOrbitNumber', 'secondaryAbsoluteOrbitNumber',
                'referenceZeroDopplerStartTime', 'secondaryZeroDopplerStartTime')
        if any(k not in ident for k in keys):
            raise ValueError(f'{path}: missing product identification')
        info = {k: text(ident[k][()]) for k in keys}
        if info['platformName'] != 'NISAR' or info['productType'] != 'GOFF':
            raise ValueError(f'{path}: expected NISAR GOFF')
        if (int(h['grid/projection'][()]) != 3031 or int(h.attrs.get('epsg', 0)) != 3031
                or text(h['metadata/processingInformation/algorithms/geocoding/floatingGeocodingInterpolation'][()]).upper() != 'BILINEAR'):
            raise ValueError(f'{path}: expected EPSG:3031 bilinear GOFF')
        x, y = h['x'][:], h['y'][:]
        if (min(len(x), len(y)) < 2 or not np.all(np.diff(x) == 80) or not np.all(np.diff(y) == -80)
                or not np.allclose(np.mod(x, 80), 40, rtol=0, atol=1e-6)
                or not np.allclose(np.mod(y, 80), 40, rtol=0, atol=1e-6)):
            raise ValueError(f'{path}: unaligned GOFF grid')
        for k in ['mask', 'surfaceHeight', *[f'layer{layer}/{k}' for k in VARIABLES]]:
            if h[k].shape != (len(y), len(x)):
                raise ValueError(f'{path}: invalid shape: {k}')
        for k in VARIABLES:
            expected = 'meters^2' if 'Variance' in k else 'meters' if k.endswith('Offset') else '1'
            if text(h[f'layer{layer}/{k}'].attrs.get('units', '')) != expected:
                raise ValueError(f'{path}: unexpected units for {k}; expected {expected}')
        if h['surfaceHeight'].attrs.get('height_reference') != 'WGS84 ellipsoid':
            raise ValueError(f'{path}: surface heights must reference WGS84 ellipsoid')
        info.update(path=str(path), sha256=checksum(path), revision=list(revision(path)),
                    x=x, y=y, earlier=utc(info['referenceZeroDopplerStartTime']),
                    later=utc(info['secondaryZeroDopplerStartTime']))
        if info['later'] <= info['earlier']:
            raise ValueError(f'{path}: acquisition endpoints are not increasing')
        # PR in the granule name denotes processing type, not data maturity.
        url = text(h.attrs.get('url', '')).upper()
        doi = text(ident['productDoi'][()]).upper() if 'productDoi' in ident else ''
        if 'PROVISIONAL' in url or doi.endswith('-P1'):
            info['maturity'] = 'PROVISIONAL'
        elif 'BETA' in url or doi.endswith('-B1'):
            info['maturity'] = 'BETA'
        elif '/NISAR_L2_GOFF_V' in url:
            info['maturity'] = 'VALIDATED'
        else:
            raise ValueError(f'{path}: cannot identify GOFF collection maturity from URL/DOI')
        info['signature'] = [info[k] for k in ('maturity', 'compositeReleaseId', 'productVersion', 'productSpecificationVersion')]
        return info


def interpolators(h):
    axes = tuple(h[f'radarGrid/{k}'][:] for k in ('heightAboveEllipsoid', 'yCoordinates', 'xCoordinates'))
    return {k: RegularGridInterpolator(axes, h[f'radarGrid/{k}'][:], bounds_error=False, fill_value=np.nan) for k in CUBES}


def native_geometry(h, points, interpolated):
    """Evaluate sensitivities at supplied (height, northing, easting) cell centers."""
    import isce3
    orbit = isce3.core.Orbit.load_from_h5(h['metadata/orbit/reference'])
    wave = 299792458./float(h['grid/centerFrequency'][()])
    look = isce3.core.LookSide.Left if text(h['identification/lookDirection'][()]).lower() == 'left' else isce3.core.LookSide.Right
    ellipsoid, doppler = isce3.core.Ellipsoid(), isce3.core.LUT2d()
    geographic = Transformer.from_crs(3031, 4326, always_xy=True)
    forward = Transformer.from_crs(4979, 4978, always_xy=True)
    inverse = Transformer.from_crs(4978, 4979, always_xy=True)
    lon, lat = geographic.transform(points[:, 2], points[:, 1])
    xyz = np.stack(forward.transform(lon, lat, points[:, 0]), -1)
    lam, phi = np.deg2rad(lon), np.deg2rad(lat)
    basis = np.stack((np.stack((-np.sin(lam), np.cos(lam), np.zeros_like(lam)), -1),
                      np.stack((-np.sin(phi)*np.cos(lam), -np.sin(phi)*np.sin(lam), np.cos(phi)), -1),
                      np.stack((np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)), -1)), -2)
    result = np.full((len(points), 3, 3), np.nan)
    def solve(p):
        lo, la, z = inverse.transform(*p)
        return isce3.geometry.geo2rdr([np.deg2rad(lo), np.deg2rad(la), z], ellipsoid, orbit, doppler, wave, look)
    for j in np.flatnonzero(np.isfinite(points).all(-1) & np.isfinite(interpolated['groundTrackVelocity'])):
        try:
            t, _ = solve(xyz[j]); position, _ = orbit.interpolate(t)
            los = np.asarray(position)-xyz[j]
            result[j, 0] = basis[j] @ (los/np.linalg.norm(los))
            for a in range(3):
                tp, rp = solve(xyz[j]+basis[j, a]); tm, rm = solve(xyz[j]-basis[j, a])
                result[j, 1, a] = (rp-rm)/2
                result[j, 2, a] = (tp-tm)/2*interpolated['groundTrackVelocity'][j]
        except RuntimeError:
            continue
    return result


def select_revisions(records, policy):
    """Resolve only counter revisions of otherwise identical granule IDs.

    The counter is a selection rule, not an accuracy ranking. Different release,
    mode, coverage/accuracy flags or acquisition identifiers remain ambiguous.
    """
    if policy not in ('error', 'latest'):
        raise ValueError('NISAR revision policy must be error or latest')
    groups = {}
    for record in records:
        key = tuple(record[k] for k in ('trackNumber', 'frameNumber', 'referenceAbsoluteOrbitNumber', 'secondaryAbsoluteOrbitNumber'))
        groups.setdefault(key, []).append(record)
    selected, superseded = [], []
    for key, candidates in sorted(groups.items()):
        if len(candidates) == 1:
            selected.extend(candidates)
            continue
        names = ', '.join(sorted(r['granuleId'] for r in candidates))
        detail = f'track {key[0]}, frame {key[1]}, reference/secondary orbits {key[2]}/{key[3]}: {names}'
        if policy == 'error':
            raise ValueError('Conflicting reprocessed copies of the same acquisition pair/frame; '+detail+
                             '. Use --nisar-revisions latest to select the highest final counter, or provide explicit files')
        parsed = [r['granuleId'].rpartition('_') for r in candidates]
        if any(not sep or not counter.isdigit() for _, sep, counter in parsed) or len({base for base, _, _ in parsed}) != 1:
            raise ValueError('Conflicting reprocessed products differ beyond their final revision counter; '+detail+
                             '. Select the intended input files explicitly')
        counters = [int(counter) for _, _, counter in parsed]
        if len(set(counters)) != len(counters):
            raise ValueError('Ambiguous duplicate revision counter; '+detail)
        winner = candidates[int(np.argmax(counters))]
        selected.append(winner)
        for r in candidates:
            if r is winner:
                continue
            item = {k: r[k] for k in ('granuleId', 'path', 'sha256')}
            item['selected_granule'] = winner['granuleId']
            superseded.append(item)
    return selected, sorted(superseded, key=lambda r: (r['granuleId'], r['path']))


def package_nisar(paths, *, layer=1, output, force=False, revisions='error', workers=None, geometry_cache=None, decimation=1):
    from .packaging import variable, strings, timestamp, content_hashes
    from .tides import output_lock
    started = perf_counter()
    decimation = decimation_factor(decimation)
    sampling = sampling_record(decimation)
    step = sampling['posting_m']
    destination = Path(output).expanduser().resolve()
    if destination.suffix != '.nc' or layer not in (1, 2, 3):
        raise ValueError('NISAR packaging needs a .nc output and layer 1, 2, or 3')
    files = []
    for p in paths:
        p = Path(p).expanduser().resolve()
        files.extend(sorted(p.glob('*.h5')) if p.is_dir() else [p])
    if not files or len(set(files)) != len(files) or any(not p.is_file() or p.suffix != '.h5' for p in files):
        raise ValueError('Select distinct completed .h5 subsets')
    records = []
    for i, p in enumerate(files):
        phase = perf_counter()
        logging.info('Inspecting/checksumming NISAR input %d/%d: %s', i+1, len(files), p.name)
        records.append(inspect(p, layer))
        logging.info('Inspected NISAR input %d/%d in %.1fs', i+1, len(files), perf_counter()-phase)
    if len({r['granuleId'] for r in records}) != len(records) or any(r['signature'] != records[0]['signature'] for r in records):
        raise ValueError('Duplicate granule or incompatible maturity/processing release')
    inspected = records
    records, superseded = select_revisions(records, revisions)
    for r in superseded:
        logging.warning('Selecting %s instead of %s (highest revision counter)', r['selected_granule'], r['granuleId'])
    records.sort(key=lambda r: ((r['earlier']+r['later'])/2, r['granuleId']))
    x = np.arange(min(r['x'][0] for r in records), max(r['x'][-1] for r in records)+40, 80.)
    y = np.arange(max(r['y'][0] for r in records), min(r['y'][-1] for r in records)-40, -80.)
    x, y = x[axis_selection(x, decimation)], y[axis_selection(y, decimation)]
    for r in records:
        r['source_selection'] = (axis_selection(r['y'], decimation), axis_selection(r['x'], decimation))
        r['y'], r['x'] = r['y'][r['source_selection'][0]], r['x'][r['source_selection'][1]]
        r['sampling'] = sampling
    if not len(x) or not len(y) or not any(len(r['x']) and len(r['y']) for r in records):
        raise ValueError('Decimation leaves no sampled GOFF cells; use a smaller factor or larger subset')
    grid = {'epsg': 3031, 'shape': [len(y), len(x)], 'transform': [x[0]-step/2, step, 0., y[0]+step/2, 0., -step]}
    implementation = {k: checksum(Path(__file__).with_name(k)) for k in ('nisar_packaging.py', 'nisar_geometry.py', 'nisar_grid.py', 'displacement.py')}
    dependencies = {}
    for name in ('numpy', 'scipy', 'pyproj'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    try:
        import isce3
        dependencies['isce3'] = isce3.__version__
    except ImportError:
        dependencies['isce3'] = None
    fingerprint = {'method': METHOD, 'implementation': implementation, 'dependencies': dependencies}
    cache = (None if geometry_cache is False else Path(geometry_cache).expanduser().resolve()
             if geometry_cache is not None else destination.parent/'.nisar-geometry-cache')
    geometry = GeometryRunner(workers, cache, fingerprint)
    spec = {'schema': NISAR_SCHEMA, 'mission': 'NISAR', 'layer': layer, 'frequency': 'A', 'polarization': 'HH',
            'release': records[0]['signature'], 'grid': grid, 'geometry_method': METHOD,
            'sampling': sampling,
            'implementation': implementation, 'geometry_dependencies': dependencies,
            'revision_selection': {'policy': revisions, 'superseded_inputs': superseded},
            'inputs': [{k: r[k] for k in ('granuleId', 'path', 'sha256')} for r in records]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(destination.parent), geometry:
        if destination.exists():
            with h5netcdf.File(destination) as old:
                if old.attrs.get('schema_version') != NISAR_SCHEMA:
                    raise ValueError('Refusing to replace an unrelated file')
                if old.attrs['package_identity'] == identity(spec) and not force:
                    logging.info('Validating existing NISAR package for reuse: %s', destination)
                    return validate_nisar(destination) | {'reused': True, 'superseded_products': superseded}
            if not force:
                raise ValueError('Output differs; choose another output or use --force')
        with tempfile.TemporaryDirectory(prefix='.nisar-package-', dir=destination.parent) as tmp:
            logging.info('Packaging %d NISAR products with %d geometry worker(s); cache: %s',
                         len(records), geometry.workers, cache if cache is not None else 'disabled')
            target = Path(tmp)/destination.name
            with h5netcdf.File(target, 'w') as out:
                out.dimensions = {'time': len(records), 'y': len(y), 'x': len(x), 'enu': 3, 'bound': 2}
                out.attrs.update(schema_version=NISAR_SCHEMA, package_identity=identity(spec), package_status='complete',
                                 mission='NISAR', source='raw', EPSG=3031, Conventions='CF-1.10', format='xarray',
                                 observation_support='geocoded_cell', geocoding_resampling='bilinear',
                                 matching_layer=layer, frequency='A', polarization='HH', geometry_method=METHOD,
                                 decimation_factor=decimation, map_posting_m=step, sampling_method='integer_decimation',
                                 data_maturity=records[0]['maturity'],
                                 revision_policy=revisions, superseded_product_count=len(superseded),
                                 axis_sign='later-minus-earlier; positive range away from sensor; positive azimuth increasing sensing time',
                                 spatial_support='correlated matching windows and bilinear resampling; center-based geometry linearization')
                for k, a in [('x', x), ('y', y)]:
                    variable(out, k, (k,), a, units='m', standard_name=f'projection_{k}_coordinate', axis=k.upper())
                crs = variable(out, 'crs', (), np.int32(0)); crs.attrs.update(CRS.from_epsg(3031).to_cf())
                crs.attrs.update(spatial_ref=CRS.from_epsg(3031).to_wkt(), GeoTransform=' '.join(map(str, grid['transform'])))
                strings(out, 'enu', ('enu',), ['east', 'north', 'up'])
                for name in ('pair_id', 'names'):
                    strings(out, name, ('time',), [r['granuleId'] for r in records])
                for label in ('earlier', 'later'):
                    timestamp(out, label+'_time', ('time',), [r[label] for r in records])
                    strings(out, label+'_spacecraft', ('time',), ['NISAR']*len(records))
                timestamp(out, 'time', ('time',), [(r['earlier']+r['later'])/2 for r in records]).attrs.update(bounds='time_bounds', axis='T')
                timestamp(out, 'time_bounds', ('time', 'bound'), [[r['earlier'], r['later']] for r in records])
                for dest, source in [('track_number', 'trackNumber'), ('frame_number', 'frameNumber'),
                                     ('reference_absolute_orbit', 'referenceAbsoluteOrbitNumber'), ('secondary_absolute_orbit', 'secondaryAbsoluteOrbitNumber')]:
                    variable(out, dest, ('time',), [int(r[source]) for r in records], dtype='i4')
                for k in ('lookDirection', 'orbitPassDirection'):
                    strings(out, k, ('time',), [r[k] for r in records])
                strings(out, 'source_metadata_json', ('time',), ['']*len(records))
                floats = {'range_displacement_m': 'm', 'azimuth_displacement_m': 'm', 'snr': '1', 'correlation_peak': '1',
                          'range_displacement_sigma_m': 'm', 'azimuth_displacement_sigma_m': 'm',
                          'range_azimuth_displacement_covariance_m2': 'm2', 'height_m': 'm',
                          'azimuth_ground_spacing_m_per_slc_line': 'm', 'ground_track_velocity_m_s': 'm s-1'}
                for k, unit in floats.items():
                    variable(out, k, ('time', 'y', 'x'), dtype='f4', units=unit, grid_mapping='crs')
                for k in VARIABLES:
                    unit = 'm2' if 'Variance' in k else 'm' if k.endswith('Offset') else '1'
                    variable(out, 'source_'+k, ('time', 'y', 'x'), dtype='f4', units=unit,
                             definition='unaltered source GOFF values, including invalid support')
                for k in ('earlier_observation_time', 'later_observation_time'):
                    timestamp(out, k, ('time', 'y', 'x'), None)
                for k in ('valid_mask', 'geometry_valid_mask', 'uncertainty_available'):
                    variable(out, k, ('time', 'y', 'x'), dtype='u1', units='1')
                variable(out, 'input_mask', ('time', 'y', 'x'), dtype='u4', units='1')
                variable(out, 'geometry_flags', ('time', 'y', 'x'), dtype='u1', units='1', flag_masks=np.array([1, 2, 4], 'u1'),
                         flag_meanings='surface_height_missing cube_unavailable native_geometry_unavailable')
                for k in ('reference_los_enu', 'range_sensitivity_enu', 'azimuth_sensitivity_enu'):
                    variable(out, k, ('time', 'enu', 'y', 'x'), dtype='f8', units='1', grid_mapping='crs')
                xx, yy = np.meshgrid(x, y)
                lon, lat = Transformer.from_crs(3031, 4326, always_xy=True).transform(xx, yy)
                for k, values, unit in [('observation_longitude', lon, 'degrees_east'), ('observation_latitude', lat, 'degrees_north')]:
                    variable(out, k, ('y', 'x'), values, units=unit)
                for i, r in enumerate(records):
                    product_started = perf_counter()
                    logging.info('Importing NISAR %s (%d/%d)', r['granuleId'], i+1, len(records))
                    with h5py.File(r['path'], 'r') as h:
                        out['source_metadata_json'][i] = metadata_json(h)
                        out['input_mask'][i] = np.uint32(255)
                        if not len(r['x']) or not len(r['y']):
                            logging.info('No cells on the selected lattice in %s; retaining empty product slice', r['granuleId'])
                            continue
                        ro = int(round((y[0]-r['y'][0])/step)); co = int(round((r['x'][0]-x[0])/step))
                        for sec, a, mask, z, cube, native in geometry.tiles(h, r, layer):
                            dest = (slice(ro+sec[0].start, ro+sec[0].stop), slice(co+sec[1].start, co+sec[1].stop)); slot = (i, *dest)
                            for k in VARIABLES: out['source_'+k][slot] = a[k]
                            finite = np.isfinite(native).all(axis=(-1, -2))
                            cube_good = np.isfinite(np.stack(list(cube.values()), -1)).all(-1).reshape(z.shape)
                            geom = finite & cube_good & np.isfinite(z)
                            valid = valid_nisar_mask(mask) & np.isfinite(a['alongTrackOffset']) & np.isfinite(a['slantRangeOffset']) & geom
                            out['input_mask'][slot] = mask; out['height_m'][slot] = z
                            out['valid_mask'][slot] = valid.astype('u1'); out['geometry_valid_mask'][slot] = geom.astype('u1')
                            out['geometry_flags'][slot] = ((~np.isfinite(z)).astype('u1')+2*(~cube_good).astype('u1')+4*(~finite).astype('u1'))
                            for j, k in enumerate(('reference_los_enu', 'range_sensitivity_enu', 'azimuth_sensitivity_enu')):
                                out[k][(i, slice(None), *dest)] = np.moveaxis(native[..., j, :], -1, 0)
                            for k, source in [('range_displacement_m', 'slantRangeOffset'), ('azimuth_displacement_m', 'alongTrackOffset'), ('snr', 'snr'), ('correlation_peak', 'correlationSurfacePeak')]:
                                out[k][slot] = np.where(valid, a[source], np.nan)
                            vr, va, cross = (a[k] for k in ('slantRangeOffsetVariance', 'alongTrackOffsetVariance', 'crossOffsetVariance'))
                            unc = covariance_valid(vr, va, cross) & valid
                            for k, val in [('range_displacement_sigma_m', np.sqrt(np.maximum(vr, 0))), ('azimuth_displacement_sigma_m', np.sqrt(np.maximum(va, 0))), ('range_azimuth_displacement_covariance_m2', cross)]:
                                out[k][slot] = np.where(unc, val, np.nan)
                            out['uncertainty_available'][slot] = unc.astype('u1')
                            gtv = cube['groundTrackVelocity'].reshape(z.shape)
                            out['ground_track_velocity_m_s'][slot] = gtv
                            out['azimuth_ground_spacing_m_per_slc_line'][slot] = gtv*float(h['metadata/processingInformation/parameters/reference/frequencyA/zeroDopplerTimeSpacing'][()])
                            for label, source, orbit in [('earlier', 'referenceZeroDopplerAzimuthTime', 'reference'), ('later', 'secondaryZeroDopplerAzimuthTime', 'secondary')]:
                                epoch = utc(h['radarGrid/'+source].attrs['units'])
                                out[label+'_observation_time'][slot] = cube[source].reshape(z.shape)+epoch
                    logging.info('Imported NISAR product %d/%d in %.1fs', i+1, len(records), perf_counter()-product_started)
                strings(out, 'provenance_json', (), json.dumps(spec, sort_keys=True))
                phase = perf_counter()
                logging.info('Hashing completed NISAR package arrays')
                out.attrs['content_hashes_json'] = json.dumps(content_hashes(out), sort_keys=True)
                logging.info('Package array hashing completed in %.1fs', perf_counter()-phase)
            phase = perf_counter()
            logging.info('Validating completed NISAR package')
            result = validate_nisar(target)
            logging.info('Package validation completed in %.1fs', perf_counter()-phase)
            phase = perf_counter()
            for i, r in enumerate(inspected):
                logging.info('Rechecking source checksum %d/%d: %s', i+1, len(inspected), r['granuleId'])
                if list(revision(r['path'])) != r['revision'] or checksum(r['path']) != r['sha256']:
                    raise ValueError('Source subset changed during packaging')
            logging.info('Source verification completed in %.1fs', perf_counter()-phase)
            os.replace(target, destination)
        elapsed = perf_counter()-started
        logging.info('Published NISAR package in %.1fs; %s', elapsed, geometry.stats)
        return result | {'path': str(destination), 'reused': False, 'superseded_products': superseded,
                         'performance': {'workers': geometry.workers, 'elapsed_seconds': elapsed, **geometry.stats}}

"""Supplemental reference geometry, independent of scientific stack identities."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from pyproj import Transformer

from .geocode import make_mapper
from .products import checksum, identity, write_json

METHOD = 'reference-enu-central-difference-v1'
IMPLEMENTATION = checksum(__file__)


def utc_seconds(value):
    """Convert recorded UTC timestamps without assuming the machine timezone."""
    if value is None:
        return np.nan
    t = datetime.fromisoformat(str(value).replace('Z', '+00:00').removeprefix('UTC='))
    return t.replace(tzinfo=timezone.utc).timestamp() if t.tzinfo is None else t.timestamp()


def retain_reference_orbit(root, orbit_record):
    """Snapshot a checksum-pinned EOF; no SAFE or DEM access is needed.

    The small companion manifest authenticates the snapshot on later offline
    invocations. Neither file participates in the coregistration identity.
    Callers hold the stack writer lock.
    """
    root = Path(root)
    path, record_path = root / 'reference_orbit.h5', root / 'reference_orbit.json'
    if path.is_file() and record_path.is_file():
        saved = json.loads(record_path.read_text())
        if saved.get('source_sha256') == orbit_record['sha256'] and checksum(path) == saved.get('sha256'):
            return path
        raise ValueError('Supplemental reference orbit is damaged or differs from the pinned orbit')
    source = Path(orbit_record['path'])
    if not source.is_file():
        raise ValueError(f'Reference orbit snapshot is missing; reconnect the pinned orbit once: {source}')
    digest = checksum(source)
    if digest != orbit_record['sha256']:
        raise ValueError('Reference orbit checksum differs from the pinned reference')
    records = []
    for node in ET.parse(source).getroot().iter():
        if node.tag.split('}')[-1] != 'OSV':
            continue
        fields = {e.tag.split('}')[-1]: e.text for e in node}
        records.append([utc_seconds(fields['UTC']), *[float(fields[k]) for k in ('X', 'Y', 'Z', 'VX', 'VY', 'VZ')]])
    data = np.asarray(records, dtype='float64')
    if data.ndim != 2 or data.shape[0] < 4 or data.shape[1] != 7 or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError('Pinned orbit has invalid state vectors')
    if checksum(source) != digest:
        raise ValueError('Pinned orbit changed during enrichment')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reference-orbit-', dir=root) as folder:
        temp = Path(folder) / path.name
        with h5py.File(temp, 'w') as h:
            h.attrs['source_sha256'] = digest
            h.create_dataset('time', data=data[:, 0])
            h.create_dataset('position', data=data[:, 1:4])
            h.create_dataset('velocity', data=data[:, 4:7])
        temp.replace(path)
    write_json(record_path, {'source_sha256': digest, 'sha256': checksum(path)})
    return path


def native_orbit(path, sources):
    import isce3
    with h5py.File(path, 'r') as h:
        times, positions, velocities = h['time'][:], h['position'][:], h['velocity'][:]
    # EOFs can span a UTC leap second, giving a 9/11 s step in otherwise
    # 10-second records. As in the scene reader, only construct a native orbit
    # around the reference acquisition; retain the complete EOF in the snapshot.
    start = min(utc_seconds(s['geometry_reference_time']) for s in sources)-60
    stop = max(utc_seconds(s['geometry_reference_time']) +
               s['coreg']['reference_grid']['length']/s['coreg']['reference_grid']['prf'] for s in sources)+60
    take = (times >= start) & (times <= stop)
    times, positions, velocities = times[take], positions[take], velocities[take]
    if len(times) < 4 or times[0] > start+30 or times[-1] < stop-30:
        raise ValueError('Reference orbit snapshot does not cover the reference acquisition')
    vectors = [isce3.core.StateVector(isce3.core.DateTime(datetime.fromtimestamp(float(t), timezone.utc).replace(tzinfo=None)), p, v)
               for t, p, v in zip(times, positions, velocities)]
    return isce3.core.Orbit(vectors)


def enu_geometry(lon, lat, height, scale, owner, sources, orbit):
    """Return LOS and observable sensitivities to one metre of local ENU motion.

    A symmetric +/-1 m ECEF perturbation uses the same zero-Doppler reference
    orbit/grid as velocity-prior projection. Azimuth sensitivity is scaled by
    local DEM-intersecting ground metres per reference line, not platform speed.
    """
    import isce3
    forward = Transformer.from_crs(4979, 4978, always_xy=True)
    inverse = Transformer.from_crs(4978, 4979, always_xy=True)
    xyz = np.stack(forward.transform(lon, lat, height), axis=-1)
    lam, phi = np.deg2rad(lon), np.deg2rad(lat)
    basis = np.stack((np.stack((-np.sin(lam), np.cos(lam), np.zeros_like(lam)), -1),
                      np.stack((-np.sin(phi)*np.cos(lam), -np.sin(phi)*np.sin(lam), np.cos(phi)), -1),
                      np.stack((np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)), -1)), -2)
    result = {k: np.full((*lon.shape, 3), np.nan) for k in ('reference_los_enu', 'range_sensitivity_enu', 'azimuth_sensitivity_enu')}
    good = np.isfinite(xyz).all(-1) & np.isfinite(scale) & (scale > 0) & (owner > 0)
    ellipsoid, doppler = isce3.core.Ellipsoid(), isce3.core.LUT2d()
    for n, index in enumerate(zip(*np.where(good))):
        if n % 10000 == 0:
            logging.info('Reference viewing geometry: %d/%d observations', n, int(good.sum()))
        grid = sources[int(owner[index])-1]['coreg']['reference_grid']
        look = isce3.core.LookSide.Right if grid['lookside'].lower() == 'right' else isce3.core.LookSide.Left
        def solve(point):
            lo, la, z = inverse.transform(*point)
            return isce3.geometry.geo2rdr([np.deg2rad(lo), np.deg2rad(la), z], ellipsoid, orbit, doppler, grid['wavelength'], look)
        try:
            t, _ = solve(xyz[index])
            position, _ = orbit.interpolate(t)
            los = np.asarray(position) - xyz[index]
            result['reference_los_enu'][index] = basis[index] @ (los / np.linalg.norm(los))
            for axis in range(3):
                tp, rp = solve(xyz[index]+basis[index][axis])
                tm, rm = solve(xyz[index]-basis[index][axis])
                result['range_sensitivity_enu'][index][axis] = (rp-rm)/2
                result['azimuth_sensitivity_enu'][index][axis] = (tp-tm)/2*grid['prf']*scale[index]
        except RuntimeError as exc:
            raise ValueError(f'Cannot derive reference geometry at sparse observation {index}: {exc}') from exc
    return result


def supplemental_geometry(root, frame, grid, centers_path, inputs):
    """Build/check a sparse-geometry cache, then map exact source associations."""
    orbit_path = retain_reference_orbit(Path(root) / 'geometry', frame['specification']['inputs']['orbit'])
    inputs.file(orbit_path)
    inputs.file(orbit_path.with_suffix('.json'))
    inputs.file(centers_path)
    with h5py.File(centers_path, 'r') as h:
        center = {k: h[k][:] for k in ('longitude_deg', 'latitude_deg', 'height_m', 'center_line', 'center_sample',
                                     'azimuth_ground_spacing_m_per_slc_line', 'geometry_valid', 'source_burst_index')}
    signature = {'method': METHOD, 'implementation': IMPLEMENTATION,
                 'geometry': frame['geometry_identity'], 'layout': frame['layout_identity'],
                 'orbit': inputs.checksums(orbit_path),
                 'centers': {k: identity(v.tolist()) for k, v in center.items() if k != 'source_burst_index'}}
    # Missing-burst masks must not change the static ownership/geometry cache.
    from .assembly import source_indices
    owner = source_indices(frame['coreg'], center['center_line'], center['center_sample'])
    center['source_burst_index'] = owner
    signature['owner'] = identity(owner.tolist())
    cache = Path(root) / 'geometry' / f'packaging_geometry_{identity(signature)}.h5'
    record = cache.with_suffix('.json')
    if cache.is_file() and record.is_file():
        saved = json.loads(record.read_text())
        if saved.get('signature') != signature or saved.get('sha256') != checksum(cache):
            raise ValueError('Supplemental packaging geometry cache checksum mismatch')
    else:
        fields = enu_geometry(center['longitude_deg'], center['latitude_deg'], center['height_m'],
                              center['azimuth_ground_spacing_m_per_slc_line'], owner, frame['sources'], native_orbit(orbit_path, frame['sources']))
        with tempfile.TemporaryDirectory(prefix='.packaging-geometry-', dir=cache.parent) as folder:
            temp = Path(folder) / cache.name
            with h5py.File(temp, 'w') as h:
                for key, value in fields.items():
                    h.create_dataset(key, data=value, compression='gzip', shuffle=True)
            temp.replace(cache)
        write_json(record, {'signature': signature, 'sha256': checksum(cache)})
    inputs.file(cache)
    inputs.file(record)
    with h5py.File(cache, 'r') as h:
        center.update({k: h[k][:] for k in h})
    x, y = Transformer.from_crs(4326, grid['epsg'], always_xy=True).transform(center['longitude_deg'], center['latitude_deg'])
    mapper = make_mapper(x, y, center['geometry_valid'].astype(bool), grid['transform'][1], 'nearest', target_grid=grid)
    # Store an integer association, including the many cells copied from one observation.
    association = np.full(mapper.shape, -1, dtype='int64')
    association.ravel()[mapper.positions] = mapper.source_indexes
    return center, association, orbit_path

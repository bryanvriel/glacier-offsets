"""Prepared-pair adapters: no SAFE discovery, SLC copies or coregistration."""
from copy import deepcopy
from pathlib import Path

import numpy as np
from osgeo import gdal

from .config import gross_values
from .matching_grid import MatchingGrid, build_grid, valid_windows
from .products import identity, write_json
from .stack_storage import read_manifest


def matching_template(cfg, frame):
    shape = tuple(frame['coreg']['reference_grid'][k] for k in ('length', 'width'))
    p = cfg['matching']
    return build_grid(shape, shape, p['window'], p['search'], p['stride'], first_center=p.get('first_center'))


def common_map_grid(cfg, frame):
    from .geocode import center_geometry, map_grid, validate_geocoding
    from pyproj import Transformer
    grid = matching_template(cfg, frame)
    epsg, spacing = validate_geocoding(cfg['geocoding'])
    lon, lat, _, _, _ = center_geometry(frame['coreg'], *grid.centers)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    result = map_grid(x, y, spacing)
    result['epsg'] = epsg
    return result


def prepared_pair(cfg, plan, frame, earlier, later, dates, target_grid, velocity_provenance=None):
    root = Path(cfg['output']['directory']) / 'pairs' / f'{earlier.date}_{later.date}'
    products = [dates[a.date] for a in (earlier, later)]
    sources = deepcopy(frame['sources'])
    for entry in sources:
        key = entry['burst_id']
        times = [a.bursts[key].start if key in a.bursts else None for a in (earlier, later)]
        interval = (times[1]-times[0]).total_seconds() if all(t is not None for t in times) else None
        if interval is not None and interval <= 0:
            raise ValueError(f'{key}: nonpositive burst interval')
        entry.update(acquisition_times=[t.isoformat() if t else None for t in times], interval_seconds=interval)
    pair_metadata = {'geometry_reference_date': plan.reference.date,
                     'geometry_identity': frame['geometry_identity'],
                     'geometry_manifest': str(Path(cfg['output']['directory'])/'geometry/manifest.json'),
                     'geometry_reference_inputs': frame['specification']['inputs'],
                     'geometry_reference_burst_times': {e['burst_id']: e['geometry_reference_time'] for e in sources},
                     'acquisition_dates': [earlier.date, later.date],
                     'axis_sign': 'later-minus-earlier on shared stack-reference grid'}
    spec = {'method': 'prepared-stack-pair-v1', 'geometry': frame['geometry_identity'],
            'dates': [p['assembly_identity'] for p in products], 'pair': pair_metadata,
            'burst_intervals': [{key: e[key] for key in ('burst_id', 'geometry_reference_time', 'acquisition_times', 'interval_seconds')}
                                for e in sources]}
    fingerprint = identity(spec)
    manifest = read_manifest(root)
    if manifest.get('coreg_identity') != fingerprint:
        coreg = deepcopy(frame['coreg'])
        coreg['assembly_sources'] = sources
        coreg['slcs'] = dict(zip(('reference', 'secondary'), (p['slc'] for p in products)))
        manifest = {'coreg_identity': fingerprint, 'coreg': coreg, 'coreg_specification': {
                    'inputs': {a.date: p['specification'] for a, p in zip((earlier, later), products)},
                    'polarization': cfg['inputs']['polarization'], 'burst_id': 'stack', 'stack': spec},
                    'stack_pair': pair_metadata, 'software': frame['software'],
                    'interval_seconds': None,
                    'acquisition_times': [a.scene.start.isoformat() for a in (earlier, later)],
                    'assembly': {'identity': fingerprint, 'method': 'fixed stack-reference ownership',
                                 'ownership': frame.get('ownership'),
                                 'sources': sources, 'source_burst_ids': [e['burst_id'] for e in sources],
                                 'interval_policy': 'exact earlier/later sensing times for each geographic burst; missing times invalid'},
                    'stages': {'coregister': 'reused aligned acquisitions; no pair coregistration'}}
    manifest['map_grid'] = target_grid
    if velocity_provenance is not None:
        manifest['velocity_provenance'] = velocity_provenance
    else:
        manifest.pop('velocity_provenance', None)
    write_json(root / 'manifest.json', manifest)
    return root, manifest


def prepare_prior(cfg, manifest, grid, reference_bursts):
    """Project pair durations through the geometry reference's orbit, never a matched date's."""
    from .assembly import source_indices
    from .velocity_prior import prepare_prior as burst_prior
    owner = source_indices(manifest['coreg'], *grid.centers)
    float_names = ('vx', 'vy', 'east_displacement_m', 'north_displacement_m',
                   'longitude_deg', 'latitude_deg', 'height_m')
    result = {name: np.full(grid.shape, np.nan) for name in float_names}
    result.update(predicted_px=np.full((*grid.shape, 2), np.nan),
                  integer_px=np.full((*grid.shape, 2), np.iinfo(np.int32).min, 'int32'),
                  valid=np.zeros(grid.shape, bool), rejection_flags=np.ones(grid.shape, 'uint8'))
    metadata = []
    for index, entry in enumerate(manifest['assembly']['sources'], 1):
        selected = owner == index
        if not selected.any() or entry['interval_seconds'] is None:
            continue
        dy, dx = entry['offset']
        local = MatchingGrid(grid.starts_line-dy, grid.starts_sample-dx, grid.window, grid.search, grid.stride)
        source = {'coreg': entry['coreg'], 'interval_seconds': entry['interval_seconds']}
        child = deepcopy(cfg)
        child['inputs']['burst_id'] = entry['burst_id']
        reference = reference_bursts[entry['burst_id']]
        prior = burst_prior(child, source, local, native_pair=(reference, reference))
        for name in result:
            result[name][selected] = prior[name][selected]
        metadata.append({'burst_id': entry['burst_id'], 'geometry_reference_time': entry['geometry_reference_time'],
                         'acquisition_times': entry['acquisition_times'], 'metadata': prior['metadata']})
    result['metadata'] = {'settings': cfg['gross_offsets']['velocity'], 'sources': metadata,
                          'geometry_reference_date': manifest['stack_pair']['geometry_reference_date'],
                          'interval_policy': 'matched burst times; projection uses pinned stack-reference orbit/grid',
                          'missing_policy': 'mask; no zero fallback'}
    return result


def match_prepared(cfg, manifest, directory, prior):
    """Match valid prepared windows, producing an empty field without native calls when needed."""
    from .ampcor import native_match, varying_match
    slcs = manifest['coreg']['slcs']
    ref, sec = (gdal.Open(slcs[side]) for side in ('reference', 'secondary'))
    if prior is None:
        grid = matching_template(cfg, {'coreg': manifest['coreg']})
        gross = np.broadcast_to(np.array(gross_values(cfg), 'int32'), (*grid.shape, 2)).copy()
        prior = {'grid': grid, 'integer_px': gross, 'valid': np.ones(grid.shape, bool),
                 'rejection_flags': np.zeros(grid.shape, 'uint8')}
        velocity = False
    else:
        grid = prior['grid']
        velocity = True
    prior['valid'] &= valid_windows(ref, sec, grid, prior['integer_px'])
    if not velocity and prior['valid'].any():
        gross = gross_values(cfg)
        bounds = (grid.starts_line[0]+gross[0]-grid.search[0] >= 0 and
                  grid.starts_line[-1]+gross[0]+grid.window[0]+grid.search[0] <= sec.RasterYSize and
                  grid.starts_sample[0]+gross[1]-grid.search[1] >= 0 and
                  grid.starts_sample[-1]+gross[1]+grid.window[1]+grid.search[1] <= sec.RasterXSize)
        if bounds:
            return native_match(slcs['reference'], slcs['secondary'], directory/'native', cfg['matching'],
                                cfg, cfg['matching']['backend'], _grid=grid, _valid_input=prior['valid'])
    result = varying_match(slcs['reference'], slcs['secondary'], directory/'native', cfg['matching'],
                           cfg, cfg['matching']['backend'], prior, ref, sec)
    if not velocity:
        result.pop('prior', None)
    return result

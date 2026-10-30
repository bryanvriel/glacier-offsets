"""Validated virtual collection of radar displacement packages.

Every observation pair is mapped to its source file and slot in a manifest;
inputs are read lazily and never merged into a new netCDF or VRT. Packages must
share CRS, grid, pinned reference/layout identities, source associations,
ownership, reference positions and geographic coordinates. Projection vectors
may differ only within the documented 1e-5 numerical-export tolerance, and each
package keeps its own vectors. Pairs are sorted by acquisition time; pair IDs
must be unique. No reprojection, bridging pair or scene-time substitution.
"""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import h5netcdf
import h5py
import numpy as np

from .velocity_basis import EPOCH_SECONDS
from .velocity_io import digest, revision, verify_hashes, write_json


MANIFEST_SCHEMA = 'glacier-offsets-velocity-collection-v1'
PACKAGE_SCHEMA = 'glacier-offsets-displacements-v1'
PROJECTION_TOLERANCE = 1e-5
TIME_UNITS = 'seconds since 1970-01-01 00:00:00'
CALENDARS = ('proleptic_gregorian', 'standard', 'gregorian')
SPACECRAFT = ('S1A', 'S1B')
REQUIRED = ('x', 'y', 'crs', 'pair_id', 'earlier_time', 'later_time', 'earlier_burst_time',
            'later_burst_time', 'earlier_spacecraft', 'later_spacecraft', 'burst_id', 'enu',
            'range_displacement_m', 'azimuth_displacement_m', 'snr', 'correlation_peak',
            'source_observation_index', 'source_burst_index', 'observation_latitude',
            'observation_longitude', 'reference_center_line', 'reference_center_sample',
            'range_sensitivity_enu', 'azimuth_sensitivity_enu')
EXACT_STATIC = ('x', 'y', 'burst_id', 'source_burst_index', 'source_observation_index',
                'reference_center_line', 'reference_center_sample',
                'observation_latitude', 'observation_longitude')
MATCHING_ATTRIBUTES = ('EPSG', 'geometry_identity', 'geometry_reference_date', 'layout_identity')
OCEAN_TIDE = ('tidal_correction_provenance_json', 'tidal_correction_available')


def days(seconds):
    """UTC epoch seconds to days since the 2000-01-01 phase origin."""
    return (np.asarray(seconds, float)-EPOCH_SECONDS)/86400.


def calendar_year(day):
    return datetime.fromtimestamp(float(day)*86400.+EPOCH_SECONDS, tz=timezone.utc).year


def year_start(year):
    return (datetime(year, 1, 1, tzinfo=timezone.utc).timestamp()-EPOCH_SECONDS)/86400.


def _text(value):
    value = np.asarray(value).ravel()[0] if np.ndim(value) else np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else str(value)


def _strings(variable):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(variable[...]).ravel()]


def _epsg(attrs):
    value = np.asarray(attrs['EPSG']).ravel()
    if value.size != 1:
        raise ValueError('EPSG attribute must hold one code')
    return int(value[0])


def _equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    return np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc' and b.dtype.kind in 'fc')


def check_content(path, cache_directory, sha256):
    """Verify internal variable checksums once per file content."""
    record = Path(cache_directory)/'verified-packages'/f'{sha256}.json'
    if record.is_file():
        return
    with h5netcdf.File(path, 'r') as h:
        verify_hashes(h)
    write_json(record, {'sha256': sha256, 'content_hashes': 'verified', 'path_at_verification': str(path)})


def _package(path):
    """Read and validate one package's metadata, timing and pair table."""
    with h5py.File(path, 'r') as h:
        missing = [k for k in REQUIRED if k not in h]
        if missing:
            raise ValueError(f'{path}: not a self-contained displacement package; missing {missing}')
        attrs = h.attrs
        if _text(attrs.get('schema_version', '')) != PACKAGE_SCHEMA or _text(attrs.get('package_status', '')) != 'complete':
            raise ValueError(f'{path}: incomplete or unsupported displacement package')
        for name in MATCHING_ATTRIBUTES:
            if name not in attrs:
                raise ValueError(f'{path}: missing {name} attribute')
        if _text(attrs.get('geocoding_resampling', '')) != 'nearest':
            raise ValueError(f'{path}: nearest-source geocoding is required')
        for name in ('range_displacement_m', 'azimuth_displacement_m'):
            if _text(h[name].attrs.get('units', '')) != 'm':
                raise ValueError(f'{path}: {name} must be in metres')
        for name in ('earlier_burst_time', 'later_burst_time', 'earlier_time', 'later_time'):
            if _text(h[name].attrs.get('units', '')) != TIME_UNITS:
                raise ValueError(f'{path}: {name} must be UTC epoch seconds')
            calendar = h[name].attrs.get('calendar')
            if calendar is not None and _text(calendar) not in CALENDARS:
                raise ValueError(f'{path}: unsupported {name} calendar')
        if set(_strings(h['enu'])) != {'east', 'north', 'up'}:
            raise ValueError(f'{path}: unknown ENU convention')
        # Ocean tides must already be corrected, with provenance and availability.
        if any(k not in h for k in OCEAN_TIDE) or _text(attrs.get('tidal_correction_status', 'complete')) != 'complete':
            raise ValueError(f'{path}: ocean-tide correction provenance/availability is required (corrections.ocean_tide: require_existing)')
        provenance = json.loads(_text(h['tidal_correction_provenance_json'][()]))
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError(f'{path}: empty ocean-tide correction provenance')
        applied = [k for k in (*attrs.keys(), *h.keys()) if 'solid_earth' in k.lower() or 'body_tide' in k.lower()]
        if applied:
            raise ValueError(f'{path}: solid-Earth correction already recorded ({applied}); it must be applied exactly once')
        validity = 'horizontal_valid_mask' if 'horizontal_valid_mask' in h else 'valid_mask'
        if validity not in h:
            raise ValueError(f'{path}: a measurement validity mask is required')
        if _text(attrs.get('source', 'raw')) == 'swath_corrected' and 'correction_available' not in h:
            raise ValueError(f'{path}: swath-corrected input lacks correction_available')
        pairs = _strings(h['pair_id'])
        earlier, later = _strings(h['earlier_spacecraft']), _strings(h['later_spacecraft'])
        unknown = sorted(set(earlier+later)-set(SPACECRAFT))
        if unknown:
            raise ValueError(f'{path}: unsupported spacecraft {unknown}; order offsets are defined for S1A/S1B')
        scene = np.stack([h['earlier_time'][:], h['later_time'][:]], -1)
        bursts = np.stack([h['earlier_burst_time'][:], h['later_burst_time'][:]], -1)
        if bursts.shape[:2] != (len(pairs), len(_strings(h['burst_id']))):
            raise ValueError(f'{path}: burst endpoint table must be (time, burst)')
        if not np.isfinite(scene).all() or np.any(scene[:, 1] <= scene[:, 0]):
            raise ValueError(f'{path}: scene acquisition labels must be finite and increasing')
        return {'attrs': {k: (_epsg(attrs) if k == 'EPSG' else _text(attrs[k])) for k in MATCHING_ATTRIBUTES},
                'source': _text(attrs.get('source', 'raw')), 'package_identity': _text(attrs.get('package_identity', '')),
                'crs_wkt': _text(h['crs'].attrs.get('crs_wkt', '')), 'validity': validity,
                'pairs': pairs, 'earlier_spacecraft': earlier, 'later_spacecraft': later,
                'scene': scene, 'bursts': bursts,
                'correction_history': sorted(k for k in h if k.endswith('_json') and 'provenance' in k or k == 'calibration_json')}


def compare_geometry(paths):
    """Exact source associations and coordinates; bounded projection-vector differences."""
    differences = []
    with h5py.File(paths[0], 'r') as base:
        base_order = _strings(base['enu'])
        for path in paths:
            with h5py.File(path, 'r') as h:
                for name in EXACT_STATIC:
                    left, right = h[name][...], base[name][...]
                    if name == 'burst_id':
                        left, right = _strings(h[name]), _strings(base[name])
                    if not _equal(left, right):
                        raise ValueError(f'{path}: incompatible {name}')
                order = _strings(h['enu'])
                result = {}
                for name in ('range_sensitivity_enu', 'azimuth_sensitivity_enu'):
                    a = h[name][...][[order.index(k) for k in ('east', 'north', 'up')]]
                    b = base[name][...][[base_order.index(k) for k in ('east', 'north', 'up')]]
                    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                        raise ValueError(f'{path}: {name} support differs')
                    finite = np.isfinite(a)
                    result[name] = float(np.max(abs(a[finite]-b[finite]))) if finite.any() else 0.
                if max(result.values()) > PROJECTION_TOLERANCE:
                    raise ValueError(f'{path}: projection vectors differ by {max(result.values()):.3g}, '
                                     f'above the {PROJECTION_TOLERANCE:g} numerical-export tolerance')
                differences.append(result)
    return differences


def collect(paths, cache_directory, verify_content=True):
    """Validate packages and return the deterministic virtual pair manifest."""
    paths = [Path(p).resolve() for p in paths]
    for path in paths:
        if not path.is_file():
            raise ValueError(f'Missing input package: {path}')
    identities = {}
    for path in paths:
        before = revision(path)
        sha = digest(path)
        if verify_content:
            check_content(path, cache_directory, sha)
        if revision(path) != before:
            raise ValueError(f'Input changed while it was checksummed: {path}')
        identities[path] = {'sha256': sha, 'revision': list(before)}
    if len({v['sha256'] for v in identities.values()}) != len(paths):
        raise ValueError('Two inputs have identical content; each package must be listed once')
    packages = {path: _package(path) for path in paths}
    # Canonical file order is independent of the YAML input order.
    paths = sorted(paths, key=lambda p: (float(packages[p]['scene'][:, 0].min()), identities[p]['sha256']))
    base = packages[paths[0]]
    for path in paths[1:]:
        for name in MATCHING_ATTRIBUTES:
            if packages[path]['attrs'][name] != base['attrs'][name]:
                raise ValueError(f'{path}: incompatible {name}')
        if packages[path]['crs_wkt'] != base['crs_wkt']:
            raise ValueError(f'{path}: incompatible CRS definition')
    differences = compare_geometry(paths)
    seen = {}
    rows = []
    for index, path in enumerate(paths):
        package = packages[path]
        for slot, pair in enumerate(package['pairs']):
            if pair in seen:
                raise ValueError(f'Duplicate pair ID {pair} in {seen[pair]} and {path}')
            seen[pair] = path
            early, late = days(package['scene'][slot])
            mid = (early+late)/2
            label = calendar_year(mid)
            earlier, later = package['earlier_spacecraft'][slot], package['later_spacecraft'][slot]
            bursts = package['bursts'][slot]
            finite = np.isfinite(bursts).all(-1) & (bursts[:, 1] > bursts[:, 0])
            rows.append({'pair_id': pair, 'file': index, 'slot': slot,
                         'calendar_early_day': float(early), 'calendar_late_day': float(late),
                         'calendar_midpoint_day': float(mid), 'year': label,
                         'within_year': calendar_year(early) == label == calendar_year(late),
                         'earlier_spacecraft': earlier, 'later_spacecraft': later,
                         'order': int(later == 'S1B')-int(earlier == 'S1B'),
                         'burst_span_days': [float(days(bursts[finite, 0].min())), float(days(bursts[finite, 1].max()))]
                         if finite.any() else None})
    rows.sort(key=lambda r: (r['calendar_early_day'], r['calendar_late_day'], r['pair_id']))
    spans = np.array([r['burst_span_days'] for r in rows if r['burst_span_days'] is not None])
    if not len(spans):
        raise ValueError('No pair has finite, increasing burst endpoints')
    years = sorted({r['year'] for r in rows})
    manifest = {'schema': MANIFEST_SCHEMA,
                'inputs': [{'path': str(p), **identities[p], 'package_identity': packages[p]['package_identity'],
                            'source': packages[p]['source'], 'validity_mask': packages[p]['validity'],
                            'pairs': len(packages[p]['pairs']), 'correction_history': packages[p]['correction_history']}
                           for p in paths],
                'grid': {**base['attrs'], 'crs_wkt': base['crs_wkt']},
                'projection_tolerance': PROJECTION_TOLERANCE, 'geometry_max_abs_differences': differences,
                'pairs': rows, 'years': years,
                'pairs_by_year': {str(y): sum(r['year'] == y for r in rows) for y in years},
                'cross_year_pairs': [r['pair_id'] for r in rows if not r['within_year']],
                'span_days': float(spans[:, 1].max()-spans[:, 0].min()),
                'year_assignment': 'calendar year of the scene-label midpoint; secular motion uses exact burst-endpoint overlaps',
                'pair_order': 'scene earlier time, scene later time, pair ID'}
    logging.info('Collected %d pairs from %d package(s); years %s', len(rows), len(paths), years)
    return manifest


def content_identity(manifest):
    """Path-independent identity of the virtual stack."""
    return {'schema': manifest['schema'], 'inputs': [i['sha256'] for i in manifest['inputs']],
            'grid': manifest['grid'], 'pairs': [[r['pair_id'], r['file'], r['slot']] for r in manifest['pairs']]}


def check_unchanged(manifest):
    """Reject input files whose revision changed since collection."""
    for item in manifest['inputs']:
        if list(revision(item['path'])) != item['revision']:
            if digest(item['path']) != item['sha256']:
                raise ValueError(f'Input package changed: {item["path"]}')

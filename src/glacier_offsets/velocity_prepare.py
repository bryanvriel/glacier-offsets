"""Extract unique radar-source observations from a virtual package collection.

Each unique source observation is stored once, with exact owning-burst
endpoints, range/azimuth displacements (m), quality values and separate
availability flags. Eligibility is retained before any optional mixed-window
exclusion. Pairs are read in bounded chunks; a partial artifact checkpoints the
completed pair count and resumes only for an identical identity.
"""
import logging

import h5py
import numpy as np

from .velocity_cache import Artifact
from .velocity_collection import content_identity, days, check_unchanged
from .velocity_io import digest
from .velocity_pixel_runtime import module_fingerprints


EXTRACTION_MODULES = ('velocity_collection', 'velocity_prepare')
OBSERVATIONS = 'observations.h5'


def pair_validity(package, slot, rows, cols, data, early, late, validity):
    """Version-1 validity rules for any package validity mask.

    Raw mixed-window radar values remain eligible when only their swath
    correction is unavailable; optional removal happens at fitting time.
    """
    mixed = (package['cross_swath_mixed'][slot][rows, cols] == 1 if 'cross_swath_mixed' in package
             else np.zeros(len(rows), bool))
    good = ((package[validity][slot][rows, cols] == 1) & np.isfinite(data).all(-1)
            & np.isfinite(early+late) & (late > early))
    if 'correction_available' in package:
        good &= (package['correction_available'][slot][rows, cols] == 1) | mixed
    good &= package['tidal_correction_available'][slot][rows, cols] == 1
    return good, mixed


def observation_artifact(cache_directory, manifest):
    identity = {'stage': 'observations', 'collection': content_identity(manifest),
                'code_sha256': module_fingerprints(EXTRACTION_MODULES)}
    return Artifact(cache_directory, 'observations', identity)


def _static(base, enu_by_file, handles):
    association = base['source_observation_index'][:]
    supported = association >= 0
    source_ids, first = np.unique(association[supported], return_index=True)
    positions = np.flatnonzero(supported)[first]
    rows, cols = np.unravel_index(positions, association.shape)
    owner = base['source_burst_index'][:][rows, cols].astype(int)-1
    if np.any(owner < 0):
        raise ValueError('A supported source has no owning burst')
    back = np.full(association.shape, -1, np.int32)
    back[supported] = np.searchsorted(source_ids, association[supported])
    geometry = []
    for h, order in zip(handles, enu_by_file):
        indices = [order.index(k) for k in ('east', 'north', 'up')]
        geometry.append(np.stack([h[k][:][indices][:, rows, cols].T
                                  for k in ('range_sensitivity_enu', 'azimuth_sensitivity_enu')], axis=1))
    return source_ids, rows, cols, owner, back, np.array(geometry)


def _create(path, manifest, handles, static, chunk):
    source_ids, rows, cols, owner, back, geometry = static
    base = handles[0]
    ns, nt = len(source_ids), len(manifest['pairs'])
    pairs = manifest['pairs']
    years = manifest['years']
    with h5py.File(path, 'w') as out:
        out.attrs.update(status='writing', completed_pairs=0, schema='glacier-offsets-velocity-observations-v2',
                         EPSG=int(manifest['grid']['EPSG']), crs_wkt=manifest['grid']['crs_wkt'],
                         geometry_policy='each input retains its own ENU projection vectors',
                         validity='valid_including_mixed retains mixed-window eligibility; removal is a fit-time policy')
        out['x'] = base['x'][:]
        out['y'] = base['y'][:]
        out['crs'] = np.int32(0)
        for key, value in dict(base['crs'].attrs).items():
            out['crs'].attrs[key] = value
        out['burst_id'] = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in base['burst_id'][:]],
                                   dtype=h5py.string_dtype())
        for key, value in [('source_id', source_ids), ('row', rows), ('col', cols), ('owner', owner),
                           ('map_source_index', back), ('geometry_by_file', geometry),
                           ('latitude', base['observation_latitude'][:][rows, cols]),
                           ('longitude', base['observation_longitude'][:][rows, cols])]:
            out.create_dataset(key, data=value, compression='gzip', compression_opts=2)
        for key in ('height_m', 'tidal_response_factor', 'azimuth_ground_spacing_m_per_slc_line'):
            value = base[key][:][rows, cols] if key in base else np.full(ns, np.nan)
            out.create_dataset(key, data=np.asarray(value, float))
        out['pair_id'] = np.array([p['pair_id'] for p in pairs], dtype=h5py.string_dtype())
        out['file_index'] = np.array([p['file'] for p in pairs], np.int32)
        out['slot'] = np.array([p['slot'] for p in pairs], np.int32)
        out['order'] = np.array([p['order'] for p in pairs], np.int8)
        out['year'] = np.array([p['year'] for p in pairs], np.int32)
        out['years'] = np.array(years, np.int32)
        out['year_index'] = np.array([years.index(p['year']) for p in pairs], np.int32)
        out['within_year'] = np.array([p['within_year'] for p in pairs], bool)
        out['calendar_early'] = np.array([p['calendar_early_day'] for p in pairs])
        out['calendar_late'] = np.array([p['calendar_late_day'] for p in pairs])
        out['input_sha256'] = np.array([i['sha256'] for i in manifest['inputs']], dtype=h5py.string_dtype())
        block = (min(2048, ns), min(16, nt))
        options = {'compression': 'gzip', 'compression_opts': 2}
        for name, dtype in [('early', 'f8'), ('late', 'f8'), ('snr', 'f4'), ('peak', 'f4')]:
            out.create_dataset(name, (ns, nt), dtype=dtype, chunks=block, fillvalue=np.nan, **options)
        for name in ('valid_including_mixed', 'mixed_window'):
            out.create_dataset(name, (ns, nt), dtype='?', chunks=block, **options)
        out.create_dataset('data', (ns, nt, 2), dtype='f8', chunks=(*block, 2), fillvalue=np.nan, **options)


def _fill(path, manifest, handles, owner, rows, cols, chunk, hook=None):
    pairs = manifest['pairs']
    with h5py.File(path, 'r+') as out:
        start = int(out.attrs['completed_pairs'])
        if start:
            logging.info('Resuming observation extraction after %d/%d pairs', start, len(pairs))
        for begin in range(start, len(pairs), chunk):
            stop = min(begin+chunk, len(pairs))
            columns = {name: [] for name in ('early', 'late', 'data', 'valid_including_mixed', 'mixed_window', 'snr', 'peak')}
            for pair in pairs[begin:stop]:
                h = handles[pair['file']]
                slot = pair['slot']
                early = days(h['earlier_burst_time'][slot])[owner]
                late = days(h['later_burst_time'][slot])[owner]
                data = np.stack([h[k][slot][rows, cols] for k in ('range_displacement_m', 'azimuth_displacement_m')], -1)
                validity = manifest['inputs'][pair['file']]['validity_mask']
                good, mixed = pair_validity(h, slot, rows, cols, data, early, late, validity)
                for name, value in (('early', early), ('late', late), ('data', data), ('valid_including_mixed', good),
                                    ('mixed_window', mixed), ('snr', h['snr'][slot][rows, cols]),
                                    ('peak', h['correlation_peak'][slot][rows, cols])):
                    columns[name].append(value)
            for name, values in columns.items():
                out[name][:, begin:stop] = np.stack(values, axis=1)
            out.attrs['completed_pairs'] = stop
            out.flush()
            logging.info('Extracted %d/%d pairs', stop, len(pairs))
            if hook is not None:
                hook('observations', stop)
        mixed = np.zeros(len(owner), bool)
        for begin in range(0, len(owner), 8192):
            mixed[begin:begin+8192] = out['mixed_window'][begin:begin+8192].any(1)
        if 'known_mixed_source' not in out:
            out['known_mixed_source'] = mixed
        out.attrs['status'] = 'complete'
        return {'unique_sources': len(owner), 'pairs': len(pairs), 'known_mixed_sources': int(mixed.sum()),
                'map_cells': int(np.sum(out['map_source_index'][:] >= 0))}


def prepare_observations(cache_directory, manifest, chunk=8, hook=None):
    """Build or verify the observation artifact; return (artifact, record)."""
    artifact = observation_artifact(cache_directory, manifest)
    record = artifact.complete()
    if record is not None:
        logging.info('Reusing verified observation cache %s', artifact.id)
        return artifact, record
    with artifact.lock():
        record = artifact.complete()
        if record is not None:
            return artifact, record
        partial, resumed = artifact.begin()
        target = partial/OBSERVATIONS
        for item in manifest['inputs']:
            if digest(item['path']) != item['sha256']:
                raise ValueError(f'Input package changed: {item["path"]}')
        handles = [h5py.File(item['path'], 'r') for item in manifest['inputs']]
        try:
            enu = [[v.decode() if isinstance(v, bytes) else str(v) for v in h['enu'][:]] for h in handles]
            static = _static(handles[0], enu, handles)
            if resumed and target.exists():
                try:
                    with h5py.File(target, 'r') as existing:
                        consistent = (existing.attrs.get('status') == 'writing'
                                      and np.array_equal(existing['source_id'][:], static[0]))
                except OSError:
                    consistent = False
                if not consistent:
                    target.unlink()
            if not target.exists():
                _create(target, manifest, handles, static, chunk)
            _, rows, cols, owner, _, _ = static
            summary = _fill(target, manifest, handles, owner, rows, cols, chunk, hook)
            check_unchanged(manifest)
        finally:
            for h in handles:
                h.close()
        record = artifact.publish([OBSERVATIONS], {'summary': summary, 'manifest': manifest,
                                                   'mixed_policy': 'eligibility retained; removal is selected by screening.remove_mixed_windows'})
        logging.info('Published observation cache %s: %d sources, %d pairs, %d known mixed sources',
                     artifact.id, summary['unique_sources'], summary['pairs'], summary['known_mixed_sources'])
        return artifact, record

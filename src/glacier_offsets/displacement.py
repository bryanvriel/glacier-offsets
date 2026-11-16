"""Portable access to native-source and geocoded-cell displacement packages.

This module deliberately has no GDAL or ISCE3 imports. GOFF cells are analysis
locations, not recovered independent radar measurements.
"""
import json

import numpy as np


NISAR_SCHEMA = 'glacier-offsets-displacements-v2'
LEGACY_SCHEMA = 'glacier-offsets-displacements-v1'


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def is_nisar(h):
    return text(h.attrs.get('schema_version', '')) == NISAR_SCHEMA


def sensitivity(h, name, slot, section):
    """Return (ENU, rows, columns) in either package profile."""
    return h[name][(slot, slice(None), *section) if is_nisar(h) else (slice(None), *section)]


def endpoints(h, slot, section):
    if is_nisar(h):
        return tuple(h[k][(slot, *section)] for k in ('earlier_observation_time', 'later_observation_time'))
    owner = h['source_burst_index'][section].astype(int)-1
    supported = owner >= 0
    return tuple(np.where(supported, h[k][slot][np.maximum(owner, 0)], np.nan)
                 for k in ('earlier_burst_time', 'later_burst_time'))


def covariance_valid(vr, va, cross):
    bound = np.sqrt(np.maximum(vr, 0).astype(float)*np.maximum(va, 0))
    return (np.isfinite(vr) & np.isfinite(va) & np.isfinite(cross) & (vr >= 0) & (va >= 0)
            & (abs(cross) <= bound*(1+1e-6)+1e-12))


def valid_nisar_mask(mask):
    low = mask & 255
    return ((low != 255) & ((low//10) % 10 > 0) & (low % 10 > 0)
            & ((mask & 0x00ffff00) == 0))


def validate_nisar(path):
    """Validate the standalone v2 profile, including all content hashes."""
    import h5netcdf
    from .velocity_io import verify_hashes, fingerprint
    with h5netcdf.File(path, 'r') as h:
        if not is_nisar(h) or h.attrs.get('mission') != 'NISAR':
            raise ValueError('Not a NISAR displacement package')
        verify_hashes(h)
        spec = json.loads(text(h['provenance_json'][()]))
        if h.attrs['package_identity'] != fingerprint(spec):
            raise ValueError('Package provenance fingerprint mismatch')
        shape = tuple(len(h.dimensions[k]) for k in ('time', 'y', 'x'))
        if not all(shape) or h.attrs.get('observation_support') != 'geocoded_cell':
            raise ValueError('Invalid GOFF observation support')
        ids = h['pair_id'][:].astype(str)
        if len(set(ids)) != len(ids) or not np.array_equal(ids, h['names'][:].astype(str)):
            raise ValueError('Pair IDs must be unique and agree with names')
        if (np.any(np.diff(h['time'][:]) < 0) or np.any(h['later_time'][:] <= h['earlier_time'][:])
                or not np.allclose(h['time'][:], (h['earlier_time'][:]+h['later_time'][:])/2, rtol=0, atol=5e-7)):
            raise ValueError('Invalid scene timing')
        grid = spec['grid']; gt = grid['transform']
        if (grid['shape'] != list(shape[1:]) or int(h.attrs['EPSG']) != 3031
                or gt[1:3] != [80., 0.] or gt[4:6] != [0., -80.]):
            raise ValueError('Invalid package grid')
        for k, expected in [('x', gt[0]+(np.arange(shape[2])+.5)*gt[1]),
                            ('y', gt[3]+(np.arange(shape[1])+.5)*gt[5])]:
            if not np.array_equal(h[k][:], expected):
                raise ValueError('Invalid pixel-center coordinates')
        for k in ('range_displacement_m', 'azimuth_displacement_m', 'snr', 'correlation_peak', 'valid_mask',
                  'earlier_observation_time', 'later_observation_time', 'geometry_valid_mask'):
            if h[k].dimensions != ('time', 'y', 'x'):
                raise ValueError(f'Invalid dimensions: {k}')
        if h['valid_mask'].dtype != np.dtype('u1') or '_FillValue' in h['valid_mask'].attrs:
            raise ValueError('Invalid integer measurement mask')
        for k in ('range_displacement_m', 'azimuth_displacement_m'):
            if h[k].dtype != np.dtype('f4') or h[k].attrs.get('units') != 'm':
                raise ValueError('Radar displacements must be float32 metres')
        for k in ('reference_los_enu', 'range_sensitivity_enu', 'azimuth_sensitivity_enu'):
            if h[k].dimensions != ('time', 'enu', 'y', 'x'):
                raise ValueError(f'Invalid geometry dimensions: {k}')
        for k in ('earlier_observation_time', 'later_observation_time'):
            if h[k].attrs.get('units') != 'seconds since 1970-01-01 00:00:00' or h[k].attrs.get('calendar') != 'proleptic_gregorian':
                raise ValueError('Observation endpoints must use UTC epoch seconds')
        for i in range(shape[0]):
            for row in range(0, shape[1], 256):
                for col in range(0, shape[2], 256):
                    section = (slice(row, row+256), slice(col, col+256)); slot = (i, *section)
                    valid = h['valid_mask'][slot]
                    if not np.isin(valid, [0, 1]).all():
                        raise ValueError('Invalid measurement mask')
                    if 'input_mask' in h and np.any((valid == 1) & ~valid_nisar_mask(h['input_mask'][slot])):
                        raise ValueError('Measurement mask accepts invalid source samples')
                    for k in ('range_displacement_m', 'azimuth_displacement_m', 'snr', 'correlation_peak'):
                        a = h[k][slot]
                        if not np.isnan(a[valid == 0]).all() or (k.endswith('_m') and not np.isfinite(a[valid == 1]).all()):
                            raise ValueError(f'Inconsistent valid support: {k}')
                    early, late = endpoints(h, i, section)
                    if np.any((valid == 1) & (~np.isfinite(early+late) | (late <= early))):
                        raise ValueError('Invalid observation endpoints')
                    for k in ('reference_los_enu', 'range_sensitivity_enu', 'azimuth_sensitivity_enu'):
                        if not np.isfinite(sensitivity(h, k, i, section)[:, valid == 1]).all():
                            raise ValueError(f'Incomplete reference geometry: {k}')
                    if 'uncertainty_available' in h:
                        unc = h['uncertainty_available'][slot]
                        sr, sa, cross = [h[k][slot] for k in ('range_displacement_sigma_m', 'azimuth_displacement_sigma_m', 'range_azimuth_displacement_covariance_m2')]
                        if not np.isin(unc, [0, 1]).all() or np.any((unc == 1) & ((valid == 0) | ~covariance_valid(sr.astype(float)**2, sa.astype(float)**2, cross))):
                            raise ValueError('Invalid formal covariance availability')
        return {'path': str(path), 'pairs': shape[0], 'shape': list(shape),
                'status': h.attrs['package_status'], 'omitted': []}

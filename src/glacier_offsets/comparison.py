"""Auditable batch/standalone product comparisons with strict motion checks.

Only named quality/formal-uncertainty fields allow numerical tolerances. Radar
coordinates, masks, gross/residual/total motion, physical displacement/velocity,
and reference geometry always require exact agreement, including missing data.
"""
import json
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal

ANCILLARY_RADAR = {'snr', 'correlation_peak', 'covariance'}
ANCILLARY_MAP = {'snr.tif', 'correlation_peak.tif',
                 'azimuth_displacement_sigma_m.tif', 'range_displacement_sigma_m.tif',
                 'azimuth_velocity_sigma_m_per_day.tif', 'range_velocity_sigma_m_per_day.tif'}
ANCILLARY_RTOL = 1e-5
ANCILLARY_ATOL = 1e-6
RADAR_REQUIRED = {'center_line', 'center_sample', 'valid', 'input_valid', 'finite',
                  'search_boundary', 'rejection_flags', *ANCILLARY_RADAR,
                  *(f'{field}/{axis}_px' for field in ('gross', 'residual', 'total') for axis in ('azimuth', 'range'))}
GEOMETRY_REQUIRED = {'longitude_deg', 'latitude_deg', 'height_m', 'geometry_valid',
                     'azimuth_ground_spacing_m_per_slc_line', 'center_line', 'center_sample'}


def compare_array(left, right, *, ancillary=False, strict=False):
    """Compare arrays and retain difference metrics even when validation fails.

    The ancillary bound is |a-b| <= atol + rtol * max(|a|, |b|). Shape, dtype,
    NaN locations, infinity signs, and every non-ancillary field remain exact.
    Absolute tolerances use each field's stored units and are recorded explicitly.
    """
    a, b = np.asarray(left), np.asarray(right)
    tolerant = ancillary and not strict
    result = {'policy': 'ancillary_tolerance' if tolerant else 'exact',
              'rtol': ANCILLARY_RTOL if tolerant else 0.,
              'atol': ANCILLARY_ATOL if tolerant else 0.,
              'shape': list(a.shape), 'dtype': str(a.dtype), 'status': 'failed'}
    if a.shape != b.shape or a.dtype != b.dtype:
        return result | {'error': f'Shape/dtype mismatch: {a.shape}/{a.dtype} versus {b.shape}/{b.dtype}'}
    floating = a.dtype.kind in 'fc'
    same = (a == b) | (np.isnan(a) & np.isnan(b)) if floating else a == b
    count = int(np.count_nonzero(~same))
    result['different_values'] = count
    if not count:
        return result | {'status': 'exact', 'max_abs_difference': 0., 'outside_tolerance': 0}
    if a.dtype.kind not in 'fciub':
        return result | {'error': 'Non-numeric values differ'}
    finite = np.isfinite(a) & np.isfinite(b)
    nonfinite_difference = int(np.count_nonzero(~same & ~finite))
    # Promote before subtraction, avoiding integer overflow and float32 rounding.
    dtype = np.complex128 if a.dtype.kind == 'c' else np.float64
    aa, bb = a[finite].astype(dtype), b[finite].astype(dtype)
    delta = np.abs(aa - bb)
    scale = np.maximum(np.abs(aa), np.abs(bb))
    limit = result['atol'] + result['rtol'] * scale
    outside = int(np.count_nonzero(delta > limit)) + nonfinite_difference
    result.update(max_abs_difference=float(delta.max()) if delta.size else None,
                  max_relative_difference=float(np.max(delta / np.maximum(scale, np.finfo(float).tiny))) if delta.size else None,
                  nonfinite_differences=nonfinite_difference, outside_tolerance=outside)
    if tolerant and not outside:
        result['status'] = 'within_tolerance'
    return result


def compare_h5(left, right, *, radar=False, strict=False, required=()):
    results = {}
    with h5py.File(left) as a, h5py.File(right) as b:
        def datasets(h):
            names = set()
            h.visititems(lambda name, obj: names.add(name) if isinstance(obj, h5py.Dataset) else None)
            return names
        aa, bb = datasets(a), datasets(b)
        required = set(required) | (RADAR_REQUIRED if radar else set())
        for name in sorted(aa | bb | required):
            if name not in aa or name not in bb:
                results[name] = {'status': 'failed', 'error': 'Required or differing dataset missing from a product'}
            else:
                results[name] = compare_array(a[name][()], b[name][()],
                    ancillary=radar and name in ANCILLARY_RADAR, strict=strict)
    return results


def compare_burst(left, right, *, strict=False):
    """Read existing products only; no source SAFE, orbit, DEM or imagery mutation."""
    left, right = Path(left), Path(right)
    results = {}
    try:
        a, b = [json.loads((p / 'manifest.json').read_text()) for p in (left, right)]
        invariants = {key: a.get(key) == b.get(key) and a.get(key) is not None
                      for key in ('coreg_identity', 'match_identity', 'geocode_identity',
                                  'interval_seconds', 'acquisition_times')}
        invariants['reference_grid'] = a['coreg']['reference_grid'] == b['coreg']['reference_grid']
        invariants['matching_backend'] = a['matching']['backend'] == b['matching']['backend']
        for relative, radar in [('radar/offsets.h5', True), ('radar/matching_grid.h5', False),
                                ('geo/center_geometry.h5', False)]:
            required = GEOMETRY_REQUIRED if relative.startswith('geo/') else {'center_line', 'center_sample'}
            values = compare_h5(left / relative, right / relative, radar=radar, strict=strict, required=required)
            results.update({f'{relative}:{key}': value for key, value in values.items()})
        geotiffs = [set(p.name for p in (root / 'geo').glob('*.tif')) for root in (left, right)]
        if not any(geotiffs):
            results['geo'] = {'status': 'failed', 'error': 'No geocoded GeoTIFF products'}
        gdal.UseExceptions()
        for name in sorted(geotiffs[0] | geotiffs[1]):
            if not all(name in names for names in geotiffs):
                results['geo/' + name] = {'status': 'failed', 'error': 'GeoTIFF missing from one run'}
                continue
            x, y = [gdal.Open(str(p / 'geo' / name)) for p in (left, right)]
            value = compare_array(x.ReadAsArray(), y.ReadAsArray(), ancillary=name in ANCILLARY_MAP, strict=strict)
            georeferencing = x.GetGeoTransform() == y.GetGeoTransform() and x.GetProjection() == y.GetProjection()
            value['georeferencing_exact'] = georeferencing
            if not georeferencing:
                value.update(status='failed', error='Georeferencing differs')
            results['geo/' + name] = value
        status = ('failed' if not all(invariants.values()) or any(v['status'] == 'failed' for v in results.values())
                  else 'within_tolerance' if any(v['status'] == 'within_tolerance' for v in results.values()) else 'exact')
        return {'result': status, 'manifest_checks': invariants, 'arrays': results}
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {'result': 'failed', 'arrays': results, 'error': f'{type(exc).__name__}: {exc}'}

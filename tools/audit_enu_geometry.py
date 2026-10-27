"""Audit packaged ENU coefficients without changing geometry or displacements.

Run in the ISCE3 environment, using only a displacement netCDF:
  python tools/audit_enu_geometry.py STACK.nc --output .runtime/enu-audit.json

The sweep includes representative observations from every burst, all declared
outliers, and the strongest extremes. A zero-Doppler implicit derivative checks
the finite differences independently. The accompanying NPZ retains every sweep
result. Numerical convergence does not establish the physical accuracy of the
DEM-intersecting azimuth scale; scale outliers remain explicitly unresolved.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5netcdf
import numpy as np
from pyproj import Transformer

from glacier_offsets.products import checksum


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def load_geometry(path):
    """Read each unique source observation once, with its pinned orbit."""
    with h5netcdf.File(path) as h:
        association = h['source_observation_index'][:]
        support = association >= 0
        ids, first = np.unique(association[support], return_index=True)
        if not len(ids):
            raise ValueError('No source observations')
        locations = np.flatnonzero(support)[first]
        lookup = np.searchsorted(ids, association[support])
        fields = {}
        names = ['observation_longitude', 'observation_latitude', 'height_m',
                 'azimuth_ground_spacing_m_per_slc_line', 'source_burst_index',
                 'reference_center_line', 'reference_center_sample',
                 'range_sensitivity_enu', 'azimuth_sensitivity_enu', 'reference_los_enu']
        if [text(v) for v in h['enu'][:]] != ['east', 'north', 'up']:
            raise ValueError('Expected east/north/up coefficient order')
        for name in names:
            value = h[name][:]
            value = np.moveaxis(value, 0, -1) if value.ndim == 3 else value
            selected = value[support][first]
            if not np.array_equal(value[support], selected[lookup], equal_nan=True):
                raise ValueError(f'Inconsistent copies of source observation: {name}')
            if not np.isfinite(selected).all():
                raise ValueError(f'Nonfinite source geometry: {name}')
            fields[name] = selected
        fields['id'] = ids
        fields['map_yx'] = np.stack(np.unravel_index(locations, association.shape), axis=1)
        fields['grids'] = [json.loads(text(v)) for v in h['reference_radar_grids_json'][:]]
        fields['burst_ids'] = [text(v) for v in h['burst_id'][:]]
        fields['burst_time'] = h['reference_burst_time'][:]
        fields['orbit'] = {name: h[f'reference_orbit_{name}'][:] for name in ('time', 'position', 'velocity')}
        fields['package_identity'] = str(h.attrs['package_identity'])
    owner = fields['source_burst_index']
    if np.any(owner < 1) or np.any(owner > len(fields['grids'])):
        raise ValueError('Invalid source-burst ownership')
    if np.any(fields['azimuth_ground_spacing_m_per_slc_line'] <= 0):
        raise ValueError('Nonpositive azimuth scale')
    return fields


def select_observations(fields, per_burst=64):
    """Include every high-scale/high-condition observation and spatial samples."""
    rg, az = (fields[f'{axis}_sensitivity_enu'] for axis in ('range', 'azimuth'))
    condition = np.linalg.cond(np.stack((rg[:, :2], az[:, :2]), axis=1))
    scale = fields['azimuth_ground_spacing_m_per_slc_line']
    aznorm = np.linalg.norm(az[:, :2], axis=1)
    owner = fields['source_burst_index']
    ratio = np.empty(len(scale))
    selected = set()
    for burst in np.unique(owner):
        indexes = np.flatnonzero(owner == burst)
        ratio[indexes] = scale[indexes]/np.median(scale[indexes])
        # Order by map coordinates rather than source IDs for spatial coverage.
        yx = fields['map_yx'][indexes]
        ordered = indexes[np.lexsort((yx[:, 1], yx[:, 0]))]
        selected.update(ordered[np.unique(np.linspace(0, len(ordered)-1, min(per_burst, len(ordered))).astype(int))])
    extreme = (condition > 3) | (ratio > 2) | (aznorm > 2) | (np.abs(az[:, 2]) > .004)
    selected.update(np.flatnonzero(extreme))
    for score in (condition, ratio, np.abs(az[:, 2]), np.linalg.norm(rg[:, :2], axis=1)):
        selected.update(np.argsort(score)[-32:])
        selected.update(np.argsort(score)[:16])
    # Include immediate map neighbors of the worst condition/scale observations.
    for index in np.unique(np.r_[np.argsort(condition)[-16:], np.argsort(ratio)[-16:]]):
        distance = np.sum((fields['map_yx']-fields['map_yx'][index])**2, axis=1)
        selected.update(np.argsort(distance)[:9])
    return np.array(sorted(selected)), condition, extreme


def native_orbit(fields, isce3):
    """Use exactly the UTC crop and construction used by package_geometry."""
    times = fields['orbit']['time']
    starts = fields['burst_time']
    stop = max(t+g['length']/g['prf'] for t, g in zip(starts, fields['grids']))+60
    take = (times >= min(starts)-60) & (times <= stop)
    if take.sum() < 4:
        raise ValueError('Insufficient orbit coverage')
    vectors = [isce3.core.StateVector(
        isce3.core.DateTime(datetime.fromtimestamp(float(t), timezone.utc).replace(tzinfo=None)), p, v)
        for t, p, v in zip(times[take], fields['orbit']['position'][take], fields['orbit']['velocity'][take])]
    return isce3.core.Orbit(vectors)


def enu_basis(lon, lat):
    lam, phi = np.deg2rad(lon), np.deg2rad(lat)
    return np.array([[-np.sin(lam), np.cos(lam), 0],
                     [-np.sin(phi)*np.cos(lam), -np.sin(phi)*np.sin(lam), np.cos(phi)],
                     [np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)]])


def central_difference(solve, point, basis, step, azimuth_scale):
    """Return range and scaled azimuth derivatives, ordered ENU."""
    result = np.empty((2, 3))
    for j, direction in enumerate(basis):
        tp, rp = solve(point+step*direction)
        tm, rm = solve(point-step*direction)
        result[:, j] = [(rp-rm)/(2*step), (tp-tm)/(2*step)*azimuth_scale]
    return result


def implicit_derivative(orbit, t, point, basis, azimuth_scale, acceleration_step=.05):
    """Differentiate (sensor(t)-point).velocity(t)=0 independently of geo2rdr.

    Range's derivative is minus ground-to-sensor LOS. The azimuth derivative
    follows from implicit differentiation of zero Doppler; only the time
    derivative of the orbital dot product is numerically evaluated here.
    """
    position, velocity = orbit.interpolate(t)
    los = np.asarray(position)-point
    def doppler(time):
        p, v = orbit.interpolate(time)
        return np.dot(np.asarray(p)-point, v)
    denominator = (doppler(t+acceleration_step)-doppler(t-acceleration_step))/(2*acceleration_step)
    return np.stack((-basis @ (los/np.linalg.norm(los)), basis @ velocity/denominator*azimuth_scale))


def settings():
    result = [{'step_m': step, 'threshold_s': threshold, 'maxiter': 50, 'delta_range_m': 10.}
              for step in (.01, .03, .1, .3, 1., 3., 10., 30.)
              for threshold in (1e-6, 1e-8, 1e-10, 1e-12)]
    result.extend({'step_m': 1., 'threshold_s': 1e-12, 'maxiter': 100, 'delta_range_m': dr}
                  for dr in (.1, 10., 100.))
    return result


def statistics(values):
    a = np.asarray(values)
    finite = a[np.isfinite(a)]
    quantiles = np.quantile(finite, [0, .5, .95, .99, 1]) if len(finite) else [None]*5
    return {key: float(value) if value is not None else None
            for key, value in zip(('min', 'median', 'p95', 'p99', 'max'), quantiles)} | {'nonfinite_count': int(a.size-len(finite))}


def finite_max(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else None


def audit(path, output, per_burst=64):
    # Pyre must not interpret this tool's command-line options.
    arguments = sys.argv
    try:
        sys.argv = arguments[:1]
        import isce3
    finally:
        sys.argv = arguments
    path, output = Path(path).resolve(), Path(output).resolve()
    if output == path or output.with_suffix('.npz') == path:
        raise ValueError('Audit output must not overwrite the input')
    digest = checksum(path)
    fields = load_geometry(path)
    selected, condition, extreme = select_observations(fields, per_burst)
    orbit = native_orbit(fields, isce3)
    ellipsoid, doppler = isce3.core.Ellipsoid(), isce3.core.LUT2d()
    forward, inverse = (Transformer.from_crs(a, b, always_xy=True) for a, b in ((4979, 4978), (4978, 4979)))
    options = settings()
    values = np.full((len(selected), len(options), 2, 3), np.nan)
    reference = np.full((len(selected), 2, 3), np.nan)
    reference_check = np.full_like(reference, np.nan)
    forward_error = np.full((len(selected), 2), np.nan)
    residual = np.full(len(selected), np.nan)
    failures = []
    for k, i in enumerate(selected):
        if k % 100 == 0:
            print(f'Geometry sweep {k}/{len(selected)}', file=sys.stderr, flush=True)
        lon, lat, z = (fields[key][i] for key in ('observation_longitude', 'observation_latitude', 'height_m'))
        point, basis = np.array(forward.transform(lon, lat, z)), enu_basis(lon, lat)
        grid = fields['grids'][int(fields['source_burst_index'][i])-1]
        side = isce3.core.LookSide.Right if grid['lookside'].lower() == 'right' else isce3.core.LookSide.Left
        scale = grid['prf']*fields['azimuth_ground_spacing_m_per_slc_line'][i]
        def solve(p, **kwargs):
            lo, la, height = inverse.transform(*p)
            return isce3.geometry.geo2rdr([np.deg2rad(lo), np.deg2rad(la), height], ellipsoid, orbit, doppler,
                                         grid['wavelength'], side, **kwargs)
        # 1e-12 s is retained as a stress test, but can be below the native
        # interpolation/root convergence floor. Do not make it the reference.
        tight = lambda p: solve(p, threshold=1e-10, maxiter=100)
        try:
            t, r = tight(point)
            reference[k] = implicit_derivative(orbit, t, point, basis, scale)
            reference_check[k] = implicit_derivative(orbit, t, point, basis, scale, .1)
            position, velocity = orbit.interpolate(t)
            residual[k] = abs(np.dot(np.asarray(position)-point, velocity))/np.linalg.norm(velocity)
            # A finite, one-sided vertical move tests the sign and a realistic
            # correction amplitude rather than just repeating a symmetric test.
            tp, rp = tight(point+3*basis[2])
            forward_error[k] = np.array([rp-r, (tp-t)*scale])-3*reference[k, :, 2]
        except RuntimeError as exc:
            failures.append({'source_id': int(fields['id'][i]), 'setting': 'reference', 'error': str(exc)})
        for j, option in enumerate(options):
            try:
                solver = lambda p: solve(p, threshold=option['threshold_s'], maxiter=option['maxiter'], delta_range=option['delta_range_m'])
                values[k, j] = central_difference(solver, point, basis, option['step_m'], scale)
            except RuntimeError as exc:
                failures.append({'source_id': int(fields['id'][i]), 'setting': option, 'error': str(exc)})
    stored = np.stack((fields['range_sensitivity_enu'][selected], fields['azimuth_sensitivity_enu'][selected]), axis=1)
    errors = np.abs(values-reference[:, None])
    # Preserve the original 0.1--10 m acceptance window even if a narrower
    # practical plateau succeeds. Stress-test convergence is reported separately.
    plateau = np.array([.1 <= s['step_m'] <= 10 and 1e-10 <= s['threshold_s'] <= 1e-8 for s in options])
    budget = 1e-5  # m observable / m ENU; <=0.03 mm bias error for 3 m vertical motion.
    selected_errors = errors[:, plateau]
    numerical_pass = np.isfinite(selected_errors).all() and float(np.max(selected_errors)) <= budget
    nominal = next(j for j, s in enumerate(options) if s['step_m'] == 1 and s['threshold_s'] == 1e-8)
    practical = np.array([.3 <= s['step_m'] <= 10 and 1e-10 <= s['threshold_s'] <= 1e-6 for s in options])
    rows = []
    for j, setting in enumerate(options):
        row = dict(setting)
        row.update(max_abs_error_m_per_m=finite_max(errors[:, j]),
                   max_vertical_error_m_per_m=finite_max(errors[:, j, :, 2]),
                   max_range_error_m_per_m=finite_max(errors[:, j, 0]),
                   max_azimuth_error_m_per_m=finite_max(errors[:, j, 1]),
                   failed_observations=int(np.any(~np.isfinite(values[:, j]), axis=(1, 2)).sum()),
                   in_acceptance_plateau=bool(plateau[j]))
        rows.append(row)
    outliers = []
    for i in np.argsort(condition)[-20:][::-1]:
        k = int(np.searchsorted(selected, i))
        outliers.append({'source_id': int(fields['id'][i]), 'burst_id': fields['burst_ids'][int(fields['source_burst_index'][i])-1],
                         'map_yx': fields['map_yx'][i].tolist(), 'longitude': float(fields['observation_longitude'][i]),
                         'latitude': float(fields['observation_latitude'][i]), 'height_m': float(fields['height_m'][i]),
                         'azimuth_scale_m_per_line': float(fields['azimuth_ground_spacing_m_per_slc_line'][i]),
                         'horizontal_condition': float(condition[i]), 'stored_coefficients_range_azimuth_by_enu': stored[k].tolist(),
                         'max_plateau_error_m_per_m': finite_max(selected_errors[k])})
    report = {'input': str(path), 'input_sha256': digest, 'package_identity': fields['package_identity'],
              'isce3_version': str(isce3.__version__), 'audit_implementation_sha256': checksum(__file__),
              'unique_source_observations': len(fields['id']), 'tested_observations': len(selected),
              'tested_bursts': sorted(set(int(v) for v in fields['source_burst_index'][selected])),
              'outlier_definition': 'condition > 3 OR scale > 2*burst median OR horizontal azimuth sensitivity norm > 2 OR abs(azimuth up sensitivity) > 0.004',
              'all_declared_outliers_tested': bool(np.isin(np.flatnonzero(extreme), selected).all()),
              'declared_outlier_observations': int(extreme.sum()),
              'selection': 'all declared outliers, per-burst spatial samples, extremes, and nearest map neighbors of strongest outliers',
              'all_geometry_condition': statistics(condition),
              'all_geometry_azimuth_scale_m_per_line': statistics(fields['azimuth_ground_spacing_m_per_slc_line']),
              'reference': 'implicit derivative of (sensor-position minus ground-position) dot sensor-velocity = 0; range derivative = minus LOS',
              'reference_acceleration_step_check_max_m_per_m': finite_max(np.abs(reference-reference_check)),
              'max_zero_doppler_alongtrack_residual_m': finite_max(residual),
              'stored_coefficient_max_error_m_per_m': finite_max(np.abs(stored-reference)),
              'stored_vs_recomputed_default_max_error_m_per_m': finite_max(np.abs(stored-values[:, nominal])),
              'three_metre_vertical_forward_error_m': {'range': statistics(abs(forward_error[:, 0])), 'azimuth': statistics(abs(forward_error[:, 1]))},
              'acceptance': {'coefficient_absolute_budget_m_per_m': budget, 'step_range_m': [.1, 10.],
                             'threshold_s_range': [1e-10, 1e-8], 'max_error_m_per_m': finite_max(selected_errors),
                             'numerical_convergence_passed': bool(numerical_pass),
                             'production_setting_max_error_m_per_m': finite_max(errors[:, nominal]),
                             'practical_plateau': {'step_range_m': [.3, 10.], 'threshold_s_range': [1e-10, 1e-6],
                                                   'max_error_m_per_m': finite_max(errors[:, practical]),
                                                   'passed': bool(np.isfinite(errors[:, practical]).all() and np.max(errors[:, practical]) <= budget)},
                             'physical_scale_accuracy': 'not established; DEM/geometry scale outliers require separate investigation',
                             'azimuth_tidal_correction_enabled': False},
              'settings': rows, 'strongest_condition_outliers': outliers, 'solver_failures': failures}
    if checksum(path) != digest:
        raise ValueError('Input changed during the audit')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'), source_id=fields['id'][selected], values=values,
                        reference=reference, stored=stored, settings_json=json.dumps(options),
                        forward_error=forward_error, condition=condition[selected], selected_index=selected)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stack')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--per-burst', type=int, default=64)
    args = parser.parse_args()
    if args.per_burst < 1:
        parser.error('--per-burst must be positive')
    result = audit(args.stack, args.output, args.per_burst)
    print(json.dumps({'report': str(args.output), 'tested': result['tested_observations'], 'acceptance': result['acceptance']}, indent=2))
    return 0 if result['acceptance']['numerical_convergence_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

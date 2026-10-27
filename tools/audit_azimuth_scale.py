"""Check retained azimuth scales against fresh pinned-DEM rdr2geo intersections.

Unlike audit_enu_geometry, this optional investigation needs the retained
reference manifest and its checksum-pinned DEM. It never changes any products.
Every declared geometry outlier and one typical observation per burst is tested.
"""
import argparse
import json
from pathlib import Path
import sys

import h5netcdf
import numpy as np
from pyproj import Geod

from glacier_offsets.products import checksum
from tools.audit_enu_geometry import load_geometry, native_orbit, select_observations


def audit(package, manifest, output):
    arguments = sys.argv
    try:
        sys.argv = arguments[:1]
        import isce3
    finally:
        sys.argv = arguments
    package, manifest, output = (Path(p).resolve() for p in (package, manifest, output))
    package_digest, manifest_digest = checksum(package), checksum(manifest)
    fields = load_geometry(package)
    saved = json.loads(manifest.read_text())
    with h5netcdf.File(package) as h:
        for key in ('geometry_identity', 'layout_identity'):
            if saved[key] != h.attrs[key]:
                raise ValueError(f'Reference manifest differs from package {key}')
    dem_record = saved['specification']['inputs']['dem']
    dem_path = Path(dem_record['path']).resolve()
    if output in (package, manifest, dem_path):
        raise ValueError('Audit output must not overwrite inputs')
    if checksum(dem_path) != dem_record['sha256']:
        raise ValueError('DEM differs from the pinned reference')
    for grid, burst, source in zip(fields['grids'], fields['burst_ids'], saved['sources'], strict=True):
        if source['burst_id'] != burst or source['coreg']['reference_grid'] != grid:
            raise ValueError('Reference burst order or radar grid differs from package')
    orbit = native_orbit(fields, isce3)
    dem = isce3.geometry.DEMInterpolator(isce3.io.Raster(str(dem_path)))
    _, _, extreme = select_observations(fields)
    selected = set(np.flatnonzero(extreme))
    owner, scale = fields['source_burst_index'], fields['azimuth_ground_spacing_m_per_slc_line']
    for burst in np.unique(owner):
        indexes = np.flatnonzero(owner == burst)
        selected.add(indexes[np.argsort(scale[indexes])[len(indexes)//2]])
    selected = sorted(selected, key=lambda i: -scale[i])
    geod, points = Geod(ellps='WGS84'), []
    for k, i in enumerate(selected):
        if k % 25 == 0:
            print(f'Native azimuth-scale audit {k}/{len(selected)}', file=sys.stderr, flush=True)
        source = saved['sources'][int(owner[i])-1]
        grid = fields['grids'][int(owner[i])-1]
        dy, dx = source['offset']
        line = fields['reference_center_line'][i]-dy
        sample = fields['reference_center_sample'][i]-dx
        time = ((isce3.core.DateTime(grid['ref_epoch'])-orbit.reference_epoch).total_seconds()
                +grid['sensing_start']+line/grid['prf'])
        distance = grid['starting_range']+sample*grid['range_pixel_spacing']
        side = isce3.core.LookSide.Right if grid['lookside'].lower() == 'right' else isce3.core.LookSide.Left
        results = []
        for halfspan in (.1, .25, .5, 1., 2.):
            for threshold in (1e-2, 1e-4, 1e-6):
                record = {'halfspan_lines': halfspan, 'rdr2geo_threshold_m': threshold}
                try:
                    a, b = [isce3.geometry.rdr2geo(time+sign*halfspan/grid['prf'], distance, orbit, side,
                            dem=dem, threshold=threshold, maxiter=100, extraiter=10) for sign in (-1, 1)]
                    _, _, length = geod.inv(*np.rad2deg(a[:2]), *np.rad2deg(b[:2]))
                    if not np.isfinite([*a, *b, length]).all():
                        raise RuntimeError('Nonfinite DEM intersection')
                    record.update(native_scale_m_per_line=float(length/(2*halfspan)),
                                  height_minus_m=float(a[2]), height_plus_m=float(b[2]))
                except RuntimeError as exc:
                    record['error'] = str(exc)
                results.append(record)
        nominal = next(r for r in results if r['halfspan_lines'] == .5 and r['rdr2geo_threshold_m'] == 1e-4)
        fresh = nominal.get('native_scale_m_per_line')
        agreement = fresh is not None and abs(scale[i]-fresh) <= .02+.01*abs(fresh)
        points.append({'source_id': int(fields['id'][i]), 'declared_outlier': bool(extreme[i]),
                       'burst_id': source['burst_id'], 'map_yx': fields['map_yx'][i].tolist(),
                       'stored_scale_m_per_line': float(scale[i]), 'nominal_native_scale_m_per_line': fresh,
                       'nominal_agreement': bool(agreement), 'local_radar_line': float(line),
                       'local_radar_sample': float(sample), 'native': results})
    if checksum(package) != package_digest or checksum(manifest) != manifest_digest or checksum(dem_path) != dem_record['sha256']:
        raise ValueError('Input changed during the audit')
    report = {'package': str(package), 'package_sha256': package_digest,
              'reference_manifest': str(manifest), 'reference_manifest_sha256': manifest_digest, 'dem': dem_record,
              'isce3_version': str(isce3.__version__), 'audit_implementation_sha256': checksum(__file__),
              'tested_observations': len(points), 'all_declared_outliers_tested': True,
              'selection': 'every declared outlier from audit_enu_geometry plus the median-scale observation in every burst',
              'agreement_tolerance': 'absolute difference <= 0.02 m/line + 1% of native scale',
              'nominal_scale_agreement_count': sum(p['nominal_agreement'] for p in points),
              'native_solve_failures': sum('error' in r for p in points for r in p['native']),
              'all_nominal_scales_agree': all(p['nominal_agreement'] for p in points),
              'azimuth_tidal_correction_enabled': False, 'points': points}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package')
    parser.add_argument('reference_manifest')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = audit(args.package, args.reference_manifest, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'points'}, indent=2))
    return 0 if report['all_nominal_scales_agree'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

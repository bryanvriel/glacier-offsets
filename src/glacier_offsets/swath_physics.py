"""Auditable sparse timing candidates on original Sentinel-1 burst coordinates.

Tagged-time corrections are kept separate from apparent image displacement.
The nearest-to-original-mid-burst annotation polynomials are held fixed;
neither the midpoint nor the fast-time origin depends on the sampled patch.
"""
import importlib
import inspect
from pathlib import Path
import logging

import numpy as np

from .geocode import sample_geometry
from .products import identity, write_json

C = 299792458.0
COMPONENTS = ('bistatic', 'fm_rate', 'combined')


def orbit_seconds(burst, lines):
    grid = burst.as_isce3_radargrid()
    epoch = (grid.ref_epoch-burst.orbit.reference_epoch).total_seconds()
    return epoch+grid.sensing_start+np.asarray(lines)*burst.azimuth_time_interval


def bistatic_seconds(burst, samples):
    rho = burst.starting_range+np.asarray(samples)*burst.range_pixel_spacing
    return (burst.iw2_mid_range+rho)/C-burst.rank/burst.prf_raw_data


def fm_rate_seconds(burst, lines, samples, llh):
    """ETAD-form FM mismatch using fixed original-mid-burst annotation selection.

    llh last axis is longitude degrees, latitude degrees, ellipsoidal metres.
    The independent source audit verifies the reader selected nearest-mid-burst
    polynomials; the installed LUT implementation is a separate comparison.
    """
    import isce3
    lines, samples = np.broadcast_arrays(lines, samples)
    shape = lines.shape
    t = orbit_seconds(burst, lines).ravel()
    rho = (burst.starting_range+samples*burst.range_pixel_spacing).ravel()
    points = np.asarray(llh).reshape(-1, 3)
    positions, velocities, accelerations = [], [], []
    dt = burst.azimuth_time_interval
    for time in t:
        position, velocity = burst.orbit.interpolate(float(time))
        _, before = burst.orbit.interpolate(float(time-dt/2))
        _, after = burst.orbit.interpolate(float(time+dt/2))
        positions.append(position)
        velocities.append(velocity)
        accelerations.append((np.asarray(after)-before)/dt)
    positions, velocities, accelerations = map(np.asarray, (positions, velocities, accelerations))
    ellipsoid = isce3.core.Ellipsoid()
    ground = np.array([ellipsoid.lon_lat_to_xyz([np.deg2rad(lon), np.deg2rad(lat), h])
                       for lon, lat, h in points])
    separation = positions-ground
    true_ka = -2*burst.radar_center_frequency/(C*np.linalg.norm(separation, axis=-1))* (
        np.sum(separation*accelerations, axis=-1)+np.sum(velocities**2, axis=-1))
    ka = np.asarray(burst.azimuth_fm_rate.eval(rho.tolist()))
    fdc = np.asarray(burst.doppler.poly1d.eval(rho.tolist()))
    mid_rho = burst.starting_range+(burst.width-1)*burst.range_pixel_spacing/2
    ka_mid = burst.azimuth_fm_rate.eval(float(mid_rho))
    fdc_mid = burst.doppler.poly1d.eval(float(mid_rho))
    mid_t = float(orbit_seconds(burst, (burst.length-1)/2))
    steering = 2*np.linalg.norm(velocities, axis=-1)/C*burst.radar_center_frequency*burst.azimuth_steer_rate
    focused = ka*steering/(ka-steering)
    doppler = fdc+focused*((t-mid_t)+(fdc_mid/ka_mid-fdc/ka))
    result = doppler*(-1/ka+1/true_ka)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite FM-rate prediction')
    return result.reshape(shape)


def mapping(record, key, lines, samples, reference=False):
    """Reference local -> secondary local, and its 2x2 coordinate Jacobian."""
    lines, samples = np.broadcast_arrays(lines, samples)
    shape = lines.shape
    jacobian = np.broadcast_to(np.eye(2), shape+(2, 2)).copy()
    if reference:
        return lines.copy(), samples.copy(), jacobian
    paths = record.get('valid_windows', {}).get(key, {}).get('offset_paths')
    if not paths or len(paths) != 2:
        raise ValueError(f'{key}: registered azimuth/range offsets unavailable')
    offsets = [sample_geometry(path, lines, samples) for path in paths]
    for i, path in enumerate(paths):
        jacobian[..., i, 0] += (sample_geometry(path, lines+.5, samples)-
                               sample_geometry(path, lines-.5, samples))
        jacobian[..., i, 1] += (sample_geometry(path, lines, samples+.5)-
                               sample_geometry(path, lines, samples-.5))
    values = [lines+offsets[0], samples+offsets[1]]
    if any(not np.isfinite(v).all() or np.any(abs(v) > 1.e7) for v in (*offsets, jacobian)):
        raise ValueError('Nonfinite or sentinel coordinate mapping')
    if np.any(abs(np.linalg.det(jacobian)) < .5):
        raise ValueError('Singular or unexpected resampling Jacobian')
    return *values, jacobian


def apparent_bias(seconds, jacobian, acquisition_dt):
    """A positive tagged-time correction means a negative native image bias.

    Convert native-line displacement through the inverse resampling Jacobian;
    do not assume that acquisition and reference pixel coordinates coincide.
    """
    displacement = np.stack([-np.asarray(seconds)/acquisition_dt, np.zeros_like(seconds)], axis=-1)
    return np.linalg.solve(jacobian, displacement[..., None])[..., 0]


def native_geometry_check(burst, lines, samples, llh, limit=15):
    import isce3
    indices = np.unique(np.linspace(0, len(np.ravel(lines))-1, min(limit, np.size(lines))).astype(int))
    errors = []
    grid = burst.as_isce3_radargrid()
    shift = (grid.ref_epoch-burst.orbit.reference_epoch).total_seconds()
    for i in indices:
        lon, lat, height = np.asarray(llh).reshape(-1, 3)[i]
        t, rho = isce3.geometry.geo2rdr([np.deg2rad(lon), np.deg2rad(lat), height],
            isce3.core.Ellipsoid(), burst.orbit, isce3.core.LUT2d(), grid.wavelength, grid.lookside)
        errors.append([(t-shift-grid.sensing_start)*grid.prf-np.ravel(lines)[i],
                       (rho-grid.starting_range)/grid.range_pixel_spacing-np.ravel(samples)[i]])
    errors = np.asarray(errors)
    return {'samples': len(indices), 'max_line_error_px': float(abs(errors[:, 0]).max()),
            'max_range_error_px': float(abs(errors[:, 1]).max()), 'errors_px': errors.tolist(),
            'passed': bool(np.all(abs(errors[:, 0]) < .02) and np.all(abs(errors[:, 1]) < .05))}


def coefficient_check(burst):
    """Independent evaluation of original annotated coefficient records."""
    import isce3
    midpoint = float(orbit_seconds(burst, (burst.length-1)/2))
    ext = burst.extended_coeffs
    results = {}
    rho = burst.starting_range+np.array([0, (burst.width-1)/2, burst.width-1])*burst.range_pixel_spacing
    for label, times, origins, coefficients, poly in (
        ('fm', ext.fm_rate_aztime_vec, ext.fm_rate_tau0_vec, ext.fm_rate_coeff_arr, burst.azimuth_fm_rate),
        ('dc', ext.dc_aztime_vec, ext.dc_tau0_vec, ext.dc_coeff_arr, burst.doppler.poly1d)):
        times = np.array([(isce3.core.DateTime(t)-burst.orbit.reference_epoch).total_seconds() for t in times])
        if not times.size:
            raise ValueError('Missing annotation coefficients')
        index = int(np.argmin(abs(times-midpoint)))
        origin = float(np.ravel(origins)[index])
        coeff = np.asarray(coefficients)[index]
        x = 2*rho/C-origin
        actual = np.polynomial.polynomial.polyval(x, coeff)
        reader = np.asarray(poly.eval(rho.tolist()))
        results[label] = {'selected_index': index, 'time_distance_seconds': float(abs(times[index]-midpoint)),
            'tau0_seconds': origin, 'coefficients': coeff.tolist(), 'max_reader_difference': float(abs(actual-reader).max()),
            'passed': bool(np.allclose(actual, reader, rtol=1.e-10, atol=1.e-7))}
    results['passed'] = all(v['passed'] for v in results.values())
    return results


def installed_grid_check(burst, dem):
    """Full-burst-span coarse grid, NEVER a cropped patch passed to the LUT API.

    The installed method's returned grid origins are checked explicitly. A
    disagreement is retained, not used to tune the sparse physical prediction.
    """
    import isce3
    grid = burst.as_isce3_radargrid()
    coarse = isce3.product.RadarGridParameters(grid.sensing_start, grid.wavelength,
        grid.prf*6/(grid.length-1), grid.starting_range, grid.range_pixel_spacing*(grid.width-1)/8,
        grid.lookside, 7, 9, grid.ref_epoch)
    lines, samples = np.meshgrid(np.linspace(0, grid.length-1, 7),
                                np.linspace(0, grid.width-1, 9), indexing='ij')
    epoch = (grid.ref_epoch-burst.orbit.reference_epoch).total_seconds()
    dem_interp = isce3.geometry.DEMInterpolator(isce3.io.Raster(str(dem)))
    llh = np.empty(lines.shape+(3,))
    for idx in np.ndindex(lines.shape):
        value = isce3.geometry.rdr2geo(float(epoch+grid.sensing_start+lines[idx]/grid.prf),
            float(grid.starting_range+samples[idx]*grid.range_pixel_spacing), burst.orbit, grid.lookside,
            dem=dem_interp, threshold=1.e-4, maxiter=100, extraiter=10)
        llh[idx] = [np.rad2deg(value[0]), np.rad2deg(value[1]), value[2]]
    lut = burst.az_fm_rate_mismatch_from_llh(llh[..., 1], llh[..., 0], llh[..., 2],
                                           isce3.core.Ellipsoid(), coarse)
    sparse = fm_rate_seconds(burst, lines, samples, llh)
    if not lut.have_data:
        return {'status': 'unavailable', 'reason': 'Installed FM routine returned empty LUT'}
    installed = np.asarray(lut.data)
    bis = burst.bistatic_delay(range_step=coarse.range_pixel_spacing, az_step=1/coarse.prf)
    bis_actual = np.asarray(bis.data)
    bis_samples = (bis.x_start+np.arange(bis_actual.shape[1])*bis.x_spacing-burst.starting_range)/burst.range_pixel_spacing
    bis_expected = np.broadcast_to(bistatic_seconds(burst, bis_samples), bis_actual.shape)
    return {'status': 'complete', 'grid_shape': list(lines.shape),
            'fm_max_difference_seconds': float(abs(installed-sparse).max()),
            'fm_max_difference_native_px': float(abs(installed-sparse).max()/burst.azimuth_time_interval),
            'returned_range_origin_m': float(lut.x_start), 'requested_range_origin_m': float(coarse.starting_range),
            'origin_difference_m': float(lut.x_start-coarse.starting_range),
            'bistatic_max_difference_seconds': float(abs(bis_actual-bis_expected).max()),
            'interpretation': 'Installed FM interpolates coefficients and derives midpoint from supplied grid; '
                              'candidate uses nearest original-mid-burst polynomials and true original range/time origins.'}


def audit(experiment):
    from tools.swath_experiment_io import read_json
    modules = ['glacier_offsets.compass_worker', 'glacier_offsets.compass_coreg',
               'glacier_offsets.inputs', 'compass.s1_rdr2geo', 'compass.s1_geo2rdr',
               'compass.s1_resample', 'compass.s1_geocode_slc', 'compass.utils.lut',
               's1reader.s1_burst_slc', 's1reader.s1_reader']
    source_files, evidence = {}, {}
    for name in modules:
        module = importlib.import_module(name)
        path = Path(inspect.getfile(module))
        source_files[str(path)] = experiment.inputs.file(path)
        lines = path.read_text().splitlines()
        tokens = ('s1_geo2rdr.run', 's1_resample.run', 's1_rdr2geo.run', 'LUT2d()', 'ResampSlc',
                  'get_az_carrier_poly', 'cumulative_correction_luts', 'bistatic_delay',
                  'az_fm_rate_mismatch', 'flag_apply_eap', 'get_nearest_polynomial', 'flatten')
        evidence[name] = [{'line': i+1, 'text': line.strip()} for i, line in enumerate(lines)
                          if any(t in line for t in tokens)]
    runconfigs = {}
    for root in (experiment.stack/'geometry/bursts', experiment.stack/'acquisitions'):
        for path in sorted(root.glob('**/compass_*.yaml')):
            import yaml
            document = yaml.safe_load(path.read_text())['runconfig']['groups']
            runconfigs[str(path)] = {'sha256': experiment.inputs.file(path),
                'flatten': document.get('processing', {}).get('resample', {}).get('flatten'),
                'reference': document['input_file_group'].get('reference_burst'),
                'correction_luts_declared': document.get('processing', {}).get('correction_luts')}
    unchanged_code = source_files[str(Path(importlib.import_module('glacier_offsets.compass_worker').__file__))]
    worker = inspect.getsource(importlib.import_module('glacier_offsets.compass_worker'))
    known_path = all(x in worker for x in ('s1_rdr2geo.run(cfg)', 's1_geo2rdr.run(cfg)', 's1_resample.run(cfg)'))
    specification = {'experiment': experiment.key, 'sources': source_files,
                     'runconfigs': {k: v['sha256'] for k, v in runconfigs.items()},
                     'implementation': experiment.implementation('swath_physics.py', 'swath_experiment_io.py')}
    report = {'status': 'complete', 'executed_radar_path_recognized': known_path,
        'evidence': evidence, 'runconfigs': runconfigs,
        'components': {
            'bistatic': {'radar_path': 'no explicit bistatic LUT application found',
                         'status': 'candidate; historical IPF bulk convention requires independent validation'},
            'fm_rate': {'radar_path': 'carrier/Doppler used in resampling; no explicit FM mismatch timing LUT',
                        'status': 'candidate; nearest-mid-burst selection audited numerically per original burst'},
            'flatten': {'configured_false': all(v['flatten'] is False for v in runconfigs.values()) if runconfigs else None},
            'geocoding_luts': {'applied_by_executed_worker': False,
                              'separate_path': 'compass.s1_geocode_slc; cumulative LUT has negative timing sign'},
            'eap': {'reader_inventory': 'flag_apply_eap=False; radar worker loader audited separately',
                    'interpretation': 'Do not infer an applied radar correction from a geocoding-only capability'}},
        'coordinate_contract': 'global reference -> source local -> local+saved offsets -> acquisition radar time/range; '
                               'no repeated relative-burst shift; negative tagged-time effect through inverse Jacobian',
        'historical_regimes': sorted({metadata(experiment, d)['ipf'] for d in experiment.dates}),
        'dem_height': 'WGS84 ellipsoidal metres inherited from pinned preparation; no vertical conversion',
        'references': ['https://sentiwiki.copernicus.eu/web/s1-processing',
            'https://dataspace.copernicus.eu/news/2024-6-12-copernicus-sentinel-1-etad-products-new-processing-baseline'],
        'physical_validation_status': 'unvalidated_candidate; source audit alone does not establish physical accuracy'}
    return experiment.finish(experiment.output/'audit.json', specification, report)


def metadata(experiment, day):
    from glacier_offsets.initialization import local_xml
    from zipfile import ZipFile
    entry = experiment.inventory[day]
    safe = Path(entry['path'])
    if safe.is_dir():
        path = safe/'manifest.safe'
        experiment.inputs.file(path)
        xml = local_xml(path.read_bytes())
    else:
        experiment.inputs.file(safe)
        with ZipFile(safe) as archive:
            xml = local_xml(archive.read(next(n for n in archive.namelist() if n.endswith('/manifest.safe'))))
    versions = sorted({e.get('version') for e in xml.iter('software') if 'IPF' in e.get('name', '').upper()})
    return {'mission': entry['mission'], 'ipf': ','.join(versions), 'date': day}


def load_native(experiment, day):
    from .inputs import inventory
    from .input_preparation import annotation_paths
    acq = experiment.acquisition(day)
    safe, orbit = acq['specification']['safe']['path'], acq['specification']['orbit']['path']
    experiment.inputs.file(orbit, acq['specification']['orbit']['sha256'])
    if Path(safe).is_dir():
        for relative in annotation_paths(Path(safe)):
            experiment.inputs.file(Path(safe)/relative)
    else:
        experiment.inputs.file(safe, acq['specification']['safe']['sha256'])
    return {b.id: b.native for b in inventory(safe, orbit, experiment.cfg['inputs']['polarization'], [1, 2])}


def native_input_products(experiment, day):
    """Fingerprint the original metadata actually consumed by the forward model.

    These enter the prediction identity independently of aligned-image identity:
    changing an annotation or orbit cannot silently reuse a physical field.
    """
    acq = experiment.acquisition(day)
    safe = Path(acq['specification']['safe']['path'])
    paths = ([safe/'manifest.safe']+[p for p in sorted((safe/'annotation').rglob('*.xml'))
                                   if not p.name.startswith('.')] if safe.is_dir() else [safe])
    result = {str(p.resolve()): experiment.inputs.file(p) for p in paths}
    orbit = acq['specification']['orbit']
    result[orbit['path']] = experiment.inputs.file(orbit['path'], orbit['sha256'])
    dem = experiment.frame['specification']['inputs']['dem']
    result[dem['path']] = experiment.inputs.file(dem['path'], dem['sha256'])
    return result


def physics(experiment):
    from tools.swath_experiment_io import read_json, load_npz, save_npz
    audit_record = experiment.checked_report(experiment.output/'audit.json')
    if not audit_record['executed_radar_path_recognized']:
        raise ValueError('Executed radar path has not passed source audit')
    native, records, checks, original_inputs = {}, {}, {}, {}
    sources = {e['burst_id']: e for e in experiment.frame['sources']}
    for link in experiment.links:
        out = experiment.output/'physics'/link['pair']
        measured = experiment.output/'measurements'/link['pair']/'report.json'
        if not measured.exists() or read_json(measured).get('status') != 'complete':
            records[link['pair']] = {'status': 'unavailable', 'error': 'overlap measurements incomplete'}
            continue
        experiment.checked_report(measured)
        for day in (link['earlier'], link['later']):
            if day not in original_inputs:
                original_inputs[day] = native_input_products(experiment, day)
        input_products = {p: digest for day in (link['earlier'], link['later'])
                          for p, digest in original_inputs[day].items()}
        spec = {'experiment': experiment.key, 'measurement': read_json(measured)['identity'],
                'audit': audit_record['identity'],
                'original_metadata_orbit_dem': input_products,
                'implementation': experiment.implementation('swath_physics.py', 'swath_experiment_io.py')}
        if experiment.current(out/'report.json', spec):
            records[link['pair']] = read_json(out/'report.json')
            continue
        data = load_npz(measured.parent/'observations.npz')
        shape = data['dual_support'].shape
        arrays = {f'{s}_{c}_{quantity}': np.full(shape, np.nan) for s in ('iw1', 'iw2')
                  for c in COMPONENTS for quantity in ('early_seconds', 'late_seconds', 'pair_px')}
        link_checks, errors = {}, []
        for day in (link['earlier'], link['later']):
            if day not in native:
                native[day] = load_native(experiment, day)
        for swath in ('iw1', 'iw2'):
            for number, key in enumerate(data['source_ids'], 1):
                key = str(key)
                take = data['dual_support'] & (data[f'{swath}_owner'] == number)
                if not take.any() or not key.endswith(swath):
                    continue
                try:
                    biases, seconds = [], []
                    entry = sources[key]
                    dy, dx = entry['offset']
                    ref_line, ref_sample = data['center_line'][take]-dy, data['center_sample'][take]-dx
                    llh5 = data[f'{swath}_geometry_llh'][take]
                    for day in (link['earlier'], link['later']):
                        acq = experiment.acquisition(day)
                        for path in acq['valid_windows'].get(key, {}).get('offset_products', []):
                            experiment.inputs.file(path, acq['alignment_products'][path])
                        burst = native[day][key]
                        reference = day == experiment.frame['specification']['anchor']['reference_date']
                        line, sample, jac = mapping(acq, key, ref_line, ref_sample, reference)
                        hy, hx = [(v-1)/2 for v in experiment.cfg['matching']['window']]
                        corner_lines, corner_samples = [], []
                        for a, b in ((0, 0), (-hy, -hx), (-hy, hx), (hy, -hx), (hy, hx)):
                            cy, cx, _ = mapping(acq, key, ref_line+a, ref_sample+b, reference)
                            corner_lines.append(cy)
                            corner_samples.append(cx)
                        geometry = native_geometry_check(burst, np.stack(corner_lines, axis=-1),
                            np.stack(corner_samples, axis=-1), llh5)
                        check_key = f'{day}/{key}'
                        if check_key not in checks:
                            coeff = coefficient_check(burst)
                            try:
                                installed = installed_grid_check(burst, experiment.cfg['inputs']['dem'])
                            except Exception as exc:
                                installed = {'status': 'unavailable', 'error': str(exc)}
                            checks[check_key] = {'coefficients': coeff, 'installed_full_burst_span': installed}
                        link_checks[check_key] = dict(checks[check_key], geometry=geometry,
                            min_jacobian_determinant=float(np.linalg.det(jac).min()), reference_identity=reference)
                        if not geometry['passed'] or not checks[check_key]['coefficients']['passed']:
                            raise ValueError(f'{check_key}: coefficient or native geometry validation failed')
                        bis = bistatic_seconds(burst, sample)
                        fm = fm_rate_seconds(burst, line, sample, llh5[:, 0, :])
                        sec = {'bistatic': bis, 'fm_rate': fm, 'combined': bis+fm}
                        seconds.append(sec)
                        biases.append({c: apparent_bias(value, jac, burst.azimuth_time_interval)[..., 0]
                                       for c, value in sec.items()})
                    for component in COMPONENTS:
                        arrays[f'{swath}_{component}_early_seconds'][take] = seconds[0][component]
                        arrays[f'{swath}_{component}_late_seconds'][take] = seconds[1][component]
                        arrays[f'{swath}_{component}_pair_px'][take] = biases[1][component]-biases[0][component]
                except Exception as exc:
                    errors.append({'swath': swath, 'burst': key, 'error': f'{type(exc).__name__}: {exc}'})
                    logging.error('Physical prediction %s %s: %s', link['pair'], key, exc)
        observed = (data['iw2_total']-data['iw1_total'])[..., 0]
        for component in COMPONENTS:
            arrays[f'{component}_prediction_px'] = arrays[f'iw2_{component}_pair_px']-arrays[f'iw1_{component}_pair_px']
            arrays[f'{component}_residual_px'] = observed-arrays[f'{component}_prediction_px']
        output = out/'predictions.npz'
        save_npz(output, **arrays)
        report = {'status': 'incomplete' if errors else 'complete', 'errors': errors, 'checks': link_checks,
                  'available_n': {c: int(np.isfinite(arrays[f'{c}_prediction_px']).sum()) for c in COMPONENTS},
                  'interpretation': 'unvalidated physical candidates; no fitted scale, timing constant, or empirical residual',
                  'audit_status': audit_record['physical_validation_status']}
        report['registered_file_roles'] = {'prediction': str(output), 'read_only_original_inputs': list(input_products)}
        records[link['pair']] = experiment.finish(out/'report.json', spec, report, [output, *input_products])
        # Only recent dates are needed for adjacent and skip-one links.
        keep = set(experiment.dates[max(0, experiment.dates.index(link['earlier'])-1):])
        for day in list(native):
            if day not in keep:
                native.pop(day)
    return {'status': 'complete' if all(r['status'] == 'complete' for r in records.values()) else 'incomplete',
            'links': {k: {'status': r['status'], 'available_n': r.get('available_n'), 'errors': r.get('errors')}
                      for k, r in records.items()}}

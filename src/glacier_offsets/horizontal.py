"""Convert packaged radar offsets to local east/north displacements."""
import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import h5netcdf
import numpy as np

from .packaging import content_hashes, strings, tiles, validate_package, variable
from .products import ChecksumCache, checksum, identity
from .swath_correction import Inputs
from .tides import json_scalar, output_lock, validate_tidal_stack

METHOD = 'radar-to-local-horizontal-zero-up-v1'
IMPLEMENTATION = {name: checksum(Path(__file__).with_name(name)) for name in ('horizontal.py', 'packaging.py')}
COMPONENTS = ('east_displacement_m', 'north_displacement_m')
COVARIANCE_INPUTS = ('range_displacement_sigma_m', 'azimuth_displacement_sigma_m',
                     'range_azimuth_displacement_covariance_m2')
COVARIANCE_OUTPUTS = ('east_displacement_sigma_m', 'north_displacement_sigma_m',
                      'east_north_displacement_covariance_m2')
DEFAULTS = {'max_condition': 3., 'max_azimuth_scale_ratio': 2., 'min_singular_value': 1e-6}


def settings(max_condition, max_azimuth_scale_ratio, vertical_motion):
    if vertical_motion != 'zero':
        raise ValueError('Only vertical_motion=zero is supported; surface-parallel motion requires reliable slope fields')
    if not np.isfinite(max_condition) or max_condition < 1:
        raise ValueError('max_condition must be finite and at least one')
    if not np.isfinite(max_azimuth_scale_ratio) or max_azimuth_scale_ratio < 1:
        raise ValueError('max_azimuth_scale_ratio must be finite and at least one')
    return DEFAULTS | {'max_condition': float(max_condition), 'max_azimuth_scale_ratio': float(max_azimuth_scale_ratio)}


def input_contract(h):
    if h.attrs.get('geocoding_resampling') != 'nearest':
        raise ValueError('Horizontal conversion requires a nearest-source displacement package')
    axes = h['enu'][:].astype(str).tolist()
    if len(axes) != 3 or set(axes) != {'east', 'north', 'up'}:
        raise ValueError('Input must identify the three ENU sensitivity axes')
    for key in ('range_displacement_m', 'azimuth_displacement_m', 'azimuth_ground_spacing_m_per_slc_line'):
        if h[key].attrs.get('units') != 'm':
            raise ValueError(f'{key} must use metres')
    for key in ('range_sensitivity_enu', 'azimuth_sensitivity_enu'):
        if h[key].dimensions != ('enu', 'y', 'x') or h[key].attrs.get('units') != '1':
            raise ValueError(f'Invalid sensitivity dimensions or units: {key}')
    covariance = all(key in h.variables for key in COVARIANCE_INPUTS)
    if covariance:
        for key in COVARIANCE_INPUTS:
            if h[key].dimensions != ('time', 'y', 'x') or h[key].attrs.get('units') != ('m2' if key.endswith('m2') else 'm'):
                raise ValueError(f'Invalid covariance dimensions or units: {key}')
    return covariance


def scale_medians(h):
    """Normalize within each burst, counting each source observation once."""
    association = h['source_observation_index'][:]
    supported = association >= 0
    ids, first = np.unique(association[supported], return_index=True)
    lookup = np.searchsorted(ids, association[supported])
    owner, scale = (h[key][:][supported] for key in ('source_burst_index', 'azimuth_ground_spacing_m_per_slc_line'))
    for name, values in [('ownership', owner), ('azimuth scale', scale)]:
        if not np.array_equal(values, values[first][lookup], equal_nan=True):
            raise ValueError(f'Repeated source observations have inconsistent {name}')
    owner, scale = owner[first], scale[first]
    result = {}
    for burst in range(1, len(h.dimensions['burst'])+1):
        values = scale[(owner == burst) & np.isfinite(scale) & (scale > 0)]
        if len(values):
            median = float(np.median(values))
            if not np.isfinite(median) or median <= 0:
                raise ValueError('Invalid burst-median azimuth scale')
            result[str(burst)] = median
    return result


def geometry_tile(h, section, options, medians):
    """Return inverse horizontal sensitivity, diagnostics and exclusion flags."""
    axes = h['enu'][:].astype(str).tolist()
    horizontal = [axes.index('east'), axes.index('north')]
    matrix = np.stack([np.moveaxis(h[f'{axis}_sensitivity_enu'][(slice(None), *section)], 0, -1)[..., horizontal]
                       for axis in ('range', 'azimuth')], axis=-2)
    owner = h['source_burst_index'][section]
    supported = (h['source_observation_index'][section] >= 0) & (owner > 0) & (owner <= len(h.dimensions['burst']))
    finite = np.isfinite(matrix).all(axis=(-1, -2)) & supported
    flags = np.where(finite, 0, 1).astype('uint8')
    safe = np.where(finite[..., None, None], matrix, np.eye(2))
    singular = np.linalg.svd(safe, compute_uv=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        condition = singular[..., 0]/singular[..., 1]
    bad = finite & (~np.isfinite(condition) | ~np.isfinite(singular).all(-1) |
                    (condition > options['max_condition']) | (singular[..., 1] < options['min_singular_value']))
    flags[bad] |= 2
    condition[~finite] = np.nan
    median = np.full(owner.shape, np.nan)
    for burst, value in medians.items():
        median[owner == int(burst)] = value
    ratio = h['azimuth_ground_spacing_m_per_slc_line'][section]/median
    limit = options['max_azimuth_scale_ratio']
    flags[supported & (~np.isfinite(ratio) | (ratio < 1/limit) | (ratio > limit))] |= 4
    ratio[~supported] = np.nan
    inverse = np.full_like(matrix, np.nan)
    good = flags == 0
    inverse[good] = np.linalg.inv(matrix[good])
    return inverse, condition, ratio, flags


def convert_tile(h, slot, inverse, geometry_flags, tidal):
    valid = h['valid_mask'][slot].astype(bool)
    flags = (~valid).astype('uint8')
    flags[geometry_flags != 0] |= 2
    if tidal:
        flags[h['tidal_correction_available'][slot] != 1] |= 4
    radar = np.stack([h[key][slot] for key in ('range_displacement_m', 'azimuth_displacement_m')], axis=-1).astype('float64')
    with np.errstate(over='ignore', invalid='ignore'):
        converted = np.einsum('...ij,...j->...i', inverse, radar)
    finite = np.isfinite(converted).all(-1) & (np.abs(converted) <= np.finfo('float32').max).all(-1)
    flags[(flags == 0) & ~finite] |= 8
    converted[flags != 0] = np.nan
    return converted.astype('float32'), flags


def covariance_tile(h, slot, inverse, valid):
    """Propagate complete formal covariance without assuming independent axes."""
    sr, sa, cross = (h[key][slot].astype('float64') for key in COVARIANCE_INPUTS)
    good = valid & np.isfinite(sr) & np.isfinite(sa) & np.isfinite(cross) & (sr >= 0) & (sa >= 0)
    # Slight rounding differences at the positive-semidefinite boundary are
    # tolerated, then clipped to that boundary before propagation.
    bound = sr*sa
    good &= abs(cross) <= bound*(1+1e-6)+1e-12
    cross = np.clip(cross, -bound, bound)
    covariance = np.stack((np.stack((sr**2, cross), -1), np.stack((cross, sa**2), -1)), -2)
    with np.errstate(invalid='ignore', over='ignore'):
        result = inverse @ covariance @ np.swapaxes(inverse, -1, -2)
        fields = np.stack((np.sqrt(np.maximum(result[..., 0, 0], 0)),
                           np.sqrt(np.maximum(result[..., 1, 1], 0)), result[..., 0, 1]), -1)
    good &= np.isfinite(fields).all(-1) & (abs(fields) <= np.finfo('float32').max).all(-1)
    fields[~good] = np.nan
    return fields.astype('float32'), good.astype('uint8')


def integer_variable(h, name, dimensions, *, masks=False, **attrs):
    key = 'flag_masks' if masks else 'flag_values'
    values = attrs.pop('flags', [0, 1])
    return variable(h, name, dimensions, dtype='u1', units='1', grid_mapping='crs',
                    **{key: np.array(values, dtype='uint8')}, **attrs)


def apply_horizontal(temp, source_spec, recipe):
    counts = {'source_valid_cells': 0, 'converted_cells': 0, 'geometry_rejected_valid_cells': 0,
              'tidal_unavailable_valid_cells': 0, 'numeric_rejected_cells': 0, 'uncertainty_available_cells': 0}
    with h5netcdf.File(temp, 'a') as h:
        shape = (len(h.dimensions['y']), len(h.dimensions['x']))
        components = [variable(h, key, ('time', 'y', 'x'), dtype='f4', units='m', grid_mapping='crs',
                      long_name=f'Local {axis} later-minus-earlier displacement; zero residual vertical motion',
                      ancillary_variables='horizontal_valid_mask horizontal_validity_flags')
                      for key, axis in zip(COMPONENTS, ('eastward', 'northward'))]
        masks = {key: integer_variable(h, key, dimensions, flag_meanings='unavailable available' if 'uncertainty' in key else 'invalid valid') for key, dimensions in (
            ('horizontal_valid_mask', ('time', 'y', 'x')), ('horizontal_geometry_valid_mask', ('y', 'x')),
            ('horizontal_uncertainty_available', ('time', 'y', 'x')))}
        geometry_flags = integer_variable(h, 'horizontal_geometry_flags', ('y', 'x'), masks=True, flags=[1, 2, 4],
            flag_meanings='missing_geometry ill_conditioned_or_weak_sensitivity azimuth_scale_outlier')
        validity_flags = integer_variable(h, 'horizontal_validity_flags', ('time', 'y', 'x'), masks=True, flags=[1, 2, 4, 8],
            flag_meanings='input_invalid geometry_rejected tidal_correction_unavailable numeric_out_of_range')
        condition = variable(h, 'horizontal_condition_number', ('y', 'x'), dtype='f8', units='1', grid_mapping='crs',
                             long_name='2-norm condition number of range/azimuth horizontal sensitivity matrix')
        ratio = variable(h, 'horizontal_azimuth_scale_ratio', ('y', 'x'), dtype='f4', units='1', grid_mapping='crs',
                         long_name='Local azimuth scale divided by the unique-source median for its burst')
        covariance = [variable(h, key, ('time', 'y', 'x'), dtype='f4', units='m2' if key.endswith('m2') else 'm',
                      grid_mapping='crs', ancillary_variables='horizontal_uncertainty_available',
                      uncertainty='propagated formal matching covariance; excludes tide, calibration and shared-acquisition errors')
                      for key in COVARIANCE_OUTPUTS] if recipe['covariance_available'] else []
        for section in tiles(shape):
            inverse, c, r, flags = geometry_tile(h, section, recipe['geometry_quality'], recipe['burst_scale_medians_m_per_line'])
            condition[section], ratio[section], geometry_flags[section] = c, r, flags
            masks['horizontal_geometry_valid_mask'][section] = (flags == 0).astype('uint8')
            for i in range(len(h.dimensions['time'])):
                slot = (i, *section)
                values, why = convert_tile(h, slot, inverse, flags, recipe['tidal_input'])
                take = why == 0
                for j, target in enumerate(components):
                    target[slot] = values[..., j]
                masks['horizontal_valid_mask'][slot], validity_flags[slot] = take.astype('uint8'), why
                available = np.zeros(take.shape, dtype='uint8')
                if covariance:
                    propagated, available = covariance_tile(h, slot, inverse, take)
                    for j, target in enumerate(covariance):
                        target[slot] = propagated[..., j]
                masks['horizontal_uncertainty_available'][slot] = available
                valid = h['valid_mask'][slot].astype(bool)
                counts['source_valid_cells'] += int(valid.sum())
                counts['converted_cells'] += int(take.sum())
                counts['geometry_rejected_valid_cells'] += int((valid & ((why & 2) != 0)).sum())
                counts['tidal_unavailable_valid_cells'] += int((valid & ((why & 4) != 0)).sum())
                counts['numeric_rejected_cells'] += int(((why & 8) != 0).sum())
                counts['uncertainty_available_cells'] += int(available.sum())
        if counts['source_valid_cells'] and not counts['converted_cells']:
            raise ValueError('No usable horizontal observations after geometry and tidal-availability checks')
        spec = dict(source_spec, postprocessing=source_spec.get('postprocessing', [])+[recipe])
        h['provenance_json'][()] = json.dumps(spec, sort_keys=True)
        strings(h, 'horizontal_conversion_provenance_json', (), json.dumps({'recipe': recipe, 'counts': counts}, sort_keys=True))
        h.attrs.update(package_identity=identity(spec), horizontal_conversion_method=METHOD,
                       horizontal_conversion_identity=identity(recipe), horizontal_vector_basis='local geographic east/north; physical metres',
                       horizontal_vertical_motion='zero residual vertical displacement in the supplied radar observations',
                       horizontal_azimuth_tidal_correction='disabled; input azimuth used as supplied',
                       horizontal_uncertainty='full formal covariance propagated' if covariance else 'unavailable; complete input covariance absent',
                       horizontal_conversion_status='empty' if not counts['source_valid_cells'] else
                       'complete' if counts['converted_cells'] == counts['source_valid_cells'] else 'partial')
        h.attrs['history'] = (str(h.attrs.get('history', ''))+'\n'+datetime.now(timezone.utc).isoformat()+' '+METHOD).strip()
        hashes = content_hashes(h)
        for key, expected in recipe['preserved_content_hashes'].items():
            if hashes.get(key) != expected:
                raise ValueError(f'Horizontal conversion unexpectedly changed {key}')
        h.attrs['content_hashes_json'] = json.dumps(hashes, sort_keys=True)
    return counts


def validate_horizontal_stack(path):
    """Check converted measurements, original fields and geometry using one file."""
    report = validate_package(path)
    with h5netcdf.File(path) as h:
        if h.attrs.get('horizontal_conversion_method') != METHOD:
            raise ValueError('File is not a supported horizontal displacement stack')
        record = json_scalar(h, 'horizontal_conversion_provenance_json')
        recipe = record['recipe']
        covariance = input_contract(h)
        if covariance != recipe['covariance_available']:
            raise ValueError('Horizontal covariance availability differs from the input contract')
        if recipe['method'] != METHOD or h.attrs.get('horizontal_conversion_identity') != identity(recipe):
            raise ValueError('Horizontal conversion provenance differs from its identity')
        if json_scalar(h, 'provenance_json').get('postprocessing', [])[-1:] != [recipe]:
            raise ValueError('Horizontal conversion is not the recorded final processing step')
        hashes = json.loads(h.attrs['content_hashes_json'])
        if any(hashes.get(key) != value for key, value in recipe['preserved_content_hashes'].items()):
            raise ValueError('Horizontal conversion changed a source variable')
        if scale_medians(h) != recipe['burst_scale_medians_m_per_line']:
            raise ValueError('Horizontal scale normalization differs from source geometry')
        names = COMPONENTS + ('horizontal_valid_mask', 'horizontal_validity_flags', 'horizontal_uncertainty_available')
        if recipe['covariance_available']:
            names += COVARIANCE_OUTPUTS
        for key in names:
            v = h[key]
            integer = key not in COMPONENTS+COVARIANCE_OUTPUTS
            if v.dimensions != ('time', 'y', 'x') or v.dtype != np.dtype('u1' if integer else 'f4'):
                raise ValueError(f'Invalid horizontal variable: {key}')
            if v.attrs.get('units') != ('1' if integer else 'm2' if key.endswith('m2') else 'm'):
                raise ValueError(f'Invalid horizontal units: {key}')
            if integer and '_FillValue' in v.attrs:
                raise ValueError(f'Integer mask/flags must not have a fill value: {key}')
        for key in ('horizontal_geometry_flags', 'horizontal_geometry_valid_mask'):
            if h[key].dimensions != ('y', 'x') or h[key].dtype != np.dtype('uint8') or '_FillValue' in h[key].attrs:
                raise ValueError(f'Invalid horizontal geometry mask/flags: {key}')
        counts = {key: 0 for key in record['counts']}
        axes = h['enu'][:].astype(str).tolist()
        for section in tiles(report['shape'][1:]):
            inverse, condition, ratio, flags = geometry_tile(h, section, recipe['geometry_quality'], recipe['burst_scale_medians_m_per_line'])
            for key, expected in [('horizontal_condition_number', condition), ('horizontal_azimuth_scale_ratio', ratio.astype('float32')),
                                  ('horizontal_geometry_flags', flags), ('horizontal_geometry_valid_mask', (flags == 0).astype('uint8'))]:
                np.testing.assert_array_equal(h[key][section], expected, err_msg=key)
            for i in range(report['pairs']):
                slot = (i, *section)
                expected, why = convert_tile(h, slot, inverse, flags, recipe['tidal_input'])
                np.testing.assert_array_equal(h['horizontal_valid_mask'][slot], (why == 0).astype('uint8'))
                np.testing.assert_array_equal(h['horizontal_validity_flags'][slot], why)
                for j, key in enumerate(COMPONENTS):
                    np.testing.assert_array_equal(h[key][slot], expected[..., j], err_msg=key)
                # Check the forward equations independently of the inverse,
                # allowing only float32 storage roundoff in EN components.
                take = why == 0
                east, north = (h[key][slot].astype('float64') for key in COMPONENTS)
                for axis in ('range', 'azimuth'):
                    sensitivity = h[f'{axis}_sensitivity_enu'][(slice(None), *section)]
                    terms = [sensitivity[axes.index('east')]*east, sensitivity[axes.index('north')]*north]
                    radar = h[f'{axis}_displacement_m'][slot]
                    tolerance = 4*np.finfo('float32').eps*(abs(terms[0])+abs(terms[1]))+1e-6
                    if np.any(abs(terms[0][take]+terms[1][take]-radar[take]) > tolerance[take]):
                        raise ValueError(f'Horizontal components do not reproduce source {axis} displacements')
                available = np.zeros(why.shape, dtype='uint8')
                if recipe['covariance_available']:
                    covariance, available = covariance_tile(h, slot, inverse, why == 0)
                    for j, key in enumerate(COVARIANCE_OUTPUTS):
                        np.testing.assert_array_equal(h[key][slot], covariance[..., j], err_msg=key)
                np.testing.assert_array_equal(h['horizontal_uncertainty_available'][slot], available)
                valid = h['valid_mask'][slot].astype(bool)
                counts['source_valid_cells'] += int(valid.sum())
                counts['converted_cells'] += int(take.sum())
                counts['geometry_rejected_valid_cells'] += int((valid & ((why & 2) != 0)).sum())
                counts['tidal_unavailable_valid_cells'] += int((valid & ((why & 4) != 0)).sum())
                counts['numeric_rejected_cells'] += int(((why & 8) != 0).sum())
                counts['uncertainty_available_cells'] += int(available.sum())
        if counts != record['counts']:
            raise ValueError('Horizontal conversion counts disagree with observations')
        return report | {'conversion': METHOD, 'horizontal_status': h.attrs['horizontal_conversion_status'], 'counts': record['counts']}


def to_horizontal(source, *, output=None, vertical_motion='zero', max_condition=3., max_azimuth_scale_ratio=2., force=False):
    options = settings(max_condition, max_azimuth_scale_ratio, vertical_motion)
    source = Path(source).expanduser().resolve()
    destination = Path(output).expanduser().resolve() if output else source.with_name(source.stem+'_horizontal.nc')
    if destination == source or (destination.exists() and os.path.samefile(destination, source)):
        raise ValueError('Horizontal conversion requires a separate output file')
    if destination.suffix.lower() != '.nc':
        raise ValueError('Horizontal output must be a netCDF .nc file')
    inputs = Inputs(ChecksumCache())
    logging.info('Validating source displacement stack')
    digest = inputs.file(source)
    with h5netcdf.File(source) as h:
        tidal = bool(h.attrs.get('tidal_correction_method'))
        if h.attrs.get('horizontal_conversion_method') or any(key in h.variables for key in COMPONENTS):
            raise ValueError('Input already has horizontal conversion; use its radar-component parent stack')
    (validate_tidal_stack if tidal else validate_package)(source)
    with h5netcdf.File(source) as h:
        covariance = input_contract(h)
        spec = json_scalar(h, 'provenance_json')
        recipe = {'method': METHOD, 'implementation': IMPLEMENTATION,
                  'input': {'path': str(source), 'sha256': digest, 'package_identity': h.attrs['package_identity']},
                  'geometry_quality': options, 'burst_scale_medians_m_per_line': scale_medians(h),
                  'vertical_motion': vertical_motion, 'azimuth_tidal_correction': 'disabled; use input unchanged',
                  'tidal_input': tidal, 'tidal_missing_policy': 'exclude from horizontal valid support',
                  'vector_basis': 'local geographic east/north, independent of map CRS axes; metres',
                  'grid_policy': 'preserve pixel centers, CRS, transform and all source observations; no interpolation',
                  'timing': 'pair displacements; preserve exact burst endpoints; midpoint is only a label',
                  'covariance_available': covariance, 'covariance_inputs': list(COVARIANCE_INPUTS),
                  'software': {'numpy': np.__version__, 'h5netcdf': h5netcdf.__version__},
                  'preserved_content_hashes': {k: v for k, v in json.loads(h.attrs['content_hashes_json']).items() if k != 'provenance_json'}}
    with output_lock(destination.parent):
        if destination.exists():
            with h5netcdf.File(destination) as h:
                if h.attrs.get('horizontal_conversion_method') != METHOD:
                    raise ValueError('Refusing to replace an unrelated output file')
                current = h.attrs.get('horizontal_conversion_identity') == identity(recipe)
            if current and not force:
                report = validate_horizontal_stack(destination)
                inputs.unchanged()
                return report | {'reused': True}
            if not force:
                raise ValueError('Output has a different conversion recipe; choose another output or use --force')
        with tempfile.TemporaryDirectory(prefix='.horizontal-stack-', dir=destination.parent) as folder:
            temp = Path(folder)/destination.name
            shutil.copyfile(source, temp)
            counts = apply_horizontal(temp, spec, recipe)
            logging.info('Converted %d/%d valid observations', counts['converted_cells'], counts['source_valid_cells'])
            report = validate_horizontal_stack(temp)
            inputs.unchanged()
            os.replace(temp, destination)
        return report | {'path': str(destination), 'reused': False}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='glacier-offsets to-horizontal', description=__doc__)
    parser.add_argument('stack', help='Radar displacement netCDF, optionally range-tide corrected')
    parser.add_argument('-o', '--output', help='Separate .nc output; default INPUT_horizontal.nc')
    parser.add_argument('--vertical-motion', choices=['zero'], default='zero', help='Assume zero residual vertical motion (default)')
    parser.add_argument('--max-condition', type=float, default=3., help='Maximum horizontal matrix condition number (default 3)')
    parser.add_argument('--max-azimuth-scale-ratio', type=float, default=2., help='Maximum scale ratio to its burst median, and reciprocal minimum (default 2)')
    parser.add_argument('--force', action='store_true', help='Rebuild conversion from the radar-component parent')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        print(json.dumps(to_horizontal(args.stack, output=args.output, vertical_motion=args.vertical_motion,
              max_condition=args.max_condition, max_azimuth_scale_ratio=args.max_azimuth_scale_ratio, force=args.force), indent=2))
        return 0
    except KeyboardInterrupt:
        logging.error('Interrupted; the input and any completed output are preserved')
        return 130
    except (ValueError, OSError, RuntimeError, ImportError, AssertionError, KeyError) as exc:
        logging.error('%s', exc)
        return 2

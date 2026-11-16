"""Pair-varying GOFF corrections and horizontal conversion."""
from datetime import datetime, timezone
import json

import h5netcdf
import numpy as np

from .displacement import endpoints, sensitivity, validate_nisar
from .packaging import variable, strings, tiles, content_hashes
from .products import identity


def finish(h, source_spec, recipe, attributes, record_name, record):
    spec = dict(source_spec, postprocessing=source_spec.get('postprocessing', [])+[recipe])
    h['provenance_json'][()] = json.dumps(spec, sort_keys=True)
    strings(h, record_name, (), json.dumps(record, sort_keys=True))
    h.attrs.update(package_identity=identity(spec), **attributes)
    h.attrs['history'] = (str(h.attrs.get('history', ''))+'\n'+datetime.now(timezone.utc).isoformat()+' '+recipe['method']).strip()
    h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)


def apply_tides_nisar(path, source_spec, model, recipe):
    from .tides import METHOD, FLOAT_FIELDS
    stats = {'corrected_cells': 0, 'uncorrected_valid_cells': 0, 'zero_response_cells': 0, 'max_abs_range_bias_m': 0.}
    with h5netcdf.File(path, 'a') as h:
        shape = (len(h.dimensions['y']), len(h.dimensions['x']))
        fields = {k: variable(h, k, ('time', 'y', 'x'), dtype='f4', units='m') for k in FLOAT_FIELDS}
        available = variable(h, 'tidal_correction_available', ('time', 'y', 'x'), dtype='u1', units='1')
        for name, a in [('tidal_response_factor', model.response_factor), ('tidal_model_ocean_mask', model.ocean_mask)]:
            variable(h, name, ('y', 'x'), a.reshape(shape).astype('f4'), units='1')
        up = h['enu'][:].astype(str).tolist().index('up')
        for i in range(len(h.dimensions['time'])):
            for section in tiles(shape):
                slot = (i, *section)
                rows, cols = np.meshgrid(np.arange(shape[0])[section[0]], np.arange(shape[1])[section[1]], indexing='ij')
                indices = (rows*shape[1]+cols).ravel()
                e, l = endpoints(h, i, section)
                valid = h['valid_mask'][slot] == 1
                finite = np.isfinite(e+l).ravel()
                early = np.full(len(indices), np.nan); late = early.copy()
                if finite.any():
                    early[finite] = model.predict(indices[finite], e.ravel()[finite])
                    late[finite] = model.predict(indices[finite], l.ravel()[finite])
                early, late = early.reshape(e.shape), late.reshape(e.shape)
                delta = late-early; bias = delta*sensitivity(h, 'range_sensitivity_enu', i, section)[up]
                take = valid & np.isfinite(bias) & model.available[indices].reshape(e.shape)
                raw = h['range_displacement_m'][slot]
                fields['range_displacement_before_tides_m'][slot] = raw
                for k, a in [('tide_height_earlier_m', early), ('tide_height_later_m', late),
                             ('tidal_vertical_displacement_m', delta), ('tidal_range_bias_m', bias)]:
                    fields[k][slot] = np.where(take, a, np.nan)
                available[slot] = take.astype('u1')
                corrected = raw.copy(); corrected[take] = raw[take].astype(float)-bias[take]
                h['range_displacement_m'][slot] = corrected
                stats['corrected_cells'] += int(take.sum()); stats['uncorrected_valid_cells'] += int((valid & ~take).sum())
                stats['zero_response_cells'] += int((take & (model.response_factor[indices].reshape(e.shape) == 0)).sum())
                if take.any():
                    stats['max_abs_range_bias_m'] = max(stats['max_abs_range_bias_m'], float(np.max(abs(bias[take]))))
        finish(h, source_spec, recipe, dict(tidal_correction_method=METHOD, tidal_correction_identity=identity(recipe),
               tidal_correction_status='partial_model_coverage' if stats['uncorrected_valid_cells'] else 'complete',
               tidal_correction_response=recipe['response'], tidal_correction_azimuth='unchanged; correction applies only to range'),
               'tidal_correction_provenance_json', recipe | {'counts': stats, 'model': model.metadata})
    return stats


def geometry(h, i, section, recipe):
    axes = h['enu'][:].astype(str).tolist(); en = [axes.index(k) for k in ('east', 'north')]
    a = np.stack([np.moveaxis(sensitivity(h, k+'_sensitivity_enu', i, section)[en], 0, -1) for k in ('range', 'azimuth')], -2)
    finite = np.isfinite(a).all(axis=(-1, -2)) & (h['geometry_valid_mask'][(i, *section)] == 1)
    safe = np.where(finite[..., None, None], a, np.eye(2)); s = np.linalg.svd(safe, compute_uv=False)
    cond = s[..., 0]/np.maximum(s[..., 1], 1e-30)
    options = recipe['geometry_quality']; flags = (~finite).astype('u1')
    flags[finite & ((cond > options['max_condition']) | (s[..., 1] < options['min_singular_value']))] |= 2
    median = recipe['burst_scale_medians_m_per_line'][str(i)]
    scale = h['azimuth_ground_spacing_m_per_slc_line'][(i, *section)]
    ratio = scale/(median if median is not None else np.nan)
    limit = options['max_azimuth_scale_ratio']; flags[~np.isfinite(ratio) | (ratio < 1/limit) | (ratio > limit)] |= 4
    inverse = np.linalg.inv(safe); inverse[flags != 0] = np.nan
    cond[~finite] = np.nan
    return inverse, cond, ratio, flags


def apply_horizontal_nisar(path, spec, recipe):
    from .horizontal import METHOD, COMPONENTS, COVARIANCE_OUTPUTS, covariance_tile, scale_medians
    stats = {'source_valid_cells': 0, 'converted_cells': 0, 'geometry_rejected_valid_cells': 0,
             'tidal_unavailable_valid_cells': 0, 'numeric_rejected_cells': 0, 'uncertainty_available_cells': 0}
    with h5netcdf.File(path, 'a') as h:
        dims = ('time', 'y', 'x'); shape = (len(h.dimensions['y']), len(h.dimensions['x']))
        out = {}
        for k in COMPONENTS+COVARIANCE_OUTPUTS:
            out[k] = variable(h, k, dims, dtype='f4', units='m2' if k.endswith('m2') else 'm')
        for k in ('horizontal_valid_mask', 'horizontal_validity_flags', 'horizontal_uncertainty_available',
                  'horizontal_geometry_flags', 'horizontal_geometry_valid_mask'):
            out[k] = variable(h, k, dims, dtype='u1', units='1')
        for k in ('horizontal_condition_number', 'horizontal_azimuth_scale_ratio'):
            out[k] = variable(h, k, dims, dtype='f8', units='1')
        for i in range(len(h.dimensions['time'])):
            for sec in tiles(shape):
                slot = (i, *sec); inv, cond, ratio, flags = geometry(h, i, sec, recipe)
                valid = h['valid_mask'][slot] == 1
                why = (~valid).astype('u1'); why[flags != 0] |= 2
                if recipe['tidal_input']:
                    why[h['tidal_correction_available'][slot] != 1] |= 4
                ra = np.stack([h[k][slot] for k in ('range_displacement_m', 'azimuth_displacement_m')], -1)
                en = np.einsum('...ij,...j->...i', inv, ra)
                why[~np.isfinite(en).all(-1) | (abs(en) > np.finfo('f4').max).any(-1)] |= 8
                good = why == 0; en[~good] = np.nan
                for j, k in enumerate(COMPONENTS): out[k][slot] = en[..., j]
                cov, unc = covariance_tile(h, slot, inv, good)
                for j, k in enumerate(COVARIANCE_OUTPUTS): out[k][slot] = cov[..., j]
                for k, a in [('horizontal_valid_mask', good), ('horizontal_validity_flags', why),
                             ('horizontal_uncertainty_available', unc), ('horizontal_geometry_flags', flags),
                             ('horizontal_geometry_valid_mask', flags == 0), ('horizontal_condition_number', cond),
                             ('horizontal_azimuth_scale_ratio', ratio)]: out[k][slot] = a
                for k, a in [('source_valid_cells', valid), ('converted_cells', good),
                             ('geometry_rejected_valid_cells', valid & (flags != 0)),
                             ('tidal_unavailable_valid_cells', valid & ((why & 4) != 0)),
                             ('numeric_rejected_cells', valid & ((why & 8) != 0)), ('uncertainty_available_cells', unc)]: stats[k] += int(a.sum())
        if stats['source_valid_cells'] and not stats['converted_cells']:
            raise ValueError('No usable horizontal observations after geometry and tidal-availability checks')
        finish(h, spec, recipe, dict(horizontal_conversion_method=METHOD, horizontal_conversion_identity=identity(recipe),
               horizontal_vector_basis='local geographic east/north; physical metres', horizontal_vertical_motion='zero residual vertical displacement',
               horizontal_conversion_status='complete' if stats['source_valid_cells'] == stats['converted_cells'] else 'partial'),
               'horizontal_conversion_provenance_json', {'recipe': recipe, 'counts': stats})
    return stats


def validate_horizontal_nisar(path):
    from .horizontal import METHOD, COMPONENTS, COVARIANCE_OUTPUTS, covariance_tile, scale_medians
    result = validate_nisar(path)
    with h5netcdf.File(path) as h:
        record = json.loads(h['horizontal_conversion_provenance_json'][()].decode()); recipe = record['recipe']
        if h.attrs.get('horizontal_conversion_method') != METHOD or h.attrs['horizontal_conversion_identity'] != identity(recipe):
            raise ValueError('Horizontal recipe mismatch')
        if scale_medians(h) != recipe['burst_scale_medians_m_per_line']:
            raise ValueError('Horizontal scale normalization differs')
        hashes = json.loads(h.attrs['content_hashes_json'])
        if any(hashes.get(k) != v for k, v in recipe['preserved_content_hashes'].items()):
            raise ValueError('Horizontal conversion changed source content')
        for i in range(result['pairs']):
            for sec in tiles(result['shape'][1:]):
                slot = (i, *sec); inv, _, _, flags = geometry(h, i, sec, recipe)
                good = h['horizontal_valid_mask'][slot] == 1
                expected = (h['valid_mask'][slot] == 1) & (flags == 0)
                if recipe['tidal_input']: expected &= h['tidal_correction_available'][slot] == 1
                if not np.array_equal(expected, good): raise ValueError('Horizontal validity mismatch')
                ra = np.stack([h[k][slot] for k in ('range_displacement_m', 'azimuth_displacement_m')], -1)
                en = np.stack([h[k][slot] for k in COMPONENTS], -1)
                np.testing.assert_allclose(en[good], np.einsum('...ij,...j->...i', inv, ra)[good], rtol=2e-6, atol=2e-6)
                if not np.isnan(en[~good]).all(): raise ValueError('Invalid horizontal support')
                cov, unc = covariance_tile(h, slot, inv, good)
                for j, k in enumerate(COVARIANCE_OUTPUTS): np.testing.assert_allclose(h[k][slot], cov[..., j], rtol=2e-6, atol=2e-6, equal_nan=True)
                if not np.array_equal(h['horizontal_uncertainty_available'][slot], unc): raise ValueError('Invalid uncertainty support')
    return result | {'conversion': METHOD}

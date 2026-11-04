"""Configuration for the separate two-year exploratory pixel inversion."""
from copy import deepcopy
import math
from numbers import Real
from pathlib import Path

import numpy as np
import yaml


DEFAULT_PARAMETERS = {
    'screening': {
        'remove_mixed_windows': False,
        'min_peak': 0.08,
        'max_range_residual_m': 2.0,
        'max_azimuth_residual_m': 5.0,
    },
    'coverage': {
        'min_pairs_per_year': 20,
        'min_pairs_total': 40,
        'min_span_days': 540.0,
        'phase_bins': 12,
        'min_phase_bins': 9,
        'min_pairs_standard': 60,
    },
    'uncertainty': {'cluster_days': 30.0},
    'display': {
        'min_speed_m_day': 0.1,
        'min_abs_azimuth_projection': 0.3,
        'max_direction_change_deg': 20.0,
        'diagnostic_min_abs_azimuth_projection': 0.5,
        'max_phase_sigma_deg': 30.0,
        'min_amplitude_over_sigma': 2.0,
        'proxy_min_speed_m_day': 0.5,
    },
}


class _UniqueLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently using the last value."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError('Configuration keys must be strings')
        if key in result:
            raise ValueError(f'Duplicate configuration key: {key}')
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _number(value, name, minimum=0, maximum=None, integer=False, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or (integer and not isinstance(value, int))):
        raise ValueError(f'{name} must be a finite {"integer" if integer else "number"}')
    if value < minimum or (positive and value == 0) or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside its allowed range')


def validate_parameters(overrides=None):
    """Merge partial sections with defaults; reject typos and invalid thresholds."""
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError('Inversion parameters must be a mapping')
    unknown = overrides.keys() - DEFAULT_PARAMETERS.keys()
    if unknown:
        raise ValueError(f'Unknown inversion sections: {sorted(unknown)}')
    result = deepcopy(DEFAULT_PARAMETERS)
    for section, values in overrides.items():
        if not isinstance(values, dict):
            raise ValueError(f'{section} must be a mapping')
        unknown = values.keys() - result[section].keys()
        if unknown:
            raise ValueError(f'Unknown {section} keys: {sorted(unknown)}')
        result[section].update(values)
        for name, value in result[section].items():
            if section == 'screening' and name == 'remove_mixed_windows':
                if type(value) is not bool:
                    raise ValueError('screening.remove_mixed_windows must be true or false')
                continue
            _number(value, f'{section}.{name}')
    s = result['screening']
    _number(s['min_peak'], 'screening.min_peak', maximum=1)
    for name in ('max_range_residual_m', 'max_azimuth_residual_m'):
        _number(s[name], f'screening.{name}', positive=True)
    c = result['coverage']
    for name in ('min_pairs_per_year', 'min_pairs_total', 'phase_bins', 'min_phase_bins', 'min_pairs_standard'):
        _number(c[name], f'coverage.{name}', integer=True, positive=True)
    if c['min_phase_bins'] > c['phase_bins']:
        raise ValueError('coverage.min_phase_bins cannot exceed phase_bins')
    if c['phase_bins'] > 127:
        raise ValueError('coverage.phase_bins cannot exceed 127 (stored bin-count dtype)')
    if c['min_pairs_standard'] < c['min_pairs_total']:
        raise ValueError('coverage.min_pairs_standard cannot be below min_pairs_total')
    _number(result['uncertainty']['cluster_days'], 'uncertainty.cluster_days', positive=True)
    d = result['display']
    for name in ('min_abs_azimuth_projection', 'diagnostic_min_abs_azimuth_projection'):
        _number(d[name], f'display.{name}', maximum=1)
    for name in ('max_direction_change_deg', 'max_phase_sigma_deg'):
        _number(d[name], f'display.{name}', maximum=180)
    if d['diagnostic_min_abs_azimuth_projection'] < d['min_abs_azimuth_projection']:
        raise ValueError('Diagnostic projection threshold cannot be below the ordinary display threshold')
    if d['proxy_min_speed_m_day'] < d['min_speed_m_day']:
        raise ValueError('Proxy speed threshold cannot be below the ordinary display threshold')
    return result


def load_config(path):
    """Read YAML, resolving input/output paths relative to its own directory."""
    path = Path(path).resolve()
    with path.open() as stream:
        raw = yaml.load(stream, Loader=_UniqueLoader)
    if not isinstance(raw, dict):
        raise ValueError('Pixel inversion YAML must be a mapping')
    allowed = set(DEFAULT_PARAMETERS) | {'version', 'input_directory', 'output', 'execution', 'regions', 'regional_support'}
    unknown = raw.keys() - allowed
    if unknown:
        raise ValueError(f'Unknown pixel inversion keys: {sorted(unknown)}')
    if type(raw.get('version', 1)) is not int or raw.get('version', 1) != 1:
        raise ValueError('Only pixel inversion configuration version 1 is supported')
    result = {'version': 1}
    for name in ('input_directory', 'output'):
        value = raw.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} must be a nonempty path')
        result[name] = str((path.parent / Path(value).expanduser()).resolve())
    if result['output'] == result['input_directory']:
        raise ValueError('Choose an output directory different from the prepared input directory')
    for name in ('regions', 'regional_support'):
        value = raw.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f'{name} must be a path or null')
        result[name] = None if value is None else str((path.parent / Path(value).expanduser()).resolve())
    execution = raw.get('execution', {})
    if not isinstance(execution, dict) or execution.keys() - {'batch_size'}:
        raise ValueError('execution accepts only batch_size')
    batch = execution.get('batch_size', 512)
    _number(batch, 'execution.batch_size', integer=True, positive=True)
    result['execution'] = {'batch_size': batch}
    result.update(validate_parameters({k: raw[k] for k in DEFAULT_PARAMETERS if k in raw}))
    return result


def parameters_from_recipe(recipe):
    """Preserve the old exclusion when replaying recipes predating this option."""
    parameters=deepcopy(recipe.get('parameters') or {})
    parameters.setdefault('screening',{}).setdefault('remove_mixed_windows',True)
    return validate_parameters(parameters)


def display_masks(estimable, az_estimable, standard, speed, angle, projection,
                  amplitude, sigma, phase_sigma, options):
    """Apply only display choices, leaving stored fitted coefficients intact."""
    minimum_projection = np.min(abs(projection), axis=1)
    along = (estimable & standard & (speed >= options['min_speed_m_day'])
             & (minimum_projection >= options['min_abs_azimuth_projection']))
    ratio = np.divide(amplitude, sigma, out=np.zeros_like(amplitude), where=sigma > 0)
    direction = ((angle <= options['max_direction_change_deg'])
                 & (minimum_projection >= options['diagnostic_min_abs_azimuth_projection']))
    return {
        'along': along,
        'proxy': along & direction & (speed >= options['proxy_min_speed_m_day']),
        'phase_quality': along & direction & (phase_sigma <= options['max_phase_sigma_deg'])
                         & (ratio >= options['min_amplitude_over_sigma']),
        'azimuth': az_estimable & standard,
        'azimuth_moving': az_estimable & standard & (speed >= options['min_speed_m_day']),
    }

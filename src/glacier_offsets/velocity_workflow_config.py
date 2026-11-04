"""Version-2 YAML for the netCDF-to-results pixel inversion workflow.

One file names the input displacement packages, cache and result directories,
corrections, annual noise model and inversion settings. Paths resolve relative
to the YAML. Scientific thresholds reuse the version-1 defaults and checks,
except that version 2 requires an explicit ``coverage.min_span_days``.
"""
from copy import deepcopy
from pathlib import Path

import yaml

from .velocity_pixel_config import DEFAULT_PARAMETERS, _UniqueLoader, _number, validate_parameters


VERSION = 2
NOISE_MODES = ('supplied', 'controls')
CONTROL_MINIMUM_SAMPLES = 40  # calibrate_noise's existing held-out residual minimum.
EXECUTION_DEFAULTS = {'solid_earth_python': None, 'batch_size': 512,
                      'preparation_chunk_pairs': 8, 'solid_earth_spacing_deg': 0.1}
TOP_LEVEL = {'version', 'inputs', 'cache_directory', 'output', 'corrections', 'noise',
             'diagnostics', 'execution', *DEFAULT_PARAMETERS}


def config_version(path):
    """Return the declared version without validating the rest of the file."""
    with Path(path).open() as stream:
        raw = yaml.load(stream, Loader=_UniqueLoader)
    if not isinstance(raw, dict):
        raise ValueError('Pixel inversion YAML must be a mapping')
    version = raw.get('version', 1)
    if type(version) is not int or version not in (1, VERSION):
        raise ValueError('Pixel inversion configuration version must be 1 or 2')
    return version


def _path(base, value, name, required=True):
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty path')
    return str((base / Path(value).expanduser()).resolve())


def _mapping(raw, name, allowed):
    value = raw.get(name, {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    unknown = value.keys() - set(allowed)
    if unknown:
        raise ValueError(f'Unknown {name} keys: {sorted(unknown)}')
    return value


def _years(value, name):
    if (not isinstance(value, list) or not value
            or any(type(y) is not int or not 1900 <= y <= 2200 for y in value)):
        raise ValueError(f'{name} must be a nonempty list of calendar years')
    if len(set(value)) != len(value):
        raise ValueError(f'{name} contains duplicate years')
    return list(value)


def load_workflow_config(path):
    """Validate a version-2 YAML before any input is opened or output created."""
    path = Path(path).expanduser().resolve()
    with path.open() as stream:
        raw = yaml.load(stream, Loader=_UniqueLoader)
    if not isinstance(raw, dict):
        raise ValueError('Pixel inversion YAML must be a mapping')
    if raw.get('version') != VERSION or type(raw.get('version')) is not int:
        raise ValueError('The netCDF-to-results workflow requires "version: 2"')
    unknown = raw.keys() - TOP_LEVEL
    if unknown:
        raise ValueError(f'Unknown pixel inversion keys: {sorted(unknown)}')
    base = path.parent
    inputs = raw.get('inputs')
    if not isinstance(inputs, list) or not inputs:
        raise ValueError('inputs must be a nonempty list of displacement netCDF paths')
    inputs = [_path(base, item, 'inputs entry') for item in inputs]
    if len(set(inputs)) != len(inputs):
        raise ValueError('inputs lists the same file more than once')
    result = {'version': VERSION, 'config_file': str(path), 'inputs': inputs,
              'cache_directory': _path(base, raw.get('cache_directory'), 'cache_directory'),
              'output': _path(base, raw.get('output'), 'output')}
    if result['output'] == result['cache_directory']:
        raise ValueError('Choose an output directory different from cache_directory')
    if result['output'] in inputs or result['cache_directory'] in inputs:
        raise ValueError('cache_directory and output must not be input files')

    corrections = _mapping(raw, 'corrections', ('ocean_tide', 'solid_earth'))
    ocean = corrections.get('ocean_tide', 'require_existing')
    if ocean != 'require_existing':
        raise ValueError('corrections.ocean_tide must be require_existing; CATS corrections are not computed here')
    solid = corrections.get('solid_earth', True)
    if type(solid) is not bool:
        raise ValueError('corrections.solid_earth must be true or false')
    result['corrections'] = {'ocean_tide': ocean, 'solid_earth': solid}

    noise = _mapping(raw, 'noise', ('mode', 'regions', 'group_by', 'block_size_m', 'files'))
    mode = noise.get('mode')
    if mode not in NOISE_MODES:
        raise ValueError('noise.mode must be supplied or controls')
    if noise.get('group_by', 'year') != 'year':
        raise ValueError('noise.group_by supports only year')
    record = {'mode': mode, 'group_by': 'year'}
    if mode == 'controls':
        if 'files' in noise:
            raise ValueError('noise.files is only used with mode: supplied')
        record['regions'] = _path(base, noise.get('regions'), 'noise.regions')
        block = noise.get('block_size_m', 3000.)
        _number(block, 'noise.block_size_m', positive=True)
        record['block_size_m'] = float(block)
        record['min_samples'] = CONTROL_MINIMUM_SAMPLES
    else:
        if 'regions' in noise or 'block_size_m' in noise:
            raise ValueError('noise.regions and block_size_m are only used with mode: controls')
        files = noise.get('files')
        if not isinstance(files, list) or not files:
            raise ValueError('noise.files must list supplied covariance files')
        entries = []
        for item in files:
            if isinstance(item, str):
                item = {'path': item}
            if not isinstance(item, dict) or item.keys() - {'path', 'years'} or 'path' not in item:
                raise ValueError('Each noise.files entry is a path or a mapping with path and optional years')
            entries.append({'path': _path(base, item['path'], 'noise.files path'),
                            'years': None if item.get('years') is None else _years(item['years'], 'noise.files years')})
        record['files'] = entries
    result['noise'] = record

    diagnostics = _mapping(raw, 'diagnostics', ('regions',))
    regions = diagnostics.get('regions', record.get('regions'))
    result['diagnostics'] = {'regions': None if regions is None else
                             (regions if regions == record.get('regions') else _path(base, regions, 'diagnostics.regions'))}

    execution = deepcopy(EXECUTION_DEFAULTS)
    execution.update(_mapping(raw, 'execution', EXECUTION_DEFAULTS))
    if execution['solid_earth_python'] is not None:
        execution['solid_earth_python'] = str(Path(execution['solid_earth_python']).expanduser()) \
            if Path(execution['solid_earth_python']).expanduser().is_absolute() else \
            _path(base, execution['solid_earth_python'], 'execution.solid_earth_python')
    for name in ('batch_size', 'preparation_chunk_pairs'):
        _number(execution[name], f'execution.{name}', integer=True, positive=True)
    _number(execution['solid_earth_spacing_deg'], 'execution.solid_earth_spacing_deg', positive=True, maximum=1)
    result['execution'] = execution

    coverage = raw.get('coverage')
    if not isinstance(coverage, dict) or 'min_span_days' not in coverage:
        raise ValueError('Version 2 requires an explicit coverage.min_span_days')
    result.update(validate_parameters({k: raw[k] for k in DEFAULT_PARAMETERS if k in raw}))
    return result


def scientific_parameters(config):
    return {k: deepcopy(config[k]) for k in DEFAULT_PARAMETERS}


def impossible_settings(config, pairs_by_year, span_days):
    """Settings the observed pair record can never satisfy, reported together."""
    c = config['coverage']
    problems = []
    if not pairs_by_year:
        return ['no pairs were collected']
    for year, count in sorted(pairs_by_year.items()):
        if count < c['min_pairs_per_year']:
            problems.append(f'{year} has {count} pairs, fewer than coverage.min_pairs_per_year={c["min_pairs_per_year"]}')
    total = sum(pairs_by_year.values())
    if total < c['min_pairs_total']:
        problems.append(f'{total} pairs in total, fewer than coverage.min_pairs_total={c["min_pairs_total"]}')
    if total < c['min_pairs_standard']:
        problems.append(f'{total} pairs in total, fewer than coverage.min_pairs_standard={c["min_pairs_standard"]}')
    if span_days < c['min_span_days']:
        problems.append(f'the collected burst intervals span {span_days:.1f} days, less than coverage.min_span_days={c["min_span_days"]:g}')
    if total < c['min_phase_bins']:
        problems.append(f'{total} pairs cannot occupy coverage.min_phase_bins={c["min_phase_bins"]} Msf phase bins')
    return problems

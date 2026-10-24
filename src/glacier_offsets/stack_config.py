"""Independent stack configuration, retaining YAML-relative inheritance."""
from datetime import date
import math
from pathlib import Path

from .config import _load_overrides, load_defaults, merge_config, validate_common

STACK_DEFAULTS = {'start_date': None, 'end_date': None, 'reference_date': None, 'processing': 'isce3',
                  'swath_merge': 'overwrite',
                  'pairing': {'neighbors': 1, 'max_separation_days': 12, 'pairs_file': None}}


def iso_date(value, label):
    # PyYAML decodes unquoted ISO dates as date objects.
    if type(value) is date:
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError(f'{label} must use YYYY-MM-DD')
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'{label} must use YYYY-MM-DD') from exc
    if parsed.isoformat() != value:
        raise ValueError(f'{label} must use YYYY-MM-DD')
    return value


def _load_stack_settings(path):
    overrides = _load_overrides(Path(path))
    cfg = merge_config(load_defaults(), overrides)
    # Preserve the original opt-in strategy's engine when omitted, while
    # honoring explicit engine choices (including inherited YAML settings).
    if (isinstance(cfg.get('stack'), dict) and
            cfg['stack'].get('processing') == 'isce2_merge_first' and
            isinstance(cfg.get('matching'), dict) and
            'engine' not in overrides.get('matching', {})):
        cfg['matching']['engine'] = 'isce2'
    return cfg


def load_stack_config(path, discovery=False):
    cfg = _load_stack_settings(path)
    return validate_stack_config(cfg, discovery=discovery)


def load_status_config(path):
    """Resolve settings for inspection even when raw-data volumes are offline."""
    cfg = _load_stack_settings(path)
    if not isinstance(cfg.get('stack'), dict):
        raise ValueError('stack must be a YAML mapping')
    for section in ('inputs', 'output', 'matching', 'gross_offsets', 'geocoding'):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f'{section} must be a YAML mapping')
    if isinstance(cfg['inputs'].get('polarization'), str):
        cfg['inputs']['polarization'] = cfg['inputs']['polarization'].upper()
    output = cfg.get('output', {}).get('directory')
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise ValueError('output.directory must be a nonempty local path')
    return cfg


def validate_stack_config(cfg, discovery=False):
    stack = cfg.get('stack')
    if not isinstance(stack, dict):
        raise ValueError('stack must be a YAML mapping')
    unknown = set(stack) - set(STACK_DEFAULTS) - {'safe_directory', 'orbit_directory'}
    if unknown:
        raise ValueError(f'Unknown stack options: {sorted(unknown)}')
    stack = merge_config(STACK_DEFAULTS, stack)
    if stack['processing'] not in ('isce3', 'isce2_merge_first'):
        raise ValueError('stack.processing must be isce3 or isce2_merge_first')
    if stack['swath_merge'] not in ('overwrite', 'midpoint'):
        raise ValueError('stack.swath_merge must be overwrite or midpoint')
    if stack['swath_merge'] == 'midpoint' and stack['processing'] != 'isce2_merge_first':
        raise ValueError('stack.swath_merge=midpoint requires stack.processing=isce2_merge_first')
    for key in ('safe_directory', 'orbit_directory'):
        value = stack.get(key)
        if not value or not Path(value).is_dir():
            raise ValueError(f'stack.{key} must be an existing directory')
    for key in ('start_date', 'end_date', 'reference_date'):
        if stack[key] is not None:
            stack[key] = iso_date(stack[key], f'stack.{key}')
    if stack['start_date'] and stack['end_date'] and stack['start_date'] > stack['end_date']:
        raise ValueError('stack.start_date must not follow end_date')
    pairing = stack['pairing']
    if not isinstance(pairing, dict) or set(pairing) - set(STACK_DEFAULTS['pairing']):
        raise ValueError('stack.pairing accepts neighbors, max_separation_days and pairs_file')
    if type(pairing['neighbors']) is not int or pairing['neighbors'] < 1:
        raise ValueError('stack.pairing.neighbors must be a positive integer')
    cap = pairing['max_separation_days']
    if cap is not None and (type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0):
        raise ValueError('stack.pairing.max_separation_days must be positive or null')
    if pairing['pairs_file'] is not None and not Path(pairing['pairs_file']).is_file():
        raise ValueError('stack.pairing.pairs_file must be an existing CSV')
    cfg['stack'] = stack
    output = cfg.get('output')
    if (not isinstance(output, dict) or not isinstance(output.get('directory'), str)
            or not Path(output['directory']).is_absolute()):
        raise ValueError('output.directory must be a nonempty local path')
    return validate_common(cfg, discovery=discovery)

"""Pinned spacecraft calibration settings; safe for metadata-only inspection."""
import hashlib
import json
import math
from pathlib import Path
import re

METHOD = 'spacecraft-swath-azimuth-v1'
DEFAULTS = {'enabled': False, 'calibration': None, 'calibration_sha256': None,
            'pairs_directory': None}
MATCHING_KEYS = ('engine', 'backend', 'window', 'search', 'stride', 'first_center',
                 'deramp', 'range_oversampling', 'raw_oversampling', 'oversampling', 'zoom')


def settings(cfg):
    value = cfg.get('stack', {}).get('swath_correction', {})
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError('stack.swath_correction accepts enabled, calibration, calibration_sha256 and pairs_directory')
    result = DEFAULTS | value
    if type(result['enabled']) is not bool:
        raise ValueError('stack.swath_correction.enabled must be boolean')
    if not result['enabled']:
        return result
    for key in ('calibration', 'pairs_directory'):
        if result[key] is not None and (not isinstance(result[key], str) or not Path(result[key]).is_absolute()):
            raise ValueError(f'stack.swath_correction.{key} must be an absolute path after YAML resolution')
    pin = result['calibration_sha256']
    if not result['calibration'] or not isinstance(pin, str) or not re.fullmatch('[0-9a-f]{64}', pin):
        raise ValueError('Enabled swath correction requires calibration and its calibration_sha256 pin')
    source = Path(cfg['output']['directory']).resolve()
    target = Path(result['pairs_directory'] or source / 'pairs_swath_corrected').resolve()
    protected = [source / name for name in ('pairs', 'acquisitions', 'geometry', '.cache')]
    if target == source or target in source.parents or any(
            target == p or target in p.parents or p in target.parents for p in protected):
        raise ValueError('swath correction pairs_directory must be separate from raw stack products')
    if target in Path(result['calibration']).resolve().parents:
        raise ValueError('Calibration must be outside the corrected pairs directory')
    result['pairs_directory'] = str(target)
    if cfg['stack'].get('processing') not in ('isce3_merge_first', 'isce2_merge_first'):
        raise ValueError('Swath correction requires a merge-first stack with source ownership')
    if cfg['matching'].get('engine', 'isce3') != 'isce3' or cfg['gross_offsets']['mode'] != 'zero':
        raise ValueError('Current swath correction supports ISCE3 matching with zero gross offsets')
    if cfg['geocoding'].get('resampling', 'nearest') != 'nearest':
        raise ValueError('Current swath correction requires nearest-source geocoding')
    return result


def load_calibration(cfg):
    options = settings(cfg)
    if not options['enabled']:
        raise ValueError('Enable stack.swath_correction before running correct-swaths')
    path = Path(options['calibration'])
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != options['calibration_sha256']:
        raise ValueError('Swath calibration checksum differs from calibration_sha256')
    cal = json.loads(data)
    if not isinstance(cal, dict) or cal.get('method') != METHOD or cal.get('units') != 'original_reference_azimuth_pixels':
        raise ValueError('Unsupported swath calibration method or units')
    for key in ('beta_px', 'beta_std_px', 'reference_prf_hz'):
        value = cal.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'Calibration {key} must be finite')
    if cal['beta_std_px'] < 0 or cal['reference_prf_hz'] <= 0:
        raise ValueError('Calibration uncertainty must be nonnegative and PRF positive')
    if cal.get('swaths') != [1, 2] or cal.get('allocation') != 'symmetric':
        raise ValueError('Calibration must describe symmetric IW1/IW2 azimuth allocation')
    scope = cal.get('scope', {})
    if not isinstance(scope, dict) or any(key not in scope for key in (
            'geometry_identity', 'relative_orbit', 'direction', 'polarization',
            'processing', 'swath_merge', 'matching', 'start_date', 'end_date')):
        raise ValueError('Calibration must record its geometry, track, matching and date scope')
    from .stack_config import iso_date
    for key in ('start_date', 'end_date'):
        iso_date(scope[key], f'calibration.scope.{key}')
    if scope['start_date'] > scope['end_date']:
        raise ValueError('Calibration scope dates are reversed')
    if not isinstance(scope['matching'], dict) or set(scope['matching']) != set(MATCHING_KEYS):
        raise ValueError('Calibration matching scope is incomplete')
    if scope['polarization'] != cfg['inputs']['polarization']:
        raise ValueError('Calibration polarization differs from this stack')
    for key in ('processing', 'swath_merge'):
        if scope[key] != cfg['stack'].get(key):
            raise ValueError(f'Calibration stack.{key} differs from this stack')
    actual = {k: cfg['matching'].get(k) for k in MATCHING_KEYS}
    # JSON comparison also distinguishes boolean flags from integer settings.
    if json.dumps(scope['matching'], sort_keys=True) != json.dumps(actual, sort_keys=True):
        raise ValueError('Calibration matching treatment differs from this stack')
    return cal, digest


def output_root(cfg):
    return Path(settings(cfg)['pairs_directory'])

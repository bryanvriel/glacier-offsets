"""Exact displacement and velocity bases in UTC elapsed days.

Temporal observation matrices are evaluated by iceutils; EN layout stays local.
"""
from datetime import datetime, timezone
import numpy as np
from .velocity_temporal import matrix as temporal_matrix

PERIOD = 14.765294
ORIGIN = '2000-01-01T00:00:00+00:00'
EPOCH_SECONDS = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()


def utc_days(values):
    """Convert ISO/datetime values to days since the fixed UTC phase origin."""
    def convert(value):
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                return np.nan
            return float((value - np.datetime64('2000-01-01T00:00:00')) / np.timedelta64(1, 'D'))
        date = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return (date.timestamp() - EPOCH_SECONDS) / 86400.
    array = np.asarray(values, dtype=object)
    return np.array([convert(x) for x in array.flat]).reshape(array.shape)


def harmonic(early, late=None, period=PERIOD, *, observation='integral'):
    """Observe fixed-origin cosine/sine velocity (integrals use elapsed days)."""
    return temporal_matrix((('periodic', 0., float(period)),), early, late, observation=observation)


def definition(early, late, background='constant', scales=(60., 120., 240.)):
    if background not in ('constant', 'linear', 'spline_l2', 'spline_l1'):
        raise ValueError('Unknown velocity background model')
    lo, hi = float(np.nanmin(early)), float(np.nanmax(late))
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError('Basis requires finite, increasing pair endpoints')
    splines = []
    if background.startswith('spline'):
        for scale in scales:
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Spline scales must be positive days')
            for center in np.arange(np.floor(lo/scale)*scale-scale, hi+2*scale, scale):
                splines.append({'center': float(center), 'scale': float(scale)})
    return {'background': background, 'reference_day': (lo+hi)/2,
            'period_days': PERIOD, 'phase_origin': ORIGIN, 'splines': splines,
            'spline_degree': 3, 'time_units': 'UTC elapsed days since 2000-01-01'}


def background_specs(basis):
    """Rebuild the stored velocity dictionary without changing its coordinates."""
    specs = [('constant',)]
    if basis['background'] != 'constant':
        specs.append(('linear', basis['reference_day']))
    specs.extend(('bspline', spec['center'], spec['scale'], basis['spline_degree'])
                 for spec in basis['splines'])
    return tuple(specs)


def background_columns(t, basis, *, displacement=False):
    """Velocity dictionary or its analytic primitive (unobservable constant omitted)."""
    specs = background_specs(basis)
    if not displacement:
        return temporal_matrix(specs, t)
    # Retain historical primitive offsets for callers evaluating a primitive.
    columns = []
    for spec in specs:
        origin = spec[1]-2*spec[2] if spec[0] == 'bspline' else basis['reference_day']
        columns.append(temporal_matrix((spec,), origin, t, observation='integral')[..., 0])
    return np.stack(columns, -1)


def design(early, late, basis, parameterization, direction):
    """Return (pair, EN, parameter) design, names and spline penalty indices."""
    bg = temporal_matrix(background_specs(basis), early, late, observation='integral')
    ht = harmonic(early, late, basis['period_days'])
    nbg = bg.shape[-1]
    nt = 4 if parameterization == 'free_en' else 2
    if parameterization not in ('free_en', 'along_flow'):
        raise ValueError('parameterization must be free_en or along_flow')
    result = np.zeros((len(bg), 2, 2*nbg+nt))
    names = []
    for j in range(nbg):
        for axis in range(2):
            result[:, axis, 2*j+axis] = bg[:, j]
            label = ('mean', 'trend')[j] if j < (1 if basis['background']=='constant' else 2) else f'spline_{j}'
            names.append(f'{label}_{("east", "north")[axis]}')
    if parameterization == 'free_en':
        for axis in range(2):
            result[:, axis, 2*nbg+2*axis:2*nbg+2*axis+2] = ht
        names += ['Msf_C_east', 'Msf_S_east', 'Msf_C_north', 'Msf_S_north']
    else:
        result[:, :, 2*nbg:] = ht[:, None, :]*np.asarray(direction)[None, :, None]
        names += ['Msf_C_along', 'Msf_S_along']
    unpen = 1 if basis['background'] == 'constant' else 2
    penalty = np.arange(2*unpen, 2*nbg)
    return result, names, penalty


def evaluate(coefficients, basis, parameterization, direction, days):
    """Evaluate local EN instantaneous velocity from the stored basis alone."""
    bg = background_columns(days, basis)
    nbg = bg.shape[-1]
    coefficients = np.asarray(coefficients)
    velocity = bg @ coefficients[:2*nbg].reshape(nbg, 2)
    ht = harmonic(days, period=basis['period_days'], observation='sample')
    target = coefficients[2*nbg:]
    if parameterization == 'free_en':
        velocity += ht @ target.reshape(2, 2).T
    else:
        velocity += (ht @ target)[:, None]*np.asarray(direction)
    return velocity

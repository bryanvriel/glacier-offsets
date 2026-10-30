"""Exact displacement and velocity bases in UTC elapsed days.

No processing/native modules are imported by the temporal fitting package.
"""
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import BSpline

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


def harmonic(early, late, period=PERIOD):
    """Integrate velocity cosine/sine analytically, returning metres per m/day."""
    early, late = np.broadcast_arrays(early, late)
    omega = 2*np.pi/period
    mid = (early+late)/2
    duration = late-early
    # sinc form avoids cancellation for short intervals and retains the sign.
    factor = duration*np.sinc(duration/period)
    return np.stack((factor*np.cos(omega*mid), factor*np.sin(omega*mid)), -1)


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


def _spline(t, spec, integrated):
    center, scale = spec['center'], spec['scale']
    knots = center+scale*np.arange(-2, 3)
    b = BSpline.basis_element(knots, extrapolate=False)
    t = np.asarray(t)
    if integrated:
        integral = b.antiderivative()
        return integral(np.clip(t, knots[0], knots[-1]))-integral(knots[0])
    return np.where((t >= knots[0]) & (t <= knots[-1]), b(np.clip(t, knots[0], knots[-1])), 0.)


def background_columns(t, basis, *, displacement=False):
    """Velocity dictionary or its analytic primitive (unobservable constant omitted)."""
    t = np.asarray(t, dtype=float)
    dt = t-basis['reference_day']
    columns = [dt if displacement else np.ones_like(t)]
    if basis['background'] != 'constant':
        columns.append(dt**2/2 if displacement else dt)
    columns.extend(_spline(t, spec, displacement) for spec in basis['splines'])
    return np.stack(columns, -1)


def design(early, late, basis, parameterization, direction):
    """Return (pair, EN, parameter) design, names and spline penalty indices."""
    bg = background_columns(late, basis, displacement=True)-background_columns(early, basis, displacement=True)
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
    omega = 2*np.pi/basis['period_days']
    ht = np.stack((np.cos(omega*np.asarray(days)), np.sin(omega*np.asarray(days))), -1)
    target = coefficients[2*nbg:]
    if parameterization == 'free_en':
        velocity += ht @ target.reshape(2, 2).T
    else:
        velocity += (ht @ target)[:, None]*np.asarray(direction)
    return velocity

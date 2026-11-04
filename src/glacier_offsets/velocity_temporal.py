"""Array adapter for iceutils temporal observation operators.

Public times are UTC elapsed days since 2000-01-01. iceutils receives naive
UTC datetimes, never numeric days (which it would interpret as decimal years).
Collections and repeated endpoint evaluations are bounded, in-process caches.
"""
from functools import lru_cache
from hashlib import sha256
from importlib import import_module
from pathlib import Path

import numpy as np


BACKEND_MODULES = (
    'iceutils.tseries.timefn.Collection', 'iceutils.tseries.timefn.observations',
    'iceutils.tseries.timefn.BasisFn', 'iceutils.tseries.timefn.CenteredBasisFn',
    'iceutils.tseries.timefn.datetimeUtils', 'iceutils.tseries.timefn.Constant',
    'iceutils.tseries.timefn.Power', 'iceutils.tseries.timefn.Trigonometry',
    'iceutils.tseries.timefn.Periodic', 'iceutils.tseries.timefn.Spline',
    'iceutils.tseries.timefn.Interface',
)


@lru_cache(maxsize=1)
def timefn():
    """Require the operator-enabled iceutils API when numerical work starts."""
    message = ('Velocity fitting requires iceutils temporal observation operators '
               '(PR #28, commit 2c95cba or later). Install the velocity extra or '
               'update iceutils in the interpreter running this command.')
    try:
        from iceutils import tseries
    except ImportError as error:
        raise ImportError(message) from error
    if tseries is None or not hasattr(tseries.timefn.TimefnCollection, 'design_matrix'):
        raise ImportError(message)
    return tseries.timefn


def backend_fingerprints():
    """Hash the actual imported operator implementation, independent of cwd."""
    timefn()
    return {name: sha256(Path(import_module(name).__file__).read_bytes()).hexdigest()
            for name in BACKEND_MODULES}


def dates(days):
    """Convert finite elapsed days to NumPy datetimes at microsecond precision."""
    micros = np.rint(np.asarray(days)*86400_000_000).astype(np.int64)
    return np.datetime64('2000-01-01', 'us') + micros.astype('timedelta64[us]')


@lru_cache(maxsize=32)
def _collection(specs):
    fn = timefn()
    collection = fn.TimefnCollection()
    for spec in specs:
        kind, *values = spec
        if kind == 'constant':
            bounds = {key: dates(value).item() for key, value in zip(('tmin', 'tmax'), values)
                      if value is not None}
            basis = fn.fnmap[kind](**bounds)
        elif kind == 'linear':
            basis = fn.fnmap[kind](tref=dates(values[0]).item(), units='days')
        elif kind == 'bspline':
            center, scale, degree = values
            half_width = scale*(degree+1)/2
            # The point evaluator uses truncated powers. Explicit support keeps
            # floating-point cancellation in distant tails from leaking velocity.
            basis = fn.fnmap[kind](tref=dates(center).item(), units='days', scale=scale, order=degree,
                                   tmin=dates(center-half_width).item(), tmax=dates(center+half_width).item())
        elif kind == 'periodic':
            origin, period = values
            basis = fn.fnmap[kind](tref=dates(origin).item(), units='days', period=period)
        else:
            raise ValueError(f'Unknown temporal basis: {kind}')
        collection.append(basis)
    return collection


@lru_cache(maxsize=16)
def _evaluate(specs, observation, endpoints):
    width = 1 if observation == 'sample' else 2
    points = np.frombuffer(endpoints, dtype=np.float64).reshape(-1, width)
    starts = dates(points[:, 0])
    ends = None if width == 1 else dates(points[:, 1])
    return _collection(specs).design_matrix(starts, observation=observation, t_end=ends, time_units='days')


def matrix(specs, start, end=None, *, observation='sample'):
    """Evaluate a fixed basis, restoring broadcast shape and missing rows.

    NaN/nonfinite endpoints yield NaN rows. Signed integrals and differences
    retain the old array helpers' behavior for reversed intervals. Zero-length
    integrals/differences are zero; zero-length averages are undefined (NaN).
    """
    if observation not in ('sample', 'integral', 'average', 'difference'):
        raise ValueError(f'Unknown temporal observation: {observation}')
    specs = tuple(specs)
    npar = sum(2 if spec[0] == 'periodic' else 1 for spec in specs)
    start = np.asarray(start, dtype=float)
    if observation == 'sample':
        if end is not None:
            raise ValueError('Sample observations do not accept endpoints')
        valid = np.isfinite(start)
        points = start[valid, None]
        sign = 1.
    else:
        if end is None:
            raise ValueError('Interval observations require endpoints')
        start, end = np.broadcast_arrays(start, np.asarray(end, dtype=float))
        valid = np.isfinite(start) & np.isfinite(end)
        zero = valid & (start == end)
        valid &= ~zero
        points = np.column_stack((np.minimum(start[valid], end[valid]), np.maximum(start[valid], end[valid])))
        sign = (np.sign(end[valid]-start[valid])[:, None]
                if observation in ('integral', 'difference') else 1.)
    result = np.full((*start.shape, npar), np.nan)
    if observation != 'sample' and observation != 'average':
        result[zero] = 0.
    if valid.any():
        # Burst endpoints repeat across pixels. Evaluate each distinct pair once.
        unique, inverse = np.unique(points, axis=0, return_inverse=True)
        result[valid] = _evaluate(specs, observation, unique.tobytes())[inverse]*sign
    return result

"""Shared inputs and refits for the block-level Msf tools.

``tools/aggregate_msf_blocks.py`` (full-record blocks) and
``tools/split_period_msf_blocks.py`` (acquisition-disjoint halves) both start
from a completed velocity-pixel result: the run's recipe and cached artifacts,
its mapped sources projected to a metric CRS, and the same square blocks. This
module keeps those steps in one place so the two tools cannot disagree about
which sources, blocks or coordinates they use. The block statistics themselves
are in :mod:`glacier_offsets.velocity_blocks`.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import yaml

from .velocity_basis import utc_days
from .velocity_blocks import block_keys
from .velocity_pixel_config import validate_parameters
from .velocity_pixel_general import MODELS, Layout, coverage, screen, solve

# Grounded slow blocks (empirical null) and floating blocks, by CATS response and speed.
CLASS_DEFAULTS = {'null_max_response': .02, 'null_max_speed': .1, 'floating_min_response': .5}
PAIR_KEYS = ('order', 'year_index', 'calendar_early', 'calendar_late', 'within_year', 'file_index')
SOURCE_PAIR_KEYS = ('early', 'late', 'data', 'peak', 'valid')


def artifact_paths(result, cache=None):
    """Recipe and the observation/solid-Earth artifacts a run used.

    The cache directory comes from the run's resolved configuration unless
    ``cache`` is given (for a cache that has moved).
    """
    result = Path(result)
    recipe = json.loads((result/'recipe.json').read_text())
    if cache is None:
        config = yaml.safe_load((result/'inversion-config.resolved.yaml').read_text())
        cache = config['cache_directory']
    cache = Path(cache)
    art = recipe['artifacts']
    paths = {'observations': cache/'observations'/art['observations']['id']/'observations.h5',
             'solid_earth': cache/'solid_earth'/art['solid_earth']['id']/'solid_earth.h5'}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f'{name} artifact {path} is missing; pass the cache directory if it moved')
    return recipe, {k: str(v) for k, v in paths.items()}


def result_parameters(recipe):
    """Recipe parameters with defaults for keys older recipes predate (which reproduce the legacy fit)."""
    return validate_parameters(recipe['parameters'])


def stored_covariances(h):
    """The covariance a stored fit used: annual array, or annual plus per-pair."""
    cov = h['covariance_by_year'][:]
    if 'covariance_by_pair' in h:
        return {'by_year': cov, 'by_pair': h['covariance_by_pair'][:]}
    return cov


def block_crs(h, override=None):
    """Projected CRS for blocks: ``override`` or the result's own CRS."""
    crs = CRS.from_user_input(override) if override else CRS.from_wkt(h.attrs.get('crs_wkt') or h['crs'].attrs['crs_wkt'])
    if not crs.is_projected:
        raise ValueError(f'Blocks need a projected CRS in metres; {crs.name} is not')
    return crs


def load_sources(result_file, model='residual_vertical', crs=None, min_abs_projection=0.):
    """Mapped sources of a result: standard coverage and an estimable fit for ``model``.

    ``min_abs_projection`` also drops sources whose smallest annual absolute
    azimuth projection of the flow axis is below it (the pixel maps' display
    rule uses 0.3). Along-flow Msf is identified from azimuth, so its noise and
    any cross-flow or systematic azimuth term scale with 1/projection.
    Returns the source indices, their projected x/y, stored C/S, mean flow
    speed, CATS response (NaN when the result has none), minimum absolute
    projection and the CRS.
    """
    if model not in MODELS:
        raise ValueError(f'model must be one of {list(MODELS)}')
    with h5py.File(result_file) as h:
        if h.attrs.get('status') not in (None, 'complete'):
            raise ValueError(f"{result_file} is not complete (status {h.attrs['status']})")
        projection = np.abs(h['azimuth_flow_projection_by_year'][:]).min(1)
        keep = h['standard_coverage'][:].astype(bool) & h[f'{model}/estimable'][:].astype(bool)
        if min_abs_projection > 0:
            keep &= projection >= min_abs_projection
        selected = np.flatnonzero(keep)
        crs = block_crs(h, crs)
        x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(h['longitude'][:][selected],
                                                                          h['latitude'][:][selected])
        out = {'selected': selected, 'x': np.asarray(x), 'y': np.asarray(y), 'cs': h[f'{model}/cs'][:][selected],
               'speed': h['mean_flow_speed'][:][selected], 'projection': projection[selected], 'crs': crs,
               'response': (h['tidal_response_factor'][:][selected] if 'tidal_response_factor' in h
                            else np.full(len(selected), np.nan))}
    if not len(selected):
        raise ValueError('No sources have standard coverage, an estimable fit and the required projection')
    return out


def assign_blocks(x, y, sizes):
    """{size: (keys, labels)}: unique (ix, iy) block keys and each source's index into them."""
    out = {}
    for size in sizes:
        ix, iy = block_keys(x, y, size)
        keys, labels = np.unique(np.stack([ix, iy], 1), axis=0, return_inverse=True)
        out[size] = (keys, labels.ravel())
    return out


def per_block(values, labels, nblocks, how):
    """Per-block reduction ('median', 'max', ...) of per-source values; NaN for empty blocks."""
    return getattr(pd.Series(values).groupby(labels), how)().reindex(range(nblocks)).to_numpy()


def classify(max_response, median_response, median_speed, *, null_max_response=CLASS_DEFAULTS['null_max_response'],
             null_max_speed=CLASS_DEFAULTS['null_max_speed'],
             floating_min_response=CLASS_DEFAULTS['floating_min_response']):
    """Grounded slow and floating block masks. NaN responses never satisfy either rule."""
    max_response, median_response, median_speed = map(np.asarray, (max_response, median_response, median_speed))
    with np.errstate(invalid='ignore'):
        grounded_slow = (max_response <= null_max_response) & (median_speed <= null_max_speed)
        floating = median_response >= floating_min_response
    return grounded_slow, floating


def median_split_date(calendar_early, calendar_late):
    """ISO date (UTC midnight) at or before the median pair midpoint."""
    mid = float(np.median((np.asarray(calendar_early)+np.asarray(calendar_late))/2))
    return str(np.datetime64('2000-01-01')+np.timedelta64(int(np.floor(mid)), 'D'))


def split_halves(calendar_early, calendar_late, split_day):
    """Pair columns of two acquisition-disjoint halves split by pair midpoint.

    Pairs whose early or late acquisition day also occurs in the other half
    are dropped from both. Returns ({'A': columns, 'B': columns}, shared days).
    """
    ce, cl = np.asarray(calendar_early), np.asarray(calendar_late)
    mid = (ce+cl)/2
    a, b = mid < split_day, mid >= split_day

    def days(mask):
        return set(np.floor(ce[mask]).tolist()) | set(np.floor(cl[mask]).tolist())
    shared = sorted(days(a) & days(b))
    touches = np.isin(np.floor(ce), shared) | np.isin(np.floor(cl), shared)
    return {'A': np.flatnonzero(a & ~touches), 'B': np.flatnonzero(b & ~touches)}, shared


def split_day_number(date):
    """Days since the 2000-01-01 UTC origin at midnight of an ISO date."""
    return float(utc_days(str(np.datetime64(date, 'D'))))


def subset_pairs(field, layout, columns):
    """Field restricted to pair columns, with calendar-year regimes renumbered.

    Returns the field, its Layout (only years and order regimes still present)
    and the original indices of the retained years.
    """
    columns = np.asarray(columns)
    old = field['year_index'][columns]
    present = np.unique(old)
    remap = np.full(len(layout.years), -1)
    remap[present] = np.arange(len(present))
    sub = {k: (v[columns] if k in PAIR_KEYS else v[:, columns] if k in SOURCE_PAIR_KEYS else v) for k, v in field.items()}
    sub['year_index'] = remap[old]
    if 'geometry_by_year' in sub:
        sub['geometry_by_year'] = sub['geometry_by_year'][present]
    years = [layout.years[k] for k in present]
    order_years = [y for i, y in enumerate(years) if np.any(sub['order'][sub['year_index'] == i] != 0)]
    spacecraft = {str(y): layout.order_spacecraft[str(y)] for y in order_years}
    return sub, Layout(years, order_years, spacecraft), present


def subset_covariances(covariances, present, columns):
    """Covariance for a pair subset: retained years, and retained pairs in pair mode."""
    if isinstance(covariances, dict):
        return {'by_year': covariances['by_year'][present], 'by_pair': covariances['by_pair'][columns]}
    return covariances[present]


def fit_pairs(field, layout, covariances, parameters, model='residual_vertical'):
    """Screen, coverage and one model fit, exactly as fit_sources does for a full record.

    Returns the fit and the sources with standard coverage and an estimable fit.
    """
    s = parameters['screening']
    options = s
    if isinstance(covariances, dict):  # pair mode bypasses the metre cuts (velocity_pixel_general.fit_sources)
        options = s | {'max_range_residual_m': np.inf, 'max_azimuth_residual_m': np.inf}
    good, axis, _, _, _ = screen(field, layout, options)
    support = coverage(field, layout, good, parameters['coverage'])
    fit = solve(field, layout, good, axis, covariances, model, cluster_days=parameters['uncertainty']['cluster_days'],
                robust_cutoff=s['robust_cutoff'], robust=s['robust'], robust_scale=s['robust_scale'])
    return fit, support['standard_coverage'] & support['fit_support'] & fit['estimable']

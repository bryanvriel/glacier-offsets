"""Aligned Sentinel-1 collections and joint range/azimuth pixel fits."""
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import json

import h5py
import numpy as np
from pyproj import CRS, Geod

from .velocity_annual_noise import check_covariance
from .velocity_basis import EPOCH_SECONDS, harmonic
from .velocity_pixel_general import Layout, TARGET_NAMES, VERTICAL_NAMES, _angles, coverage, predict, screen, solve
from .velocity_vertical import vertical_columns


def relative_orbit(path):
    with h5py.File(path, 'r') as h:
        keys = ('earlier_relative_orbit', 'later_relative_orbit', 'earlier_orbit_direction', 'later_orbit_direction')
        if not all(k in h and h[k].shape == h['pair_id'].shape for k in keys):
            raise ValueError(f'{path}: expected Sentinel-1 orbit metadata for every pair')
        labels = np.unique(np.concatenate([h[k][:] for k in keys[:2]]))
        directions = set(np.concatenate([h[k][:].astype(str) for k in keys[2:]]))
        if (len(labels) != 1 or not np.issubdtype(labels.dtype, np.integer) or not 1 <= labels[0] <= 175
                or len(directions) != 1 or not directions <= {'ASCENDING', 'DESCENDING'}):
            raise ValueError(f'{path}: expected one Sentinel-1 relative orbit and flight direction')
        return int(labels[0]), directions.pop()


def ordered_tracks(paths):
    tracks = sorted((track, path, direction) for path in paths for track, direction in [relative_orbit(path)])
    if len({track for track, _, _ in tracks}) != len(tracks):
        raise ValueError('Multitrack collections require one input per distinct Sentinel-1 relative orbit')
    return tracks


def pair_rows(tracks, manifests):
    """Concatenate tracks in orbit order with globally unique observation identities."""
    rows = []
    for i, ((track, _, direction), manifest) in enumerate(zip(tracks, manifests, strict=True)):
        namespace = f'S1:T{track:03d}'
        for p in manifest['pairs']:
            row = dict(p, track=track, direction=direction, file=i, native_pair_id=p['pair_id'],
                       pair_id=f"{namespace}:{p['pair_id']}", noise_group=f"{namespace}:{p['year']}")
            for endpoint, spacecraft in [('early', 'earlier'), ('late', 'later')]:
                seconds = EPOCH_SECONDS + 86400*p[f'calendar_{endpoint}_day']
                iso = datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec='microseconds')
                row[f'{endpoint}_acquisition'] = f"{namespace}:{p[f'{spacecraft}_spacecraft']}:{iso}"
            rows.append(row)
    return rows


def overlap_axis(axes, direction=1):
    """Return exact common cell centers and corresponding input slices."""
    axes = [np.asarray(axis, dtype=float) for axis in axes]
    if any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() for a in axes):
        raise ValueError('Each input needs finite regular x/y coordinates')
    step = axes[0][1] - axes[0][0]
    tolerance = abs(step)*1e-8
    if direction*step <= 0 or any(not np.allclose(np.diff(a), step, rtol=0, atol=tolerance) for a in axes):
        raise ValueError('Multitrack grids need the same regular map posting and axis order')
    offsets = np.array([(a[0]-axes[0][0])/step for a in axes])
    if not np.allclose(offsets, np.rint(offsets), rtol=0, atol=1e-8):
        raise ValueError('Multitrack grids must share the same lattice without resampling')
    offsets = np.rint(offsets).astype(int)
    begin = int(offsets.max())
    end = min(offset+len(a) for offset, a in zip(offsets, axes))
    if end <= begin:
        raise ValueError('Multitrack grids have no common extent')
    sections = [slice(begin-i, end-i) for i in offsets]
    common = axes[0][sections[0]]
    if any(not np.allclose(a[s], common, rtol=0, atol=tolerance) for a, s in zip(axes, sections)):
        raise ValueError('Multitrack cell centers differ inside the overlap')
    return common, sections, abs(step)


@dataclass
class Overlap:
    x: np.ndarray
    y: np.ndarray
    crs: dict
    row: np.ndarray
    col: np.ndarray
    back: np.ndarray
    native_index: np.ndarray
    native_id: np.ndarray
    chip_separation_m: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray


def common_grid(handles):
    target = CRS.from_epsg(3031)
    for h in handles:
        if not CRS.from_wkt(h.attrs['crs_wkt']).equals(target, ignore_axis_order=True):
            raise ValueError('Multitrack grids must use EPSG:3031')
    x, cols, dx = overlap_axis([h['x'][:] for h in handles])
    y, rows, dy = overlap_axis([h['y'][:] for h in handles], -1)
    indexes = np.stack([h['map_source_index'][r, c] for h, r, c in zip(handles, rows, cols)], axis=-1)
    supported = (indexes >= 0).all(-1)
    if not supported.any():
        raise ValueError('No map cell is supported by every track')
    native, first, inverse = np.unique(indexes[supported], axis=0, return_index=True, return_inverse=True)
    row, col = np.unravel_index(np.flatnonzero(supported)[first], supported.shape)
    # Spatial ordering keeps consecutive batches near the same native chunks on both tracks.
    ordering = np.lexsort((col, row, col//32, row//32))
    ordered_inverse = np.empty_like(ordering)
    ordered_inverse[ordering] = np.arange(len(ordering))
    back = np.full(supported.shape, -1, dtype=np.int32)
    back[supported] = ordered_inverse[inverse]
    native, row, col = native[ordering], row[ordering], col[ordering]
    ids = np.stack([h['source_id'][:][native[:, i]] for i, h in enumerate(handles)], -1)
    lon = np.stack([h['longitude'][:][native[:, i]] for i, h in enumerate(handles)], -1)
    lat = np.stack([h['latitude'][:][native[:, i]] for i, h in enumerate(handles)], -1)
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError('An overlapping native source has missing coordinates')
    geod = Geod(ellps='WGS84')
    separation = np.maximum.reduce([geod.inv(lon[:, i], lat[:, i], lon[:, j], lat[:, j])[2]
                                    for i, j in combinations(range(len(handles)), 2)])
    crs = {**dict(handles[0]['crs'].attrs), 'GeoTransform': f'{x[0]-dx/2} {dx} 0 {y[0]+dy/2} 0 {-dy}'}
    return Overlap(x, y, crs, row, col, back, native, ids, separation, lon, lat)


class JointLayout(Layout):
    def __init__(self, years, rows):
        super().__init__(years, order_years=[])
        active = sorted({(p['track'], p['year']) for p in rows if p['order']})
        self.terms = tuple((f'order_T{track:03d}_{year}', track, year) for track, year in active)

    @property
    def joint_names(self):
        return (tuple(f'{axis}_velocity_{y}' for y in self.years for axis in ('east', 'north'))
                + tuple(term[0] for term in self.terms) + VERTICAL_NAMES + TARGET_NAMES)

    def order_columns(self, field):
        for name, track, year in self.terms:
            take = (field['track'] == track) & (field['year_index'] == self.years.index(year))
            yield name, field['order']*take

    def record(self):
        return super().record() | {'track_year_order': {name: {'track': track, 'year': year}
                                                       for name, track, year in self.terms}}


def covariances_for(rows, records):
    by_group = {f'S1:T{int(track):03d}:{year}': value for track, record in records.items()
                for year, value in record['covariance_ra_m2_by_year'].items()}
    missing = {p['noise_group'] for p in rows} - by_group.keys()
    if missing:
        raise ValueError(f'Missing track/year covariance: {", ".join(sorted(missing))}')
    return np.array([check_covariance(by_group[key], key) for key in [p['noise_group'] for p in rows]])


# The dependent target cache is virtual. Read only rows needed by this fit batch.
class Track:
    def __init__(self, stack, entry, rows, layout, covariance):
        self.number = entry['track']
        self.rows = rows
        self.obs = stack.enter_context(h5py.File(entry['observations'], 'r', rdcc_nbytes=32*1024**2))
        self.body = (stack.enter_context(h5py.File(entry['solid_earth'], 'r', rdcc_nbytes=32*1024**2))
                     if entry['solid_earth'] is not None else None)
        local = Layout.from_observations(self.obs)
        self.layout = Layout(layout.years, local.order_years, local.order_spacecraft)
        self.year_index = np.asarray([layout.years.index(p['year']) for p in rows])
        self.covariance = covariance
        self.owner = self.obs['owner'][:]
        self.order = self.obs['order'][:].astype(float)
        self.early_calendar = self.obs['calendar_early'][:]
        self.late_calendar = self.obs['calendar_late'][:]
        self.within_year = self.obs['within_year'][:]
        self.geometry = self.obs['geometry_by_file'][0]
        self.mixed = self.obs['known_mixed_source'][:]
        # Each owning burst has a single exact endpoint pair per observation.
        owners, indexes = np.unique(self.owner, return_index=True)
        take = np.sort(indexes)
        early, late = (self.obs[k][take] for k in ('early', 'late'))
        self.owner_index = np.zeros(owners.max()+1, dtype=int)
        self.owner_index[self.owner[take]] = np.arange(len(take))
        self.basis = {'overlap': np.nan_to_num(layout.overlaps(early, late)),
                      'vertical': vertical_columns(early, late), 'msf': np.nan_to_num(harmonic(early, late))}
        self.endpoints = {'early': early, 'late': late}
        self.ids = np.asarray([p['pair_id'] for p in rows])

    def field(self, index, remove_mixed):
        unique, inverse = np.unique(index, return_inverse=True)
        burst = self.owner_index[self.owner[index]]
        data = self.obs['data'][unique]
        if self.body is not None:
            data -= self.body['set_ra_m'][unique]
        valid = self.obs['valid_including_mixed'][unique] & np.isfinite(data).all(-1)
        if remove_mixed:
            valid &= ~self.mixed[unique, None]
        return {**{k: v[burst] for k, v in self.endpoints.items()}, 'data': data[inverse],
                'valid': valid[inverse], 'peak': self.obs['peak'][unique][inverse],
                'order': self.order, 'calendar_early': self.early_calendar, 'calendar_late': self.late_calendar,
                'within_year': self.within_year, 'file_index': np.zeros(len(self.rows), int),
                'geometry_by_file': self.geometry[index][None], 'year_index': self.year_index,
                'track': np.full(len(self.rows), self.number), 'pair_id': self.ids,
                'temporal_basis': {k: v[burst] for k, v in self.basis.items()}}


def joint_field(parts):
    return ({k: np.concatenate([f[k] for f in parts], axis=1) for k in ('early', 'late', 'data', 'peak', 'valid')}
            | {k: np.concatenate([f[k] for f in parts]) for k in
               ('order', 'year_index', 'calendar_early', 'calendar_late', 'within_year', 'track', 'pair_id')}
            | {'file_index': np.concatenate([np.full(f['early'].shape[1], i) for i, f in enumerate(parts)]),
               'geometry_by_file': np.concatenate([f['geometry_by_file'] for f in parts]),
               'temporal_basis': {k: np.concatenate([f['temporal_basis'][k] for f in parts], axis=1)
                                  for k in parts[0]['temporal_basis']}})


def screen_tracks(fields, tracks, options):
    """Combine order-removed native backgrounds in a covariance-weighted EN solve."""
    masks, normal, rhs = [], 0., 0.
    for field, track in zip(fields, tracks, strict=True):
        good, _, velocity, _, _ = screen(field, track.layout, options)
        masks.append(good)
        geometry = field['geometry_by_file'][0, :, :, :2]
        supported = np.isfinite(geometry).all(axis=(-1, -2))
        geometry = np.where(supported[:, None, None], geometry, 0.)
        weights = np.linalg.inv(track.covariance)
        n = np.einsum('sci,tcd,sdj->stij', geometry, weights, geometry)
        for_year = (track.year_index[None] == np.arange(len(track.layout.years))[:, None])
        use = good & field['within_year'][None]
        part = np.einsum('kt,st,stij->ksij', for_year, use, n)
        normal = normal + part
        rhs = rhs + np.einsum('ksij,ksj->ksi', part, np.nan_to_num(velocity))
    velocity = np.einsum('ksij,ksj->ksi', np.linalg.pinv(normal, hermitian=True), rhs)
    available = np.linalg.det(normal) > 0
    velocity = np.where(available[..., None], velocity, np.nan)
    mean = np.sum(np.nan_to_num(velocity), axis=0)/np.maximum(available.sum(axis=0), 1)[:, None]
    speed = np.linalg.norm(mean, axis=-1)
    # Keep a conventional direction for quiescent controls and expose their speed.
    axis = np.divide(mean, speed[:, None], out=np.tile([1., 0.], (len(mean), 1)), where=speed[:, None] > 1e-8)
    return masks, axis, velocity, speed


def solver_options(config):
    s = config['screening']
    return dict(cluster_days=config['uncertainty']['cluster_days'], robust_cutoff=s['robust_cutoff'],
                robust='none' if s['robust'] == 'auto' else s['robust'], robust_scale=s['robust_scale'], batched=True)


def fit_batch(tracks, native, layout, config):
    screen_options, cov_options = config['screening'], config['coverage']
    fields = [track.field(native[:, i], screen_options['remove_mixed_windows']) for i, track in enumerate(tracks)]
    masks, axis, velocity, speed = screen_tracks(fields, tracks, screen_options)
    field = joint_field(fields)
    good = np.concatenate(masks, axis=1)
    totals = coverage(field, layout, good, cov_options)
    track_support = [coverage(f, t.layout, g, cov_options) for f, t, g in zip(fields, tracks, masks)]
    enough_by_track = np.stack([(s['counts'] >= s['min_pairs_by_year']).all(1) for s in track_support], axis=1)
    eligible = totals['fit_support'] & enough_by_track.all(1)
    covariance = np.concatenate([t.covariance for t in tracks])
    options = solver_options(config)
    groups = [field['track'] == t.number for t in tracks]
    joint = solve(field, layout, good & eligible[:, None], axis, {'by_pair': covariance},
                  annual_uncertainty=True, information_groups=groups, **options)
    valid = joint['estimable'] & eligible
    residual = field['data'] - predict(field, layout, joint['coefficients'], joint['names'], axis)
    rms = []
    comparators = {}
    offset = 0
    for f, t, g, support in zip(fields, tracks, masks, track_support):
        section = slice(offset, offset+g.shape[1])
        r = residual[:, section]
        value = np.sum(np.where(g[..., None], r**2, 0.), axis=1)
        rms.append(np.sqrt(value/np.maximum(g.sum(1), 1)[:, None]))
        offset = section.stop
        direct = solve(f, t.layout, g & support['fit_support'][:, None], axis,
                       {'by_pair': t.covariance}, 'azimuth_direct', **options)
        comparators[t.number] = mask_fit(direct, direct['estimable'] & support['fit_support'])
    projections = np.stack([np.sum(f['geometry_by_file'][0, :, 1, :2]*axis, axis=-1) for f in fields], axis=1)
    values = {'flow_axis_EN': axis, 'mean_flow_speed': speed, 'year_direction_change_deg': _angles(velocity),
              'clean_pairs_by_track_year': np.stack([s['counts'] for s in track_support], axis=1),
              'clean_pairs_by_year': totals['counts'], 'phase_bins': totals['phase_bins'], 'span_days': totals['span_days'],
              'fit_support': eligible, 'standard_coverage': totals['standard_coverage'] & eligible,
              'azimuth_flow_projection_by_track': projections, 'rms_by_track_m': np.where(
                   valid[:, None, None], np.stack(rms, axis=1), np.nan)}
    return values, mask_fit(joint, valid), comparators


def mask_fit(result, valid):
    result['estimable'] = valid
    for key in ('coefficients', 'cs', 'amplitude', 'phase_deg', 'amplitude_sigma', 'phase_sigma_deg',
                'coefficient_covariance', 'rms_m', 'annual_en_covariance', 'target_min_information',
                'target_information', 'target_condition', 'target_min_information_by_group'):
        if key in result:
            value = result[key]
            result[key] = np.where(valid.reshape((-1,)+(1,)*(value.ndim-1)), value, np.nan)
    return result

"""Synthetic self-contained displacement packages for the version-2 workflow tests.

Packages follow the displacement-v1 schema used by the real stacks: exact
per-burst endpoints, range/azimuth metres, ENU sensitivities, copied map cells,
ocean-tide provenance/availability and internal content hashes. Displacements
are generated from known annual secular velocities (integrated over exact
year overlaps), spacecraft-order offsets, residual vertical M2/O1 and
along-flow Msf, plus optional correlated range/azimuth noise.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import h5netcdf
import h5py
import numpy as np
from pyproj import CRS, Transformer

from glacier_offsets.velocity_basis import EPOCH_SECONDS, harmonic
from glacier_offsets.velocity_io import variable_hashes
from glacier_offsets.velocity_pixel_general import Layout
from glacier_offsets.velocity_vertical import vertical_columns


GRID = {'ny': 4, 'nx': 6, 'dx': 200., 'x0': -2135000., 'y0': 990000.}
BURST_OFFSET_S = (0., 2.75)


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def acquisitions(start, count, step=6):
    """Alternating S1A/S1B acquisitions at the fixed hour of the real Larsen stack."""
    first = utc(start)+timedelta(hours=8, minutes=2, seconds=11)
    return [(first+timedelta(days=step*i), 'S1A' if i % 2 == 0 else 'S1B') for i in range(count)]


class Truth:
    """Known model, one unique source per two map cells (copied cells share sources)."""

    def __init__(self, years, seed=4, controls=()):
        rng = np.random.default_rng(seed)
        self.ny, self.nx = GRID['ny'], GRID['nx']
        self.ns = self.ny*self.nx//2
        self.years = list(years)
        self.velocity = {y: rng.normal(0, .05, (self.ns, 2))+[.6+.1*k, .35-.05*k] for k, y in enumerate(years)}
        self.order = {y: -.3+.1*k for k, y in enumerate(years)}
        self.vertical = rng.normal(0, .05, (self.ns, 4))
        self.msf = rng.normal(0, .01, (self.ns, 2))+[.05, .03]
        for s in controls:
            for y in years:
                self.velocity[y][s] = 0.
            self.vertical[s] = 0.
            self.msf[s] = 0.
        self.controls = list(controls)

    def geometry(self):
        """(radar, ENU, y, x) sensitivities and copied-cell associations."""
        association = (np.arange(self.ny*self.nx)//2).reshape(self.ny, self.nx)
        owner = 1+association % 2  # copied cells share their source's owning burst
        base = np.array([[[-.55, .12, -.83], [.11, .99, .003]]])
        ramp = np.linspace(0, .02, self.ns)[:, None, None]
        per_source = base+ramp
        sensitivity = per_source[association]  # y, x, radar, ENU
        return association, owner, np.moveaxis(sensitivity, (2, 3), (0, 1))

    def axis(self):
        mean = np.mean([self.velocity[y] for y in self.years], axis=0)
        speed = np.linalg.norm(mean, axis=1)[:, None]
        return np.divide(mean, speed, out=np.tile([1., 0.], (len(mean), 1)), where=speed > 1e-8)


def write_package(path, truth, pairs, *, noise=None, seed=0, perturb=0., grid_shift=0.,
                  mixed=None, tide_unavailable=None, horizontal=False, drop=(), extra_attrs=None,
                  correction_unavailable=None, dtype='f4'):
    """Write pairs [(early_dt, late_dt, sc1, sc2)] as one package and set its content hashes."""
    rng = np.random.default_rng(seed)
    association, owner_map, sensitivity = truth.geometry()
    sensitivity = sensitivity+perturb
    ny, nx, nt = truth.ny, truth.nx, len(pairs)
    x = GRID['x0']+GRID['dx']*np.arange(nx)+grid_shift
    y = GRID['y0']-GRID['dx']*np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    # Observation positions are the first copied cell of each source.
    first = np.array([np.argwhere(association == s)[0] for s in range(truth.ns)])
    lon, lat = Transformer.from_crs(3031, 4326, always_xy=True).transform(xx, yy)
    lon_s, lat_s = lon[first[:, 0], first[:, 1]], lat[first[:, 0], first[:, 1]]
    lon, lat = lon_s[association], lat_s[association]
    owner_s = owner_map[first[:, 0], first[:, 1]]-1
    layout = Layout(truth.years)
    geom = np.moveaxis(sensitivity[:, :, first[:, 0], first[:, 1]], 2, 0)  # source, radar, ENU
    seconds = lambda d: d.timestamp()
    scene = np.array([[seconds(e), seconds(l)] for e, l, _, _ in pairs])
    bursts = scene[:, None, :]+np.array(BURST_OFFSET_S)[None, :, None]  # time, burst, endpoint
    early = (bursts[:, owner_s, 0].T-EPOCH_SECONDS)/86400.
    late = (bursts[:, owner_s, 1].T-EPOCH_SECONDS)/86400.
    overlap = np.nan_to_num(layout.overlaps(early, late))
    order = np.array([int(b == 'S1B')-int(a == 'S1B') for _, _, a, b in pairs], float)
    label = [layout.years.index(datetime.fromtimestamp((s[0]+s[1])/2, tz=timezone.utc).year) for s in scene]
    data = np.zeros((truth.ns, nt, 2))
    for k, year in enumerate(truth.years):
        data += overlap[:, :, k, None]*np.einsum('sci,si->sc', geom[..., :2], truth.velocity[year])[:, None]
    for t, k in enumerate(label):
        data[:, t, 1] += order[t]*truth.order[truth.years[k]]
    data += np.einsum('stj,sj->st', vertical_columns(early, late), truth.vertical)[..., None]*geom[:, None, :, 2]
    projection = np.einsum('sci,si->sc', geom[..., :2], truth.axis())
    data += np.einsum('stj,sj->st', harmonic(early, late), truth.msf)[..., None]*projection[:, None]
    if noise is not None:
        data += rng.multivariate_normal([0, 0], noise, size=(truth.ns, nt))
    displacement = np.moveaxis(data[association], 2, 0)  # time, y, x, channel
    crs = CRS.from_epsg(3031)
    with h5netcdf.File(path, 'w') as h:
        h.dimensions = {'time': nt, 'y': ny, 'x': nx, 'burst': 2, 'enu': 3}
        h.attrs.update(schema_version='glacier-offsets-displacements-v1', package_status='complete', source='raw',
                       EPSG=np.array([3031]), geometry_identity='geometry-a', geometry_reference_date='2017-01-01',
                       layout_identity='layout-a', geocoding_resampling='nearest', tidal_correction_status='complete',
                       package_identity=f'synthetic-{Path(path).stem}', **(extra_attrs or {}))
        if horizontal:
            h.attrs['horizontal_vector_basis'] = 'local geographic east/north; physical metres'

        def var(name, dims, value, dtype=None, **attrs):
            if name in drop:
                return
            v = h.create_variable(name, dims, dtype=dtype or np.asarray(value).dtype)
            v[...] = np.asarray(value, dtype=object) if isinstance(value, str) else value
            v.attrs.update(attrs)
        var('x', ('x',), x, units='m')
        var('y', ('y',), y, units='m')
        v = h.create_variable('crs', (), dtype='i4')
        v.attrs.update(crs_wkt=crs.to_wkt(), spatial_ref=crs.to_wkt(), grid_mapping_name='polar_stereographic')
        time_units = {'units': 'seconds since 1970-01-01 00:00:00', 'calendar': 'proleptic_gregorian'}
        var('earlier_time', ('time',), scene[:, 0], **time_units)
        var('later_time', ('time',), scene[:, 1], **time_units)
        var('earlier_burst_time', ('time', 'burst'), bursts[..., 0], **time_units)
        var('later_burst_time', ('time', 'burst'), bursts[..., 1], **time_units)
        strings = h5py.string_dtype()
        var('pair_id', ('time',), np.array([f'{e:%Y-%m-%d}_{l:%Y-%m-%d}' for e, l, _, _ in pairs], dtype=object), strings)
        var('earlier_spacecraft', ('time',), np.array([a for _, _, a, _ in pairs], dtype=object), strings)
        var('later_spacecraft', ('time',), np.array([b for _, _, _, b in pairs], dtype=object), strings)
        var('burst_id', ('burst',), np.array(['t038_080975_iw1', 't038_080975_iw2'], dtype=object), strings)
        var('enu', ('enu',), np.array(['east', 'north', 'up'], dtype=object), strings)
        var('source_observation_index', ('y', 'x'), association.astype('i8'))
        var('source_burst_index', ('y', 'x'), owner_map.astype('i4'), units='1')
        var('observation_longitude', ('y', 'x'), lon, units='degrees_east')
        var('observation_latitude', ('y', 'x'), lat, units='degrees_north')
        var('reference_center_line', ('y', 'x'), association.astype(float)*10, units='1')
        var('reference_center_sample', ('y', 'x'), association.astype(float)*20, units='1')
        var('height_m', ('y', 'x'), np.full((ny, nx), 50.), units='m')
        var('tidal_response_factor', ('y', 'x'), np.full((ny, nx), .9, np.float32), units='1')
        var('range_sensitivity_enu', ('enu', 'y', 'x'), sensitivity[0], units='1')
        var('azimuth_sensitivity_enu', ('enu', 'y', 'x'), sensitivity[1], units='1')
        var('range_displacement_m', ('time', 'y', 'x'), displacement[..., 0].astype(dtype), units='m')
        var('azimuth_displacement_m', ('time', 'y', 'x'), displacement[..., 1].astype(dtype), units='m')
        var('snr', ('time', 'y', 'x'), np.full((nt, ny, nx), 12., np.float32), units='1')
        var('correlation_peak', ('time', 'y', 'x'), np.full((nt, ny, nx), .4, np.float32), units='1')
        ones = np.ones((nt, ny, nx), np.uint8)
        var('valid_mask', ('time', 'y', 'x'), ones, units='1')
        if horizontal:
            var('horizontal_valid_mask', ('time', 'y', 'x'), ones, units='1')
        tide = ones.copy()
        if tide_unavailable is not None:
            for t, s in tide_unavailable:
                tide[t][association == s] = 0
        var('tidal_correction_available', ('time', 'y', 'x'), tide, units='1')
        if mixed is not None:
            flag = np.zeros_like(ones)
            available = ones.copy()
            for t, s in mixed:
                flag[t][association == s] = 1
                available[t][association == s] = 0
            for t, s in correction_unavailable or ():
                available[t][association == s] = 0
            var('cross_swath_mixed', ('time', 'y', 'x'), flag, units='1')
            var('correction_available', ('time', 'y', 'x'), available, units='1')
        var('tidal_correction_provenance_json', (), json.dumps({'method': 'cats2008-vertical-range-v1', 'synthetic': True}), strings)
    with h5netcdf.File(path, 'r') as h:
        hashes = variable_hashes(h)
    with h5py.File(path, 'r+') as h:
        h.attrs['content_hashes_json'] = json.dumps(hashes)
    return path


def pair_list(start, count, step=6):
    acq = acquisitions(start, count+1, step)
    return [(acq[i][0], acq[i+1][0], acq[i][1], acq[i+1][1]) for i in range(count)]


def config_text(inputs, cache, output, noise, *, min_span=100, extra=''):
    lines = ['version: 2', 'inputs:']+[f'  - {p}' for p in inputs]
    lines += [f'cache_directory: {cache}', f'output: {output}', 'corrections:', '  ocean_tide: require_existing',
              '  solid_earth: false', 'noise:']+['  '+line for line in noise.strip().splitlines()]
    lines += ['coverage:', f'  min_span_days: {min_span}', '  min_pairs_per_year: 10', '  min_pairs_total: 20',
              '  min_pairs_standard: 20', 'execution:', '  batch_size: 5', '  preparation_chunk_pairs: 7']
    return '\n'.join(lines)+'\n'+extra

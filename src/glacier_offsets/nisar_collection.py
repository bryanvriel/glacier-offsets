"""Portable, aligned GOFF collections with pair-varying geometry and timing."""
from contextlib import ExitStack
import json
from pathlib import Path

import h5netcdf
import h5py
import numpy as np
from pyproj import CRS

from .displacement import is_nisar, text, validate_nisar
from .velocity_basis import EPOCH_SECONDS, utc_days
from .velocity_io import digest, revision


def frame_candidates(members, priority, read_valid, acquisition_pairs=None):
    """Shared per-location ownership; invalid low frames allow higher frames."""
    chosen, candidates = {}, {}
    for t in priority:
        r = members[t]
        if acquisition_pairs is not None and r['acquisition_pair'] not in acquisition_pairs:
            continue
        valid = read_valid(t)
        used = chosen.setdefault(r['acquisition_pair'], np.zeros(valid.shape, bool))
        candidates[t] = valid & ~used
        used |= valid
    return candidates


class NisarCollection:
    """GOFF map locations; repeated frames do not create extra measurements."""
    mission = 'NISAR'

    def __init__(self, paths, start=None, end=None, *, cache_directory=None, verified_inputs=None):
        self.paths = sorted([Path(p).resolve() for p in paths], key=str)
        self.start = -np.inf if start is None else float(utc_days(start))
        self.end = np.inf if end is None else float(utc_days(end))
        self.stack = ExitStack()
        self.cache_directory = cache_directory
        self.verified_inputs = {str(Path(i['path']).resolve()): i for i in (verified_inputs or [])}

    def __enter__(self):
        try:
            if len(set(self.paths)) != len(self.paths): raise ValueError('Duplicate package paths')
            self.handles = [self.stack.enter_context(h5netcdf.File(p)) for p in self.paths]
            self.raw_handles = [self.stack.enter_context(h5py.File(p)) for p in self.paths]
            self.datasets = {}
            self.sources = []; self.members = []; self.pairs = []; ids = set(); signatures = []
            self.revisions = [revision(p) for p in self.paths]
            for fi, (path, h) in enumerate(zip(self.paths, self.handles)):
                if not is_nisar(h): raise ValueError('Joint NISAR/Sentinel-1 fitting is not supported')
                if h.attrs.get('source', 'raw') != 'raw': raise ValueError('Sentinel-1 swath corrections are not supported on NISAR packages')
                if self.cache_directory is None:
                    validate_nisar(path)
                    sha = digest(path)
                else:
                    from .nisar_validation import verified_nisar
                    sha = verified_nisar(path, self.cache_directory, self.verified_inputs.get(str(path)))
                if 'tidal_correction_available' not in h or 'tidal_correction_provenance_json' not in h:
                    raise ValueError('Ocean-tide correction provenance/availability is required')
                if any('solid_earth' in k.lower() or 'body_tide' in k.lower() for k in (*h.attrs, *h.variables)):
                    raise ValueError('Solid-Earth correction must be applied exactly once at fitting')
                spec = json.loads(text(h['provenance_json'][()]))
                signatures.append((spec['frequency'], spec['polarization'], spec['layer'], tuple(spec['release'])))
                self.sources.append({'path': str(path), 'sha256': sha, 'package_identity': h.attrs['package_identity'],
                                     'correction_history': [text(h[k][()]) for k in h.variables if k.endswith('provenance_json') and k != 'provenance_json']})
                for slot, pair in enumerate(h['pair_id'][:].astype(str)):
                    if pair in ids: raise ValueError(f'Duplicate pair ID: {pair}')
                    ids.add(pair)
                    e, l = (float(h[k][slot]) for k in ('earlier_time', 'later_time'))
                    if (l-EPOCH_SECONDS)/86400 < self.start or (e-EPOCH_SECONDS)/86400 >= self.end:
                        continue
                    track, frame = int(h['track_number'][slot]), int(h['frame_number'][slot])
                    ref, sec = int(h['reference_absolute_orbit'][slot]), int(h['secondary_absolute_orbit'][slot])
                    year = int(np.datetime64(int(((e+l)/2)*1e6), 'us').astype('datetime64[Y]').astype(int))+1970
                    self.members.append({'pair_id': pair, 'file': fi, 'slot': slot, 'track': track, 'frame': frame,
                                         'acquisition_pair': f'T{track:03d}:{ref}:{sec}', 'reference_acquisition': f'NISAR:{ref}',
                                         'secondary_acquisition': f'NISAR:{sec}', 'earlier': e, 'later': l, 'year': year,
                                         'noise_group': f'T{track:03d}:{year}'})
            if any(s != signatures[0] for s in signatures):
                raise ValueError('NISAR collections must share layer, frequency, polarization and processing release')
            self.members.sort(key=lambda r: (r['earlier'], r['later'], r['pair_id']))
            if not self.members: raise ValueError('No NISAR pair overlaps the selected period')
            keys = [(r['acquisition_pair'], r['frame']) for r in self.members]
            if len(set(keys)) != len(keys): raise ValueError('Conflicting reprocessed copies of the same acquisition pair/frame')
            self.priority = sorted(range(len(self.members)), key=lambda i: (self.members[i]['frame'], self.members[i]['pair_id']))
            self.x = np.arange(min(h['x'][0] for h in self.handles), max(h['x'][-1] for h in self.handles)+40, 80.)
            self.y = np.arange(max(h['y'][0] for h in self.handles), min(h['y'][-1] for h in self.handles)-40, -80.)
            self.shape = (len(self.y), len(self.x)); self.enu = ['east', 'north', 'up']
            self.offsets = [(int(round((self.y[0]-h['y'][0])/80)), int(round((h['x'][0]-self.x[0])/80))) for h in self.handles]
            self.association = np.arange(np.prod(self.shape), dtype='i8').reshape(self.shape)
            self.owner = np.zeros(self.shape, int)
            self.pairs = [(r['file'], r['slot'], r['pair_id'], np.array([(r['earlier']-EPOCH_SECONDS)/86400]),
                           np.array([(r['later']-EPOCH_SECONDS)/86400]), np.array([True])) for r in self.members]
            self.selected_span_days = (max(r['later'] for r in self.members)-min(r['earlier'] for r in self.members))/86400
            return self
        except BaseException:
            self.stack.close(); raise

    def __exit__(self, *args): self.stack.close()

    def check_revisions(self):
        if self.revisions != [revision(p) for p in self.paths]: raise ValueError('A source netCDF changed during fitting')

    def sections(self, size=64):
        for r in range(0, self.shape[0], size):
            for c in range(0, self.shape[1], size): yield slice(r, min(r+size, self.shape[0])), slice(c, min(c+size, self.shape[1]))

    def read(self, member, name, section, *, vector=False):
        h = self.handles[member['file']]; off = self.offsets[member['file']]
        shape = tuple(s.stop-s.start for s in section)
        result = np.full(((3,)+shape if vector else shape), np.nan)
        dr0, dc0 = max(section[0].start, off[0]), max(section[1].start, off[1])
        dr1, dc1 = min(section[0].stop, off[0]+len(h.dimensions['y'])), min(section[1].stop, off[1]+len(h.dimensions['x']))
        if dr0 >= dr1 or dc0 >= dc1: return result
        source = (slice(dr0-off[0], dr1-off[0]), slice(dc0-off[1], dc1-off[1]))
        dest = (slice(dr0-section[0].start, dr1-section[0].start), slice(dc0-section[1].start, dc1-section[1].start))
        key = (member['file'], name)
        if key not in self.datasets:
            self.datasets[key] = (self.raw_handles[member['file']][name], h[name].dimensions[0] == 'time')
        ds, temporal = self.datasets[key]
        prefix = (member['slot'], slice(None)) if vector else (member['slot'],) if temporal else ()
        result[(slice(None), *dest) if vector else dest] = ds[(*prefix, *source)]
        return result

    def tile(self, section, indices=None, *, radar_only=False):
        selected = list(range(len(self.members))) if indices is None else list(indices)
        acquisition_pairs = {self.members[t]['acquisition_pair'] for t in selected}
        keys = ('early', 'late', 'data_ra', 'valid', 'snr', 'peak', 'geometry', 'overlap_suppressed')
        out = {k: [] for k in (keys if radar_only else (*keys, 'data', 'inverse', 'tidal_range_bias'))}
        candidates = frame_candidates(self.members, self.priority,
                                      lambda t: self.read(self.members[t], 'valid_mask', section) == 1,
                                      acquisition_pairs)
        for t in selected:
            r = self.members[t]
            e, l = (self.read(r, k, section) for k in ('earlier_observation_time', 'later_observation_time'))
            e, l = (e-EPOCH_SECONDS)/86400, (l-EPOCH_SECONDS)/86400
            geom = np.stack([np.moveaxis(self.read(r, k, section, vector=True), 0, -1) for k in ('range_sensitivity_enu', 'azimuth_sensitivity_enu')], -2)
            finite = np.isfinite(geom[..., :2]).all(axis=(-1, -2))
            ra = np.stack([self.read(r, k, section) for k in ('range_displacement_m', 'azimuth_displacement_m')], -1)
            if not radar_only or 'horizontal_valid_mask' not in self.handles[r['file']]:
                inv = np.linalg.pinv(np.where(finite[..., None, None], geom[..., :2], np.eye(2)))
                data = np.einsum('...ij,...j->...i', inv, ra)
            good = candidates[t] & finite & np.isfinite(e+l) & (l > e) & (e >= self.start) & (l < self.end)
            good &= self.read(r, 'tidal_correction_available', section) == 1
            h = self.handles[r['file']]
            if 'horizontal_valid_mask' in h: good &= self.read(r, 'horizontal_valid_mask', section) == 1
            good &= np.isfinite(ra if radar_only and 'horizontal_valid_mask' in h else data).all(-1)
            for k, v in [('early', e), ('late', l), ('data_ra', ra), ('valid', good),
                         ('snr', self.read(r, 'snr', section)), ('peak', self.read(r, 'correlation_peak', section)),
                         ('geometry', geom),
                         ('overlap_suppressed', (self.read(r, 'valid_mask', section) == 1) & ~candidates[t])]: out[k].append(v)
            if not radar_only:
                for k, v in [('data', data), ('inverse', inv), ('tidal_range_bias', self.read(r, 'tidal_range_bias_m', section))]:
                    out[k].append(v)
        return {k: np.array(v) for k, v in out.items()}

    def observation_position(self, row, col): return float(self.x[col]), float(self.y[row])
    def spatial_block(self, row, col): return f'{int(np.floor(self.x[col]/3000))}:{int(np.floor(self.y[row]/3000))}'

    def point(self, row, col):
        a = self.tile((slice(row, row+1), slice(col, col+1)))
        out = {k: v[:, 0, 0] for k, v in a.items()}
        out.update(row=row, col=col, source_id=int(self.association[row, col]), mission='NISAR',
                   spatial_block=self.spatial_block(row, col),
                   pair_id=np.array([r['pair_id'] for r in self.members]), track=np.array([r['track'] for r in self.members]),
                   reference_acquisition=np.array([r['reference_acquisition'] for r in self.members]),
                   secondary_acquisition=np.array([r['secondary_acquisition'] for r in self.members]),
                   noise_group=np.array([r['noise_group'] for r in self.members]))
        out['tidal_range_bias'] = out.pop('tidal_range_bias')
        return out


def collect_nisar(paths, cache_directory):
    from .velocity_collection import calendar_year
    with NisarCollection(paths, cache_directory=cache_directory) as c:
        pairs = []
        for r in c.members:
            e, l = ((r[k]-EPOCH_SECONDS)/86400 for k in ('earlier', 'later'))
            pairs.append(r | {'calendar_early_day': e, 'calendar_late_day': l, 'calendar_midpoint_day': (e+l)/2,
                             'within_year': calendar_year(e) == calendar_year(l), 'order': 0,
                             'earlier_spacecraft': 'NISAR', 'later_spacecraft': 'NISAR'})
        years = sorted({r['year'] for r in pairs})
        grid = {'EPSG': 3031, 'crs_wkt': CRS.from_epsg(3031).to_wkt(), 'shape': list(c.shape),
                'transform': [float(c.x[0]-40), 80., 0., float(c.y[0]+40), 0., -80.]}
        return {'schema': 'glacier-offsets-nisar-collection-v2', 'mission': 'NISAR', 'grid': grid,
                'inputs': [s | {'revision': list(revision(s['path'])), 'validity_mask': 'valid_mask', 'source': 'raw',
                               'pairs': sum(r['file'] == i for r in pairs)} for i, s in enumerate(c.sources)],
                'pairs': pairs, 'years': years, 'order_spacecraft': {},
                'pairs_by_year': {str(y): len({r['acquisition_pair'] for r in pairs if r['year'] == y}) for y in years},
                'cross_year_pairs': [r['pair_id'] for r in pairs if not r['within_year']], 'span_days': c.selected_span_days,
                'overlap_policy': 'lowest valid frame number then granule ID per acquisition pair and map location',
                'overlap_groups': {k: [r['pair_id'] for r in pairs if r['acquisition_pair'] == k]
                                   for k in sorted({r['acquisition_pair'] for r in pairs}) if sum(r['acquisition_pair'] == k for r in pairs) > 1}}

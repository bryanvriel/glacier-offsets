"""Standalone netCDF input, source fingerprints and atomic fit products."""
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5netcdf
import h5py
import numpy as np
from .velocity_basis import EPOCH_SECONDS, utc_days, evaluate

SCHEMA = 'glacier-offsets-velocity-fit-v1'
STATIC = ('x', 'y', 'burst_id', 'enu', 'source_burst_index', 'source_observation_index',
          'range_sensitivity_enu', 'azimuth_sensitivity_enu', 'reference_center_line', 'reference_center_sample')


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(serializable(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix='.'+path.name)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(serializable(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_json_variable(h, name):
    value = h[name][...]
    def parse(v):
        return json.loads(v.decode() if isinstance(v, bytes) else v)
    if value.ndim:
        return [parse(v) for v in value.ravel()]
    return parse(value.item())


def revision(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def variable_hashes(h):
    """Reproduce displacement-package hashes without importing native modules."""
    result = {}
    for name, variable in h.variables.items():
        sha = hashlib.sha256()
        if variable.dimensions[-2:] == ('y', 'x'):
            for index in np.ndindex(variable.shape[:-2]):
                for row in range(0, variable.shape[-2], 256):
                    for col in range(0, variable.shape[-1], 256):
                        sha.update(np.asarray(variable[(*index, slice(row,row+256), slice(col,col+256))]).tobytes())
        elif variable.dtype.kind in 'OSU':
            values = np.asarray(variable[...]).ravel()
            sha.update(json.dumps([s.decode() if isinstance(s,bytes) else str(s) for s in values]).encode())
        else:
            sha.update(np.asarray(variable[...]).tobytes())
        result[name] = sha.hexdigest()
    return result


def verify_hashes(h):
    if json.loads(h.attrs.get('content_hashes_json', '{}')) != variable_hashes(h):
        raise ValueError('Input netCDF content checksum mismatch')


@contextmanager
def writer_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory/'.velocity.lock', 'a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another velocity fitter is using this output directory') from exc
        yield


class Collection:
    """Read only netCDFs; pair IDs, not possibly duplicate midpoints, are keys."""
    def __init__(self, paths, start=None, end=None):
        self.paths = [Path(p).resolve() for p in paths]
        self.start = -np.inf if start is None else float(utc_days(start))
        self.end = np.inf if end is None else float(utc_days(end))
        self.stack = ExitStack()

    def __enter__(self):
        try:
            self.handles = [self.stack.enter_context(h5netcdf.File(p, 'r')) for p in self.paths]
            self.revisions = [revision(p) for p in self.paths]
            self.sources = []
            self.pairs = []
            ids = set()
            base = self.handles[0]
            for path, h in zip(self.paths, self.handles):
                verify_hashes(h)
                if h.attrs.get('schema_version') != 'glacier-offsets-displacements-v1' or h.attrs.get('package_status') != 'complete':
                    raise ValueError(f'{path}: incomplete or unsupported displacement package')
                required = (*STATIC, 'earlier_burst_time', 'later_burst_time', 'pair_id', 'horizontal_valid_mask',
                            'east_displacement_m', 'north_displacement_m', 'snr', 'correlation_peak')
                if any(k not in h.variables for k in required):
                    raise ValueError(f'{path}: a self-contained horizontal package is required')
                if h.attrs.get('horizontal_vector_basis') != 'local geographic east/north; physical metres':
                    raise ValueError('Horizontal components must preserve local geographic EN')
                for attr in ('EPSG', 'geometry_identity', 'geometry_reference_date', 'layout_identity', 'geocoding_resampling'):
                    if attr not in h.attrs or h.attrs[attr] != base.attrs[attr]:
                        raise ValueError(f'Incompatible reference geometry/grid: {attr}')
                if h.attrs['geocoding_resampling'] != 'nearest':
                    raise ValueError('Fitting requires nearest-source observations')
                if h.attrs.get('source')=='swath_corrected' and 'correction_available' not in h.variables:
                    raise ValueError('Swath-corrected input lacks its correction availability mask')
                for name in STATIC:
                    if h[name].dimensions != base[name].dimensions:
                        raise ValueError(f'Incompatible source associations: {name}')
                    left, right = h[name][:], base[name][:]
                    equal = np.array_equal(left, right, equal_nan=True) if left.dtype.kind in 'fc' else np.array_equal(left, right)
                    if not equal:
                        raise ValueError(f'Incompatible grid or source associations: {name}')
                for name in ('east_displacement_m', 'north_displacement_m'):
                    if h[name].dimensions != ('time', 'y', 'x') or h[name].attrs.get('units') != 'm':
                        raise ValueError(f'{name} must be (time,y,x) displacement metres')
                for name in ('earlier_burst_time', 'later_burst_time'):
                    if h[name].dimensions != ('time', 'burst') or h[name].attrs.get('units') != 'seconds since 1970-01-01 00:00:00':
                        raise ValueError('Exact burst endpoints must be UTC epoch seconds')
                    if h[name].attrs.get('calendar') not in ('proleptic_gregorian', 'standard', 'gregorian'):
                        raise ValueError('Unsupported burst time calendar')
                early, late = [(h[k][:]-EPOCH_SECONDS)/86400 for k in ('earlier_burst_time', 'later_burst_time')]
                for slot, pair in enumerate(h['pair_id'][:].astype(str)):
                    if pair in ids:
                        raise ValueError(f'Duplicate pair ID: {pair}')
                    ids.add(pair)
                    available = np.isfinite(early[slot]+late[slot]) & (late[slot] > early[slot])
                    # Retain missing bursts as unavailable; never substitute scene times.
                    available &= (early[slot] >= self.start) & (late[slot] < self.end)
                    self.pairs.append((len(self.sources), slot, pair, early[slot], late[slot], available))
                corrections = {k: read_json_variable(h, k) for k in
                    ('calibration_json', 'correction_provenance_json', 'tidal_correction_provenance_json',
                     'horizontal_conversion_provenance_json') if k in h.variables}
                self.sources.append({'path': str(path), 'sha256': digest(path), 'package_identity': str(h.attrs.get('package_identity', '')),
                                     'correction_history': corrections})
            self.pairs.sort(key=lambda p: (np.min(p[3][np.isfinite(p[3])]) if np.isfinite(p[3]).any() else np.inf,p[2]))
            self.x, self.y = base['x'][:], base['y'][:]
            self.owner = base['source_burst_index'][:].astype(int)
            self.association = base['source_observation_index'][:].astype('int64')
            self.shape = self.association.shape
            supported = self.association>=0
            ids,first,inverse_ids = np.unique(self.association[supported],return_index=True,return_inverse=True)
            for name in ('source_burst_index','reference_center_line','reference_center_sample',
                         'range_sensitivity_enu','azimuth_sensitivity_enu'):
                array = base[name][:]
                values = array[...,supported]
                expected = values[...,first][...,inverse_ids]
                if not np.array_equal(values,expected,equal_nan=True):
                    raise ValueError('Copied source observations have inconsistent '+name)
            self.enu = base['enu'][:].astype(str).tolist()
            self.early = np.array([p[3] for p in self.pairs])
            self.late = np.array([p[4] for p in self.pairs])
            selection = np.array([p[5] for p in self.pairs])
            if not selection.any():
                raise ValueError('No exact burst intervals fall within the requested period')
            self.selected_span_days = float(self.late[selection].max()-self.early[selection].min())
            self.selected_early = self.early[selection]
            self.selected_late = self.late[selection]
            self.check_revisions()
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        self.stack.close()

    def check_revisions(self):
        if self.revisions != [revision(p) for p in self.paths]:
            raise ValueError('A source netCDF changed during fitting')

    def sections(self, size=64):
        for row in range(0, self.shape[0], size):
            for col in range(0, self.shape[1], size):
                yield (slice(row, min(row+size, self.shape[0])), slice(col, min(col+size, self.shape[1])))

    def tile(self, section):
        owner = self.owner[section]-1
        valid_owner = (owner >= 0) & (owner < self.early.shape[1]) & (self.association[section]>=0)
        owner = np.clip(owner, 0, self.early.shape[1]-1)
        keys = ('east_displacement_m', 'north_displacement_m', 'horizontal_valid_mask', 'snr', 'correlation_peak')
        cached = [{key: h[key][(slice(None), *section)] for key in keys} for h in self.handles]
        correction = [h['correction_available'][(slice(None),*section)] if h.attrs.get('source')=='swath_corrected'
                      else np.ones_like(cached[i]['horizontal_valid_mask']) for i,h in enumerate(self.handles)]
        values = {key: np.array([cached[p[0]][key][p[1]] for p in self.pairs]) for key in keys}
        early, late = self.early[:, owner], self.late[:, owner]
        available = np.array([p[5] for p in self.pairs])[:, owner] & valid_owner[None]
        data = np.stack((values['east_displacement_m'], values['north_displacement_m']), -1).astype(float)
        valid = (values['horizontal_valid_mask']==1) & available & np.isfinite(data).all(-1)
        # Unsupported mixed-swath observations remain in the input for audit,
        # but are not treated as calibrated observations in temporal inference.
        valid &= np.array([correction[p[0]][p[1]] for p in self.pairs])==1
        data[~valid] = np.nan
        h = self.handles[0]
        axes = [self.enu.index(k) for k in ('east', 'north', 'up')]
        geometry = np.stack([np.moveaxis(h[k][(slice(None), *section)], 0, -1)[..., axes]
                             for k in ('range_sensitivity_enu', 'azimuth_sensitivity_enu')], -2)
        safe = np.where(np.isfinite(geometry[..., :2]).all(axis=(-1, -2))[..., None, None], geometry[..., :2], np.eye(2))
        inverse = np.linalg.pinv(safe)
        return {'early': early, 'late': late, 'data': data, 'valid': valid, 'snr': values['snr'],
                'peak': values['correlation_peak'], 'inverse': inverse, 'geometry': geometry}

    def observation_position(self,row,col):
        """Read physical source positions once, before any time-series sampling."""
        if not hasattr(self,'_observation_xy'):
            h=self.handles[0]
            if 'observation_longitude' in h.variables:
                from pyproj import Transformer
                transform=Transformer.from_crs(4326,int(h.attrs['EPSG']),always_xy=True)
                self._observation_xy=transform.transform(h['observation_longitude'][:],h['observation_latitude'][:])
            else:
                self._observation_xy=np.meshgrid(self.x,self.y)
        return tuple(float(v[row,col]) for v in self._observation_xy)

    def spatial_block(self,row,col):
        x,y=self.observation_position(row,col)
        return f'{int(np.floor(x/3000))}:{int(np.floor(y/3000))}'

    def point(self, row, col):
        tile = self.tile((slice(row, row+1), slice(col, col+1)))
        result = {k: v[:, 0, 0] for k, v in tile.items() if k not in ('inverse', 'geometry')}
        result.update(inverse=tile['inverse'][0, 0], geometry=tile['geometry'][0, 0],
                      row=int(row), col=int(col), source_id=int(self.association[row, col]))
        # Use physical observation positions, not copied map-cell positions, for blocks.
        h = self.handles[0]
        result['spatial_block'] = self.spatial_block(row,col)
        tide = []
        calibration = []
        for file_index, slot, _, _, _, _ in self.pairs:
            source = self.handles[file_index]
            if 'tidal_range_bias_m' in source.variables:
                tide.append(float(source['tidal_range_bias_m'][slot, row, col]))
            if 'calibration_json' in source.variables:
                record = read_json_variable(source, 'calibration_json')
                earlier = str(source['earlier_spacecraft'][slot].decode())
                later = str(source['later_spacecraft'][slot].decode())
                burst = int(self.owner[row, col])-1
                name = source['burst_id'][burst].decode()
                sign = -.5 if 'iw1' in name else .5 if 'iw2' in name else np.nan
                delta = int(later.endswith('B'))-int(earlier.endswith('B'))
                std = sign*delta*float(record['beta_std_px'])*float(source['azimuth_ground_spacing_m_per_slc_line'][row, col])
                calibration.append(std*result['inverse'][:, 1])
        if len(tide)==len(self.pairs):
            result['tidal_range_bias'] = np.asarray(tide)
        if len(calibration)==len(self.pairs):
            result['calibration_std_en'] = np.asarray(calibration)
        return result


def save_fit(path, collection, results, parameterization, recipe):
    """Write one tile/whole-grid fit with a self-contained analytic velocity model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.nc')
    os.close(fd)
    try:
        first = next(iter(results.values()), None)
        if first is None:
            raise ValueError('No estimable velocity fits')
        nparam = len(first['coefficients'])
        with h5netcdf.File(temp, 'w') as h:
            h.dimensions = {'y': len(collection.y), 'x': len(collection.x), 'parameter': nparam,
                            'parameter2': nparam, 'en': 2, 'target': len(first['target_indices']),
                            'target2': len(first['target_indices'])}
            h.attrs.update(schema_version=SCHEMA, status='complete', parameterization=parameterization,
                           EPSG=collection.handles[0].attrs['EPSG'], phase_convention='atan2(S,C); radians; v=C*cos(omega*tau)+S*sin(omega*tau)',
                           vector_basis='local geographic east/north', uncertainty='model conditional; empirical RA noise; paired block draws reported separately',
                           recipe_json=json.dumps(serializable(recipe), sort_keys=True), recipe_identity=fingerprint(recipe),
                           basis_json=json.dumps(first['basis']), created_utc=datetime.now(timezone.utc).isoformat())
            for name, dim, data in [('x', ('x',), collection.x), ('y', ('y',), collection.y),
                                    ('source_observation_index', ('y', 'x'), collection.association)]:
                v = h.create_variable(name, dim, dtype=data.dtype)
                v[:] = data
            crs = h.create_variable('crs', (), dtype='i4')
            crs[()] = 0
            crs.attrs.update(dict(collection.handles[0]['crs'].attrs))
            text = h.create_variable('parameter_name', ('parameter',), dtype=h5py.string_dtype())
            text[:] = first['names']
            units = h.create_variable('parameter_units',('parameter',),dtype=h5py.string_dtype())
            units[:] = ['m day-2' if name.startswith('trend_') else 'm day-1' for name in first['names']]
            indices = h.create_variable('target_parameter_index',('target',),dtype='i4')
            indices[:] = first['target_indices']
            fields = {
                'coefficients': ('parameter', 'y', 'x'), 'coefficient_covariance': ('parameter', 'parameter2', 'y', 'x'),
                'flow_direction': ('en', 'y', 'x'), 'target_information': ('target', 'target2', 'y', 'x')}
            scalar = ('amplitude', 'phase', 'amplitude_std', 'phase_std', 'cross_amplitude', 'ellipse_major', 'ellipse_minor',
                      'ellipse_orientation','ellipse_rotation','p_value','along_p_value', 'global_condition', 'target_rank', 'target_data_rank', 'n_pairs', 'span_days', 'whitened_rms',
                      'mean_robust_factor', 'phase_coverage', 'residual_rms_m')
            fields.update({k: ('y', 'x') for k in scalar})
            variables = {}
            for name, dims in fields.items():
                chunks = tuple(1 if d not in ('y', 'x') else min(64, len(h.dimensions[d])) for d in dims)
                variables[name] = h.create_variable(name, dims, dtype='f8', fillvalue=np.nan, chunks=chunks, compression='gzip')
                variables[name].attrs['grid_mapping'] = 'crs'
                if name=='global_condition':
                    variables[name].attrs['definition']='SVD condition of standardized whitened design including the specified penalty'
                if name in ('amplitude', 'amplitude_std', 'cross_amplitude', 'ellipse_major', 'ellipse_minor'):
                    variables[name].attrs['units'] = 'm day-1'
                elif name in ('phase', 'phase_std','ellipse_orientation'):
                    variables[name].attrs['units'] = 'radian'
            masks = {k: h.create_variable(k, ('y', 'x'), dtype='u1', chunks=(min(64,len(collection.y)),min(64,len(collection.x))), compression='gzip')
                     for k in ('fit_valid', 'detection_valid', 'phase_valid')}
            for mask in masks.values():
                mask[:] = 0
            for (row, col), result in results.items():
                from .velocity_fit import diagnostics
                diag = diagnostics(result)
                values = result | diag | {'coefficient_covariance': result['covariance'], 'flow_direction': result['direction'],
                        'mean_robust_factor': np.mean(result['robust_factors']), 'residual_rms_m': np.sqrt(np.mean(result['residual']**2))}
                for name, variable in variables.items():
                    variable[(..., row, col)] = values.get(name, np.nan)
                masks['fit_valid'][row, col] = result.get('fit_valid', False)
                masks['detection_valid'][row, col] = result.get('detection_valid', False)
                masks['phase_valid'][row, col] = result.get('phase_valid', False)
        collection.check_revisions()
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def evaluate_file(path, times, row, col):
    """Evaluate a standalone fit netCDF without its inputs or processing stack."""
    with h5netcdf.File(path, 'r') as h:
        if h.attrs.get('schema_version') != SCHEMA or h.attrs.get('status') != 'complete':
            raise ValueError('Incomplete or unsupported velocity fit')
        if not h['fit_valid'][row, col]:
            raise ValueError('Velocity coefficients are not estimable at this location')
        return evaluate(h['coefficients'][:, row, col], json.loads(h.attrs['basis_json']),
                        h.attrs['parameterization'], h['flow_direction'][:, row, col], utc_days(times))


def export_daily(path, output, start, end):
    """Explicit full-grid daily export; fitting never creates these cubes itself.

    Start is inclusive and end is exclusive, both UTC. The coefficient netCDF is
    the only scientific input; parent displacement packages may be offline.
    """
    path,output = Path(path).resolve(),Path(output).resolve()
    if path==output:
        raise ValueError('Daily export must differ from coefficient input')
    days = np.arange(float(utc_days(start)),float(utc_days(end)),1.)
    if not len(days):
        raise ValueError('Daily export end must follow start')
    before = revision(path)
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temp = tempfile.mkstemp(dir=output.parent,suffix='.nc');os.close(fd)
    try:
        with h5netcdf.File(path) as source,h5netcdf.File(temp,'w') as h:
            if source.attrs.get('schema_version')!=SCHEMA:
                raise ValueError('Daily export requires a velocity coefficient netCDF')
            ny,nx = (len(source.dimensions[k]) for k in ('y','x'))
            h.dimensions = {'time':len(days),'y':ny,'x':nx}
            h.attrs.update(schema_version='glacier-offsets-daily-velocity-v1',EPSG=source.attrs['EPSG'],
                           coefficient_input_sha256=digest(path),vector_basis='local geographic east/north')
            time = h.create_variable('time',('time',),dtype='f8');time[:]=days
            time.attrs.update(units='days since 2000-01-01 00:00:00',calendar='proleptic_gregorian')
            for name in ('x','y','crs','fit_valid','detection_valid','phase_valid'):
                v = source[name]
                target = h.create_variable(name,v.dimensions,dtype=v.dtype)
                target[...] = v[...];target.attrs.update(dict(v.attrs))
            fields = [h.create_variable(name,('time','y','x'),dtype='f4',fillvalue=np.nan,
                        chunks=(1,min(64,ny),min(64,nx)),compression='gzip') for name in ('east_velocity','north_velocity')]
            for v in fields:v.attrs.update(units='m day-1',grid_mapping='crs')
            basis = json.loads(source.attrs['basis_json'])
            for row in range(0,ny,64):
                for col in range(0,nx,64):
                    section=(slice(row,min(row+64,ny)),slice(col,min(col+64,nx)))
                    mask=source['fit_valid'][section].astype(bool)
                    coefficients=source['coefficients'][(slice(None),*section)]
                    direction=source['flow_direction'][(slice(None),*section)]
                    velocity=np.full((len(days),*mask.shape,2),np.nan)
                    for r,c in zip(*np.where(mask)):
                        velocity[:,r,c]=evaluate(coefficients[:,r,c],basis,source.attrs['parameterization'],direction[:,r,c],days)
                    for axis,v in enumerate(fields):v[(slice(None),*section)]=velocity[...,axis]
        if revision(path)!=before:raise ValueError('Coefficient input changed during daily export')
        os.replace(temp,output)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return str(output)

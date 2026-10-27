"""Correct a self-contained displacement netCDF for CATS2008 vertical tides."""
import argparse
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile

import h5netcdf
import numpy as np

from .packaging import CALENDAR, TIME_UNITS, content_hashes, strings, tiles, validate_package, variable
from .products import ChecksumCache, checksum, identity
from .swath_correction import Inputs
from .tide_model import Cats2008

METHOD = 'cats2008-vertical-range-v1'
IMPLEMENTATION = {name: checksum(Path(__file__).with_name(name)) for name in ('tides.py', 'tide_model.py')}
FLOAT_FIELDS = ('range_displacement_before_tides_m', 'tide_height_earlier_m', 'tide_height_later_m',
                'tidal_vertical_displacement_m', 'tidal_range_bias_m')


def json_scalar(h, name):
    value = h[name][()]
    return json.loads(value.decode() if isinstance(value, bytes) else value)


@contextmanager
def output_lock(parent):
    """Serialize derived-file publication independently of any processing stack."""
    parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(parent, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another writer is using the output directory') from exc
        yield
    finally:
        os.close(fd)


def observation_geometry(h):
    """Deduplicate copied map cells without treating them as independent data."""
    association = h['source_observation_index'][:]
    supported = association >= 0
    ids, first = np.unique(association[supported], return_index=True)
    if not len(ids):
        raise ValueError('Input has no reference observation geometry for tidal prediction')
    locations = np.flatnonzero(supported)[first]
    up = h['enu'][:].astype(str).tolist().index('up')
    fields = {'longitude': h['observation_longitude'][:], 'latitude': h['observation_latitude'][:],
              'owner': h['source_burst_index'][:], 'range_up': h['range_sensitivity_enu'][up]}
    output = {}
    lookup = np.searchsorted(ids, association[supported])
    for key, a in fields.items():
        output[key] = a.ravel()[locations]
        if not np.array_equal(a[supported], output[key][lookup], equal_nan=True):
            raise ValueError(f'Repeated source observations have inconsistent {key}')
    if (not np.isfinite(output['longitude']).all() or not np.isfinite(output['latitude']).all() or
            not np.isfinite(output['range_up']).all() or np.any(output['owner'] <= 0)):
        raise ValueError('Incomplete source observation geometry')
    return ids, output


def apply_tides(temp, source_spec, geometry, ids, model, recipe):
    """Change range only, retaining raw values where model predictions are absent."""
    stats = {'corrected_cells': 0, 'uncorrected_valid_cells': 0, 'zero_response_cells': 0, 'max_abs_range_bias_m': 0.}
    with h5netcdf.File(temp, 'a') as h:
        original_hashes = json.loads(h.attrs['content_hashes_json'])
        shape = (len(h.dimensions['y']), len(h.dimensions['x']))
        fields = {key: variable(h, key, ('time', 'y', 'x'), dtype='f4', units='m', grid_mapping='crs') for key in FLOAT_FIELDS}
        fields['tidal_correction_available'] = variable(h, 'tidal_correction_available', ('time', 'y', 'x'), dtype='u1', units='1',
            flag_values=np.array([0, 1], dtype='uint8'), flag_meanings='unavailable available')
        fields['tidal_range_bias_m'].attrs['definition'] = 'bias subtracted from range = (later tide - earlier tide) * range_sensitivity_enu[up]'
        fields['range_displacement_before_tides_m'].attrs['definition'] = 'exact input range values, before ocean-tide correction'
        for key, values in [('tidal_response_factor', model.response_factor), ('tidal_model_ocean_mask', model.ocean_mask)]:
            target = variable(h, key, ('y', 'x'), dtype='f4', units='1', grid_mapping='crs')
            for section in tiles(shape):
                association = h['source_observation_index'][section]
                supported = association >= 0
                a = np.full(association.shape, np.nan, dtype='float32')
                a[supported] = values[np.searchsorted(ids, association[supported])]
                target[section] = a
        h['tidal_response_factor'].attrs['definition'] = ('model flexure percent / 100, interpolated; values above one are retained'
            if recipe['response'] == 'model' else 'unit response; freely floating assumption')
        h['tidal_model_ocean_mask'].attrs['definition'] = 'linearly interpolated model land=0/ocean=1 field; not an observation quality mask'
        owners = geometry['owner']
        by_burst = {j: np.flatnonzero(owners == j) for j in range(1, len(h.dimensions['burst'])+1)}
        cache = OrderedDict()

        def heights(burst, epoch):
            key = (burst, float(epoch))
            if key not in cache:
                cache[key] = model.predict(by_burst[burst], epoch)
                if len(cache) > 32:
                    cache.popitem(last=False)
            cache.move_to_end(key)
            return cache[key]

        names = h['pair_id'][:].astype(str)
        for i, name in enumerate(names):
            logging.info('Correcting tides for %s (%d/%d)', name, i+1, len(names))
            early, late = (np.full(len(ids), np.nan) for _ in range(2))
            for burst, indices in by_burst.items():
                if not len(indices):
                    continue
                for key, out in [('earlier_burst_time', early), ('later_burst_time', late)]:
                    t = float(h[key][i, burst-1])
                    if np.isfinite(t):
                        out[indices] = heights(burst, t)
            available = model.available & np.isfinite(early) & np.isfinite(late)
            delta = late-early
            bias = delta*geometry['range_up']
            for section in tiles(shape):
                slot = (i, *section)
                raw = h['range_displacement_m'][slot]
                valid = h['valid_mask'][slot].astype(bool)
                association = h['source_observation_index'][section]
                supported = association >= 0
                if np.any(valid & ~supported):
                    raise ValueError(f'{name}: valid observation lacks a source association')
                lookup = np.searchsorted(ids, association[supported])
                take = np.zeros(raw.shape, dtype=bool)
                take[supported] = available[lookup]
                take &= valid
                for key, values in [('tide_height_earlier_m', early), ('tide_height_later_m', late),
                                    ('tidal_vertical_displacement_m', delta), ('tidal_range_bias_m', bias)]:
                    a = np.full(raw.shape, np.nan, dtype='float32')
                    a[supported] = values[lookup]
                    a[~take] = np.nan
                    fields[key][slot] = a
                fields['range_displacement_before_tides_m'][slot] = raw
                fields['tidal_correction_available'][slot] = take.astype('uint8')
                corrected = raw.copy()
                # Subtract the double-precision projected tide before returning
                # to the input float32 displacement storage precision.
                full_bias = np.full(raw.shape, np.nan)
                full_bias[supported] = bias[lookup]
                corrected[take] = (raw[take].astype('float64')-full_bias[take]).astype('float32')
                h['range_displacement_m'][slot] = corrected
                response = np.full(raw.shape, np.nan)
                response[supported] = model.response_factor[lookup]
                stats['corrected_cells'] += int(take.sum())
                stats['uncorrected_valid_cells'] += int((valid & ~take).sum())
                stats['zero_response_cells'] += int((take & (response == 0)).sum())
                if take.any():
                    stats['max_abs_range_bias_m'] = max(stats['max_abs_range_bias_m'], float(np.max(np.abs(full_bias[take]))))
        history = source_spec.get('postprocessing', []) + [recipe]
        spec = dict(source_spec, postprocessing=history)
        h['provenance_json'][()] = json.dumps(spec, sort_keys=True)
        strings(h, 'tidal_correction_provenance_json', (), json.dumps(recipe | {'model': model.metadata, 'counts': stats}, sort_keys=True))
        h.attrs.update(package_identity=identity(spec), tidal_correction_method=METHOD,
                       tidal_correction_identity=identity(recipe), tidal_correction_response=recipe['response'],
                       tidal_correction_status='partial_model_coverage' if stats['uncorrected_valid_cells'] else 'complete',
                       tidal_correction_azimuth='unchanged; correction applies only to range',
                       range_basis='slant range; positive away from reference sensor; no division by sin(incidence)',
                       tidal_missing_policy='retain original displacement and valid mask; correction availability is separate')
        h['range_displacement_m'].attrs['tidal_correction'] = 'vertical ocean tide removed; slant-range metres retained'
        h['range_displacement_m'].attrs['ancillary_variables'] = 'tidal_range_bias_m tidal_correction_available range_displacement_before_tides_m'
        h.attrs['history'] = (str(h.attrs.get('history', ''))+'\n'+datetime.now(timezone.utc).isoformat()+' '+METHOD).strip()
        hashes = content_hashes(h)
        for name, expected in original_hashes.items():
            if name not in ('range_displacement_m', 'provenance_json') and hashes.get(name) != expected:
                raise ValueError(f'Tidal processing unexpectedly changed {name}')
        h.attrs['content_hashes_json'] = json.dumps(hashes, sort_keys=True)
    if stats['uncorrected_valid_cells']:
        logging.warning('%d valid cells have no modeled tide correction; original displacements retained with availability=0',
                        stats['uncorrected_valid_cells'])
    return stats


def validate_tidal_stack(path):
    """Validate a derived stack and its range-correction arithmetic in isolation."""
    result = validate_package(path)
    with h5netcdf.File(path, 'r') as h:
        if h.attrs.get('tidal_correction_method') != METHOD:
            raise ValueError('File is not a supported tide-corrected displacement stack')
        record = json_scalar(h, 'tidal_correction_provenance_json')
        recipe = {k: v for k, v in record.items() if k not in ('model', 'counts')}
        if h.attrs['tidal_correction_identity'] != identity(recipe):
            raise ValueError('Tidal correction provenance differs from its identity')
        if h['tidal_correction_available'].dtype != np.dtype('u1') or '_FillValue' in h['tidal_correction_available'].attrs:
            raise ValueError('Tidal availability must be an integer 0/1 mask without a fill value')
        shape = result['shape']
        up = h['enu'][:].astype(str).tolist().index('up')
        for i in range(shape[0]):
            for section in tiles(shape[1:]):
                slot = (i, *section)
                available = h['tidal_correction_available'][slot]
                valid = h['valid_mask'][slot].astype(bool)
                if not np.isin(available, [0, 1]).all() or np.any((available == 1) & ~valid):
                    raise ValueError('Tidal availability is inconsistent with valid observations')
                take = available == 1
                raw, corrected = h['range_displacement_before_tides_m'][slot], h['range_displacement_m'][slot]
                if not np.array_equal(raw[~take], corrected[~take], equal_nan=True):
                    raise ValueError('Unavailable tidal corrections changed original displacements')
                early, late = h['tide_height_earlier_m'][slot], h['tide_height_later_m'][slot]
                delta, bias = h['tidal_vertical_displacement_m'][slot], h['tidal_range_bias_m'][slot]
                if not all(np.isfinite(a[take]).all() and np.isnan(a[~take]).all() for a in (early, late, delta, bias)):
                    raise ValueError('Tidal fields have inconsistent availability')
                np.testing.assert_allclose(late[take]-early[take], delta[take], rtol=2e-6, atol=1e-6)
                sensitivity = h['range_sensitivity_enu'][(up, *section)]
                np.testing.assert_allclose(bias[take], delta[take]*sensitivity[take], rtol=2e-6, atol=1e-6)
                np.testing.assert_allclose(raw[take].astype(float)-bias[take], corrected[take], rtol=2e-6, atol=2e-6)
        return result | {'correction': METHOD, 'tidal_correction_status': h.attrs['tidal_correction_status'], 'counts': record['counts']}


def correct_tides(source, model_path, *, output=None, response='model', force=False):
    source = Path(source).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    destination = Path(output).expanduser().resolve() if output else source.with_name(source.stem+'_tide_corrected.nc')
    if destination in (source, model_path) or (destination.exists() and (os.path.samefile(destination, source) or os.path.samefile(destination, model_path))):
        raise ValueError('Tidal correction requires a separate output file; input/model cannot be overwritten')
    if destination.suffix.lower() != '.nc':
        raise ValueError('Tidal output must be a netCDF .nc file')
    inputs = Inputs(ChecksumCache())
    logging.info('Validating source displacement stack')
    source_digest = inputs.file(source)
    validate_package(source)
    with h5netcdf.File(source, 'r') as h:
        spec = json_scalar(h, 'provenance_json')
        if h.attrs.get('tidal_correction_method') or any(s.get('method') == METHOD for s in spec.get('postprocessing', [])):
            raise ValueError('Input already has an ocean-tide correction; use the uncorrected parent stack')
        for key in ('earlier_burst_time', 'later_burst_time'):
            if h[key].attrs.get('units') != TIME_UNITS or h[key].attrs.get('calendar') != CALENDAR:
                raise ValueError('Tidal endpoints must use the package UTC Unix-seconds time convention')
        ids, geometry = observation_geometry(h)
        parent_identity = h.attrs['package_identity']
    logging.info('Checking local CATS2008 model')
    model_digest = inputs.file(model_path)
    model = Cats2008(model_path, geometry['longitude'], geometry['latitude'], response=response)
    recipe = {'method': METHOD, 'implementation': IMPLEMENTATION, 'response': response,
              'input': {'path': str(source), 'sha256': source_digest, 'package_identity': parent_identity},
              'model_file': {'path': str(model_path), 'sha256': model_digest}, 'pyTMD_version': model.version,
              'timing': 'exact UTC earlier/later burst starts from netCDF, indexed by source burst',
              'formula': 'range_corrected = range_input - (eta_later - eta_earlier) * range_sensitivity_enu[up]',
              'basis': 'slant_range', 'interpolation': 'linear complex harmonics; no extrapolation', 'infer_minor': True,
              'unavailable': 'retain_input_with_availability_zero', 'azimuth': 'unchanged'}
    with output_lock(destination.parent):
        if destination.exists():
            with h5netcdf.File(destination, 'r') as h:
                if h.attrs.get('tidal_correction_method') != METHOD:
                    raise ValueError('Refusing to replace an unrelated output file')
                current = h.attrs.get('tidal_correction_identity') == identity(recipe)
            if current and not force:
                report = validate_tidal_stack(destination)
                inputs.unchanged()
                return report | {'reused': True}
            if not force:
                raise ValueError('Output has a different correction recipe; choose another output or use --force')
        with tempfile.TemporaryDirectory(prefix='.tidal-stack-', dir=destination.parent) as folder:
            temp = Path(folder) / destination.name
            shutil.copyfile(source, temp)
            apply_tides(temp, spec, geometry, ids, model, recipe)
            report = validate_tidal_stack(temp)
            inputs.unchanged()
            os.replace(temp, destination)
        return report | {'path': str(destination), 'reused': False}


def main(argv=None):
    parser = argparse.ArgumentParser(prog='glacier-offsets correct-tides', description=__doc__)
    parser.add_argument('stack', help='Self-contained displacement netCDF stack')
    parser.add_argument('--model', required=True, help='Local CATS2008_v2023.nc (TMD3) file')
    parser.add_argument('-o', '--output', help='Separate output .nc; default: INPUT_tide_corrected.nc')
    parser.add_argument('--response', choices=('model', 'floating'), default='model',
                        help='model: embedded flexure/grounding response (default); floating: full ocean-tide response')
    parser.add_argument('--force', action='store_true', help='Rebuild this derived output; never reapply tides to a corrected input')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        result = correct_tides(args.stack, args.model, output=args.output, response=args.response, force=args.force)
        print(json.dumps(result, indent=2))
        return 0
    except KeyboardInterrupt:
        logging.error('Interrupted; source and previous completed output are preserved')
        return 130
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, ImportError, AssertionError) as exc:
        logging.error('%s', exc)
        return 2

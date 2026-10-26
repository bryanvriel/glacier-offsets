"""Clean spacecraft-correction maps, retaining raw mixed-footprint observations.

Correction applies only where the frozen experiment supports it. Other original
observations are retained without correction, with an explicit availability mask.
No smoothing, filling, production mutation, or model refitting is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal
from pyproj import Transformer

from glacier_offsets.geocode import make_mapper, write_geotiff


def render(stack, experiment, pair, output):
    inputs = {}

    def checked(path, expected=None):
        path = Path(path).resolve()
        before = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'Changing input: {path}')
        if expected is not None and digest != expected:
            raise ValueError(f'Checksum mismatch: {path}')
        inputs[str(path)] = dict(sha256=digest, size=after.st_size, mtime_ns=after.st_mtime_ns)
        return path

    preview = experiment / 'previews' / pair
    report = json.loads(checked(preview / 'report.json').read_text())
    original = stack / 'pairs' / pair
    manifest = json.loads(checked(original / 'manifest.json').read_text())
    if manifest['match_identity'] != report['specification']['match_identity']:
        raise ValueError('Production matching identity changed')
    checked(original / 'radar/offsets.h5', report['raw_input_sha256'])
    source = (preview / 'derived_radar.npz').resolve()
    with np.load(checked(source, report['products'][str(source)])) as z:
        total = z['raw_total_azimuth_px']
        scale = z['azimuth_scale_m_per_px']
        interval = z['interval_seconds']
        available = z['spacecraft_correction_available']
        bias = z['spacecraft_bias_to_subtract_px']
        mixed = z['cross_swath_mixed']
    with h5py.File(checked(original / 'geo/center_geometry.h5'), 'r') as h:
        metadata = json.loads(h.attrs['metadata_json'])
        if metadata['match_identity'] != manifest['match_identity']:
            raise ValueError('Cached center geometry belongs to a different match')
        lon, lat = h['longitude_deg'][:], h['latitude_deg'][:]
        geometry_valid = h['geometry_valid'][:]
    with h5py.File(original / 'radar/offsets.h5', 'r') as h:
        base = h['input_valid'][:] & ((h['rejection_flags'][:] & 7) == 0)
        base &= np.isfinite(total) & np.isfinite(h['total/range_px'][:])
    base &= geometry_valid & np.isfinite(interval) & (interval > 0)
    grid = manifest['map_grid']
    x, y = Transformer.from_crs(4326, grid['epsg'], always_xy=True).transform(lon, lat)
    mapper = make_mapper(x, y, base, grid['transform'][1], 'nearest', target_grid=grid)
    production_path = checked(original / 'geo/azimuth_displacement_m.tif')
    ds = gdal.Open(str(production_path))
    production = ds.ReadAsArray()
    raw = mapper.map(total * scale)
    # Prove that every displayed raw pixel uses the production source operator.
    np.testing.assert_array_equal(raw, production)
    applied_px = np.where(available, -bias, 0.)
    corrected = mapper.map((total + applied_px) * scale)
    applied = mapper.map(applied_px * scale)
    corrected_velocity = mapper.map((total + applied_px) * scale / (interval / 86400))
    mapped_available = mapper.map(available.astype('float32'))
    mapped_mixed = mapper.map(mixed.astype('float32'))
    unchanged = np.isfinite(raw) & (mapped_available == 0)
    np.testing.assert_array_equal(corrected[unchanged], raw[unchanged])
    np.testing.assert_allclose(corrected - raw, applied, atol=2e-4, rtol=1e-3, equal_nan=True)
    delta = report['predictions_px']['spacecraft']
    common_path = (preview / 'spacecraft_corrected_azimuth_displacement.tif').resolve()
    old = gdal.Open(str(checked(common_path, report['products'][str(common_path)]))).ReadAsArray()
    selected = np.isfinite(old)
    np.testing.assert_array_equal(corrected[selected], old[selected])
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'units': 'm', 'pair': pair, 'model': 'leave-one-link-out spacecraft',
                'relative_IW2_minus_IW1_contrast_px': delta,
                'allocation': 'subtract IW1=-delta/2, IW2=+delta/2',
                'spatial_assumption': 'constant relative bias per swath in radar pixels',
                'unsupported_policy': 'retain raw observations; correction_available=0',
                'cross_swath_mixed_policy': 'retain raw observations, no correction',
                'resampling': 'unchanged production nearest-source operator',
                'display': 'no overlap outlines, smoothing, or filling'}
    fields = {'raw_azimuth_displacement_m': raw,
              'spacecraft_corrected_with_raw_unsupported_azimuth_displacement_m': corrected,
              'spacecraft_corrected_with_raw_unsupported_azimuth_velocity_m_per_day': corrected_velocity,
              'spacecraft_correction_added_m': applied,
              'spacecraft_correction_available': mapped_available, 'cross_swath_mixed': mapped_mixed}
    for name, value in fields.items():
        units = '0/1' if name in ('spacecraft_correction_available', 'cross_swath_mixed') else (
            'm/day' if 'velocity' in name else 'm')
        write_geotiff(output / (name + '.tif'), value, mapper.transform, grid['epsg'], metadata | {'units': units})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x0, dx, _, y0, _, dy = mapper.transform
    extent = [x0, x0 + raw.shape[1]*dx, y0 + raw.shape[0]*dy, y0]
    for name, values, titles, figsize in (
        ('corrected_clean.png', [corrected], ['Spacecraft-corrected azimuth displacement'], (10, 8)),
        ('comparison_clean.png', [raw, corrected], ['Raw', 'Spacecraft corrected'], (16, 7))):
        fig, axes = plt.subplots(1, len(values), figsize=figsize, layout='constrained', squeeze=False)
        for ax, value, title in zip(axes[0], values, titles):
            im = ax.imshow(value, extent=extent, origin='upper', cmap='RdBu_r',
                           vmin=-5, vmax=5, interpolation='nearest')
            ax.set_title(title)
            ax.set_xlabel('Easting (m, EPSG:3031)')
            ax.set_ylabel('Northing (m, EPSG:3031)')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.9, label='Azimuth displacement (m)')
        fig.suptitle(pair.replace('_', ' to '))
        fig.supxlabel('Mixed-swath and unsupported windows retain their raw measurements', fontsize=9)
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
    for path, record in list(inputs.items()):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Consumed input changed: {path}')
    record = {'inputs': inputs, 'metadata': metadata,
              'checks': {'raw_equals_production_exactly': True,
                         'supported_corrected_equals_existing_preview_exactly': True,
                         'unsupported_observations_unchanged_exactly': True,
                         'inputs_unchanged': True},
              'map_pixels': {'raw': int(np.isfinite(raw).sum()),
                             'corrected': int((mapped_available == 1).sum()),
                             'retained_raw_unsupported': int(unchanged.sum()),
                             'cross_swath_mixed': int((mapped_mixed == 1).sum())}}
    (output / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record['map_pixels'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', type=Path, default=Path('work/larsen/stack'))
    parser.add_argument('--experiment', type=Path, default=Path('work/larsen/swath-experiment-2017h1'))
    parser.add_argument('--pair', default='2017-01-19_2017-01-25')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.stack.resolve(), args.experiment.resolve(), args.pair, args.output.resolve())


if __name__ == '__main__':
    main()

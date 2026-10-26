"""Production correction contracts: units, ownership, isolation and resumption."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal
from pyproj import Transformer
import yaml

from glacier_offsets.ampcor import matching_specification
from glacier_offsets.cli import main
from glacier_offsets.config import load_defaults
from glacier_offsets.geocode import geocoding_specification, make_mapper, map_grid, write_geotiff
from glacier_offsets.products import checksum, identity, write_json
from glacier_offsets.stack_config import load_status_config
from glacier_offsets.stack_status import inspect_stack
from glacier_offsets.swath_correction import correction_arrays, correction_status, execute_corrections
from glacier_offsets.swath_correction_config import METHOD, MATCHING_KEYS, load_calibration, settings
from glacier_offsets.swath_ownership import pure_swath_footprints


def fake_plot(stage, *args):
    (stage / 'diagnostics/comparison.png').write_bytes(b'figure')


class ProductionSwathCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve()
        self.root = self.folder / 'stack'
        self.pair = '2017-01-01_2017-01-07'
        self.raw = self.root / 'pairs' / self.pair
        for name in ('radar', 'geo'):
            (self.raw / name).mkdir(parents=True)
        (self.root / 'geometry').mkdir()
        self.target = self.root / 'pairs_swath_corrected'
        self.cfg = load_defaults()
        self.cfg.update(output={'directory': str(self.root)}, stack={
            'processing': 'isce3_merge_first', 'swath_merge': 'midpoint',
            'safe_directory': str(self.folder / 'offline-safe'),
            'orbit_directory': str(self.folder / 'offline-orbits')})
        self.cfg['inputs'].update(polarization='HH', subswaths=[1, 2])
        self.cfg['matching'].update(engine='isce3', backend='cpu', deramp=0,
                                    window=[8, 8], search=[2, 2], stride=[6, 6])
        self.cfg['geocoding'].update(epsg=3031, spacing_m=200, resampling='nearest', mask='input')
        self.ids = ['t038_080975_iw1', 't038_080976_iw1', 't038_080975_iw2']
        owner_image = np.ones((64, 128), 'uint16')
        owner_image[32:, :64] = 2
        owner_image[:, 64:] = 3
        path = self.root / 'geometry/source.tif'
        ds = gdal.GetDriverByName('GTiff').Create(str(path), 128, 64, 1, gdal.GDT_UInt16)
        ds.GetRasterBand(1).WriteArray(owner_image)
        ds = None
        frame_spec = {'reference': 'fixture'}
        self.frame = {'status': 'complete', 'specification': frame_spec,
                      'geometry_identity': identity(frame_spec), 'products': {str(path): checksum(path)},
                      'coreg': {'source_index': str(path)},
                      'sources': [{'burst_id': k} for k in self.ids]}
        write_json(self.root / 'geometry/manifest.json', self.frame)
        self.line, self.sample = np.meshgrid([13.5, 31.5, 37.5, 49.5],
                                            [13.5, 31.5, 55.5, 61.5, 67.5, 85.5, 103.5], indexing='ij')
        self.owner = owner_image[np.rint(self.line).astype(int), np.rint(self.sample).astype(int)]
        self.swath = np.where(self.owner == 3, 2, 1)
        self.interval = 6 * 86400 + self.owner.astype(float)
        self.scale = 14 + self.sample * .001
        self.total = 3.25 + np.where(self.swath == 1, -.026 / 2, .026 / 2)
        self.base = np.ones(self.total.shape, bool)
        self.base[0, 0] = False
        self.quality = self.base.copy()
        self.quality[1, 5] = False
        x, y = self.sample * 200, 1e6 - self.line * 200
        self.lon, self.lat = Transformer.from_crs(3031, 4326, always_xy=True).transform(x, y)
        x, y = Transformer.from_crs(4326, 3031, always_xy=True).transform(self.lon, self.lat)
        grid = map_grid(x, y, 200) | {'epsg': 3031}
        self.mapper = make_mapper(x, y, self.base, 200, 'nearest', target_grid=grid)
        dates = self.pair.split('_')
        self.scenes = [dict(date=day, path=str(self.folder / f'{mission}_IW_SLC_{day}.SAFE'),
                            mission=mission, relative_orbit=38, direction='DESCENDING')
                       for day, mission in zip(dates, ['S1A', 'S1B'])]
        sources = {s['date']: {'safe': {'path': s['path'], 'sha256': 'fixture'}} for s in self.scenes}
        coreg = {'source_index': str(path), 'reference_grid': {'prf': 10., 'range_pixel_spacing': 2.}}
        m = {'coreg_identity': 'raw-coreg', 'coreg': coreg,
             'coreg_specification': {'polarization': 'HH', 'inputs': sources},
             'stack_pair': {'geometry_identity': self.frame['geometry_identity'], 'acquisition_dates': dates},
             'assembly': {'source_burst_ids': self.ids}, 'map_grid': grid,
             'stages': {'match': 'complete', 'geocode': 'complete'}}
        m['match_identity'] = identity(matching_specification(self.cfg, m))
        m['geocode_identity'] = identity(geocoding_specification(self.cfg, m))
        m['matching'] = {'matching': self.cfg['matching'], 'coreg_identity': m['coreg_identity'],
                         'input_provenance': sources, 'source_burst_ids': self.ids}
        self.manifest = m
        write_json(self.raw / 'manifest.json', m)
        with h5py.File(self.raw / 'radar/offsets.h5', 'w') as h:
            for name, value in {'center_line': self.line, 'center_sample': self.sample,
                                'source_burst_index': self.owner, 'interval_seconds': self.interval,
                                'input_valid': self.base, 'valid': self.quality,
                                'rejection_flags': np.zeros(self.total.shape, 'uint8'),
                                'seam_crossing': np.broadcast_to(np.arange(4)[:, None] == 1, self.total.shape),
                                'gross/azimuth_px': np.zeros(self.total.shape),
                                'residual/azimuth_px': self.total, 'total/azimuth_px': self.total,
                                'total/range_px': np.full(self.total.shape, -1.5),
                                'covariance': np.full(self.total.shape + (3,), .2)}.items():
                h.create_dataset(name, data=value)
            h.attrs['metadata_json'] = json.dumps(m['matching'])
        with h5py.File(self.raw / 'geo/center_geometry.h5', 'w') as h:
            for name, value in {'longitude_deg': self.lon, 'latitude_deg': self.lat,
                                'geometry_valid': np.ones(self.total.shape, bool),
                                'azimuth_ground_spacing_m_per_slc_line': self.scale,
                                'center_line': self.line, 'center_sample': self.sample,
                                'source_burst_index': self.owner, 'interval_seconds': self.interval}.items():
                h.create_dataset(name, data=value)
            h.attrs['metadata_json'] = json.dumps({'match_identity': m['match_identity']})
        raw_map = self.mapper.map(self.total * self.scale)
        for name, array in {'azimuth_displacement_m': raw_map,
                            'azimuth_velocity_m_per_day': self.mapper.map(self.total * self.scale / (self.interval / 86400)),
                            'range_displacement_m': self.mapper.map(np.full(self.total.shape, -3)),
                            'range_velocity_m_per_day': self.mapper.map(-3 / (self.interval / 86400)),
                            'azimuth_displacement_sigma_m': self.mapper.map(np.full(self.total.shape, .2)),
                            'valid_mask': np.isfinite(raw_map).astype('float32')}.items():
            write_geotiff(self.raw / 'geo' / (name + '.tif'), array, self.mapper.transform, 3031, {'units': 'fixture'})
        write_json(self.root / 'stack_manifest.json', {'pairs': {self.pair: {'active': True}}, 'inventory': self.scenes})
        self.cal_path = self.folder / 'calibration.json'
        self.cal = {'method': METHOD, 'units': 'original_reference_azimuth_pixels',
                    'beta_px': .026, 'beta_std_px': .009, 'reference_prf_hz': 10.,
                    'swaths': [1, 2], 'allocation': 'symmetric',
                    'scope': {'geometry_identity': self.frame['geometry_identity'],
                              'relative_orbit': 38, 'direction': 'DESCENDING', 'polarization': 'HH',
                              'processing': 'isce3_merge_first', 'swath_merge': 'midpoint',
                              'matching': {k: self.cfg['matching'].get(k) for k in MATCHING_KEYS},
                              'start_date': '2017-01-01', 'end_date': '2017-12-31'},
                    'provenance': {'training_dates': dates}}
        self.cfg['stack']['swath_correction'] = {'enabled': True, 'calibration': str(self.cal_path)}
        self.save_cal()
        self.plot = patch('glacier_offsets.swath_correction.plot_comparison', side_effect=fake_plot)
        self.plot_mock = self.plot.start()
        self.addCleanup(self.plot.stop)

    def save_cal(self):
        write_json(self.cal_path, self.cal)
        self.cfg['stack']['swath_correction']['calibration_sha256'] = checksum(self.cal_path)

    def snapshot(self):
        return {str(p): (checksum(p), p.stat().st_mtime_ns) for folder in (self.raw, self.root / 'geometry')
                for p in folder.rglob('*') if p.is_file()}

    def corrected(self):
        return self.target / self.pair

    def test_known_signed_fractional_bias_preserves_common_motion_and_same_satellite(self):
        swath = np.array([1, 2, 1, 2])
        for contrast in (-1, 0, 1):
            delta = contrast * .026
            original = -2.125 + np.where(swath == 1, -delta / 2, delta / 2)
            corrected, bias, available, sigma = correction_arrays(
                original, swath, swath, np.ones(4, bool), delta, abs(contrast) * .009)
            np.testing.assert_allclose(corrected, -2.125)
            np.testing.assert_array_equal(available, True)
            np.testing.assert_allclose(sigma, abs(contrast) * .0045)
            self.assertAlmostEqual(bias[1] - bias[0], delta)

    def test_full_pair_scope_units_ownership_masks_and_raw_isolation(self):
        before = self.snapshot()
        with patch('glacier_offsets.inputs.inventory', side_effect=AssertionError('discovered SAFE')), \
             patch('glacier_offsets.swath_models.train', side_effect=AssertionError('fitted data')):
            self.assertEqual(execute_corrections(self.cfg), 0)
        self.assertEqual(before, self.snapshot())
        with h5py.File(self.corrected() / 'radar/offsets.h5') as h:
            available = h['correction_available'][:]
            corrected = h['corrected_total/azimuth_px'][:]
            np.testing.assert_allclose(corrected[available], 3.25)
            np.testing.assert_array_equal(corrected[~available], self.total[~available])
            np.testing.assert_array_equal(h['total/azimuth_px'][:], self.total)
            np.testing.assert_array_equal(h['gross/azimuth_px'][:], 0)
            np.testing.assert_array_equal(h['residual/azimuth_px'][:], self.total)
            self.assertTrue(h['same_swath_burst_join'][1, 0])
            self.assertTrue(available[1, 0])  # Same-swath source joins are supported.
            self.assertFalse(available[2, 3])  # Full search straddles IW1/IW2.
            np.testing.assert_allclose(h['correction_std/azimuth_px'][:][available], .0045)
        for name in ('range_displacement_m.tif', 'range_velocity_m_per_day.tif', 'azimuth_displacement_sigma_m.tif'):
            self.assertEqual(checksum(self.corrected() / 'geo' / name), checksum(self.raw / 'geo' / name))
        path = self.corrected() / 'geo/azimuth_velocity_m_per_day.tif'
        actual = gdal.Open(str(path)).ReadAsArray()
        expected = self.mapper.map(corrected * self.scale / (self.interval / 86400))
        np.testing.assert_array_equal(actual, expected)
        displacement = gdal.Open(str(self.corrected() / 'geo/azimuth_displacement_m.tif')).ReadAsArray()
        self.assertTrue(np.isnan(displacement).any())  # Input holes remain holes.
        self.assertEqual(correction_status(self.cfg, self.manifest, self.pair)['status'], 'complete')

    def test_resume_ignores_global_progress_and_detects_damaged_outputs(self):
        self.assertEqual(execute_corrections(self.cfg), 0)
        old_time = (self.corrected() / 'manifest.json').stat().st_mtime_ns
        catalog = read(self.root / 'stack_manifest.json')
        catalog['progress'] = 'unrelated progress update'
        write_json(self.root / 'stack_manifest.json', catalog)
        self.assertEqual(execute_corrections(self.cfg), 0)
        self.assertEqual(self.plot_mock.call_count, 1)
        self.assertEqual(old_time, (self.corrected() / 'manifest.json').stat().st_mtime_ns)
        (self.corrected() / 'geo/azimuth_displacement_m.tif').write_bytes(b'damaged')
        self.assertEqual(execute_corrections(self.cfg), 0)
        self.assertEqual(self.plot_mock.call_count, 2)

    def test_appledouble_sidecars_are_not_inputs_or_completion_products(self):
        hidden = self.raw / 'geo/._range_displacement_m.tif'
        hidden.write_bytes(b'AppleDouble, not a TIFF')
        def plot_with_sidecars(stage, *args):
            fake_plot(stage)
            (stage / '._calibration.json').write_bytes(b'metadata')
            (stage / 'geo/._range_displacement_m.tif').write_bytes(b'metadata')
        self.plot_mock.side_effect = plot_with_sidecars
        self.assertEqual(execute_corrections(self.cfg), 0)
        record = read(self.corrected() / 'manifest.json')
        self.assertNotIn(str(hidden), record['inputs'])
        self.assertTrue(all(not any(x.startswith('.') for x in Path(p).parts) for p in record['products']))
        self.assertEqual(execute_corrections(self.cfg), 0)
        self.assertEqual(self.plot_mock.call_count, 1)

    def test_calibration_change_only_regenerates_derived_outputs(self):
        self.assertEqual(execute_corrections(self.cfg), 0)
        before = self.snapshot()
        self.cal['beta_px'] = .04
        self.save_cal()
        self.assertEqual(correction_status(self.cfg, self.manifest, self.pair)['status'], 'stale')
        self.assertEqual(execute_corrections(self.cfg), 0)
        self.assertEqual(before, self.snapshot())
        m = read(self.corrected() / 'manifest.json')
        self.assertEqual(m['metadata']['pair_contrast_px'], .04)

    def test_changed_raw_offsets_cannot_reuse_corrected_completion_or_stale_maps(self):
        self.assertEqual(execute_corrections(self.cfg), 0)
        with h5py.File(self.raw / 'radar/offsets.h5', 'a') as h:
            h['total/azimuth_px'][2, 0] += 1
        self.assertEqual(correction_status(self.cfg, self.manifest, self.pair)['status'], 'stale')
        self.assertEqual(execute_corrections(self.cfg), 1)
        self.assertEqual(self.plot_mock.call_count, 1)

    def test_missing_range_product_is_an_incomplete_raw_pair(self):
        (self.raw / 'geo/range_displacement_m.tif').unlink()
        self.assertEqual(execute_corrections(self.cfg), 1)
        self.assertFalse((self.corrected() / 'manifest.json').exists())

    def test_interruption_and_independent_pair_failure(self):
        self.plot_mock.side_effect = KeyboardInterrupt
        before = self.snapshot()
        with self.assertRaises(KeyboardInterrupt):
            execute_corrections(self.cfg)
        self.assertEqual(read(self.target / 'correction_status.json')['status'], 'interrupted')
        self.assertFalse((self.corrected() / 'manifest.json').exists())
        self.assertEqual(before, self.snapshot())
        self.plot_mock.side_effect = fake_plot
        bad = '2017-01-07_2017-01-13'
        catalog = read(self.root / 'stack_manifest.json')
        catalog['pairs'][bad] = {'active': True}
        write_json(self.root / 'stack_manifest.json', catalog)
        self.assertEqual(execute_corrections(self.cfg), 1)
        report = read(self.target / 'correction_status.json')
        self.assertEqual(report['pairs'][self.pair]['status'], 'complete')
        self.assertEqual(report['pairs'][bad]['status'], 'failed')
        self.assertEqual(execute_corrections(self.cfg), 1)
        self.assertTrue(read(self.target / 'correction_status.json')['pairs'][self.pair]['reused'])

    def test_scope_unknown_mission_and_missing_predictions_fail_closed(self):
        for key, bad in [('beta_px', None), ('beta_px', float('nan')), ('beta_px', True), ('beta_std_px', -1)]:
            original = self.cal[key]
            self.cal[key] = bad
            self.save_cal()
            with self.assertRaises(ValueError):
                load_calibration(self.cfg)
            self.cal[key] = original
        self.save_cal()
        self.cal['scope']['matching']['range_oversampling'] = 2
        self.save_cal()
        with self.assertRaisesRegex(ValueError, 'matching treatment'):
            execute_corrections(self.cfg)
        self.cal['scope']['matching']['range_oversampling'] = self.cfg['matching']['range_oversampling']
        self.save_cal()
        catalog = read(self.root / 'stack_manifest.json')
        catalog['inventory'][1]['mission'] = 'S1C'
        write_json(self.root / 'stack_manifest.json', catalog)
        self.assertEqual(execute_corrections(self.cfg), 1)
        self.assertFalse((self.corrected() / 'manifest.json').exists())

    def test_config_pin_inheritance_and_output_alias_protection(self):
        parent = self.folder / 'cfg/base.yaml'
        parent.parent.mkdir()
        cfg = deepcopy(self.cfg)
        cfg['stack']['swath_correction']['calibration'] = '../calibration.json'
        parent.write_text(yaml.safe_dump(cfg))
        child = self.folder / 'child.yaml'
        child.write_text('extends: cfg/base.yaml\n')
        resolved = load_status_config(child)
        self.assertEqual(resolved['stack']['swath_correction']['calibration'], str(self.cal_path))
        for target in (self.root, self.raw, self.root / 'pairs', self.root.parent,
                       self.root / 'geometry' / 'derived'):
            cfg = deepcopy(self.cfg)
            cfg['stack']['swath_correction']['pairs_directory'] = str(target)
            with self.assertRaisesRegex(ValueError, 'separate'):
                settings(cfg)
        symlink = self.folder / 'alias'
        symlink.symlink_to(self.root / 'pairs')
        cfg['stack']['swath_correction']['pairs_directory'] = str(symlink)
        with self.assertRaisesRegex(ValueError, 'separate'):
            settings(cfg)
        self.cal_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_calibration(self.cfg)
        for pin in (None, 123, True, 'not-a-checksum'):
            cfg = deepcopy(self.cfg)
            cfg['stack']['swath_correction']['calibration_sha256'] = pin
            with self.assertRaises(ValueError):
                settings(cfg)

    def test_malformed_calibration_container_never_becomes_a_prediction(self):
        for invalid in ([], None, 'calibration', {'method': METHOD}):
            write_json(self.cal_path, invalid)
            self.cfg['stack']['swath_correction']['calibration_sha256'] = checksum(self.cal_path)
            with self.assertRaises(ValueError):
                load_calibration(self.cfg)

    def test_cli_runs_offline_without_native_or_upstream_processing(self):
        path = self.folder / 'stack.yaml'
        path.write_text(yaml.safe_dump(self.cfg))
        with patch('glacier_offsets.stack_cli.load_stack_config', side_effect=AssertionError('raw discovery')), \
             patch('glacier_offsets.stack_workflow._execute_stack', side_effect=AssertionError('production run')):
            self.assertEqual(main(['stack', 'correct-swaths', str(path), '--pair', self.pair]), 0)
        (self.corrected() / 'geo/correction_available.tif').unlink()
        self.assertEqual(correction_status(self.cfg, self.manifest, self.pair)['status'], 'incomplete')

    def test_run_hook_is_opt_in_and_correction_failure_keeps_raw_result(self):
        from glacier_offsets.stack_workflow import execute_stack
        cfg = deepcopy(self.cfg)
        cfg['stack']['swath_correction']['enabled'] = False
        with patch('glacier_offsets.stack_workflow._execute_stack', return_value=0) as raw, \
             patch('glacier_offsets.swath_correction.execute_corrections', return_value=1) as correct:
            self.assertEqual(execute_stack(cfg, 'run'), 0)
            correct.assert_not_called()
            self.assertEqual(execute_stack(self.cfg, 'run'), 1)
            correct.assert_called_once()
            self.assertEqual(raw.call_count, 2)

    def test_run_rejects_a_changed_calibration_before_any_upstream_force(self):
        from glacier_offsets.stack_workflow import execute_stack
        self.cal_path.write_text('{}')
        with patch('glacier_offsets.stack_workflow._execute_stack') as raw:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                execute_stack(self.cfg, 'run', force=True)
            raw.assert_not_called()

    def test_output_symlink_cannot_redirect_writes_into_raw_pair(self):
        self.assertEqual(execute_corrections(self.cfg), 0)
        before = self.snapshot()
        shutil.rmtree(self.corrected() / 'geo')
        (self.corrected() / 'geo').symlink_to(self.raw / 'geo', target_is_directory=True)
        self.assertEqual(execute_corrections(self.cfg, force=True), 1)
        self.assertEqual(before, self.snapshot())

    def test_quality_mask_remains_the_production_mask(self):
        self.cfg['geocoding']['mask'] = 'quality'
        self.manifest['geocode_identity'] = identity(geocoding_specification(self.cfg, self.manifest))
        write_json(self.raw / 'manifest.json', self.manifest)
        x, y = Transformer.from_crs(4326, 3031, always_xy=True).transform(self.lon, self.lat)
        mapper = make_mapper(x, y, self.quality, 200, 'nearest', target_grid=self.manifest['map_grid'])
        for path in (self.raw / 'geo').glob('*.tif'):
            values = gdal.Open(str(path)).ReadAsArray()
            values[~mapper.valid] = np.nan
            write_geotiff(path, values, mapper.transform, 3031, {'units': 'fixture'})
        self.assertEqual(execute_corrections(self.cfg), 0)
        result = gdal.Open(str(self.corrected() / 'geo/azimuth_displacement_m.tif')).ReadAsArray()
        np.testing.assert_array_equal(np.isfinite(result), mapper.valid)

    def test_source_metadata_changes_and_mutation_during_read_are_rejected(self):
        from glacier_offsets.swath_correction import Inputs
        from glacier_offsets.products import ChecksumCache
        inputs = Inputs(ChecksumCache())
        inputs.file(self.raw / 'manifest.json')
        data = read(self.raw / 'manifest.json')
        data['assembly']['source_burst_ids'].reverse()
        write_json(self.raw / 'manifest.json', data)
        with self.assertRaisesRegex(ValueError, 'changed between reads'):
            inputs.unchanged()
        self.assertEqual(execute_corrections(self.cfg), 1)

    def test_date_additive_contrasts_cancel_over_a_year_and_skip_same_satellite(self):
        labels = np.array([0, 1, 0, 1, 0])
        contrasts = .026 * np.diff(labels)
        self.assertAlmostEqual(contrasts.sum(), 0)
        # Arbitrary missing intermediate dates still telescope through endpoints.
        self.assertAlmostEqual(.026 * (labels[-1] - labels[1]) + .026 * (labels[1] - labels[0]), 0)


def read(path):
    return json.loads(Path(path).read_text())


if __name__ == '__main__':
    unittest.main()

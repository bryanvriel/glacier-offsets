"""Displacement packages, offline compatibility, and native viewing geometry."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import h5netcdf
import h5py
import numpy as np
from osgeo import gdal
import yaml

import test_production_swath_correction as correction_fixture
from glacier_offsets.cli import main
from glacier_offsets.geocode import write_geotiff
from glacier_offsets.package_geometry import enu_geometry, retain_reference_orbit
from glacier_offsets.packaging import package_stack, validate_package, PRIMARY
from glacier_offsets.products import checksum, identity, write_json
from glacier_offsets.stack_config import load_status_config
from glacier_offsets.stack_lock import stack_lock
from glacier_offsets.swath_correction import execute_corrections


class PackagingTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing physically signed IW1/IW2 correction fixture.
        self.fixture = f = correction_fixture.ProductionSwathCorrectionTests()
        self.addCleanup(f.doCleanups)
        f.setUp()
        self.cfg, self.root, self.raw = f.cfg, f.root, f.raw
        self.cfg['packaging'] = {'source': 'raw', 'output': None}
        self.frame = f.frame
        self.grid = f.manifest['map_grid']
        self.frame.update(layout_identity='layout', map_grid=self.grid)
        epoch = datetime(2017, 1, 1)
        orbit_path = self.root / 'pinned.EOF'
        orbit_path.write_text('<Earth_Explorer_File><List_of_OSVs>' + ''.join(
            '<OSV><UTC>UTC='+ (epoch+timedelta(seconds=i*10-120)).isoformat()+'</UTC>' +
            ''.join(f'<{k}>{v}</{k}>' for k, v in zip(('X','Y','Z','VX','VY','VZ'),
                     (7078137, (i*10-120)*7500, 500000, 0, 7500, 0)))+'</OSV>' for i in range(30)) +
            '</List_of_OSVs></Earth_Explorer_File>')
        self.frame['specification']['inputs'] = {'orbit': {'path': str(orbit_path), 'sha256': checksum(orbit_path)}}
        self.frame['geometry_identity'] = identity(self.frame['specification'])
        f.cal['scope']['geometry_identity'] = self.frame['geometry_identity']
        f.save_cal()
        self.orbit_path = orbit_path
        for j, s in enumerate(self.frame['sources']):
            s.update(geometry_reference_time=(epoch+timedelta(seconds=j)).isoformat(), offset=[0, 0],
                     coreg={'reference_grid': {'length': 64, 'width': 128, 'prf': 10., 'lookside': 'right', 'wavelength': .05}})
        # The package keeps exact unequal burst-start intervals in its timing table.
        m = f.manifest
        m['stack_pair']['geometry_identity'] = self.frame['geometry_identity']
        m['stack_pair']['geometry_reference_date'] = '2017-01-01'
        m['acquisition_times'] = [epoch.isoformat(), (epoch+timedelta(days=6)).isoformat()]
        m['assembly']['sources'] = [dict(s, interval_seconds=6*86400+j+1,
            acquisition_times=[(epoch+timedelta(seconds=j)).isoformat(),
                               (epoch+timedelta(days=6, seconds=2*j+1)).isoformat()]) for j, s in enumerate(self.frame['sources'])]
        m['geocoding'] = {'epsg': 3031, 'spacing_m': 200, 'shape': self.grid['shape'],
                          'transform': self.grid['transform'], 'resampling': 'nearest', 'source_mask': 'input'}
        write_json(self.raw / 'manifest.json', m)
        write_json(self.raw / 'geo/geocoding.json', m['geocoding'])
        with h5py.File(self.raw / 'geo/center_geometry.h5', 'a') as h:
            h.create_dataset('height_m', data=np.zeros(f.total.shape))
        for key, a in [('snr', f.mapper.map(np.full(f.total.shape, 8.))),
                       ('correlation_peak', f.mapper.map(np.full(f.total.shape, np.nan)))]:
            write_geotiff(self.raw / 'geo' / f'{key}.tif', a, f.mapper.transform, 3031, {})
        write_json(self.root / 'geometry/manifest.json', self.frame)
        catalog = json.loads((self.root / 'stack_manifest.json').read_text())
        catalog['map_grid'] = self.grid
        catalog['pairs'].update({'2017-01-07_2017-01-13': {'active': True}, 'inactive': {'active': False}})
        write_json(self.root / 'stack_manifest.json', catalog)
        self.native = patch('glacier_offsets.package_geometry.native_orbit', return_value=object())
        self.native.start(); self.addCleanup(self.native.stop)
        def geometry(lon, lat, height, scale, owner, sources, orbit):
            los = np.broadcast_to([0., .6, .8], lon.shape+(3,))
            return {'reference_los_enu': los, 'range_sensitivity_enu': -los,
                    'azimuth_sensitivity_enu': np.broadcast_to([1., 0., 0.], los.shape)}
        self.geometry = patch('glacier_offsets.package_geometry.enu_geometry', side_effect=geometry)
        self.geometry_mock = self.geometry.start(); self.addCleanup(self.geometry.stop)

    def test_roundtrip_partial_quality_missing_and_offline_resume(self):
        report = package_stack(self.cfg)
        self.assertEqual(report['status'], 'partial')
        self.assertEqual(report['pairs'], 1)
        self.assertEqual(len(report['omitted']), 1)
        path = Path(report['path'])
        with h5netcdf.File(path, 'r') as h:
            self.assertEqual(h['valid_mask'].dtype, np.dtype('uint8'))
            self.assertNotIn('_FillValue', h['valid_mask'].attrs)
            valid = h['valid_mask'][0].astype(bool)
            self.assertTrue(valid.any())
            self.assertTrue(np.isnan(h['correlation_peak'][0]).all())
            for key in PRIMARY:
                original = gdal.Open(str(self.raw / 'geo' / f'{key}.tif')).ReadAsArray()
                actual = h[key][0]
                if key != 'valid_mask':
                    original = np.where(valid, original, np.nan)
                np.testing.assert_array_equal(actual, original)
            np.testing.assert_array_equal(h['pair_interval_seconds'][0], [518401, 518402, 518403])
            self.assertEqual(h.attrs['EPSG'], 3031)
        before = (checksum(path), path.stat().st_mtime_ns)
        self.orbit_path.unlink()
        with patch('glacier_offsets.package_geometry.enu_geometry', side_effect=AssertionError('recomputed geometry')):
            self.assertTrue(package_stack(self.cfg)['reused'])
        self.assertEqual(before, (checksum(path), path.stat().st_mtime_ns))
        copy = self.fixture.folder / 'standalone.nc'
        shutil.copyfile(path, copy)
        with patch('glacier_offsets.packaging.read_json', side_effect=AssertionError('read stack')):
            self.assertEqual(validate_package(copy)['pairs'], 1)

    def test_corrected_selection_legacy_and_tampered_output(self):
        # Original pin covers input bytes; the saved copy is normalized JSON.
        self.fixture.cal_path.write_text(json.dumps(self.fixture.cal, separators=(',', ':')))
        self.cfg['stack']['swath_correction']['calibration_sha256'] = checksum(self.fixture.cal_path)
        self.assertEqual(execute_corrections(self.cfg, pairs=[self.fixture.pair]), 0)
        self.cfg['packaging']['source'] = 'swath_corrected'
        report = package_stack(self.cfg)
        with h5netcdf.File(report['path'], 'r') as h:
            actual = h['azimuth_displacement_m'][0]
            original = gdal.Open(str(self.fixture.corrected() / 'geo/azimuth_displacement_m.tif')).ReadAsArray()
            np.testing.assert_array_equal(actual, original)
            self.assertIn('correction_available', h.variables)
            self.assertIn('calibration_json', h.variables)
        path = self.fixture.corrected() / 'geo/azimuth_displacement_m.tif'
        with path.open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            package_stack(self.cfg)

    def test_duplicate_midpoints_empty_pair_and_sorted_labels(self):
        f = self.fixture
        second = '2016-12-31_2017-01-08'
        folder = self.root / 'pairs' / second
        shutil.copytree(self.raw, folder)
        m = deepcopy(f.manifest)
        m['stack_pair']['acquisition_dates'] = second.split('_')
        m['acquisition_times'] = ['2016-12-31T00:00:00', '2017-01-08T00:00:00']
        for j, source in enumerate(m['assembly']['sources']):
            source['acquisition_times'] = [(datetime(2016, 12, 31)+timedelta(seconds=j)).isoformat(),
                                           (datetime(2017, 1, 8)+timedelta(seconds=2*j+1)).isoformat()]
            source['interval_seconds'] = 8*86400+j+1
        with h5py.File(folder / 'radar/offsets.h5', 'a') as h:
            h['interval_seconds'][...] = 8*86400+self.fixture.owner.astype(float)
        write_json(folder / 'manifest.json', m)
        for key in PRIMARY:
            value = np.zeros(self.grid['shape'], dtype='float32') if key == 'valid_mask' else np.full(self.grid['shape'], np.nan)
            write_geotiff(folder / 'geo' / f'{key}.tif', value, self.grid['transform'], 3031, {})
        catalog = json.loads((self.root / 'stack_manifest.json').read_text())
        catalog['pairs'][second] = {'active': True}
        write_json(self.root / 'stack_manifest.json', catalog)
        report = package_stack(self.cfg)
        with h5netcdf.File(report['path']) as h:
            self.assertEqual(h['time'][0], h['time'][1])
            self.assertEqual(h['pair_id'][:].astype(str).tolist(), [second, f.pair])
            self.assertFalse(h['valid_mask'][0].any())
            self.assertTrue(h['valid_mask'][1].any())

    def test_config_inheritance_cli_and_linear_rejection(self):
        parent = self.fixture.folder / 'parent.yaml'
        parent.write_text(yaml.safe_dump(self.cfg | {'packaging': {'output': 'custom.nc'}}))
        child = self.fixture.folder / 'child.yaml'
        child.write_text('extends: parent.yaml\n')
        cfg = load_status_config(child)
        self.assertEqual(cfg['packaging']['output'], str(self.fixture.folder / 'custom.nc'))
        with patch('glacier_offsets.stack_workflow.execute_stack', side_effect=AssertionError('upstream processing')):
            self.assertEqual(main(['package', str(child)]), 0)
        cfg['geocoding']['resampling'] = 'linear'
        with self.assertRaisesRegex(ValueError, 'nearest-source'):
            package_stack(cfg)

    def test_microsecond_scene_endpoints_survive_storage(self):
        m = deepcopy(self.fixture.manifest)
        m['acquisition_times'] = ['2017-01-01T00:00:00.123456', '2017-01-07T00:00:00.654321']
        write_json(self.raw / 'manifest.json', m)
        report = package_stack(self.cfg)
        with h5netcdf.File(report['path']) as h:
            self.assertAlmostEqual(float(h['earlier_time'][0]), 1483228800.123456, places=6)
            self.assertAlmostEqual(float(h['later_time'][0]), 1483747200.654321, places=6)

    def test_pinned_orbit_snapshot_detects_corruption(self):
        record = self.frame['specification']['inputs']['orbit']
        path = retain_reference_orbit(self.root / 'geometry', record)
        self.orbit_path.unlink()
        self.assertEqual(retain_reference_orbit(self.root / 'geometry', record), path)
        with h5py.File(path, 'a') as h:
            h['position'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'damaged'):
            retain_reference_orbit(self.root / 'geometry', record)

    def test_saved_processing_backend_override_does_not_require_a_yaml_change(self):
        cfg = deepcopy(self.cfg)
        cfg['matching']['backend'] = 'auto'
        with patch('glacier_offsets.environment.select_backend', side_effect=AssertionError('probed native backend')):
            self.assertEqual(package_stack(cfg)['pairs'], 1)

    def test_atomic_failure_lock_and_damaged_package_rebuild(self):
        path = Path(package_stack(self.cfg)['path'])
        before = checksum(path)
        with stack_lock(self.root), self.assertRaisesRegex(ValueError, 'writer'):
            package_stack(self.cfg)
        with patch('glacier_offsets.packaging.write_package', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            package_stack(self.cfg, force=True)
        self.assertEqual(before, checksum(path))
        with h5netcdf.File(path, 'a') as h:
            h['range_displacement_m'][0, 0, 0] = 123.
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_package(path)
        self.assertFalse(package_stack(self.cfg)['reused'])
        validate_package(path)

    def test_stale_and_incompatible_pairs_are_omitted_without_publication(self):
        for kind in ('stale', 'grid'):
            m = deepcopy(self.fixture.manifest)
            if kind == 'stale':
                m['geocode_identity'] = 'stale'
            else:
                m['map_grid'] = self.grid | {'epsg': 3413}
            write_json(self.raw / 'manifest.json', m)
            with self.assertRaisesRegex(ValueError, 'No eligible'):
                package_stack(self.cfg)
        self.assertFalse((self.root / 'packages').exists())


class NativePackageGeometryTests(unittest.TestCase):
    def test_native_enu_projection_signs_and_local_azimuth_scale(self):
        import isce3
        epoch = datetime(2020, 1, 1)
        states = [isce3.core.StateVector(isce3.core.DateTime(epoch+timedelta(seconds=t)),
                    [7078137., 7500*t, 500000.], [0, 7500., 0]) for t in range(-40, 41, 10)]
        orbit = isce3.core.Orbit(states)
        zero = np.zeros((1, 1))
        source = [{'coreg': {'reference_grid': {'lookside': 'right', 'wavelength': .05, 'prf': 100.}}}]
        fields = enu_geometry(zero, zero, zero, zero+10, np.ones((1, 1), 'i4'), source, orbit)
        expected_los = np.array([0., 500000., 700000.])/np.hypot(500000, 700000)
        np.testing.assert_allclose(fields['reference_los_enu'][0, 0], expected_los, atol=1e-10)
        np.testing.assert_allclose(fields['range_sensitivity_enu'][0, 0], -expected_los, atol=1e-6)
        np.testing.assert_allclose(fields['azimuth_sensitivity_enu'][0, 0], [1000/7500, 0, 0], atol=1e-5)

"""Native-geometry scheduling, mask screening, cache integrity and restart."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5netcdf
import h5py
import numpy as np
from numpy.testing import assert_array_equal

from glacier_offsets.nisar_packaging import package_nisar, CUBES, VARIABLES
from test_nisar import enriched, test_geometry


def wide_subset(path):
    enriched(path)
    with h5py.File(path, 'a') as h:
        x = -2198520.+np.arange(257)*80
        for key in ('x', 'radarGrid/xCoordinates'):
            del h[key]; h[key] = x
        for key in ('mask', 'surfaceHeight', *['layer1/'+k for k in VARIABLES]):
            value, attrs = h[key][0, 0], dict(h[key].attrs)
            del h[key]
            h.create_dataset(key, data=np.full((2, 257), value)).attrs.update(attrs)
        h['mask'][0, 0] = 100
        for key in CUBES:
            key = 'radarGrid/'+key
            value, attrs = h[key][0, 0, 0], dict(h[key].attrs)
            del h[key]
            h.create_dataset(key, data=np.full((2, 2, 257), value)).attrs.update(attrs)
    return path


class NisarGeometryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = enriched(self.root/'subset.h5')
        self.output = self.root/'displacements.nc'

    def run_package(self, **kwargs):
        return package_nisar([self.source], output=self.output, workers=1, **kwargs)

    def test_screening_before_solves_and_covariance_independent(self):
        with h5py.File(self.source, 'a') as h:
            h['layer1/alongTrackOffsetVariance'][0, 0] = -1
            h['layer1/slantRangeOffset'][0, 1] = np.nan
            h['surfaceHeight'][1, 1] = np.nan
        with patch('glacier_offsets.nisar_packaging.native_geometry', side_effect=test_geometry) as native:
            result = self.run_package()
        self.assertEqual(len(native.call_args.args[1]), 1)
        self.assertEqual(result['performance']['evaluated_cells'], 1)
        self.assertEqual(result['performance']['skipped_cells'], 3)
        with h5netcdf.File(self.output) as h:
            self.assertEqual(h['valid_mask'][:].sum(), 1)
            self.assertEqual(h['uncertainty_available'][:].sum(), 0)
            self.assertEqual(h['source_slantRangeOffset'][0, 1, 0], -3)
            self.assertTrue(np.isnan(h['range_sensitivity_enu'][0, :, 1, 0]).all())

    def test_empty_tile_never_imports_native_solver(self):
        with h5py.File(self.source, 'a') as h: h['mask'][:] = 255
        with patch('glacier_offsets.nisar_packaging.native_geometry', side_effect=AssertionError('must skip')):
            result = self.run_package()
        self.assertEqual(result['pairs'], 1)
        self.assertEqual(result['performance']['evaluated_cells'], 0)
        with h5netcdf.File(self.output) as h: self.assertEqual(h['valid_mask'][:].sum(), 0)

    def test_cache_reuse_damage_and_source_changes(self):
        cache = self.root/'shared-cache'
        with patch('glacier_offsets.nisar_packaging.native_geometry', side_effect=test_geometry) as native:
            first = self.run_package(geometry_cache=cache)
            second = self.run_package(force=True, geometry_cache=cache)
            self.assertEqual(native.call_count, 1)
            self.assertEqual(second['performance']['cache_hits'], 1)
            self.assertEqual(second['performance']['cached_cells'], 3)
            with h5netcdf.File(self.output) as h: old_identity = h.attrs['package_identity']
            # A second output and a different worker count share scientific identity.
            other = self.root/'other.nc'
            package_nisar([self.source], output=other, workers=3, geometry_cache=cache)
            self.assertEqual(native.call_count, 1)
            with h5netcdf.File(other) as h: self.assertEqual(h.attrs['package_identity'], old_identity)
            entry = next(cache.rglob('*.npz'))
            with np.load(entry) as h: key, values, digest = str(h['key']), h['native'], str(h['sha256'])
            values[0, 0, 0, 0] += .1
            np.savez_compressed(entry, key=key, native=values, sha256=digest)
            with self.assertLogs(level='WARNING') as logs:
                self.run_package(force=True, geometry_cache=cache)
            self.assertIn('damaged NISAR geometry cache', '\n'.join(logs.output))
            self.assertEqual(native.call_count, 2)
            with h5py.File(self.source, 'a') as h: h['surfaceHeight'][0, 0] += 1
            result = self.run_package(force=True, geometry_cache=cache)
            self.assertEqual(native.call_count, 3)
            self.assertEqual(result['performance']['cache_hits'], 0)
            self.assertEqual(first['performance']['evaluated_cells'], 3)

    def test_no_cache_and_cli_options(self):
        with patch('glacier_offsets.nisar_packaging.native_geometry', side_effect=test_geometry) as native:
            self.run_package(geometry_cache=False)
            self.run_package(force=True, geometry_cache=False)
            self.assertEqual(native.call_count, 2)
        self.assertFalse((self.root/'.nisar-geometry-cache').exists())
        from glacier_offsets.packaging import main
        with patch('glacier_offsets.nisar_packaging.package_nisar', return_value={}) as importer:
            self.assertEqual(main(['--nisar-subsets', 'subsets', '-o', 'out.nc', '--workers', '6', '--geometry-cache', 'cache']), 0)
            self.assertEqual(importer.call_args.kwargs['workers'], 6)
            self.assertEqual(importer.call_args.kwargs['geometry_cache'], 'cache')
            self.assertEqual(main(['--nisar-subsets', 'subsets', '-o', 'out.nc', '--no-geometry-cache']), 0)
            self.assertIs(importer.call_args.kwargs['geometry_cache'], False)
        self.assertEqual(main(['stack.yaml', '--workers', '2']), 2)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            package_nisar([self.source], output=self.output, workers=0)

    def test_cache_separates_layer_and_method_and_recovers_truncation(self):
        with h5py.File(self.source, 'a') as h: h.copy('layer1', 'layer2')
        with patch('glacier_offsets.nisar_packaging.native_geometry', side_effect=test_geometry) as native:
            self.run_package()
            entry = next((self.root/'.nisar-geometry-cache').rglob('*.npz'))
            entry.write_bytes(entry.read_bytes()[:25])
            with self.assertLogs(level='WARNING'): self.run_package(force=True)
            self.assertEqual(native.call_count, 2)
            with patch('glacier_offsets.nisar_packaging.METHOD', 'changed-method'):
                self.run_package(force=True)
            self.assertEqual(native.call_count, 3)
            self.run_package(force=True, layer=2)
            self.assertEqual(native.call_count, 4)

    def test_failed_cache_publication_leaves_no_partial_tile(self):
        from glacier_offsets.nisar_geometry import write_tile
        path = self.root/'cache'/'tile.npz'
        with patch('glacier_offsets.nisar_geometry.np.savez_compressed', side_effect=RuntimeError('write failed')):
            with self.assertRaisesRegex(RuntimeError, 'write failed'):
                write_tile(path, 'key', np.zeros((2, 2, 3, 3)))
        self.assertFalse(path.exists())
        self.assertEqual(list(path.parent.iterdir()), [])

    def test_restart_preserves_finished_tiles_and_old_output(self):
        self.source = wide_subset(self.source)
        with patch('glacier_offsets.nisar_packaging.native_geometry', test_geometry): self.run_package()
        previous = self.output.read_bytes()
        cache = self.root/'restart-cache'
        calls = 0
        def interrupted(h, points, cube):
            nonlocal calls
            calls += 1
            if calls == 2: raise RuntimeError('interrupted geometry')
            return test_geometry(h, points, cube)
        with patch('glacier_offsets.nisar_packaging.native_geometry', interrupted):
            with self.assertRaisesRegex(RuntimeError, 'interrupted geometry'):
                self.run_package(force=True, geometry_cache=cache)
        self.assertEqual(previous, self.output.read_bytes())
        self.assertEqual(len(list(cache.rglob('*.npz'))), 1)
        with patch('glacier_offsets.nisar_packaging.native_geometry', side_effect=test_geometry) as native:
            result = self.run_package(force=True, geometry_cache=cache)
        self.assertEqual(native.call_count, 1)
        self.assertEqual(result['performance']['cache_hits'], 1)

    def test_parallel_pipeline_matches_serial_and_logs_progress(self):
        self.source = wide_subset(self.source)
        with patch('glacier_offsets.nisar_packaging.native_geometry', test_geometry):
            self.run_package(geometry_cache=False)
            # Exercise asynchronous scheduling with an instrumentable worker;
            # real spawn/ISCE3 parity is additionally checked on the LAR2 subsets.
            with patch('glacier_offsets.nisar_geometry.ProcessPoolExecutor',
                       side_effect=lambda max_workers, **_: ThreadPoolExecutor(max_workers=max_workers)) as executor:
                with self.assertLogs(level='INFO') as logs:
                    other = self.root/'parallel.nc'
                    result = package_nisar([self.source], output=other, workers=2, geometry_cache=False)
            executor.assert_called_once()
        self.assertEqual(result['performance']['evaluated_cells'], 513)
        with h5netcdf.File(self.output) as a, h5netcdf.File(other) as b:
            self.assertEqual(a.attrs['content_hashes_json'], b.attrs['content_hashes_json'])
            for key in a.variables: assert_array_equal(a[key][()], b[key][()])
        messages = '\n'.join(logs.output)
        for phrase in ('tiles 2/2', 'skipped', 'Hashing completed', 'Validating completed', 'Published NISAR'):
            self.assertIn(phrase, messages)


if __name__ == '__main__':
    unittest.main()

"""Horizontal inversion, QC, optional covariance and standalone publication."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import h5netcdf
import numpy as np

import test_packaging as package_fixture
from test_tides import FakeModel
from glacier_offsets.cli import main
from glacier_offsets.horizontal import COVARIANCE_INPUTS, COVARIANCE_OUTPUTS, to_horizontal, validate_horizontal_stack
from glacier_offsets.packaging import content_hashes, package_stack, variable
from glacier_offsets.products import checksum
from glacier_offsets.products import identity
from glacier_offsets.tides import correct_tides, output_lock


class HorizontalTests(unittest.TestCase):
    def setUp(self):
        f = package_fixture.PackagingTests()
        self.addCleanup(f.doCleanups)
        f.setUp()
        self.folder = f.fixture.folder
        self.source = Path(package_stack(f.cfg)['path'])
        self.output = self.folder/'horizontal.nc'
        with h5netcdf.File(self.source, 'a') as h:
            self.valid = h['valid_mask'][0].astype(bool)
            association = h['source_observation_index'][:]
            self.association = association
            self.selected = np.flatnonzero(self.valid)[0]
            self.source_id = association.ravel()[self.selected]
            self.copied_source = association == self.source_id
            shape = self.valid.shape
            # Signed, oblique and mutually orthogonal horizontal rows: known
            # inverse = [[-1.2, .8], [1.6, .6]], condition number 2.
            self.up = -np.sqrt(.75)
            for key, row in [('range_sensitivity_enu', [-.3, .4, self.up]),
                             ('azimuth_sensitivity_enu', [.8, .6, .002]),
                             ('reference_los_enu', [.3, -.4, -self.up])]:
                h[key][...] = np.broadcast_to(np.array(row)[:, None, None], (3, *shape))
            y, x = np.indices(shape)
            self.east, self.north = 40.+.01*x, -12.+.02*y
            h['range_displacement_m'][0] = np.where(self.valid, -.3*self.east+.4*self.north, np.nan).astype('float32')
            h['azimuth_displacement_m'][0] = np.where(self.valid, .8*self.east+.6*self.north, np.nan).astype('float32')
            self.rehash(h)
        self.original = checksum(self.source)

    @staticmethod
    def rehash(h):
        h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)

    def add_covariance(self, cross=.5, *, complete=True):
        with h5netcdf.File(self.source, 'a') as h:
            for key, value in zip(COVARIANCE_INPUTS, (2., 3., cross)):
                if key.endswith('m2') and not complete:
                    continue
                variable(h, key, ('time', 'y', 'x'), np.where(self.valid, value, np.nan)[None].astype('float32'),
                         units='m2' if key.endswith('m2') else 'm')
            self.rehash(h)

    def test_signed_roundtrip_grid_endpoints_quality_and_integer_masks(self):
        result = to_horizontal(self.source, output=self.output)
        self.assertFalse(result['reused'])
        self.assertEqual(result['horizontal_status'], 'complete')
        self.assertEqual(checksum(self.source), self.original)
        with h5netcdf.File(self.source) as before, h5netcdf.File(self.output) as after:
            for key in before.variables:
                if key != 'provenance_json':
                    np.testing.assert_array_equal(before[key][()], after[key][()], err_msg=key)
            for key, truth in [('east_displacement_m', self.east), ('north_displacement_m', self.north)]:
                actual = after[key][0]
                np.testing.assert_allclose(actual[self.valid], truth[self.valid], atol=5e-6)
                self.assertTrue(np.isnan(actual[~self.valid]).all())
                self.assertEqual(after[key].dtype, np.dtype('float32'))
                self.assertEqual(after[key].attrs['units'], 'm')
                self.assertLessEqual(after[key].chunks[1], 256)
                self.assertEqual(after[key].chunks[0], 1)
            for key in ('horizontal_valid_mask', 'horizontal_geometry_valid_mask', 'horizontal_geometry_flags',
                        'horizontal_uncertainty_available', 'horizontal_validity_flags'):
                self.assertEqual(after[key].dtype, np.dtype('uint8'))
                self.assertNotIn('_FillValue', after[key].attrs)
            np.testing.assert_array_equal(after['horizontal_valid_mask'][0], self.valid)
            self.assertFalse(after['horizontal_uncertainty_available'][:].any())
            self.assertTrue(np.isnan(after['correlation_peak'][:]).all())
            self.assertNotIn('data', after.variables)
            self.assertIn('local geographic east/north', after.attrs['horizontal_vector_basis'])
        # xarray must not decode valid zero-valued integer flags/masks as NaN.
        import xarray as xr
        with xr.open_dataset(self.output, engine='h5netcdf') as ds:
            self.assertEqual(ds.horizontal_valid_mask.dtype, np.dtype('uint8'))
            self.assertEqual(ds.horizontal_geometry_flags.dtype, np.dtype('uint8'))
            np.testing.assert_array_equal(ds.horizontal_valid_mask[0].values, self.valid)
        validate_horizontal_stack(self.output)

    def test_full_covariance_and_invalid_covariance_do_not_remove_observations(self):
        self.add_covariance()
        with h5netcdf.File(self.source, 'a') as h:
            cross = h[COVARIANCE_INPUTS[2]][0]
            cross[self.copied_source] = 99.  # Not positive semidefinite.
            h[COVARIANCE_INPUTS[2]][0] = cross
            self.rehash(h)
        to_horizontal(self.source, output=self.output)
        with h5netcdf.File(self.output) as h:
            good = self.valid & ~self.copied_source
            for key, truth in zip(COVARIANCE_OUTPUTS, (np.sqrt(10.56), 3.8, -3.08)):
                np.testing.assert_allclose(h[key][0][good], truth, rtol=1e-6)
                self.assertTrue(np.isnan(h[key][0][self.copied_source]).all())
            np.testing.assert_array_equal(h['horizontal_valid_mask'][0], self.valid)
            self.assertFalse(h['horizontal_uncertainty_available'][0][self.copied_source].any())

    def test_partial_covariance_does_not_assume_zero_cross_covariance(self):
        self.add_covariance(complete=False)
        to_horizontal(self.source, output=self.output)
        with h5netcdf.File(self.output) as h:
            self.assertNotIn(COVARIANCE_OUTPUTS[0], h.variables)
            self.assertFalse(h['horizontal_uncertainty_available'][:].any())

    def test_geometry_scale_and_singular_matrix_exclusions_preserve_source_mask(self):
        with h5netcdf.File(self.source, 'a') as h:
            scale = h['azimuth_ground_spacing_m_per_slc_line'][:]
            scale[self.copied_source] *= 3
            h['azimuth_ground_spacing_m_per_slc_line'][...] = scale
            other = np.unique(self.association[self.valid & ~self.copied_source])[0]
            self.singular_source = self.association == other
            azimuth = h['azimuth_sensitivity_enu'][:]
            azimuth[:2, self.singular_source] = np.array([-.3, .4])[:, None]
            h['azimuth_sensitivity_enu'][...] = azimuth
            self.rehash(h)
        result = to_horizontal(self.source, output=self.output)
        self.assertEqual(result['horizontal_status'], 'partial')
        with h5netcdf.File(self.output) as h:
            self.assertTrue(((h['horizontal_geometry_flags'][:][self.copied_source] & 4) != 0).all())
            self.assertTrue(((h['horizontal_geometry_flags'][:][self.singular_source] & 2) != 0).all())
            excluded = self.copied_source | self.singular_source
            self.assertTrue(np.isnan(h['east_displacement_m'][0][excluded]).all())
            np.testing.assert_array_equal(h['valid_mask'][0], self.valid)

    def test_range_tide_input_keeps_azimuth_unchanged_and_excludes_unavailable_tides(self):
        model = self.folder/'model.nc'
        model.write_bytes(b'fixture')
        tidal = self.folder/'tidal.nc'
        with patch('glacier_offsets.tides.Cats2008', FakeModel):
            correct_tides(self.source, model, output=tidal)
        to_horizontal(tidal, output=self.output)
        with h5netcdf.File(tidal) as before, h5netcdf.File(self.output) as h:
            available = before['tidal_correction_available'][0] == 1
            np.testing.assert_array_equal(h['horizontal_valid_mask'][0], self.valid & available)
            np.testing.assert_array_equal(h['azimuth_displacement_m'][:], before['azimuth_displacement_m'][:])
            r, a = before['range_displacement_m'][0], before['azimuth_displacement_m'][0]
            np.testing.assert_allclose(h['east_displacement_m'][0][available], (-1.2*r+.8*a)[available], atol=1e-5)
            np.testing.assert_allclose(h['north_displacement_m'][0][available], (1.6*r+.6*a)[available], atol=1e-5)
            self.assertIn('disabled', h.attrs['horizontal_azimuth_tidal_correction'])
            self.assertTrue(np.isnan(h['east_displacement_m'][0][~available]).all())
        with self.assertRaisesRegex(ValueError, 'parent before horizontal'):
            correct_tides(self.output, model, output=self.folder/'wrong-order.nc')

    def test_empty_completed_observations_are_retained(self):
        with h5netcdf.File(self.source, 'a') as h:
            h['valid_mask'][...] = np.zeros(h['valid_mask'].shape, dtype='uint8')
            for key in ('range_displacement_m', 'azimuth_displacement_m', 'snr', 'correlation_peak'):
                h[key][...] = np.full(h[key].shape, np.nan, dtype='float32')
            self.rehash(h)
        result = to_horizontal(self.source, output=self.output)
        self.assertEqual(result['horizontal_status'], 'empty')
        with h5netcdf.File(self.output) as h:
            self.assertEqual(len(h.dimensions['time']), 1)
            self.assertTrue(np.isnan(h['east_displacement_m'][:]).all())

    def test_duplicate_midpoints_unequal_burst_intervals_and_empty_slice_survive(self):
        import h5py
        doubled = self.folder/'duplicate-midpoints.nc'
        with h5netcdf.File(self.source) as a, h5netcdf.File(doubled, 'w') as b:
            b.dimensions = {key: 2 if key == 'time' else len(value) for key, value in a.dimensions.items()}
            b.attrs.update(dict(a.attrs))
            for key, source in a.variables.items():
                dtype = h5py.string_dtype('utf-8') if source.dtype.kind == 'O' else source.dtype
                target = b.create_variable(key, source.dimensions, dtype=dtype, fillvalue=source.attrs.get('_FillValue'))
                target.attrs.update({k: v for k, v in source.attrs.items() if k != '_FillValue'})
                values = source[()]
                if source.dimensions and source.dimensions[0] == 'time':
                    values = np.concatenate((values, values), axis=0)
                target[...] = values
            b['pair_id'][...] = ['pair-a', 'pair-b']
            b['names'][...] = ['pair-a', 'pair-b']
            # Same midpoint, a different pair duration and microsecond endpoints.
            for key, delta in [('earlier_time', -86400.123456), ('later_time', 86400.123456),
                               ('earlier_burst_time', -86400.123456), ('later_burst_time', 86400.123456)]:
                b[key][1] = b[key][1]+delta
            b['time_bounds'][1] = [b['earlier_time'][1], b['later_time'][1]]
            b['pair_interval_seconds'][1] = b['later_burst_time'][1]-b['earlier_burst_time'][1]
            b['valid_mask'][1] = np.zeros(self.valid.shape, dtype='uint8')
            for key in ('range_displacement_m', 'azimuth_displacement_m', 'snr', 'correlation_peak'):
                b[key][1] = np.full(self.valid.shape, np.nan, dtype='float32')
            spec = json.loads(b['provenance_json'][()].decode())
            spec['pairs'] = ['pair-a', 'pair-b']
            b['provenance_json'][()] = json.dumps(spec, sort_keys=True)
            b.attrs['package_identity'] = identity(spec)
            self.rehash(b)
        to_horizontal(doubled, output=self.output)
        with h5netcdf.File(doubled) as before, h5netcdf.File(self.output) as after:
            self.assertEqual(after['time'][0], after['time'][1])
            self.assertEqual(after['pair_id'][:].astype(str).tolist(), ['pair-a', 'pair-b'])
            for key in ('earlier_time', 'later_time', 'earlier_burst_time', 'later_burst_time', 'pair_interval_seconds'):
                np.testing.assert_array_equal(before[key][:], after[key][:])
            self.assertTrue(after['horizontal_valid_mask'][0].any())
            self.assertFalse(after['horizontal_valid_mask'][1].any())

    def test_offline_reuse_force_and_recipe_changes(self):
        to_horizontal(self.source, output=self.output)
        before = self.output.stat().st_mtime_ns
        with patch('glacier_offsets.inputs.discover', side_effect=AssertionError('upstream access')):
            self.assertTrue(to_horizontal(self.source, output=self.output)['reused'])
        self.assertEqual(before, self.output.stat().st_mtime_ns)
        with self.assertRaisesRegex(ValueError, 'different conversion recipe'):
            to_horizontal(self.source, output=self.output, max_condition=4)
        self.assertFalse(to_horizontal(self.source, output=self.output, max_condition=4, force=True)['reused'])
        with self.assertRaisesRegex(ValueError, 'already has horizontal'):
            to_horizontal(self.output, output=self.folder/'twice.nc', force=True)
        with h5netcdf.File(self.source, 'a') as h:
            a = h['range_displacement_m'][0]
            a[self.valid] += 1
            h['range_displacement_m'][0] = a
            self.rehash(h)
        with self.assertRaisesRegex(ValueError, 'different conversion recipe'):
            to_horizontal(self.source, output=self.output)
        to_horizontal(self.source, output=self.output, force=True)

    def test_input_and_output_tampering_is_detected(self):
        to_horizontal(self.source, output=self.output)
        with h5netcdf.File(self.output, 'a') as h:
            a = h['east_displacement_m'][0]
            a[self.valid] += 3
            h['east_displacement_m'][0] = a
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_horizontal_stack(self.output)
        with h5netcdf.File(self.output, 'a') as h:
            self.rehash(h)
        with self.assertRaises(AssertionError):
            validate_horizontal_stack(self.output)
        with h5netcdf.File(self.source, 'a') as h:
            a = h['range_displacement_m'][0]
            a[self.valid] += 1
            h['range_displacement_m'][0] = a
        with self.assertRaisesRegex(ValueError, 'checksum'):
            to_horizontal(self.source, output=self.folder/'bad-input.nc')

    def test_no_convertible_data_interruption_input_change_and_writer_preserve_output(self):
        to_horizontal(self.source, output=self.output)
        before = checksum(self.output)
        with output_lock(self.folder), self.assertRaisesRegex(ValueError, 'writer'):
            to_horizontal(self.source, output=self.output, force=True)
        with patch('glacier_offsets.horizontal.os.replace', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            to_horizontal(self.source, output=self.output, force=True)
        self.assertEqual(checksum(self.output), before)
        from glacier_offsets.horizontal import apply_horizontal
        def changed(*args):
            result = apply_horizontal(*args)
            with self.source.open('ab') as stream:
                stream.write(b'changed')
            return result
        with patch('glacier_offsets.horizontal.apply_horizontal', side_effect=changed), self.assertRaisesRegex(ValueError, 'changed between'):
            to_horizontal(self.source, output=self.output, force=True)
        self.assertEqual(checksum(self.output), before)
        with self.assertRaisesRegex(ValueError, 'No usable horizontal'):
            to_horizontal(self.source, output=self.output, max_condition=1.1, force=True)
        self.assertEqual(checksum(self.output), before)

    def test_parameters_units_aliases_cli_and_unrelated_outputs(self):
        self.assertEqual(main(['to-horizontal', str(self.source), '-o', str(self.output)]), 0)
        for kwargs in ({'max_condition': .5}, {'max_condition': float('nan')}, {'max_azimuth_scale_ratio': float('inf')},
                       {'max_azimuth_scale_ratio': .5}, {'vertical_motion': 'surface_parallel'}):
            with self.assertRaises(ValueError):
                to_horizontal(self.source, **kwargs)
        with self.assertRaisesRegex(ValueError, 'separate output'):
            to_horizontal(self.source, output=self.source, force=True)
        alias = self.folder/'alias.nc'
        os.link(self.source, alias)
        with self.assertRaisesRegex(ValueError, 'separate output'):
            to_horizontal(self.source, output=alias, force=True)
        with self.assertRaisesRegex(ValueError, 'unrelated output'):
            self.unrelated_output()
        with h5netcdf.File(self.source, 'a') as h:
            h['range_displacement_m'].attrs['units'] = 'pixels'
        with self.assertRaisesRegex(ValueError, 'must use metres'):
            to_horizontal(self.source, output=self.folder/'bad-units.nc')

    def unrelated_output(self):
        path = self.folder/'unrelated.nc'
        with h5netcdf.File(path, 'w'):
            pass
        to_horizontal(self.source, output=path, force=True)


if __name__ == '__main__':
    unittest.main()

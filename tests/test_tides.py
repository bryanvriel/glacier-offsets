"""Tidal projection, standalone stack processing, and pyTMD parity."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5netcdf
import numpy as np

import test_packaging as package_fixture
from glacier_offsets.cli import main
from glacier_offsets.packaging import content_hashes, package_stack
from glacier_offsets.products import checksum
from glacier_offsets.tides import correct_tides, output_lock, validate_tidal_stack


class FakeModel:
    calls = []

    def __init__(self, path, longitude, latitude, *, response):
        self.version = '3.0.9'
        self.metadata = {'test': True, 'response': response}
        self.response_factor = np.ones(len(longitude))
        self.response_factor[:3] = 0.
        self.response_factor[-1] = 1.08
        self.available = np.ones(len(longitude), dtype=bool)
        self.available[1] = False
        self.ocean_mask = np.ones(len(longitude))
        self.ocean_mask[0] = 0.

    def predict(self, indices, seconds):
        self.calls.append(float(seconds))
        # Deliberately depends on exact burst endpoints, not pair midpoint.
        value = ((seconds-1483228800.)/86400.)*self.response_factor[indices]
        return np.where(self.available[indices], value, np.nan)


class TidalStackTests(unittest.TestCase):
    def setUp(self):
        f = package_fixture.PackagingTests()
        self.addCleanup(f.doCleanups)
        f.setUp()
        self.folder = f.fixture.folder
        self.source = Path(package_stack(f.cfg)['path'])
        self.original = checksum(self.source)
        self.model = self.folder / 'CATS2008_v2023.nc'
        self.model.write_bytes(b'model-fixture')
        self.output = self.folder / 'corrected.nc'
        self.mock = patch('glacier_offsets.tides.Cats2008', FakeModel)
        self.mock.start(); self.addCleanup(self.mock.stop)
        FakeModel.calls = []

    def test_vertical_projection_sign_slant_basis_grounding_and_missing_model(self):
        report = correct_tides(self.source, self.model, output=self.output)
        self.assertGreater(report['counts']['corrected_cells'], 0)
        self.assertGreater(report['counts']['uncorrected_valid_cells'], 0)
        self.assertGreater(report['counts']['zero_response_cells'], 0)
        with h5netcdf.File(self.source) as before, h5netcdf.File(self.output) as after:
            raw, corrected = before['range_displacement_m'][0], after['range_displacement_m'][0]
            available = after['tidal_correction_available'][0].astype(bool)
            delta = after['tidal_vertical_displacement_m'][0]
            bias = after['tidal_range_bias_m'][0]
            # Upward tide decreases range: removing it ADDS z*cos(theta).
            np.testing.assert_allclose(bias[available], -.8*delta[available], rtol=1e-6)
            np.testing.assert_allclose(corrected[available], raw[available]+.8*delta[available], rtol=1e-6)
            np.testing.assert_array_equal(raw[~available], corrected[~available])
            ground = (after['tidal_response_factor'][:] == 0) & available
            np.testing.assert_array_equal(raw[ground], corrected[ground])
            for key in ('azimuth_displacement_m', 'valid_mask', 'snr', 'correlation_peak', 'reference_los_enu', 'earlier_burst_time'):
                np.testing.assert_array_equal(after[key][:], before[key][:])
            self.assertEqual(after['valid_mask'].dtype, np.dtype('uint8'))
            self.assertNotIn('_FillValue', after['tidal_correction_available'].attrs)
            self.assertEqual(after.attrs['range_basis'].split(';')[0], 'slant range')
            self.assertEqual(len(json.loads(after['provenance_json'][()].decode())['postprocessing']), 1)
        self.assertEqual(self.original, checksum(self.source))
        self.assertIn(1483228801., FakeModel.calls)
        self.assertIn(1483747205., FakeModel.calls)
        validate_tidal_stack(self.output)

    def test_offline_input_restart_and_double_correction_guard(self):
        with patch('glacier_offsets.packaging.inspect_pair', side_effect=AssertionError('opened stack outputs')):
            self.assertFalse(correct_tides(self.source, self.model, output=self.output)['reused'])
        revision = self.output.stat().st_mtime_ns
        self.assertTrue(correct_tides(self.source, self.model, output=self.output)['reused'])
        self.assertEqual(revision, self.output.stat().st_mtime_ns)
        with self.assertRaisesRegex(ValueError, 'already has'):
            correct_tides(self.output, self.model, output=self.folder/'double.nc', force=True)
        with self.assertRaisesRegex(ValueError, 'separate'):
            correct_tides(self.source, self.model, output=self.source, force=True)
        with self.assertRaisesRegex(ValueError, 'different correction'):
            correct_tides(self.source, self.model, output=self.output, response='floating')

    def test_failures_leave_source_and_previous_output_untouched(self):
        correct_tides(self.source, self.model, output=self.output)
        old = checksum(self.output)
        with output_lock(self.folder), self.assertRaisesRegex(ValueError, 'writer'):
            correct_tides(self.source, self.model, output=self.output)
        with patch('glacier_offsets.tides.apply_tides', side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            correct_tides(self.source, self.model, output=self.output, force=True)
        self.assertEqual(old, checksum(self.output))
        self.assertEqual(self.original, checksum(self.source))
        with patch('glacier_offsets.tides.apply_tides', side_effect=RuntimeError('bad model')), self.assertRaises(RuntimeError):
            correct_tides(self.source, self.model, output=self.output, force=True)
        self.assertEqual(old, checksum(self.output))

    def test_cli_and_correction_field_integrity(self):
        self.assertEqual(main(['correct-tides', str(self.source), '--model', str(self.model), '-o', str(self.output)]), 0)
        with h5netcdf.File(self.output, 'a') as h:
            h['tidal_vertical_displacement_m'][0, 1, 1] = 123.
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_tidal_stack(self.output)

    def test_model_revision_change_during_processing_never_publishes(self):
        from glacier_offsets.tides import apply_tides
        def changed(*args, **kwargs):
            result = apply_tides(*args, **kwargs)
            self.model.write_bytes(b'changed model')
            return result
        with patch('glacier_offsets.tides.apply_tides', side_effect=changed), self.assertRaisesRegex(ValueError, 'changed between'):
            correct_tides(self.source, self.model, output=self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.original, checksum(self.source))


class CatsModelTests(unittest.TestCase):
    def setUp(self):
        try:
            import pyTMD
            import xarray as xr
        except ImportError:
            self.skipTest('optional tidal dependencies are not installed')
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'CATS2008_v2023.nc'
        names = 'm2 s2 n2 k2 k1 o1 p1 q1 mf mm'.split()
        y, x = np.meshgrid(np.arange(4.), np.arange(5.), indexing='ij')
        real = np.stack([(j+1)*.01+x*.002+y*.003 for j in range(10)])
        imag = real*.4
        ds = xr.Dataset({'hRe': (('constituents', 'y', 'x'), real), 'hIm': (('constituents', 'y', 'x'), imag),
                         'mask': (('y','x'), np.ones_like(x)), 'wct': (('y','x'), np.ones_like(x)*100),
                         'flexure': (('y','x'), np.broadcast_to(np.linspace(0, 110, 5), x.shape)),
                         'mapping': ((), 0)},
                        coords={'x':np.arange(5.), 'y':np.arange(4.), 'constituents':np.arange(10)}, attrs={'Title':'CATS2008_v2023'})
        ds['mapping'].attrs['spatial_proj4'] = '+proj=longlat +datum=WGS84 +type=crs'
        ds['constituents'].attrs['constituent_order'] = ' '.join(names)
        for k in ['hRe','hIm']: ds[k].attrs['units']='m'
        ds.to_netcdf(self.path, engine='h5netcdf', encoding={k:{'dtype':'int16','scale_factor':.0001,'_FillValue':-32767} for k in ['hRe','hIm']})

    def test_prediction_matches_public_pytmd_compute_and_complex_scaling(self):
        from glacier_offsets.tide_model import Cats2008
        import pyTMD
        from datetime import datetime, timezone
        lon, lat = np.array([0., 1.2, 3.8]), np.array([1., 1.5, 2.1])
        epoch = 1483228800.123456
        model = Cats2008(self.path, lon, lat)
        definition = {'format':'TMD3', 'name':'CATS2008-v2023', 'projection':'+proj=longlat +datum=WGS84 +type=crs',
                      'z':{'model_file':str(self.path),'units':'m','variable':'tide_ocean'}}
        # A tiny independent standard model driver provides the parity oracle.
        times = np.full(3, np.datetime64(datetime.fromtimestamp(epoch, timezone.utc).replace(tzinfo=None)))
        expected = pyTMD.compute.tide_elevations(lon, lat, times, directory=None,
            definition_file=io.StringIO(json.dumps(definition)), standard='datetime', type='trajectory', apply_flexure=True)
        np.testing.assert_allclose(model.predict(np.arange(3), epoch), expected.values.reshape(-1), atol=1e-9)
        self.assertEqual(model.response_factor[0], 0.)
        self.assertGreater(model.response_factor[-1], 1.)
        self.assertTrue(model.available.all())

    def test_outside_domain_remains_unavailable_and_missing_flexure_fails(self):
        from glacier_offsets.tide_model import Cats2008
        model = Cats2008(self.path, np.array([100.]), np.array([40.]))
        self.assertFalse(model.available.any())
        self.assertTrue(np.isnan(model.predict(np.array([0]), 1483228800.)).all())
        import xarray as xr
        with xr.open_dataset(self.path) as h:
            ds = h.drop_vars('flexure').load()
        other = self.path.with_name('no-flexure.nc')
        ds.to_netcdf(other, engine='h5netcdf')
        with self.assertRaisesRegex(ValueError, 'flexure field'):
            Cats2008(other, np.array([1.]), np.array([1.]))

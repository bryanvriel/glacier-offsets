"""Mission-aware control-box screening from corrected horizontal packages."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5netcdf
import h5py
import numpy as np

from glacier_offsets.packaging import content_hashes
from test_nisar import synthetic, corrected

spec = importlib.util.spec_from_file_location('grounded_controls', Path(__file__).resolve().parents[1]/'tools/build_grounded_control_regions.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class GroundedControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def nisar(self, n=24):
        path = corrected(synthetic(self.root/f'raw-{n}.nc', n=n, tracks=(75,)))
        with h5netcdf.File(path, 'a') as h:
            h['tidal_response_factor'][:] = 0
            h['tidal_model_ocean_mask'][:] = 0
            # One location has a one-day interval, others have 24 days.
            # Scene labels still span 12 days, deliberately testing cell times.
            early = h['earlier_observation_time'][:]
            dt = np.broadcast_to([[1., 24.], [24., 24.]], early.shape)
            h['later_observation_time'][:] = early+dt*86400
            h['east_displacement_m'][:] = dt*.02
            h['north_displacement_m'][:] = dt*.01
            h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
        return path

    def build(self, path, verify=False):
        output = self.root/'regions.json'
        builder.main(argparse.Namespace(package=path, output=output, cell=3000., min_supported=2, verify_package=verify))
        return json.loads(output.read_text())

    def test_nisar_exact_intervals_and_map_support(self):
        result = self.build(self.nisar())
        self.assertEqual(len(result['regions']), 1)
        region = result['regions'][0]
        self.assertEqual(region['sources'], 4)
        self.assertEqual(region['supported_sources'], 4)
        self.assertAlmostEqual(region['median_speed_m_day'], np.hypot(.02, .01), places=7)
        self.assertEqual(result['provenance']['observation_support'], 'geocoded_cell')

    def test_screening_does_not_read_geometry_or_validate_unused_arrays(self):
        path = self.nisar()
        reads = []
        original = h5py.Dataset.__getitem__
        def recorded(dataset, key):
            reads.append(dataset.name)
            return original(dataset, key)
        with patch.object(h5py.Dataset, '__getitem__', recorded), patch(
                'glacier_offsets.displacement.validate_nisar', side_effect=AssertionError('full validation is opt-in')):
            self.assertEqual(len(self.build(path)['regions']), 1)
        self.assertFalse(any('sensitivity' in key or key.startswith('/source_') for key in reads))
        with patch('glacier_offsets.displacement.validate_nisar') as verify:
            verified = self.build(path, verify=True)
            verify.assert_called_once_with(path)
            self.assertEqual(verified['provenance']['array_verification'], 'full')
        # Non-grounded boxes need only coverage, not endpoint or EN time series.
        with h5netcdf.File(path, 'a') as h: h['tidal_response_factor'][:] = 1
        reads.clear()
        with patch.object(h5py.Dataset, '__getitem__', recorded):
            self.assertEqual(self.build(path)['regions'], [])
        self.assertNotIn('/earlier_observation_time', reads)
        self.assertNotIn('/east_displacement_m', reads)

    def test_fast_samples_match_fitting_reader_with_overlap_fallback(self):
        from glacier_offsets.nisar_collection import NisarCollection
        path = self.nisar(n=48)
        with h5netcdf.File(path, 'a') as h:
            for key in ('reference_absolute_orbit', 'secondary_absolute_orbit'):
                h[key][24:36] = h[key][:12]
            h['frame_number'][24:36] = 145
            h['valid_mask'][0, 0, 0] = 0
            for key in ('range_displacement_m', 'azimuth_displacement_m', 'snr', 'correlation_peak'):
                h[key][0, 0, 0] = np.nan
            h['horizontal_valid_mask'][0, 0, 0] = 0
            h['correlation_peak'][1, 0, 1] = 0
            h['tidal_correction_available'][2, 1, 0] = 0
            h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
        with NisarCollection([path]) as collection:
            tile = collection.tile((slice(0, 2), slice(0, 2)))
            good = tile['valid'] & (tile['peak'] >= .1)
            expected = good.sum(axis=0)
        with h5py.File(path) as h:
            row, col, support, speed = builder.nisar_samples(h, 3000., 2)
        np.testing.assert_array_equal(support, expected[row, col])
        np.testing.assert_allclose(speed, np.hypot(.02, .01), atol=1e-8)

    def test_frame_overlaps_do_not_inflate_temporal_support(self):
        path = self.nisar()
        with h5netcdf.File(path, 'a') as h:
            for key in ('reference_absolute_orbit', 'secondary_absolute_orbit'):
                h[key][12:] = h[key][:12]
            h['frame_number'][12:] = 145
            h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
        self.assertEqual(self.build(path)['regions'], [])

    def test_chunk_aligned_reads_across_multiple_tiles(self):
        path = self.root/'chunked.nc'
        with h5py.File(path, 'w') as h:
            h.attrs.update(schema_version='glacier-offsets-displacements-v2', mission='NISAR',
                           package_status='complete', package_identity=builder.fingerprint({}), EPSG=3031)
            h['provenance_json'] = b'{}'
            h['tidal_correction_provenance_json'] = b'{}'
            h['horizontal_conversion_provenance_json'] = b'{}'
            h['x'] = 40.+np.arange(512)*80; h['y'] = [120., 40.]
            h['pair_id'] = np.array([f'pair_{i}'.encode() for i in range(24)])
            for key, values in [('frame_number', np.full(24, 144)), ('track_number', np.full(24, 75)),
                                ('reference_absolute_orbit', np.arange(24)), ('secondary_absolute_orbit', np.arange(24)+100)]:
                h[key] = values
            for key in ('tidal_response_factor', 'tidal_model_ocean_mask'): h[key] = np.zeros((2, 512))
            for key, value in [('valid_mask', 1), ('horizontal_valid_mask', 1), ('tidal_correction_available', 1),
                               ('correlation_peak', .8), ('east_displacement_m', .24), ('north_displacement_m', .12),
                               ('earlier_observation_time', 1700000000.), ('later_observation_time', 1700000000.+12*86400)]:
                ds = h.create_dataset(key, data=np.full((24, 2, 512), value), chunks=(1, 2, 256), compression='gzip')
                if key.endswith('observation_time'): ds.attrs['units'] = 'seconds since 1970-01-01 00:00:00'
        original = h5py.Dataset.__getitem__
        reads = []
        def aligned(ds, key):
            if ds.ndim == 3:
                reads.append(ds.name)
                self.assertEqual(key[-1].start % 256, 0)
                self.assertEqual(key[-1].stop-key[-1].start, 256)
            return original(ds, key)
        with h5py.File(path) as h, patch.object(h5py.Dataset, '__getitem__', aligned):
            row, col, support, speed = builder.nisar_samples(h, 3000., 2)
        self.assertEqual(len(row), 1024)
        np.testing.assert_array_equal(support, 24)
        np.testing.assert_allclose(speed, np.hypot(.02, .01))
        self.assertEqual(reads.count('/east_displacement_m'), 2)

    def test_short_record_and_grounding_rules(self):
        self.assertEqual(self.build(self.nisar(n=12))['regions'], [])
        path = self.nisar()
        with h5netcdf.File(path, 'a') as h:
            h['tidal_model_ocean_mask'][0, 0] = np.nan
            h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
        self.assertEqual(self.build(path)['regions'], [])

    def test_missing_horizontal_fields_report_action(self):
        path = synthetic(self.root/'raw.nc', n=4)
        with self.assertRaisesRegex(ValueError, 'tide-corrected horizontal'):
            self.build(path)

    def test_legacy_source_deduplication(self):
        path = self.root/'legacy.nc'
        with h5py.File(path, 'w') as h:
            h.attrs['EPSG'] = 3031
            h['source_observation_index'] = [[0, 0], [1, 2]]
            h['x'] = [40., 120.]; h['y'] = [120., 40.]
            h['tidal_response_factor'] = np.zeros((2, 2)); h['tidal_model_ocean_mask'] = np.zeros((2, 2))
            h['earlier_time'] = np.arange(24)*86400.; h['later_time'] = (np.arange(24)+12)*86400.
            for key, value in [('east_displacement_m', .24), ('north_displacement_m', .12),
                               ('horizontal_valid_mask', 1), ('correlation_peak', .8)]:
                h[key] = np.full((24, 2, 2), value)
        result = self.build(path)
        self.assertEqual(len(result['regions']), 1)
        self.assertEqual(result['regions'][0]['sources'], 3)
        self.assertEqual(result['provenance']['observation_support'], 'native_source')
        self.assertEqual(result['provenance']['package_sha256'], builder.digest(path))


if __name__ == '__main__':
    unittest.main()

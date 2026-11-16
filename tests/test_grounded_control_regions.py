"""Mission-aware control-box screening from corrected horizontal packages."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

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

    def build(self, path):
        output = self.root/'regions.json'
        builder.main(argparse.Namespace(package=path, output=output, cell=3000., min_supported=2))
        return json.loads(output.read_text())

    def test_nisar_exact_intervals_and_map_support(self):
        result = self.build(self.nisar())
        self.assertEqual(len(result['regions']), 1)
        region = result['regions'][0]
        self.assertEqual(region['sources'], 4)
        self.assertEqual(region['supported_sources'], 4)
        self.assertAlmostEqual(region['median_speed_m_day'], np.hypot(.02, .01), places=7)
        self.assertEqual(result['provenance']['observation_support'], 'geocoded_cell')

    def test_frame_overlaps_do_not_inflate_temporal_support(self):
        path = self.nisar()
        with h5netcdf.File(path, 'a') as h:
            for key in ('reference_absolute_orbit', 'secondary_absolute_orbit'):
                h[key][12:] = h[key][:12]
            h['frame_number'][12:] = 145
            h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
        self.assertEqual(self.build(path)['regions'], [])

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


if __name__ == '__main__':
    unittest.main()

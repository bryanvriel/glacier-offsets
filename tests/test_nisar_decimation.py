"""Integer GOFF sampling retains observation values and downstream geometry."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5netcdf
import h5py
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from glacier_offsets.nisar_packaging import package_nisar, VARIABLES, CUBES
from glacier_offsets.nisar_grid import axis_selection, sampling_record
from glacier_offsets.displacement import validate_nisar
from glacier_offsets.packaging import content_hashes
from glacier_offsets.products import identity
from test_nisar import enriched, test_geometry, synthetic, corrected, ZeroTide


def subset(path, frame=144, shift=0):
    enriched(path, frame=frame, xshift=shift)
    with h5py.File(path, 'a') as h:
        x = h['x'][0]+80.*np.arange(8); y = h['y'][0]-80.*np.arange(8)
        for key, values in [('x', x), ('y', y), ('radarGrid/xCoordinates', x[[0, -1]]), ('radarGrid/yCoordinates', y[[0, -1]])]:
            del h[key]; h[key] = values
        ids = np.arange(64).reshape(8, 8)
        for key in ('mask', 'surfaceHeight', *['layer1/'+k for k in VARIABLES]):
            value, attrs, dtype = h[key][0, 0], dict(h[key].attrs), h[key].dtype
            del h[key]
            a = np.full((8, 8), value, dtype=dtype)
            if key in ('surfaceHeight', 'layer1/alongTrackOffset', 'layer1/slantRangeOffset'): a += ids*.01
            h.create_dataset(key, data=a).attrs.update(attrs)
        h['mask'][2, 2] = 100
        h['layer1/alongTrackOffsetVariance'][5, 5] = -1
        for key in ('referenceZeroDopplerAzimuthTime', 'secondaryZeroDopplerAzimuthTime'):
            h['radarGrid/'+key][:] += np.arange(8).reshape(2, 2, 2)*.01
    return path


def coarse_synthetic(path, factor=3):
    """Known physical forward model on a declared coarse analysis lattice."""
    from pyproj import Transformer
    with h5netcdf.File(path, 'a') as h:
        step = 80.*factor
        x = 40+step*np.ceil((h['x'][0]-40)/step)+np.arange(2)*step
        y = 40+step*np.floor((h['y'][0]-40)/step)-np.arange(2)*step
        h['x'][:] = x; h['y'][:] = y
        xx, yy = np.meshgrid(x, y)
        lon, lat = Transformer.from_crs(3031, 4326, always_xy=True).transform(xx, yy)
        h['observation_longitude'][:] = lon; h['observation_latitude'][:] = lat
        spec = json.loads(h['provenance_json'][()].decode())
        spec['grid']['transform'] = [x[0]-step/2, step, 0., y[0]+step/2, 0., -step]
        spec['sampling'] = sampling_record(factor)
        h['provenance_json'][()] = json.dumps(spec)
        h.attrs['package_identity'] = identity(spec)
        h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
    return path


class DecimationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_retained_values_geometry_and_cache_identity(self):
        source = subset(self.root/'input.h5')
        with patch('glacier_offsets.nisar_packaging.native_geometry', side_effect=test_geometry) as native:
            base = self.root/'native.nc'; out = self.root/'coarse.nc'
            package_nisar([source], output=base, workers=1)
            report = package_nisar([source], output=out, decimation=3, workers=1)
            self.assertEqual(report['shape'], [1, 2, 2])
            self.assertEqual(len(native.call_args.args[1]), 3)  # fourth point has invalid source mask
            report = package_nisar([source], output=out, decimation=3, workers=1, force=True)
            self.assertEqual(report['performance']['cache_hits'], 1)
            self.assertEqual(native.call_count, 2)
        with h5netcdf.File(base) as a, h5netcdf.File(out) as b:
            sx, sy = axis_selection(a['x'][:], 3), axis_selection(a['y'][:], 3)
            for key in a.variables:
                dims = a[key].dimensions
                if dims[-2:] == ('y', 'x'):
                    assert_array_equal(b[key][:], a[key][(..., sy, sx)])
            self.assertEqual(b.attrs['map_posting_m'], 240.)
            spec = json.loads(b['provenance_json'][()].decode())
            self.assertEqual(spec['sampling'], sampling_record(3))
        validate_nisar(out)
        with self.assertRaisesRegex(ValueError, 'differs'):
            package_nisar([source], output=out, decimation=2, workers=1)

    def test_common_lattice_shifted_files_and_mixed_grid_rejection(self):
        from glacier_offsets.nisar_collection import NisarCollection
        a = subset(self.root/'a.h5')
        b = subset(self.root/'b.h5', frame=145, shift=80)
        with patch('glacier_offsets.nisar_packaging.native_geometry', test_geometry):
            pa, pb, native = self.root/'a.nc', self.root/'b.nc', self.root/'native.nc'
            package_nisar([a], output=pa, decimation=3, workers=1)
            package_nisar([b], output=pb, decimation=3, workers=1)
            package_nisar([a], output=native, workers=1)
        # Raw inputs remain unchanged; each collection consumes corrected copies.
        with patch('glacier_offsets.tides.Cats2008', ZeroTide):
            from glacier_offsets.tides import correct_tides
            from glacier_offsets.horizontal import to_horizontal
            model = self.root/'model.nc'; model.write_bytes(b'model')
            paths = [to_horizontal(correct_tides(p, model)['path'])['path'] for p in (pa, pb, native)]
        with NisarCollection(paths[:2]) as c:
            self.assertEqual(c.spacing, 240.)
            assert_array_equal(np.diff(c.x), 240.)
            self.assertEqual(c.tile((slice(0, 2), slice(0, 3)))['data'].shape[-1], 2)
        with self.assertRaisesRegex(ValueError, 'posting'):
            with NisarCollection([paths[1], paths[2]]): pass

    def test_empty_product_retained_and_invalid_factor_atomic(self):
        a = subset(self.root/'a.h5'); b = enriched(self.root/'b.h5', frame=145)
        out = self.root/'coarse.nc'
        with patch('glacier_offsets.nisar_packaging.native_geometry', side_effect=test_geometry) as native:
            result = package_nisar([a, b], output=out, workers=1, decimation=3)
            self.assertEqual(result['pairs'], 2)
            self.assertEqual(native.call_count, 1)
        with h5netcdf.File(out) as h:
            self.assertEqual(h['valid_mask'][1].sum(), 0)
            assert_array_equal(h['input_mask'][1], 255)
        before = out.read_bytes()
        for bad in (0, -1, True, 1.5, '3'):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                package_nisar([a], output=out, decimation=bad, force=True)
        with self.assertRaisesRegex(ValueError, 'no sampled'):
            package_nisar([b], output=out, decimation=100, force=True)
        self.assertEqual(out.read_bytes(), before)

    def test_cli_dispatch(self):
        from glacier_offsets.packaging import main
        with patch('glacier_offsets.nisar_packaging.package_nisar', return_value={}) as importer:
            self.assertEqual(main(['--nisar-subsets', 'inputs', '--decimation', '3', '-o', 'out.nc']), 0)
            self.assertEqual(importer.call_args.kwargs['decimation'], 3)
        self.assertEqual(main(['stack.yaml', '--decimation', '3']), 2)

    def test_sampling_alignment_is_validated(self):
        source = subset(self.root/'input.h5'); out = self.root/'coarse.nc'
        with patch('glacier_offsets.nisar_packaging.native_geometry', test_geometry):
            package_nisar([source], output=out, workers=1, decimation=3)
        with h5netcdf.File(out, 'a') as h:
            h['x'][:] += 80
            spec = json.loads(h['provenance_json'][()].decode())
            spec['grid']['transform'][0] += 80
            h['provenance_json'][()] = json.dumps(spec)
            h.attrs['package_identity'] = identity(spec)
            h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
        with self.assertRaisesRegex(ValueError, 'alignment'): validate_nisar(out)

    def test_fit_velocity_inspect_and_map_coarse_horizontal(self):
        from copy import deepcopy
        from glacier_offsets.velocity_io import Collection, fingerprint
        from glacier_offsets.velocity_inspect import propose
        from glacier_offsets.velocity_recovery import DEFAULTS, map_stage, run_recipe
        from glacier_offsets.velocity_basis import definition
        path = corrected(coarse_synthetic(synthetic(self.root/'raw.nc', n=32)))
        out = self.root/'pilot'; out.mkdir()
        regions = self.root/'regions.json'
        regions.write_text(json.dumps({'EPSG': 3031, 'regions': [{'id': 'target', 'role': 'target', 'bounds': [-2198480, 1074880, -2198000, 1075360]}]}))
        cfg = deepcopy(DEFAULTS) | {'inputs': [str(path)], 'output': str(out), 'regions': str(regions)}
        cfg['map'].update(min_pairs_one_year=10, min_days_one_year=30)
        with Collection([path]) as c:
            inspection = propose(c, out)
            self.assertEqual(inspection['aliases'][0]['sampling']['viewing_geometries_at_this_point'], 4)
            with self.assertRaisesRegex(ValueError, 'validated pilot'): map_stage(cfg, c)
            candidate = {'weighting': 'constant', 'basis': definition(c.point(0, 0)['early'], c.point(0, 0)['late'], 'constant'), 'penalty': 0.}
            # Fixture only: production predictive/recovery acceptance is unchanged.
            record = {'recipe_identity': fingerprint(run_recipe(cfg, c)), 'map_allowed': True,
                      'selected': {'free_en': candidate}, 'noise': {'covariance_ra_m2': [[.01, 0], [0, .04]]}}
            (out/'pilot.json').write_text(json.dumps(record))
            self.assertEqual(map_stage(cfg, c)['status'], 'complete')
            with h5netcdf.File(out/'map_free_en.nc') as h:
                assert_array_equal(np.diff(h['x'][:]), 240.)
                assert_array_equal(h['analysis_location_index'][:], c.association)

    def test_coarse_grid_both_fit_readers_and_pixel_replay(self):
        import yaml
        from glacier_offsets.velocity_workflow import run_workflow
        from glacier_offsets.velocity_workflow_products import RunView, export_maps, verify_v2
        from glacier_offsets.velocity_io import Collection
        from glacier_offsets.velocity_collection import collect
        from glacier_offsets.velocity_prepare import prepare_observations
        path = corrected(coarse_synthetic(synthetic(self.root/'raw.nc')))
        with Collection([path]) as c:
            self.assertEqual(c.spacing, 240.)
            p = c.point(0, 0)
            self.assertTrue(p['valid'].any())
        groups = {f'T{track:03d}:{year}': [[.01, .002], [.002, .04]] for track in (53, 75, 104, 154) for year in (2024, 2025, 2026)}
        (self.root/'noise.json').write_text(json.dumps({'covariance_ra_m2_by_group': groups}))
        config = self.root/'fit.yaml'
        config.write_text(yaml.safe_dump({'version': 2, 'inputs': [str(path)], 'cache_directory': str(self.root/'cache'),
                                         'output': str(self.root/'fit'), 'corrections': {'solid_earth': False},
                                         'noise': {'mode': 'supplied', 'group_by': 'track_year', 'files': ['noise.json']},
                                         'coverage': {'min_span_days': 540}}))
        run_workflow(config, 'fit')
        with h5py.File(self.root/'fit/source_coefficients.h5') as h:
            names = json.loads(h['residual_vertical'].attrs['parameter_names'])
            assert_allclose(h['residual_vertical/coefficients'][:, names.index('Msf_C')], .03, atol=1e-5)
        view = RunView.v2(self.root/'fit'); export_maps(view); verify_v2(view)
        with h5netcdf.File(self.root/'fit/coefficient_maps.nc') as h:
            assert_array_equal(np.diff(h['x'][:]), 240.)


if __name__ == '__main__': unittest.main()

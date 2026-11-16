"""GOFF import, paired corrections and multitrack scientific recovery."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5netcdf
import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from glacier_offsets.displacement import NISAR_SCHEMA, validate_nisar, valid_nisar_mask
from glacier_offsets.nisar_packaging import package_nisar, CUBES, VARIABLES
from glacier_offsets.packaging import variable, strings, timestamp, content_hashes
from glacier_offsets.products import identity
from glacier_offsets.velocity_basis import EPOCH_SECONDS, harmonic
from glacier_offsets.velocity_vertical import vertical_columns


def synthetic(path, n=236, tracks=(53, 75, 104, 154), frame_duplicate=False):
    """Independent linear forward model at four geographic analysis locations."""
    from pyproj import CRS, Transformer
    from glacier_offsets.velocity_pixel_general import Layout
    start = datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()
    early = start+np.arange(n)*3*86400.+np.array([10.5 if t % 2 else 21.75 for t in range(n)])*3600
    late = early+12*86400.+.17
    track = np.array([tracks[i % len(tracks)] for i in range(n)])
    xx, yy = np.meshgrid([-2198520., -2198440.], [1075400., 1075320.])
    offset = np.array([[.0, .13], [.26, .39]])
    e = early[:, None, None]+offset; l = late[:, None, None]+offset
    years = sorted(set(datetime.fromtimestamp(float(t), timezone.utc).year for t in (early+late)/2))
    layout = Layout(years, [])
    overlap = layout.overlaps((e-EPOCH_SECONDS)/86400, (l-EPOCH_SECONDS)/86400)
    en = np.stack([np.sum(overlap, axis=-1)*.4, np.sum(overlap, axis=-1)*.2], -1)
    h = harmonic((e-EPOCH_SECONDS)/86400, (l-EPOCH_SECONDS)/86400)
    axis = np.array([2., 1.])/np.sqrt(5.)
    en += np.einsum('tyxj,j->tyx', h, [.03, -.02])[..., None]*axis
    up = np.einsum('tyxj,j->tyx', vertical_columns((e-EPOCH_SECONDS)/86400, (l-EPOCH_SECONDS)/86400), [.08, -.03, .04, .02])
    geom = np.empty((n, 2, 2, 2, 3))
    for i in range(n):
        ascending = track[i] in (75, 104)
        eps = .002*np.sin(i/13)
        gr = np.array([-.64 if ascending else .64, -.12, -.76]); gr /= np.linalg.norm(gr)
        ga = np.array([-.15+eps, .988 if ascending else -.988, .015+eps])
        geom[i, ..., 0, :] = gr; geom[i, ..., 1, :] = ga
    ra = np.einsum('tyxci,tyxi->tyxc', geom[..., :2], en)+geom[..., 2]*up[..., None]
    grid = {'epsg': 3031, 'shape': [2, 2], 'transform': [xx[0, 0]-40, 80., 0., yy[0, 0]+40, 0., -80.]}
    spec = {'schema': NISAR_SCHEMA, 'mission': 'NISAR', 'grid': grid, 'frequency': 'A', 'polarization': 'HH', 'layer': 1, 'release': ['PR', 'P05023', '1.0.11', '1.5.0']}
    with h5netcdf.File(path, 'w') as out:
        out.dimensions = {'time': n, 'y': 2, 'x': 2, 'enu': 3, 'bound': 2}
        out.attrs.update(schema_version=NISAR_SCHEMA, mission='NISAR', package_status='complete', EPSG=3031,
                         observation_support='geocoded_cell', geocoding_resampling='bilinear', package_identity=identity(spec), source='raw')
        for k, a in [('x', xx[0]), ('y', yy[:, 0])]: variable(out, k, (k,), a, units='m')
        variable(out, 'crs', (), np.int32(0)).attrs.update(CRS.from_epsg(3031).to_cf())
        strings(out, 'enu', ('enu',), ['east', 'north', 'up'])
        ids = [f'NISAR_synthetic_{t:03d}_{i:04d}' for i, t in enumerate(track)]
        for k in ('pair_id', 'names'): strings(out, k, ('time',), ids)
        timestamp(out, 'time', ('time',), (early+late)/2)
        for k, a in [('earlier_time', early), ('later_time', late)]: timestamp(out, k, ('time',), a)
        for k, a in [('earlier_observation_time', e), ('later_observation_time', l)]: timestamp(out, k, ('time', 'y', 'x'), a)
        for k in ('earlier_spacecraft', 'later_spacecraft'): strings(out, k, ('time',), ['NISAR']*n)
        for k, a in [('track_number', track), ('frame_number', np.full(n, 144)), ('reference_absolute_orbit', np.arange(n)+1000), ('secondary_absolute_orbit', np.arange(n)+1173)]: variable(out, k, ('time',), a, dtype='i4')
        for j, k in enumerate(('range_displacement_m', 'azimuth_displacement_m')): variable(out, k, ('time', 'y', 'x'), ra[..., j].astype('f4'), units='m')
        for k, a in [('snr', 30.), ('correlation_peak', .9), ('height_m', 50.), ('azimuth_ground_spacing_m_per_slc_line', 4.34), ('range_displacement_sigma_m', .1), ('azimuth_displacement_sigma_m', .2), ('range_azimuth_displacement_covariance_m2', .005)]:
            variable(out, k, ('time', 'y', 'x'), np.full((n, 2, 2), a, 'f4'), units='m2' if k.endswith('m2') else 'm' if k.endswith('_m') or 'spacing' in k else '1')
        for k in ('valid_mask', 'geometry_valid_mask'): variable(out, k, ('time', 'y', 'x'), np.ones((n, 2, 2), 'u1'), units='1')
        for j, k in enumerate(('range_sensitivity_enu', 'azimuth_sensitivity_enu')): variable(out, k, ('time', 'enu', 'y', 'x'), np.moveaxis(geom[..., j, :], -1, 1), units='1')
        variable(out, 'reference_los_enu', ('time', 'enu', 'y', 'x'), -np.moveaxis(geom[..., 0, :], -1, 1), units='1')
        lon, lat = Transformer.from_crs(3031, 4326, always_xy=True).transform(xx, yy)
        for k, a in [('observation_longitude', lon), ('observation_latitude', lat)]: variable(out, k, ('y', 'x'), a)
        strings(out, 'provenance_json', (), json.dumps(spec, sort_keys=True))
        out.attrs['content_hashes_json'] = json.dumps(content_hashes(out), sort_keys=True)
    return path


class ZeroTide:
    version = 'synthetic'
    metadata = {'synthetic': True}
    def __init__(self, path, lon, lat, response='model'):
        self.available = np.ones(len(lon), bool); self.response_factor = np.ones(len(lon)); self.ocean_mask = np.ones(len(lon))
    def predict(self, indices, time): return np.zeros(len(indices))


def corrected(path, horizontal=True):
    from glacier_offsets.tides import correct_tides
    from glacier_offsets.horizontal import to_horizontal
    model = path.parent/'model.nc'; model.write_bytes(b'synthetic')
    with patch('glacier_offsets.tides.Cats2008', ZeroTide):
        result = correct_tides(path, model)
    tide = Path(result['path'])
    return Path(to_horizontal(tide)['path']) if horizontal else tide


def enriched(path, xshift=0, frame=144):
    """A small enriched subset; native geometry is supplied independently in tests."""
    x = np.array([-2198520., -2198440.])+xshift; y = np.array([1075400., 1075320.])
    with h5py.File(path, 'w') as h:
        h.attrs['epsg'] = 3031; h['x'] = x; h['y'] = y
        h.attrs['url'] = 'https://example.invalid/NISAR_L2_GOFF_PROVISIONAL_V1/subset.h5'
        h['mask'] = np.array([[111, 111], [100, 111]], 'u4')
        d = h.create_dataset('surfaceHeight', data=np.full((2, 2), 50., 'f4')); d.attrs.update(height_reference='WGS84 ellipsoid', units='meters')
        for k, value in zip(VARIABLES, [2., -3., .04, .01, .005, 30., .8]):
            d = h.create_dataset('layer1/'+k, data=np.full((2, 2), value, 'f4'))
            d.attrs['units'] = 'meters^2' if 'Variance' in k else 'meters' if k.endswith('Offset') else '1'
        h['grid/centerFrequency'] = 1.239e9; h['grid/projection'] = 3031
        h['radarGrid/xCoordinates'] = x; h['radarGrid/yCoordinates'] = y; h['radarGrid/heightAboveEllipsoid'] = [0., 100.]
        for k in CUBES:
            value = 38090. if 'AzimuthTime' in k else 6600. if k == 'groundTrackVelocity' else 1.
            d = h.create_dataset('radarGrid/'+k, data=np.full((2, 2, 2), value))
            if 'AzimuthTime' in k: d.attrs['units'] = 'seconds since 2026-09-23T00:00:00' if k.startswith('reference') else 'seconds since 2026-10-05T00:00:00'
        for label in ('reference', 'secondary'):
            h[f'metadata/orbit/{label}/time'] = np.arange(4)*10.
            h[f'metadata/orbit/{label}/position'] = np.zeros((4, 3)); h[f'metadata/orbit/{label}/velocity'] = np.zeros((4, 3))
        group = h.require_group('metadata/processingInformation/parameters/reference/frequencyA')
        for k, value in [('zeroDopplerTimeSpacing', 1/1520), ('slantRangeSpacing', 3.12), ('slantRangeStart', 900000.), ('numberOfAzimuthLines', 50000), ('numberOfRangeSamples', 50000)]: group[k] = value
        h['metadata/processingInformation/algorithms/geocoding/floatingGeocodingInterpolation'] = np.bytes_('BILINEAR')
        ids = {'granuleId': f'NISAR_L2_PR_GOFF_test_{frame}', 'productType': 'GOFF', 'platformName': 'NISAR', 'productVersion': '1.0.11', 'productSpecificationVersion': '1.5.0', 'compositeReleaseId': 'P05023',
               'trackNumber': 75, 'frameNumber': frame, 'lookDirection': 'Left', 'orbitPassDirection': 'Ascending', 'referenceAbsoluteOrbitNumber': 1000, 'secondaryAbsoluteOrbitNumber': 1173,
               'referenceZeroDopplerStartTime': '2026-09-23T10:34:45', 'secondaryZeroDopplerStartTime': '2026-10-05T10:34:45'}
        for k, v in ids.items(): h['identification/'+k] = np.bytes_(v) if isinstance(v, str) else v
    return path


def test_geometry(h, points, cube):
    result = np.empty((len(points), 3, 3))
    result[:, 0] = [.6, .0, .8]; result[:, 1] = [-.6, 0., -.8]; result[:, 2] = [0., 1., .01]
    return result


class NisarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)

    def test_mask_and_integrity(self):
        assert_array_equal(valid_nisar_mask(np.array([111, 11, 100, 110, 101, 255, 111+(1<<8)], 'u4')), [1, 1, 0, 0, 0, 0, 0])
        path = synthetic(self.root/'raw.nc', n=4)
        self.assertEqual(validate_nisar(path)['pairs'], 4)
        with h5netcdf.File(path, 'a') as h: h['range_displacement_m'][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'checksum'): validate_nisar(path)

    def test_enriched_import_union_atomic_reuse_and_covariance(self):
        directory = self.root/'subsets'; directory.mkdir()
        enriched(directory/'one.h5'); enriched(directory/'two.h5', xshift=80, frame=145)
        with h5py.File(directory/'one.h5', 'a') as h: h['layer1/alongTrackOffsetVariance'][0, 1] = -1
        (directory/'active.h5.part').write_bytes(b'incomplete')
        out = self.root/'package.nc'
        with patch('glacier_offsets.nisar_packaging.native_geometry', test_geometry):
            result = package_nisar([directory], output=out)
            self.assertEqual(result['shape'], [2, 2, 3])
            self.assertTrue(package_nisar([directory], output=out)['reused'])
        with h5netcdf.File(out) as h:
            assert_allclose(h['range_displacement_m'][0, 0, :2], [-3, -3])
            self.assertTrue(np.isnan(h['range_displacement_m'][0, 1, 0]))
            assert_allclose(h['range_displacement_sigma_m'][0, 0, 0], .1)
            assert_allclose(h['range_azimuth_displacement_covariance_m2'][0, 0, 0], .005)
            self.assertTrue(np.isnan(h['range_displacement_sigma_m'][0, 0, 1]))
            self.assertEqual(h['valid_mask'][0, 0, 1], 1)
            self.assertEqual(h.attrs['data_maturity'], 'PROVISIONAL')
            self.assertEqual(h['input_mask'][0, 1, 0], 100)
            self.assertEqual(h['source_slantRangeOffset'][0, 1, 0], -3)
            self.assertEqual(h['later_observation_time'][0, 0, 0]-h['earlier_observation_time'][0, 0, 0], 12*86400.)
        before = out.read_bytes()
        with h5py.File(directory/'one.h5', 'a') as h: del h['surfaceHeight']
        with self.assertRaisesRegex(ValueError, 'regenerate'): package_nisar([directory], output=out, force=True)
        self.assertEqual(before, out.read_bytes())

    def test_frame_overlap_ownership_and_reprocessed_rejection(self):
        from glacier_offsets.nisar_collection import NisarCollection, collect_nisar
        path = synthetic(self.root/'raw.nc', n=2, tracks=(75,))
        with h5netcdf.File(path, 'a') as h:
            h['reference_absolute_orbit'][1] = h['reference_absolute_orbit'][0]
            h['secondary_absolute_orbit'][1] = h['secondary_absolute_orbit'][0]
            h['frame_number'][1] = 145
            for name in ('earlier_time', 'later_time', 'time'):
                h[name][1] = h[name][0]+10
            for name in ('earlier_observation_time', 'later_observation_time'):
                h[name][1] = h[name][0]+10
            h['valid_mask'][0, 0, 0] = 0
            for name in ('range_displacement_m', 'azimuth_displacement_m', 'snr', 'correlation_peak'): h[name][0, 0, 0] = np.nan
            h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
        path = corrected(path)
        with NisarCollection([path]) as c:
            a = c.tile((slice(0, 2), slice(0, 2)))
            assert_array_equal(a['valid'].sum(0), np.ones((2, 2)))
            self.assertEqual(a['overlap_suppressed'].sum(), 3)
            self.assertTrue(a['valid'][1, 0, 0])
        manifest = collect_nisar([path], self.root/'cache')
        self.assertEqual(len(manifest['overlap_groups']), 1)
        with h5netcdf.File(path, 'a') as h:
            h['frame_number'][1] = 144
            h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
        with self.assertRaisesRegex(ValueError, 'reprocessed'):
            with NisarCollection([path]): pass

    def test_variable_time_solid_earth(self):
        from glacier_offsets.velocity_collection import collect
        from glacier_offsets.velocity_prepare import prepare_observations
        from glacier_offsets.velocity_solid_earth import prepare_solid_earth
        native = Path('/Users/briel/miniforge3/envs/isce3/bin/python')
        if not native.exists(): self.skipTest('Optional native PySolid worker not available')
        path = corrected(synthetic(self.root/'raw.nc', n=4))
        manifest = collect([path], self.root/'cache')
        from glacier_offsets.velocity_solid_earth import Interrupted
        def interrupt(stage, progress):
            if progress == 2: raise Interrupted('test checkpoint')
        with self.assertRaises(Interrupted): prepare_observations(self.root/'cache', manifest, chunk=2, hook=interrupt)
        obs, _ = prepare_observations(self.root/'cache', manifest, chunk=2)
        with h5py.File(obs.path('observations.h5')) as h, h5netcdf.File(path) as source:
            for i in range(4):
                expected = np.stack([source[k][i] for k in ('range_displacement_m', 'azimuth_displacement_m')], -1).reshape(4, 2)
                assert_array_equal(h['data'][:, i], expected)
        worker, _ = prepare_solid_earth(self.root/'cache', obs, {'solid_earth_python': str(native), 'solid_earth_spacing_deg': .1, 'preparation_chunk_pairs': 2})
        with h5py.File(worker.path('solid_earth.h5')) as h:
            self.assertTrue(np.isfinite(h['set_ra_m'][:]).all())
            self.assertLess(h.attrs['max_ENU_endpoint_difference_error_m'], 1e-5)

    def test_grouped_temporal_calibration_and_injection(self):
        from glacier_offsets.nisar_fit import learn_noise_nisar
        from glacier_offsets.velocity_pilot import point_arrays, fit_point, _noise_draw
        from glacier_offsets.velocity_basis import definition
        rng = np.random.default_rng(101); n = 120
        early = 8767.+np.arange(n)*6+.43; late = early+12.000003
        track = np.array([53, 75, 104, 154]*(n//4))
        years = [int(str(np.datetime64('2000-01-01')+np.timedelta64(int(t), 'D'))[:4]) for t in (early+late)/2]
        groups = np.array([f'T{t:03d}:{y}' for t, y in zip(track, years)])
        g = np.empty((n, 2, 3))
        g[:, 0] = [-.6, -.1, -.79]; g[:, 1] = [-.15, .988, .015]
        g[track == 53, :, :2] *= -1; inv = np.linalg.inv(g[..., :2])
        base = {'early': early, 'late': late, 'valid': np.ones(n, bool), 'inverse': inv, 'geometry': g,
                'peak': np.full(n, .9), 'snr': np.full(n, 30.), 'pair_id': np.array([f'p{i}' for i in range(n)]),
                'noise_group': groups, 'track': track, 'mission': 'NISAR'}
        points = []
        for i in range(12):
            ra = rng.normal(0, .1, (n, 2))
            points.append(base | {'data': np.einsum('tij,tj->ti', inv, ra), 'role': 'control',
                                 'region': f'control{i}', 'spatial_block': f'{i}:0', 'source_id': i})
        noise, records = learn_noise_nisar(points, None, None, {'mode': 'pair', 'min_pair_blocks': 3, 'shrinkage_blocks': 20.})
        self.assertEqual(set(noise['by_group']), set(groups))
        self.assertEqual(set(noise['pair_calibration']['pairs']), set(base['pair_id']))
        for p, group in noise['pair_noise_group'].items(): self.assertIn(p, noise['by_group'][group]['pair_calibration']['pairs'])
        direction = np.array([2., 1.])/np.sqrt(5.)
        signal = harmonic(early, late) @ [.03, -.02]
        target = base | {'data': (late-early)[:, None]*[.4, .2]+signal[:, None]*direction,
                         'role': 'target', 'region': 'target', 'spatial_block': '20:0', 'source_id': 20}
        e, l, data, covariance, good = point_arrays(target, noise, 'quality_robust')
        self.assertTrue(np.linalg.eigvalsh(covariance).min() > 0)
        candidate = {'weighting': 'constant', 'basis': definition(early, late, 'constant'), 'penalty': 0.}
        result = fit_point(target, noise, candidate, 'along_flow', direction=direction)
        self.assertGreaterEqual(result['n_pairs'], 100)
        draw = _noise_draw(rng, target, noise, records)
        self.assertEqual(draw.shape, (n, 2)); self.assertTrue(np.isfinite(draw).all())

    def test_paired_cats_predictions_match_scalar(self):
        try:
            import pyTMD
            import xarray as xr
        except ImportError:
            self.skipTest('Optional pyTMD predictor not installed')
        from glacier_offsets.tide_model import Cats2008
        names = ['m2', 's2', 'n2', 'k2', 'k1', 'o1', 'p1', 'q1', 'mf', 'mm']
        rng = np.random.default_rng(15)
        model = object.__new__(Cats2008)
        model.local = xr.Dataset({k: xr.DataArray(rng.normal(size=4)+1j*rng.normal(size=4), dims='point', attrs={'units': 'm'}) for k in names})
        model.metadata = {'constituents': names}; model.available = np.ones(4, bool)
        times = 1790000000.+np.array([.0, .13, .26, .39]); indices = np.arange(4)
        paired = model.predict(indices, times)
        scalar = np.array([model.predict(np.array([i]), float(t))[0] for i, t in enumerate(times)])
        assert_allclose(paired, scalar, rtol=0, atol=1e-12)

    def test_temporal_inspect_and_map_gate(self):
        from copy import deepcopy
        from glacier_offsets.velocity_io import Collection, fingerprint
        from glacier_offsets.velocity_inspect import propose
        from glacier_offsets.velocity_recovery import DEFAULTS, map_stage, run_recipe
        from glacier_offsets.velocity_basis import definition
        path = corrected(synthetic(self.root/'raw.nc', n=32))
        out = self.root/'pilot'; out.mkdir()
        regions = self.root/'regions.json'
        regions.write_text(json.dumps({'EPSG': 3031, 'regions': [{'id': 'target', 'role': 'target', 'bounds': [-2198560, 1075280, -2198400, 1075440]}]}))
        cfg = deepcopy(DEFAULTS) | {'inputs': [str(path)], 'output': str(out), 'regions': str(regions)}
        cfg['map'].update(min_pairs_one_year=10, min_days_one_year=30)
        with Collection([path]) as c:
            inspection = propose(c, out)
            self.assertEqual(inspection['aliases'][0]['sampling']['viewing_geometries_at_this_point'], 4)
            with self.assertRaisesRegex(ValueError, 'validated pilot'): map_stage(cfg, c)
            # Unit fixture for an accepted pilot; production pilot acceptance remains unchanged.
            candidate = {'weighting': 'constant', 'basis': definition(c.point(0, 0)['early'], c.point(0, 0)['late'], 'constant'), 'penalty': 0.}
            record = {'recipe_identity': fingerprint(run_recipe(cfg, c)), 'map_allowed': True,
                      'selected': {'free_en': candidate}, 'noise': {'covariance_ra_m2': [[.01, 0], [0, .04]]}}
            (out/'pilot.json').write_text(json.dumps(record))
            self.assertEqual(map_stage(cfg, c)['status'], 'complete')
            with h5netcdf.File(out/'map_free_en.nc') as h:
                self.assertIn('analysis_location_index', h.variables)
                assert_array_equal(h['analysis_location_index'][:], c.association)

    def test_corrections_horizontal_and_dynamic_inverse(self):
        from glacier_offsets.tides import validate_tidal_stack
        from glacier_offsets.horizontal import validate_horizontal_stack
        from glacier_offsets.velocity_io import Collection
        from glacier_offsets.velocity_pilot import point_arrays
        path = corrected(synthetic(self.root/'raw.nc', n=12))
        self.assertEqual(validate_horizontal_stack(path)['pairs'], 12)
        validate_tidal_stack(path)
        with Collection([path]) as c:
            point = c.point(0, 0)
            self.assertEqual(point['inverse'].shape, (12, 2, 2))
            noise = {'covariance_ra_m2': [[1., .1], [.1, 2.]]}
            _, _, _, cov, good = point_arrays(point, noise, 'constant')
            expected = point['inverse'][good] @ np.array(noise['covariance_ra_m2']) @ point['inverse'][good].transpose(0, 2, 1)
            assert_allclose(cov, expected)
            self.assertGreater(np.max(abs(point['inverse'][0]-point['inverse'][1])), .5)

    def test_prepare_joint_fit_and_replay(self):
        from glacier_offsets.velocity_workflow import run_workflow
        from glacier_offsets.velocity_workflow_products import RunView, export_maps, verify_v2
        import yaml
        path = corrected(synthetic(self.root/'raw.nc'))
        groups = {f'T{track:03d}:{year}': [[.01, .002], [.002, .04]] for track in (53, 75, 104, 154) for year in (2024, 2025, 2026)}
        (self.root/'noise.json').write_text(json.dumps({'covariance_ra_m2_by_group': groups}))
        cfg = {'version': 2, 'inputs': [str(path)], 'cache_directory': str(self.root/'cache'), 'output': str(self.root/'fit'),
               'corrections': {'solid_earth': False}, 'noise': {'mode': 'supplied', 'group_by': 'track_year', 'files': ['noise.json']},
               'coverage': {'min_span_days': 540}}
        config = self.root/'fit.yaml'; config.write_text(yaml.safe_dump(cfg))
        run_workflow(config, 'fit')
        with h5py.File(self.root/'fit/source_coefficients.h5') as h:
            self.assertTrue(h['residual_vertical/estimable'][:].all())
            names = json.loads(h['residual_vertical'].attrs['parameter_names'])
            coef = h['residual_vertical/coefficients'][:]
            assert_allclose(coef[:, names.index('Msf_C')], .03, atol=1e-5)
            assert_allclose(coef[:, names.index('Msf_S')], -.02, atol=1e-5)
            for year in (2024, 2025):
                assert_allclose(coef[:, names.index(f'east_velocity_{year}')], .4, atol=1e-5)
            self.assertFalse(h['azimuth_direct/estimable'][:].any())
            for track in (53, 75, 104, 154): self.assertIn(f'azimuth_track_{track:03d}', h)
        view = RunView.v2(self.root/'fit'); export_maps(view); verify_v2(view)
        run_workflow(config, 'fit')

    def test_short_record_and_layer_rejection(self):
        from glacier_offsets.velocity_collection import collect
        from glacier_offsets.velocity_workflow import run_workflow
        import yaml
        path = corrected(synthetic(self.root/'short.nc', n=2))
        manifest = collect([path], self.root/'cache')
        self.assertEqual(manifest['mission'], 'NISAR')
        noise = self.root/'noise.json'; noise.write_text(json.dumps({'covariance_ra_m2_by_group': {}}))
        cfg = {'version': 2, 'inputs': [str(path)], 'cache_directory': str(self.root/'cache'), 'output': str(self.root/'fit'),
               'corrections': {'solid_earth': False}, 'noise': {'mode': 'supplied', 'group_by': 'track_year', 'files': ['noise.json']}, 'coverage': {'min_span_days': 540}}
        config = self.root/'fit.yaml'; config.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError, 'cannot be satisfied'): run_workflow(config, 'fit')


if __name__ == '__main__': unittest.main()

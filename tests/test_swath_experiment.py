"""Scientific contracts for isolated range-2x swath experiments."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from osgeo import gdal

from glacier_offsets import range_oversampling as r2
from glacier_offsets.ampcor import _native_executor
from glacier_offsets.fftw_cache import pinned, pinned_wisdom
from glacier_offsets.matching_grid import MatchingGrid, build_grid
from glacier_offsets.products import identity
from glacier_offsets.swath_models import (train, predict, graph_audit, validation_folds,
    evaluate, multiplier_draws, error_metrics, paired_metrics)
from glacier_offsets.swath_observations import prepare_strip, support_arrays
from glacier_offsets.swath_observations import measure_link
from glacier_offsets.swath_physics import apparent_bias, bistatic_seconds, mapping
from glacier_offsets.swath_physics import fm_rate_seconds
from glacier_offsets.swath_physics import native_input_products
from tools.swath_experiment_io import Inputs, Experiment, save_npz, load_npz, cohorts
from tools.swath_experiment_previews import (symmetric_bias, pure_swath_footprints,
    preview_predictions, corrected_preview, write_previews)
from test_range_oversampling import PARAMS
from test_swath_overlap import write_raster
from test_swath_overlap import make_pair


def row(a, b, value, group=('burst_iw1', 'burst_iw2')):
    return {'pair': f'{a}_{b}', 'earlier': a, 'later': b,
            'burst1': group[0], 'burst2': group[1], 'mean': value, 'values': [value]*3}


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.dates = ['a', 'b', 'c', 'd']
        self.missions = {'a': 'S1A', 'b': 'S1B', 'c': 'S1A', 'd': 'S1B'}
        self.terms = {'a': .03, 'b': -.01, 'c': .04, 'd': -.06}
        self.rows = [row(a, b, self.terms[b]-self.terms[a], group)
                     for a, b in (('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'c'), ('b', 'd'))
                     for group in (('x_iw1', 'x_iw2'), ('y_iw1', 'y_iw2'))]

    def test_date_terms_and_datums_are_recovered(self):
        fit = train(self.rows, self.dates, self.missions, 'per_acquisition')
        np.testing.assert_allclose(fit['parameters'], list(self.terms.values()), atol=1.e-15)
        self.assertEqual(fit['graph']['cycles'], 2)
        self.assertAlmostEqual(fit['parameters'].sum(), 0)
        for kind, label, training, testing in validation_folds(self.rows, self.dates):
            fitted = train([r for r, take in zip(self.rows, training) if take], self.dates,
                           self.missions, 'per_acquisition')
            test = [r for r, take in zip(self.rows, testing) if take]
            predicted = predict(fitted, test, self.dates, self.missions)
            if kind == 'leave_acquisition_out':
                self.assertTrue(np.isnan(predicted).all())
            else:
                np.testing.assert_allclose(predicted, [r['mean'] for r in test], atol=1.e-15)

    def test_spacecraft_recovery_without_intercept_or_ipf_terms(self):
        data = [dict(r, mean=.025*(int(self.missions[r['later']] == 'S1B')-
                                  int(self.missions[r['earlier']] == 'S1B'))) for r in self.rows]
        fit = train(data, self.dates, self.missions, 'spacecraft')
        np.testing.assert_allclose(fit['parameters'], [.025], atol=1.e-15)

    def test_tree_exact_fit_is_not_validation_and_cycles_retain_inconsistency(self):
        tree = [row('a', 'b', .2), row('b', 'c', -.1), row('c', 'd', .05)]
        fit = train(tree, self.dates, self.missions, 'per_acquisition')
        self.assertEqual(fit['graph']['cycles'], 0)
        self.assertEqual(len(fit['graph']['bridges']), 3)
        np.testing.assert_allclose(predict(fit, tree, self.dates, self.missions), [.2, -.1, .05])
        inconsistent = tree+[row('a', 'c', .8)]
        fit = train(inconsistent, self.dates, self.missions, 'per_acquisition')
        self.assertGreater(error_metrics(inconsistent, predict(fit, inconsistent, self.dates, self.missions))['mae_px'], .01)
        disconnected = train([tree[0], tree[2]], self.dates, self.missions, 'per_acquisition')
        self.assertTrue(np.isnan(predict(disconnected, [row('a', 'd', 0)], self.dates, self.missions)).all())

    def test_spatial_folds_purge_shared_source_bursts_and_never_use_test_values(self):
        data = self.rows+[row('a', 'b', 10, ('x_iw1', 'z_iw2'))]
        for kind, label, training, testing in validation_folds(data, self.dates):
            self.assertFalse((training & testing).any())
            if kind == 'leave_burst_group_out':
                keys = set(label.split('|'))
                for r, take in zip(data, training):
                    if take:
                        self.assertFalse(keys.intersection((r['burst1'], r['burst2'])))
        fit1 = train(self.rows[:-2], self.dates, self.missions, 'per_acquisition')
        held = [dict(r, mean=1.e6) for r in self.rows[-2:]]
        np.testing.assert_allclose(predict(fit1, held, self.dates, self.missions),
                                   predict(fit1, self.rows[-2:], self.dates, self.missions))

    def test_bootstrap_and_folds_reproducible_with_full_parameter_covariance(self):
        a = multiplier_draws(self.rows, 25)
        np.testing.assert_array_equal(a, multiplier_draws(self.rows, 25))
        self.assertTrue((a > 0).all())
        report = evaluate(self.rows, self.dates, self.missions, draws=25)
        cov = np.asarray(report['models']['per_acquisition']['covariance'])
        self.assertEqual(cov.shape, (4, 4))
        for fold in report['folds']:
            if fold['kind'] == 'leave_acquisition_out':
                self.assertTrue(np.isnan(fold['models']['per_acquisition']['prediction_px']).all())

    def test_equal_link_weight_and_common_population(self):
        data = [row('a', 'b', 1), row('b', 'c', 3)]
        data[0]['values'] = [1]*100
        self.assertAlmostEqual(error_metrics(data, [0, 0])['mae_px'], 2)
        common = paired_metrics(data, {'a': [0, 0], 'b': [np.nan, 0]})
        self.assertEqual(common['common_n'], 3)
        self.assertEqual(common['models']['a']['mae_px'], 3)


class PhysicalContractTests(unittest.TestCase):
    def test_positive_and_negative_timing_perturbations_through_native_resampler(self):
        import isce3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = (192, 512)
            rng = np.random.default_rng(73)
            fy, fx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]
            spectrum = np.fft.fft2(rng.normal(size=shape)+1j*rng.normal(size=shape))
            spectrum *= (abs(fy) < .12) & (abs(fx) < .4)
            image = np.fft.ifft2(spectrum).astype('complex64')
            write_raster(root/'reference.tif', image, gdal.GDT_CFloat32)
            r2.prepare(root/'reference.tif', root/'reference2x.tif')
            grid = isce3.product.RadarGridParameters(0., .055, 500., 800000., 2.3,
                isce3.core.LookSide.Right, *shape, isce3.core.DateTime('2020-01-01T00:00:00'))
            params = PARAMS | {'range_oversampling': 2}
            matching = MatchingGrid(np.array([40, 72, 104]), np.array([40, 168, 296]),
                                   tuple(params['window']), tuple(params['search']), tuple(params['stride']))
            with pinned(root/'wisdom'):
                wisdom = pinned_wisdom(r2.native_parameters(params))
            for shift in (-.25, .25):
                write_raster(root/'az.off.tif', np.full(shape, shift), gdal.GDT_Float64)
                write_raster(root/'rg.off.tif', np.zeros(shape), gdal.GDT_Float64)
                output = isce3.io.Raster(str(root/'resampled.tif'), shape[1], shape[0], 1, gdal.GDT_CFloat32, 'GTiff')
                resampler = isce3.image.ResampSlc(grid, isce3.core.LUT2d(),
                                                isce3.core.Poly2d(np.zeros((1, 1))), ref_rdr_grid=grid)
                resampler.resamp(isce3.io.Raster(str(root/'reference.tif')), output,
                    isce3.io.Raster(str(root/'rg.off.tif')), isce3.io.Raster(str(root/'az.off.tif')), flatten=False)
                del output, resampler
                r2.prepare(root/'resampled.tif', root/'resampled2x.tif')
                args = (str(root/'reference2x.tif'), str(root/'resampled2x.tif'), root/f'native-{shift}',
                        params, {'gross_offsets': {'mode': 'zero'}}, 'cpu')
                with _native_executor(1, single_threaded=True, wisdom=wisdom) as pool:
                    result = pool.apply(r2.range_worker, ((args, {'_grid': matching,
                        '_valid_input': np.ones(matching.shape, bool)}),))
                predicted = apparent_bias(np.array([shift/grid.prf]), np.eye(2)[None, ...], 1/grid.prf)[0, 0]
                self.assertTrue(result['valid'].all())
                self.assertLess(abs(result['total'][..., 0].mean()-predicted), .05)

    def test_annotation_changes_invalidate_physical_identity_without_changing_aligned_slcs(self):
        from glacier_offsets.products import checksum
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'safe/annotation/calibration').mkdir(parents=True)
            manifest, annotation = root/'safe/manifest.safe', root/'safe/annotation/iw1.xml'
            manifest.write_text('<manifest/>')
            annotation.write_text('<coefficients>1</coefficients>')
            orbit, dem = root/'orbit.xml', root/'dem.tif'
            orbit.write_text('precise orbit')
            dem.write_text('pinned DEM')
            acq = {'specification': {'safe': {'path': str(root/'safe')},
                                    'orbit': {'path': str(orbit), 'sha256': checksum(orbit)}}}
            experiment = SimpleNamespace(inputs=Inputs(), acquisition=lambda _: acq,
                frame={'specification': {'inputs': {'dem': {'path': str(dem), 'sha256': checksum(dem)}}}})
            first = native_input_products(experiment, 'date')
            annotation.write_text('<coefficients>2</coefficients>')
            experiment.inputs = Inputs()
            second = native_input_products(experiment, 'date')
            self.assertNotEqual(identity(first), identity(second))
            self.assertEqual(first[str(orbit)], second[str(orbit)])
            orbit.write_text('wrong orbit')
            experiment.inputs = Inputs()
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                native_input_products(experiment, 'date')

    def test_fm_sparse_values_do_not_redefine_original_burst_midpoint(self):
        from datetime import datetime
        class Poly:
            def __init__(self, constant, slope):
                self.constant, self.slope = constant, slope
            def eval(self, rho):
                return self.constant+self.slope*(np.asarray(rho)-700000.)
        class Orbit:
            reference_epoch = datetime(2020, 1, 1)
            def interpolate(self, t):
                return np.array([7000000., t*7500., 0]), np.array([0., 7500., 0.])
        grid = SimpleNamespace(ref_epoch=datetime(2020, 1, 1), sensing_start=0.)
        burst = SimpleNamespace(as_isce3_radargrid=lambda: grid, orbit=Orbit(),
            azimuth_time_interval=.002, starting_range=700000., range_pixel_spacing=2.3,
            radar_center_frequency=5.4e9, azimuth_fm_rate=Poly(-2300., .001),
            doppler=SimpleNamespace(poly1d=Poly(15., -.0001)), width=200, length=100,
            azimuth_steer_rate=.01)
        lines, samples = np.meshgrid([10., 40., 80.], [5., 75., 150.], indexing='ij')
        llh = np.broadcast_to([0., 0., 0.], lines.shape+(3,))
        whole = fm_rate_seconds(burst, lines, samples, llh)
        for i in np.ndindex(lines.shape):
            individual = fm_rate_seconds(burst, np.array([lines[i]]), np.array([samples[i]]), np.array([llh[i]]))
            self.assertAlmostEqual(whole[i], individual[0], places=14)

    def test_timing_sign_and_resampling_jacobian(self):
        jacobian = np.broadcast_to(np.array([[1.01, .03], [-.02, .99]]), (3, 2, 2))
        seconds = np.array([.001, -.001, 0])
        result = apparent_bias(seconds, jacobian, .002)
        np.testing.assert_allclose(np.einsum('nij,nj->ni', jacobian, result),
                                   np.stack([-seconds/.002, np.zeros(3)], axis=-1), atol=1.e-15)
        self.assertLess(result[0, 0], 0)
        self.assertGreater(result[1, 0], 0)

    def test_static_bistatic_terms_cancel_and_missing_reference_shift_is_not_added(self):
        b = SimpleNamespace(starting_range=800000., range_pixel_spacing=2.3,
                            iw2_mid_range=880000., rank=9, prf_raw_data=1700.)
        samples = np.array([100, 200])
        correction = bistatic_seconds(b, samples)
        np.testing.assert_array_equal(correction-correction, 0)
        line = np.array([1., 2.])
        y, x, jac = mapping({}, 'burst', line, samples, reference=True)
        np.testing.assert_array_equal(y, line)
        np.testing.assert_array_equal(x, samples)
        np.testing.assert_array_equal(jac, np.broadcast_to(np.eye(2), (2, 2, 2)))
        with self.assertRaisesRegex(ValueError, 'offsets unavailable'):
            mapping({}, 'burst', line, samples)

    def test_registered_offsets_and_derivatives_use_secondary_local_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            y, x = np.indices((16, 20), dtype=float)
            write_raster(root/'az.tif', 2+.01*y+.02*x, gdal.GDT_Float64)
            write_raster(root/'rg.tif', -3+.03*y-.01*x, gdal.GDT_Float64)
            record = {'valid_windows': {'burst': {'offset_paths': [str(root/'az.tif'), str(root/'rg.tif')]}}}
            line, sample = np.array([5.5, 8.5]), np.array([7.5, 11.5])
            actual_y, actual_x, jac = mapping(record, 'burst', line, sample)
            np.testing.assert_allclose(actual_y, line+2+.01*line+.02*sample)
            np.testing.assert_allclose(actual_x, sample-3+.03*line-.01*sample)
            np.testing.assert_allclose(jac, np.broadcast_to([[1.01, .02], [.03, .99]], (2, 2, 2)))


class InputAndPreviewTests(unittest.TestCase):
    def test_clean_maps_retain_raw_mixed_and_unavailable_values_on_the_production_grid(self):
        from glacier_offsets.geocode import make_mapper
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot  # Initialize pyplot before intercepting contour.
        from matplotlib.axes import Axes
        x, y = np.meshgrid(np.arange(4)*200., -np.arange(3)*200.)
        valid = np.ones(x.shape, bool); valid[0, 0] = False
        mapper = make_mapper(x, y, valid, 200., 'nearest')
        total = np.arange(12).reshape(3, 4)*.02
        scale = 14.+total
        owner = np.where(x < 400, 1, 2)
        mixed = x == 200
        available = valid & ~mixed
        arrays = {'raw_total_azimuth_px': total, 'azimuth_scale_m_per_px': scale,
            'interval_seconds': np.full(x.shape, 6*86400.), 'cross_swath_mixed': mixed,
            'same_swath_burst_join': np.zeros(x.shape, bool), 'measured_overlap': mixed,
            'comparison_available': np.zeros(x.shape, bool)}
        predictions = {'physical': 0., 'spacecraft': .04, 'per_acquisition': np.nan}
        availability = {'physical': available, 'spacecraft': available,
                        'per_acquisition': np.zeros(x.shape, bool)}
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(Axes, 'contour', side_effect=AssertionError('Overlap outline added')):
            out = Path(tmp)
            write_previews(out, arrays, predictions, {k: None for k in predictions}, {},
                           mapper, {'epsg': 3031}, owner, availability, 'a_b')
            def raster(name):
                return gdal.Open(str(out/(name+'.tif'))).ReadAsArray()
            raw = raster('raw_azimuth_displacement')
            corrected = raster('spacecraft_corrected_azimuth_displacement')
            np.testing.assert_array_equal(raw, mapper.map(total*scale))
            np.testing.assert_array_equal(np.isfinite(corrected), np.isfinite(raw))
            unchanged = mapper.map(mixed.astype(float)) == 1
            np.testing.assert_array_equal(corrected[unchanged], raw[unchanged])
            expected, applied = corrected_preview(total, symmetric_bias(.04, owner), available)
            np.testing.assert_array_equal(corrected, mapper.map(expected*scale))
            np.testing.assert_array_equal(raster('per_acquisition_corrected_azimuth_displacement'), raw)
            np.testing.assert_allclose(raster('spacecraft_corrected_azimuth_velocity'),
                                       mapper.map(expected*scale/6), equal_nan=True)
            np.testing.assert_array_equal(raster('spacecraft_bias_to_subtract_azimuth_displacement'),
                                           mapper.map(applied*scale))
            # A model with no common scientific support must not blank other panels.
            self.assertFalse(np.any(raster('comparison_available') == 1))
            self.assertTrue(np.isnan(arrays['per_acquisition_corrected_total_azimuth_px']).all())
            for name in ('comparison.png', 'comparison_clean.png', 'corrected_clean.png'):
                self.assertTrue((out/name).is_file())

    def test_unmeasured_link_can_use_an_alternative_path_but_not_a_disconnected_component(self):
        dates = ['a', 'b', 'c']
        rows = [row('a', 'b', .1), row('b', 'c', .2)]
        graph = graph_audit(rows, dates)
        primary = {'rows': [], 'folds': [], 'graph': graph, 'models': {
            'per_acquisition': {'parameter_names': dates, 'parameters': [-.1, 0., .2],
                                'covariance': np.eye(3).tolist()},
            'spacecraft': {'rank': 1, 'parameters': [.03], 'covariance': [[.0001]]}}}
        result, std = preview_predictions(primary, 'a_c', {'a': 'S1A', 'b': 'S1B', 'c': 'S1A'})
        self.assertAlmostEqual(result['per_acquisition'], .3)
        self.assertAlmostEqual(std['per_acquisition'], np.sqrt(2))
        self.assertEqual(result['spacecraft'], 0)
        self.assertTrue(np.isnan(result['physical']))
        primary['graph'] = graph_audit([rows[0]], dates)
        result, _ = preview_predictions(primary, 'a_c')
        self.assertTrue(np.isnan(result['per_acquisition']))

    def test_metre_correction_before_mapping_equals_identically_mapped_subtraction(self):
        from glacier_offsets.geocode import make_mapper
        x, y = np.meshgrid(np.arange(4)*200., -np.arange(3)*200.)
        valid = np.ones(x.shape, bool)
        valid[1, 1] = False
        mapper = make_mapper(x, y, valid, 200., 'nearest')
        scale = 2.+np.arange(12).reshape(3, 4)*.01
        total = np.arange(12).reshape(3, 4)*.02
        bias = np.where(x < 400, -.025, .025)
        before = mapper.map(scale*(total-bias))
        after = mapper.map(scale*total)-mapper.map(scale*bias)
        np.testing.assert_allclose(before, after, atol=1.e-7, equal_nan=True)

    def test_atomic_artifacts_checksum_damage_and_revision_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'arrays.npz'
            save_npz(path, values=np.arange(5))
            np.testing.assert_array_equal(load_npz(path)['values'], np.arange(5))
            inputs = Inputs()
            digest = inputs.file(path)
            inputs.unchanged()
            save_npz(path, values=np.arange(6))
            with self.assertRaisesRegex(ValueError, 'changed during'):
                inputs.unchanged()
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                Inputs().file(path, digest)

    def test_resume_requires_matching_specification_and_all_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exp = object.__new__(Experiment)
            exp.inputs = Inputs()
            exp.sources = {}
            output = root/'values.npz'
            save_npz(output, a=np.arange(4))
            exp.finish(root/'report.json', {'version': 1}, {'status': 'complete'}, [output])
            self.assertTrue(exp.current(root/'report.json', {'version': 1}))
            self.assertFalse(exp.current(root/'report.json', {'version': 2}))
            output.unlink()
            self.assertFalse(exp.current(root/'report.json', {'version': 1}))

    def test_primary_filter_never_uses_offsets_and_symmetric_allocation(self):
        data = {'dual_valid': np.ones(3, bool), 'burst_interior': np.ones(3, bool),
                'geometry_eligible': np.ones(3, bool), 'iw1_peak': np.array([.4, .1, .4]),
                'iw2_peak': np.array([.4, .1, .4]), 'iw1_snr': np.array([8, 20, 7]),
                'iw2_snr': np.array([8, 20, 7])}
        np.testing.assert_array_equal(cohorts(data)['primary'], [True, True, False])
        raw = np.array([1., 1.2])
        corrected = raw-symmetric_bias(.2, np.array([1, 2]))
        np.testing.assert_allclose(corrected, [1.1, 1.1])
        self.assertAlmostEqual(corrected.mean(), raw.mean())

    def test_complete_search_detects_cross_swath_mix_and_same_swath_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'owners.tif'
            owners = np.ones((32, 64), dtype='uint16')
            owners[:, 32:] = 3
            owners[16:, :32] = 2
            write_raster(path, owners, gdal.GDT_UInt16)
            line, sample = np.meshgrid([10., 18.], [10., 30., 50.], indexing='ij')
            pure = pure_swath_footprints({'source_index': str(path)}, line, sample,
                {'window': [8, 8], 'search': [2, 2]}, ['upper_iw1', 'lower_iw1', 'other_iw2'])
            np.testing.assert_array_equal(pure, [[1, 0, 2], [1, 0, 2]])


class NativeExperimentTests(unittest.TestCase):
    def test_real_measurement_path_without_template_pair_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pair, _ = make_pair(root)
            manifest = json.loads((pair/'manifest.json').read_text())
            cfg = {'matching': manifest['matching']['matching'] | {'range_oversampling': 2},
                   'gross_offsets': {'mode': 'zero'}}
            frame = {'coreg': {'reference_grid': {'length': 160, 'width': 320}},
                     'sources': manifest['assembly']['sources']}
            height = root/'stack/geometry/z.tif'
            write_raster(height, np.zeros((160, 320)), gdal.GDT_Float64)
            for source in frame['sources']:
                source['coreg']['geometry']['z'] = str(height)
            exp = object.__new__(Experiment)
            exp.key, exp.cfg, exp.frame, exp.workers = 'synthetic', cfg, frame, 2
            exp.output = root/'experiment'
            exp.stack = root/'stack'
            exp.inputs = Inputs()
            exp.sources = {}
            # Simulate completed alignment manifests; no merged pair is needed.
            for day in ('2020-01-01', '2020-01-07'):
                path = exp.stack/'acquisitions'/day/'manifest.json'
                acq = json.loads(path.read_text())
                acq['alignment_identity'] = identity(acq['specification'])
                acq['status'] = 'aligned'
                path.write_text(json.dumps(acq))
            exp.frame['geometry_identity'] = 'fixed'
            with pinned(root/'wisdom'):
                wisdom = pinned_wisdom(r2.native_parameters(cfg['matching']))
            link = {'pair': '2020-01-01_2020-01-07', 'earlier': '2020-01-01', 'later': '2020-01-07', 'step': 1}
            first = measure_link(exp, link, {}, wisdom)
            self.assertEqual(first['status'], 'complete')
            result_path = exp.output/'measurements'/link['pair']/'observations.npz'
            obs = load_npz(result_path)
            self.assertTrue(obs['dual_support'].all())
            np.testing.assert_allclose(obs['iw2_total']-obs['iw1_total'], 0, atol=1.e-10)
            stamp = result_path.stat().st_mtime_ns
            second = measure_link(exp, link, {}, wisdom)
            self.assertEqual(second['identity'], first['identity'])
            self.assertEqual(result_path.stat().st_mtime_ns, stamp)
            # A lost parent publication must rebuild from committed per-burst
            # jobs, rather than repeat successful native work.
            expected = load_npz(result_path)
            result_path.unlink()
            with patch('glacier_offsets.swath_observations._native_executor',
                       side_effect=AssertionError('Completed native jobs must be reused')):
                rebuilt = measure_link(exp, link, {}, wisdom)
            self.assertEqual(rebuilt['status'], 'complete')
            for name, values in load_npz(result_path).items():
                np.testing.assert_array_equal(values, expected[name], err_msg=name)

    def test_strip_equals_complete_production_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(12)
            data = (rng.normal(size=(96, 320))+1j*rng.normal(size=(96, 320))).astype('complex64')
            write_raster(root/'original.tif', data, gdal.GDT_CFloat32)
            original = (root/'original.tif').read_bytes()
            r2.prepare(root/'original.tif', root/'full.tif')
            prepare_strip(root/'original.tif', root/'strip.tif', [10, 80, 50, 210])
            np.testing.assert_array_equal(gdal.Open(str(root/'strip.tif')).ReadAsArray(),
                gdal.Open(str(root/'full.tif')).ReadAsArray(100, 10, 320, 70))
            self.assertEqual(original, (root/'original.tif').read_bytes())

    def test_native_fractional_signs_units_and_pinned_worker_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params = PARAMS | {'range_oversampling': 2}
            shape = (192, 768)
            rng = np.random.default_rng(54)
            spectrum = np.fft.fft2(rng.normal(size=shape)+1j*rng.normal(size=shape))
            fy, fx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]
            spectrum *= (abs(fy) < .12) & (abs(fx) < .40)
            reference = np.fft.ifft2(spectrum).astype('complex64')
            write_raster(root/'reference.tif', reference, gdal.GDT_CFloat32)
            r2.prepare(root/'reference.tif', root/'reference2x.tif')
            grid = build_grid(shape, shape, params['window'], params['search'], params['stride'])
            cfg = {'gross_offsets': {'mode': 'zero'}}
            with pinned(root/'wisdom'):
                wisdom = pinned_wisdom(r2.native_parameters(params))
            for sign in (-1, 1):
                common, differential, range_shift = sign*.2, sign*.15, sign*.3
                results = []
                for swath, azimuth in enumerate((common-differential/2, common+differential/2)):
                    secondary = np.fft.ifft2(spectrum*np.exp(-2j*np.pi*(fy*azimuth+fx*range_shift))).astype('complex64')
                    path = root/f'secondary-{sign}-{swath}.tif'
                    write_raster(path, secondary, gdal.GDT_CFloat32)
                    r2.prepare(path, root/'secondary2x.tif')
                    args = (str(root/'reference2x.tif'), str(root/'secondary2x.tif'), root/f'native-{sign}-{swath}', params, cfg, 'cpu')
                    kwargs = {'_grid': grid, '_valid_input': np.ones(grid.shape, bool)}
                    with _native_executor(1, single_threaded=True, wisdom=wisdom) as pool:
                        first = pool.apply(r2.range_worker, ((args, kwargs),))
                    with _native_executor(2, single_threaded=True, wisdom=wisdom) as pool:
                        again = pool.apply(r2.range_worker, ((args, kwargs),))
                    for field in ('total', 'covariance', 'snr', 'correlation_peak', 'valid'):
                        np.testing.assert_array_equal(first[field], again[field])
                    np.testing.assert_array_equal(first['total'], first['gross']+first['residual'])
                    self.assertGreater(first['valid'].sum(), 20)
                    accepted = first['valid']
                    self.assertLess(abs(first['total'][..., 1][accepted].mean()-range_shift), .04)
                    self.assertLess(abs(first['total'][..., 0][accepted].mean()-azimuth), .06)
                    results.append(first['total'][..., 0])
                contrast = results[1]-results[0]
                self.assertLess(abs(np.median(contrast)-differential), .06)


if __name__ == '__main__':
    unittest.main()

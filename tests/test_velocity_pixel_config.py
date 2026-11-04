"""Configuration changes must reach the fitter without altering legacy defaults."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import yaml

from glacier_offsets.velocity_io import digest, write_json
from glacier_offsets.velocity_pixel import coverage, screen_years, solve
from glacier_offsets.velocity_pixel_config import (
    DEFAULT_PARAMETERS, display_masks, load_config, parameters_from_recipe, validate_parameters,
)
from test_velocity_pixel import fixture

from glacier_offsets import velocity_pixel_inversion as runner


class ConfigTests(unittest.TestCase):
    def test_paths_defaults_and_partial_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'experiment.yaml'
            config.write_text('input_directory: cached\noutput: result\ncoverage:\n  min_pairs_standard: 45\n')
            result = load_config(config)
            self.assertEqual(result['input_directory'], str((Path(tmp) / 'cached').resolve()))
            self.assertEqual(result['output'], str((Path(tmp) / 'result').resolve()))
            self.assertEqual(result['coverage']['min_pairs_standard'], 45)
            self.assertEqual(result['coverage']['min_pairs_per_year'], 20)
            self.assertEqual(result['execution']['batch_size'], 512)
        self.assertFalse(DEFAULT_PARAMETERS['screening']['remove_mixed_windows'])
        self.assertTrue(parameters_from_recipe({})['screening']['remove_mixed_windows'])
        self.assertFalse(parameters_from_recipe({'parameters':DEFAULT_PARAMETERS})['screening']['remove_mixed_windows'])

    def test_invalid_values_and_typos_fail(self):
        invalid = [
            {'coverage': {'min_pairs_standrad': 45}},
            {'coverage': {'min_pairs_standard': 39}},
            {'coverage': {'min_pairs_per_year': True}},
            {'coverage': {'phase_bins': 3}},
            {'coverage': {'phase_bins': 128}},
            {'coverage': {'min_pairs_total': 40.5}},
            {'screening': {'min_peak': float('nan')}},
            {'screening': {'remove_mixed_windows': 'false'}},
            {'screening': {'remove_mixed_windows': 0}},
            {'screening': {'max_range_residual_m': 0}},
            {'uncertainty': {'cluster_days': -1}},
            {'display': {'min_abs_azimuth_projection': 1.1}},
            {'screening': None},
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_parameters(value)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.yaml'
            for text in ('output: a\noutput: b\n', 'input_directory: a\noutput: a\n',
                         'input_directory: a\noutput: b\nexecution:\n  batch_size: 0\n',
                         'input_directory: a\noutput: b\nstack: {}\n'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):load_config(path)

    def test_default_screen_and_solver_are_unchanged(self):
        field, axis, *_ = fixture()
        cov = np.array([np.eye(2), np.eye(2)])
        good, actual_axis, *_ = screen_years(field)
        explicit, explicit_axis, *_ = screen_years(field, DEFAULT_PARAMETERS['screening'])
        assert_array_equal(good, explicit);assert_array_equal(actual_axis, explicit_axis)
        implicit = solve(field, good, axis, cov)
        explicit = solve(field, good, axis, cov, cluster_days=30)
        for name in ('coefficients', 'coefficient_covariance', 'calendar_clusters'):
            assert_array_equal(implicit[name], explicit[name])

    def test_pair_coverage_is_separate_from_fit_and_bins_are_configurable(self):
        field, *_ = fixture();good = np.zeros_like(field['valid'])
        good[:, :25] = True;good[:, 60:85] = True;good[:, -1] = True
        old = coverage(field, good)
        self.assertTrue(old['fit_support'].all());self.assertFalse(old['standard_coverage'].any())
        options = validate_parameters({'coverage': {'min_pairs_standard': 50}})['coverage']
        lowered = coverage(field, good, options)
        self.assertTrue(lowered['standard_coverage'].all())
        assert_array_equal(old['fit_support'], lowered['fit_support'])
        for update in ({'min_pairs_per_year': 26}, {'min_pairs_total': 52}, {'min_span_days': 800}):
            options = validate_parameters({'coverage': update})['coverage']
            self.assertFalse(coverage(field, good, options)['fit_support'].any())
        options = validate_parameters({'coverage': {'phase_bins': 6, 'min_phase_bins': 6}})['coverage']
        self.assertTrue((coverage(field, good, options)['phase_bins'] == 6).all())

    def test_peak_and_residual_thresholds_reach_yearwise_screen(self):
        field, *_ = fixture()
        field['peak'][:, 8] = .05
        self.assertFalse(screen_years(field)[0][:, 8].any())
        loose = validate_parameters({'screening': {'min_peak': .04}})['screening']
        self.assertTrue(screen_years(field, loose)[0][:, 8].all())
        field['data'][0, 65] += [3, 6]
        self.assertFalse(screen_years(field, loose)[0][0, 65])
        loose.update(max_range_residual_m=4, max_azimuth_residual_m=7)
        self.assertTrue(screen_years(field, loose)[0][0, 65])
        # Lowering peak also affects the post-screen background used for the axis.
        field['peak'][:] = .05
        good, axis, *_ = screen_years(field, loose)
        self.assertTrue(good[:, 0].all());self.assertTrue(np.isfinite(axis).all())
        self.assertTrue(np.all(abs(axis[:, 1]) > .1))

    def test_cluster_length_changes_uncertainty_only(self):
        field, axis, *_ = fixture()
        field['data'] += np.random.default_rng(5).normal(size=field['data'].shape) * [.1, .3]
        cov = np.array([np.eye(2), np.eye(2)])
        before = solve(field, field['valid'], axis, cov)
        after = solve(field, field['valid'], axis, cov, cluster_days=60)
        assert_array_equal(before['coefficients'], after['coefficients'])
        self.assertTrue((after['calendar_clusters'] < before['calendar_clusters']).all())
        self.assertGreater(np.max(abs(before['coefficient_covariance']-after['coefficient_covariance'])), 1e-6)

    def test_display_settings_change_masks_only(self):
        args = [np.ones(3, bool), np.ones(3, bool), np.ones(3, bool), np.array([.05, .2, .6]),
                np.zeros(3), np.ones((3, 2))*.6, np.ones(3)*.1, np.ones(3)*.02, np.ones(3)*15]
        original = deepcopy(args)
        default = display_masks(*args, DEFAULT_PARAMETERS['display'])
        options = validate_parameters({'display': {'min_speed_m_day': .01, 'max_phase_sigma_deg': 10}})['display']
        changed = display_masks(*args, options)
        assert_array_equal(default['along'], [False, True, True])
        self.assertTrue(changed['along'].all());self.assertFalse(changed['phase_quality'].any())
        for before, after in zip(original, args):assert_array_equal(before, after)


class RunnerTests(unittest.TestCase):
    def prepared(self, directory):
        field, *_ = fixture();directory.mkdir()
        with h5py.File(directory / 'observations.h5', 'w') as h:
            for key in ('early','late','data','valid','peak','order','year_index','calendar_early','calendar_late'):
                h[key] = field[key]
            h['source_id'] = np.arange(5)
            h['file_index'] = field['year_index']
            h['geometry_by_file'] = field['geometry_by_year']
        with h5py.File(directory / 'solid_earth_tides.h5', 'w') as h:
            h['set_ra_m'] = np.zeros_like(field['data'])
        write_json(directory / 'inputs.json', {'cache_sha256': digest(directory / 'observations.h5')})
        write_json(directory / 'solid_earth_tides.json', {'output_sha256': digest(directory / 'solid_earth_tides.h5')})
        write_json(directory / 'noise.json', {'covariance_by_year': [np.eye(2), np.eye(2)]})

    def test_yaml_cli_runs_reuses_and_refuses_changed_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);self.prepared(root / 'cache')
            path = root / 'run.yaml'
            config = {'input_directory': 'cache', 'output': 'result',
                      'coverage': {'min_pairs_standard': 45}, 'execution': {'batch_size': 2}}
            path.write_text(yaml.safe_dump(config))
            runner.main(['--config', str(path)])
            result = root / 'result';frozen = digest(result / 'source_coefficients.h5')
            with h5py.File(result / 'source_coefficients.h5') as fits:
                self.assertTrue(fits['residual_vertical/estimable'][:].all())
                self.assertTrue(fits['standard_coverage'][:].all())
                field, *_ = fixture();good, axis, *_ = screen_years(field)
                expected = solve(field, good, axis, np.array([np.eye(2), np.eye(2)]))
                assert_allclose(fits['residual_vertical/coefficients'][:], expected['coefficients'], atol=1e-9)
            resolved = yaml.safe_load((result / 'inversion-config.resolved.yaml').read_text())
            self.assertEqual(resolved['coverage']['min_pairs_standard'], 45)
            recipe = json.loads((result / 'recipe.json').read_text())
            self.assertIn('>=45', recipe['standard_coverage'])
            self.assertEqual(recipe['noise_sha256'], digest(result / 'noise.json'))
            runner.main(['--config', str(path), '--batch-size', '3'])
            self.assertEqual(digest(result / 'source_coefficients.h5'), frozen)
            config['screening'] = {'min_peak': .1};path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(SystemExit, '2'):
                runner.main(['--config', str(path)])
            self.assertEqual(digest(result / 'source_coefficients.h5'), frozen)
            self.assertFalse((root / 'cache' / 'recipe.json').exists())

    def test_cli_rejects_invalid_config_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);path = root / 'bad.yaml'
            path.write_text('input_directory: absent\noutput: result\ncoverage:\n  min_pairs_totl: 40\n')
            with self.assertRaises(SystemExit):runner.main(['--config', str(path)])
            self.assertFalse((root / 'result').exists())


if __name__ == '__main__':unittest.main()

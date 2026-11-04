"""Physical contracts at the elapsed-day / iceutils operator boundary."""
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.interpolate import BSpline

from glacier_offsets.velocity_basis import harmonic, definition, design, PERIOD
from glacier_offsets.velocity_pixel_general import Layout
from glacier_offsets.velocity_pixel_runtime import module_fingerprints, verify_code
from glacier_offsets.velocity_temporal import matrix, timefn
from glacier_offsets.velocity_vertical import vertical_columns
from glacier_offsets.velocity_inspect import M2_PERIOD, O1_PERIOD


class TemporalOperatorTests(unittest.TestCase):
    def test_broadcast_missing_reversed_and_zero_intervals(self):
        early = np.array([[6209.123456], [np.nan], [6210.]])
        late = np.array([6209.123456, 6215., 6200.])
        actual = harmonic(early, late)
        self.assertEqual(actual.shape, (3, 3, 2))
        self.assertTrue(np.isnan(actual[1]).all())
        assert_array_equal(actual[0, 0], [0., 0.])
        for row in (0, 2):
            for col in range(3):
                a, b = early[row, 0], late[col]
                expected = [quad(lambda day: fun(2*np.pi*day/PERIOD), a, b, epsabs=1e-10)[0]
                            for fun in (np.cos, np.sin)]
                assert_allclose(actual[row, col], expected, atol=2e-11)
        assert_array_equal(harmonic(7., 7.), [0., 0.])
        self.assertTrue(np.isnan(harmonic(7., 7., observation='average')).all())
        self.assertEqual(harmonic(np.empty((0, 2)), np.empty((0, 2))).shape, (0, 2, 2))

    def test_average_and_sample_keep_fixed_utc_phase_and_day_units(self):
        early = np.array([6209.25, 6268.123456, 7363.999999])
        late = early + [6., 12., .001]
        integrated = harmonic(early, late)
        average = harmonic(early, late, observation='average')
        assert_allclose(average*(late-early)[:, None], integrated, atol=2e-11)
        assert_allclose(harmonic(late, early), -integrated, atol=1e-14)
        assert_allclose(harmonic(late, early, observation='average'), average, atol=1e-14)
        expected = np.column_stack((np.cos(2*np.pi*early/PERIOD), np.sin(2*np.pi*early/PERIOD)))
        assert_allclose(harmonic(early, observation='sample'), expected, atol=2e-11)

    def test_spline_dictionary_matches_independent_displacement_integrals(self):
        early = np.array([6209.25, 6250.5, 6500.125, 6800.75])
        late = early + [6., 12., 30., 24.]
        basis = definition(early, late, 'spline_l2')
        actual, names, penalized = design(early, late, basis, 'free_en', [1., 0.])
        background = [late-early, ((late-basis['reference_day'])**2-(early-basis['reference_day'])**2)/2]
        for spec in basis['splines']:
            knots = spec['center']+spec['scale']*np.arange(-2, 3)
            primitive = BSpline.basis_element(knots, extrapolate=False).antiderivative()
            background.append(primitive(np.clip(late, knots[0], knots[-1]))
                              - primitive(np.clip(early, knots[0], knots[-1])))
        expected = np.stack(background, axis=-1)
        assert_allclose(actual[:, 0, :2*len(background):2], expected, atol=2e-10)
        assert_allclose(actual[:, 1, 1:2*len(background):2], expected, atol=2e-10)
        self.assertEqual(names[-4:], ['Msf_C_east', 'Msf_S_east', 'Msf_C_north', 'Msf_S_north'])
        assert_array_equal(penalized, np.arange(4, 2*len(background)))

    def test_vertical_observes_height_difference_not_velocity_integral(self):
        early = np.array([[6209.123456, np.nan, 6210.]])
        late = np.array([[6209.823456, 6215., 6210.]])
        expected = []
        for period in (M2_PERIOD, O1_PERIOD):
            omega = 2*np.pi/period
            expected.extend([np.cos(omega*late)-np.cos(omega*early),
                             np.sin(omega*late)-np.sin(omega*early)])
        assert_allclose(vertical_columns(early, late), np.nan_to_num(np.stack(expected, axis=-1)), atol=5e-10)

    def test_spline_samples_have_compact_support_even_far_from_center(self):
        samples = matrix((('bspline', 6209., 60., 3),), [6209., 6000., 16000.])
        assert_allclose(samples[:, 0], [2/3, 0., 0.], atol=1e-15)

    def test_regimes_include_gap_years_crossings_and_leap_day(self):
        from glacier_offsets.velocity_basis import utc_days
        layout = Layout([2017, 2019, 2020])
        early = utc_days(['2016-12-31', '2018-12-30', '2019-12-31', '2020-02-28', '2021-01-01'])
        late = early + [3., 3., 3., 3., 3.]
        assert_array_equal(layout.overlaps(early, late), [[3, 0, 0], [2, 1, 0], [0, 1, 2], [0, 0, 3], [0, 0, 3]])
        self.assertTrue(np.isnan(layout.overlaps(np.nan, 5.)).all())
        assert_array_equal(layout.overlaps(early, early), np.zeros((5, 3)))

    def test_backend_changes_are_part_of_replay_identity(self):
        fingerprints = module_fingerprints(('velocity_temporal',))
        self.assertIn('iceutils.tseries.timefn.Trigonometry', fingerprints)
        verify_code(fingerprints)
        fingerprints['iceutils.tseries.timefn.Trigonometry'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'changed'):
            verify_code(fingerprints)

    def test_cross_validation_cannot_reuse_a_changed_operator(self):
        from glacier_offsets.velocity_recovery import implementation, cv_numerical_recipe
        cfg = {'start': None, 'end': None, 'sampling': {}, 'models': {}}
        recipe = {'source_fingerprints': [], 'regions_sha256': 'regions', 'implementation': implementation()}
        original = cv_numerical_recipe(cfg, recipe)
        for module in ('velocity_temporal.py', 'iceutils.tseries.timefn.Spline'):
            changed = dict(recipe, implementation=recipe['implementation'] | {module: 'changed'})
            self.assertNotEqual(cv_numerical_recipe(cfg, changed), original)

    def test_stale_backend_has_actionable_error(self):
        timefn.cache_clear()
        try:
            with patch('iceutils.tseries', None), self.assertRaisesRegex(ImportError, 'PR #28'):
                timefn()
        finally:
            timefn.cache_clear()


if __name__ == '__main__':
    unittest.main()

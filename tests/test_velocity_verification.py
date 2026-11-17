"""RMS replay stays strict while respecting diagnostic storage precision."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from glacier_offsets.velocity_verification import check_physical_rms, check_stored_rms
from glacier_offsets.velocity_pixel_general import Layout, solve, predict
from test_nisar_performance import field


class VerificationTests(unittest.TestCase):
    def test_large_rms_reproduces_false_failure_at_float32_storage(self):
        f = field(ns=12, nt=95)
        f['order'] = np.zeros(95)
        f['calendar_early'] = f['early'].mean(axis=0)
        f['calendar_late'] = f['late'].mean(axis=0)
        f['year_index'] = (f['calendar_early'] >= 9132.).astype(int)
        f['data'] += np.random.default_rng(801).normal(0, 200., f['data'].shape)
        axis = np.tile([2., 1.], (12, 1))/np.sqrt(5.)
        covariance = np.tile([[40000., 10000.], [10000., 90000.]], (2, 1, 1))
        layout = Layout([2024, 2025], [])
        good = f['valid']
        result = solve(f, layout, good, axis, covariance, robust='huber')
        self.assertTrue(result['estimable'].all())
        residual = np.where(good[..., None], f['data']-predict(f, layout, result['coefficients'], result['names'], axis), 0.)
        independent = np.sqrt(np.sum(residual**2, axis=1)/good.sum(axis=1)[:, None])
        stored = result['rms_m'].astype('f4')
        self.assertGreater(np.max(abs(independent-stored)), 1e-6)
        self.assertLess(check_physical_rms(independent, result['rms_m']), 1e-10)
        self.assertGreater(check_stored_rms(result['rms_m'], stored), 1e-6)

    def test_wrong_physical_prediction_still_fails_at_original_tolerance(self):
        with self.assertRaisesRegex(ValueError, 'source 42.*difference=0.0001'):
            check_physical_rms([[200.0001, 100.]], [[200., 100.]], source_ids=[42])

    def test_storage_bounds_and_mismatched_diagnostics(self):
        for dtype in ('f4', 'f8'):
            values = np.array([[0., 1., 32., 128., 1e5]], dtype=dtype)
            check_stored_rms(values.astype(float), values)
            with self.assertRaisesRegex(ValueError, 'Stored radar RMS'):
                check_stored_rms(values.astype(float)+1., values)
        for bad in (np.nan, np.inf, -1.):
            with self.assertRaises(ValueError): check_stored_rms([[bad]], np.array([[1.]], 'f4'))
            with self.assertRaises(ValueError): check_physical_rms([[bad]], [[1.]])
        check_stored_rms([[123.456789]], np.array([[123.456789]], 'f4'))


if __name__ == '__main__': unittest.main()

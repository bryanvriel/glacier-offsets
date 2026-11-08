"""Regression checks for covariance cliffs, missing quality and pair transfer."""
from copy import deepcopy
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from glacier_offsets.velocity_basis import definition, design
from glacier_offsets.velocity_fit import covariance_ra, fit


def evans_noise():
    """Frozen Evans pilot model: the lower-domain fallback gave 3,000x precision."""
    return {
        'covariance_ra_m2': [[.22820270109529964, .11246938034764503],
                             [.11246938034764505, 1.9423586242132762]],
        'floors_m': [.183126596763899, .29606948642036246],
        'correlation': .16893094168319808,
        'quality_models': {'snr_peak': {
            'coefficients': [[12.237355441611212, -2.9616468846507193, 0.],
                             [14.280958845812494, -3.0587911816229805, 0.]],
            'domain': [[1.8081468343734741, -3.8929944038391113],
                       [5.348765850067139, -.26471516489982605]]}},
    }


def joint_noise():
    return {'covariance_ra_m2': [[1, .6], [.6, 1]], 'floors_m': [.01, .01],
            'correlation': .6, 'quality_models': {'snr_peak': {
                'coefficients': [[3., -1., -1.], [3., -1., -1.]],
                'domain': [[0., np.log(.1)], [np.log(100.), 0.]]}}}


class CovarianceTransferTests(unittest.TestCase):
    def test_evans_lower_boundary_has_no_precision_cliff(self):
        noise = evans_noise()
        edge = np.exp(noise['quality_models']['snr_peak']['domain'][0][0])
        cov, details = covariance_ra(noise, edge*np.array([.1, 1-1e-6, 1., 1+1e-6]),
                                     [.2]*4, True, return_diagnostics=True)
        assert_allclose(cov[0], cov[2], rtol=1e-14)
        assert_allclose(cov[1], cov[2], rtol=1e-14)
        assert_allclose(cov[3], cov[2], rtol=4e-6)
        # These scales, rather than the old 0.48/1.39 m fallback, must persist.
        assert_allclose(np.sqrt(np.diag(cov[0])), [31.22227, 79.44839], rtol=1e-5)
        assert_allclose(np.linalg.inv(cov[0]), np.linalg.inv(cov[2]), rtol=1e-14)
        self.assertEqual(details['status'][0], 'clipped')
        self.assertTrue(details['clipped_low'][0, 0])

    def test_all_finite_domain_edges_clamp_continuously(self):
        noise = joint_noise()
        for feature, edges, other in [(0, [1., 100.], .5), (1, [.1, 1.], 10.)]:
            for edge in edges:
                values = np.full((3, 2), other)
                values[:, feature] = edge*np.array([1-1e-7, 1., 1+1e-7])
                cov = covariance_ra(noise, values[:, 0], values[:, 1], True)
                assert_allclose(cov[0], cov[1], rtol=2e-7)
                assert_allclose(cov[2], cov[1], rtol=2e-7)
        cov, details = covariance_ra(noise, [.01, 1., 100., 1e6], [.01, .1, 1., 100.],
                                     True, return_diagnostics=True)
        assert_allclose(cov[0], cov[1])
        assert_allclose(cov[2], cov[3])
        assert_array_equal(details['clipped_low'][0], [True, True])
        assert_array_equal(details['clipped_high'][3], [True, True])

    def test_precision_is_monotone_through_clamped_boundaries(self):
        noise = joint_noise()
        snr = np.geomspace(.01, 1e4, 100)
        cov = covariance_ra(noise, snr, np.full(100, .5), True)
        # With common slopes this must hold in all directions, including RA sums.
        self.assertGreaterEqual(np.linalg.eigvalsh(cov[:-1]-cov[1:]).min(), -1e-10)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0.)

    def test_missing_snr_is_worst_boundary_with_pooled_precision_floor(self):
        noise = joint_noise()
        snr = [np.nan, np.inf, -np.inf, 0., -10., 1.]
        cov, details = covariance_ra(noise, snr, [.5]*6, True, return_diagnostics=True)
        for candidate in cov[:5]:
            assert_allclose(candidate, cov[5])
            self.assertGreaterEqual(np.linalg.eigvalsh(candidate-noise['covariance_ra_m2']).min(), -1e-12)
        assert_array_equal(details['status'][:5], ['missing_quality']*5)
        self.assertTrue(details['missing_features'][:5, 0].all())

    def test_missing_peak_does_not_switch_to_overconfident_snr_model(self):
        noise = joint_noise()
        noise['quality_models']['snr'] = {
            'coefficients': [[-10., 0.], [-10., 0.]], 'domain': [[0.], [np.log(100.)]]}
        cov, details = covariance_ra(noise, [10.]*6, [np.nan, np.inf, -np.inf, 0., -1., .1],
                                     True, return_diagnostics=True)
        for candidate in cov[:5]:
            assert_allclose(candidate, cov[5])
        assert_array_equal(details['model'], ['snr_peak']*6)
        self.assertTrue(details['missing_features'][:5, 1].all())

    def test_missing_quality_never_beats_pooled_in_a_correlated_direction(self):
        noise = joint_noise()
        noise['quality_models']['snr_peak']['coefficients'] = [[-10., -.1, -.2], [-8., -.2, -.1]]
        cov = covariance_ra(noise, [np.nan, 10.], [.5, np.nan], True)
        baseline = np.array(noise['covariance_ra_m2'])
        for candidate in cov:
            self.assertGreaterEqual(np.linalg.eigvalsh(candidate-baseline).min(), -1e-12)
            self.assertLessEqual(np.linalg.eigvalsh(np.linalg.inv(candidate)-np.linalg.inv(baseline)).max(), 1e-12)

    def test_existing_directional_precision_cap_is_preserved(self):
        noise = joint_noise()
        noise['quality_models']['snr_peak']['coefficients'] = [[-10., -.1, -.2], [-8., -.2, -.1]]
        cov = covariance_ra(noise, [1., 10., 100., 1e9], [.1, .5, 1., 100.], True)
        baseline = np.array(noise['covariance_ra_m2'])
        for candidate in cov:
            self.assertGreaterEqual(np.linalg.eigvalsh(candidate-.01*baseline).min(), -1e-12)
            self.assertLessEqual(np.linalg.eigvalsh(np.linalg.inv(candidate)-100*np.linalg.inv(baseline)).max(), 1e-9)

    def test_snr_only_model_uses_same_boundary_policy(self):
        noise = joint_noise()
        noise['quality_models'] = {'snr': {'coefficients': [[3., -1.], [3., -1.]],
                                          'domain': [[0.], [np.log(100.)]]}}
        cov, details = covariance_ra(noise, [.1, 1., 100., 1000.], [np.nan]*4, True,
                                     return_diagnostics=True)
        assert_allclose(cov[0], cov[1])
        assert_allclose(cov[2], cov[3])
        assert_array_equal(details['model'], ['snr']*4)
        self.assertFalse(details['missing_features'].any())

    def test_absent_model_and_constant_mode_are_explicit(self):
        noise = joint_noise()
        noise['quality_models'] = {}
        expected = np.tile(noise['covariance_ra_m2'], (2, 1, 1))
        cov, details = covariance_ra(noise, [np.nan, 1.], [np.nan, .5], True,
                                     return_diagnostics=True)
        assert_allclose(cov, expected)
        assert_array_equal(details['status'], ['no_quality_model']*2)
        cov, details = covariance_ra(noise, [np.nan, 1.], [np.nan, .5],
                                     return_diagnostics=True)
        assert_allclose(cov, expected)
        assert_array_equal(details['status'], ['constant']*2)

    def test_pair_covariance_map_preserves_boundary_fix_and_diagnostics(self):
        noise = evans_noise()
        edge = np.exp(noise['quality_models']['snr_peak']['domain'][0][0])
        pairs = [np.array([[3., -.2], [-.2, 2.]]), np.array([[.1, .05], [.05, 4.]])]
        paired = deepcopy(noise)
        paired['pair_calibration'] = {'pairs': {str(i): {'covariance_ra_m2': p.tolist()} for i,p in enumerate(pairs)}}
        pooled, expected = covariance_ra(noise, [edge/2]*2, [.2]*2, True, return_diagnostics=True)
        actual, details = covariance_ra(paired, [edge/2]*2, [.2]*2, True, ['0', '1'],
                                        return_diagnostics=True)
        for i,pair in enumerate(pairs):
            t = np.linalg.cholesky(pair) @ np.linalg.inv(np.linalg.cholesky(noise['covariance_ra_m2']))
            assert_allclose(actual[i], t @ pooled[i] @ t.T)
        for key in expected:
            assert_array_equal(details[key], expected[key])

    def test_low_quality_outlier_does_not_take_over_harmonic_fit(self):
        early = np.arange(120)*6.+6209.25
        late = early+6
        basis = definition(early, late)
        matrix = design(early, late, basis, 'along_flow', [1., 0.])[0]
        truth = np.array([1., .2, .08, .06])
        data = np.einsum('ijk,k->ij', matrix, truth)
        data[20] += [100., -100.]
        noise = {'covariance_ra_m2': np.eye(2)*.01, 'floors_m': [.01, .01], 'correlation': 0.,
                 'quality_models': {'snr': {'coefficients': [[np.log(100.), -2.]]*2,
                                            'domain': [[0.], [np.log(100.)]]}}}
        snr = np.full(120, 100.);snr[20] = .9
        cov = covariance_ra(noise, snr, np.full(120, np.nan), True)
        repaired = fit(early, late, data, cov, basis, 'along_flow', [1.,0.])
        legacy = cov.copy();legacy[20] = noise['covariance_ra_m2']
        broken = fit(early, late, data, legacy, basis, 'along_flow', [1.,0.])
        self.assertLess(np.linalg.norm(repaired['coefficients'][-2:]-truth[-2:]), 1e-3)
        self.assertGreater(np.linalg.norm(broken['coefficients'][-2:]-truth[-2:]), .1)

    def test_bad_shapes_and_model_domains_fail_explicitly(self):
        noise = joint_noise()
        with self.assertRaisesRegex(ValueError, 'one-dimensional'):
            covariance_ra(noise, [1.,2.], [.5], True)
        noise['quality_models']['snr_peak']['domain'] = [[1.,0.], [0.,1.]]
        with self.assertRaisesRegex(ValueError, 'calibration domain'):
            covariance_ra(noise, [1.], [.5], True)


if __name__ == '__main__':
    unittest.main()

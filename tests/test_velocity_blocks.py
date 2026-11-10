"""Block aggregation: shared-cluster covariance, debiasing, tests and FDR."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from glacier_offsets.velocity_blocks import BlockAccumulator, benjamini_hochberg, block_keys, combine, statistics
from glacier_offsets.velocity_pixel_general import Layout, solve
from test_velocity_pixel import fixture


class BlockTests(unittest.TestCase):
    def test_shared_errors_are_not_averaged_away(self):
        g = 10
        rng = np.random.default_rng(3)
        one = rng.normal(size=(1, g, 2))
        acc = BlockAccumulator(2, g)
        point = np.tile(np.eye(2), (2, 1, 1))
        # Block 0: two sources with identical (fully shared) influences.
        acc.add([0, 0], np.zeros((2, 2)), np.concatenate([one, one]), np.ones(2), point)
        # Block 1: two sources whose influences live in disjoint clusters.
        a, b = one.copy(), one.copy()
        a[:, g//2:] = 0
        b[:, :g//2] = 0
        acc.add([1, 1], np.zeros((2, 2)), np.concatenate([a, b]), np.ones(2), point)
        out = acc.finish()
        single = one[0].T@one[0]
        assert_allclose(out['covariance'][0], single)
        assert_allclose(out['covariance'][1], single/4)
        self.assertEqual(out['clusters'][0], g)

    def test_accumulator_matches_direct_regional_sandwich_and_combine(self):
        field, axis, truth, *_ = fixture()
        layout = Layout([2017, 2018])
        cov = np.tile(np.diag([.01, .04]), (2, 1, 1))
        rng = np.random.default_rng(5)
        field['data'] = field['data']+rng.normal(size=field['data'].shape)*[.1, .2]
        fit = solve(field, layout, field['valid'], axis, cov, robust='huber', robust_scale='source')
        labels = np.array([0, 0, 1, 1, 1])
        acc = BlockAccumulator(2, fit['cluster_influence'].shape[1])
        acc.add(labels, fit['cs'], fit['cluster_influence'], fit['cluster_cr1'], fit['coefficient_covariance'])
        out = acc.finish()
        for b in (0, 1):
            take = labels == b
            infl = fit['cluster_influence'][take].mean(0)
            assert_allclose(out['covariance'][b], fit['cluster_cr1'][take].mean()*infl.T@infl, rtol=1e-12)
            assert_allclose(out['mean'][b], fit['cs'][take].mean(0))
        # The block mean of a single source is that source's own sandwich.
        acc1 = BlockAccumulator(1, fit['cluster_influence'].shape[1])
        acc1.add([0], fit['cs'][:1], fit['cluster_influence'][:1], fit['cluster_cr1'][:1], fit['coefficient_covariance'][:1])
        assert_allclose(acc1.finish()['covariance'][0], fit['coefficient_covariance'][0], rtol=1e-12)
        assert_allclose(combine(out['influence'], out['cr1'], [1, 0]), out['covariance'][0], rtol=1e-12)
        # Difference of two blocks sharing clusters is not the sum of their variances.
        diff = combine(out['influence'], out['cr1'], [1, -1])
        self.assertFalse(np.allclose(diff, out['covariance'][0]+out['covariance'][1]))

    def test_statistics_debias_and_small_sample_reference(self):
        mean = np.array([[0., 0.], [.3, .4]])
        cov = np.tile(np.eye(2)*.01, (2, 1, 1))
        s = statistics(mean, cov, [24, 24])
        assert_allclose(s['amplitude'], [0., .5])
        assert_allclose(s['amplitude_debiased'], [0., np.sqrt(.25-.02)])
        assert_allclose(s['wald'], [0., 25.])
        self.assertEqual(s['p_value'][0], 1.)
        inflated = statistics(mean, cov, [24, 24], inflation=2.)
        assert_allclose(inflated['wald'], [0., 25/4])
        # The F reference is more conservative than chi-square with 2 dof.
        self.assertGreater(s['p_value'][1], np.exp(-25/2))
        assert_array_equal(np.isnan(statistics(mean, cov, [2, 24])['p_value']), [True, False])

    def test_benjamini_hochberg(self):
        p = np.array([.001, .008, .039, .041, .042, .06, .074, .205, np.nan])
        assert_array_equal(benjamini_hochberg(p, .05), [True, True, False, False, False, False, False, False, False])
        assert_array_equal(benjamini_hochberg(np.array([.5, np.nan])), [False, False])

    def test_null_block_rejection_rate_with_shared_cluster_errors(self):
        base, axis, truth, *_ = fixture()
        layout = Layout([2017, 2018])
        cov = np.tile(np.diag([.04, .16]), (2, 1, 1))
        rng = np.random.default_rng(11)
        ns = 20
        idx = np.arange(ns) % 5
        field = {k: (v[idx] if k in ('early', 'late', 'data', 'peak', 'valid') else v) for k, v in base.items()}
        field['geometry_by_year'] = base['geometry_by_year'][:, idx]
        ax = axis[idx]
        from glacier_offsets.velocity_basis import harmonic
        from glacier_offsets.velocity_pixel_general import pair_geometry
        signal = (np.einsum('stj,sj->st', harmonic(field['early'], field['late']), truth[idx])[..., None]
                  * np.einsum('stci,si->stc', pair_geometry(field)[..., :2], ax))
        clean = field['data']-signal
        p, design = [], []
        for _ in range(150):
            shared = rng.normal(size=(clean.shape[1], 2))*[.3, .6]  # pair errors common to all sources
            field['data'] = clean+shared[None]+rng.normal(size=clean.shape)*[.1, .2]
            fit = solve(field, layout, field['valid'], ax, cov)
            acc = BlockAccumulator(1, fit['cluster_influence'].shape[1])
            acc.add(np.zeros(ns, int), fit['cs'], fit['cluster_influence'], fit['cluster_cr1'], fit['coefficient_covariance'])
            out = acc.finish()
            p.append(statistics(out['mean'], out['covariance'], out['clusters'])['p_value'][0])
            design.append(out['design_effect'][0])
        self.assertGreater(np.mean(design), 3.)  # shared errors dominate the block variance
        rate = np.mean(np.array(p) < .05)
        self.assertLess(rate, .13)
        self.assertGreater(np.mean(np.array(p) < .5), .3)

    def test_block_keys(self):
        ix, iy = block_keys([0., 2999., 3000., -1.], [0., 0., 0., 0.], 3000.)
        assert_array_equal(ix, [0, 0, 1, -1])


if __name__ == '__main__':
    unittest.main()

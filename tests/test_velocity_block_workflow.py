"""Shared block-tool steps: halves, pair subsets, classes, blocks and accumulator merging."""
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from glacier_offsets.velocity_basis import utc_days
from glacier_offsets.velocity_block_workflow import (assign_blocks, classify, fit_pairs, median_split_date, per_block,
                                                     split_day_number, split_halves, subset_covariances, subset_pairs)
from glacier_offsets.velocity_blocks import BlockAccumulator
from glacier_offsets.velocity_pixel_config import validate_parameters
from glacier_offsets.velocity_pixel_general import Layout, solve
from test_velocity_pixel import fixture


class BlockWorkflowTests(unittest.TestCase):
    def test_halves_are_acquisition_disjoint(self):
        early = np.array([0., 6., 12., 18., 24.])+.3
        late = early+6
        late[2] = 30.3  # long pair with midpoint after the split
        parts, shared = split_halves(early, late, 15.)
        # Day 12 is the late acquisition of pair 1 (half A) and the early one of pair 2 (half B).
        self.assertEqual(shared, [12.])
        assert_array_equal(parts['A'], [0])
        assert_array_equal(parts['B'], [3, 4])
        parts, shared = split_halves(early[[0, 1, 3, 4]], late[[0, 1, 3, 4]], 15.)
        assert_array_equal(parts['A'], [0, 1])
        assert_array_equal(parts['B'], [2, 3])
        self.assertEqual(shared, [])

    def test_split_dates(self):
        self.assertEqual(split_day_number('2018-06-12'), float(utc_days('2018-06-12')))
        early = utc_days(['2017-01-01', '2018-06-01', '2019-01-01'])
        self.assertEqual(median_split_date(early, early+6), '2018-06-04')

    def test_subset_of_all_pairs_reproduces_the_full_fit(self):
        field, axis, *_ = fixture()
        layout = Layout([2017, 2018])
        cov = np.tile(np.diag([.01, .04]), (2, 1, 1))
        params = validate_parameters({})
        full, use = fit_pairs(field, layout, cov, params)
        columns = np.arange(field['early'].shape[1])
        sub, sublayout, present = subset_pairs(field, layout, columns)
        part, use_part = fit_pairs(sub, sublayout, subset_covariances(cov, present, columns), params)
        assert_array_equal(present, [0, 1])
        assert_array_equal(use, use_part)
        assert_array_equal(part['cs'], full['cs'])

    def test_one_year_subset_matches_an_independent_one_year_field(self):
        field, axis, truth, *_ = fixture()
        layout = Layout([2017, 2018])
        cov = np.tile(np.diag([.01, .04]), (2, 1, 1))
        columns = np.flatnonzero(field['year_index'] == 1)
        sub, sublayout, present = subset_pairs(field, layout, columns)
        self.assertEqual(sublayout.years, (2018,))
        self.assertEqual(sublayout.order_years, (2018,))
        assert_array_equal(present, [1])
        assert_array_equal(sub['year_index'], 0)
        assert_array_equal(sub['geometry_by_year'], field['geometry_by_year'][[1]])
        assert_array_equal(sub['data'], field['data'][:, columns])
        params = validate_parameters({'coverage': {'min_pairs_per_year': 5, 'min_pairs_total': 20,
                                                   'min_pairs_standard': 20, 'min_span_days': 270}})
        fit, use = fit_pairs(sub, sublayout, subset_covariances(cov, present, columns), params)
        self.assertTrue(use.all())
        alone = {k: (v[columns] if k in ('order', 'calendar_early', 'calendar_late')
                     else v[:, columns] if k in ('early', 'late', 'data', 'peak', 'valid') else v) for k, v in field.items()}
        alone['year_index'] = np.zeros(len(columns), int)
        alone['geometry_by_year'] = field['geometry_by_year'][[1]]
        expected, _ = fit_pairs(alone, Layout([2018]), cov[[1]], params)
        assert_array_equal(fit['cs'], expected['cs'])
        # Given the true flow axis the noise-free one-year fit is exact (the screen estimates its own axis).
        exact = solve(sub, sublayout, sub['valid'], axis, subset_covariances(cov, present, columns))
        assert_allclose(exact['cs'], truth, atol=1e-12)

    def test_pair_mode_covariance_follows_the_subset(self):
        by_year = np.arange(3)[:, None, None]*np.eye(2)
        by_pair = np.arange(10)[:, None, None]*np.eye(2)
        sub = subset_covariances({'by_year': by_year, 'by_pair': by_pair}, np.array([1, 2]), np.array([3, 7]))
        assert_array_equal(sub['by_year'][:, 0, 0], [1, 2])
        assert_array_equal(sub['by_pair'][:, 0, 0], [3, 7])
        assert_array_equal(subset_covariances(by_year, np.array([2]), np.array([3]))[:, 0, 0], [2])

    def test_classes_and_block_assignment(self):
        grounded, floating = classify([.01, .01, .3, np.nan], [0, 0, .6, np.nan], [.05, .5, 1., .01])
        assert_array_equal(grounded, [True, False, False, False])
        assert_array_equal(floating, [False, False, True, False])
        blocks = assign_blocks(np.array([-1., 1., 2999., 3001.]), np.array([0., 0., 10., 10.]), [3000.])
        keys, labels = blocks[3000.]
        assert_array_equal(keys, [[-1, 0], [0, 0], [1, 0]])
        assert_array_equal(labels, [0, 1, 1, 2])
        assert_array_equal(per_block([1., 2., 4., 8.], labels, 4, 'max'), [1., 4., 8., np.nan])

    def test_accumulator_merge_equals_one_pass(self):
        rng = np.random.default_rng(4)
        labels = np.array([0, 1, 1, 2, 0, 2])
        cs, influence = rng.normal(size=(6, 2)), rng.normal(size=(6, 5, 2))
        cr1, point = rng.uniform(1, 2, 6), np.tile(np.eye(2), (6, 1, 1))
        whole = BlockAccumulator(3, 5)
        whole.add(labels, cs, influence, cr1, point)
        first, second = BlockAccumulator(3, 5), BlockAccumulator(3, 5)
        first.add(labels[:4], cs[:4], influence[:4], cr1[:4], point[:4])
        second.add(labels[4:], cs[4:], influence[4:], cr1[4:], point[4:])
        merged, single = first.merge(second).finish(), whole.finish()
        for key in ('mean', 'covariance', 'influence', 'cr1', 'design_effect'):
            assert_allclose(merged[key], single[key], atol=1e-14)
        with self.assertRaisesRegex(ValueError, 'different'):
            first.merge(BlockAccumulator(3, 4))


if __name__ == '__main__':
    unittest.main()

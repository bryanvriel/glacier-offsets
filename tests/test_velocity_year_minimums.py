"""Per-year clean-pair minimums scaled by each calendar year's covered span."""
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from glacier_offsets.velocity_basis import utc_days
from glacier_offsets.velocity_pixel_config import validate_parameters
from glacier_offsets.velocity_pixel_general import Layout, coverage, year_pair_minimums
from glacier_offsets.velocity_workflow import manifest_year_minimums
from glacier_offsets.velocity_workflow_config import impossible_settings
from test_velocity_pixel import fixture


def calendar(start, stop, step=6.):
    early = np.arange(float(utc_days(start)), float(utc_days(stop))-step, step)+.3
    return early, early+step


class YearMinimumTests(unittest.TestCase):
    def setUp(self):
        # From late June 2017, all of 2018, to early June 2019 (the Evans record shape).
        e1, l1 = calendar('2017-06-26', '2018-01-01')
        e2, l2 = calendar('2018-01-01', '2019-01-01')
        e3, l3 = calendar('2019-01-01', '2019-06-10')
        self.early, self.late = np.r_[e1, e2, e3], np.r_[l1, l2, l3]
        self.index = np.r_[np.zeros(len(e1), int), np.ones(len(e2), int), np.full(len(e3), 2)]
        self.years = [2017, 2018, 2019]

    def test_none_keeps_one_minimum(self):
        assert_array_equal(year_pair_minimums(self.early, self.late, self.index, self.years, 10), [10, 10, 10])

    def test_span_scales_partial_years_up(self):
        need = year_pair_minimums(self.early, self.late, self.index, self.years, 10, 'span')
        assert_array_equal(need, [6, 10, 5])  # 186/365 -> 5.1 -> 6; 2018 full -> 10; 155/365 -> 4.2 -> 5
        assert_array_equal(year_pair_minimums(self.early, self.late, self.index, self.years, 1, 'span'), [1, 1, 1])
        with self.assertRaisesRegex(ValueError, 'scaling'):
            year_pair_minimums(self.early, self.late, self.index, self.years, 10, 'density')

    def test_coverage_uses_scaled_minimum(self):
        field, *_ = fixture()
        layout = Layout([2017, 2018])
        good = field['valid'].copy()
        good[:, np.flatnonzero(field['year_index'] == 0)[8:]] = False  # 8 clean pairs in 2017 for every source
        options = validate_parameters({'coverage': {'min_pairs_per_year': 10, 'min_pairs_total': 40,
                                                    'min_pairs_standard': 40, 'min_span_days': 300}})['coverage']
        self.assertFalse(coverage(field, layout, good, options)['fit_support'].any())
        # Truncate 2017 to its last 70 days: required 2017 pairs become ceil(10*70/365) = 2.
        late2017 = np.flatnonzero(field['year_index'] == 0)
        keep = np.r_[late2017[-12:], np.flatnonzero(field['year_index'] == 1)]
        sub = {k: (v[keep] if k in ('order', 'year_index', 'calendar_early', 'calendar_late')
                   else v[:, keep] if k in ('early', 'late', 'data', 'peak', 'valid') else v) for k, v in field.items()}
        g = good[:, keep].copy()
        g[:, :12] = False
        g[:, :2] = True
        scaled = options | {'min_pairs_per_year_scaling': 'span'}
        result = coverage(sub, layout, g, scaled)
        self.assertEqual(result['min_pairs_by_year'][0], 2)
        self.assertTrue(result['fit_support'].all())
        self.assertFalse(coverage(sub, layout, g, options)['fit_support'].any())

    def test_feasibility_and_manifest_agree(self):
        pairs = [{'calendar_early_day': e, 'calendar_late_day': l, 'year': self.years[k]}
                 for e, l, k in zip(self.early, self.late, self.index)]
        manifest = {'years': self.years, 'pairs': pairs}
        coverage_options = validate_parameters({'coverage': {'min_pairs_per_year': 10,
                                                             'min_pairs_per_year_scaling': 'span'}})['coverage']
        minimums = manifest_year_minimums(manifest, coverage_options)
        self.assertEqual(minimums, {2017: 6, 2018: 10, 2019: 5})
        config = {'coverage': validate_parameters({'coverage': {'min_pairs_per_year': 10, 'min_pairs_total': 10,
                                                                'min_pairs_standard': 10, 'min_span_days': 10}})['coverage']}
        problems = impossible_settings(config, {2017: 5, 2018: 60, 2019: 5}, 700., minimums)
        self.assertEqual(problems, ['2017 has 5 pairs, fewer than coverage.min_pairs_per_year=10 scaled to 6 for its covered span'])
        self.assertEqual(len(impossible_settings(config, {2017: 5, 2018: 60, 2019: 5}, 700.)), 2)

    def test_choice_is_validated(self):
        self.assertEqual(validate_parameters({})['coverage']['min_pairs_per_year_scaling'], 'none')
        with self.assertRaisesRegex(ValueError, 'coverage.min_pairs_per_year_scaling'):
            validate_parameters({'coverage': {'min_pairs_per_year_scaling': 'density'}})


if __name__ == '__main__':
    unittest.main()

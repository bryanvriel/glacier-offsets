"""Interior cut margins, coverage partitions and non-overlap preservation."""
import unittest

import numpy as np

from glacier_offsets.stack_ownership import interior_ownership


def source(name, offset, window):
    return {'burst_id': name, 'offset': offset, 'valid_window': window}


def owners(sources):
    result = np.zeros((400, 80), 'uint16')
    count = np.zeros_like(result)
    for index, entry in enumerate(sources, 1):
        dy, dx = entry['offset']
        for a, b, c, d in entry['ownership_windows']:
            result[a+dy:b+dy, c+dx:d+dx] = index
            count[a+dy:b+dy, c+dx:d+dx] += 1
    return result, count


class InteriorOwnershipTests(unittest.TestCase):
    def test_shifted_range_support_keeps_unshared_pixels_and_one_owner(self):
        original = [source('upper', [0, 0], [4, 188, 4, 50]),
                    source('lower', [128, 10], [4, 188, 4, 50])]
        selected, policy = interior_ownership(original)
        self.assertNotIn('ownership_windows', original[0])
        self.assertEqual(policy['cuts'][0]['cut_global_line'], 160)
        actual, count = owners(selected)
        expected = np.zeros_like(actual)
        expected[4:188, 4:50] = 1
        expected[132:316, 14:60] = 2
        expected[132:160, 14:50] = 1
        np.testing.assert_array_equal(actual, expected)
        self.assertLessEqual(count.max(), 1)

    def test_gaps_single_sources_and_touching_ranges_are_not_trimmed(self):
        original = [source('a', [0, 0], [4, 100, 4, 40]),
                    source('b', [120, 0], [4, 100, 4, 40]),
                    source('c', [130, 0], [4, 100, 40, 60])]
        selected, policy = interior_ownership(original)
        self.assertEqual(policy['cuts'], [])
        for a, b in zip(original, selected):
            self.assertEqual(b['ownership_windows'], [a['valid_window']])
        single, _ = interior_ownership(original[:1])
        self.assertEqual(single[0]['ownership_windows'], [original[0]['valid_window']])

    def test_narrow_overlap_fails_and_minimum_margin_is_enforced(self):
        for offset in [181, 182, 190]:
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'edge margin'):
                interior_ownership([source('a', [0, 0], [0, 196, 0, 50]),
                                    source('b', [offset, 0], [0, 196, 0, 50])])
        _, policy = interior_ownership([source('a', [0, 0], [0, 196, 0, 50]),
                                       source('b', [180, 0], [0, 196, 0, 50])])
        self.assertEqual(policy['cuts'][0]['cut_global_line'], 188)

    def test_three_overlapping_sources_have_complete_disjoint_ownership(self):
        selected, _ = interior_ownership([source(str(i), [i*80, 0], [0, 200, 0, 50]) for i in range(3)])
        actual, count = owners(selected)
        np.testing.assert_array_equal(count[:360, :50], 1)
        self.assertTrue((actual[:140, :50] == 1).all())
        self.assertTrue((actual[140:220, :50] == 2).all())
        self.assertTrue((actual[220:360, :50] == 3).all())


if __name__ == '__main__':
    unittest.main()

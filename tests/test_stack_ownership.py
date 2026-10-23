"""Interior cut margins, coverage partitions and non-overlap preservation."""
import unittest
from itertools import permutations

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

    def test_interleaved_swaths_have_order_independent_complete_ownership(self):
        # Inventory is swath-major, so the first IW2 burst is earlier than
        # the second IW1 burst despite appearing later in the source list.
        original = [source('a_iw1', [0, 0], [4, 188, 4, 50]),
                    source('b_iw1', [128, 0], [4, 188, 4, 50]),
                    source('a_iw2', [48, 20], [4, 188, 4, 50]),
                    source('b_iw2', [160, 20], [4, 188, 4, 50])]
        chronological = sorted(original, key=lambda entry: entry['offset'][0])
        expected_sources, expected_policy = interior_ownership(chronological)
        expected, _ = owners(expected_sources)
        names = [entry['burst_id'] for entry in chronological]
        union = np.zeros_like(expected, dtype=bool)
        for entry in original:
            dy, dx = entry['offset']
            a, b, c, d = entry['valid_window']
            union[a+dy:b+dy, c+dx:d+dx] = True
        for order in permutations(original):
            with self.subTest(order=[e['burst_id'] for e in order]):
                selected, policy = interior_ownership(order)
                self.assertEqual([s['burst_id'] for s in selected], [s['burst_id'] for s in order])
                actual, count = owners(selected)
                # Remap categorical codes to the same geographic source IDs.
                codes = np.array([0] + [names.index(s['burst_id'])+1 for s in selected])
                np.testing.assert_array_equal(codes[actual], expected)
                np.testing.assert_array_equal(count, union.astype(count.dtype))
                cuts = lambda p: sorted(str(sorted(c.items())) for c in p['cuts'])
                self.assertEqual(cuts(policy), cuts(expected_policy))
        self.assertTrue(all('ownership_windows' not in s for s in original))
        # Across swaths the switch runs in range, regardless of radar time.
        self.assertEqual(expected[80, 36], names.index('a_iw1')+1)
        self.assertEqual(expected[80, 37], names.index('a_iw2')+1)

    def test_true_nested_or_equal_time_overlaps_still_fail_with_burst_ids(self):
        for first, second in [(source('a_iw1', [0, 0], [0, 200, 0, 50]),
                               source('b_iw1', [20, 20], [0, 100, 0, 50])),
                              (source('a_iw1', [0, 0], [0, 200, 0, 50]),
                               source('b_iw1', [0, 20], [0, 200, 0, 50]))]:
            for items in ([first, second], [second, first]):
                with self.assertRaisesRegex(ValueError, 'iw[12].*iw[12].*non-nested'):
                    interior_ownership(items)

    def test_range_cuts_allow_equal_or_nested_times_on_different_swaths(self):
        for offset, stop in [(0, 200), (20, 100)]:
            selected, policy = interior_ownership([
                source('a_iw1', [0, 0], [0, 200, 0, 50]),
                source('a_iw2', [offset, 20], [0, stop, 0, 50])])
            actual, count = owners(selected)
            self.assertEqual(policy['cuts'][0]['cut_global_sample'], 35)
            self.assertEqual(actual[offset+1, 34], 1)
            self.assertEqual(actual[offset+1, 35], 2)
            self.assertEqual(count[offset+1, :70].min(), 1)

    def test_range_margin_and_nested_range_are_checked(self):
        for dx in [35, 40]:
            with self.subTest(dx=dx), self.assertRaisesRegex(ValueError, 'samples.*edge margin'):
                interior_ownership([source('a_iw1', [0, 0], [0, 100, 0, 50]),
                                    source('a_iw2', [0, dx], [0, 100, 0, 50])])
        _, policy = interior_ownership([source('a_iw1', [0, 0], [0, 100, 0, 50]),
                                        source('a_iw2', [0, 34], [0, 100, 0, 50])])
        self.assertEqual(policy['cuts'][0]['cut_global_sample'], 42)
        for dx in [0, 10]:
            with self.subTest(dx=dx), self.assertRaisesRegex(ValueError, 'non-nested swath support'):
                interior_ownership([source('a_iw1', [0, 0], [0, 100, 0, 70]),
                                    source('a_iw2', [0, dx], [0, 100, 0, 50])])

    def test_staggered_range_fringe_is_partitioned_as_a_swath_union(self):
        # A one-sample fringe from the upper IW1 burst must not be treated as
        # a separate narrow cross-swath overlap at the four-burst junction.
        original = [source('a_iw1', [0, 0], [0, 188, 0, 50]),
                    source('b_iw1', [128, 1], [0, 188, 0, 50]),
                    source('a_iw2', [48, 20], [0, 188, 0, 50]),
                    source('b_iw2', [160, 21], [0, 188, 0, 50]),
                    source('a_iw3', [64, 40], [0, 188, 0, 40])]
        selected, _ = interior_ownership(original)
        _, count = owners(selected)
        union = np.zeros_like(count)
        for e in original:
            dy, dx = e['offset']; a,b,c,d = e['valid_window']
            union[a+dy:b+dy,c+dx:d+dx] = 1
        np.testing.assert_array_equal(count, union)


if __name__ == '__main__':
    unittest.main()

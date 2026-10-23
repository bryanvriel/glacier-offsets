"""Partial selection policy: optimal compatible coverage and deterministic ties."""
from itertools import product
import random
import unittest

from glacier_offsets.burst_download_selection import select_contiguous_subsets


def rows(ids, swath):
    return [{'granule':f'{swath}-{i}', 'swath':swath, 'burst_id':i} for i in ids]


class PartialSelectionTests(unittest.TestCase):
    def test_interior_gap_trims_neighboring_swath_with_deterministic_tie(self):
        available={'IW1':rows([10,11,13,14],'IW1'), 'IW2':rows(range(10,15),'IW2')}
        requested={r['granule'] for group in available.values() for r in group}
        selected,ranges=select_contiguous_subsets(available,requested)
        self.assertEqual(ranges,{'IW1':[10,11],'IW2':[10,12]})
        self.assertEqual(selected,{'IW1-10','IW1-11','IW2-10','IW2-11','IW2-12'})
        for group in available.values():group.reverse()
        self.assertEqual(select_contiguous_subsets(dict(reversed(list(available.items()))),requested),
                         (selected,ranges))

    def test_iw1_priority_precedes_total_aoi_count(self):
        available={'IW1':rows([10,11,12,14,15],'IW1'), 'IW2':rows(range(10,17),'IW2')}
        # End IW2 bursts are context. The preferred solution preserves three IW1
        # observations even though the later interval could keep more IW2 AOI.
        requested={f'IW1-{i}' for i in (10,11,12,14,15)}|{f'IW2-{i}' for i in (10,14,15,16)}
        _,ranges=select_contiguous_subsets(available,requested)
        self.assertEqual(ranges,{'IW1':[10,12],'IW2':[10,13]})

    def test_all_observed_swaths_must_retain_aoi_coverage(self):
        available={'IW1':rows([10,11],'IW1'), 'IW2':rows([15,16],'IW2')}
        requested={r['granule'] for group in available.values() for r in group}
        with self.assertRaisesRegex(ValueError,'every observed swath'):
            select_contiguous_subsets(available,requested)
        # Context alone cannot stand in for the selected swath's AOI imagery.
        available['IW2']=rows(range(10,17),'IW2')
        with self.assertRaisesRegex(ValueError,'every observed swath'):
            select_contiguous_subsets(available,requested)

    def test_duplicate_products_are_not_silently_chosen(self):
        available={'IW1':rows([10,10,11],'IW1')}
        with self.assertRaisesRegex(ValueError,'ambiguous duplicate'):
            select_contiguous_subsets(available,{'IW1-10','IW1-11'})

    def test_iw1_iw3_with_iw2_helper_and_single_swath(self):
        available={'IW1':rows([10,11,13],'IW1'),'IW3':rows([10,11,12],'IW3')}
        requested={r['granule'] for group in available.values() for r in group}
        _,ranges=select_contiguous_subsets(available,requested)
        self.assertEqual(ranges,{'IW1':[10,11],'IW3':[10,12]})
        selected,ranges=select_contiguous_subsets({'IW2':rows([10,12,13,14],'IW2')},
                                                {'IW2-10','IW2-12','IW2-13','IW2-14'})
        self.assertEqual(ranges,{'IW2':[12,14]})
        self.assertEqual(selected,{'IW2-12','IW2-13','IW2-14'})

    def test_three_swath_optimum_against_exhaustive_small_cases(self):
        # Independent exhaustive oracle verifies the DP's global optimum,
        # including cases where choosing each swath greedily would fail.
        rng=random.Random(712)
        for _ in range(18):
            available={s:rows([i for i in range(10,15) if rng.random()>.3],s)
                       for s in ('IW1','IW2','IW3')}
            if any(not r for r in available.values()):continue
            requested={r['granule'] for group in available.values() for r in group}
            intervals=[]
            for swath,group in available.items():
                ids={r['burst_id'] for r in group}
                intervals.append([(lo,hi) for lo in sorted(ids) for hi in sorted(ids)
                                  if lo<=hi and set(range(lo,hi+1))<=ids])
            possibilities=[]
            for bounds in product(*intervals):
                if all(abs(a[0]-b[0])<=1 and abs(a[1]-b[1])<=1 for a,b in zip(bounds,bounds[1:])):
                    lengths=[hi-lo+1 for lo,hi in bounds]
                    possibilities.append(((-lengths[0],-sum(lengths),-sum(lengths),bounds),bounds))
            with self.subTest(available=available):
                if not possibilities:
                    with self.assertRaises(ValueError):select_contiguous_subsets(available,requested)
                else:
                    expected=min(possibilities)[1]
                    _,actual=select_contiguous_subsets(available,requested)
                    self.assertEqual(list(actual.values()),[list(x) for x in expected])


if __name__=='__main__':unittest.main()

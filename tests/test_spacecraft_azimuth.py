"""Same-ground order contrasts and temporal cancellation of timing terms."""
import unittest
import numpy as np
from tools.investigate_spacecraft_azimuth import counterbalanced_effect,differential_bistatic_seconds,eligible,summarize,temporal_closure


class SpacecraftControlsTests(unittest.TestCase):
    def test_empty_quality_selection_retains_counts_without_an_estimate(self):
        result=summarize(np.array([]),np.array([],dtype=int),np.random.default_rng(0))
        self.assertEqual(result['n'],0);self.assertEqual(result['absolute_gt_0_25_n'],0)
        self.assertIsNone(result['mean_px']);self.assertIsNone(result['ci95_px'])
        self.assertEqual(result['group_counts'],[])

    def test_reversing_order_is_separated_from_same_ground_even_component(self):
        location=np.array([.01,-.02,.04]);odd,even=counterbalanced_effect(.03+location,-.03+location)
        np.testing.assert_allclose(odd,.03);np.testing.assert_allclose(even,location)

    def test_selection_does_not_depend_on_agreement_or_offset_values(self):
        v={'dual_valid':np.array([True,True]),'burst_interior':np.array([True,True]),
           'iw1_peak':np.array([.4,.5]),'iw2_peak':np.array([.4,.5]),
           'iw1_total':np.array([[100,200],[0,0]]),'iw2_total':np.array([[-100,-200],[1,1]])}
        self.assertTrue(eligible(v).all());v['iw2_peak'][0]=.2
        np.testing.assert_array_equal(eligible(v),[False,True])

    def test_static_rank_pri_and_shared_midrange_terms_cancel_between_dates(self):
        a={'rank':9,'prf_raw_Hz':1700.,'iw2_mid_range_m':887499.}
        b={'rank':8,'prf_raw_Hz':1450.,'iw2_mid_range_m':887499.}
        early=differential_bistatic_seconds(a,b,850000.,850000.)
        late=differential_bistatic_seconds(a|{'iw2_mid_range_m':887498.9},b|{'iw2_mid_range_m':887498.9},850001.,850001.)
        self.assertAlmostEqual(early,late,places=15)

    def test_date_additive_biases_cancel_but_pair_specific_effect_remains(self):
        a=np.array([[.01,.03],[-.02,.04]])
        b=np.array([[.02,.04],[.03,.06]])
        c=np.array([[-.01,.01],[.04,.07]])
        np.testing.assert_allclose(temporal_closure(b-a,c-b,c-a),0,atol=1e-15)
        np.testing.assert_allclose(temporal_closure(b-a+.03,c-b,c-a),.03)


if __name__=='__main__':unittest.main()

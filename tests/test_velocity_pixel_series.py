import unittest
import numpy as np
from numpy.testing import assert_allclose
from test_velocity_pixel import fixture
from glacier_offsets.velocity_pixel_series import evaluate_series
from glacier_offsets.velocity_basis import PERIOD

class SeriesTests(unittest.TestCase):
    def setup_data(self):
        f,u,cs,v,bias,z=fixture()
        coef=np.column_stack([v[0],v[1],np.broadcast_to(bias,(len(u),2)),z,cs])
        return f,u,coef,cs

    def test_known_physical_prediction_and_along_projection(self):
        f,u,coef,_=self.setup_data();result=evaluate_series(f,coef,u)
        assert_allclose(result['predicted_radar_displacement_m'],f['data'],atol=1e-12)
        assert_allclose(result['observed_along_equivalent_m_day'],result['modeled_along_pair_average_m_day'],atol=1e-12)

    def test_range_noise_does_not_change_azimuth_derived_observation(self):
        f,u,coef,_=self.setup_data();before=evaluate_series(f,coef,u)
        f['data'][...,0]+=np.random.default_rng(6).normal(size=f['early'].shape)*10
        after=evaluate_series(f,coef,u)
        assert_allclose(after['observed_along_equivalent_m_day'],before['observed_along_equivalent_m_day'],atol=0)

    def test_azimuth_noise_has_correct_interval_and_geometry_scaling(self):
        f,u,coef,_=self.setup_data();before=evaluate_series(f,coef,u);f['data'][...,1]+=.12
        after=evaluate_series(f,coef,u)
        expected=.12/(before['duration_days']*before['azimuth_flow_projection'])
        assert_allclose(after['observed_along_equivalent_m_day']-before['observed_along_equivalent_m_day'],expected,atol=1e-12)

    def test_model_comparison_uses_interval_average_not_instantaneous_value(self):
        f,u,coef,cs=self.setup_data();r=evaluate_series(f,coef,u);mid=r['midpoint_days'];w=2*np.pi/PERIOD
        instant=cs[:,0,None]*np.cos(w*mid)+cs[:,1,None]*np.sin(w*mid)
        assert_allclose(r['modeled_along_Msf_pair_average_m_day'],np.sinc(r['duration_days']/PERIOD)*instant,atol=1e-12)
        self.assertGreater(np.max(abs(r['modeled_along_Msf_pair_average_m_day']-instant)),.01)

if __name__=='__main__':unittest.main()

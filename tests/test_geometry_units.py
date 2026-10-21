import unittest
import numpy as np
from glacier_offsets.geocode import physical_fields

class UnitsTests(unittest.TestCase):
    def test_units_signs_exact_interval_and_total(self):
        gross=np.array([10.,-10]);residual=np.array([-2.,2.])
        result=physical_fields(gross+residual,np.array([2.,-2]),2.3,np.array([14.,13.]),6*86400+1)
        np.testing.assert_allclose(result['range_displacement_m'],[18.4,-18.4])
        np.testing.assert_allclose(result['azimuth_displacement_m'],[28.,-26.])
        np.testing.assert_allclose(result['range_velocity_m_per_day'],np.array([18.4,-18.4])/(6+1/86400))
    def test_exact_zero_and_invalid_interval(self):
        for array in physical_fields(np.zeros(3),np.zeros(3),2.3,np.array([14.,13.,15.]),12*86400).values():
            np.testing.assert_array_equal(array,np.zeros(3))
        with self.assertRaises(ValueError):physical_fields(0,0,2.3,14,0)

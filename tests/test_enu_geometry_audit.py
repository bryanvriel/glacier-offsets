"""Independent checks for the read-only geometry convergence audit."""
from datetime import datetime, timedelta
import json
import unittest

import numpy as np
from pyproj import Transformer

from tools.audit_enu_geometry import central_difference, enu_basis, finite_max, implicit_derivative, select_observations, statistics


class GeometryAuditTests(unittest.TestCase):
    def test_failed_solver_summaries_remain_strict_json(self):
        summary = {'errors': statistics([np.nan, np.nan]), 'maximum': finite_max([np.nan])}
        result = json.loads(json.dumps(summary, allow_nan=False))
        self.assertIsNone(result['maximum'])
        self.assertIsNone(result['errors']['max'])
        self.assertEqual(result['errors']['nonfinite_count'], 2)

    def test_enu_basis_is_orthonormal_and_right_handed(self):
        basis = enu_basis(-62., -67.)
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-15)
        np.testing.assert_allclose(np.cross(basis[0], basis[1]), basis[2], atol=1e-15)

    def test_selection_includes_every_declared_outlier_and_burst(self):
        n = 600
        fields = {'range_sensitivity_enu': np.tile([1., 0., -.8], (n, 1)),
                  'azimuth_sensitivity_enu': np.tile([0., 1., .002], (n, 1)),
                  'azimuth_ground_spacing_m_per_slc_line': np.full(n, 14.),
                  'source_burst_index': np.repeat([1, 2, 3], n//3),
                  'map_yx': np.stack(np.unravel_index(np.arange(n), (20, 30)), axis=1)}
        # Many outliers ensure selection is not just a top-N list.
        fields['azimuth_ground_spacing_m_per_slc_line'][201:251] = 100
        fields['azimuth_sensitivity_enu'][501] = [0, 1, .02]
        fields['azimuth_sensitivity_enu'][580] = [1, 1e-5, 0]
        selected, _, extreme = select_observations(fields, per_burst=3)
        self.assertTrue(np.isin(np.flatnonzero(extreme), selected).all())
        self.assertEqual(set(fields['source_burst_index'][selected]), {1, 2, 3})
        self.assertIn(501, selected)
        self.assertIn(580, selected)

    def test_native_derivatives_against_known_straight_orbit(self):
        import isce3
        epoch = datetime(2020, 1, 1)
        start, velocity = np.array([7078137., 0., 500000.]), np.array([500., 7500., 30.])
        states = [isce3.core.StateVector(isce3.core.DateTime(epoch+timedelta(seconds=t)),
                  start+t*velocity, velocity) for t in range(-40, 41, 10)]
        orbit = isce3.core.Orbit(states)
        basis, point = enu_basis(0., 0.), np.array([6378137., 0., 0.])
        inverse = Transformer.from_crs(4978, 4979, always_xy=True)
        ellipsoid, doppler = isce3.core.Ellipsoid(), isce3.core.LUT2d()
        def solve(p, threshold):
            lo, la, height = inverse.transform(*p)
            return isce3.geometry.geo2rdr([np.deg2rad(lo), np.deg2rad(la), height],
                ellipsoid, orbit, doppler, .05, isce3.core.LookSide.Right, threshold=threshold)
        t, _ = solve(point, 1e-12)
        position, _ = orbit.interpolate(t)
        los = np.asarray(position)-point
        expected = np.stack((-basis @ (los/np.linalg.norm(los)), basis @ velocity/np.dot(velocity, velocity)*1000))
        np.testing.assert_allclose(implicit_derivative(orbit, t, point, basis, 1000), expected, atol=1e-11)
        # Vertical azimuth response is explicitly nonzero in this fixture.
        self.assertGreater(abs(expected[1, 2]), .001)
        for step in (.1, 1., 10.):
            for threshold in (1e-6, 1e-8, 1e-10, 1e-12):
                with self.subTest(step=step, threshold=threshold):
                    actual = central_difference(lambda p: solve(p, threshold), point, basis, step, 1000)
                    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from scipy.sparse import csr_matrix
from mb1_muscle_balance import AMPLITUDE, SETS, SIGNS, directions, selection


class MuscleBalanceTests(unittest.TestCase):
    def setUp(self):
        self.joints = [f"joint_{leg}{axis}" for axis in SETS for leg in ("LF", "LM", "LH", "RF", "RM", "RH")]
        positive, negative = np.zeros((12, 4)), np.zeros((12, 4))
        positive[:6, 0], negative[:6, 1] = 1., 1.
        positive[6:, 2], negative[6:, 3] = .5, .5
        self.positive, self.negative = csr_matrix(positive), csr_matrix(negative)
        self.weights = np.array([2., -2., 2., -2., 2.])
        self.targets = np.array([0, 1, 2, 3, 0])

    def test_axes_use_actual_positive_negative_pools_and_shared_targets(self):
        vectors, _ = directions(self.weights, self.weights, self.targets,
                                self.positive, self.negative, self.joints)
        np.testing.assert_allclose(vectors["Femur"], [AMPLITUDE, -AMPLITUDE, 0, 0, AMPLITUDE])
        np.testing.assert_allclose(vectors["Tibia"], [0, 0, AMPLITUDE, -AMPLITUDE, 0])
        # The additive direction is signed drive, independently of transmitter sign.
        self.assertLess(self.weights[1] + vectors["Femur"][1] * abs(self.weights[1]), self.weights[1])

    def test_zero_axis_overlap_and_headroom_exclusion(self):
        weights = self.weights.copy()
        weights[0] *= 1.5
        vectors, _ = directions(weights, self.weights, self.targets,
                                self.positive, self.negative, self.joints)
        self.assertEqual(vectors["Femur"][0], 0)
        with self.assertRaises(ValueError):
            directions(self.weights, self.weights, self.targets, self.positive, self.positive, self.joints)

    def test_reject_invalid_reference_or_mapping(self):
        for targets in (self.targets.astype(float), np.array([0, 1, 2, 3, 4])):
            with self.assertRaises(ValueError):
                directions(self.weights, self.weights, targets, self.positive, self.negative, self.joints)
        with self.assertRaises(ValueError):
            directions(-self.weights, self.weights, self.targets, self.positive, self.negative, self.joints)

    def test_selection_tie_order_and_angular_guard(self):
        parent = np.tile([0., 1.], (6, 1))
        arms = {(axis, sign): parent + [1e-3, 0] for axis in SETS for sign in SIGNS}
        self.assertEqual(selection(parent, arms)["retained_candidate"], "Femur/-1")
        arms["Femur", -1][0, 1] += .01
        self.assertEqual(selection(parent, arms)["retained_candidate"], "Femur/+1")
        for value in arms.values():
            value[0, 0] = -1e-3
        self.assertIsNone(selection(parent, arms)["retained_candidate"])
        del arms["Tibia", 1]
        with self.assertRaises(ValueError):
            selection(parent, arms)


if __name__ == "__main__":
    unittest.main()

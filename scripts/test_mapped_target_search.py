import unittest

import numpy as np

from mapped_target_search import AMPLITUDE, SETS, SIGNS, directions, selection


class MatchedTargetTests(unittest.TestCase):
    def test_disjoint_balanced_edges_with_symmetric_headroom(self):
        original = np.array([2., -2., 2., -2., 2., -2., 2., -2.])
        weights = original.copy()
        weights[0] *= 1.5
        mask = np.arange(8) < 4
        offsets, eligible = directions(weights, original, mask)
        self.assertEqual(eligible, {"mapped": 3, "remaining": 4})
        for label, available in (("mapped", mask), ("remaining", ~mask)):
            vector = offsets[label]
            self.assertFalse(vector[~available].any())
            self.assertEqual(sorted(vector[vector != 0]), [-AMPLITUDE, AMPLITUDE])
        self.assertEqual(offsets["mapped"][0], 0)

    def test_reward_gain_cannot_override_angular_regression(self):
        parent = np.tile([0., 2.], (6, 1))
        arms = {(label, sign): parent.copy() for label in SETS for sign in SIGNS}
        arms["mapped", -1][:, 0] = 1e-3
        arms["mapped", -1][0, 1] += .01
        arms["mapped", 1][:, 0] = 1e-4
        self.assertEqual(selection(parent, arms)["retained_signs"], {"mapped": 1, "remaining": None})
        del arms["remaining", 1]
        with self.assertRaises(ValueError):
            selection(parent, arms)


if __name__ == "__main__":
    unittest.main()

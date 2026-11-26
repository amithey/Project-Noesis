import unittest

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from om1_target_coverage import counts


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.positive = csr_matrix([[1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        self.negative = csr_matrix([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]])

    def test_missing_targets_overlap_and_pool_coverage(self):
        result = counts([0, 3], self.positive, self.negative)
        self.assertEqual(result, {
            "selected_targets": 2, "joint_rows": 3,
            "positive_nonempty_pools": 2, "positive_covered_pools": 1,
            "positive_selected_targets": 1,
            "negative_nonempty_pools": 2, "negative_covered_pools": 1,
            "negative_selected_targets": 1,
            "selected_targets_both_signs": 1, "selected_targets_mapped": 1,
            "selected_targets_unmapped": 1,
        })

    def test_all_nonempty_pools_covered(self):
        result = counts([0, 1, 2], self.positive, self.negative)
        self.assertEqual(result["selected_targets_unmapped"], 0)
        self.assertEqual(result["selected_targets_both_signs"], 2)
        self.assertEqual(result["positive_covered_pools"], 2)
        self.assertEqual(result["negative_covered_pools"], 2)

    def test_empty_pools_and_explicit_zeros(self):
        zero = coo_matrix(([0.0], ([2], [0])), shape=(3, 4))
        result = counts([0, 1], zero, zero)
        self.assertEqual(result["selected_targets_unmapped"], 2)
        self.assertEqual(result["positive_nonempty_pools"], 0)
        self.assertEqual(result["negative_covered_pools"], 0)

    def test_reject_invalid_indices(self):
        for bad in ([0, 0], [-1], [4], [0.0], [True], [[0]], np.array([], dtype=int)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                counts(bad, self.positive, self.negative)

    def test_reject_invalid_matrices(self):
        for bad in (csr_matrix((2, 4)), self.positive.toarray(),
                    csr_matrix(np.full((3, 4), np.nan)), -self.positive):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                counts([0], bad, self.negative)


if __name__ == "__main__":
    unittest.main()

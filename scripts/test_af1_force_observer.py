import unittest
import numpy as np
from af1_force_observer import summarize


class ForceObserverTests(unittest.TestCase):
    def test_exact_bounds_warmup_and_longest_sequence(self):
        result = summarize(np.array([[2., -2.], [0., 0.], [2., 0.], [2., -2.], [0., -2.]]),
                           np.tile([-2., 2.], (2, 1)), [1, 1], dt=.1, warmup=.1)
        self.assertEqual(result["bound_hit_fraction"], .5)
        self.assertEqual(result["peak_normalized_force"], 1.)
        self.assertEqual(result["longest_bound_hit_seconds"], .2)

    def test_interior_force(self):
        result = summarize(np.array([[.1], [-.2]]), [[-1., 2.]], [1], warmup=0)
        self.assertEqual(result["bound_hit_fraction"], 0.)
        self.assertEqual(result["peak_normalized_force"], .2)

    def test_reject_bad_force_or_disabled_limit(self):
        for force, enabled in (([[float("nan")]], [1]), ([[3.]], [1]), ([[0.]], [0])):
            with self.subTest(force=force), self.assertRaises(ValueError):
                summarize(force, [[-1., 1.]], enabled, warmup=0)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from cs1_command_audit import reconstruct


class CommandLimiterTests(unittest.TestCase):
    def test_constant_and_ramp(self):
        command, active = reconstruct([[.03], [.06], [.09]], [0])
        np.testing.assert_allclose(command[:, 0], [.03, .06, .09], atol=1e-15)
        self.assertFalse(active.any())

    def test_step_then_reversal(self):
        command, active = reconstruct([[1.], [1.], [-1.], [-1.]], [0])
        np.testing.assert_allclose(command[:, 0], [.06, .12, .06, 0], atol=1e-15)
        self.assertTrue(active.all())

    def test_neutral_offset_and_bad_input(self):
        command, _ = reconstruct([[2.03]], [2])
        np.testing.assert_allclose(command, [[2.03]])
        for desired, rest in (([], [0]), ([[float("nan")]], [0]), ([[1, 2]], [0])):
            with self.subTest(desired=desired), self.assertRaises(ValueError):
                reconstruct(desired, rest)


if __name__ == "__main__":
    unittest.main()

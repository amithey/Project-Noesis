import unittest
import run_mb1_muscle_balance as mb1
from run_mb1_completion_mc1 import schedule


class CompletionScheduleTests(unittest.TestCase):
    def test_exact_missing_cells_with_one_additional_baseline(self):
        original = list(mb1.jobs())
        cells = schedule([cell[0] for cell in original[:18]])
        self.assertEqual(cells[0], original[0])
        self.assertEqual(cells[1:], original[18:])
        self.assertEqual(len(cells), 8)
        self.assertEqual(len(set(cell[0] for cell in [*original[:18], *cells[1:]])), 25)

    def test_reject_missing_duplicate_reordered_or_extra_prefix(self):
        names = [cell[0] for cell in list(mb1.jobs())[:18]]
        for bad in (names[:-1], [names[0], *names[:-1]], names[::-1], [*names, "extra"]):
            with self.subTest(prefix=bad), self.assertRaises(ValueError):
                schedule(bad)


if __name__ == "__main__":
    unittest.main()

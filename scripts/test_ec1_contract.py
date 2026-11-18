import copy
import unittest
from ec1_contract import BANKS, TRAIN, decision, jobs, lineage


def fixture():
    banks = {}
    for bank in ("parent", *BANKS):
        better = bank == "carry"
        banks[bank] = {}
        for seed in TRAIN:
            banks[bank][seed] = {
                "return": -.6 if better else -.8,
                "physical": {"mean_PS1_quality": .2 if better else .1,
                             "mean_COM_speed_mm_s": 9. if better else 10.,
                             "mean_angular_speed_rad_s": 9. if better else 10.,
                             "mean_non_tarsal_fraction": .1 if better else .2},
                "anatomical": {"mode_fractions": {"tarsal_only": .9 if better else .8},
                               "mean_non_tarsal_fraction": .1 if better else .2},
                "summary": {"standing_fraction": .95 if better else .91,
                            "mean_stance_slip_mm_s": 1. if better else 2.,
                            "heading_forward_distance_mm": 1. if better else 0.,
                            "net_forward_displacement_mm": 1. if better else 0.,
                            "total_leg_steps": 1 if better else 0,
                            "tripod_coordination": {"tripod_coordination_score": .5 if better else .4}},
            }
    return banks


class EpisodicCarryTests(unittest.TestCase):
    def test_fixed_lineage_and_complete_allocation(self):
        self.assertEqual(lineage("stage1"), ("parent", 1010))
        self.assertEqual(lineage("carry"), ("stage1", 1011))
        self.assertEqual(lineage("reset"), ("parent", 1011))
        schedule = jobs()
        self.assertEqual(len(schedule), 22)
        self.assertEqual(sum(job["phase"] == "train" for job in schedule), 3)
        self.assertEqual(len({job["name"] for job in schedule}), 22)
        with self.assertRaises(ValueError):
            lineage("best")

    def test_positive_fixture_without_adoption(self):
        result = decision(fixture())
        self.assertTrue(result["endpoint_meets_training_criteria"])
        self.assertFalse(result["model_promoted"])
        self.assertFalse(result["walking_proven"])

    def test_later_reward_cannot_override_earlier_skill_regression(self):
        banks = fixture()
        banks["stage1"][1010]["physical"]["mean_PS1_quality"] = .21
        result = decision(banks)
        self.assertFalse(result["endpoint_meets_training_criteria"])
        self.assertTrue(result["comparisons"]["parent"]["passes"])
        self.assertFalse(result["comparisons"]["stage1"]["passes"])

    def test_standing_threshold_and_missing_context(self):
        banks = fixture()
        banks["carry"][1010]["summary"]["standing_fraction"] = .89
        self.assertFalse(decision(banks)["endpoint_meets_training_criteria"])
        bad = copy.deepcopy(banks)
        del bad["carry"][1015]
        with self.assertRaises(ValueError):
            decision(bad)


if __name__ == "__main__":
    unittest.main()

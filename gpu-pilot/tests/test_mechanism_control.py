import importlib.util

import numpy as np
import pytest

from noesis_gpu import mechanism_control as mc

pytestmark = pytest.mark.skipif(importlib.util.find_spec("noesis.plasticity") is None, reason="Requires pinned Noesis core")


def test_learning_off_runs_the_identical_trial_without_any_update():
    row = mc.birth(3, .2, learning=False, steps=200)
    assert row["updates"] == 0 and row["weights_equal_start"] and row["D"] == 0.0


def test_birth_is_deterministic_and_negative_arm_shares_noise_with_positive():
    a, b = mc.birth(5, .2, steps=200), mc.birth(5, .2, steps=200)
    assert a == b
    assert mc.birth(5, 0.0, steps=200)["D"] != a["D"]  # the planted term really changes the learner's input


def test_bounds_and_signs_hold_for_both_arms_even_with_a_large_planted_effect():
    for kappa in (0.0, .8):
        row = mc.birth(2, kappa, lr=2.0, steps=300)
        assert row["ratio_min"] >= .5 - 1e-9 and row["ratio_max"] <= 1.5 + 1e-9 and row["signs_kept"]


def test_a_strong_planted_effect_is_acquired_and_an_absent_one_is_not():
    seeds = range(8)
    pos = [mc.birth(s, .8, steps=400)["D"] for s in seeds]
    neg = [mc.birth(s, 0.0, steps=400)["D"] for s in seeds]
    assert mc.z_score(pos) > 3 and abs(mc.z_score(neg)) < 3


def test_registered_verdict_fails_when_the_positive_control_is_absent():
    out = mc.run_control(range(4), kappa=0.0, steps=150)
    assert not out["verdict"]["positive_z_ge_3"] and not out["verdict"]["registered_pass"]
    assert out["verdict"]["learning_off_exact"] and out["verdict"]["bounds_and_signs"]

import importlib.util

import numpy as np
import pytest

from noesis_gpu.learning import score_transition
from noesis_gpu.reward_profiles import DEFAULT_PROFILE, PROFILES, resolve_objective

needs_core = pytest.mark.skipif(importlib.util.find_spec("noesis.embodied_metrics") is None,
                                reason="Requires pinned Noesis core")
DT_MS = 100.


def state(x=0., up=1., load=1., foot_v=0.):
    return {"position": np.array([x, 0., 1.]), "up_body": np.array([0., 0., up]),
            "velocity": np.zeros(42), "heading": np.array([1., 0., 0.]),
            "feet_position": np.zeros((6, 3)) + np.array([foot_v * DT_MS / 1000, 0., 0.]),
            "foot_load": np.full(6, load)}


def test_profile_names_and_default():
    assert DEFAULT_PROFILE == "standing"
    assert resolve_objective() == "curriculum-standing-v7"
    assert resolve_objective("gripped-steps") == "curriculum-gripped-step-v8"
    assert resolve_objective("continuous-walking") == "standing-gate-smooth-grip-v6"
    for objective in set(PROFILES.values()):
        assert resolve_objective(objective) == objective
    with pytest.raises(ValueError):
        resolve_objective("nope")


@needs_core
def test_every_profile_objective_exists_in_core():
    from noesis.embodied_metrics import GROUNDED_OBJECTIVES
    assert set(PROFILES.values()) <= set(GROUNDED_OBJECTIVES)


@needs_core
def test_score_transition_default_is_standing_and_keeps_fields():
    a, b = state(), state(x=.2)
    default = score_transition(a, b, DT_MS)
    assert default == score_transition(a, b, DT_MS, objective="curriculum-standing-v7")
    assert default == score_transition(a, b, DT_MS, objective="standing")
    assert default["grounded_progress"] == 0.  # standing stage gives no advancement credit
    for key in ("reward", "forward_speed_mm_s", "supporting_feet", "stance_slip_mm_s",
                "grounded_progress", "grip_factor", "standing_quality"):
        assert key in default


@needs_core
@pytest.mark.parametrize("profile", ["gripped-steps", "continuous-walking"])
def test_slipping_and_falling_are_not_steps(profile):
    before = state()
    gripped = score_transition(before, state(x=.2), DT_MS, profile)  # 2 mm/s, planted feet
    slipping = score_transition(before, state(x=.2, foot_v=50.), DT_MS, profile)
    fallen = score_transition(before, state(x=.2, up=.1), DT_MS, profile)
    airborne = score_transition(before, state(x=.2, load=0.), DT_MS, profile)
    assert gripped["grounded_progress"] > 0
    assert slipping["grounded_progress"] < .2 * gripped["grounded_progress"]
    assert slipping["reward"] < gripped["reward"]
    assert fallen["grounded_progress"] == 0 and fallen["reward"] < 0
    assert airborne["grounded_progress"] == 0 and airborne["reward"] < 0

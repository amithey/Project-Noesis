import numpy as np
import pytest

from noesis.embodied_metrics import CYCLE_WEIGHT, GROUNDED_OBJECTIVES, locomotion_reward

V8, V9 = "curriculum-gripped-step-v8", "curriculum-gripped-cycle-v9"


def reward(objective=V9, credit=0., weight=None, dx=.1, foot_v=0., loads=None):
    loads = np.ones(6) if loads is None else loads
    extra = {} if weight is None else {"cycle_weight": weight}
    return locomotion_reward([0, 0, .8], [dx, 0, .8], 1, [0], .2, [1, 0, 0], loads,
                             np.tile([foot_v, 0, 0], (6, 1)), 0., objective=objective,
                             cycle_credit=credit, **extra)


def test_default_weight_is_unchanged_and_equals_an_explicit_default():
    assert CYCLE_WEIGHT == .1
    implicit, explicit = reward(credit=.7), reward(credit=.7, weight=CYCLE_WEIGHT)
    assert implicit == explicit and implicit["cycle_weight"] == CYCLE_WEIGHT


@pytest.mark.parametrize("weight", [.1, .3, .5, 1.])
def test_bonus_scales_linearly_with_the_weight_before_clipping(weight):
    base = reward(credit=0.)["reward_unclipped"]
    bonus = reward(credit=.8, weight=weight)
    grip = bonus["grip_factor"]
    assert bonus["reward_unclipped"] - base == pytest.approx(weight * .8 * grip)
    assert bonus["cycle_weight"] == weight
    assert bonus["reward"] == float(np.clip(bonus["reward_unclipped"], -1, 1))


def test_larger_weight_pays_more_until_the_clip_and_never_more_than_one():
    rewards = [reward(credit=1., weight=w, dx=.2)["reward"] for w in (.1, .3, .5, 1.)]
    assert rewards == sorted(rewards) and max(rewards) <= 1.0 and rewards[0] < rewards[-1]


@pytest.mark.parametrize("bad", [0, -.1, 1.01, 2., float("nan"), float("inf")])
def test_weight_range_is_enforced(bad):
    with pytest.raises(ValueError):
        reward(credit=.5, weight=bad)


def test_other_objectives_reject_a_non_default_weight_but_accept_the_default():
    for objective in GROUNDED_OBJECTIVES[:-1]:
        reward(objective, weight=CYCLE_WEIGHT)  # the default is a no-op everywhere
        with pytest.raises(ValueError):
            reward(objective, weight=.5)


def test_changing_the_weight_does_not_touch_v8_part_or_other_fields():
    low, high = reward(credit=.6, weight=.1), reward(credit=.6, weight=.9)
    for key in ("grounded_progress", "grip_factor", "stance_slip_mm_s", "supporting_feet", "standing_quality"):
        assert low[key] == high[key]
    v8 = reward(V8)
    assert reward(credit=0., weight=.9)["reward"] == v8["reward"]  # no credit, no difference from v8

import numpy as np
import pytest

from noesis.embodied_metrics import GROUNDED_OBJECTIVES, locomotion_reward

V8, V9 = "curriculum-gripped-step-v8", "curriculum-gripped-cycle-v9"


def reward(objective, dx=.2, foot_v=0., loads=None, credit=0., up=1., obstacle=0., vel=0.):
    loads = np.ones(6) if loads is None else loads
    return locomotion_reward([0, 0, .8], [dx, 0, .8], up, [vel] * 5, .2, [1, 0, 0], loads,
                             np.tile([foot_v, 0, 0], (6, 1)), obstacle, objective=objective,
                             cycle_credit=credit)


def test_unclipped_is_v9_only_and_does_not_leak_into_older_objectives():
    assert "reward_unclipped" in reward(V9)
    for objective in GROUNDED_OBJECTIVES[:-1]:
        assert "reward_unclipped" not in reward(objective)


def test_clipping_the_unclipped_value_gives_the_reward_exactly():
    rng = np.random.default_rng(1)
    saturated = 0
    for _ in range(400):
        r = reward(V9, dx=rng.uniform(-.5, 1.5), foot_v=rng.uniform(0, 60),
                   loads=rng.uniform(0, 1, 6) * (rng.random(6) > .3), credit=rng.uniform(0, 1),
                   up=rng.uniform(-1, 1), obstacle=rng.uniform(0, 1.5), vel=rng.uniform(0, 3))
        assert r["reward"] == float(np.clip(r["reward_unclipped"], -1, 1))
        saturated += abs(r["reward_unclipped"]) > 1
    assert saturated > 0  # the sample reaches the clip (low side), so the check is not vacuous


def test_unclipped_exposes_the_lost_bonus_when_v8_saturates():
    fast = reward(V9, dx=.6, credit=1.)  # 3 mm/s, planted feet: v8 part already at the clip
    assert fast["reward"] == 1.0 and fast["reward_unclipped"] > 1.0
    slow = reward(V9, dx=.1, credit=1.)
    assert slow["reward"] == slow["reward_unclipped"] < 1.0  # bonus fully delivered below the clip


def test_adding_the_field_does_not_change_the_reward():
    for kwargs in ({}, {"dx": .6}, {"foot_v": 50.}, {"credit": 1.}, {"loads": np.zeros(6)},
                   {"up": .2}, {"obstacle": 1.}):
        a = reward(V9, **kwargs)
        b = reward(V8, **{k: v for k, v in kwargs.items() if k != "credit"})
        if kwargs.get("credit", 0.) == 0.:
            assert a["reward"] == b["reward"]
        else:
            assert a["reward"] >= b["reward"]  # the bonus can only add, and is bounded by the clip
        assert a["reward"] <= 1.0
    with pytest.raises(ValueError):
        reward(V9, credit=1.5)

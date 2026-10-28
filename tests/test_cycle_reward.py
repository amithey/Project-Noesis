import numpy as np
import pytest

from noesis.embodied_metrics import (
    CYCLE_WEIGHT,
    GROUNDED_OBJECTIVES,
    LegCycleTracker,
    locomotion_reward,
)

V8, V9 = "curriculum-gripped-step-v8", "curriculum-gripped-cycle-v9"
STEPS = 500  # 2.5 s at 5 ms: covers the 2 s speed window, so "no credit" assertions are not vacuous


def drive(tracker, loads_at, speed=2., standing=True, steps=STEPS):
    credit = 0.
    for i in range(steps):
        credit = tracker.update(loads_at(i), speed, standing)
    return credit


def stepping_legs(i):
    """Each leg lifts for 30 ms every 100 ms, staggered; >=4 feet always loaded."""
    loads = np.ones(6)
    for leg in range(6):
        if (i + 3 * leg) % 20 < 6 and leg % 2 == 0:
            loads[leg] = 0.
    return loads


def test_v9_is_registered_last_so_default_is_unchanged():
    assert V9 in GROUNDED_OBJECTIVES
    assert GROUNDED_OBJECTIVES[0] == "standing-first-linear-slip-v3"
    assert GROUNDED_OBJECTIVES[-1] == V9


def test_real_lift_touchdown_cycles_earn_credit_after_a_full_window():
    t = LegCycleTracker()
    assert drive(t, stepping_legs, steps=20) == 0.  # window not yet full
    credit = drive(t, stepping_legs)
    assert 0 < credit <= .5  # only the three stepping legs move


def test_static_standing_and_stepping_in_place_earn_no_credit():
    assert drive(LegCycleTracker(), lambda i: np.ones(6)) == 0.
    assert drive(LegCycleTracker(), stepping_legs, speed=0.) == 0.
    assert drive(LegCycleTracker(), stepping_legs, speed=.2) == 0.  # below 0.25 mm/s
    assert drive(LegCycleTracker(), stepping_legs, speed=.3) > 0.  # same stepping, enough progress


def test_contact_chatter_is_not_a_cycle():
    def chatter(i):
        loads = np.ones(6)
        loads[:3] = float(i % 2)  # flips every 5 ms, shorter than the 15 ms minimum phase
        return loads
    assert drive(LegCycleTracker(), chatter) == 0.


def test_lifting_legs_without_support_or_when_fallen_earns_nothing():
    def hover(i):  # only two feet ever loaded
        loads = np.zeros(6)
        loads[:2] = 1
        loads[0] = 0. if i % 20 < 6 else 1.
        return loads
    assert drive(LegCycleTracker(), hover) == 0.
    assert drive(LegCycleTracker(), stepping_legs, standing=False) == 0.


def test_first_landing_without_a_prior_lift_is_not_a_cycle():
    def late_landing(i):
        return np.zeros(6) if i < 30 else np.ones(6)
    assert drive(LegCycleTracker(), late_landing, steps=STEPS) == 0.


def test_tracker_rejects_bad_input_and_resets():
    t = LegCycleTracker()
    for bad in (np.ones(5), np.full(6, np.nan), -np.ones(6)):
        with pytest.raises(ValueError):
            t.update(bad, 1., True)
    with pytest.raises(ValueError):
        t.update(np.ones(6), np.nan, True)
    with pytest.raises(ValueError):
        LegCycleTracker(window_ms=0)
    drive(t, stepping_legs)
    t.reset()
    assert drive(t, stepping_legs, steps=10) == 0.


def reward(objective, loads=None, foot_v=0., credit=0., dx=.2):
    loads = np.ones(6) if loads is None else loads
    return locomotion_reward([0, 0, .8], [dx, 0, .8], 1, [0], .2, [1, 0, 0], loads,
                             np.tile([foot_v, 0, 0], (6, 1)), 0., objective=objective,
                             cycle_credit=credit)


def test_v9_without_credit_equals_v8_and_credit_adds_bounded_bonus():
    base = reward(V8)
    zero = reward(V9)
    assert zero["reward"] == base["reward"] and "cycle_credit" not in base
    assert zero["cycle_credit"] == 0
    full = reward(V9, credit=1.)
    assert full["reward"] == pytest.approx(min(base["reward"] + CYCLE_WEIGHT * full["grip_factor"], 1))
    assert reward(V9, credit=.5)["reward"] < full["reward"]


def test_sliding_scales_the_bonus_down_and_fallen_gets_none():
    planted = reward(V9, credit=1.)["reward"] - reward(V9)["reward"]
    sliding = reward(V9, foot_v=50., credit=1.)["reward"] - reward(V9, foot_v=50.)["reward"]
    assert planted > 5 * sliding >= 0
    airborne = reward(V9, loads=np.zeros(6), credit=1.)["reward"] - reward(V9, loads=np.zeros(6))["reward"]
    assert airborne == 0


@pytest.mark.parametrize("bad", [-.1, 1.1, np.nan, np.inf])
def test_credit_range_is_enforced(bad):
    with pytest.raises(ValueError):
        reward(V9, credit=bad)


def test_other_objectives_reject_cycle_credit():
    for objective in GROUNDED_OBJECTIVES[:-1]:
        with pytest.raises(ValueError):
            reward(objective, credit=.5)

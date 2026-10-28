import numpy as np
import pytest

from noesis.embodied_metrics import LegCycleTracker
from test_cycle_reward import stepping_legs

DT = .005


def run(tracker, speeds, standing=True):
    credit = []
    for i, speed in enumerate(speeds):
        credit.append(tracker.update(stepping_legs(i), speed, standing))
    return np.array(credit)


def test_no_credit_during_the_first_speed_window_then_credit_for_sustained_progress():
    credit = run(LegCycleTracker(), np.full(600, 1.))
    assert (credit[:399] == 0).all()  # 2 s = 400 samples: the window is not yet full
    assert (credit[400:] > 0).any()


def test_jittery_motion_with_no_net_progress_earns_nothing_even_with_cycles():
    # +10 mm/s for 250 ms then -10 for 250 ms: a 250 ms window inside a +phase looks fast, the 2 s mean is 0
    speeds = np.tile(np.r_[np.full(50, 10.), np.full(50, -10.)], 6)
    assert (run(LegCycleTracker(), speeds) == 0).all()
    short = LegCycleTracker(speed_window_ms=250., min_forward_mm_s=.5)
    assert (run(short, speeds) > 0).any()  # the old short gate would have paid on this jitter


def test_gate_opens_only_after_enough_sustained_progress_accumulates():
    speeds = np.r_[np.zeros(520), np.full(80, 5.)]  # the 2 s mean reaches 0.25 after 20 samples of 5 mm/s
    credit = run(LegCycleTracker(), speeds)
    assert (credit[:539] == 0).all()
    assert (credit[539:] > 0).any()


@pytest.mark.parametrize("mean, expected", [(.2, False), (.24, False), (.26, True), (1., True)])
def test_threshold_is_on_the_trailing_two_second_mean(mean, expected):
    credit = run(LegCycleTracker(), np.full(600, mean))
    assert bool((credit[400:] > 0).any()) is expected


def test_speed_window_is_configurable_and_validated():
    short = LegCycleTracker(speed_window_ms=250.)
    assert (run(short, np.full(100, 1.))[60:] > 0).any()  # opens after 250 ms of progress
    for bad in (0, -5, np.nan, np.inf):
        with pytest.raises(ValueError):
            LegCycleTracker(speed_window_ms=bad)
    assert LegCycleTracker().min_forward == .25 and LegCycleTracker().speed_window == 400


def test_reset_clears_the_speed_history():
    tracker = LegCycleTracker()
    run(tracker, np.full(600, 1.))
    tracker.reset()
    assert (run(tracker, np.full(300, 1.)) == 0).all()  # window must refill after a reset

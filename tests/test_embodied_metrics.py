import numpy as np
import pytest

from noesis.embodied_metrics import body_reward


def score(x=0.0, height=0.8, up=1.0, velocity=(0.0, 0.0)):
    return body_reward([0, 0, 0.8], [x, 0, height], up, velocity, 0.1)


def test_forward_upright_walking_beats_rest_backward_and_fallen_motion():
    walking = score(x=0.5)
    assert walking["forward_speed_mm_s"] == 5.0
    assert walking["height_mm"] == 0.8
    assert walking["uprightness"] == 1.0
    assert walking["effort"] == 0.0
    assert walking["reward"] > score()["reward"] > score(x=-0.5)["reward"]
    assert score(x=0.5, up=0.2)["reward"] < 0
    assert score(x=0.5, height=0.1)["reward"] < 0


@pytest.mark.parametrize("height,up", [(0.29, 1.0), (1.51, 1.0), (0.8, 0.69)])
def test_ineligible_posture_has_no_progress_credit(height, up):
    assert score(x=1.0, height=height, up=up)["reward"] == score(
        height=height, up=up
    )["reward"]


def test_launch_speed_gives_no_progress_and_speed_credit_saturates():
    assert score(x=10)["reward"] == score()["reward"]
    assert score(x=1)["reward"] == score(x=2)["reward"]


def test_effort_penalty_and_extreme_values_stay_bounded():
    moving = score(x=0.5, velocity=(10, -10))
    assert 0 < moving["effort"] < 1
    assert moving["reward"] < score(x=0.5)["reward"]
    extreme = score(x=-1e300, up=-1, velocity=(1e300, -1e300))
    assert -1 <= extreme["reward"] <= 1
    assert 0 <= extreme["effort"] <= 1
    assert all(np.isfinite(value) for value in extreme.values())


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
def test_invalid_time_rejected(dt):
    with pytest.raises(ValueError, match="dt_seconds"):
        body_reward([0, 0, 1], [0, 0, 1], 1, [0], dt)


@pytest.mark.parametrize("up", [-1.1, 1.1, np.nan, np.inf])
def test_invalid_uprightness_rejected(up):
    with pytest.raises(ValueError, match="uprightness"):
        score(up=up)


@pytest.mark.parametrize("previous,current,velocity", [
    ([0, 0], [0, 0, 1], [0]),
    ([0, 0, 1], [0, 0], [0]),
    ([0, 0, np.nan], [0, 0, 1], [0]),
    ([0, 0, 1], [0, np.inf, 1], [0]),
    ([0, 0, 1], [0, 0, 1], []),
    ([0, 0, 1], [0, 0, 1], [[0]]),
    ([0, 0, 1], [0, 0, 1], [np.nan]),
])
def test_invalid_arrays_rejected(previous, current, velocity):
    with pytest.raises(ValueError):
        body_reward(previous, current, 1, velocity, 0.1)


def test_inputs_are_not_modified():
    previous, current, velocity = np.array([0., 0., 0.8]), np.array([0.5, 0., 0.8]), np.array([2.])
    originals = [a.copy() for a in (previous, current, velocity)]
    body_reward(previous, current, 1, velocity, 0.1)
    for actual, expected in zip((previous, current, velocity), originals, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_grounded_reward_rejects_airborne_translation_and_penalizes_sliding():
    from noesis.embodied_metrics import locomotion_reward
    def run(loads, feet):
        return locomotion_reward([0, 0, .8], [.5, 0, .8], 1, [0], .1,
                                 [1, 0, 0], loads, feet)
    stationary = np.zeros((6, 3))
    supported = run(np.ones(6), stationary)
    airborne = run(np.zeros(6), stationary)
    sliding = run(np.ones(6), np.tile([5, 0, 0], (6, 1)))
    assert supported['reward'] > airborne['reward']
    assert airborne['grounded_progress'] == 0
    assert supported['supporting_feet'] == 6
    assert sliding['reward'] < supported['reward']
    assert sliding['stance_slip_mm_s'] == 5


def test_grounded_reward_tracks_thorax_heading_not_fixed_world_axis():
    from noesis.embodied_metrics import locomotion_reward
    r = locomotion_reward([0, 0, .8], [0, .5, .8], 1, [0], .1,
                          [0, 1, 0], np.ones(6), np.zeros((6, 3)))
    assert r['forward_speed_mm_s'] == 5 and r['grounded_progress'] == .5
    with pytest.raises(ValueError):
        locomotion_reward([0, 0, .8], [0, 0, .8], 1, [0], .1,
                          [0, 1, 0], [-1] * 6, np.zeros((6, 3)))


def test_skating_cannot_earn_progress_and_persistent_obstacle_contact_costs_reward():
    from noesis.embodied_metrics import locomotion_reward
    args = ([0, 0, .8], [.01, 0, .8], 1., np.zeros(3), .01, [1, 0, 0], np.ones(6))
    planted = locomotion_reward(*args, np.zeros((6, 3)))
    skating = locomotion_reward(*args, np.tile([1, 0, 0], (6, 1)))
    obstacle = locomotion_reward(*args, np.zeros((6, 3)), 1.)
    assert planted['grounded_progress'] > 0
    assert skating['grip_factor'] < .1
    assert obstacle['reward'] < planted['reward']

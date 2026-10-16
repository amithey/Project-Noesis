import numpy as np
import pytest

from noesis.virtual_agent import AgentParams, SteeringAgent, path_efficiency, plume, steering_signal


def test_plume_peaks_at_food_and_decays():
    assert plume(0, 0) == 1.0 and plume(10, 0) < plume(5, 0) < 1.0


def test_steering_signal_sign_and_silence():
    assert steering_signal(3.0, 1.0) == 0.5 and steering_signal(1.0, 3.0) == -0.5
    assert steering_signal(0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        steering_signal(-1.0, 1.0)


def test_left_activity_turns_left_and_sign_flips():
    p = AgentParams(rot_diffusion_rad_sqrt_s=0.0)
    left = SteeringAgent((0, 0), 0.0, p)
    for _ in range(20):
        left.step(0.05, 30.0, 10.0)
    assert left.theta > 0
    flipped = SteeringAgent((0, 0), 0.0, p, sign=-1.0)
    for _ in range(20):
        flipped.step(0.05, 30.0, 10.0)
    assert flipped.theta < 0


def test_bias_cancels_constant_asymmetry():
    p = AgentParams(rot_diffusion_rad_sqrt_s=0.0)
    a = SteeringAgent((0, 0), 0.0, p, bias=steering_signal(30.0, 10.0))
    for _ in range(40):
        a.step(0.05, 30.0, 10.0)
    assert abs(a.theta) < 1e-9


def test_no_input_means_diffusion_only_and_constant_speed():
    a = SteeringAgent((0, 0), 0.0, AgentParams(rot_diffusion_rad_sqrt_s=0.0))
    for _ in range(20):
        a.step(0.05, None, None)
    assert a.theta == 0.0 and a.x == pytest.approx(8.0)


def test_antennae_are_on_opposite_sides():
    a = SteeringAgent((0, 0), 0.0)
    left, right = a.antenna_positions()
    assert left[1] > 0 > right[1]


def test_bearing_error_and_efficiency():
    a = SteeringAgent((0, 0), 0.0)
    assert a.bearing_error((0, 5)) == pytest.approx(np.pi / 2)
    assert path_efficiency([(0, 0), (1, 0), (2, 0)]) == pytest.approx(1.0)
    assert path_efficiency([(0, 0), (1, 0), (0, 0)]) == 0.0


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        AgentParams(speed_mm_s=float("nan"))
    with pytest.raises(ValueError):
        AgentParams(smooth_s=0.0)

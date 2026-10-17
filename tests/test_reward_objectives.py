"""Opt-in reward semantics, preserving every physical measurement."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from noesis.embodied_metrics import GROUNDED_OBJECTIVES, locomotion_reward

V3, V4 = GROUNDED_OBJECTIVES


def reference(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stationary(up=0.9, height=0.8, feet=3, slip=0.0, objective=V4):
    loads = np.zeros(6)
    loads[:feet] = 1.0
    return locomotion_reward(
        [0, 0, height],
        [0, 0, height],
        up,
        np.zeros(3),
        0.005,
        [1, 0, 0],
        loads,
        np.tile([slip, 0, 0], (6, 1)),
        objective=objective,
    )


def test_v3_default_is_exact_and_v4_changes_only_reward_on_physical_samples():
    v3 = reference("standing_first_v3_reference.py")
    v4 = reference("standing_gate_v4_reference.py")
    rng = np.random.default_rng(71)
    for _ in range(1000):
        previous = np.array([0.0, 0.0, 0.8])
        args = (
            previous,
            previous + rng.uniform(-0.1, 0.1, 3),
            rng.uniform(-1, 1),
            rng.uniform(-20, 20, 42),
            0.005,
            rng.normal(size=3),
            rng.uniform(0, 1, 6) * (rng.random(6) > 0.4),
            rng.uniform(-100, 100, (6, 3)),
            rng.uniform(0, 1.3),
        )
        old = v3.locomotion_reward(*args)
        default = locomotion_reward(*args)
        explicit = locomotion_reward(*args, objective=V3)
        candidate = locomotion_reward(*args, objective=V4)
        assert old == default == explicit
        assert set(old) == set(candidate)
        assert {k: v for k, v in old.items() if k != "reward"} == {
            k: v for k, v in candidate.items() if k != "reward"
        }
        assert candidate["reward"] == pytest.approx(v4.reward(candidate), abs=1e-14)


def test_v4_does_not_trade_standing_time_for_one_extra_loaded_foot():
    for objective in (V3, V4):
        stable = stationary(feet=2, slip=15.0, objective=objective)["reward"]
        risky = stationary(feet=3, slip=14.0, objective=objective)["reward"]
        fallen = stationary(up=-1.0, height=0.2, feet=0, objective=objective)["reward"]
        delta = (0.9 * stable + 0.1 * fallen) - (0.8 * risky + 0.2 * fallen)
        assert delta > 0 if objective == V4 else delta < 0


@pytest.mark.parametrize("kwargs", [{"up": 0.69}, {"height": 0.29}, {"height": 1.51}, {"feet": 0}])
def test_v4_no_standing_credit_without_actual_posture_height_or_ground(kwargs):
    assert stationary(**kwargs)["reward"] < 0


def test_unknown_reward_objective_is_rejected():
    with pytest.raises(ValueError, match="unknown grounded reward"):
        stationary(objective="unknown")

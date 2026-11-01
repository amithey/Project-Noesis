import importlib.util

import numpy as np
import pytest
from noesis_gpu import chain_diagnostic as cd
from noesis_gpu.controller import FrozenBrain
from test_learning import state
from test_learning_rate_and_log import HI, LO, REST
from test_weight_integrity import build

pytestmark = pytest.mark.skipif(importlib.util.find_spec("noesis.circuit_learning") is None,
                                reason="Requires pinned Noesis core")


class FakeBody:
    """Deterministic stand-in with FlyGymBatch's call shape; a physical body is NOT simulated."""

    def __init__(self):
        self.t = 0
        self.commands = []

    def observe(self):
        out = {key: np.asarray(value)[None].copy() for key, value in state().items()}
        out["angles"] = np.full((1, 42), .01 * np.sin(self.t / 7.0))
        return out

    def advance(self, commands, adhesion, steps):
        assert commands.shape == (1, 42) and adhesion.shape == (1, 6)
        self.commands.append(commands.copy())
        self.t += 1


def arm_run(tmp_path, arm, seed=4, steps=120):
    cn, parent = build(tmp_path)
    brain = cd.ChainBrain(cn, REST, LO, HI, checkpoint=parent, arm=arm, seed=seed)
    return brain, cd.run_arm(brain, FakeBody(), steps, 50, 5.0), cn, parent


def test_off_arm_is_bit_identical_to_the_plain_frozen_brain_so_recording_changes_nothing(tmp_path):
    brain, arrays, cn, parent = arm_run(tmp_path, "off")
    plain = FrozenBrain(cn, REST, LO, HI, seed=4, checkpoint=parent)
    body, commands = FakeBody(), []
    for _ in range(120):
        obs = {key: value[0] for key, value in body.observe().items()}
        command, adhesion = plain.step(obs, 5.0)
        body.advance(command[None], adhesion[None], 50)
        commands.append(command)
    np.testing.assert_array_equal(arrays["command"], np.stack(commands))
    assert plain.total_spikes == arrays["total_spikes"].sum()
    assert brain.weights_unchanged() and plain.initial_weights_sha256 == brain.initial_weights_sha256


def test_zero_current_arm_is_identical_to_off_at_every_recorded_stage(tmp_path):
    _, off, _, _ = arm_run(tmp_path, "off")
    zero_brain, zero, _, _ = arm_run(tmp_path, "zero")
    report = cd.invariance_report(off, zero)
    assert report["identical"], report
    assert zero_brain.exploration.state.any()  # the exploration stream really advanced in the zero arm
    assert cd.first_divergence(zero, off) == {key: None for key in cd.RECORDED}


def test_active_arm_injects_the_logged_current_and_never_updates_weights(tmp_path):
    brain, active, _, _ = arm_run(tmp_path, "active", steps=200)
    assert len(brain.circuit_idx) > 0 and active["drive"].any()
    assert np.abs(active["drive"]).max() <= 1.5 + 1e-6
    assert brain.weights_unchanged() and brain.updates == 0
    with pytest.raises(RuntimeError):
        brain.learn(.5, 5.0)


def test_active_drive_equals_the_zero_arm_stream_it_replaces(tmp_path):
    a, _, _, _ = arm_run(tmp_path, "active", steps=60)
    z, _, _, _ = arm_run(tmp_path, "zero", steps=60)
    np.testing.assert_array_equal(a.exploration.state, z.exploration.state)
    assert np.array_equal(a.circuit_idx, z.circuit_idx)


def test_arm_is_validated(tmp_path):
    cn, parent = build(tmp_path)
    with pytest.raises(ValueError):
        cd.ChainBrain(cn, REST, LO, HI, checkpoint=parent, arm="learning")


def _planted(lag, noise, steps=1500, joints=8, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(steps, joints))
    y = np.roll(x, lag, axis=0) * 1.0 + noise * rng.normal(size=(steps, joints))
    return x, y


def test_coupling_recovers_a_planted_lag_and_rejects_an_unrelated_response():
    x, y = _planted(7, noise=1.0)
    found = cd.coupling(x, y, max_lag=20, permutations=200)
    assert found["best_lag_steps"] == 7 and found["sign"] == 1 and found["reference_rank"] < .01
    assert found["observed_r2"] > found["reference_p99"]
    unrelated = cd.coupling(x, np.random.default_rng(9).normal(size=x.shape), max_lag=20, permutations=200)
    assert unrelated["reference_rank"] > .05


def test_coupling_skips_unusable_joints_and_rejects_bad_shapes():
    x, y = _planted(3, noise=.5)
    x[:, 0] = np.nan
    y[:, 1] = 0
    assert cd.coupling(x, y, max_lag=10, permutations=50)["joints_used"] == 6
    with pytest.raises(ValueError):
        cd.coupling(x, y[:-1])
    assert cd.coupling(x[:100], y[:100])["status"] == "insufficient"


def test_joint_predictor_is_agonist_minus_antagonist_with_nan_for_missing_pools():
    drive = np.array([[1., 3., 10., 0.], [2., 4., 20., 0.]])
    pos = np.array([[1, 1, 0, 0], [0, 0, 1, 0]], bool)
    neg = np.array([[0, 0, 1, 1], [0, 0, 0, 0]], bool)
    out = cd.joint_predictor(drive, pos, neg)
    np.testing.assert_allclose(out[:, 0], [2 - 5, 3 - 10])
    assert np.isnan(out[:, 1]).all()


def test_chain_report_negative_control_scores_the_arms_that_never_received_the_current(tmp_path):
    arms = {}
    for arm in cd.ARMS:
        _, arms[arm], _, _ = arm_run(tmp_path, arm, steps=500)
    n = len(arms["active"]["drive_idx"])
    pos, neg = np.zeros((42, n), bool), np.zeros((42, n), bool)
    pos[0, : n // 2], neg[0, n // 2:] = True, True
    report = cd.chain_report(arms, pos, neg, max_lag=10, permutations=50)
    assert report["invariance_off_vs_zero"]["identical"]
    assert set(report["coupling"]) == set(cd.ARMS) and set(report["coupling"]["off"]) == set(cd.STAGES)

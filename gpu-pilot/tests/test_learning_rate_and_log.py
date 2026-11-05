import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_learning import state
from test_weight_integrity import build

from noesis_gpu import learning, train
from noesis_gpu.learning import DEFAULT_LEARNING_RATE, MAX_LEARNING_RATE, LearningBrain
from noesis_gpu.profile_run import COMPLETED, SIDECAR, main

pytestmark = pytest.mark.skipif(importlib.util.find_spec("noesis.circuit_learning") is None,
                                reason="Requires pinned Noesis core")
REST, LO, HI = np.zeros(42), np.full(42, -.7), np.full(42, .7)


def run(brain, steps=150):
    brain.learning_active = True
    for i in range(steps):
        brain.step(state(), 5)
        brain.learn(.6 if i % 4 else -.5, 5)
    return brain


def delta(brain, parent_weights):
    return brain.net.data[brain.learner.edge_indices] - parent_weights


def test_default_learning_rate_is_the_pilot_value_and_explicit_default_is_identical(tmp_path):
    cn, parent = build(tmp_path)
    a = run(LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=3))
    b = run(LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=3, learning_rate=DEFAULT_LEARNING_RATE))
    assert DEFAULT_LEARNING_RATE == .2 and a.learning_rate == .2
    np.testing.assert_array_equal(a.net.data, b.net.data)
    assert a.learner.learning_rate == .2


def test_learning_rate_scales_the_update_size_without_changing_its_direction(tmp_path):
    cn, parent = build(tmp_path)
    base = LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=3)
    start = base.net.data[base.learner.edge_indices].copy()
    fast = run(LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=3, learning_rate=.2))
    slow = run(LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=3, learning_rate=.02))
    d_fast, d_slow = delta(fast, start), delta(slow, start)
    cosine = float(d_fast @ d_slow / (np.linalg.norm(d_fast) * np.linalg.norm(d_slow)))
    assert cosine > .99  # same noise, same credit signal: same direction
    assert np.linalg.norm(d_slow) / np.linalg.norm(d_fast) == pytest.approx(.1, rel=.15)
    assert slow.learner.learning_rate == .02


@pytest.mark.parametrize("bad", [0, -.1, MAX_LEARNING_RATE + .1, float("nan"), float("inf")])
def test_learning_rate_range_is_enforced(tmp_path, bad):
    cn, parent = build(tmp_path)
    with pytest.raises(ValueError):
        LearningBrain(cn, REST, LO, HI, checkpoint=parent, learning_rate=bad)


def test_bounds_still_hold_at_the_largest_rate(tmp_path):
    cn, parent = build(tmp_path)
    brain = run(LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=1, learning_rate=MAX_LEARNING_RATE), 300)
    ratio = np.abs(brain.net.data[brain.learner.edge_indices]) / np.abs(brain.learner.initial_weights)
    assert ratio.min() >= .5 - 1e-6 and ratio.max() <= 1.5 + 1e-6
    assert np.array_equal(np.sign(brain.net.data[brain.learner.edge_indices]),
                          np.sign(brain.learner.initial_weights))


def test_exploration_log_is_off_by_default_and_never_changes_learning(tmp_path):
    cn, parent = build(tmp_path)
    plain = run(LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=5))
    logged = run(LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=5, record_exploration=True))
    assert plain.exploration_log() is None
    np.testing.assert_array_equal(plain.net.data, logged.net.data)  # logging is read-only


def test_exploration_log_contents_are_aligned_bounded_and_reproducible(tmp_path):
    cn, parent = build(tmp_path)
    brain = LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=5, record_exploration=True)
    for i in range(10):  # inactive steps: nothing is injected, so nothing is logged
        brain.step(state(), 5)
        brain.learn(.5, 5)
    assert len(brain.exploration_log()["credit"]) == 0
    run(brain, 120)
    log = brain.exploration_log()
    assert log["credit"].shape == (120, len(brain.learner.motor_indices))
    assert log["drive"].shape[0] == 120 and log["drive"].shape[1] == len(log["drive_idx"])
    assert np.abs(log["credit"]).max() <= 1 and np.abs(log["drive"]).max() <= brain.exploration_mv_ms + 1e-6
    assert np.abs(log["credit"]).max() > 0 and np.abs(log["drive"]).max() > 0
    np.testing.assert_array_equal(log["motor_indices"], brain.learner.motor_indices)
    again = run(LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=5, record_exploration=True), 120)
    np.testing.assert_array_equal(again.exploration_log()["credit"][:, 0], log["credit"][:, 0])  # same seed
    other = run(LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=6, record_exploration=True), 120)
    assert not np.array_equal(other.exploration_log()["credit"], log["credit"])


def argv(tmp_path, *extra):
    return ["--noesis-source", str(tmp_path), "--data", str(tmp_path), "--checkpoint",
            str(tmp_path / "parent.npz"), "--output", str(tmp_path / "out"), *extra]


def fake_train(cn, parent, steps=30):
    def run_fake(args):
        brain = learning.LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=2)
        run(brain, steps)
        args.output.mkdir(parents=True)
        report = {"status": COMPLETED, "reward_objective": "curriculum-standing-v7", "seed": args.seed,
                  "rate": brain.learning_rate}
        (args.output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report
    return SimpleNamespace(parser=train.parser, run=run_fake)


def test_profile_run_passes_the_rate_and_writes_the_exploration_log(tmp_path):
    cn, parent = build(tmp_path)
    code = main(argv(tmp_path, "--profile", "gripped-steps", "--learning-rate", "0.05", "--log-exploration"),
                fake_train(cn, parent))
    assert code == 0
    sidecar = json.loads((tmp_path / "out" / SIDECAR).read_text(encoding="utf-8"))
    assert sidecar["learning_rate"] == .05 and sidecar["exploration_log"]["steps"] == 30
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["learning_rate"] == .05 and report["rate"] == .05  # the brain really used it
    with np.load(tmp_path / "out" / "exploration-log.npz") as z:
        assert z["credit"].shape[0] == 30


def test_profile_run_defaults_leave_the_rate_and_log_alone(tmp_path):
    cn, parent = build(tmp_path)
    assert main(argv(tmp_path, "--profile", "gripped-steps"), fake_train(cn, parent)) == 0
    sidecar = json.loads((tmp_path / "out" / SIDECAR).read_text(encoding="utf-8"))
    assert "learning_rate" not in sidecar and "exploration_log" not in sidecar
    assert not (tmp_path / "out" / "exploration-log.npz").exists()
    assert json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))["rate"] == .2


@pytest.mark.parametrize("bad", ["0", "-1", "2.5", "nan"])
def test_profile_run_rejects_a_bad_rate_before_creating_output(tmp_path, bad):
    cn, parent = build(tmp_path)
    assert main(argv(tmp_path, "--profile", "gripped-steps", "--learning-rate", bad), fake_train(cn, parent)) == 2
    assert not (tmp_path / "out").exists()

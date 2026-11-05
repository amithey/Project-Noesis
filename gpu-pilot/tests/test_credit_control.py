import importlib.util
import json

import numpy as np
import pytest
from test_learning_rate_and_log import HI, LO, REST, argv, delta, fake_train, run
from test_weight_integrity import build

from noesis_gpu.learning import LearningBrain
from noesis_gpu.profile_run import SIDECAR, main

pytestmark = pytest.mark.skipif(importlib.util.find_spec("noesis.circuit_learning") is None,
                                reason="Requires pinned Noesis core")


def birth(tmp_path, control, seed=4, steps=300):
    cn, parent = build(tmp_path)
    return run(LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=seed, record_exploration=True,
                             credit_control=control), steps)


def test_control_leaves_injected_current_and_logged_real_credit_unchanged(tmp_path):
    real, ctrl = birth(tmp_path, None), birth(tmp_path, "independent")
    a, b = real.exploration_log(), ctrl.exploration_log()
    np.testing.assert_array_equal(a["drive"], b["drive"])  # same body-side perturbation
    np.testing.assert_array_equal(a["credit"], b["credit"])  # the real innovation is still recorded
    assert "applied_credit" not in a and b["applied_credit"].shape == b["credit"].shape


def test_applied_credit_is_independent_but_identically_distributed(tmp_path):
    ctrl = birth(tmp_path, "independent", steps=400).exploration_log()
    real, applied = ctrl["credit"].ravel(), ctrl["applied_credit"].ravel()
    keep = (real != 0) & (applied != 0)
    assert keep.sum() > 1000
    assert abs(np.corrcoef(real[keep], applied[keep])[0, 1]) < .05
    assert np.abs(applied).max() <= 1
    assert applied[keep].std() == pytest.approx(real[keep].std(), rel=.1)


def test_control_updates_match_in_size_but_not_in_direction_and_respect_bounds(tmp_path):
    cn, parent = build(tmp_path)
    start_brain = LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=4)
    start = start_brain.net.data[start_brain.learner.edge_indices].copy()
    real, ctrl = birth(tmp_path, None), birth(tmp_path, "independent")
    dr, dc = delta(real, start), delta(ctrl, start)
    assert 0 < np.linalg.norm(dc) / np.linalg.norm(dr) < 3 and np.linalg.norm(dc) / np.linalg.norm(dr) > .33
    assert abs(float(dr @ dc / (np.linalg.norm(dr) * np.linalg.norm(dc)))) < .3
    ratio = np.abs(ctrl.net.data[ctrl.learner.edge_indices]) / np.abs(ctrl.learner.initial_weights)
    assert ratio.min() >= .5 - 1e-6 and ratio.max() <= 1.5 + 1e-6
    assert np.array_equal(np.sign(ctrl.net.data[ctrl.learner.edge_indices]), np.sign(ctrl.learner.initial_weights))


def test_control_is_reproducible_and_depends_on_the_training_seed(tmp_path):
    a, b = birth(tmp_path, "independent", 4), birth(tmp_path, "independent", 4)
    c = birth(tmp_path, "independent", 5)
    np.testing.assert_array_equal(a.net.data, b.net.data)
    assert not np.array_equal(a.net.data, c.net.data)


def test_control_is_inert_when_learning_is_inactive_and_validated(tmp_path):
    cn, parent = build(tmp_path)
    with pytest.raises(ValueError):
        LearningBrain(cn, REST, LO, HI, checkpoint=parent, credit_control="shuffled")
    brain = LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=1, credit_control="independent")
    before = brain.net.data.copy()
    for _ in range(20):
        brain.learn(.5, 5)
    np.testing.assert_array_equal(before, brain.net.data)


def test_profile_run_records_the_control_in_the_sidecar_only_when_asked(tmp_path):
    cn, parent = build(tmp_path)
    assert main(argv(tmp_path, "--profile", "gripped-steps", "--credit-control", "independent"),
                fake_train(cn, parent)) == 0
    assert json.loads((tmp_path / "out" / SIDECAR).read_text(encoding="utf-8"))["credit_control"] == "independent"
    plain = tmp_path / "plain"
    plain.mkdir()
    cn2, parent2 = build(plain)
    assert main(argv(plain, "--profile", "gripped-steps"), fake_train(cn2, parent2)) == 0
    assert "credit_control" not in json.loads((plain / "out" / SIDECAR).read_text(encoding="utf-8"))

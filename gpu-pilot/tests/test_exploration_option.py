"""E1: lowering the exploration amplitude never raises the original 1.5 mV/ms cap."""
import importlib.util

import numpy as np
import pytest
from test_learning import state
from test_weight_integrity import build

pytestmark = pytest.mark.skipif(importlib.util.find_spec("noesis.circuit_learning") is None,
                                reason="Requires pinned Noesis core")
REST, LO, HI = np.zeros(42), np.full(42, -.7), np.full(42, .7)


def run(brain, steps=150):
    brain.learning_active = True
    for i in range(steps):
        brain.step(state(), 5)
        brain.learn(.6 if i % 4 else -.5, 5)


def test_default_is_unchanged_original_amplitude(tmp_path):
    from noesis_gpu.learning import MAX_EXPLORATION_MV_MS, LearningBrain
    cn, parent = build(tmp_path)
    brain = LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=3)
    assert MAX_EXPLORATION_MV_MS == 1.5 and brain.exploration_mv_ms == 1.5
    explicit = LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=3, exploration_mv_ms=1.5)
    run(brain), run(explicit)
    np.testing.assert_array_equal(brain.net.data, explicit.net.data)  # identical to default
    assert 1.0 < brain.max_observed_exploration <= 1.5


def test_lower_amplitude_scales_exploration_and_keeps_bounds(tmp_path):
    from noesis_gpu.learning import LearningBrain
    cn, parent = build(tmp_path)
    half = LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=3, exploration_mv_ms=.75)
    full = LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=3)
    run(half), run(full)
    assert 0 < half.max_observed_exploration <= .75
    assert half.max_observed_exploration < full.max_observed_exploration
    reference = full.learner.initial_weights
    ratio = np.abs(half.net.data[half.learner.edge_indices]) / np.abs(reference)
    assert ratio.min() >= .5 - 1e-6 and ratio.max() <= 1.5 + 1e-6
    assert np.array_equal(np.sign(half.net.data[half.learner.edge_indices]), np.sign(reference))


@pytest.mark.parametrize("bad", [0, -.1, 1.51, 3., float("nan"), float("inf")])
def test_cap_cannot_be_raised_or_disabled(tmp_path, bad):
    from noesis_gpu.learning import LearningBrain
    cn, parent = build(tmp_path)
    with pytest.raises(ValueError):
        LearningBrain(cn, REST, LO, HI, checkpoint=parent, exploration_mv_ms=bad)


def test_frozen_brain_ignores_exploration_option_and_stays_exact(tmp_path):
    from noesis_gpu.learning import LearningBrain
    cn, parent = build(tmp_path)
    brain = LearningBrain(cn, REST, LO, HI, checkpoint=parent, seed=3, exploration_mv_ms=.75)
    for _ in range(20):
        brain.step(state(), 5)
        brain.learn(.5, 5)  # learning inactive: no exploration drive, no update
    assert brain.updates == 0 and brain.max_observed_exploration == 0 and brain.weights_unchanged()

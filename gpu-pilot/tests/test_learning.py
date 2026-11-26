import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from noesis_gpu.learning import score_transition, standing_mask


def state():
    return {"position": np.array([0., 0., 1.]), "up_body": np.array([0., 0., 1.]),
            "omega": np.zeros(3), "angles": np.zeros(42), "velocity": np.zeros(42),
            "heading": np.array([1., 0., 0.]), "feet_position": np.zeros((6, 3)),
            "load": np.ones(6), "foot_load": np.zeros(6)}


def test_whole_leg_contact_cannot_count_as_foot_support():
    if importlib.util.find_spec("noesis.embodied_metrics") is None:
        pytest.skip("Requires current Noesis source")
    a = state()
    assert not standing_mask(a)
    unsupported = score_transition(a, a, 5.)
    a["foot_load"][0] = 1
    supported = score_transition(a, a, 5.)
    assert standing_mask(a)
    assert supported["reward"] > unsupported["reward"]


def test_learning_uses_original_reference_bounds_and_freeze_is_exact(tmp_path):
    if importlib.util.find_spec("noesis.circuit_learning") is None:
        pytest.skip("Requires current isolated Noesis source")
    import pandas as pd
    from noesis.connectome import Connectome
    from noesis.lif import LIFNetwork
    from noesis.plasticity import MotorPlasticity
    from scipy import sparse

    from noesis_gpu.bridge import LEGS
    from noesis_gpu.controller import FrozenBrain
    from noesis_gpu.learning import LearningBrain

    rows = []
    for leg in LEGS:
        side = "left" if leg[0] == "L" else "right"
        pos = {"F": "front", "M": "middle", "H": "hind"}[leg[1]]
        for cls, sub, primary, flow in (
                ("leg_motor_neuron", f"{pos}_leg_motor_neuron", "tibia_flexor", "efferent"),
                ("leg_motor_neuron", f"{pos}_leg_motor_neuron", "tibia_extensor", "efferent"),
                ("sensory_neuron", f"{pos}_leg_claw_chordotonal_organ_neuron", "claw", "afferent")):
            rows.append({"Class": cls, "Sub Class": sub, "Primary Cell Type": primary,
                         "Flow": flow, "Nerve": f"{side}_leg", "Soma side": side})
    table = pd.DataFrame(rows)
    cn = Connectome(table, sparse.csr_matrix(np.ones((len(table), len(table)), dtype=np.float32)))
    targets = np.flatnonzero(table["Class"].eq("leg_motor_neuron"))
    original = LIFNetwork(cn.W)
    learner = MotorPlasticity(original, targets, learning_rate=.2, tau_eligibility_ms=100,
                              rule="perturbation", normalize_reward_error=True,
                              bias_correct_reward_error=True)
    original.data[learner.edge_indices] *= .9
    checkpoint = tmp_path / "parent.npz"
    learner.save(checkpoint)
    body = SimpleNamespace(rest=np.zeros(42), lo=np.full(42, -.7), hi=np.full(42, .7))
    trained = LearningBrain(cn, body.rest, body.lo, body.hi, checkpoint=checkpoint,
                            seed=7, bg_hz=100)
    frozen = FrozenBrain(cn, body.rest, body.lo, body.hi, checkpoint=checkpoint,
                          seed=7, bg_hz=100)
    np.testing.assert_array_equal(trained.learner.initial_weights, 1)
    for _ in range(10):
        a, _ = trained.step(state(), 5)
        b, _ = frozen.step(state(), 5)
        np.testing.assert_array_equal(a, b)
        trained.learn(.5, 5)
    assert trained.updates == 0
    assert trained.weights_unchanged()
    trained.learning_active = True
    for i in range(200):
        trained.step(state(), 5)
        trained.learn(.6 if i % 5 else -.4, 5)
    assert trained.updates == 200
    assert 0 < trained.max_observed_exploration <= 1.5
    assert not trained.weights_unchanged()
    assert frozen.weights_unchanged()
    assert np.all((trained.net.data >= .5) & (trained.net.data <= 1.5))
    assert np.all(trained.net.data > 0)
    second = tmp_path / "child.npz"
    trained.learner.save(second)
    again = LearningBrain(cn, body.rest, body.lo, body.hi, checkpoint=second)
    np.testing.assert_array_equal(again.learner.initial_weights, 1)
    np.testing.assert_array_equal(again.net.data, trained.net.data)

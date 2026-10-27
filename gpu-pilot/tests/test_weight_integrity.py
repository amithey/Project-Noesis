"""Multi-cycle save/load integrity of learning brains (synthetic connectome)."""
import importlib.util

import numpy as np
import pytest
from test_learning import state

pytestmark = pytest.mark.skipif(importlib.util.find_spec("noesis.circuit_learning") is None,
                                reason="Requires pinned Noesis core")


def build(tmp_path):
    import pandas as pd
    from noesis.connectome import Connectome
    from noesis.lif import LIFNetwork
    from noesis.plasticity import MotorPlasticity
    from scipy import sparse

    from noesis_gpu.bridge import LEGS

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
    w = np.ones((len(table), len(table)), dtype=np.float32)
    w[:, ::3] *= -1  # mixed signs so sign preservation is meaningful
    cn = Connectome(table, sparse.csr_matrix(w))
    targets = np.flatnonzero(table["Class"].eq("leg_motor_neuron"))
    net = LIFNetwork(cn.W)
    learner = MotorPlasticity(net, targets, learning_rate=.2, tau_eligibility_ms=100,
                              rule="perturbation", normalize_reward_error=True,
                              bias_correct_reward_error=True)
    path = tmp_path / "parent.npz"
    learner.save(path)
    return cn, path


def test_cycles_keep_topology_signs_bounds_and_independence(tmp_path):
    from noesis_gpu.learning import LearningBrain

    cn, parent = build(tmp_path)
    rest, lo, hi = np.zeros(42), np.full(42, -.7), np.full(42, .7)
    reference = LearningBrain(cn, rest, lo, hi, checkpoint=parent, seed=1)
    ref_data = reference.net.data.copy()
    ref_struct = (reference.net.indices.copy(), reference.net.indptr.copy())
    parent_reference = reference.learner.initial_weights.copy()
    assert np.all(np.abs(parent_reference) == 1) and (parent_reference < 0).any()
    current = parent
    for cycle in range(3):
        brain = LearningBrain(cn, rest, lo, hi, checkpoint=current, seed=cycle)
        other = LearningBrain(cn, rest, lo, hi, checkpoint=parent, seed=99)
        assert not np.shares_memory(brain.net.data, other.net.data)
        np.testing.assert_array_equal(brain.learner.initial_weights,
                                      parent_reference)  # original reference kept
        brain.learning_active = True
        for i in range(150):
            brain.step(state(), 5)
            brain.learn(.6 if i % 4 else -.5, 5)
        np.testing.assert_array_equal(other.net.data, ref_data)  # sibling unaffected by learning
        np.testing.assert_array_equal(brain.net.indices, ref_struct[0])
        np.testing.assert_array_equal(brain.net.indptr, ref_struct[1])
        assert np.array_equal(np.sign(brain.net.data), np.sign(ref_data))
        ratio = np.abs(brain.net.data) / np.abs(ref_data)
        assert ratio.min() >= .5 - 1e-6 and ratio.max() <= 1.5 + 1e-6
        current = tmp_path / f"child{cycle}.npz"
        brain.learner.save(current)

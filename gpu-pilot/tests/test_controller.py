from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from noesis_gpu.bridge import LEGS
from noesis_gpu.controller import FrozenPopulation, import_checkpoint


@pytest.fixture
def connectome():
    from noesis.connectome import Connectome

    rows = []
    for leg in LEGS:
        side = "left" if leg[0] == "L" else "right"
        position = {"F": "front", "M": "middle", "H": "hind"}[leg[1]]
        for muscle in ("tibia_flexor", "tibia_extensor"):
            rows.append({"Flow": "efferent", "Class": "leg_motor_neuron",
                         "Sub Class": f"{position}_leg_motor_neuron",
                         "Primary Cell Type": muscle, "Nerve": f"{side}_leg", "Soma side": side})
        rows.append({"Flow": "afferent", "Class": "sensory_neuron",
                     "Sub Class": f"{position}_leg_claw_chordotonal_organ_neuron",
                     "Primary Cell Type": "claw", "Nerve": f"{side}_leg", "Soma side": side})
    rows.append({"Flow": "afferent", "Class": "sensory_neuron",
                 "Sub Class": "johnstons_organ_C_neuron", "Primary Cell Type": "JO",
                 "Nerve": "left_antenna", "Soma side": "left"})
    n = len(rows)
    return Connectome(pd.DataFrame(rows), sparse.csr_matrix(np.ones((n, n), dtype=np.float32)))


def observations(worlds):
    return {"angles": np.zeros((worlds, 42)), "velocity": np.zeros((worlds, 42)),
            "load": np.ones((worlds, 6)), "up_body": np.tile([0, 0, 1], (worlds, 1)),
            "omega": np.zeros((worlds, 3))}


def body(worlds):
    return SimpleNamespace(n_worlds=worlds, rest=np.zeros(42), lo=np.full(42, -0.7),
                           hi=np.full(42, 0.7))


def test_world_zero_is_independent_of_batch_size_and_other_world_state(connectome):
    one = FrozenPopulation(connectome, body(1), seed=11, bg_hz=100)
    two = FrozenPopulation(connectome, body(2), seed=11, bg_hz=100)
    for tick in range(20):
        obs = observations(2)
        obs["angles"][1] = np.sin(tick)
        a, _ = one.step(observations(1), 5)
        b, _ = two.step(obs, 5)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(one.brains[0].net.v, two.brains[0].net.v)
        np.testing.assert_array_equal(one.brains[0].net.g, two.brains[0].net.g)
    assert one.brains[0].total_spikes > 0
    assert all(brain.weights_unchanged() for brain in one.brains + two.brains)
    assert not np.shares_memory(two.brains[0].net.v, two.brains[1].net.v)
    assert not np.shares_memory(two.brains[0].muscles.a_pos, two.brains[1].muscles.a_pos)
    assert np.shares_memory(two.brains[0].net.data, two.brains[1].net.data)
    with pytest.raises(ValueError):
        two.brains[1].net.data[0] = 9


def test_memory_budget_rejects_before_allocating_whole_population(connectome):
    with pytest.raises(MemoryError, match="Reduce --worlds"):
        FrozenPopulation(connectome, body(100), memory_budget_gib=1e-8)


def test_checkpoint_import_uses_existing_validation_and_keeps_signs(connectome, tmp_path):
    from noesis.lif import LIFNetwork
    from noesis.plasticity import MotorPlasticity

    net = LIFNetwork(connectome.W, seed=0)
    learner = MotorPlasticity(net, np.array([0, 1]), max_change=0.5)
    net.data[learner.edge_indices] *= 1.1
    path = tmp_path / "weights.npz"
    learner.save(path)
    other = LIFNetwork(connectome.W, seed=1)
    import_checkpoint(other, path)
    np.testing.assert_array_equal(other.data, net.data)
    with np.load(path, allow_pickle=False) as cp:
        values = {key: cp[key] for key in cp.files}
    values["weights"] = -values["weights"]
    np.savez(path, **values)
    untouched = other.data.copy()
    with pytest.raises(ValueError):
        import_checkpoint(other, path)
    np.testing.assert_array_equal(other.data, untouched)


def test_actual_upstream_body_and_brain_close_the_loop(connectome):
    from importlib.metadata import version

    if version("flygym") != "2.1.0":
        pytest.skip("Requires isolated FlyGym 2.1 runtime")
    from noesis_gpu.backend import FlyGymBatch

    physical = FlyGymBatch(1, "cpu")
    brain = FrozenPopulation(connectome, physical, seed=12, bg_hz=100)
    for _ in range(4):
        observations = physical.observe()
        commands, pads = brain.step(observations, 5)
        physical.advance(commands, pads, 50)
    assert brain.brains[0].total_spikes > 0
    assert brain.brains[0].weights_unchanged()
    assert np.isfinite(physical.observe()["position"]).all()


def test_version_four_checkpoint_weights_only_import(connectome, tmp_path):
    import inspect

    from noesis.lif import LIFNetwork
    from noesis.plasticity import MotorPlasticity

    if "bias_correct_reward_error" not in inspect.signature(MotorPlasticity).parameters:
        pytest.skip("This legacy Noesis source predates version-four checkpoints")
    original = LIFNetwork(connectome.W, seed=3)
    learner = MotorPlasticity(original, np.array([0, 1]), learning_rate=0.2,
                              tau_eligibility_ms=100, rule="perturbation",
                              normalize_reward_error=True, bias_correct_reward_error=True)
    original.data[learner.edge_indices] *= 0.9
    path = tmp_path / "v4.npz"
    learner.save(path)
    with np.load(path, allow_pickle=False) as cp:
        assert int(cp["version"]) == 4
    frozen = LIFNetwork(connectome.W, seed=4)
    import_checkpoint(frozen, path)
    np.testing.assert_array_equal(frozen.data, original.data)

import numpy as np
import pytest
from scipy import sparse

from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity


def network():
    return LIFNetwork(sparse.csr_matrix(np.array([
        [0., 3., 4.], [0., 0., -2.], [1., 0., 0.],
    ])))


def test_rewarded_coactivity_changes_only_existing_motor_edges_and_preserves_signs():
    net = network()
    original = net.data.copy()
    indices, indptr = net.indices.copy(), net.indptr.copy()
    learner = MotorPlasticity(net, [2], learning_rate=1)
    result = learner.update([0, 1, 2], 100, 1)
    assert result["changed_edges"] == 2
    assert np.all(np.abs(net.data[learner.edge_indices]) > [4, 2])
    assert np.array_equal(np.sign(net.data), np.sign(original))
    other = np.setdiff1d(np.arange(len(original)), learner.edge_indices)
    assert np.array_equal(net.data[other], original[other])
    assert np.array_equal(net.indices, indices) and np.array_equal(net.indptr, indptr)


@pytest.mark.parametrize("reward", [-1, 1])
def test_weights_remain_bounded_under_extreme_learning(reward):
    net = network()
    learner = MotorPlasticity(net, [2], learning_rate=1e300, max_change=.25)
    for _ in range(20):
        learner.update([0, 1, 2], 100, reward)
    relative = net.data[learner.edge_indices] / learner.initial_weights
    assert np.allclose(relative, 1 + reward * .25)
    assert np.array_equal(np.sign(net.data[learner.edge_indices]), [1, -1])


def test_no_reward_or_no_coactivity_does_not_change_weights():
    for spikes, reward in (([0, 1, 2], 0), ([0, 1], 1), ([], 1)):
        net = network()
        before = net.data.copy()
        learner = MotorPlasticity(net, [2])
        learner.update(spikes, 100, reward)
        assert np.array_equal(net.data, before)


def test_checkpoint_continuation_is_reproducible(tmp_path):
    a, b = network(), network()
    first, second = MotorPlasticity(a, [2]), MotorPlasticity(b, [2])
    first.update([0, 1, 2], 100, 1)
    path = tmp_path / "learned.npz"
    first.save(path)
    second.load(path, restore_traces=True)
    for spikes, reward in (([1, 2], -.5), ([0, 2], .8), ([], 0)):
        assert first.update(spikes, 50, reward) == second.update(spikes, 50, reward)
        assert np.array_equal(a.data, b.data)


def test_default_checkpoint_load_transfers_weights_with_fresh_episode_traces(tmp_path):
    learner = MotorPlasticity(network(), [2])
    learner.update([0, 1, 2], 100, 1)
    path = tmp_path / "episode.npz"
    learner.save(path)
    fresh = MotorPlasticity(network(), [2])
    fresh.load(path)
    assert np.array_equal(fresh.net.data, learner.net.data)
    assert not fresh.activity.any() and not fresh.eligibility.any()
    assert fresh.reward_baseline == 0


def test_frozen_loaded_metrics_report_real_changes_without_advancing_state(tmp_path):
    trained = MotorPlasticity(network(), [2], learning_rate=1)
    update = trained.update([0, 1, 2], 100, 1)
    path = tmp_path / "trained.npz"
    trained.save(path)
    frozen = MotorPlasticity(network(), [2], learning_rate=1)
    assert frozen.metrics()["max_relative_change"] == 0
    frozen.load(path)
    weights = frozen.net.data.copy()
    for _ in range(3):
        result = frozen.metrics()
        expected = np.abs(weights[frozen.edge_indices] / frozen.initial_weights - 1)
        assert result["plastic_edges"] == 2
        assert result["mean_relative_change"] == pytest.approx(expected.mean())
        assert result["max_relative_change"] == pytest.approx(expected.max())
        assert result["mean_relative_change"] > 0
        assert all(result[key] == update[key] for key in result)
    assert np.array_equal(frozen.net.data, weights)
    assert not frozen.activity.any() and not frozen.eligibility.any()
    assert frozen.reward_baseline == 0


@pytest.mark.parametrize("mismatch", ["topology", "initial", "parameters", "motor"])
def test_checkpoint_rejects_incompatible_network_or_configuration(tmp_path, mismatch):
    path = tmp_path / "weights.npz"
    MotorPlasticity(network(), [2]).save(path)
    net = network()
    if mismatch == "topology":
        net.indices[0] = 0
    if mismatch == "initial":
        net.data[1] += .1
    learner = MotorPlasticity(net, [1] if mismatch == "motor" else [2],
                             learning_rate=.3 if mismatch == "parameters" else .02)
    before = net.data.copy()
    with pytest.raises(ValueError, match="mismatch"):
        learner.load(path)
    assert np.array_equal(net.data, before)


def test_checkpoint_rejects_corrupt_weight_without_partial_mutation(tmp_path):
    learner = MotorPlasticity(network(), [2])
    path = tmp_path / "weights.npz"
    learner.save(path)
    with np.load(path) as saved:
        state = {key: saved[key] for key in saved.files}
    state["weights"][0] = np.nan
    np.savez(path, **state)
    before = learner.net.data.copy()
    with pytest.raises(ValueError):
        learner.load(path)
    assert np.array_equal(learner.net.data, before)


@pytest.mark.parametrize("spikes,dt,reward", [
    ([0], 0, 1), ([0], np.inf, 1), ([0], 100, np.nan),
    ([3], 100, 1), ([-1], 100, 1), ([.5], 100, 1),
])
def test_invalid_update_is_atomic(spikes, dt, reward):
    learner = MotorPlasticity(network(), [2])
    before = learner.net.data.copy()
    with pytest.raises(ValueError):
        learner.update(spikes, dt, reward)
    assert np.array_equal(learner.net.data, before)
    assert not learner.activity.any() and learner.reward_baseline == 0


def test_empty_motor_set_is_valid():
    learner = MotorPlasticity(network(), [])
    result = learner.update([0, 1], 100, 1)
    assert result["plastic_edges"] == 0
    assert result["max_relative_change"] == 0


def test_perturbation_reward_credits_excitation_and_disinhibition():
    net = network()
    learner = MotorPlasticity(net, [2], rule='perturbation', learning_rate=1)
    learner.update([0, 1], 100, 1, perturbation=[1.])
    assert net.data[learner.edge_indices][0] > 4
    assert abs(net.data[learner.edge_indices][1]) < 2
    assert np.array_equal(np.sign(net.data[learner.edge_indices]), [1, -1])
    assert (learner.eligibility > 0).any() and (learner.eligibility < 0).any()


def test_perturbation_without_exploration_cannot_credit_tonic_activity():
    net = network()
    original = net.data.copy()
    learner = MotorPlasticity(net, [2], rule='perturbation')
    for _ in range(100):
        learner.update([0, 1, 2], 5, 1)
    np.testing.assert_array_equal(net.data, original)


@pytest.mark.parametrize('tau_ms', [50., 100.])
def test_delayed_action_reward_changes_relevant_motor_not_unrelated_motor(tau_ms):
    # Fixed delayed-reward bandit: only motor 2's exploration predicts reward.
    # Motor 3 sees identical presynaptic activity but unrelated exploration.
    net = LIFNetwork(sparse.csr_matrix(np.array([
        [0., 0., 4., 4.], [0., 0., -2., -2.],
        [0., 0., 0., 0.], [0., 0., 0., 0.],
    ])))
    learner = MotorPlasticity(net, [2, 3], rule='perturbation', learning_rate=.2,
                             tau_eligibility_ms=tau_ms)
    rng = np.random.default_rng(13)
    noise = rng.choice([-1., 1.], (2000, 2))
    for k, action in enumerate(noise):
        reward = noise[k - 5, 0] if k >= 5 else 0
        learner.update([0, 1], 5, reward, perturbation=action)
    learned = net.data[learner.edge_indices] / learner.initial_weights
    targets = net.indices[learner.edge_indices]
    relevant = np.abs(learned[targets == 2] - 1).mean()
    irrelevant = np.abs(learned[targets == 3] - 1).mean()
    assert relevant > .05 and irrelevant < .04
    assert learned[0] > 1 and learned[2] < 1


def test_signed_eligibility_checkpoint_transfer_and_rule_mismatch(tmp_path):
    a, b = network(), network()
    first = MotorPlasticity(a, [2], rule='perturbation')
    first.update([0, 1], 100, 1, perturbation=[1.])
    path = tmp_path / 'perturbed.npz'
    first.save(path)
    second = MotorPlasticity(b, [2], rule='perturbation')
    second.load(path, restore_traces=True)
    assert np.any(second.eligibility < 0)
    first.update([0], 5, -.5, perturbation=[-.2])
    second.update([0], 5, -.5, perturbation=[-.2])
    np.testing.assert_array_equal(a.data, b.data)
    with pytest.raises(ValueError, match='mismatch'):
        MotorPlasticity(network(), [2]).load(path)


@pytest.mark.parametrize('perturbation', [[1., 0.], [np.nan], [1.01]])
def test_bad_perturbation_does_not_mutate(perturbation):
    learner = MotorPlasticity(network(), [2], rule='perturbation')
    original = learner.net.data.copy()
    with pytest.raises(ValueError):
        learner.update([0], 5, 1, perturbation=perturbation)
    np.testing.assert_array_equal(learner.net.data, original)
    assert not learner.activity.any() and learner.reward_baseline == 0

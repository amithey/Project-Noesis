"""Bias correction must remove birth-score drift without breaking saved weights."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from noesis import plasticity
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity


def network():
    return LIFNetwork(sparse.csr_matrix(np.array([[0., 3., 4.], [0., 0., -2.], [1., 0., 0.]])))


def learner(*, corrected=True):
    assert Path(plasticity.__file__).is_relative_to(Path(__file__).resolve().parents[1] / 'src')
    return MotorPlasticity(network(), [2], learning_rate=.2, tau_eligibility_ms=100,
                           rule='perturbation', normalize_reward_error=True,
                           bias_correct_reward_error=corrected)


def test_constant_reward_does_not_credit_random_motor_innovations():
    a = learner()
    initial = a.net.data.copy()
    for i in range(200):
        result = a.update([0, 1], 5, .6, perturbation=[(-1)**i*.5])
        assert result['prediction_error'] == result['learning_signal'] == 0
        assert np.array_equal(a.net.data, initial)
    assert a.eligibility.any() and a.reward_normalization_mass > 0


def test_common_reward_offset_does_not_change_action_credit():
    a, b = learner(), learner()
    for i, reward in enumerate([-.1, .05, .2, -.2, .1]*12):
        p = [(-1)**i*.4]
        x = a.update([0, 1], 5, reward, perturbation=p)
        y = b.update([0, 1], 5, reward+.5, perturbation=p)
        assert np.isclose(x['prediction_error'], y['prediction_error'], atol=1e-14, rtol=0)
        assert np.isclose(x['learning_signal'], y['learning_signal'], atol=1e-12, rtol=0)
        assert np.allclose(a.net.data, b.net.data, atol=1e-14, rtol=0)


def test_variable_duration_moments_match_explicit_weighted_history():
    a = learner()
    rewards, errors, durations = [], [], []
    for dt, reward in [(5, .5), (7, .4), (11, .6), (3, -.2), (19, .3)]:
        raw = reward-a.reward_baseline if rewards else 0.
        rewards.append(reward)
        errors.append(raw)
        durations.append(dt)
        result = a.update([0, 1], dt, reward, perturbation=[.4])
        weights = np.array([(1-np.exp(-d/2000))*np.exp(-sum(durations[j+1:])/2000)
                            for j, d in enumerate(durations)])
        assert np.isclose(a.reward_normalization_mass, weights.sum(), atol=1e-14)
        assert np.isclose(a.reward_baseline, np.average(rewards, weights=weights), atol=1e-14)
        assert np.isclose(a.reward_variance, np.average(np.square(errors), weights=weights), atol=1e-14)
        assert np.isclose(result['learning_signal'], np.clip(raw/max(.01, np.sqrt(a.reward_variance)), -1, 1))


def test_new_checkpoint_restores_mass_and_exact_learning_continuation(tmp_path):
    a, b = learner(), learner()
    for reward in [.4, .5, -.2, .7]:
        a.update([0, 1], 5, reward, perturbation=[.4])
    path = tmp_path / 'new.npz'
    a.save(path)
    with np.load(path, allow_pickle=False) as z:
        assert int(z['version']) == 4 and bool(z['bias_correct_reward_error'])
    b.load(path, restore_traces=True)
    assert a.reward_normalization_mass == b.reward_normalization_mass
    for reward in [.1, -.8, .2]:
        assert a.update([0, 1], 5, reward, perturbation=[-.2]) == b.update([0, 1], 5, reward, perturbation=[-.2])
        assert np.array_equal(a.net.data, b.net.data)
        assert a.reward_normalization_mass == b.reward_normalization_mass


def test_default_load_keeps_weights_and_resets_all_birth_statistics(tmp_path):
    a, b = learner(), learner()
    for reward in [.3, -.2, .8]:
        a.update([0, 1], 5, reward, perturbation=[.5])
    path = tmp_path / 'new.npz'
    a.save(path)
    b.load(path)
    assert np.array_equal(a.net.data, b.net.data)
    assert b.reward_baseline == b.reward_variance == b.reward_normalization_mass == 0
    assert not b.activity.any() and not b.eligibility.any()


@pytest.mark.parametrize('source_corrected', [False, True])
def test_cross_normalization_requires_explicit_weight_only_import(tmp_path, source_corrected):
    a = learner(corrected=source_corrected)
    b = learner(corrected=not source_corrected)
    for reward in [.3, -.2, .8]:
        a.update([0, 1], 5, reward, perturbation=[.5])
    path = tmp_path / 'bank.npz'
    a.save(path)
    before = b.net.data.copy()
    with pytest.raises(ValueError):
        b.load(path)
    assert np.array_equal(b.net.data, before)
    b.load(path, weights_only=True)
    assert np.array_equal(b.net.data, a.net.data)
    assert b.reward_normalization_mass == b.reward_baseline == b.reward_variance == 0


@pytest.mark.parametrize('bad_mass', [np.nan, np.inf, -.01, 1.01, np.array([.2]), True])
def test_malformed_mass_rejected_before_mutation(tmp_path, bad_mass):
    a = learner()
    path = tmp_path / 'bank.npz'
    a.save(path)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data['reward_normalization_mass'] = np.array(bad_mass)
    np.savez_compressed(path, **data)
    before = a.net.data.copy()
    with pytest.raises(ValueError):
        a.load(path, restore_traces=True)
    assert np.array_equal(a.net.data, before) and a.reward_normalization_mass == 0


@pytest.mark.parametrize('normalized', [False, True])
def test_legacy_default_matches_archived_updates_and_checkpoint_bytes(tmp_path, normalized):
    old_path = Path(__file__).with_name('plasticity_v3_reference.py')
    spec = importlib.util.spec_from_file_location('archived_plasticity', old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    opts = {'learning_rate': .2, 'tau_eligibility_ms': 100, 'rule': 'perturbation', 'normalize_reward_error': normalized}
    a = old.MotorPlasticity(network(), [2], **opts)
    b = MotorPlasticity(network(), [2], **opts)
    for i, reward in enumerate([.6, -.2, .8, 0., -1.]):
        kwargs = {'perturbation': [(-1)**i*.3]}
        assert a.update([0, 1, 2], 5+i, reward, **kwargs) == b.update([0, 1, 2], 5+i, reward, **kwargs)
        assert np.array_equal(a.net.data, b.net.data)
    x, y = tmp_path / 'old.npz', tmp_path / 'new.npz'
    a.save(x)
    b.save(y)
    assert x.read_bytes() == y.read_bytes()


def test_bias_correction_requires_normalization():
    with pytest.raises(ValueError):
        MotorPlasticity(network(), [2], rule='perturbation', bias_correct_reward_error=True)


def test_extreme_inputs_and_alternating_rewards_keep_original_graph_bounds():
    a = MotorPlasticity(network(), [2], learning_rate=1e300, tau_eligibility_ms=100,
                        rule='perturbation', normalize_reward_error=True, bias_correct_reward_error=True)
    indices, indptr, original = a.net.indices.copy(), a.net.indptr.copy(), a.net.data.copy()
    for dt, reward in [(1e-200, .4), (1e300, 1), (5, -1), (5, .8), (5, -.8)]:
        a.update([0, 1], dt, reward, perturbation=[.8])
    assert np.isfinite(a.net.data).all()
    assert np.array_equal(a.net.indices, indices) and np.array_equal(a.net.indptr, indptr)
    assert np.array_equal(np.sign(a.net.data), np.sign(original))
    assert np.all(np.abs(a.net.data-original) <= .5*np.abs(original))
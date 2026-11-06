"""CL1 preregistered tiny closed-loop LIF readout review; never a fly/gait test."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse

root = Path(__file__).resolve().parents[1]
core = root / 'work/Noesis-Core/src'
sys.path.insert(0, str(core))
from noesis.lif import LIFNetwork, LIFParams, _seed
from noesis.plasticity import MotorPlasticity

CORE_PATHS = [core / 'noesis/lif.py', core / 'noesis/plasticity.py']
hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in CORE_PATHS]
started = time.monotonic()


def network(seed, weights=None):
    w = sparse.csr_matrix(([25., -10.], ([0, 1], [2, 2])), shape=(3, 3))
    net = LIFNetwork(w, LIFParams(dt=.5, da=1.), seed=seed)
    learner = MotorPlasticity(net, [2], learning_rate=.2, tau_eligibility_ms=100, max_change=.5,
                              rule='perturbation', normalize_reward_error=True, bias_correct_reward_error=True)
    if weights is not None:
        net.data[:] = weights
    return net, learner


def advance(net, rng, current=None):
    _seed(int(rng.integers(0, 2**31)))
    extra = {} if current is None else dict(current_idx=np.array([2]), current_mv_per_ms=np.array([current]))
    _, spikes = net.run(5., drive_idx=np.array([0, 1]), drive_rate_hz=np.array([40., 20.]),
                        drive_weight=400., background_hz=3., background_weight=40., **extra)
    return spikes


def train(seed, mode, yoked=None):
    net, learner = network(seed)
    neural_rng = np.random.default_rng(seed)
    exploration_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x434C31]))
    state = rate = 0.
    decay, alpha = np.exp(-5./100), -np.expm1(-5./100)
    rewards, applied, firing = [], [], []
    for step in range(1000):
        draw = float(exploration_rng.normal()) if step >= 10 else 0.
        if step >= 10:
            state = decay*state + np.sqrt(1-decay**2)*draw
        spikes = advance(net, neural_rng, 1.5*np.tanh(state) if step >= 10 else None)
        rate += alpha*(np.count_nonzero(spikes == 2)*200. - rate)
        reward = float(np.tanh(rate/100.))
        if step >= 10:
            use = float(yoked[step-10]) if mode == 'yoked' else reward
            if mode != 'off':
                learner.update(spikes, 5., use, perturbation=np.array([np.tanh(draw)]))
            applied.append(use)
        rewards.append(reward)
        firing.append(rate)
    ratio = net.data / np.array([25., -10.])
    assert np.isfinite(ratio).all() and ((ratio >= .5) & (ratio <= 1.5)).all()
    assert np.array_equal(np.sign(net.data), [1., -1.])
    if mode == 'off':
        np.testing.assert_array_equal(net.data, np.array([25., -10.]))
    return net.data.copy(), np.array(rewards), np.array(applied), np.array(firing)


def probe(weights, seed):
    net, _ = network(seed, weights)
    rng = np.random.default_rng(seed)
    rate = 0.
    rewards = []
    alpha = -np.expm1(-5./100)
    for _ in range(400):
        spikes = advance(net, rng)
        rate += alpha*(np.count_nonzero(spikes == 2)*200. - rate)
        rewards.append(float(np.tanh(rate/100.)))
    return float(np.mean(rewards[20:]))


seeds, probes = list(range(50000, 50016)), list(range(60000, 60004))
baseline = np.array([probe(np.array([25., -10.]), seed) for seed in probes])
rows, evidence = [], {}
for seed in seeds:
    positive = train(seed, 'online')
    control_reward = np.roll(positive[1][10:], 300)
    yoked = train(seed, 'yoked', control_reward)
    off = train(seed, 'off')
    row = dict(seed=seed, weights={}, probe_returns={}, mean_frozen_delta={}, weight_direction={})
    for mode, trial in [('online', positive), ('yoked', yoked), ('off', off)]:
        weights, reward, applied, firing = trial
        returns = np.array([probe(weights, p) for p in probes])
        row['weights'][mode] = weights.tolist()
        row['probe_returns'][mode] = returns.tolist()
        row['mean_frozen_delta'][mode] = float(np.mean(returns-baseline))
        relative = np.abs(weights)/[25., 10.] - 1.
        row['weight_direction'][mode] = float(np.mean(relative*np.array([1., -1.])))
        for key, values in [('weights', weights), ('actual_reward', reward), ('applied_reward', applied), ('firing_ema_hz', firing)]:
            evidence[f'{seed}_{mode}_{key}'] = values
    np.testing.assert_array_equal(row['probe_returns']['off'], baseline)
    np.testing.assert_array_equal(np.sort(positive[2]), np.sort(yoked[2]))
    row['online_minus_yoked'] = row['mean_frozen_delta']['online'] - row['mean_frozen_delta']['yoked']
    rows.append(row)


def stats(values):
    values = np.asarray(values)
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    return dict(mean=float(values.mean()), standard_error=se,
                z=float(values.mean()/se) if se else None,
                positive_births=int((values > 0).sum()), independent_births=len(values))


a = stats([r['mean_frozen_delta']['online'] for r in rows])
b = stats([r['online_minus_yoked'] for r in rows])
passed = bool(a['mean'] > 0 and a['z'] is not None and a['z'] >= 3 and
              b['mean'] > 0 and b['z'] is not None and b['z'] >= 3)
assert hashes == [hashlib.sha256(p.read_bytes()).hexdigest() for p in CORE_PATHS]
out = root / 'outputs/closed-loop-readout-cl1-20261106'
out.mkdir(exist_ok=False)
np.savez_compressed(out / 'numerical-evidence.npz', baseline=baseline, **evidence)
result = dict(status='completed_synthetic_closed_loop_CL1', seconds=time.monotonic()-started,
              core_commit='42c20769376bda5694b4c1c66ca75e52cbd9f1f4', source_sha256=hashes,
              train_seeds=seeds, frozen_synthetic_probe_seeds=probes, baseline_probe_returns=baseline.tolist(),
              online_minus_parent=a, online_minus_yoked=b, registered_screen_pass=passed,
              all_original_bounds_and_signs=True, all_learning_off_exact=True, yoked_reward_marginals_exact=True,
              rows=rows, physical_simulation=False, checkpoint_written=False, walking_proven=False,
              limits=['Tiny arbitrary signed3-neuron graph and monotone firing objective, no fly, muscles or physics.',
                      'Four reused synthetic probe streams averaged within each independent training birth; not64independent observations.',
                      'Yoked reward has exactly matching marginal but shifted timing; it is not an independent reward generator.',
                      'A failed screen is not a proof that learning is absent; no tuning or repeat is authorized by this result.'])
(out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({key:result[key] for key in ('status','seconds','online_minus_parent','online_minus_yoked','registered_screen_pass')}))

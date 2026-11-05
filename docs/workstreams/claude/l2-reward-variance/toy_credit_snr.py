"""Can the actual MotorPlasticity perturbation rule acquire a *known* effect, given realistic reward noise?

A synthetic positive control (no fly, no physics): random sparse network, Poisson presynaptic spikes at 20 Hz, motor
exploration innovations built like CircuitExploration (OU state tau 100 ms, innovation = tanh(white noise)), and a reward
that depends causally on the perturbation of a few "good" motor neurons plus fast + slow noise sized from the real
training rewards (var ~0.17, ~77% of it faster than 5 Hz). The learner is the core class, with the pilot's settings
(rule perturbation, normalized + bias-corrected error, eligibility tau 100 ms, max_change 0.5).

Question: for which reward-signal strength, credit lag and learning rate does a 990-step (5 s) birth move the weights
consistently toward the true effect across independent seeds? Real-fly conclusions need real-fly measurements; this
only tests the rule's signal-to-noise under stated assumptions.

usage: toy_credit_snr.py <core src dir> [quick]
"""
import sys
from types import SimpleNamespace

import numpy as np
from scipy import sparse

sys.path.insert(0, sys.argv[1])
from noesis.plasticity import MotorPlasticity  # noqa: E402

QUICK = len(sys.argv) > 2
N, N_MOTOR, GOOD, IN_DEG = 300, 30, 6, 40
DT = 5.0
STEPS = 990
SEEDS = 8 if QUICK else 12


def make_net(rng):
    rows, cols, vals = [], [], []
    for post in range(N_MOTOR):  # every motor neuron gets IN_DEG inputs from the rest of the network
        pre = rng.choice(np.arange(N_MOTOR, N), IN_DEG, replace=False)
        for p in pre:
            rows.append(p)
            cols.append(post)
            vals.append(rng.choice([-1., 1.], p=[.3, .7]) * rng.uniform(.5, 2.))
    # sparse background so the CSR has other rows too
    for _ in range(600):
        a, b = rng.integers(0, N, 2)
        rows.append(a)
        cols.append(b)
        vals.append(rng.uniform(-1, 1))
    m = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64)
    m.sum_duplicates()
    return SimpleNamespace(n=N, data=m.data.copy(), indptr=m.indptr.copy(), indices=m.indices.copy())


def one_birth(seed, kappa, lag_ms, lr, steps=STEPS, base=0.3, tau_elig=100.0, sig_fast=0.30, sig_slow=0.20):
    rng = np.random.default_rng(seed)
    net = make_net(np.random.default_rng(1234))  # same network for every birth: only the noise differs
    motor = np.arange(N_MOTOR)
    learner = MotorPlasticity(net, motor, learning_rate=lr, tau_eligibility_ms=tau_elig, max_change=.5,
                              rule="perturbation", normalize_reward_error=True, bias_correct_reward_error=True)
    decay = np.exp(-DT / 100.)
    state = np.zeros(N_MOTOR)
    lag = int(round(lag_ms / DT))
    history = np.zeros((steps + lag + 1, GOOD))
    slow = 0.
    slow_decay = np.exp(-DT / 2000.)
    for t in range(steps):
        noise = rng.normal(size=N_MOTOR)
        innovation = np.tanh(noise)
        state = decay * state + np.sqrt(1 - decay ** 2) * noise
        x = np.tanh(state)  # normalized exploration current direction actually injected
        history[t + lag] = x[:GOOD]
        slow = slow_decay * slow + np.sqrt(1 - slow_decay ** 2) * rng.normal() * sig_slow
        causal = kappa * history[t][:GOOD].mean()
        reward = float(np.clip(base + causal + slow + sig_fast * rng.normal(), -1, 1))
        spikes = np.flatnonzero(rng.random(N) < 0.1)  # 20 Hz at 5 ms
        learner.update(spikes, DT, reward, perturbation=innovation)
    # signed change toward net excitation: excitatory magnitude up, inhibitory magnitude down
    rel = (np.abs(net.data[learner.edge_indices]) / np.abs(learner.initial_weights) - 1) * np.sign(learner.initial_weights)
    post = net.indices[learner.edge_indices]
    return float(rel[post < GOOD].mean()), float(rel[post >= GOOD].mean()), rel


def sweep():
    print(f"{'kappa':>6s} {'lag ms':>6s} {'lr/s':>5s} | {'good-edge change':>17s} {'control change':>15s} {'z (good vs ctrl)':>17s} {'births moving the right way':>28s} {'cross-seed cos (mean)':>22s}")
    results = {}
    for lr in (0.02, 0.2, 2.0):
        for lag in (0, 50, 150):
            for kappa in (0.0, 0.05, 0.2, 0.5):
                good, ctrl, rels = [], [], []
                for s in range(SEEDS):
                    g, c, rel = one_birth(s, kappa, lag, lr)
                    good.append(g)
                    ctrl.append(c)
                    rels.append(rel)
                good, ctrl = np.array(good), np.array(ctrl)
                diff = good - ctrl
                z = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)) + 1e-12)
                R = np.array(rels)
                cos = []
                for i in range(len(R)):
                    for j in range(i + 1, len(R)):
                        cos.append(float(R[i] @ R[j] / (np.linalg.norm(R[i]) * np.linalg.norm(R[j]))))
                results[(lr, lag, kappa)] = (diff.mean(), z, float((diff > 0).mean()), float(np.mean(cos)))
                print(f"{kappa:6.2f} {lag:6d} {lr:5.2f} | {good.mean():+17.5f} {ctrl.mean():+15.5f} {z:17.2f} "
                      f"{(diff > 0).mean():28.0%} {np.mean(cos):22.3f}", flush=True)
    return results


if __name__ == "__main__":
    sweep()

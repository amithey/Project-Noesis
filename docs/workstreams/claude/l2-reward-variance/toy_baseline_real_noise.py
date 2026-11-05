"""Does a shorter reward baseline raise the rule's signal-to-noise when the noise is the REAL reward? Synthetic positive control.

Same toy as toy_credit_snr.py (real MotorPlasticity, planted causal effect of 6 motor neurons) but the reward noise is a
real training reward series (v8, 8 births from the L1 runs) instead of synthetic noise. The learner always subtracts its
own 2 s baseline; a shorter baseline is emulated by first high-passing the reward, r_in = r - baseline_tau(r) with a causal
exponential baseline (the learner's 2 s baseline then sees an already zero-mean signal). Variants: native (no pre-filter),
500, 200, 100, 50 ms, and previous-step.

usage: toy_baseline_real_noise.py <core src> <L1 run dir containing m1..m8> [seeds]
"""
import sys
from pathlib import Path

import numpy as np

CORE, RUN = sys.argv[1], Path(sys.argv[2])
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
sys.argv = [sys.argv[0], CORE, "quick"]
import toy_credit_snr as toy  # noqa: E402
from noesis.plasticity import MotorPlasticity  # noqa: E402

DT = toy.DT
NOISE = []
for i in range(8):
    z = np.load(RUN / f"m{i + 1}-seed{1000 + i}" / "training-observations.npz")
    NOISE.append(z["reward"][10:].astype(float))


def birth(seed, kappa, lag_ms, prefilter, lr=0.2, steps=990):
    rng = np.random.default_rng(seed)
    net = toy.make_net(np.random.default_rng(1234))
    motor = np.arange(toy.N_MOTOR)
    learner = MotorPlasticity(net, motor, learning_rate=lr, tau_eligibility_ms=100.0, max_change=.5, rule="perturbation",
                              normalize_reward_error=True, bias_correct_reward_error=True)
    decay = np.exp(-DT / 100.)
    state = np.zeros(toy.N_MOTOR)
    lag = int(round(lag_ms / DT))
    history = np.zeros((steps + lag + 1, toy.GOOD))
    noise = NOISE[seed % 8][:steps]
    noise = noise - noise.mean()
    baseline, previous = None, None
    for t in range(steps):
        draw = rng.normal(size=toy.N_MOTOR)
        innovation = np.tanh(draw)
        state = decay * state + np.sqrt(1 - decay ** 2) * draw
        history[t + lag] = np.tanh(state)[:toy.GOOD]
        reward = float(np.clip(0.3 + noise[t] + kappa * history[t].mean(), -1, 1))
        if prefilter == "native":
            r_in = reward
        elif prefilter == "previous":
            r_in = reward - (reward if previous is None else previous)
            previous = reward
        else:
            tau = float(prefilter)
            baseline = reward if baseline is None else baseline
            r_in = reward - baseline
            baseline += (1 - np.exp(-DT / tau)) * (reward - baseline)
        spikes = np.flatnonzero(rng.random(toy.N) < 0.1)
        learner.update(spikes, DT, float(np.clip(r_in, -1, 1)), perturbation=innovation)
    rel = (np.abs(net.data[learner.edge_indices]) / np.abs(learner.initial_weights) - 1) * np.sign(learner.initial_weights)
    post = net.indices[learner.edge_indices]
    return float(rel[post < toy.GOOD].mean() - rel[post >= toy.GOOD].mean())


print(f"{SEEDS} seeds, real v8 reward noise (variance of the noise {np.mean([n.var() for n in NOISE]):.3f}); z of good-vs-control edge change")
print(f"{'kappa':>6s} {'lag ms':>6s} | " + " ".join(f"{v:>9s}" for v in ("native", "500", "200", "100", "50", "previous")))
for kappa in (0.1, 0.2):
    for lag in (0, 50, 150):
        cells = []
        for variant in ("native", "500", "200", "100", "50", "previous"):
            d = np.array([birth(s, kappa, lag, variant) for s in range(SEEDS)])
            cells.append(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))
        print(f"{kappa:6.2f} {lag:6d} | " + " ".join(f"{c:9.2f}" for c in cells), flush=True)

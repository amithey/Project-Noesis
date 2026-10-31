"""Stage A of proposal.md. usage: toy_credit_variants.py <core src> <L1 run dir containing m1..m8> [seeds]"""
import sys
from pathlib import Path

import numpy as np

CORE, RUN = sys.argv[1], Path(sys.argv[2])
SEEDS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
sys.path.insert(0, "docs/workstreams/claude/l2-reward-variance")
sys.argv = [sys.argv[0], CORE, "quick"]
import toy_credit_snr as toy  # noqa: E402
from noesis.plasticity import MotorPlasticity  # noqa: E402

DT = toy.DT
NOISE = [np.load(RUN / f"m{i + 1}-seed{1000 + i}" / "training-observations.npz")["reward"][10:].astype(float) for i in range(8)]


def birth(seed, kappa, lag_ms, credit, lr=0.2, steps=990):
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
    for t in range(steps):
        draw = rng.normal(size=toy.N_MOTOR)
        state = decay * state + np.sqrt(1 - decay ** 2) * draw
        history[t + lag] = np.tanh(state)[:toy.GOOD]
        reward = float(np.clip(0.3 + noise[t] + kappa * history[t].mean(), -1, 1))
        spikes = np.flatnonzero(rng.random(toy.N) < 0.1)
        perturbation = np.tanh(draw) if credit == "innovation" else np.tanh(state)
        learner.update(spikes, DT, reward, perturbation=perturbation)
    rel = (np.abs(net.data[learner.edge_indices]) / np.abs(learner.initial_weights) - 1) * np.sign(learner.initial_weights)
    post = net.indices[learner.edge_indices]
    return float(rel[post < toy.GOOD].mean() - rel[post >= toy.GOOD].mean())


print(f"{SEEDS} seeds; z of good-vs-control edge change; C1 = credit tanh(state) vs current innovation")
print(f"{'kappa':>6s} {'lag ms':>6s} | {'innovation':>10s} {'state(C1)':>10s} {'ratio':>6s}")
for kappa in (0.1, 0.2):
    for lag in (0, 50, 150):
        z = {}
        for credit in ("innovation", "state"):
            d = np.array([birth(s, kappa, lag, credit) for s in range(SEEDS)])
            z[credit] = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
        print(f"{kappa:6.2f} {lag:6d} | {z['innovation']:10.2f} {z['state']:10.2f} {z['state'] / z['innovation'] if z['innovation'] else float('nan'):6.2f}", flush=True)

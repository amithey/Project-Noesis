"""Positive/negative synthetic control for the REAL ``MotorPlasticity`` (perturbation rule, pilot settings).

A synthetic mechanism test only: no fly, no physics. A planted relation makes the reward depend on the exploration state of a
few "good" motor neurons (``kappa`` > 0 positive control, ``kappa`` = 0 matched negative control with the same noise, spikes and
perturbation draws). The learner is then run (or not run: learning-off) and the statistic is how much more the incoming edges of
the good targets moved toward the planted effect than the other targets' edges. Success of the positive control shows that the
rule CAN pick up a relation of this size under these assumptions; it does not identify the cause of any physical failure and is
never evidence about the fly.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

N, N_MOTOR, GOOD, IN_DEG, DT = 300, 30, 6, 40, 5.0
STEPS = 990
SPIKE_P = 0.1  # Bernoulli per 5 ms step per neuron (~20 Hz)


def make_net(seed=1234):
    """Random sparse network whose first N_MOTOR neurons are motor targets; returns a minimal object for MotorPlasticity."""
    from noesis.lif import LIFNetwork

    rng = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    for post in range(N_MOTOR):
        for pre in rng.choice(np.arange(N_MOTOR, N), IN_DEG, replace=False):
            rows.append(pre)
            cols.append(post)
            vals.append(rng.choice([-1., 1.], p=[.3, .7]) * rng.uniform(.5, 2.))
    for _ in range(600):
        a, b = rng.integers(0, N, 2)
        rows.append(a)
        cols.append(b)
        vals.append(rng.uniform(-1, 1))
    return LIFNetwork(sparse.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=np.float64))


def birth(seed, kappa, learning=True, lr=0.2, steps=STEPS, noise_sd=0.41, lag_steps=0):
    """One birth. Returns the statistic D (good-target minus other-target mean signed relative edge change) and diagnostics.

    ``learning=False`` runs the identical trial (same draws, same rewards) without ever calling ``update``.
    """
    from noesis.plasticity import MotorPlasticity

    rng = np.random.default_rng(seed)
    net = make_net()
    motor = np.arange(N_MOTOR)
    learner = MotorPlasticity(net, motor, learning_rate=lr, tau_eligibility_ms=100.0, max_change=.5, rule="perturbation",
                              normalize_reward_error=True, bias_correct_reward_error=True)
    start = net.data.copy()
    decay = np.exp(-DT / 100.)
    state, history = np.zeros(N_MOTOR), np.zeros((steps + lag_steps + 1, GOOD))
    noise = noise_sd * rng.normal(size=steps)  # same stream for every kappa/learning setting of this seed
    spikes_rng = np.random.default_rng(seed + 7_000_003)
    updates = 0
    for t in range(steps):
        draw = rng.normal(size=N_MOTOR)
        state = decay * state + np.sqrt(1 - decay ** 2) * draw
        history[t + lag_steps] = np.tanh(state)[:GOOD]
        reward = float(np.clip(0.3 + noise[t] + kappa * history[t].mean(), -1, 1))
        spikes = np.flatnonzero(spikes_rng.random(N) < SPIKE_P)
        if learning:
            learner.update(spikes, DT, reward, perturbation=np.tanh(draw))
            updates += 1
    edges = learner.edge_indices
    rel = (np.abs(net.data[edges]) / np.abs(learner.initial_weights) - 1) * np.sign(learner.initial_weights)
    post = net.indices[edges]
    return {"D": float(rel[post < GOOD].mean() - rel[post >= GOOD].mean()), "updates": updates,
            "weights_equal_start": bool(np.array_equal(net.data, start)),
            "ratio_min": float((np.abs(net.data[edges]) / np.abs(learner.initial_weights)).min()),
            "ratio_max": float((np.abs(net.data[edges]) / np.abs(learner.initial_weights)).max()),
            "signs_kept": bool(np.array_equal(np.sign(net.data[edges]), np.sign(learner.initial_weights)))}


def z_score(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean() / (values.std(ddof=1) / np.sqrt(len(values)) + 1e-12))


def run_control(seeds, kappa=0.2, lr=0.2, steps=STEPS):
    """Positive, matched negative and learning-off arms over the same seeds; plus the registered verdict."""
    arms = {"positive": [birth(s, kappa, True, lr, steps) for s in seeds],
            "negative": [birth(s, 0.0, True, lr, steps) for s in seeds],
            "learning_off_positive": [birth(s, kappa, False, lr, steps) for s in seeds]}
    d = {name: [row["D"] for row in rows] for name, rows in arms.items()}
    summary = {name: {"mean_D": float(np.mean(v)), "z": z_score(v) if np.std(v) > 0 else 0.0} for name, v in d.items()}
    off_exact = all(r["weights_equal_start"] and r["updates"] == 0 and r["D"] == 0.0 for r in arms["learning_off_positive"])
    bounds = all(0.5 - 1e-9 <= r["ratio_min"] and r["ratio_max"] <= 1.5 + 1e-9 and r["signs_kept"]
                 for name in ("positive", "negative") for r in arms[name])
    verdict = {"positive_z_ge_3": summary["positive"]["z"] >= 3.0 and summary["positive"]["mean_D"] > 0,
               "negative_abs_z_lt_3": abs(summary["negative"]["z"]) < 3.0,
               "negative_mean_lt_third_of_positive": abs(summary["negative"]["mean_D"]) < summary["positive"]["mean_D"] / 3.0,
               "learning_off_exact": off_exact, "bounds_and_signs": bounds}
    verdict["registered_pass"] = all(verdict.values())
    return {"seeds": list(seeds), "kappa": kappa, "lr": lr, "steps": steps, "summary": summary, "verdict": verdict, "rows": arms}

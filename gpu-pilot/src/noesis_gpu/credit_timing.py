"""Offline credit-timing analysis: does the exploration perturbation predict the body's response and the reward, and at which lag?

Reads the ``exploration-log.npz`` written by ``profile_run --log-exploration`` and the run's
``training-observations.npz``. For each leg it asks how well that leg's exploration current (a few principal
components of the injected current) explains (a) the leg's own joint velocity and (b) the learner's actual normalized error signal,
at lags of 0..max_lag control steps (5 ms each). Significance comes from circularly shifting the response
inside each birth (keeps its autocorrelation), taking the maximum R^2 over lags and legs for every shift
(family-wise). Descriptive and read-only: it never touches weights, learning or checkpoints.
"""

from __future__ import annotations

import numpy as np

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
DT_MS = 5.0
NEUROMERE = {"T1_PRONM": 0, "T2_MESONM": 1, "T3_METANM": 2}  # front, middle, hind
JOINTS_PER_LEG = 7


def assign_legs(neurons, neuron_ids):
    """Leg index (0..5, order LF LM LH RF RM RH) per neuron id, -1 when the neuromere or side is ambiguous."""
    side = neurons["Soma side"].to_numpy()
    region = neurons["Top in/out region"].to_numpy()
    out = np.full(len(neuron_ids), -1, dtype=int)
    for k, n in enumerate(neuron_ids):
        position = NEUROMERE.get(str(region[n]))
        if position is None or side[n] not in ("left", "right"):
            continue
        out[k] = (0 if side[n] == "left" else 3) + position
    return out


def learning_signal(reward):
    """The learner's ACTUAL normalized error signal for a reward sequence, from the core's own code.

    Replays ``MotorPlasticity.update`` (perturbation rule, normalized and bias-corrected error, as in the pilot)
    on a one-edge network with zero perturbation, so ``learning_signal`` is exactly what the learner used (reward
    minus its bias-corrected 2 s baseline, divided by the 2 s RMS, clipped to [-1, 1]); the first value is 0.
    Not interchangeable with ``ema2s_reward_residual``.
    """
    from types import SimpleNamespace

    from noesis.plasticity import MotorPlasticity

    net = SimpleNamespace(n=2, data=np.array([1.0]), indptr=np.array([0, 1, 1]), indices=np.array([1]))
    learner = MotorPlasticity(net, [1], learning_rate=0.2, tau_eligibility_ms=100, max_change=0.5, rule="perturbation",
                              normalize_reward_error=True, bias_correct_reward_error=True)
    empty, zero = np.empty(0, dtype=np.int64), np.zeros(1)
    return np.array([learner.update(empty, DT_MS, float(r), perturbation=zero)["learning_signal"] for r in reward])


def ema2s_reward_residual(reward, tau_ms=2000.0):
    """Reward minus a plain 2 s exponential baseline (no normalization, no bias correction): an approximation only."""
    a = 1 - np.exp(-DT_MS / tau_ms)
    out, baseline = np.empty(len(reward)), float(reward[0])
    for i, value in enumerate(reward):
        out[i] = value - baseline
        baseline += a * (value - baseline)
    return out


def leg_features(drive, legs, leg, components=4):
    """Top principal components of one leg's injected current (time x components), plus the PCA basis."""
    columns = np.flatnonzero(legs == leg)
    block = drive[:, columns] - drive[:, columns].mean(axis=0)
    if block.shape[1] == 0:
        raise ValueError("leg has no explored neurons")
    u, s, _ = np.linalg.svd(block, full_matrices=False)
    k = min(components, len(s))
    return u[:, :k] * s[:k]


def _lagged(x, lag):
    out = np.zeros_like(x)
    if lag:
        out[lag:] = x[:-lag]
    else:
        out[:] = x
    return out


class LagFit:
    """OLS of a response on lagged features for lags 0..max_lag, pooled over births, with fast re-scoring.

    The lagged design matrices and their normal equations are built once; ``r2`` then scores any response
    (for example a circularly shifted one) with a few matrix products. The first ``skip`` rows of every birth are
    dropped after lagging so all lags use the same rows. Each birth is centred separately.
    """

    def __init__(self, features, max_lag, skip):
        self.skip = skip
        columns = features[0].shape[1]
        self.X = np.empty((max_lag + 1, sum(len(x) - skip for x in features), columns))
        for lag in range(max_lag + 1):
            rows = []
            for x in features:
                lagged = _lagged(x, lag)[skip:]
                rows.append(lagged - lagged.mean(axis=0))
            self.X[lag] = np.vstack(rows)
        self.inverse = np.stack([np.linalg.pinv(m.T @ m) for m in self.X])

    def r2(self, response):
        y = np.concatenate([r[self.skip:] - r[self.skip:].mean() for r in response])
        xty = np.einsum("lnk,n->lk", self.X, y)
        explained = np.einsum("lk,lkj,lj->l", xty, self.inverse, xty)
        return explained / max(float(y @ y), 1e-12)


def lag_r2(features, response, max_lag, skip):
    """R^2 of an OLS fit of the response on the lagged features, for lags 0..max_lag, pooled over births."""
    return LagFit(features, max_lag, skip).r2(response)


def circular_shift(y, rng, minimum):
    shift = int(rng.integers(minimum, len(y) - minimum))
    return np.roll(y, shift)


def analyse(features_by_leg, responses_by_leg, max_lag=100, permutations=200, seed=0, min_shift=200):
    """For each leg: R^2 curve, best lag, and the family-wise p over (legs x lags).

    features_by_leg[leg] / responses_by_leg[leg]: lists with one array per birth. The family-wise null is the
    maximum R^2 over all legs and lags for each set of independent circular shifts.
    """
    rng = np.random.default_rng(seed)
    skip = max_lag
    fits = {leg: LagFit(features_by_leg[leg], max_lag, skip) for leg in features_by_leg}
    observed = {leg: fits[leg].r2(responses_by_leg[leg]) for leg in features_by_leg}
    null_max = np.zeros(permutations)
    for p in range(permutations):
        best = 0.0
        for leg in features_by_leg:
            shifted = [circular_shift(y, rng, min_shift) for y in responses_by_leg[leg]]
            best = max(best, float(fits[leg].r2(shifted).max()))
        null_max[p] = best
    result = {}
    for leg, curve in observed.items():
        lag = int(curve.argmax())
        result[leg] = {"best_lag_steps": lag, "best_lag_ms": lag * DT_MS, "r2": float(curve[lag]),
                       "family_p": float((1 + np.sum(null_max >= curve[lag])) / (permutations + 1)),
                       "curve": curve.tolist()}
    return result, {"null_max_p95": float(np.percentile(null_max, 95)), "null_max_p99": float(np.percentile(null_max, 99))}


def load_birth(log_path, observations_path, neurons, components=4):
    """Features per leg and responses for one birth. Row i of the log is the i-th learning-active step."""
    with np.load(log_path) as z:
        drive, drive_idx = z["drive"].astype(float), z["drive_idx"]
    with np.load(observations_path) as o:
        reward, velocity, active = o["reward"].astype(float), o["velocity"].astype(float), o["learning_active"].astype(bool)
    rows = np.flatnonzero(active)
    if len(rows) != len(drive):
        raise ValueError("exploration log rows must equal the number of learning-active observation rows")
    legs = assign_legs(neurons, drive_idx)
    signal = learning_signal(reward[rows])  # exactly the learner's error signal over the learning-active steps
    joint = velocity[rows].reshape(len(rows), 6, JOINTS_PER_LEG)
    features, joint_response = {}, {}
    for leg in range(6):
        features[leg] = leg_features(drive, legs, leg, components)
        block = joint[:, leg, :] - joint[:, leg, :].mean(axis=0)
        u, s, _ = np.linalg.svd(block, full_matrices=False)
        joint_response[leg] = u[:, 0] * s[0]  # first principal component of the leg's joint velocities
    return features, joint_response, signal, legs


def run(log_dirs, neurons, max_lag=100, permutations=200, seed=0, components=4):
    """Analyse several births (each directory holds exploration-log.npz and training-observations.npz)."""
    from pathlib import Path
    births = [load_birth(Path(d) / "exploration-log.npz", Path(d) / "training-observations.npz", neurons, components)
              for d in log_dirs]
    features = {leg: [b[0][leg] for b in births] for leg in range(6)}
    reward = {leg: [b[2] for b in births] for leg in range(6)}
    joints = {leg: [b[1][leg] for b in births] for leg in range(6)}
    reward_result, reward_null = analyse(features, reward, max_lag, permutations, seed)
    joint_result, joint_null = analyse(features, joints, max_lag, permutations, seed + 1)
    return {"legs": LEGS, "births": len(births), "max_lag_ms": max_lag * DT_MS,
            "learning_signal": {"per_leg": reward_result, "null": reward_null},
            "joint_velocity": {"per_leg": joint_result, "null": joint_null}}


def main(argv=None):
    import argparse
    import json
    from pathlib import Path

    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("births", nargs="+", type=Path, help="run directories with exploration-log.npz and training-observations.npz")
    ap.add_argument("--neurons", type=Path, required=True, help="data/processed/neurons.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-lag", type=int, default=100)
    ap.add_argument("--permutations", type=int, default=500)
    ap.add_argument("--components", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    result = run(args.births, pd.read_parquet(args.neurons), args.max_lag, args.permutations, args.seed, args.components)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for kind in ("joint_velocity", "learning_signal"):
        for leg, entry in result[kind]["per_leg"].items():
            print(f"{kind:16s} {LEGS[leg]}: best lag {entry['best_lag_ms']:5.0f} ms  R2 {entry['r2']:.4f}  family p {entry['family_p']:.4f}")
        print(f"{kind:16s} null max R2: p95 {result[kind]['null']['null_max_p95']:.4f}  p99 {result[kind]['null']['null_max_p99']:.4f}")


if __name__ == "__main__":
    main()

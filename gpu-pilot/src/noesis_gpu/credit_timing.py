"""Offline credit-timing analysis: does the exploration perturbation predict the body's response and the learner's error signal, and at which lag?

Reads the ``exploration-log.npz`` written by ``profile_run --log-exploration`` and the run's
``training-observations.npz``. For each leg it asks how well that leg's exploration current explains (a) the
leg's own seven joint velocities and (b) the learner's actual normalized error signal, at lags of 0..max_lag
control steps (5 ms each).

Cross-birth validity: the predictor basis is computed ONCE per leg from the pooled, centred currents of all
births (unsupervised: no response is used) and applied unchanged to every birth, and the joint response is the
full 7-joint velocity vector, never a per-birth principal component. A per-birth PCA has an arbitrary sign and
rotation, which can erase a perfect relation when births are pooled (Codex's counterexample, in the tests).

Significance: the response is circularly shifted inside each birth (keeps its autocorrelation); for every set of
shifts the maximum R^2 over lags and legs is kept (family-wise null). Descriptive and read-only: it never touches
weights, learning or checkpoints.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
DT_MS = 5.0
NEUROMERE = {"T1_PRONM": 0, "T2_MESONM": 1, "T3_METANM": 2}  # front, middle, hind
JOINTS_PER_LEG = 7


NERVE_NEUROMERE = {"prothoracic": 0, "mesothoracic": 1, "metathoracic": 2}
LEG_NERVE = re.compile(r"^(left|right)_(prothoracic|mesothoracic|metathoracic)_leg_nerve$")
REASONS = ("nerve", "region", "non_leg_nerve", "region_no_consensus", "region_ambiguous", "no_side_or_region")


def assign_legs_detailed(neurons, neuron_ids):
    """(leg, reason) per neuron id; leg 0..5 in the order LF LM LH RF RM RH, -1 when it cannot be assigned.

    Rules, in order: (1) a leg nerve name (``left_prothoracic_leg_nerve`` ...) gives side and neuromere (motor neurons);
    (2) another annotated nerve (ventral/accessory/dorsal/abdominal) is a non-leg nerve and stays unassigned; (3) without a nerve,
    an exact neuromere label (``T1_PRONM``/``T2_MESONM``/``T3_METANM``) plus the soma side; (4) a label with a ``.NO_CONS`` suffix
    (no consensus) or two neuromeres stays unassigned and is counted under its own reason, never guessed.
    """
    soma = neurons["Soma side"].to_numpy()
    nerve = neurons["Nerve"].to_numpy() if "Nerve" in neurons.columns else None
    region = neurons["Top in/out region"].to_numpy()
    legs, reasons = np.full(len(neuron_ids), -1, dtype=int), []
    for k, n in enumerate(neuron_ids):
        name = nerve[n] if nerve is not None and isinstance(nerve[n], str) else None
        if name is not None:
            match = LEG_NERVE.match(name)
            if match:
                legs[k] = (0 if match.group(1) == "left" else 3) + NERVE_NEUROMERE[match.group(2)]
                reasons.append("nerve")
            else:
                reasons.append("non_leg_nerve")
            continue
        label = str(region[n])
        if label.endswith(".NO_CONS"):
            reasons.append("region_no_consensus")
        elif label in NEUROMERE and soma[n] in ("left", "right"):
            legs[k] = (0 if soma[n] == "left" else 3) + NEUROMERE[label]
            reasons.append("region")
        elif "." in label:
            reasons.append("region_ambiguous")
        else:
            reasons.append("no_side_or_region")
    return legs, reasons


def assign_legs(neurons, neuron_ids):
    """Leg index per neuron id (-1 when unassigned); see ``assign_legs_detailed`` for the reasons."""
    return assign_legs_detailed(neurons, neuron_ids)[0]


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


def leg_basis(drives, legs, leg, components=4):
    """Shared predictor basis for one leg: (mean, projection) from the pooled currents of ALL births.

    ``drives`` is a list of (time x neurons) arrays with identical neuron order. The basis never depends on any
    response and is applied unchanged to every birth, so features have the same meaning in every birth.
    """
    columns = np.flatnonzero(legs == leg)
    if len(columns) == 0:
        raise ValueError("leg has no explored neurons")
    pooled = np.vstack([d[:, columns] for d in drives])
    mean = pooled.mean(axis=0)
    _, s, vt = np.linalg.svd(pooled - mean, full_matrices=False)
    k = min(components, len(s))
    return {"columns": columns, "mean": mean, "projection": vt[:k].T}


def project(drive, basis):
    """Features (time x components) of one birth in the shared basis."""
    return (drive[:, basis["columns"]] - basis["mean"]) @ basis["projection"]


def _lagged(x, lag):
    out = np.zeros_like(x)
    if lag:
        out[lag:] = x[:-lag]
    else:
        out[:] = x
    return out


def _columns(y):
    y = np.asarray(y, dtype=float)
    return y[:, None] if y.ndim == 1 else y


class LagFit:
    """OLS of a response on lagged features for lags 0..max_lag, pooled over births, with fast re-scoring.

    The response may have several columns (the seven joints of a leg): the coefficients are shared by all births and
    R^2 is the explained fraction of the TOTAL variance over all columns. The lagged design matrices and their normal
    equations are built once; ``r2`` then scores any response (for example a circularly shifted one) with a few matrix
    products. The first ``skip`` rows of every birth are dropped after lagging so all lags use the same rows; each
    birth is centred separately.
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
        y = np.vstack([_columns(r)[self.skip:] - _columns(r)[self.skip:].mean(axis=0) for r in response])
        xty = np.einsum("lnk,nj->lkj", self.X, y)
        explained = np.einsum("lkj,lkm,lmj->l", xty, self.inverse, xty)
        return explained / max(float((y * y).sum()), 1e-12)


def lag_r2(features, response, max_lag, skip):
    """R^2 of an OLS fit of the response on the lagged features, for lags 0..max_lag, pooled over births."""
    return LagFit(features, max_lag, skip).r2(response)


def circular_shift(y, rng, minimum):
    shift = int(rng.integers(minimum, len(y) - minimum))
    return np.roll(y, shift, axis=0)


def analyse(features_by_leg, responses_by_leg, max_lag=100, permutations=200, seed=0, min_shift=200):
    """For each leg: R^2 curve, best lag, and the family-wise p over (legs x lags).

    features_by_leg[leg] / responses_by_leg[leg]: lists with one array per birth (features: time x components,
    response: time or time x columns). The family-wise null is the maximum R^2 over all legs and lags for each set of
    independent circular shifts.
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
                       "family_rank": float((1 + np.sum(null_max >= curve[lag])) / (permutations + 1)),
                       "rank_is_calibrated_p_value": False,
                       "curve": curve.tolist()}
    return result, {"null_max_p95": float(np.percentile(null_max, 95)), "null_max_p99": float(np.percentile(null_max, 99))}


def load_birth(log_path, observations_path):
    """Raw per-birth arrays. Row i of the log is the i-th learning-active control step."""
    with np.load(log_path) as z:
        drive, drive_idx = z["drive"].astype(float), z["drive_idx"]
    with np.load(observations_path) as o:
        reward, velocity, active = o["reward"].astype(float), o["velocity"].astype(float), o["learning_active"].astype(bool)
    rows = np.flatnonzero(active)
    if len(rows) != len(drive):
        raise ValueError("exploration log rows must equal the number of learning-active observation rows")
    signal = learning_signal(reward[rows])  # exactly the learner's error signal over the learning-active steps
    joints = velocity[rows].reshape(len(rows), 6, JOINTS_PER_LEG)
    return {"drive": drive, "drive_idx": drive_idx, "signal": signal, "joints": joints}


INTERPRETATION = ("Descriptive development association only. Permutation ranks are not calibrated p-values; stationarity and "
                  "exchangeability are not established; no causal conclusion; a non-significant rank does not show that learning is absent.")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coverage_report(legs, reasons, drives, components, tolerance=1e-9):
    """Explicit coverage: how many explored neurons reach each leg, which are unassigned and why, which are constant."""
    pooled_std = np.vstack(drives).std(axis=0)
    constant = pooled_std <= tolerance
    report = {"explored_neurons": len(legs),
              "assigned": int((legs >= 0).sum()),
              "unassigned_by_reason": {r: int(sum(1 for x in reasons if x == r)) for r in REASONS if r not in ("nerve", "region")},
              "assigned_by": {r: int(sum(1 for x in reasons if x == r)) for r in ("nerve", "region")},
              "constant_columns": int(constant.sum()),
              "constant_columns_per_birth": [int((d.std(axis=0) <= tolerance).sum()) for d in drives],
              "min_usable_columns_per_leg": components + 1, "per_leg": {}}
    for leg in range(6):
        assigned = legs == leg
        usable = assigned & ~constant
        report["per_leg"][leg] = {"assigned": int(assigned.sum()), "constant": int((assigned & constant).sum()),
                                  "usable": int(usable.sum()),
                                  "status": "ok" if usable.sum() >= components + 1 else "INSUFFICIENT_COVERAGE"}
    return report, np.where(constant, -1, legs)


def provenance(log_dirs, neurons_sha256, parameters):
    from pathlib import Path

    from noesis import plasticity

    return {"inputs": [{"directory": str(d),
                        "exploration_log_sha256": file_sha256(Path(d) / "exploration-log.npz"),
                        "observations_sha256": file_sha256(Path(d) / "training-observations.npz")} for d in log_dirs],
            "neurons_table_sha256": neurons_sha256, "module_sha256": file_sha256(__file__),
            "plasticity_source_sha256": file_sha256(plasticity.__file__), "parameters": parameters,
            "numpy": np.__version__}


def per_birth_summary(features, responses, max_lag, skip):
    """Best lag and R^2 of each birth analysed on its own with the shared basis (consistency across births)."""
    summary = []
    for i in range(len(next(iter(features.values())))):
        entry = {}
        for leg in features:
            curve = LagFit([features[leg][i]], max_lag, skip).r2([responses[leg][i]])
            lag = int(curve.argmax())
            entry[leg] = {"best_lag_ms": lag * DT_MS, "r2": float(curve[lag])}
        summary.append(entry)
    return summary


def run(log_dirs, neurons, max_lag=100, permutations=200, seed=0, components=4, neurons_sha256=None):
    """Analyse several births (each directory holds exploration-log.npz and training-observations.npz).

    Legs with fewer than ``components + 1`` usable (assigned, non-constant) explored neurons are reported as
    INSUFFICIENT_COVERAGE and not analysed.
    """
    from pathlib import Path
    births = [load_birth(Path(d) / "exploration-log.npz", Path(d) / "training-observations.npz") for d in log_dirs]
    if any(not np.array_equal(b["drive_idx"], births[0]["drive_idx"]) for b in births):
        raise ValueError("births explore different neurons; the shared predictor basis needs identical neuron order")
    legs, reasons = assign_legs_detailed(neurons, births[0]["drive_idx"])
    drives = [b["drive"] for b in births]
    coverage, usable_legs = coverage_report(legs, reasons, drives, components)
    analysed = [leg for leg in range(6) if coverage["per_leg"][leg]["status"] == "ok"]
    features = {leg: [project(b["drive"], basis) for b in births]
                for leg, basis in ((leg, leg_basis(drives, usable_legs, leg, components)) for leg in analysed)}
    signal = {leg: [b["signal"] for b in births] for leg in analysed}
    joints = {leg: [b["joints"][:, leg, :] for b in births] for leg in analysed}
    result = {"legs": LEGS, "births": len(births), "max_lag_ms": max_lag * DT_MS, "analysed_legs": analysed,
              "coverage": coverage, "interpretation": INTERPRETATION,
              "provenance": provenance(log_dirs, neurons_sha256,
                                       {"max_lag": max_lag, "permutations": permutations, "seed": seed, "components": components})}
    if analysed:
        for key, responses, offset in (("learning_signal", signal, 0), ("joint_velocity", joints, 1)):
            per_leg, null = analyse(features, responses, max_lag, permutations, seed + offset)
            result[key] = {"per_leg": per_leg, "null": null,
                           "per_birth": per_birth_summary(features, responses, max_lag, max_lag)}
    for leg in range(6):
        if leg not in analysed:
            for key in ("learning_signal", "joint_velocity"):
                result.setdefault(key, {"per_leg": {}, "null": None, "per_birth": []})["per_leg"][leg] = {"status": "INSUFFICIENT_COVERAGE"}
    return result


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
    result = run(args.births, pd.read_parquet(args.neurons), args.max_lag, args.permutations, args.seed, args.components,
                 neurons_sha256=file_sha256(args.neurons))
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    cov = result["coverage"]
    print(f"coverage: {cov['assigned']}/{cov['explored_neurons']} explored neurons assigned to a leg; unassigned {cov['unassigned_by_reason']}; "
          f"constant columns {cov['constant_columns']}")
    for kind in ("joint_velocity", "learning_signal"):
        for leg, entry in result[kind]["per_leg"].items():
            if entry.get("status") == "INSUFFICIENT_COVERAGE":
                print(f"{kind:16s} {LEGS[int(leg)]}: INSUFFICIENT_COVERAGE")
                continue
            print(f"{kind:16s} {LEGS[int(leg)]}: best lag {entry['best_lag_ms']:5.0f} ms  R2 {entry['r2']:.4f}  rank {entry['family_rank']:.4f}")
        if result[kind]["null"]:
            print(f"{kind:16s} null max R2: p95 {result[kind]['null']['null_max_p95']:.4f}  p99 {result[kind]['null']['null_max_p99']:.4f}")
    print(INTERPRETATION)


if __name__ == "__main__":
    main()

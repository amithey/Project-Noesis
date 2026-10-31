"""Offline H-bal test (preregistered in README 'הצעה הבאה', criteria fixed here BEFORE running).

H-bal: a larger tilt_scale lowers saturation of the Johnston's-organ afferents during ordinary standing and increases how well
the population activity encodes the body tilt, with no topology change.
Success (all): for tilt_scale 0.6 vs the current 0.3 -- saturated share (rate > 0.95 of span, over neuron-time) falls by >= 10
points, AND leave-one-probe-seed-out ridge R^2 of decoding the 3-D tilt vector from the JO activations rises by >= 0.05.
Else H-bal is rejected. Scales 1.0 and 0.15 are reported only. Descriptive: encodes tilt, says nothing about behaviour.
usage: balance_tilt_scale.py <core src> <neurons.parquet> <out.json> <run dir> [<run dir> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from noesis.balance import UPRIGHT, BalanceMap  # noqa: E402

neurons, out, dirs = pd.read_parquet(sys.argv[2]), sys.argv[3], [Path(d) for d in sys.argv[4:]]
tilts = []
for d in dirs:
    z = np.load(d / "observations.npz")
    tilts.append(z["up_body"][10:, 0] - UPRIGHT)


def decode_r2(X, Y, groups, lam=1.0):
    scores = []
    for g in range(len(groups)):
        tr = np.concatenate([i for k, i in enumerate(groups) if k != g])
        te = groups[g]
        mu = X[tr].mean(0)
        A = X[tr] - mu
        W = np.linalg.solve(A.T @ A + lam * len(tr) * np.eye(X.shape[1]), A.T @ (Y[tr] - Y[tr].mean(0)))
        P = (X[te] - mu) @ W + Y[tr].mean(0)
        scores.append(1 - ((Y[te] - P) ** 2).sum() / ((Y[te] - Y[te].mean(0)) ** 2).sum())
    return float(np.mean(scores))


Y = np.concatenate(tilts)
idx, start = [], 0
for t in tilts:
    idx.append(np.arange(start, start + len(t)))
    start += len(t)
res = {"runs": [str(d) for d in dirs], "tilt_norm_p50": float(np.median(np.linalg.norm(Y, axis=1))), "scales": {}}
for scale in (0.15, 0.3, 0.6, 1.0):
    bm = BalanceMap(neurons, tilt_scale=scale, seed=0)
    G = np.maximum(0, np.tanh(Y @ bm.jo_dir.T / scale))  # JO activation 0..1 (span-normalised), neuron-time
    res["scales"][str(scale)] = {"n_jo": int(G.shape[1]), "saturated_share": float((G > .95).mean()),
                                 "dead_share": float((G < .05).mean()), "mean_activation": float(G.mean()),
                                 "tilt_decode_r2_loso": decode_r2(G, Y, idx)}
a, b = res["scales"]["0.3"], res["scales"]["0.6"]
res["delta_saturated_points"] = 100 * (b["saturated_share"] - a["saturated_share"])
res["delta_decode_r2"] = b["tilt_decode_r2_loso"] - a["tilt_decode_r2_loso"]
res["H_bal_success"] = bool(res["delta_saturated_points"] <= -10 and res["delta_decode_r2"] >= .05)
Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
for k, v in res["scales"].items():
    print(k, {x: round(y, 4) for x, y in v.items()})
print({k: res[k] for k in ("delta_saturated_points", "delta_decode_r2", "H_bal_success")})

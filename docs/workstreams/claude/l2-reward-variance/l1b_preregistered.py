"""L1b: pre-registered test of the two exploratory joint findings on NEW births (written and committed BEFORE the data exist).

Hypotheses (from the post hoc L1 finding): the exploration current explains joint velocity through the controller's muscle pools for
  H1  joint_RFTarsus1 at a lag of 190 ms +- 20 ms
  H2  joint_LFTarsus1 at a lag of  25 ms +- 20 ms
Predictor per joint: mean current of explored agonist neurons minus mean current of explored antagonist neurons (build_muscle_matrices).
Statistic per joint: the maximum R^2 over the declared lag window (+-4 control steps), pooled over births (one feature, shared coefficient).
Null: circular shifts of the response inside each birth (>= 200 steps), 1000 draws; for every draw the maximum of the two joints' window
maxima is kept (family-wise over the two declared joints); rank = share of draws with null max >= observed.
Success (fixed in advance): BOTH joints have rank <= 0.05 on all 8 births AND in each half of the births (seeds 1008-1011 and 1012-1015)
analysed separately with the same procedure; anything else rejects the finding. The sign of the fitted coefficient is reported, not tested.
No other joint, lag or response is part of this test.

usage: l1b_preregistered.py <core src> <pilot src> <neurons.parquet> <out.json> <run dir> [<run dir> ...]    (8 dirs: seeds 1008..1015 in order)
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [sys.argv[1], sys.argv[2]]
from noesis.motor import build_muscle_matrices  # noqa: E402

from noesis_gpu import credit_timing as ct  # noqa: E402
from noesis_gpu.bridge import LEGACY_JOINTS  # noqa: E402

HYPOTHESES = {"joint_RFTarsus1": 190.0, "joint_LFTarsus1": 25.0}
WINDOW_STEPS, PERMUTATIONS, MIN_SHIFT = 4, 1000, 200
neurons, out_path, dirs = pd.read_parquet(sys.argv[3]), sys.argv[4], sys.argv[5:]
if len(dirs) != 8:
    raise SystemExit("exactly 8 births (seeds 1008-1015) are required")
gpos, gneg = build_muscle_matrices(neurons, list(LEGACY_JOINTS))
births = [ct.load_birth(f"{d}/exploration-log.npz", f"{d}/training-observations.npz") for d in dirs]
idx = births[0]["drive_idx"]
pos, neg = gpos[:, idx].toarray() > 0, gneg[:, idx].toarray() > 0


def joint_data(name, subset):
    j = LEGACY_JOINTS.index(name)
    if not pos[j].any() or not neg[j].any():
        raise SystemExit(f"{name}: no explored agonist/antagonist pool")
    w = pos[j] / pos[j].sum() - neg[j] / neg[j].sum()
    features = [(births[i]["drive"] @ w)[:, None] for i in subset]
    responses = [births[i]["joints"].reshape(len(births[i]["drive"]), 42)[:, j] for i in subset]
    sign = float(np.sign(np.mean([np.corrcoef(f[:, 0], r)[0, 1] for f, r in zip(features, responses)])))
    return features, responses, sign


def test(subset, seed):
    rng = np.random.default_rng(seed)
    fits, observed, resp, signs = {}, {}, {}, {}
    for name, lag_ms in HYPOTHESES.items():
        features, responses, sign = joint_data(name, subset)
        centre = int(round(lag_ms / ct.DT_MS))
        lags = slice(centre - WINDOW_STEPS, centre + WINDOW_STEPS + 1)
        fit = ct.LagFit(features, 100, 100)  # same rows for every lag; lags above 100 steps are not used
        curve = fit.r2(responses)
        fits[name], resp[name], signs[name] = (fit, lags), responses, sign
        observed[name] = {"window_max_r2": float(curve[lags].max()), "best_lag_ms": float((lags.start + int(curve[lags].argmax())) * ct.DT_MS)}
    null_max = np.zeros(PERMUTATIONS)
    for p in range(PERMUTATIONS):
        best = 0.0
        for name in HYPOTHESES:
            fit, lags = fits[name]
            shifted = [ct.circular_shift(r, rng, MIN_SHIFT) for r in resp[name]]
            best = max(best, float(fit.r2(shifted)[lags].max()))
        null_max[p] = best
    return {name: {**observed[name], "coefficient_sign": signs[name],
                   "rank": float((1 + np.sum(null_max >= observed[name]["window_max_r2"])) / (PERMUTATIONS + 1))}
            for name in HYPOTHESES}, {"null_max_p95": float(np.percentile(null_max, 95)), "null_max_p99": float(np.percentile(null_max, 99))}


result = {"hypotheses": HYPOTHESES, "window_ms": WINDOW_STEPS * ct.DT_MS, "permutations": PERMUTATIONS, "births": len(births)}
for name, subset in (("all8", range(8)), ("half1_seeds1008_1011", range(4)), ("half2_seeds1012_1015", range(4, 8))):
    result[name], result[name + "_null"] = test(list(subset), seed=11)
ok = all(result[k][h]["rank"] <= 0.05 for k in ("all8", "half1_seeds1008_1011", "half2_seeds1012_1015") for h in HYPOTHESES)
result["preregistered_success"] = bool(ok)
result["provenance"] = {"inputs": [ct.file_sha256(f"{d}/exploration-log.npz") for d in dirs], "module_sha256": ct.file_sha256(__file__)}
json.dump(result, open(out_path, "w"), indent=1)
for k in ("all8", "half1_seeds1008_1011", "half2_seeds1012_1015"):
    print(k, {h: (round(v["window_max_r2"], 4), v["best_lag_ms"], round(v["rank"], 3)) for h, v in result[k].items()}, "null p99", round(result[k + "_null"]["null_max_p99"], 4))
print("PREREGISTERED SUCCESS:", ok)

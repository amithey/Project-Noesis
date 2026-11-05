"""Offline analysis of the registered chain diagnostic (rule in protocol.md, fixed before data).
usage: analyse_chain.py <core src> <pilot src> <neurons.parquet> <run dir> <out.json>   (run dir holds seed-10006 .. seed-10011)"""
import json
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [sys.argv[1], sys.argv[2]]
from noesis.motor import build_muscle_matrices  # noqa: E402

from noesis_gpu import chain_diagnostic as cd  # noqa: E402
from noesis_gpu.bridge import LEGACY_JOINTS  # noqa: E402

neurons, run, out = pd.read_parquet(sys.argv[3]), sys.argv[4], sys.argv[5]
SEEDS = (10006, 10007, 10010, 10011)
gpos, gneg = build_muscle_matrices(neurons, list(LEGACY_JOINTS))
result = {"seeds": {}, "reference": "descriptive circular-shift reference; not power or MDE"}
for seed in SEEDS:
    arms = {}
    for arm in cd.ARMS:
        with np.load(f"{run}/seed-{seed}/chain-{arm}.npz") as z:
            arms[arm] = {k: z[k] for k in z.files}
    idx = arms["active"]["drive_idx"]
    pos, neg = gpos[:, idx].toarray() > 0, gneg[:, idx].toarray() > 0
    result["seeds"][seed] = cd.chain_report(arms, pos, neg, max_lag=40, permutations=1000, min_shift=200, seed=0)
    result["seeds"][seed]["driven_neurons"] = int(len(idx))
    result["seeds"][seed]["joints_with_both_pools"] = int((pos.any(axis=1) & neg.any(axis=1)).sum())
    result["seeds"][seed]["active_max_abs_current"] = float(np.abs(arms["active"]["drive"]).max())
    result["seeds"][seed]["total_spikes"] = {a: int(arms[a]["total_spikes"].sum()) for a in cd.ARMS}


def coupled(entry):
    return entry.get("status") == "ok" and entry["reference_rank"] <= 0.05


reading = {}
for stage in cd.STAGES:
    act = [result["seeds"][s]["coupling"]["active"][stage] for s in SEEDS]
    signs = {e.get("sign") for e in act if coupled(e)}
    n_act = sum(coupled(e) for e in act)
    n_neg = {a: sum(coupled(result["seeds"][s]["coupling"][a][stage]) for s in SEEDS) for a in ("off", "zero")}
    reading[stage] = {"active_coupled": n_act, "same_sign": len(signs) <= 1, "negative_controls_coupled": n_neg,
                      "reaches": bool(n_act >= 3 and len(signs) <= 1 and all(v <= 1 for v in n_neg.values()))}
result["reading"] = reading
json.dump(result, open(out, "w"), indent=1)
for stage, r in reading.items():
    print(stage, r)
for s in SEEDS:
    print(s, {st: (round(e["observed_r2"], 4) if e.get("observed_r2") is not None else None, e.get("best_lag_steps"), round(e.get("reference_rank", 9), 3))
              for st, e in result["seeds"][s]["coupling"]["active"].items()}, "div", {k: v for k, v in result["seeds"][s]["first_divergence_active_vs_off"].items() if k in ("circuit_spikes", "a_pos", "command", "obs_velocity")})

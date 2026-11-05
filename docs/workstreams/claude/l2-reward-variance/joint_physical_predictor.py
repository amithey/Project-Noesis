"""Exploratory (post hoc): does the exploration current predict joint velocity through the controller's own muscle pools?

Per joint j the predictor is the pool difference p_j(t) = mean(current of explored agonist neurons) - mean(current of explored
antagonist neurons), taken from noesis.motor.build_muscle_matrices (the same pools the muscle model uses); the response is
that joint's velocity. One feature per joint, lags 0-500 ms, family-wise permutation null over all joints and lags
(circular shifts of the response inside each birth). Read-only.

usage: joint_physical_predictor.py <core src> <pilot src> <neurons.parquet> <run dir> [<run dir> ...]
"""
import sys

import numpy as np
import pandas as pd

sys.path[:0] = [sys.argv[1], sys.argv[2]]
from noesis.motor import build_muscle_matrices  # noqa: E402

from noesis_gpu import credit_timing as ct  # noqa: E402
from noesis_gpu.bridge import LEGACY_JOINTS  # noqa: E402

neurons = pd.read_parquet(sys.argv[3])
dirs = sys.argv[4:]
gpos, gneg = build_muscle_matrices(neurons, list(LEGACY_JOINTS))
births = [ct.load_birth(f"{d}/exploration-log.npz", f"{d}/training-observations.npz") for d in dirs]
idx = births[0]["drive_idx"]
pos = gpos[:, idx].toarray()
neg = gneg[:, idx].toarray()
usable = [j for j in range(42) if (pos[j] > 0).any() and (neg[j] > 0).any()]
print(f"{len(dirs)} births; explored neurons {len(idx)}; joints with explored agonist AND antagonist pools: {len(usable)}/42; "
      f"explored neurons in any pool: {int(((pos > 0) | (neg > 0)).any(axis=0).sum())}")
features, responses = {}, {}
for j in usable:
    w_pos = pos[j] / max((pos[j] > 0).sum(), 1) * (pos[j] > 0)
    w_neg = neg[j] / max((neg[j] > 0).sum(), 1) * (neg[j] > 0)
    w = (pos[j] > 0) / max((pos[j] > 0).sum(), 1) - (neg[j] > 0) / max((neg[j] > 0).sum(), 1)
    features[j] = [(b["drive"] @ w)[:, None] for b in births]
    responses[j] = [b["joints"].reshape(len(b["drive"]), 42)[:, j] for b in births]
    del w_pos, w_neg
result, null = ct.analyse(features, responses, max_lag=100, permutations=300, seed=5)
order = sorted(result, key=lambda j: -result[j]["r2"])[:8]
print(f"null max R2: p95 {null['null_max_p95']:.4f}  p99 {null['null_max_p99']:.4f}")
for j in order:
    r = result[j]
    print(f"{LEGACY_JOINTS[j]:22s} best lag {r['best_lag_ms']:5.0f} ms  R2 {r['r2']:.4f}  rank {r['family_rank']:.3f}")
print(f"joints with rank <= 0.05: {sum(result[j]['family_rank'] <= 0.05 for j in result)} of {len(result)}")

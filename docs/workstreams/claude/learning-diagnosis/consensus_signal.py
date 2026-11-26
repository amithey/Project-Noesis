"""Is there a consistent (reward-driven) direction in birth-to-birth weight changes, or mostly noise-realization?

Uses one run per independent training seed per parent (the same seed replays the same exploration noise).
Edges with delta exactly 0 in a birth (silent presynaptic input) are excluded from sign statistics.

usage: consensus_signal.py <outputs dir>
"""
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return z["weights"].astype(float), z["initial_weights"].astype(float), z["motor_indices"].copy()


def rel(path):
    w, w0, m = load(path)
    return np.abs(w) / np.abs(w0) - 1.0, w0, m


reports = {}
for ck in OUT.rglob("weights-last.npz"):
    rp = ck.parent / "report.json"
    meta = json.loads(rp.read_text(encoding="utf-8")) if rp.is_file() else {}
    reports[sha(ck)] = (ck, meta)
paths = {h: v[0] for h, v in reports.items()}
for ck in OUT.rglob("*.npz"):
    if ck.name in ("weights-last.npz", "observations.npz", "training-observations.npz"):
        continue
    try:
        with np.load(ck, allow_pickle=False) as z:
            if {"weights", "initial_weights", "motor_indices"} <= set(z.files):
                paths.setdefault(sha(ck), ck)
    except Exception:  # noqa: BLE001
        pass

groups = {}
for h, (ck, meta) in reports.items():
    parent = meta.get("import_checkpoint_sha256")
    if parent in paths:
        groups.setdefault(parent, {}).setdefault(meta.get("seed"), (ck, meta))

for parent, by_seed in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    if len(by_seed) < 3:
        continue
    rp, w0, m = rel(paths[parent])
    D = []
    for seed, (ck, meta) in sorted(by_seed.items(), key=lambda kv: str(kv[0])):
        r, w0c, mc = rel(ck)
        if not (np.array_equal(w0c, w0) and np.array_equal(mc, m)):
            continue
        D.append(r - rp)
    D = np.array(D)
    k, n = D.shape
    changed = D != 0
    all_changed = changed.all(axis=0)
    print(f"\nparent {parent[:8]}: {k} independent seeds, {n} plastic edges")
    print(f"  edges changed in each birth: {[f'{c.mean():.1%}' for c in changed]}; changed in ALL: {all_changed.mean():.1%}; changed in NONE: {(~changed.any(axis=0)).mean():.1%}")
    sub = D[:, all_changed]
    signs = np.sign(sub)
    unanimous = float(np.mean(np.all(signs == signs[0], axis=0)))
    print(f"  among edges changed in all births: same sign in all {k} births {unanimous:.2%} (chance {2 * 0.5 ** k:.2%})")
    cos = [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in combinations(sub, 2)]
    print(f"  pairwise cosine on those edges: mean {np.mean(cos):+.4f} range [{min(cos):+.4f}, {max(cos):+.4f}]  (null +-{1 / np.sqrt(sub.shape[1]):.4f})")
    mean_d = sub.mean(axis=0)
    ratio = float(np.sum(mean_d ** 2) / np.mean(np.sum(sub ** 2, axis=1)))
    print(f"  energy of the mean change / mean energy of one birth's change: {ratio:.3f}  (independent noise -> {1 / k:.3f}; identical births -> 1.0)")
    # reliability: split-half correlation of per-edge deltas (seed groups)
    half = k // 2
    a, b = sub[:half].mean(axis=0), sub[half:2 * half].mean(axis=0)
    print(f"  split-half correlation of per-edge change ({half} vs {half} seeds): {float(np.corrcoef(a, b)[0, 1]):+.4f}")

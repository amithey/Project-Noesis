"""Geometry of learned-weight changes across births, from checkpoints only (no connectome, no physics).

For every stored training birth with a known parent checkpoint: relative magnitude change vs the ORIGINAL
reference r = |w| / |w_initial| - 1, birth change d = r_child - r_parent, and how consistent d is across
independent siblings (same parent, different training seed). Read-only.

usage: checkpoint_geometry.py <outputs dir> <out.json>
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
        w, w0 = z["weights"].astype(float), z["initial_weights"].astype(float)
        return dict(w=w, w0=w0, motor=z["motor_indices"].copy(), params=z["parameters"].copy())


def rel(c):
    return np.abs(c["w"]) / np.abs(c["w0"]) - 1.0


# index every stored weights-last.npz by its own hash, with the report that produced it
index = {}
for ck in OUT.rglob("weights-last.npz"):
    rep_path = ck.parent / "report.json"
    meta = {}
    if rep_path.is_file():
        try:
            meta = json.loads(rep_path.read_text(encoding="utf-8"))
        except ValueError:
            meta = {}
    h = sha(ck)
    index[h] = dict(path=ck, dir=ck.parent.name, objective=meta.get("reward_objective"), seed=meta.get("seed"),
                    parent=meta.get("import_checkpoint_sha256"), lr=None)
print(f"{len(index)} stored weights-last.npz checkpoints")

# group children by parent hash (parent must itself be a stored file or known by path elsewhere)
extra_parents = {}
for ck in OUT.rglob("*.npz"):
    if ck.name.startswith("weights") or ck.name in ("observations.npz", "training-observations.npz"):
        continue
    try:
        with np.load(ck, allow_pickle=False) as z:
            if {"weights", "initial_weights", "motor_indices"} <= set(z.files):
                extra_parents[sha(ck)] = ck
    except Exception:  # noqa: BLE001 - unrelated npz files
        pass
lookup = {h: v["path"] for h, v in index.items()}
lookup.update(extra_parents)

groups = {}
for h, meta in index.items():
    if meta["parent"] in lookup:
        groups.setdefault(meta["parent"], []).append(h)

report = []
for parent_hash, children in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    parent = load(lookup[parent_hash])
    rp = rel(parent)
    print(f"\nparent {parent_hash[:8]}  (mean |r| {np.abs(rp).mean():.4f}, max {np.abs(rp).max():.3f}, edges {len(rp)})  children {len(children)}")
    deltas, names = [], []
    for h in children:
        c = load(index[h]["path"])
        if not (np.array_equal(c["w0"], parent["w0"]) and np.array_equal(c["motor"], parent["motor"])):
            print("   skipped (different reference or targets):", index[h]["dir"])
            continue
        rc = rel(c)
        d = rc - rp
        deltas.append(d)
        label = f"{index[h]['dir']}"
        names.append(label)
        print(f"   {label:34s} obj {str(index[h]['objective'])[:30]:30s} seed {index[h]['seed']}  |d| rms {np.sqrt((d**2).mean()):.5f}  "
              f"mean|d| {np.abs(d).mean():.5f}  retained-vs-parent cos {float(rc @ rp / (np.linalg.norm(rc) * np.linalg.norm(rp))):.4f}  "
              f"lr {c['params'][0]:g}")
    if len(deltas) >= 2:
        D = np.array(deltas)
        cos = []
        for i, j in combinations(range(len(D)), 2):
            a, b = D[i], D[j]
            cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
        # sign agreement per edge across all siblings, against the chance level for k independent fair signs
        k = len(D)
        signs = np.sign(D)
        unanimous = float(np.mean(np.all(signs == signs[0], axis=0)))
        chance = 2 * 0.5 ** k
        # mean delta direction vs each sibling's delta
        mean_d = D.mean(axis=0)
        print(f"   pairwise cosine of birth changes: mean {np.mean(cos):+.4f}  min {min(cos):+.4f}  max {max(cos):+.4f}  "
              f"(null for random independent changes ~ 0 +- {1 / np.sqrt(D.shape[1]):.4f})")
        print(f"   edges with unanimous sign across {k} siblings: {unanimous:.3%} (chance {chance:.3%})")
        print(f"   norm of mean change / mean norm of change: {np.linalg.norm(mean_d) / np.mean(np.linalg.norm(D, axis=1)):.3f} "
              f"(1/sqrt(k)={1 / np.sqrt(k):.3f} if independent, 1.0 if identical)")
        report.append(dict(parent=parent_hash, children=names, pairwise_cosine=cos, unanimous_sign=unanimous,
                           chance=chance, mean_over_norm=float(np.linalg.norm(mean_d) / np.mean(np.linalg.norm(D, axis=1)))))

# ---- independent seeds versus same-seed pairs (the same training seed replays the same exploration noise) ----
print("\n=== same seed vs different seed (per parent) ===")
summary = []
for parent_hash, children in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    parent = load(lookup[parent_hash])
    rp = rel(parent)
    by_seed = {}
    for h in children:
        c = load(index[h]["path"])
        if not (np.array_equal(c["w0"], parent["w0"]) and np.array_equal(c["motor"], parent["motor"])):
            continue
        by_seed.setdefault(index[h]["seed"], []).append((index[h]["objective"], index[h]["dir"], rel(c) - rp, h))
    def cosine(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    same = []
    for seed, items in by_seed.items():
        for (oa, da, a, ha), (ob, db, b, hb) in combinations(items, 2):
            if ha != hb:
                same.append((seed, oa, ob, cosine(a, b)))
    reps = [items[0][2] for _, items in sorted(by_seed.items(), key=lambda kv: str(kv[0]))]
    seeds = sorted(by_seed, key=str)
    cross = [cosine(a, b) for a, b in combinations(reps, 2)]
    print(f"parent {parent_hash[:8]}: independent seeds {seeds}")
    if same:
        print("   same seed, different run (cosine of the birth change): " + ", ".join(f"seed {sd}: {c:.4f} [{str(o1)[-9:]} vs {str(o2)[-9:]}]" for sd, o1, o2, c in same))
    if cross:
        print(f"   different seeds (one run per seed), pairwise cosine: mean {np.mean(cross):+.4f} min {min(cross):+.4f} max {max(cross):+.4f}  (null +-{1 / np.sqrt(len(reps[0])):.4f})")
        R = np.array(reps)
        unanimous = float(np.mean(np.all(np.sign(R) == np.sign(R[0]), axis=0)))
        print(f"   edges with the same sign in all {len(reps)} independent births: {unanimous:.2%} (chance {2 * 0.5 ** len(reps):.2%})")
    summary.append(dict(parent=parent_hash, seeds=[str(x) for x in seeds], same_seed=same, cross=cross))

json.dump(report, open(sys.argv[2], "w"), indent=1)

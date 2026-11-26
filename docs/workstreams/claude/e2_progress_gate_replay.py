"""Prototype tighter progress gates for the v9 cycle bonus and replay them on stored runs. Read-only."""
import glob
import os
import sys

import numpy as np

sys.path[:0] = [r"C:/dev/noesis-claude-core/src"]
from noesis.embodied_metrics import LegCycleTracker  # noqa: E402

OUT = r"C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/gpu-pilot/outputs"
DT = .005
GATES = {  # name: (window seconds, mean forward speed mm/s over that window, short 0.25 s mean must also be > 0)
    "current 0.25s>=0.5": (.25, .5, False),
    "1s>=0.25": (1., .25, False),
    "1s>=0.5": (1., .5, False),
    "2s>=0.25": (2., .25, False),
    "1s>=0.25 & 0.25s>0": (1., .25, True),
}


def load(path):
    z = np.load(path)

    def get(key, ndim):
        return z[key][:, 0] if z[key].ndim > ndim else z[key]
    return {"position": get("position", 2), "up": get("up_body", 2), "heading": get("heading", 2),
            "load": get("foot_load", 2)}


def series(obs):
    n = len(obs["position"])
    tracker = LegCycleTracker(dt_seconds=DT, min_forward_mm_s=0.)
    speed = np.zeros(n)
    base = np.zeros(n)
    for i in range(1, n):
        h = obs["heading"][i][:2] / max(np.linalg.norm(obs["heading"][i][:2]), 1e-12)
        speed[i] = (obs["position"][i][:2] - obs["position"][i - 1][:2]) @ h / DT
        height = obs["position"][i][2]
        standing = bool(obs["up"][i][2] >= .7 and .3 <= height <= 1.5 and (obs["load"][i] > 1e-5).any())
        # speed fed as +1 so the tracker's own speed gate is open: isolates cycles + support + standing
        base[i] = tracker.update(obs["load"][i], 1., standing)
    return speed, base


def trailing_mean(speed, seconds):
    k = round(seconds / DT)
    cs = np.cumsum(np.r_[0, speed])
    out = np.full(len(speed), np.nan)
    out[k - 1:] = (cs[k:] - cs[:len(speed) - k + 1]) / k
    return out


def gated(speed, base, window_s, minimum, short):
    mean = trailing_mean(speed, window_s)
    ok = np.nan_to_num(mean, nan=-1.) >= minimum
    if short:
        ok &= np.nan_to_num(trailing_mean(speed, .25), nan=-1.) > 0
    return np.where(ok & (base > 0), base, 0.)


runs = {}
for pattern in ("gripped-v8-00*-train-*/training-observations.npz", "gripped-v8-00*-candidate-*/observations.npz",
                "standing-001-candidate-*/observations.npz"):
    for p in sorted(glob.glob(f"{OUT}/{pattern}")):
        runs[os.path.basename(os.path.dirname(p))] = series(load(p))

print("share of steps paying credit (after warm-up), per gate")
print(f"{'run':34s} " + " ".join(f"{g[:18]:>18s}" for g in GATES))
agg = {g: [0, 0, 0, 0.] for g in GATES}  # paid, steps, paid while trailing 2 s mean forward <= 0, credit sum
for name, (speed, base) in runs.items():
    two = np.nan_to_num(trailing_mean(speed, 2.), nan=0.)[10:]
    cells = []
    for g, (w, m, s) in GATES.items():
        credit = gated(speed, base, w, m, s)[10:]
        paid = credit > 0
        agg[g][0] += int(paid.sum())
        agg[g][1] += len(paid)
        agg[g][2] += int((paid & (two <= 0)).sum())
        agg[g][3] += float(credit.sum())
        cells.append(f"{paid.mean():18.1%}")
    print(f"{name:34s} " + " ".join(cells))
print("\naggregate over all runs")
print(f"{'gate':24s} {'paid steps':>10s} {'share':>7s} {'paid while 2s mean fwd<=0':>28s} {'total credit':>13s}")
for g, (paid, n, bad, total) in agg.items():
    print(f"{g:24s} {paid:10d} {paid / n:7.1%} {bad:15d} ({bad / max(paid, 1):5.1%}) {total:13.1f}")
parent = [k for k in runs if k.startswith("standing-001")]
cand = [k for k in runs if k.startswith("gripped") and "candidate" in k]
print("\nparent vs v8-candidate frozen runs: mean share of steps paying credit")
print(f"{'gate':24s} {'parent':>8s} {'candidates':>11s}")
for g, (w, m, s) in GATES.items():
    pr = np.mean([np.mean(gated(*runs[k], w, m, s)[10:] > 0) for k in parent])
    cr = np.mean([np.mean(gated(*runs[k], w, m, s)[10:] > 0) for k in cand])
    print(f"{g:24s} {pr:8.1%} {cr:11.1%}")
print("\nmean forward speed over each frozen run (mm/s)")
for k in parent + cand:
    print(f"  {k:34s} {runs[k][0][10:].mean():6.3f}")

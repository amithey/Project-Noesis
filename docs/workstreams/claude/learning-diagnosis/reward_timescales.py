"""At which timescales does the training reward vary? (what the credit rule has to work with). Read-only.

usage: reward_timescales.py <outputs dir>
The learner uses error = reward - EMA_2s(reward), normalized by its 2 s RMS; exploration noise has a 100 ms
correlation time and the eligibility trace 100 ms. Fast reward variation is what a 100 ms perturbation can explain.
"""
import glob
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
DT = 5.0


def ema(x, tau_ms):
    a = 1 - np.exp(-DT / tau_ms)
    y = np.empty_like(x)
    s = x[0]
    for i, v in enumerate(x):
        s += a * (v - s)
        y[i] = s
    return y


rows = []
for path in sorted(glob.glob(str(OUT / "**" / "training-observations.npz"), recursive=True)):
    z = np.load(path)
    r = z["reward"].astype(float)[10:]  # learning is active from sample 10
    total = r.var()
    err2s = r - ema(r, 2000.)  # what the learner uses (before normalization)
    fast = r - ema(r, 200.)    # variation faster than ~200 ms
    vfast = r - ema(r, 50.)
    lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    rows.append((Path(path).parent.name, total, err2s.var(), fast.var(), vfast.var(), lag1, r.mean()))
print(f"{'run':36s} {'var(r)':>8s} {'var err2s':>10s} {'var >5Hz':>9s} {'var >20Hz':>10s} {'share >5Hz':>11s} {'share >20Hz':>12s} {'lag-1 corr':>10s}")
for name, tot, e2, f, vf, l1, mean in rows:
    print(f"{name:36s} {tot:8.4f} {e2:10.4f} {f:9.4f} {vf:10.4f} {f / tot:11.1%} {vf / tot:12.1%} {l1:10.3f}")
arr = np.array([[r[1], r[3], r[4], r[5]] for r in rows])
print(f"\nmedian over {len(rows)} births: var(r) {np.median(arr[:, 0]):.4f}; share of reward variance faster than ~5 Hz: "
      f"{np.median(arr[:, 1] / arr[:, 0]):.1%}; faster than ~20 Hz: {np.median(arr[:, 2] / arr[:, 0]):.1%}")

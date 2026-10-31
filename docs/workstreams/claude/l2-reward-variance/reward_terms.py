"""Which terms of the v8 reward carry its fast (non-credit) variance? Offline, from stored observations. Read-only.

Rebuilds the unclipped v8 reward from the core's own result fields per control step (and checks that it equals the
core's reward after clipping), then splits its variance by term and by timescale.

usage: reward_terms.py <core src> <run dir> [<run dir> ...]      (each with training-observations.npz)
"""
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from noesis.embodied_metrics import locomotion_reward  # noqa: E402

DT = 5.0
KEYS = ("position", "up_body", "velocity", "heading", "feet_position", "foot_load")
TERMS = ("standing", "support", "progress", "effort", "slip")


def ema(x, tau_ms):
    a = 1 - np.exp(-DT / tau_ms)
    out, s = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        s += a * (v - s)
        out[i] = s
    return out


def terms_for(obs):
    n = len(obs["position"])
    rows = []
    for i in range(1, n):
        before = {k: obs[k][i - 1] for k in KEYS}
        after = {k: obs[k][i] for k in KEYS}
        dt = DT / 1000
        r = locomotion_reward(before["position"], after["position"], float(np.clip(after["up_body"][2], -1, 1)), after["velocity"],
                              dt, before["heading"], after["foot_load"], (after["feet_position"] - before["feet_position"]) / dt,
                              objective="curriculum-gripped-step-v8")
        standing = bool(r["supporting_feet"] >= 1 and .3 <= r["height_mm"] <= 1.5 and r["uprightness"] >= .7)
        sq = min(r["supporting_feet"] / 3, 1)
        t = {"standing": .4 * standing - .8 * (1 - standing), "support": .1 * standing * sq, "progress": .6 * r["grounded_progress"],
             "effort": -.05 * r["effort"], "slip": -.03 * r["stance_slip_mm_s"]}
        total = sum(t.values()) - .2 * r["obstacle_cost"]
        assert abs(np.clip(total, -1, 1) - r["reward"]) < 1e-9, (total, r["reward"])
        rows.append([t[k] for k in TERMS] + [r["reward"]])
    return np.array(rows)


all_rows = []
for d in sys.argv[2:]:
    z = np.load(f"{d}/training-observations.npz")
    obs = {k: (z[k][:, 0] if z[k].ndim > {"position": 2, "up_body": 2, "velocity": 2, "heading": 2, "feet_position": 3, "foot_load": 2}[k] else z[k]) for k in KEYS}
    rows = terms_for(obs)[9:]  # learning-active part
    all_rows.append(rows)
R = np.vstack(all_rows)
total = R[:, :-1].sum(axis=1)
print(f"{len(all_rows)} runs, {len(R)} steps; variance of the unclipped reward {total.var():.4f}")
print(f"{'term':10s} {'mean':>8s} {'var':>8s} {'share of var':>13s} {'fast share (>5 Hz)':>19s} {'fast var':>9s} {'share of fast var':>18s}")
fast_total = 0.0
fasts = {}
for k, name in enumerate(TERMS):
    x = np.concatenate([ema_part for ema_part in [rows[:, k] for rows in all_rows]])
    f = np.concatenate([rows[:, k] - ema(rows[:, k], 200.) for rows in all_rows])
    fasts[name] = f
    fast_total += f.var()
for k, name in enumerate(TERMS):
    x = R[:, k]
    f = fasts[name]
    print(f"{name:10s} {x.mean():8.4f} {x.var():8.4f} {x.var() / total.var():13.1%} {f.var() / x.var():19.1%} {f.var():9.4f} {f.var() / fast_total:18.1%}")
fast_sum = np.concatenate([rows[:, :-1].sum(axis=1) - ema(rows[:, :-1].sum(axis=1), 200.) for rows in all_rows])
print(f"total fast variance {fast_sum.var():.4f} = {fast_sum.var() / total.var():.1%} of the total")
print("correlation of the terms' fast parts:")
names = list(fasts)
for a in range(len(names)):
    print("  ", names[a].ljust(9), " ".join(f"{np.corrcoef(fasts[names[a]], fasts[names[b]])[0, 1]:+.2f}" for b in range(len(names))))

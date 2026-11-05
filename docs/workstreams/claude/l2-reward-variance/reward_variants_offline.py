"""Offline variance of candidate reward variants on stored trajectories (no simulation, no learning).

Variants (all keep every threshold and the progress/slip/effort terms except where stated):
  v8        : the current reward.
  smooth    : the boolean `standing` in the standing/support terms replaced by the core's continuous `standing_quality`
              (posture ramp x support, already part of the v3-v7 family).
  smooth+ema: smooth, with the per-step forward speed smoothed by a causal 50 ms EMA before the progress credit
              (adds about 50 ms of lag).
usage: reward_variants_offline.py <core src> <run dir> [...]
"""
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from noesis.embodied_metrics import locomotion_reward  # noqa: E402

DT, KEYS = 5.0, ("position", "up_body", "velocity", "heading", "feet_position", "foot_load")


def ema(x, tau_ms):
    a = 1 - np.exp(-DT / tau_ms)
    out, s = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        s += a * (v - s)
        out[i] = s
    return out


def run_terms(d):
    z = np.load(f"{d}/training-observations.npz")
    obs = {k: (z[k][:, 0] if z[k].ndim > {"position": 2, "up_body": 2, "velocity": 2, "heading": 2, "feet_position": 3, "foot_load": 2}[k] else z[k]) for k in KEYS}
    rows = []
    for i in range(10, len(obs["position"])):
        b = {k: obs[k][i - 1] for k in KEYS}
        a = {k: obs[k][i] for k in KEYS}
        dt = DT / 1000
        r = locomotion_reward(b["position"], a["position"], float(np.clip(a["up_body"][2], -1, 1)), a["velocity"], dt, b["heading"],
                              a["foot_load"], (a["feet_position"] - b["feet_position"]) / dt, objective="curriculum-gripped-step-v8")
        standing = bool(r["supporting_feet"] >= 1 and .3 <= r["height_mm"] <= 1.5 and r["uprightness"] >= .7)
        rows.append((standing, r["standing_quality"], min(r["supporting_feet"] / 3, 1), r["forward_speed_mm_s"], r["grip_factor"],
                     r["grounded_progress"], r["effort"], r["stance_slip_mm_s"], r["obstacle_cost"], r["height_mm"]))
    return np.array(rows, dtype=float)


def reward(rows, variant):
    st, sqq, sup, speed, grip, prog, eff, slip, obst, _ = rows.T
    if variant == "v8":
        s = st
        progress = prog
    else:
        s = sqq
        progress = prog
        if variant == "smooth+ema":
            sp = ema(speed, 50.)
            progress = np.where(st > 0, np.clip(sp / 2., -1, 1) * grip, 0.) * (np.abs(sp) <= 30)
    return np.clip(.4 * s - .8 * (1 - s) + .1 * s * sup + .6 * progress - .05 * eff - .03 * slip - .2 * obst, -1, 1)


runs = [run_terms(d) for d in sys.argv[2:]]
out = {}
for variant in ("v8", "smooth", "smooth+ema"):
    series = [reward(r, variant) for r in runs]
    out[variant] = series
ref = np.concatenate(out["v8"])
print(f"{'variant':12s} {'variance':>9s} {'fast >5Hz':>10s} {'fast share':>11s} {'mean':>7s} {'corr to v8':>11s} {'corr to v8 (1 s smoothed)':>26s}")
for variant, series in out.items():
    x = np.concatenate(series)
    fast = np.concatenate([s - ema(s, 200.) for s in series])
    low = np.concatenate([ema(s, 1000.) for s in series])
    low0 = np.concatenate([ema(s, 1000.) for s in out["v8"]])
    print(f"{variant:12s} {x.var():9.4f} {fast.var():10.4f} {fast.var() / x.var():11.1%} {x.mean():7.3f} {np.corrcoef(x, ref)[0, 1]:11.3f} {np.corrcoef(low, low0)[0, 1]:26.4f}")

"""Prototype fixes for the v9 cycle-bonus saturation. Scratch only; the repo is not modified."""
import itertools
import sys

import numpy as np

sys.path.insert(0, r"C:/dev/noesis-claude-core/src")
from noesis.embodied_metrics import locomotion_reward  # noqa: E402

V8 = "curriculum-gripped-step-v8"
W = .1
OUT = r"C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/gpu-pilot/outputs"


def call(speed, foot_v, loads, up=1., height=.8, effort_vel=0., objective=V8):
    dt = .2
    return locomotion_reward([0, 0, height], [speed * dt, 0, height], up, [effort_vel] * 5, dt,
                             [1, 0, 0], loads, np.tile([foot_v, 0, 0], (6, 1)), 0., objective=objective)


def raw_v8(r, speed_ok=True):
    """Unclipped v8 value rebuilt from the result fields (checked against the core below)."""
    standing = r["supporting_feet"] >= 1 and .3 <= r["height_mm"] <= 1.5 and r["uprightness"] >= .7
    sq = min(r["supporting_feet"] / 3, 1)
    return (.4 * standing - .8 * (1 - standing) + .1 * standing * sq + .6 * r["grounded_progress"]
            - .05 * r["effort"] - .03 * r["stance_slip_mm_s"] - .2 * r["obstacle_cost"]), standing


def options(raw, standing, grip, credit):
    bonus = credit * grip * standing
    clip = lambda x: float(np.clip(x, -1, 1))  # noqa: E731
    return {
        "current (v8 + .1 b, clip)": clip(raw + W * bonus),
        "A scale v8 by .9": clip((1 - W) * raw + W * bonus),
        "B scale only positive part": clip(raw - W * max(raw, 0) + W * bonus),
        "C cap v8 at .9 then add": clip(min(raw, 1 - W) + W * bonus),
    }


rows = []
grid = itertools.product([0., .5, 1., 1.5, 2., 3.], [0., 2., 10., 50.], [np.ones(6), np.r_[np.ones(4), 0, 0]],
                         [0., .5, 1.])
for speed, foot_v, loads, credit in grid:
    r = call(speed, foot_v, loads)
    raw, standing = raw_v8(r)
    assert abs(np.clip(raw, -1, 1) - r["reward"]) < 1e-9, (raw, r["reward"])  # rebuild matches the core
    rows.append((speed, foot_v, credit, raw, r["grip_factor"], standing, r["reward"]))
print("rebuilt v8 formula matches the core on", len(rows), "grid points")

# 1. how much of the bonus survives, per option, over states where credit can legally apply
names = list(options(0, 1, 1, 1))
kept = {n: [] for n in names}
for speed, foot_v, credit, raw, grip, standing, _ in rows:
    if credit != 1. or not standing:
        continue
    base = options(raw, standing, grip, 0.)
    full = options(raw, standing, grip, 1.)
    ideal = W * grip  # what the bonus is worth if never clipped
    for n in names:
        kept[n].append(0. if ideal == 0 else (full[n] - base[n]) / ideal)
print("\nfraction of the intended bonus actually delivered (credit=1, standing states)")
for n in names:
    k = np.array(kept[n])
    print(f"  {n:30s} mean {k.mean():.3f}  min {k.min():.3f}  states below .95: {(k < .95).sum()}/{len(k)}")

# 2. ordering of states without credit must not change (the learner only sees reward order/scale)
base_vals = {n: [] for n in names}
for speed, foot_v, credit, raw, grip, standing, _ in rows:
    if credit == 0.:
        for n in names:
            base_vals[n].append(options(raw, standing, grip, 0.)[n])
ref = np.array(base_vals["current (v8 + .1 b, clip)"])
print("\nrank agreement with plain v8 when credit=0 (Kendall-like: share of pairs ordered the same)")
for n in names:
    v = np.array(base_vals[n])
    a = np.sign(ref[:, None] - ref[None, :])
    b = np.sign(v[:, None] - v[None, :])
    mask = a != 0
    print(f"  {n:30s} same order {np.mean(a[mask] == b[mask]):.3f}  max abs change {np.abs(v - ref).max():.3f}")

# 3. exploit check: can credit make a worse physical state outrank a better one?
print("\nbonus vs progress trade-off (credit=1 at 1 mm/s vs credit=0 at higher speed, planted feet, standing)")
for n in names:
    def val(speed, credit):
        r = call(speed, 0., np.ones(6))
        raw, standing = raw_v8(r)
        return options(raw, standing, r["grip_factor"], credit)[n]
    slow_credit = val(1., 1.)
    out = [(s, round(val(s, 0.), 3)) for s in (1.2, 1.5, 2.0)]
    beaten = [s for s, v in out if v <= slow_credit]
    print(f"  {n:30s} 1 mm/s+credit {slow_credit:.3f}; no-credit rewards {out}; outranked by credit up to speed {max(beaten) if beaten else 'none'}")

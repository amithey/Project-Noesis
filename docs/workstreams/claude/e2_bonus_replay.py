"""Replay stored observations through the v9 tracker/reward: how much cycle bonus reaches learning?

Counterfactual only: the stored runs were trained with v7/v8, so this says what v9 WOULD have paid on
those trajectories, not what a v9-trained fly does. Read-only.
"""
import glob
import os
import sys

import numpy as np

sys.path[:0] = [r"C:/dev/noesis-claude-core/src", sys.argv[1]]  # core with v9; scratch copy of Gemini's gait metrics
from noesis.embodied_metrics import LegCycleTracker, locomotion_reward  # noqa: E402
from noesis_gpu.gait_metrics import extract_leg_step_cycles  # noqa: E402

OUT = r"C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/gpu-pilot/outputs"
DT = .005
V8, V9 = "curriculum-gripped-step-v8", "curriculum-gripped-cycle-v9"


def load(path):
    z = np.load(path)
    out = {}
    for key in ("position", "up_body", "velocity", "heading", "feet_position", "foot_load"):
        a = z[key]
        out[key] = a[:, 0] if a.ndim > {"position": 2, "up_body": 2, "velocity": 2, "heading": 2,
                                        "feet_position": 3, "foot_load": 2}[key] else a
    return out


def replay(obs, skip=10):
    tracker = LegCycleTracker(dt_seconds=DT)
    n = len(obs["position"])
    intended = delivered = 0.
    credit_steps = lost_steps = steps = 0
    credit_sum = 0.
    cycles = np.zeros(6, int)
    for i in range(1, n):
        b = {k: v[i - 1] for k, v in obs.items()}
        a = {k: v[i] for k, v in obs.items()}
        dt = DT
        heading = a["heading"]
        h = heading[:2] / max(np.linalg.norm(heading[:2]), 1e-12)
        speed = float((a["position"][:2] - b["position"][:2]) @ h / dt)
        height = a["position"][2]
        standing = bool(a["up_body"][2] >= .7 and .3 <= height <= 1.5 and (a["foot_load"] > 1e-5).any())
        credit = tracker.update(a["foot_load"], speed, standing)
        cycles += tracker.cycles[-1]
        if i < skip:
            continue
        args = (b["position"], a["position"], float(np.clip(a["up_body"][2], -1, 1)), a["velocity"], dt,
                b["heading"], a["foot_load"], (a["feet_position"] - b["feet_position"]) / dt)
        r8 = locomotion_reward(*args, objective=V8)
        r9 = locomotion_reward(*args, objective=V9, cycle_credit=credit)
        steps += 1
        if credit > 0:
            credit_steps += 1
            credit_sum += credit
            want = .1 * credit * r9["grip_factor"] * standing
            intended += want
            delivered += r9["reward"] - r8["reward"]
            lost_steps += (r9["reward_unclipped"] > 1)
    return steps, credit_steps, credit_sum, intended, delivered, lost_steps, cycles


def gemini_cycles(obs, skip=10):
    counts = []
    for leg in range(6):
        c = extract_leg_step_cycles(obs["foot_load"][skip:, leg], obs["feet_position"][skip:, leg], DT * 1000)
        counts.append(c["step_count"])
    return np.array(counts)


paths = []
for pattern in ("gripped-v8-00*-train-*/training-observations.npz", "gripped-v8-00*-candidate-*/observations.npz",
                "standing-001-candidate-*/observations.npz"):
    paths += sorted(glob.glob(f"{OUT}/{pattern}"))
print(f"{'run':34s} {'steps':>5s} {'credit>0':>8s} {'meanC':>6s} {'intended':>9s} {'delivered':>9s} {'kept':>6s} {'clip-lost':>9s} | cycles causal vs Gemini")
tot_i = tot_d = 0.
for p in paths:
    name = os.path.basename(os.path.dirname(p))
    obs = load(p)
    steps, cs, csum, intended, delivered, lost, cyc = replay(obs)
    g = gemini_cycles(obs)
    kept = delivered / intended if intended > 0 else float("nan")
    tot_i += intended
    tot_d += delivered
    print(f"{name:34s} {steps:5d} {cs / steps:8.1%} {csum / max(cs, 1):6.2f} {intended:9.3f} {delivered:9.3f} {kept:6.1%} {lost:9d} | "
          f"{cyc.sum():3d} vs {g.sum():3d}  per-leg diff {np.abs(cyc - g).max()}")
print(f"\nall runs: intended {tot_i:.3f}, delivered {tot_d:.3f}, kept {tot_d / tot_i:.1%}" if tot_i else "\nno bonus paid on any run")

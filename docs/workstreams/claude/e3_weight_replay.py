"""How would a larger cycle-bonus weight behave on the stored E2 training trajectories? Counterfactual, read-only."""
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["C:/dev/noesis-claude-core/src", "C:/dev/noesis-claude-flygym2/gpu-pilot/src"]
from noesis_gpu.learning import CycleScoring  # noqa: E402

OUT = Path(sys.argv[1])
KEYS = ("position", "up_body", "velocity", "heading", "feet_position", "foot_load")
WEIGHTS = (.1, .3, .5, 1.)
print(f"{'birth':12s} {'weight':>6s} {'mean bonus/step':>16s} {'bonus/reward-std':>17s} {'clipped steps':>14s} {'delivered':>10s}")
for seed in (1001, 1000, 1002):
    z = np.load(OUT / f"train-v9-{seed}" / "training-observations.npz")
    obs = {k: z[k] for k in KEYS}
    scorer = CycleScoring()
    rows = []
    for i in range(1, len(obs["position"])):
        b = {k: v[i - 1] for k, v in obs.items()}
        a = {k: v[i] for k, v in obs.items()}
        r = scorer(b, a, 5.)
        # v8 part before clipping: remove the weight-0.1 bonus from the v9 unclipped value
        unit = r["cycle_credit"] * r["grip_factor"] * float(a["up_body"][2] >= .7 and .3 <= a["position"][2] <= 1.5
                                                           and (a["foot_load"] > 1e-5).any())
        rows.append((r["reward_unclipped"] - .1 * unit, unit))
    base, unit = np.array(rows).T
    v8 = np.clip(base, -1, 1)
    for w in WEIGHTS:
        v9 = np.clip(base + w * unit, -1, 1)
        intended = (w * unit).sum()
        delivered = (v9 - v8).sum()
        print(f"v9-{seed:<8d} {w:6.1f} {(v9 - v8).mean():16.4f} {(v9 - v8).std() / max(v8.std(), 1e-9):17.3f} "
              f"{((base + w * unit) > 1).mean():14.1%} {delivered / max(intended, 1e-9):10.1%}")
    print(f"   reward std (v8 part) {v8.std():.3f}, mean {v8.mean():.3f}, steps with credit {(unit > 0).mean():.1%}")

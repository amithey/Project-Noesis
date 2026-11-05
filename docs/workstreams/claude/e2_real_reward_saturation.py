"""How often does v8 actually sit near the +1 clip in the real training/frozen data? Read-only."""
import glob
import os

import numpy as np

OUT = r"C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/gpu-pilot/outputs"
for path in sorted(glob.glob(OUT + "/gripped-v8-00*-train-*/training-observations.npz")):
    z = np.load(path)
    r = z["reward"][z["learning_active"].astype(bool)] if z["learning_active"].any() else z["reward"]
    pos = z["position"]
    head = z["heading"]
    step = np.diff(pos[:, :2], axis=0)
    speed = np.einsum("ij,ij->i", step, head[:-1, :2] / np.maximum(np.linalg.norm(head[:-1, :2], axis=1, keepdims=True), 1e-9)) / .005
    print(os.path.basename(os.path.dirname(path))[:28],
          f"n={len(r)} reward mean {r.mean():.3f} p90 {np.percentile(r, 90):.3f} max {r.max():.3f} "
          f">=0.9: {(r >= .9).mean():.3%} >=0.99: {(r >= .99).mean():.3%} | fwd speed mm/s p50 {np.median(speed):.2f} p95 {np.percentile(speed, 95):.2f}")

"""Why do the causal tracker and Gemini's offline step counts differ? Definition or causality?"""
import sys

import numpy as np

sys.path[:0] = [r"C:/dev/noesis-claude-core/src", sys.argv[1]]
from noesis.embodied_metrics import LegCycleTracker  # noqa: E402
from noesis_gpu.gait_metrics import debounce_contact, extract_leg_step_cycles  # noqa: E402

OUT = r"C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/gpu-pilot/outputs"
for name in ("gripped-v8-003-candidate-10007", "gripped-v8-004-candidate-10006", "standing-001-candidate-10006"):
    z = np.load(f"{OUT}/{name}/observations.npz")
    load = z["foot_load"]
    load = load[:, 0] if load.ndim == 3 else load
    load, pos = load[10:], z["feet_position"][10:]
    pos = pos[:, 0] if pos.ndim == 4 else pos
    tracker = LegCycleTracker()
    causal = np.zeros(6, int)
    for row in load:
        tracker.update(row, 2., True)
        causal += tracker.cycles[-1]
    gem, td_offline, lift_td = [], [], []
    for leg in range(6):
        gem.append(extract_leg_step_cycles(load[:, leg], pos[:, leg], 5.)["step_count"])
        d = debounce_contact(load[:, leg] > 1e-5, 5.)
        diff = np.diff(d.astype(int))
        td = np.flatnonzero(diff == 1)
        lo = np.flatnonzero(diff == -1)
        td_offline.append(len(td))
        # lift -> touchdown pairs: touchdowns that have a liftoff before them
        lift_td.append(int(sum(1 for t in td if (lo < t).any())))
    print(name)
    print("  Gemini step_count (cycles)      ", gem, sum(gem))
    print("  offline debounced touchdowns    ", td_offline, sum(td_offline))
    print("  offline lift->touchdown pairs   ", lift_td, sum(lift_td))
    print("  causal tracker lift->touchdown  ", causal.tolist(), int(causal.sum()))

"""Aligned tracker vs Gemini's offline count: exactness check on stored runs and random signals."""
import glob
import sys
import time

import numpy as np

sys.path[:0] = [r"C:/dev/noesis-claude-core/src", sys.argv[1]]
from noesis.embodied_metrics import LegCycleTracker  # noqa: E402
from noesis.embodied_metrics import debounce_contact as core_debounce  # noqa: E402
from noesis_gpu.gait_metrics import debounce_contact as gem_debounce  # noqa: E402

OUT = r"C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/gpu-pilot/outputs"
rng = np.random.default_rng(0)

# 1. the copied function equals Gemini's on random contact signals, incl. flutter
bad = 0
for _ in range(3000):
    n = int(rng.integers(1, 120))
    sig = rng.random(n) < rng.uniform(.1, .9)
    for ms in (5., 10., 15., 20., 30.):
        bad += not np.array_equal(core_debounce(sig, 5., ms), gem_debounce(sig, 5., ms))
print("debounce_contact core == Gemini on 15000 random cases:", bad == 0, f"(mismatches {bad})")


def offline_event_times(raw):
    d = gem_debounce(raw, 5., 15.)
    diff = np.diff(d.astype(int))
    tds = np.flatnonzero(diff == 1) + 1
    los = np.flatnonzero(diff == -1) + 1
    return [int(t) for t in tds if (los < t).any()]


def tracker_event_times(load, delay_ms):
    tr = LegCycleTracker(delay_ms=delay_ms)
    ev = [[] for _ in range(6)]
    for i, row in enumerate(load):
        tr.update(row, 1., True)
        for leg in np.flatnonzero(tr.cycles[-1]):
            ev[leg].append(i - tr.delay)
    return ev


runs = []
for pattern in ("gripped-v8-00*-train-*/training-observations.npz", "gripped-v8-00*-candidate-*/observations.npz",
                "standing-001-candidate-*/observations.npz"):
    runs += sorted(glob.glob(f"{OUT}/{pattern}"))
data = []
for p in runs:
    z = np.load(p)
    load = z["foot_load"]
    data.append((load[:, 0] if load.ndim == 3 else load)[10:])

print("\nstored runs: events identical to Gemini's offline count (exact sample times)")
print(f"{'delay ms':>8s} {'Gemini events':>14s} {'tracker events':>15s} {'exact matches':>14s} {'missing':>8s} {'extra':>6s}")
for delay in (0., 10., 15., 20., 25., 30., 40.):
    g = t = m = 0
    for load in data:
        ev = tracker_event_times(load, delay)
        for leg in range(6):
            off = offline_event_times(load[:, leg] > 1e-5)
            # events read out after the end of the signal cannot appear yet: ignore offline events in the last delay samples
            horizon = len(load) - 1 - round(delay / 5.)
            off_vis = [x for x in off if x <= horizon]
            g += len(off_vis)
            t += len(ev[leg])
            m += len(set(off_vis) & set(ev[leg]))
    print(f"{delay:8.0f} {g:14d} {t:15d} {m:14d} {g - m:8d} {t - m:6d}")

load = data[0]
tr = LegCycleTracker()
start = time.perf_counter()
for row in load[:500]:
    tr.update(row, 1., True)
print(f"\nupdate cost: {(time.perf_counter() - start) / 500 * 1e6:.0f} us per control step (6 legs)")

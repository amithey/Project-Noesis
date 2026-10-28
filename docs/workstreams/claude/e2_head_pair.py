"""Event-by-event comparison of the core cycle tracker (head given by sys.argv) with Gemini's gait_metrics head.

usage: head_pair.py <dir with noesis_gpu/gait_metrics.py of Gemini's head> <label> <out.json>
The core under test is whatever C:/dev/noesis-claude-core/src currently holds.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path[:0] = [r"C:/dev/noesis-claude-core/src", sys.argv[1]]
from noesis.embodied_metrics import LegCycleTracker  # noqa: E402
from noesis_gpu.gait_metrics import debounce_contact, extract_leg_step_cycles  # noqa: E402

OUT = r"C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/gpu-pilot/outputs"
WARMUP = 10  # samples (50 ms) dropped before analysis, as in the evaluation
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def gemini_events(raw):
    d = debounce_contact(raw, 5., 15.)
    diff = np.diff(d.astype(int))
    tds = np.flatnonzero(diff == 1) + 1
    los = np.flatnonzero(diff == -1) + 1
    return [int(t) for t in tds if (los < t).any()]


def tracker_events(load):
    tracker = LegCycleTracker()
    events = [[] for _ in range(6)]
    for i, row in enumerate(load):
        tracker.update(row, 1., True)
        for leg in np.flatnonzero(tracker.cycles[-1]):
            events[leg].append(i)  # same sample as the evaluation touchdown (no read-out delay)
    return events, 0


paths = []
for pattern in ("gripped-v8-00*-train-*/training-observations.npz", "gripped-v8-00*-candidate-*/observations.npz",
                "standing-001-candidate-*/observations.npz",
                "exploration-comparison-20261028-2100/frozen-*/observations.npz"):
    paths += sorted(glob.glob(f"{OUT}/{pattern}"))
rows, tot = [], dict(gemini=0, tracker=0, exact=0, step_count=0, samples=0)
for p in paths:
    z = np.load(p)
    load = z["foot_load"]
    load = (load[:, 0] if load.ndim == 3 else load)[WARMUP:]
    pos = z["feet_position"]
    pos = (pos[:, 0] if pos.ndim == 4 else pos)[WARMUP:]
    trk, delay = tracker_events(load)
    name = os.path.relpath(os.path.dirname(p), OUT).replace("\\", "/")
    for leg in range(6):
        g = gemini_events(load[:, leg] > 1e-5)
        step_count = extract_leg_step_cycles(load[:, leg], pos[:, leg], 5.)["step_count"]
        visible = [t for t in g if t <= len(load) - 1]
        exact = len(set(visible) & set(trk[leg]))
        rows.append(dict(run=name, leg=LEGS[leg], gemini_events=len(g), gemini_step_count=step_count,
                         gemini_visible=len(visible), tracker=len(trk[leg]), exact=exact,
                         missing=len(visible) - exact, extra=len(trk[leg]) - exact,
                         gemini_times=visible, tracker_times=trk[leg]))
        tot["gemini"] += len(visible)
        tot["tracker"] += len(trk[leg])
        tot["exact"] += exact
        tot["step_count"] += step_count
    tot["samples"] += load.shape[0] * 6
    rows[-1]["input_sha256"] = sha(p)
gem_file = os.path.join(sys.argv[1], "noesis_gpu", "gait_metrics.py")
core_file = r"C:/dev/noesis-claude-core/src/noesis/embodied_metrics.py"
summary = dict(label=sys.argv[2], tracker_delay_samples=delay, warmup_samples=WARMUP, runs=len(paths), totals=tot,
               gemini_gait_metrics_sha256=sha(gem_file), core_embodied_metrics_sha256=sha(core_file))
json.dump(dict(summary=summary, rows=rows), open(sys.argv[3], "w"), indent=1)
print(sys.argv[2], json.dumps(summary))
by_run = {}
for r in rows:
    d = by_run.setdefault(r["run"], [0, 0, 0])
    d[0] += r["gemini_visible"]
    d[1] += r["tracker"]
    d[2] += r["exact"]
print(f"{'run':62s} {'Gemini':>6s} {'tracker':>7s} {'exact':>6s}")
for k, v in by_run.items():
    print(f"{k:62s} {v[0]:6d} {v[1]:7d} {v[2]:6d}")

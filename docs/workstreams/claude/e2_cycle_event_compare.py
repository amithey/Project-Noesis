"""Event-by-event, leg-by-leg comparison of the causal LegCycleTracker and Gemini's offline cycle count."""
import glob
import os
import sys

import numpy as np

sys.path[:0] = [r"C:/dev/noesis-claude-core/src", sys.argv[1]]
from noesis.embodied_metrics import LegCycleTracker  # noqa: E402
from noesis_gpu.gait_metrics import debounce_contact, extract_leg_step_cycles  # noqa: E402

OUT = r"C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/gpu-pilot/outputs"
MIN = 3  # 15 ms at 5 ms
TOL = 4  # samples: causal detection lags the true flip by MIN-1, plus slack


def longest_false(raw, a, b):
    best = cur = 0
    for v in raw[a:b]:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


def offline_events(raw):
    """Touchdowns that follow a lift in the offline-debounced signal, with how robust the lift was."""
    d = debounce_contact(raw, 5., 15.)
    diff = np.diff(d.astype(int))
    tds = np.flatnonzero(diff == 1) + 1
    los = np.flatnonzero(diff == -1) + 1
    events = []
    for t in tds:
        prior = los[los < t]
        if not len(prior):
            continue
        lift_start = prior[-1]
        sustained = longest_false(raw, lift_start, t) >= MIN  # raw has a real >=15 ms gap inside the lift
        events.append((int(t), bool(sustained)))
    return events


def causal_events(load):
    tracker = LegCycleTracker()
    times = [[] for _ in range(6)]
    for i, row in enumerate(load):
        tracker.update(row, 1., True)
        for leg in np.flatnonzero(tracker.cycles[-1]):
            times[leg].append(i - (MIN - 1))  # first sample of the persistent flip
    return times


def load_run(path):
    z = np.load(path)
    load = z["foot_load"]
    load = load[:, 0] if load.ndim == 3 else load
    return load[10:]


runs = []
for pattern in ("gripped-v8-00*-train-*/training-observations.npz", "gripped-v8-00*-candidate-*/observations.npz",
                "standing-001-candidate-*/observations.npz"):
    runs += sorted(glob.glob(f"{OUT}/{pattern}"))

tot = dict(gem=0, causal=0, matched=0, off_sustained_missed=0, off_flutter_missed=0, causal_only=0,
           off_sustained=0, off_flutter=0)
print(f"{'run':34s} {'Gemini':>6s} {'causal':>6s} {'matched':>7s} {'Gem-only sustained':>18s} {'Gem-only flutter':>16s} {'causal-only':>11s}")
per_leg = np.zeros((6, 3), int)
for p in runs:
    load = load_run(p)
    cau = causal_events(load)
    row = dict(gem=0, causal=0, matched=0, ss=0, ff=0, co=0)
    for leg in range(6):
        raw = load[:, leg] > 1e-5
        off = offline_events(raw)
        used = set()
        matched = 0
        for t, sustained in off:
            hit = next((k for k, c in enumerate(cau[leg]) if k not in used and abs(c - t) <= TOL), None)
            if hit is None:
                if sustained:
                    row["ss"] += 1
                else:
                    row["ff"] += 1
            else:
                used.add(hit)
                matched += 1
            tot["off_sustained" if sustained else "off_flutter"] += 1
        row["gem"] += len(off)
        row["causal"] += len(cau[leg])
        row["matched"] += matched
        row["co"] += len(cau[leg]) - len(used)
        per_leg[leg] += (len(off), len(cau[leg]), matched)
    print(f"{os.path.basename(os.path.dirname(p)):34s} {row['gem']:6d} {row['causal']:6d} {row['matched']:7d} "
          f"{row['ss']:18d} {row['ff']:16d} {row['co']:11d}")
    tot["gem"] += row["gem"]
    tot["causal"] += row["causal"]
    tot["matched"] += row["matched"]
    tot["off_sustained_missed"] += row["ss"]
    tot["off_flutter_missed"] += row["ff"]
    tot["causal_only"] += row["co"]

print("\nall runs:", tot)
g = tot["gem"]
print(f"Gemini events {g}: {tot['off_sustained']} follow a lift with a real >=15 ms raw gap, "
      f"{tot['off_flutter']} follow a lift made only of alternating <15 ms flips ({tot['off_flutter'] / g:.1%})")
print(f"causal tracker found {tot['causal']} ({tot['causal'] / g:.1%} of Gemini); matched {tot['matched']} within +-{TOL} samples")
print(f"unmatched Gemini events: {tot['off_sustained_missed']} sustained, {tot['off_flutter_missed']} flutter-derived; "
      f"causal-only events: {tot['causal_only']}")
print("\nper leg (all runs): Gemini / causal / matched")
for leg, name in enumerate(("LF", "LM", "LH", "RF", "RM", "RH")):
    print(f"  {name}: {per_leg[leg][0]:4d} / {per_leg[leg][1]:4d} / {per_leg[leg][2]:4d}")

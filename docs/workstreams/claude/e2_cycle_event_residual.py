import glob, os, sys
import numpy as np
sys.path[:0] = [r"C:/dev/noesis-claude-core/src", sys.argv[1]]
exec(open(sys.argv[2], encoding="utf-8").read().split("runs = []")[0])  # reuse defs from event_compare
runs = []
for pattern in ("gripped-v8-00*-train-*/training-observations.npz", "gripped-v8-00*-candidate-*/observations.npz", "standing-001-candidate-*/observations.npz"):
    runs += sorted(glob.glob(f"{OUT}/{pattern}"))
off_only, cau_only = [], []
for p in runs:
    load = load_run(p)
    cau = causal_events(load)
    for leg in range(6):
        raw = load[:, leg] > 1e-5
        off = offline_events(raw)
        used = set()
        for t, s in off:
            hit = next((k for k, c in enumerate(cau[leg]) if k not in used and abs(c - t) <= TOL), None)
            if hit is None:
                near = min((abs(c - t) for c in cau[leg]), default=999)
                d = debounce_contact(raw, 5., 15.)
                lo = np.flatnonzero(np.diff(d.astype(int)) == -1) + 1
                lift_len = t - lo[lo < t][-1]
                # raw run lengths inside the lift
                off_only.append((near, int(lift_len), longest_false(raw, int(t - lift_len), t)))
            else:
                used.add(hit)
        for k, c in enumerate(cau[leg]):
            if k not in used:
                near = min((abs(c - t) for t, _ in off), default=999)
                cau_only.append(near)
print("Gemini-only events:", len(off_only))
print("  distance to nearest causal event (samples):", np.bincount(np.minimum([o[0] for o in off_only], 40))[:41].tolist())
print("  offline lift length (samples) histogram 0..12:", np.bincount(np.minimum([o[1] for o in off_only], 12), minlength=13).tolist())
print("  longest raw False run inside the lift 0..8:", np.bincount(np.minimum([o[2] for o in off_only], 8), minlength=9).tolist())
print("causal-only events:", len(cau_only), " distance to nearest Gemini event:", np.bincount(np.minimum(cau_only, 40))[:41].tolist())

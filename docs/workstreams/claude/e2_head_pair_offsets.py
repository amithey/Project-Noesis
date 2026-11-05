import json, sys
import numpy as np
d = json.load(open(sys.argv[1]))
offs, unm_g, unm_t, n_g, n_t = [], 0, 0, 0, 0
for r in d["rows"]:
    g, t = list(r["gemini_times"]), list(r["tracker_times"])
    n_g += len(g); n_t += len(t)
    used = set()
    for x in g:
        best = min(((abs(y - x), k) for k, y in enumerate(t) if k not in used), default=None)
        if best and best[0] <= 12:
            used.add(best[1]); offs.append(t[best[1]] - x)
        else:
            unm_g += 1
    unm_t += len(t) - len(used)
offs = np.array(offs)
print("matched", len(offs), "of Gemini", n_g, "/ tracker", n_t, " Gemini-only", unm_g, " tracker-only", unm_t)
print("tracker_time - gemini_time histogram:", {int(k): int(v) for k, v in zip(*np.unique(offs, return_counts=True))})

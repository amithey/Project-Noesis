"""Apply the preregistered rule of protocol.md to the noise-control child (usage: analyse_control.py <hub> <out.json>)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["gpu-pilot/src"]
from noesis_gpu.evaluate import summarize  # noqa: E402

hub, out = Path(sys.argv[1]), sys.argv[2]
t1b, ctl = hub / "gpu-pilot/outputs/duration-t1b-guarded-20261031", hub / "gpu-pilot/outputs/credit-noise-control-20261031"
SEEDS = (10006, 10007, 10010, 10011)
rows = {}
for s in SEEDS:
    pair = {}
    for name, d in (("parent", f"frozen-parent-{s}"), ("real", f"frozen-child25-{s}"), ("control", None)):
        path = ctl / f"frozen-control-child25-{s}" if name == "control" else t1b / d
        if not path.exists() and name == "parent":
            path = None
        pair[name] = summarize(path)[1] if path else None
    rows[s] = pair
# parents are stored in cases.json of T1B
cases = json.loads((t1b / "cases.json").read_text(encoding="utf-8-sig"))
for s in SEEDS:
    if rows[s]["parent"] is None:
        rows[s]["parent"] = cases["parent"][str(s)]


def pick(c):
    return {k: c[k] for k in ("standing_fraction", "mean_stance_slip_mm_s", "heading_forward_distance_mm",
                              "net_forward_displacement_mm", "total_leg_steps", "locomotion_regime")}


table = {s: {k: pick(v) for k, v in rows[s].items()} for s in SEEDS}
dh = [rows[s]["real"]["heading_forward_distance_mm"] - rows[s]["control"]["heading_forward_distance_mm"] for s in SEEDS]
dn = [rows[s]["real"]["net_forward_displacement_mm"] - rows[s]["control"]["net_forward_displacement_mm"] for s in SEEDS]
stand_ok = sum(rows[s]["real"]["standing_fraction"] >= rows[s]["control"]["standing_fraction"] for s in SEEDS)
slip_ok = sum(rows[s]["real"]["mean_stance_slip_mm_s"] <= rows[s]["control"]["mean_stance_slip_mm_s"] for s in SEEDS)
real_better = bool(np.median(dh) >= .25 and np.median(dn) >= .25 and stand_ok >= 3 and slip_ok >= 3)
ctl_better = bool(np.median(dh) <= -.25 and np.median(dn) <= -.25 and stand_ok <= 1 and slip_ok <= 1)
# structural
r_tr = np.load(t1b / "train25/exploration-log.npz")
c_tr = np.load(ctl / "train25-control/exploration-log.npz")
drive_equal = bool(np.array_equal(r_tr["drive"], c_tr["drive"]))
credit_equal = bool(np.array_equal(r_tr["credit"], c_tr["credit"]))
applied_diff = "applied_credit" in c_tr.files
res = {"per_seed": table, "median_heading_real_minus_control_mm": float(np.median(dh)),
       "median_net_real_minus_control_mm": float(np.median(dn)), "standing_real_not_lower_count": int(stand_ok),
       "slip_real_not_higher_count": int(slip_ok), "real_better_than_control": real_better,
       "control_better_than_real": ctl_better, "verdict": "REAL_BETTER" if real_better else ("CONTROL_BETTER" if ctl_better else "INDISTINGUISHABLE"),
       "drive_arrays_equal_to_T1B": drive_equal, "real_credit_logged_equal_to_T1B": credit_equal,
       "applied_credit_present": applied_diff, "drive_shape": list(r_tr["drive"].shape)}
Path(out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
for s in SEEDS:
    print(s, {k: (round(v["standing_fraction"], 3), round(v["mean_stance_slip_mm_s"], 2), round(v["heading_forward_distance_mm"], 3),
                  round(v["net_forward_displacement_mm"], 3), v["locomotion_regime"]) for k, v in table[s].items()})
print({k: res[k] for k in res if k != "per_seed"})

"""Preregistered replication rule (protocol.md addendum 286af64). usage: analyse_replication.py <hub> <out.json>"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["gpu-pilot/src"]
from noesis_gpu.evaluate import summarize  # noqa: E402

hub, out = Path(sys.argv[1]), sys.argv[2]
O = hub / "gpu-pilot/outputs"
PS = (10006, 10007, 10010, 10011)
rep = O / "credit-noise-replication-20261031"
dirs = {
    1000: {"real": lambda p: O / f"duration-t1b-guarded-20261031/frozen-child25-{p}",
           "control": lambda p: O / f"credit-noise-control-20261031/frozen-control-child25-{p}"},
    1001: {a: (lambda p, a=a: rep / f"frozen-{a}-1001-{p}") for a in ("real", "control")},
    1002: {a: (lambda p, a=a: rep / f"frozen-{a}-1002-{p}") for a in ("real", "control")},
}
KEYS = ("standing_fraction", "mean_stance_slip_mm_s", "heading_forward_distance_mm", "net_forward_displacement_mm", "locomotion_regime")
pairs, pool_h, pool_n = {}, [], []
for seed, arms in dirs.items():
    rows = {a: {p: summarize(f(p))[1] for p in PS} for a, f in arms.items()}
    dh = [rows["real"][p]["heading_forward_distance_mm"] - rows["control"][p]["heading_forward_distance_mm"] for p in PS]
    dn = [rows["real"][p]["net_forward_displacement_mm"] - rows["control"][p]["net_forward_displacement_mm"] for p in PS]
    st = sum(rows["real"][p]["standing_fraction"] >= rows["control"][p]["standing_fraction"] for p in PS)
    sl = sum(rows["real"][p]["mean_stance_slip_mm_s"] <= rows["control"][p]["mean_stance_slip_mm_s"] for p in PS)
    better = np.median(dh) >= .25 and np.median(dn) >= .25 and st >= 3 and sl >= 3
    worse = np.median(dh) <= -.25 and np.median(dn) <= -.25 and st <= 1 and sl <= 1
    pairs[seed] = {"verdict": "REAL_BETTER" if better else ("CONTROL_BETTER" if worse else "INDISTINGUISHABLE"),
                   "median_dheading": float(np.median(dh)), "median_dnet": float(np.median(dn)), "standing_ok": int(st), "slip_ok": int(sl),
                   "table": {a: {p: {k: rows[a][p][k] for k in KEYS} for p in PS} for a in rows}}
    pool_h += dh
    pool_n += dn
ph, pn = float(np.median(pool_h)), float(np.median(pool_n))
rb = sum(v["verdict"] == "REAL_BETTER" for v in pairs.values())
cb = sum(v["verdict"] == "CONTROL_BETTER" for v in pairs.values())
overall = "REAL_BETTER" if rb >= 2 and ph >= .25 and pn >= .25 else ("CONTROL_BETTER" if cb >= 2 and ph <= -.25 and pn <= -.25 else "INDISTINGUISHABLE")
res = {"pairs": pairs, "pooled_median_dheading": ph, "pooled_median_dnet": pn, "pooled_n": len(pool_h),
       "pooled_positive_heading": int(sum(x > 0 for x in pool_h)), "pooled_positive_net": int(sum(x > 0 for x in pool_n)),
       "real_better_pairs": rb, "control_better_pairs": cb, "overall_verdict": overall}


def weights(seed):
    p = np.load(O / "e4-exploration-20261029/train-control-1000/weights-last.npz", allow_pickle=False)["weights"].astype(float)
    r = np.load((O / "duration-t1b-guarded-20261031/train25" if seed == 1000 else rep / f"train25-real-{seed}") / "weights-last.npz",
                allow_pickle=False)["weights"].astype(float) - p
    c = np.load((O / "credit-noise-control-20261031/train25-control" if seed == 1000 else rep / f"train25-control-{seed}") / "weights-last.npz",
                allow_pickle=False)["weights"].astype(float) - p
    return {"norm_real": float(np.linalg.norm(r)), "norm_control": float(np.linalg.norm(c)),
            "cosine": float(r @ c / np.linalg.norm(r) / np.linalg.norm(c))}


res["weights"] = {str(s): weights(s) for s in dirs}
Path(out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
for s, v in pairs.items():
    print(s, v["verdict"], round(v["median_dheading"], 3), round(v["median_dnet"], 3), v["standing_ok"], v["slip_ok"])
print({k: res[k] for k in ("pooled_median_dheading", "pooled_median_dnet", "pooled_positive_heading", "pooled_positive_net", "overall_verdict", "weights")})

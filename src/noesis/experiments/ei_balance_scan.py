"""Does a different excitation/inhibition balance give the network a wide sparse, stimulus-graded regime?

The connectome is not edited: the scan multiplies all inhibitory (negative) entries of the weight matrix
by ``g_I`` at network-construction time, and sweeps spontaneous background rate. For each (g_I, background):
settle 0.5 s, measure 1 s of baseline, then 0.7 s with a moderate odour-like drive (``--n-orn`` random
olfactory receptor neurons, 80 Hz, weight ``--stim-weight``). Baseline regime comes from
``operating_point.classify``; the response is 'silent' (PN rise < 1 Hz), 'graded' (rise 1-40 Hz and under
40% of neurons active) or 'ignites'. A 'good' point is baseline-sparse AND graded; the figure of merit of a
balance is the number of consecutive background rates (on a geometric grid) that are good.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.embodiment import load_training_brain
from noesis.experiments.operating_point import classify
from noesis.experiments.sense_audit import readout_groups
from noesis.lif import LIFNetwork, LIFParams


def scale_inhibition(W, g_inh: float):
    """Copy of W with every negative entry multiplied by g_inh (>= 0). Positive entries unchanged."""
    if not np.isfinite(g_inh) or g_inh < 0:
        raise ValueError("g_inh must be finite and non-negative")
    out = W.copy()
    out.data = np.where(out.data < 0, out.data * g_inh, out.data)
    return out


def response_label(pn_rise_hz: float, active_fraction: float) -> str:
    if pn_rise_hz < 1.0:
        return "silent"
    if pn_rise_hz <= 40.0 and active_fraction < 0.4:
        return "graded"
    return "ignites"


def widest_good_run(points: list[dict]) -> int:
    """Longest run of consecutive (sorted by background) points that are baseline-sparse and graded."""
    best = run = 0
    for p in sorted(points, key=lambda q: q["bg_hz"]):
        run = run + 1 if p["good"] else 0
        best = max(best, run)
    return best


def measure_point(W, groups, orn, bg_hz, bg_w, stim_w, seed, rate_hz=80.0) -> dict:
    net = LIFNetwork(W, LIFParams(dt=0.5, da=1.0), seed=seed)
    bg = {"background_hz": bg_hz, "background_weight": bg_w}
    net.run(500.0, **bg)
    _, sp = net.run(1000.0, **bg)
    base = np.bincount(sp, minlength=W.shape[0]) / 1.0
    _, sp = net.run(700.0, orn, rate_hz, stim_w, **bg)
    stim = np.bincount(sp, minlength=W.shape[0]) / 0.7
    b = {k: float(base[v].mean()) for k, v in groups.items() if k in ("pn", "kc", "mbon", "dn")}
    b["active_fraction"] = float(np.mean(base > 0))
    s = {k: float(stim[v].mean()) for k, v in groups.items() if k in ("pn", "kc", "mbon", "dn")}
    s["active_fraction"] = float(np.mean(stim > 0))
    base_regime = classify(b["pn"], b["kc"], b["active_fraction"])
    resp = response_label(s["pn"] - b["pn"], s["active_fraction"])
    return {"bg_hz": bg_hz, "baseline": b, "stim": s, "baseline_regime": base_regime,
            "response": resp, "good": bool(base_regime == "sparse" and resp == "graded")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/selectivity-cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/ei_balance_scan.json"))
    ap.add_argument("--g-inh", default="1,1.5,2,3,4")
    ap.add_argument("--bg-hz", default="1,1.5,2,3,5,8")
    ap.add_argument("--bg-weight", type=float, default=60.0)
    ap.add_argument("--stim-weight", type=float, default=100.0)
    ap.add_argument("--n-orn", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cn, _ = load_training_brain(a.data, a.cache, "corrected")
    groups = readout_groups(cn.neurons)
    orn_all = np.flatnonzero((cn.neurons["Flow"].eq("afferent")
                              & cn.neurons["Class"].eq("olfactory_receptor_neuron")).to_numpy())
    orn = np.random.default_rng(0).choice(orn_all, size=a.n_orn, replace=False)
    report = {}
    for g in (float(x) for x in a.g_inh.split(",")):
        W = scale_inhibition(cn.W, g)
        pts = []
        for bg in (float(x) for x in a.bg_hz.split(",")):
            p = measure_point(W, groups, orn, bg, a.bg_weight, a.stim_weight, a.seed)
            pts.append(p)
            print(f"g_I {g:g} bg {bg:g} Hz: base {p['baseline_regime']:9s} PN {p['baseline']['pn']:6.1f} "
                  f"KC {p['baseline']['kc']:5.1f} act {p['baseline']['active_fraction']:.1%} | stim {p['response']:7s} "
                  f"PN {p['stim']['pn']:6.1f} act {p['stim']['active_fraction']:.1%} {'GOOD' if p['good'] else ''}",
                  flush=True)
        report[str(g)] = {"points": pts, "widest_good_run": widest_good_run(pts)}
        print(f"g_I {g:g}: widest run of good background rates = {report[str(g)]['widest_good_run']}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

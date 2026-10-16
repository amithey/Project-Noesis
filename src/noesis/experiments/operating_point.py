"""Where does spontaneous activity leave the brain: silent, sparse, or saturated?

Brain only, no stimulus. For each (background rate, background weight) pair, settle 0.5 s, then
measure 1 s. Reports mean rates of olfactory projection neurons, Kenyon cells, MBONs, descending
neurons, and the fraction of all neurons that spike. A sensory audit is only informative at an
operating point where the olfactory pathway is neither silent nor already near its ceiling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.embodiment import load_training_brain
from noesis.experiments.sense_audit import readout_groups
from noesis.lif import LIFNetwork, LIFParams


def classify(pn_hz: float, kc_hz: float, active_fraction: float) -> str:
    """Coarse regime label: 'silent', 'sparse' or 'saturated' (thresholds are heuristic)."""
    if active_fraction < 0.005:
        return "silent"
    if pn_hz > 60 or kc_hz > 20:
        return "saturated"
    return "sparse"


def measure(W, groups, bg_hz: float, bg_w: float, seed: int, settle_ms: float, measure_ms: float) -> dict:
    net = LIFNetwork(W, LIFParams(dt=0.5, da=1.0), seed=seed)
    net.run(settle_ms, background_hz=bg_hz, background_weight=bg_w)
    _, spikes = net.run(measure_ms, background_hz=bg_hz, background_weight=bg_w)
    hz = np.bincount(spikes, minlength=W.shape[0]) / (measure_ms * 1e-3)
    out = {k: float(hz[v].mean()) for k, v in groups.items() if k in ("pn", "kc", "mbon", "dn", "pam", "ppl1")}
    out["active_fraction"] = float(np.mean(hz > 0))
    out["regime"] = classify(out["pn"], out["kc"], out["active_fraction"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/sense-audit-cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/operating_point.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", default="1:60,2:60,3:60,2:90,3:90,5:60,5:90",
                    help="comma list of background_hz:background_weight")
    a = ap.parse_args()
    cn, _ = load_training_brain(a.data, a.cache, "corrected")
    groups = readout_groups(cn.neurons)
    report = {}
    for item in a.grid.split(","):
        hz, w = (float(x) for x in item.split(":"))
        r = measure(cn.W, groups, hz, w, a.seed, 500.0, 1000.0)
        report[item] = r
        print(f"bg {hz:g} Hz w{w:g}: {r['regime']:9s} PN {r['pn']:6.1f} KC {r['kc']:6.2f} MBON {r['mbon']:6.2f} "
              f"DN {r['dn']:6.2f} PPL1 {r['ppl1']:6.1f} active {r['active_fraction']:.1%}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

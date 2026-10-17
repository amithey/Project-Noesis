"""Does scaling inhibition (``g_I``) damage the leg motor and proprioceptive pathway? Brain only, no body.

Mimics the locomotion training setup: corrected connectome (no weak-edge/APL additions), LIF dt 0.5 ms with
adaptation 1.0, background 3 Hz x weight 60, leg afferents driven through ``SensoryMap`` at weight 400. All
inhibitory weights are multiplied by ``g_I`` (``ei_balance_scan.scale_inhibition``). Per g_I and seed, 0.5 s
settle then 1 s measured, in five conditions:

  base      background only
  rest      all leg afferents at a neutral standing state (moderate load)
  flex      as rest, but the left-front tibia is flexed (+0.6 rad): claw organs high, hook organs low
  extend    as rest, but the left-front tibia is extended (-0.6 rad)
  current   as rest, plus +1.5 mV/ms into the left-front tibia flexor motor neurons (exploration-sized current)

Reported: mean rate of all leg motor neurons and the fraction above 1 Hz; and the left-front tibia reflex gain =
(flexor pool minus extensor pool rate | flex) - (same | extend), plus the pool response to the injected current.
A reflex that the network no longer produces at larger g_I would show up as a gain near zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.embodiment import load_training_brain
from noesis.experiments.ei_balance_scan import scale_inhibition
from noesis.lif import LIFNetwork, LIFParams
from noesis.motor import leg_motor_table
from noesis.sensory import LEGS, LegState, SensoryMap

FLEXORS = ("tibia_flexor", "accessory_tibia_flexor")
EXTENSORS = ("tibia_extensor",)


def reflex_gain(flex_flexor: float, flex_extensor: float, ext_flexor: float, ext_extensor: float) -> float:
    """(flexor - extensor | flexed tibia) - (flexor - extensor | extended tibia), Hz."""
    return (flex_flexor - flex_extensor) - (ext_flexor - ext_extensor)


def standing_state(tibia_lf: float = 0.0, load: float = 1.0) -> dict[str, LegState]:
    return {leg: LegState(tibia=tibia_lf if leg == "LF" else 0.0, load=load) for leg in LEGS}


def run_condition(W, sensory: SensoryMap, mn: dict, name: str, seed: int, a) -> dict:
    net = LIFNetwork(W, LIFParams(dt=0.5, da=1.0), seed=seed)
    bg = {"background_hz": a.bg_hz, "background_weight": a.bg_weight}
    tibia = {"flex": 0.6, "extend": -0.6}.get(name, 0.0)
    drive_idx, rates = None, 0.0
    if name != "base":
        drive_idx, rates = sensory.idx, sensory.drive(standing_state(tibia))
    cur = {}
    if name == "current":
        cur = {"current_idx": mn["lf_flexor"], "current_mv_per_ms": a.current_mv_ms}
    net.run(a.settle_ms, drive_idx, rates, a.sensory_weight, **bg, **cur)
    _, sp = net.run(a.measure_ms, drive_idx, rates, a.sensory_weight, **bg, **cur)
    hz = np.bincount(sp, minlength=W.shape[0]) / (a.measure_ms * 1e-3)
    return {"motor_mean_hz": float(hz[mn["all"]].mean()), "motor_active": float(np.mean(hz[mn["all"]] > 1.0)),
            "lf_flexor_hz": float(hz[mn["lf_flexor"]].mean()), "lf_extensor_hz": float(hz[mn["lf_extensor"]].mean()),
            "afferent_mean_hz": float(hz[sensory.idx].mean()), "brain_active": float(np.mean(hz > 0))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/motor-cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/motor_g_inh.json"))
    ap.add_argument("--g-inh", default="1,4,12")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bg-hz", type=float, default=3.0)
    ap.add_argument("--bg-weight", type=float, default=60.0)
    ap.add_argument("--sensory-weight", type=float, default=400.0)
    ap.add_argument("--current-mv-ms", type=float, default=1.5)
    ap.add_argument("--settle-ms", type=float, default=500.0)
    ap.add_argument("--measure-ms", type=float, default=1000.0)
    a = ap.parse_args()

    cn, _ = load_training_brain(a.data, a.cache, "corrected")
    table = leg_motor_table(cn.neurons)
    sel = lambda muscles: table[table.leg.eq("LF") & table.muscle.isin(muscles)]["idx"].to_numpy()
    mn = {"all": table["idx"].to_numpy(), "lf_flexor": sel(FLEXORS), "lf_extensor": sel(EXTENSORS)}
    sensory = SensoryMap(cn.neurons)
    print(f"{len(mn['all'])} leg motor neurons (LF tibia flexor {len(mn['lf_flexor'])}, extensor "
          f"{len(mn['lf_extensor'])}); {len(sensory.idx)} leg afferents", flush=True)
    report = {}
    for g in (float(x) for x in a.g_inh.split(",")):
        W = scale_inhibition(cn.W, g)
        runs = {c: [run_condition(W, sensory, mn, c, s, a) for s in range(a.seeds)]
                for c in ("base", "rest", "flex", "extend", "current")}
        mean = lambda c, k: float(np.mean([r[k] for r in runs[c]]))
        gains = [reflex_gain(f["lf_flexor_hz"], f["lf_extensor_hz"], e["lf_flexor_hz"], e["lf_extensor_hz"])
                 for f, e in zip(runs["flex"], runs["extend"], strict=True)]
        entry = {"mean": {c: {k: mean(c, k) for k in runs[c][0]} for c in runs},
                 "reflex_gain_hz": {"mean": float(np.mean(gains)), "values": gains},
                 "current_response_hz": mean("current", "lf_flexor_hz") - mean("rest", "lf_flexor_hz")}
        report[str(g)] = entry
        print(f"g_I {g:g}: motor mean Hz base {mean('base', 'motor_mean_hz'):.2f} rest {mean('rest', 'motor_mean_hz'):.2f} "
              f"| active>1Hz base {mean('base', 'motor_active'):.0%} rest {mean('rest', 'motor_active'):.0%} "
              f"| afferents {mean('rest', 'afferent_mean_hz'):.1f} Hz | brain active {mean('rest', 'brain_active'):.1%} "
              f"| LF reflex gain {np.mean(gains):+.2f} Hz {np.round(gains, 2).tolist()} "
              f"| current response {entry['current_response_hz']:+.2f} Hz", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

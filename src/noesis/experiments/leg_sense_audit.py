"""Why do the leg senses tilt the body? Audit the leg-afferent -> leg-motor-neuron pathway.

Part A (structure): propagate each (organ, leg) sensory group through the row-normalised signed
connectome for 1..3 hops and tabulate the influence on leg motor neurons, then check whether
left and right legs receive mirror-symmetric influence.

Part B (dynamics): run the awake brain alone with the sensory drive a *resting* body would
produce, and compare joint-level motor drive (agonist - antagonist rate) between conditions and
between mirror-image joints. No body, no learning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from noesis.connectome import load_cached
from noesis.lif import LIFNetwork, LIFParams
from noesis.motor import RateTracker, build_motor_matrix, leg_motor_table
from noesis.sensory import LEGS, LegState, SensoryMap

DOFS = ("Coxa", "Coxa_roll", "Coxa_yaw", "Femur", "Femur_roll", "Tibia", "Tarsus1")
JOINTS = [f"joint_{leg}{d}" for leg in LEGS for d in DOFS]  # NeuroMechFly order
REST_LOAD = {"LF": 0.0, "LM": 0.9, "LH": 5.8, "RF": 0.0, "RM": 0.9, "RH": 5.3}  # measured at rest
MIRROR = {"L": "R", "R": "L"}


def normalised(W: sparse.csr_matrix) -> sparse.csr_matrix:
    """Each presynaptic neuron's outgoing weights scaled to sum |w| = 1 (signs kept)."""
    out = np.asarray(abs(W).sum(axis=1)).ravel()
    out[out == 0] = 1.0
    return sparse.diags(1.0 / out) @ W


def structural(cn, smap: SensoryMap, hops: int = 3):
    """Per-group signed influence on every neuron, summed over 1..hops hops."""
    M = normalised(cn.W).tocsr()
    mn = leg_motor_table(cn.neurons)
    rows = []
    for kind, leg, idx in smap.groups:
        v = np.zeros(cn.n)
        v[idx] = 1.0 / len(idx)
        total = np.zeros(cn.n)
        per_hop = []
        for _ in range(hops):
            v = M.T @ v  # row vector times M
            total += v
            per_hop.append(v[mn["idx"].to_numpy()].sum())
        infl = total[mn["idx"].to_numpy()]
        rows.append({"kind": kind, "leg": leg, "n_sensory": len(idx), "net": infl.sum(),
                     "abs": np.abs(infl).sum(), "hop1": per_hop[0], "hop2": per_hop[1],
                     "hop3": per_hop[2], "mn_idx": mn["idx"].to_numpy(), "infl": infl})
    return mn, rows


def structural_report(mn, rows, neurons) -> dict:
    out: dict = {}
    tab = pd.DataFrame([{k: r[k] for k in ("kind", "leg", "n_sensory", "net", "abs", "hop1", "hop2", "hop3")}
                        for r in rows])
    tab["side"] = tab["leg"].str[0]
    tab["segment"] = tab["leg"].str[1]
    # which organ classes push the motor neurons hardest (total |influence| per sensory neuron)
    by_kind = tab.groupby("kind").apply(
        lambda g: pd.Series({"abs_per_neuron": g["abs"].sum() / g["n_sensory"].sum(),
                             "net_per_neuron": g["net"].sum() / g["n_sensory"].sum(),
                             "n": g["n_sensory"].sum()}), include_groups=False)
    out["by_kind"] = by_kind.sort_values("abs_per_neuron", ascending=False).round(5).to_dict("index")
    # which muscles receive the most (sum over all groups)
    infl_total = np.zeros(len(mn))
    for r in rows:
        infl_total += r["infl"]
    m = mn.assign(infl=infl_total, a=np.abs(infl_total))
    by_muscle = m.groupby("muscle").agg(n=("idx", "size"), net=("infl", "sum"), abs=("a", "sum"))
    by_muscle["net_per_mn"] = by_muscle["net"] / by_muscle["n"]
    out["by_muscle"] = by_muscle.sort_values("abs", ascending=False).round(4).head(10).to_dict("index")
    # mirror symmetry: total sensory input onto left vs right motor neurons, per organ kind
    m["side"] = m["leg"].str[0]
    sym = {}
    for kind in sorted(tab["kind"].unique()):
        infl_k = np.zeros(len(mn))
        for r in rows:
            if r["kind"] == kind:
                infl_k += r["infl"]
        sub = m.assign(infl=infl_k)
        left = sub.loc[sub.side == "L", "infl"].sum()
        right = sub.loc[sub.side == "R", "infl"].sum()
        sym[kind] = {"left_net": round(left, 5), "right_net": round(right, 5),
                     "asym": round((left - right) / (abs(left) + abs(right) + 1e-12), 3)}
    out["symmetry_by_kind"] = sym
    # inventory asymmetry (data completeness): neurons per side
    n = neurons
    leg_aff = n[n["Flow"].eq("afferent") & n["Sub Class"].fillna("").str.contains("_leg_")]
    side = leg_aff["Nerve"].fillna("").str.extract(r"^(left|right)_")[0].fillna(leg_aff["Soma side"])
    out["afferents_per_side"] = side.value_counts().to_dict()
    out["motor_per_side"] = m["side"].value_counts().to_dict()
    return out


def rest_state(kind_filter: str | None, corrected_neutral: bool) -> dict[str, LegState]:
    return {leg: LegState(load=REST_LOAD[leg]) for leg in LEGS}


def sensory_rates(smap: SensoryMap, only: set[str] | None, neutral_proprio: bool) -> np.ndarray:
    """Rates a resting body would give. ``only``: silence every other organ class (base rate).
    ``neutral_proprio``: claw/hook at base rate instead of the sigmoid midpoint."""
    rates = smap.drive(rest_state(None, False))
    for k, (kind, _, _) in enumerate(smap.groups):
        sl = slice(smap._bounds[k], smap._bounds[k + 1])
        if (only is not None and kind not in only) or (neutral_proprio and kind in ("claw", "hook")):
            rates[sl] = smap.base_hz
    return rates


def dynamics(cn, smap: SensoryMap, G, mn, adapt: float, seed: int, settle_ms: float, window_ms: float):
    legs_mn = mn["idx"].to_numpy()
    kinds = sorted({k for k, _, _ in smap.groups})
    conditions = {"no leg senses": None, "rest drive (as in body sim)": (None, False),
                  "rest drive, claw/hook neutral": (None, True)}
    for k in kinds:
        conditions[f"only {k}"] = ({k}, False)
    results = {}
    for name, spec in conditions.items():
        net = LIFNetwork(cn.W, LIFParams(dt=0.5, da=adapt), seed=seed)
        kw = {"background_hz": 5.0, "background_weight": 90.0}
        if spec is not None:
            kw.update(drive_idx=smap.idx, drive_rate_hz=sensory_rates(smap, spec[0], spec[1]),
                      drive_weight=400.0)
        net.run(settle_ms, **kw)
        tr = RateTracker(cn.n, 50.0)
        us = []
        for _ in range(round(window_ms / 5.0)):
            _, sp = net.run(5.0, **kw)
            us.append(G @ tr.update(sp, 5.0))
        u = np.mean(us, axis=0)  # mean agonist-antagonist rate per joint (Hz)
        rate = tr.rate[legs_mn].mean()
        mir = []
        for j, name_j in enumerate(JOINTS):
            side, seg, dof = name_j[len("joint_")], name_j[len("joint_") + 1], name_j[len("joint_") + 2:]
            if side == "L":
                jr = JOINTS.index(f"joint_R{seg}{dof}")
                a, b = u[j], u[jr]
                if abs(a) + abs(b) > 5.0:  # ignore silent joints
                    mir.append((name_j[6:], round(float(a), 1), round(float(b), 1),
                                round(float((a - b) / (abs(a) + abs(b))), 2)))
        results[name] = {"mean_leg_mn_rate_hz": round(float(rate), 2),
                         "mean_abs_joint_drive_hz": round(float(np.abs(u).mean()), 1),
                         "mirror_pairs(L,R,asym)": mir}
        print(f"  {name:32s} legMN {rate:6.2f} Hz | mean |joint drive| {np.abs(u).mean():6.1f} Hz", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--adapt", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle-ms", type=float, default=1000.0)
    ap.add_argument("--window-ms", type=float, default=1500.0)
    ap.add_argument("--out", type=Path, default=Path("outputs/leg_sense_audit.json"))
    a = ap.parse_args()

    cn = load_cached(a.data / "raw", a.data / "processed")
    smap = SensoryMap(cn.neurons)
    G = build_motor_matrix(cn.neurons, JOINTS)

    print("== A. structure")
    mn, rows = structural(cn, smap)
    report = {"structure": structural_report(mn, rows, cn.neurons)}
    print(json.dumps(report["structure"], indent=1, default=float))
    print("== B. dynamics")
    report["dynamics"] = dynamics(cn, smap, G, mn, a.adapt, a.seed, a.settle_ms, a.window_ms)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()

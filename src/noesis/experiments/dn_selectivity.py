"""Do descending neurons (the brain's output to the nerve cord) carry odour identity and side?

Brain only, corrected connectome, graded APL, same drive as ``mb_selectivity``. Two tests, per neuron
population vector, each with a shuffle null:

1. Odour identity: four odours (disjoint glomerulus sets) x ``--reps`` independently seeded networks.
   The code is the vector of descending-neuron (DN) rates. Identity index = mean Pearson correlation
   between repeats of one odour minus mean correlation between odours; null permutes odour labels.
2. Side: the same odours driven through left-side or right-side receptors only. (a) Same machinery with
   the class "left-driven" vs "right-driven". (b) Lateralisation: S = [mean rate of left-soma DNs minus
   right-soma DNs] under left drive minus the same under right drive, averaged over trials; the null
   randomly reassigns the DN soma-side labels (keeping the left/right counts). S > 0 means a DN
   population responds more on the side of the stimulated receptors. The mapping from DN activity to
   a turn direction is NOT assumed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.connectome import load_all_pairs
from noesis.embodiment import load_training_brain
from noesis.experiments.mb_selectivity import glomeruli, identity_null, present
from noesis.experiments.sense_audit import receptor_sides
from noesis.mushroom_body import populations, with_weak_pathway_edges


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two rate vectors; 0 if either is constant (no pattern to compare)."""
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def lateralisation(trials: list[dict], left_dn: np.ndarray, right_dn: np.ndarray, n_perm: int,
                   all_dn: np.ndarray, seed: int = 0) -> dict:
    """S = mean over matched trials of (L-R DN rate | left drive) - (L-R DN rate | right drive).

    ``trials`` items: {'set': id, 'side': 'left'|'right', 'rate': per-neuron rate vector}.
    Null: reassign soma-side labels among ``all_dn`` keeping the group sizes.
    """
    def stat(l_idx, r_idx):
        diffs = []
        for sid in sorted({t["set"] for t in trials}):
            by = {s: np.mean([t["rate"] for t in trials if t["set"] == sid and t["side"] == s], axis=0)
                  for s in ("left", "right")}
            diffs.append((by["left"][l_idx].mean() - by["left"][r_idx].mean())
                         - (by["right"][l_idx].mean() - by["right"][r_idx].mean()))
        return float(np.mean(diffs))

    observed = stat(left_dn, right_dn)
    rng = np.random.default_rng(seed)
    null = np.array([stat(*np.split(rng.permutation(all_dn)[:len(left_dn) + len(right_dn)], [len(left_dn)]))
                     for _ in range(n_perm)])
    return {"S_hz": observed, "null_sd_hz": float(null.std()),
            "p_two_sided": float((1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12)) / (n_perm + 1))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/selectivity-cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/dn_selectivity.json"))
    ap.add_argument("--apl-gain", type=float, default=1024.0)
    ap.add_argument("--weights", default="400,100")
    ap.add_argument("--rate", type=float, default=80.0)
    ap.add_argument("--odours", type=int, default=4)
    ap.add_argument("--glom-per-odour", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--side-sets", type=int, default=3)
    ap.add_argument("--side-reps", type=int, default=2)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--settle-ms", type=float, default=200.0)
    ap.add_argument("--stim-ms", type=float, default=500.0)
    a = ap.parse_args()

    cn, _ = load_training_brain(a.data, a.cache, "corrected")
    pops = populations(cn.neurons)
    W = with_weak_pathway_edges(cn, load_all_pairs(a.data / "raw", a.data / "processed", cn.neurons), pops)
    glom = glomeruli(cn.neurons, pops)
    names = list(glom)
    perm = list(np.random.default_rng(1).permutation(names))  # same odour sets as mb_selectivity
    odour_sets = [perm[i * a.glom_per_odour:(i + 1) * a.glom_per_odour] for i in range(a.odours)]
    side_sets = odour_sets[:a.side_sets]
    dn_mask = cn.neurons["Super Class"].eq("descending").to_numpy()
    dn = np.flatnonzero(dn_mask)
    soma = cn.neurons["Soma side"].to_numpy()
    left_dn, right_dn = dn[soma[dn] == "left"], dn[soma[dn] == "right"]
    side = receptor_sides(cn.neurons).to_numpy()
    print(f"{len(dn)} descending neurons (left {len(left_dn)}, right {len(right_dn)})", flush=True)
    report = {"apl_gain": a.apl_gain, "rate_hz": a.rate, "odours": [list(map(str, o)) for o in odour_sets],
              "weights": {}}
    stim_s = a.stim_ms * 1e-3
    for weight in (float(w) for w in a.weights.split(",")):
        kw = {"rate_hz": a.rate, "weight": weight, "gain": a.apl_gain, "settle_ms": a.settle_ms,
              "stim_ms": a.stim_ms}
        codes, active = {}, []
        for o, gl in enumerate(odour_sets):
            drive = np.concatenate([glom[g]["orn"] for g in gl])
            for rep in range(a.reps):
                rate = present(W, pops, drive, seed=200 + 10 * o + rep, **kw)[dn] / stim_s
                codes.setdefault(o, []).append(rate)
                active.append(float(np.mean(rate > 0)))
        odour = identity_null(codes, a.n_perm, sim=corr)
        print(f"w{weight:g} odour identity (DN rate-vector correlation): index {odour['observed']:+.3f} "
              f"(null {odour['null_mean']:+.3f} ± {odour['null_sd']:.3f}) p={odour['p_value']:.3f} "
              f"| DN active {np.mean(active):.1%}", flush=True)
        trials, side_codes = [], {0: [], 1: []}
        for k, gl in enumerate(side_sets):
            for s_i, s in enumerate(("left", "right")):
                orn = np.concatenate([glom[g]["orn"] for g in gl])
                drive = orn[side[orn] == s]
                for rep in range(a.side_reps):
                    counts = present(W, pops, drive, seed=500 + 100 * k + 10 * s_i + rep, **kw)
                    rate = counts / stim_s
                    trials.append({"set": k, "side": s, "rate": rate})
                    side_codes[s_i].append(rate[dn])
        side_id = identity_null(side_codes, a.n_perm, sim=corr)
        lat = lateralisation(trials, left_dn, right_dn, a.n_perm, dn)
        print(f"w{weight:g} side identity: index {side_id['observed']:+.3f} "
              f"(null {side_id['null_mean']:+.3f} ± {side_id['null_sd']:.3f}) p={side_id['p_value']:.3f} "
              f"| lateralisation S {lat['S_hz']:+.2f} Hz (null sd {lat['null_sd_hz']:.2f}) "
              f"p={lat['p_two_sided']:.3f}", flush=True)
        report["weights"][str(weight)] = {"odour_identity": odour, "dn_active_fraction": float(np.mean(active)),
                                          "side_identity": side_id, "lateralisation": lat}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()

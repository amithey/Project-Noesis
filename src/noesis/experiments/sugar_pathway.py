"""Which dopaminergic cell types does sugar actually reach?

Brain only, corrected connectome, graded APL, same drive as the foraging assay (sugar gustatory neurons,
200 Hz, weight 400, 500 ms after a 200 ms settle). Conditions: sugar; no stimulus; bitter and water
gustatory neurons; and size-matched random sets of afferent (sensory) neurons. For each dopaminergic cell
type the sugar-evoked rate change is compared with the same-network control changes; a type counts as
sugar-selective only if sugar exceeds EVERY control by at least ``--min-excess-hz``.

In the fly, PAM neurons are widely reported to carry appetitive (reward) signals and PPL1 neurons aversive
ones; that expectation comes from the literature, is not verified here, and is not assumed in the test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.connectome import load_all_pairs
from noesis.embodiment import load_training_brain
from noesis.experiments.mb_selectivity import present
from noesis.mushroom_body import populations, with_weak_pathway_edges


def type_table(rate: dict[str, np.ndarray], base: np.ndarray, groups: dict[str, np.ndarray],
               min_excess_hz: float) -> list[dict]:
    """Per cell type: mean rate change under sugar and under each control; flag sugar-selective types."""
    rows = []
    for name, idx in groups.items():
        delta = {c: float(r[idx].mean() - base[idx].mean()) for c, r in rate.items()}
        sugar = delta.pop("sugar")
        excess = sugar - max(delta.values()) if delta else sugar
        rows.append({"type": name, "n": len(idx), "sugar_delta_hz": sugar,
                     "control_delta_hz": delta, "excess_over_best_control_hz": float(excess),
                     "sugar_selective": bool(excess >= min_excess_hz)})
    return sorted(rows, key=lambda r: -r["excess_over_best_control_hz"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/selectivity-cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/sugar_pathway.json"))
    ap.add_argument("--apl-gain", type=float, default=1024.0)
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--weight", type=float, default=400.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--random-sets", type=int, default=3)
    ap.add_argument("--min-excess-hz", type=float, default=2.0)
    ap.add_argument("--settle-ms", type=float, default=200.0)
    ap.add_argument("--stim-ms", type=float, default=500.0)
    a = ap.parse_args()

    cn, _ = load_training_brain(a.data, a.cache, "corrected")
    pops = populations(cn.neurons)
    W = with_weak_pathway_edges(cn, load_all_pairs(a.data / "raw", a.data / "processed", cn.neurons), pops)
    function = cn.neurons["Function"].fillna("")
    afferent = cn.neurons["Flow"].eq("afferent").to_numpy()
    pick = lambda token: np.flatnonzero((function.str.contains(token) & cn.neurons["Flow"].eq("afferent")).to_numpy())
    sugar, bitter, water = pick("sugar"), pick("bitter"), pick("water")
    others = np.setdiff1d(np.flatnonzero(afferent), sugar)
    rng = np.random.default_rng(0)
    conditions = {"sugar": sugar, "bitter": bitter, "water": water}
    for k in range(a.random_sets):
        conditions[f"random_afferent_{k}"] = rng.choice(others, size=len(sugar), replace=False)
    print(f"sugar GRNs {len(sugar)}, bitter {len(bitter)}, water {len(water)}; DAN {len(pops.dan)}", flush=True)

    kw = {"rate_hz": a.rate, "weight": a.weight, "gain": a.apl_gain, "settle_ms": a.settle_ms,
          "stim_ms": a.stim_ms}
    stim_s = a.stim_ms * 1e-3
    rates = {c: np.mean([present(W, pops, idx, seed=40 + s, **kw) / stim_s for s in range(a.seeds)], axis=0)
             for c, idx in conditions.items()}
    base = np.mean([present(W, pops, np.zeros(0, np.int64), seed=40 + s, **kw) / stim_s
                    for s in range(a.seeds)], axis=0)
    ctype = cn.neurons["Primary Cell Type"].fillna("").to_numpy()
    groups = {t: pops.dan[ctype[pops.dan] == t] for t in sorted(set(ctype[pops.dan]))}
    rows = type_table(rates, base, groups, a.min_excess_hz)
    for r in rows[:15]:
        ctrl = " ".join(f"{c.split('_')[0][:3]}{v:+.1f}" for c, v in r["control_delta_hz"].items())
        print(f"{r['type']:10s} n={r['n']:3d} sugar {r['sugar_delta_hz']:+6.2f} | controls {ctrl} | "
              f"excess {r['excess_over_best_control_hz']:+6.2f} {'SELECTIVE' if r['sugar_selective'] else ''}",
              flush=True)
    selective = [r["type"] for r in rows if r["sugar_selective"]]
    print(f"{len(selective)} of {len(rows)} dopaminergic types are sugar-selective: {selective}")
    report = {"n_sugar": len(sugar), "settings": {"rate": a.rate, "weight": a.weight, "seeds": a.seeds},
              "rows": rows, "selective_types": selective}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

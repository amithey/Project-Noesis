"""Side selectivity of literature-nominated steering descending-neuron (DN) types.

Same drive and graded APL as ``dn_selectivity``; here only the ipsilateral side signal is analysed, per
pre-specified DN group instead of the pooled population. The groups are fixed BEFORE looking at results:
DNa01, DNa02, DNa03, DNa04 individually, their union, the whole numbered DNa family (``DNa\\d\\d``), the numbered
DNb family, and DNp09. Which of these truly steer the fly is taken from memory of the literature and is
NOT verified here; the groups are candidates, not established steering cells.

Statistic S (per group) = mean over odour sets of [(left-soma minus right-soma rate | left receptors driven)
- (same | right receptors driven)]. Null: draw random DN sets of the same left/right sizes from all DNs
(5,000 draws), so a group must beat random same-sized DN groups. Bonferroni correction over the groups.
Per-trial DN rate vectors are saved so other groups can be analysed without re-simulating.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.connectome import load_all_pairs
from noesis.embodiment import load_training_brain
from noesis.experiments.dn_selectivity import lateralisation
from noesis.experiments.mb_selectivity import glomeruli, present
from noesis.experiments.sense_audit import receptor_sides
from noesis.mushroom_body import populations, with_weak_pathway_edges

GROUPS = {
    "DNa01": r"^DNa01$", "DNa02": r"^DNa02$", "DNa03": r"^DNa03$", "DNa04": r"^DNa04$",
    "DNa01-04": r"^DNa0[1-4]$", "DNa family": r"^DNa\d\d$", "DNb family": r"^DNb\d\d$", "DNp09": r"^DNp09$",
}


def group_stats(trials, dn_all, soma, ctype, n_perm, seed=0) -> dict:
    """S, mean left/right rates and permutation p for every pre-specified group."""
    import re
    out = {}
    for name, pattern in GROUPS.items():
        members = np.array([i for i in dn_all if re.match(pattern, ctype[i])], dtype=np.int64)
        left, right = members[soma[members] == "left"], members[soma[members] == "right"]
        if not len(left) or not len(right):
            out[name] = {"n_left": len(left), "n_right": len(right), "status": "unavailable"}
            continue
        res = lateralisation(trials, left, right, n_perm, dn_all, seed=seed)
        by = {s: np.mean([t["rate"] for t in trials if t["side"] == s], axis=0) for s in ("left", "right")}
        res.update(n_left=len(left), n_right=len(right),
                   left_soma_hz={s: float(by[s][left].mean()) for s in by},
                   right_soma_hz={s: float(by[s][right].mean()) for s in by})
        out[name] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/selectivity-cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/dn_subtypes.json"))
    ap.add_argument("--rates-out", type=Path, default=Path("outputs/dn_subtype_rates.npz"))
    ap.add_argument("--apl-gain", type=float, default=1024.0)
    ap.add_argument("--weights", default="400,100")
    ap.add_argument("--rate", type=float, default=80.0)
    ap.add_argument("--glom-per-odour", type=int, default=5)
    ap.add_argument("--sets", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--settle-ms", type=float, default=200.0)
    ap.add_argument("--stim-ms", type=float, default=500.0)
    a = ap.parse_args()

    cn, _ = load_training_brain(a.data, a.cache, "corrected")
    pops = populations(cn.neurons)
    W = with_weak_pathway_edges(cn, load_all_pairs(a.data / "raw", a.data / "processed", cn.neurons), pops)
    glom = glomeruli(cn.neurons, pops)
    perm = list(np.random.default_rng(1).permutation(list(glom)))
    sets = [perm[i * a.glom_per_odour:(i + 1) * a.glom_per_odour] for i in range(a.sets)]
    dn = np.flatnonzero(cn.neurons["Super Class"].eq("descending").to_numpy())
    soma = cn.neurons["Soma side"].to_numpy()
    ctype = cn.neurons["Primary Cell Type"].fillna("").to_numpy()
    side = receptor_sides(cn.neurons).to_numpy()
    stim_s = a.stim_ms * 1e-3
    report, saved = {"groups": list(GROUPS), "bonferroni_alpha": 0.05 / len(GROUPS), "weights": {}}, {}
    for weight in (float(w) for w in a.weights.split(",")):
        kw = {"rate_hz": a.rate, "weight": weight, "gain": a.apl_gain, "settle_ms": a.settle_ms,
              "stim_ms": a.stim_ms}
        trials = []
        for k, gl in enumerate(sets):
            orn = np.concatenate([glom[g]["orn"] for g in gl])
            for s_i, s in enumerate(("left", "right")):
                drive = orn[side[orn] == s]
                for rep in range(a.reps):
                    counts = present(W, pops, drive, seed=900 + 100 * k + 10 * s_i + rep, **kw)
                    full = np.zeros(cn.n)
                    full[dn] = counts[dn] / stim_s
                    trials.append({"set": k, "side": s, "rate": full})
        saved[f"w{weight:g}"] = np.stack([t["rate"][dn] for t in trials])
        saved[f"w{weight:g}_meta"] = np.array([[t["set"], t["side"] == "right"] for t in trials])
        stats = group_stats(trials, dn, soma, ctype, a.n_perm)
        report["weights"][str(weight)] = stats
        for name, r in stats.items():
            if r.get("status") == "unavailable":
                print(f"w{weight:g} {name:11s} unavailable (L{r['n_left']}/R{r['n_right']})", flush=True)
                continue
            print(f"w{weight:g} {name:11s} L{r['n_left']:2d}/R{r['n_right']:2d} S {r['S_hz']:+6.2f} Hz "
                  f"(random-group sd {r['null_sd_hz']:.2f}) p={r['p_two_sided']:.4f} | "
                  f"left-soma Lstim {r['left_soma_hz']['left']:.1f} Rstim {r['left_soma_hz']['right']:.1f} Hz | "
                  f"right-soma Lstim {r['right_soma_hz']['left']:.1f} Rstim {r['right_soma_hz']['right']:.1f} Hz",
                  flush=True)
    saved["dn_idx"] = dn
    a.rates_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.rates_out, **saved)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()

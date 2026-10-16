"""Does the olfactory pathway carry odour identity? Per-glomerulus and per-Kenyon-cell selectivity.

Brain only, with the graded APL used by the conditioning assay. Two tests, both against shuffle nulls:

1. Projection-neuron (PN) selectivity: drive the receptor neurons of ONE glomerulus; compare the
   firing of the uniglomerular PNs of that glomerulus ("own") with random PN sets of the same size
   drawn from all PNs. Selectivity needs own >> random (permutation p).
2. Kenyon-cell (KC) identity: four odours (disjoint glomerulus sets), each presented twice on
   independently seeded networks. The set of KCs that spike is the odour's code. Identity index =
   mean Jaccard overlap within an odour (across repeats) minus mean overlap between odours; the null
   permutes which of the eight codes belong to which odour.

Earlier work in this repo (docs/findings.md) found ~74-76% KC overlap between odours even with strong
APL inhibition. This script measures that again with controls rather than assuming it.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from noesis.connectome import load_all_pairs
from noesis.embodiment import load_training_brain
from noesis.lif import LIFNetwork, LIFParams
from noesis.mushroom_body import GradedAPL, populations, with_weak_pathway_edges

CHUNK_MS = 5.0


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard overlap of two boolean masks; 0 when both are empty (no code to compare)."""
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def identity_index(codes: dict[int, list[np.ndarray]], sim=jaccard) -> float:
    """Mean within-class similarity minus mean between-class similarity (default Jaccard of masks)."""
    within = [sim(a, b) for reps in codes.values() for a, b in itertools.combinations(reps, 2)]
    flat = [(o, m) for o, reps in codes.items() for m in reps]
    between = [sim(m1, m2) for (o1, m1), (o2, m2) in itertools.combinations(flat, 2) if o1 != o2]
    if not within or not between:
        raise ValueError("need >= 2 repeats per odour and >= 2 odours")
    return float(np.mean(within) - np.mean(between))


def identity_null(codes: dict[int, list[np.ndarray]], n_perm: int, seed: int = 0, sim=jaccard) -> dict:
    """Permute odour labels over all codes (keeping group sizes); return observed index and p."""
    observed = identity_index(codes, sim)
    flat = [m for reps in codes.values() for m in reps]
    sizes = [len(reps) for reps in codes.values()]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        order = rng.permutation(len(flat))
        shuffled, start = {}, 0
        for k, size in enumerate(sizes):
            shuffled[k] = [flat[i] for i in order[start:start + size]]
            start += size
        null.append(identity_index(shuffled, sim))
    null = np.asarray(null)
    return {"observed": observed, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p_value": float((1 + np.count_nonzero(null >= observed - 1e-12)) / (n_perm + 1))}


def pn_selectivity(rate_hz: np.ndarray, pn: np.ndarray, own: np.ndarray, n_perm: int, seed: int = 0) -> dict:
    """Own-PN mean rate vs random PN sets of equal size. rate_hz is per neuron, pn/own are indices."""
    if not len(own):
        raise ValueError("glomerulus has no uniglomerular PNs")
    own_mean = float(rate_hz[own].mean())
    rest = np.setdiff1d(pn, own)
    rng = np.random.default_rng(seed)
    draws = np.array([rate_hz[rng.choice(rest, size=len(own), replace=False)].mean() for _ in range(n_perm)])
    return {"own_hz": own_mean, "other_hz": float(rate_hz[rest].mean()), "random_set_mean_hz": float(draws.mean()),
            "p_value": float((1 + np.count_nonzero(draws >= own_mean - 1e-12)) / (n_perm + 1)), "n_own": len(own)}


def present(W, pops, drive: np.ndarray, rate_hz: float, weight: float, seed: int, gain: float,
            settle_ms: float, stim_ms: float, bg_hz: float = 0.0, bg_w: float = 60.0) -> np.ndarray:
    """Fresh network, settle, then drive for ``stim_ms``; return per-neuron spike counts in the drive window."""
    net = LIFNetwork(W, LIFParams(dt=0.5, da=1.0), seed=seed)
    apl = GradedAPL(net, pops, gain=gain)
    counts, cur = np.zeros(net.n), {}
    for phase_ms, d in ((settle_ms, None), (stim_ms, drive)):
        for _ in range(round(phase_ms / CHUNK_MS)):
            _, sp = net.run(CHUNK_MS, drive_idx=d, drive_rate_hz=rate_hz, drive_weight=weight,
                            background_hz=bg_hz, background_weight=bg_w, **cur)
            cur = apl.current(sp, CHUNK_MS)
            if d is not None:
                counts += np.bincount(sp, minlength=net.n)
    return counts


def glomeruli(neurons, pops) -> dict[str, dict]:
    """glomerulus name -> {'orn': indices, 'pn': uniglomerular PN indices}; only those with both."""
    ctype = neurons["Primary Cell Type"].fillna("").to_numpy()
    out = {}
    for name in sorted({t[4:] for t in ctype[pops.orn] if t.startswith("ORN_")}):
        orn = pops.orn[ctype[pops.orn] == f"ORN_{name}"]
        pn = np.array([i for i in pops.pn if ctype[i].startswith(name + "_")], dtype=np.int64)
        if len(orn) and len(pn):
            out[name] = {"orn": orn, "pn": pn}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/selectivity-cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/mb_selectivity.json"))
    ap.add_argument("--apl-gain", type=float, default=1024.0)
    ap.add_argument("--weights", default="400,100")
    ap.add_argument("--rate", type=float, default=80.0)
    ap.add_argument("--n-glomeruli", type=int, default=10)
    ap.add_argument("--odours", type=int, default=4)
    ap.add_argument("--glom-per-odour", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--settle-ms", type=float, default=200.0)
    ap.add_argument("--stim-ms", type=float, default=500.0)
    ap.add_argument("--background-hz", type=float, default=0.0)
    ap.add_argument("--background-weight", type=float, default=60.0)
    a = ap.parse_args()

    cn, _ = load_training_brain(a.data, a.cache, "corrected")
    pops = populations(cn.neurons)
    W = with_weak_pathway_edges(cn, load_all_pairs(a.data / "raw", a.data / "processed", cn.neurons), pops)
    glom = glomeruli(cn.neurons, pops)
    names = list(glom)
    rng = np.random.default_rng(0)
    chosen = list(rng.permutation(names)[:a.n_glomeruli])
    # odours use disjoint glomeruli: slices of one permutation
    perm = list(np.random.default_rng(1).permutation(names))
    odour_sets = [perm[i * a.glom_per_odour:(i + 1) * a.glom_per_odour] for i in range(a.odours)]
    print(f"{len(names)} glomeruli with ORNs and uniglomerular PNs; testing {len(chosen)}; "
          f"{a.odours} odours of {a.glom_per_odour}", flush=True)
    report = {"apl_gain": a.apl_gain, "rate_hz": a.rate, "odours": [list(map(str, o)) for o in odour_sets],
              "weights": {}}
    for weight in (float(w) for w in a.weights.split(",")):
        kw = {"rate_hz": a.rate, "weight": weight, "gain": a.apl_gain, "settle_ms": a.settle_ms,
              "stim_ms": a.stim_ms, "bg_hz": a.background_hz, "bg_w": a.background_weight}
        stim_s = a.stim_ms * 1e-3
        pn_rows = []
        for k, g in enumerate(chosen):
            counts = present(W, pops, glom[g]["orn"], seed=k, **kw)
            r = pn_selectivity(counts / stim_s, pops.pn, glom[g]["pn"], a.n_perm, seed=k)
            r["glomerulus"] = str(g)
            r["all_pn_hz"] = float(counts[pops.pn].mean() / stim_s)
            pn_rows.append(r)
            print(f"w{weight:g} PN {g:8s} own {r['own_hz']:6.1f} Hz vs other {r['other_hz']:6.1f} "
                  f"(random sets {r['random_set_mean_hz']:6.1f}) p={r['p_value']:.3f}", flush=True)
        codes, kc_active = {}, []
        for o, gl in enumerate(odour_sets):
            drive = np.concatenate([glom[g]["orn"] for g in gl])
            for rep in range(2):
                counts = present(W, pops, drive, seed=100 + 10 * o + rep, **kw)
                mask = counts[pops.kc] > 0
                codes.setdefault(o, []).append(mask)
                kc_active.append(float(mask.mean()))
        ident = identity_null(codes, a.n_perm)
        between = [jaccard(m1, m2) for (o1, m1), (o2, m2) in itertools.combinations(
            [(o, m) for o, reps in codes.items() for m in reps], 2) if o1 != o2]
        within = [jaccard(a_, b_) for reps in codes.values() for a_, b_ in itertools.combinations(reps, 2)]
        print(f"w{weight:g} KC active {np.mean(kc_active):.1%} | within-odour Jaccard {np.mean(within):.2f} "
              f"between {np.mean(between):.2f} | identity index {ident['observed']:+.3f} "
              f"(null {ident['null_mean']:+.3f} ± {ident['null_sd']:.3f}) p={ident['p_value']:.3f}", flush=True)
        report["weights"][str(weight)] = {
            "pn": pn_rows, "kc_active_fraction": float(np.mean(kc_active)),
            "kc": {**ident, "within": float(np.mean(within)), "between": float(np.mean(between))}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()

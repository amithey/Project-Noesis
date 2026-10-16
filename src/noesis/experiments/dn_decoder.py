"""Can a linear read-out of descending-neuron rates tell which side an odour came from, on unseen odours?

Brain only, corrected connectome with all inhibitory weights scaled by ``--g-inh``, graded APL.
Collection: ``--sets`` disjoint odour sets (5 glomeruli each); each is presented through left-side receptors
only and right-side receptors only, ``--reps`` independently seeded networks per side. Features per trial:
log(1 + rate) of (a) all descending neurons, (b) projection neurons, (c) a size-matched random set of
non-sensory, non-descending neurons.

Decoder: ridge classifier in closed form (dual form, p >> n). Evaluation is leave-one-ODOUR-SET-out, so the
test odour was never seen in training; features are standardised on the training trials only. Ridge strengths
are pre-specified (not tuned on the test set) and all are reported. The null shuffles side labels within each
odour set (keeping the pairing) and repeats the whole leave-one-set-out evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.connectome import load_all_pairs
from noesis.embodiment import load_training_brain
from noesis.experiments.ei_balance_scan import scale_inhibition
from noesis.experiments.mb_selectivity import glomeruli, present
from noesis.experiments.sense_audit import receptor_sides
from noesis.mushroom_body import populations, with_weak_pathway_edges


def standardise(train: np.ndarray, test: np.ndarray):
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def ridge_predict(x_train, y_train, x_test, lam_per_feature: float) -> np.ndarray:
    """Dual-form ridge to +/-1 targets; returns signed scores for x_test. lam = lam_per_feature * n_features."""
    if set(np.unique(y_train)) - {-1, 1}:
        raise ValueError("labels must be -1 / +1")
    lam = lam_per_feature * x_train.shape[1]
    gram = x_train @ x_train.T + lam * np.eye(len(x_train))
    alpha = np.linalg.solve(gram, y_train - y_train.mean())
    return (x_test @ (x_train.T @ alpha)) + y_train.mean()


def loso_accuracy(x: np.ndarray, y: np.ndarray, groups: np.ndarray, lam_per_feature: float) -> dict:
    """Leave-one-group-out accuracy; each held-out group is a whole odour set unseen in training."""
    correct, total, per = 0, 0, {}
    for g in np.unique(groups):
        te = groups == g
        xtr, xte = standardise(x[~te], x[te])
        pred = np.where(ridge_predict(xtr, y[~te], xte, lam_per_feature) >= 0, 1, -1)
        c = int((pred == y[te]).sum())
        correct, total, per[int(g)] = correct + c, total + int(te.sum()), c / int(te.sum())
    return {"accuracy": correct / total, "per_set": per}


def permutation_null(x, y, groups, lam_per_feature, n_perm: int, seed: int = 0) -> dict:
    """Shuffle side labels within each odour set; null distribution of leave-one-set-out accuracy."""
    rng = np.random.default_rng(seed)
    observed = loso_accuracy(x, y, groups, lam_per_feature)["accuracy"]
    null = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = y.copy()
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            y_perm[idx] = y[rng.permutation(idx)]
        null[i] = loso_accuracy(x, y_perm, groups, lam_per_feature)["accuracy"]
    return {"accuracy": observed, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p_value": float((1 + np.count_nonzero(null >= observed - 1e-12)) / (n_perm + 1))}


def collect(a) -> Path:
    cn, _ = load_training_brain(a.data, a.cache, "corrected")
    pops = populations(cn.neurons)
    W = scale_inhibition(with_weak_pathway_edges(
        cn, load_all_pairs(a.data / "raw", a.data / "processed", cn.neurons), pops), a.g_inh)
    glom = glomeruli(cn.neurons, pops)
    perm = list(np.random.default_rng(1).permutation(list(glom)))
    per = a.glom_per_odour
    sets = [perm[i * per:(i + 1) * per] for i in range(a.sets)]
    n = cn.neurons
    dn = np.flatnonzero(n["Super Class"].eq("descending").to_numpy())
    side = receptor_sides(n).to_numpy()
    sensory = n["Flow"].isin(["afferent", "efferent"]).to_numpy()
    pool = np.setdiff1d(np.flatnonzero(~sensory), np.concatenate([dn, pops.pn]))
    rand = np.random.default_rng(0).choice(pool, size=len(dn), replace=False)
    kw = {"rate_hz": a.rate, "weight": a.weight, "gain": a.apl_gain, "settle_ms": a.settle_ms, "stim_ms": a.stim_ms}
    feats = {"dn": [], "pn": [], "random": []}
    meta = []
    stim_s = a.stim_ms * 1e-3
    for k, gl in enumerate(sets):
        orn = np.concatenate([glom[g]["orn"] for g in gl])
        for s_i, s in enumerate(("left", "right")):
            drive = orn[side[orn] == s]
            for rep in range(a.reps):
                counts = present(W, pops, drive, seed=3000 + 100 * k + 10 * s_i + rep, **kw) / stim_s
                for name, idx in (("dn", dn), ("pn", pops.pn), ("random", rand)):
                    feats[name].append(np.log1p(counts[idx]))
                meta.append((k, 1 if s == "left" else -1, rep))
        print(f"collected odour set {k + 1}/{len(sets)}", flush=True)
    out = a.rates_out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=np.array(meta), g_inh=a.g_inh, **{k: np.stack(v) for k, v in feats.items()})
    return out


def analyse(path: Path, lams, n_perm: int) -> dict:
    d = np.load(path)
    meta = d["meta"]
    groups, y = meta[:, 0], meta[:, 1]
    report = {"n_trials": len(y), "n_sets": len(np.unique(groups)), "g_inh": float(d["g_inh"]), "features": {}}
    for name in ("dn", "pn", "random"):
        x = d[name]
        report["features"][name] = {"n_features": int(x.shape[1]), "active_fraction": float(np.mean(x > 0)),
                                    "by_lambda": {}}
        for lam in lams:
            res = permutation_null(x, y, groups, lam, n_perm)
            res["per_set"] = loso_accuracy(x, y, groups, lam)["per_set"]
            report["features"][name]["by_lambda"][str(lam)] = res
            print(f"{name:6s} lam {lam:g}/feat: held-out accuracy {res['accuracy']:.3f} "
                  f"(null {res['null_mean']:.3f} ± {res['null_sd']:.3f}) p={res['p_value']:.4f}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/selectivity-cache"))
    ap.add_argument("--rates-out", type=Path, default=Path("outputs/dn_decoder_rates.npz"))
    ap.add_argument("--out", type=Path, default=Path("outputs/dn_decoder.json"))
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--g-inh", type=float, default=12.0)
    ap.add_argument("--apl-gain", type=float, default=1024.0)
    ap.add_argument("--rate", type=float, default=80.0)
    ap.add_argument("--weight", type=float, default=400.0)
    ap.add_argument("--sets", type=int, default=10)
    ap.add_argument("--glom-per-odour", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--settle-ms", type=float, default=200.0)
    ap.add_argument("--stim-ms", type=float, default=500.0)
    ap.add_argument("--lambdas", default="0.01,0.1,1")
    ap.add_argument("--n-perm", type=int, default=500)
    a = ap.parse_args()
    path = a.rates_out if a.analyse_only else collect(a)
    report = analyse(path, [float(x) for x in a.lambdas.split(",")], a.n_perm)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

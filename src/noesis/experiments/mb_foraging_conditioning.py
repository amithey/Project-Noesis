"""Does hunger change what an odour-sugar pairing teaches the mushroom body?

Brain only. A fresh network per run: odour A is paired with sugar-receptor drive (reward reaches the
dopaminergic cells only through the connectome) and odour B is never paired. MBON responses to A and
B are measured with plasticity frozen before and after. Hunger scales the sugar-receptor drive via
``HomeostaticState.sensory_gain``; it does not touch plasticity directly.

Conditions: {fed, hungry} x {sugar pairing, no reward}. The learning index of a run is
(dA - dB) / before_A, with d = after - before MBON rate: negative means A became relatively weaker
(conditioned depression). A conditioned effect needs: sugar index < no-reward index, paired by seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.connectome import load_all_pairs
from noesis.drives import HomeostaticState
from noesis.embodiment import load_training_brain
from noesis.experiments.mb_conditioning import _run
from noesis.lif import LIFNetwork, LIFParams
from noesis.mushroom_body import DopamineGatedPlasticity, populations, with_weak_pathway_edges

STATES = {"fed": 1.0, "hungry": 0.2}


def learning_index(before: dict, after: dict) -> float | None:
    """(dA - dB) / before_A of MBON Hz; None when odour A did not drive the MBONs at baseline."""
    if before["A"] <= 0:
        return None
    return float(((after["A"] - before["A"]) - (after["B"] - before["B"])) / before["A"])


def paired_effect(sugar: list, none: list) -> dict:
    """Mean/SD of (sugar - none) learning index over seeds present in both; missing seeds dropped."""
    pairs = [(s, n) for s, n in zip(sugar, none, strict=True) if s is not None and n is not None]
    if not pairs:
        return {"n": 0, "mean": None, "sd": None}
    diff = np.array([s - n for s, n in pairs])
    return {"n": len(diff), "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)) if len(diff) > 1 else None,
            "negative_in": int((diff < 0).sum())}


def run_one(W, pops, odor, sugar_idx, state: str, reward: bool, seed: int, a) -> dict:
    net = LIFNetwork(W, LIFParams(dt=0.5, da=1.0), seed=seed)
    plast = DopamineGatedPlasticity(net, pops, eta=a.eta)

    def test():
        out = {}
        for name, idx in odor.items():
            (_, mbon, _), _ = _run(net, plast, pops, a.test_ms, idx, a.odor_hz, 400.0, learn=False)
            out[name] = mbon
            _run(net, plast, pops, 300.0, None, 0.0, 0.0, learn=False)
        return out

    before = test()
    if reward:
        gain = HomeostaticState(energy=STATES[state]).sensory_gain("sugar")
        idx = np.concatenate([odor["A"], sugar_idx])
        rate = np.concatenate([np.full(len(odor["A"]), a.odor_hz), np.full(len(sugar_idx), a.sugar_hz * gain)])
    else:
        idx, rate = odor["A"], np.full(len(odor["A"]), a.odor_hz)
    (_, _, dan), _ = _run(net, plast, pops, a.train_ms, idx, rate, 400.0, learn=True)
    after = test()
    return {"before": before, "after": after, "index": learning_index(before, after),
            "train_dan_hz": float(dan), "weight_fraction": float(plast.mean_weight_fraction)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/foraging-cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/mb_foraging.json"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train-ms", type=float, default=2000.0)
    ap.add_argument("--test-ms", type=float, default=600.0)
    ap.add_argument("--eta", type=float, default=5e-4)
    ap.add_argument("--odor-hz", type=float, default=80.0)
    ap.add_argument("--sugar-hz", type=float, default=200.0)
    ap.add_argument("--glomeruli", type=int, default=5)
    a = ap.parse_args()

    cn, _ = load_training_brain(a.data, a.cache, "corrected")
    pairs = load_all_pairs(a.data / "raw", a.data / "processed", cn.neurons)
    pops = populations(cn.neurons)
    W = with_weak_pathway_edges(cn, pairs, pops)
    sugar_idx = np.flatnonzero(cn.neurons["Function"].fillna("").str.contains("sugar").to_numpy())
    glom = cn.neurons.iloc[pops.orn]["Primary Cell Type"].fillna("?").to_numpy()
    results = {}
    for seed in range(a.seeds):
        types = np.random.default_rng(seed).permutation(np.unique(glom[glom != "?"]))
        odor = {n: pops.orn[np.isin(glom, types[i * a.glomeruli:(i + 1) * a.glomeruli])]
                for i, n in enumerate("AB")}
        for state in STATES:
            for reward in (True, False):
                key = f"{state}/{'sugar' if reward else 'none'}"
                r = run_one(W, pops, odor, sugar_idx, state, reward, seed, a)
                results.setdefault(key, []).append(r)
                print(f"seed {seed} {key:12s} index {r['index']} | MBON A {r['before']['A']:.2f}->{r['after']['A']:.2f} "
                      f"B {r['before']['B']:.2f}->{r['after']['B']:.2f} | DAN {r['train_dan_hz']:.2f} Hz "
                      f"| weights {r['weight_fraction']:.4f}", flush=True)
    summary = {s: paired_effect([r["index"] for r in results[f"{s}/sugar"]],
                                [r["index"] for r in results[f"{s}/none"]]) for s in STATES}
    print("sugar minus no-reward learning index (paired by seed):", json.dumps(summary))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"runs": results, "summary": summary}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

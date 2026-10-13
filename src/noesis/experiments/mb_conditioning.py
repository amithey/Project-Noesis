"""Brain-only associative-conditioning assay for the mushroom-body plasticity.

The classic fly experiment, on the connectome with no body: odour A is paired with dopamine
(training), then the response of the mushroom-body output neurons to A and to a different odour B
is measured with plasticity frozen. Learning predicts a drop for A and little change for B.

Reward can arrive two ways: ``direct`` stimulates the dopaminergic neurons (the optogenetic
reward assay used on real flies), ``sugar`` stimulates sugar-taste neurons so the dopamine
neurons are driven only through the connectome. ``--reward none`` is the control.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from noesis.connectome import load_all_pairs, load_cached
from noesis.lif import LIFNetwork, LIFParams
from noesis.mushroom_body import (
    DopamineGatedPlasticity,
    GradedAPL,
    compartmentalize_mbons,
    kc_rescale_factors,
    mbon_input_compartments,
    populations,
    rescale_kc_edges,
    with_weak_pathway_edges,
)

CHUNK_MS = 5.0


def _run(net, plast, pops, ms, drive_idx, rate, weight, learn, apl=None):
    """Run ``ms`` ms; return mean Hz of (KC, MBON, DAN) populations and per-neuron spike counts."""
    plast.enabled = learn
    counts = np.zeros(net.n)
    cur = {}
    for _ in range(round(ms / CHUNK_MS)):
        _, sp = net.run(CHUNK_MS, drive_idx=drive_idx, drive_rate_hz=rate, drive_weight=weight, **cur)
        plast.step(sp, CHUNK_MS)
        if apl is not None:
            cur = apl.current(sp, CHUNK_MS)
        counts += np.bincount(sp, minlength=net.n)
    sec = ms * 1e-3
    hz = lambda idx: counts[idx].sum() / max(len(idx), 1) / sec
    return (hz(pops.kc), hz(pops.mbon), hz(pops.dan)), counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--reward", choices=("direct", "sugar", "none"), default="direct")
    ap.add_argument("--train-ms", type=float, default=2000.0)
    ap.add_argument("--test-ms", type=float, default=600.0)
    ap.add_argument("--eta", type=float, default=5e-4)
    ap.add_argument("--glomeruli", type=int, default=5, help="glomeruli (receptor types) per odour")
    ap.add_argument("--odor-hz", type=float, default=80.0)
    ap.add_argument("--adapt", type=float, default=1.0)
    ap.add_argument("--mbon-axonal-weight", type=float, default=-1.0,
                    help="weight of synapses onto MBON axons (outside the MB); <0 = point neurons")
    ap.add_argument("--kc-rescale", action="store_true",
                    help="bring BANC Kenyon-cell synapse counts to FAFB totals (needs <data>/fafb)")
    ap.add_argument("--sparseness-only", action="store_true", help="only measure KC sparseness")
    ap.add_argument("--apl-uniform", action="store_true",
                    help="APL inhibits every KC equally (BANC lacks APL synapses on ~24% of KCs)")
    ap.add_argument("--apl-gain", type=float, default=0.0, help="0 = spiking APL; >0 = graded APL")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cn = load_cached(a.data / "raw", a.data / "processed")
    pairs = load_all_pairs(a.data / "raw", a.data / "processed", cn.neurons)
    pops = populations(cn.neurons)
    W = with_weak_pathway_edges(cn, pairs, pops)
    if a.mbon_axonal_weight >= 0:
        comp = mbon_input_compartments(a.data / "raw", a.data / "processed", cn.neurons, pops.mbon)
        W = compartmentalize_mbons(W, comp, cn.neurons, a.mbon_axonal_weight)
        print(f"MBON inputs: dendritic (MB) x1, axonal x{a.mbon_axonal_weight:g}")
    if a.kc_rescale:
        import json
        cache = a.data / "processed" / "kc_rescale_factors.json"
        if cache.exists():
            factors = {tuple(k.split("->")): v for k, v in json.loads(cache.read_text()).items()}
        else:
            ctype = cn.neurons["Primary Cell Type"].fillna("?").to_numpy()
            factors = kc_rescale_factors(pairs, ctype, a.data / "fafb")
            cache.write_text(json.dumps({"->".join(k): v for k, v in factors.items()}, indent=1))
        W = rescale_kc_edges(W, cn.neurons["Primary Cell Type"].fillna("?").to_numpy(), factors)
        print("KC synapses rescaled to FAFB: " + ", ".join(f"{a_}->{b_} x{f:.1f}" for (a_, b_), f in
                                                     sorted(factors.items())))
    print(f"ORN {len(pops.orn)}  PN {len(pops.pn)}  KC {len(pops.kc)}  MBON {len(pops.mbon)}  "
          f"DAN {len(pops.dan)} | edges {cn.W.nnz} -> {W.nnz} (+{W.nnz - cn.W.nnz} weak olfactory/MB)")

    net = LIFNetwork(W, LIFParams(dt=0.5, da=a.adapt), seed=a.seed)
    apl = GradedAPL(net, pops, gain=a.apl_gain, uniform=a.apl_uniform) if a.apl_gain > 0 else None
    plast = DopamineGatedPlasticity(net, pops, eta=a.eta)
    print(f"plastic KC->MBON synapses: {len(plast.pos)}")

    rng = np.random.default_rng(a.seed)
    # an odour = the receptor neurons of a few glomeruli; A and B use disjoint glomeruli
    glom = cn.neurons.iloc[pops.orn]["Primary Cell Type"].fillna("?").to_numpy()
    types = rng.permutation(np.unique(glom[glom != "?"]))
    odor = {name: pops.orn[np.isin(glom, types[i * a.glomeruli:(i + 1) * a.glomeruli])]
            for i, name in enumerate("AB")}
    print(f"odour A: {len(odor['A'])} receptor neurons {sorted(types[:a.glomeruli])}")
    print(f"odour B: {len(odor['B'])} receptor neurons {sorted(types[a.glomeruli:2 * a.glomeruli])}")
    sugar = np.flatnonzero(cn.neurons["Function"].fillna("").str.contains("sugar").to_numpy())

    def test(label):
        out, active = {}, {}
        for name, idx in odor.items():
            out[name], cnt = _run(net, plast, pops, a.test_ms, idx, a.odor_hz, 400.0, learn=False, apl=apl)
            active[name] = cnt[pops.kc] > 0
            _run(net, plast, pops, 300.0, None, 0.0, 0.0, learn=False)  # let activity settle
        print(f"{label:10s} KC / MBON / DAN Hz  odour A: {out['A'][0]:5.2f} {out['A'][1]:6.2f} {out['A'][2]:5.2f}"
              f"   odour B: {out['B'][0]:5.2f} {out['B'][1]:6.2f} {out['B'][2]:5.2f}", flush=True)
        a_, b_ = active["A"], active["B"]
        print(f"           KCs active: A {a_.sum()} ({a_.mean():.0%})  B {b_.sum()} ({b_.mean():.0%})  "
              f"overlap {(a_ & b_).sum() / max((a_ | b_).sum(), 1):.0%} of the union", flush=True)
        return out

    before = test("before")
    if a.sparseness_only:
        return

    # training: odour A (+ reward)
    if a.reward == "none":
        idx, rate, w = odor["A"], a.odor_hz, 400.0
        drive = (idx, rate, w)
    else:
        reward = pops.dan if a.reward == "direct" else sugar
        idx = np.concatenate([odor["A"], reward])
        rate = np.concatenate([np.full(len(odor["A"]), a.odor_hz),
                               np.full(len(reward), 60.0 if a.reward == "direct" else 200.0)])
        drive = (idx, rate, 400.0)
    (kc, mb, da), _ = _run(net, plast, pops, a.train_ms, *drive, learn=True, apl=apl)
    print(f"training ({a.reward}): KC {kc:.2f} Hz, MBON {mb:.2f} Hz, DAN {da:.2f} Hz | "
          f"plastic weights at {plast.mean_weight_fraction:.3f} of connectome value", flush=True)

    after = test("after")
    for name in "AB":
        b, f = before[name][1], after[name][1]
        change = (f - b) / b * 100 if b > 0 else float("nan")
        print(f"odour {name}: MBON response {b:.2f} -> {f:.2f} Hz ({change:+.1f}%)")


if __name__ == "__main__":
    main()

"""Audit the two prerequisites of learning from the world: a sparse odour code and a taste->dopamine path.

1. Odour code: stimulate the receptor neurons of 5 glomeruli and measure, layer by layer, how specific
   the response stays (projection neurons on/off target, Kenyon cells active, APL rate).
2. Reward path: excitatory hop distance from sugar-taste neurons to the reward dopamine neurons (PAM),
   and PAM firing with and without sugar in the awake (background-active) brain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from noesis.connectome import load_all_pairs, load_cached, presynaptic_signs
from noesis.lif import LIFNetwork, LIFParams
from noesis.mushroom_body import populations, with_weak_pathway_edges


def odour_code(cn, pairs, pops, seed: int, glomeruli: int) -> None:
    n = cn.neurons
    ctype = n["Primary Cell Type"].fillna("")
    W = with_weak_pathway_edges(cn, pairs, pops)
    glom = ctype.iloc[pops.orn].to_numpy()
    types = np.random.default_rng(seed).permutation(np.unique(glom[glom != ""]))[:glomeruli]
    stim = pops.orn[np.isin(glom, types)]
    names = [t.replace("ORN_", "") for t in types]
    on = np.array([any(t.startswith(g + "_") for g in names) for t in ctype.iloc[pops.pn]])
    net = LIFNetwork(W, LIFParams(dt=0.5, da=1.0), seed=seed)
    _, i = net.run(600.0, stim, 80.0, 400.0)
    hz = np.bincount(i, minlength=cn.n) / 0.6
    pn = hz[pops.pn]
    print(f"odour = glomeruli {names} ({len(stim)} receptor neurons, {on.sum()} matching PNs)")
    print(f"  PNs on target  {pn[on].mean():5.0f} Hz, {(pn[on] > 0).mean():4.0%} active")
    print(f"  PNs off target {pn[~on].mean():5.0f} Hz, {(pn[~on] > 0).mean():4.0%} active")
    print(f"  Kenyon cells   {hz[pops.kc].mean():5.1f} Hz, {(hz[pops.kc] > 0).mean():4.0%} active "
          f"(real flies: ~5-10%)")
    print(f"  APL            {', '.join(f'{x:.0f}' for x in hz[pops.apl])} Hz (spiking ceiling ~400 Hz at dt 0.5 ms)")
    ln = np.flatnonzero(n["Class"].fillna("").str.contains("antennal_lobe_local").to_numpy())
    sign = presynaptic_signs(n)
    to_pn = pairs[pairs["pre"].isin(set(ln)) & pairs["post"].isin(set(pops.pn))]
    s = sign[to_pn["pre"].to_numpy()]
    print(f"  AL local neurons -> PNs: {to_pn['syn'][s > 0].sum()} excitatory vs "
          f"{to_pn['syn'][s < 0].sum()} inhibitory synapses; LNs fire {hz[ln].mean():.0f} Hz")


def reward_path(cn, pairs, seed: int) -> None:
    n = cn.neurons
    ctype = n["Primary Cell Type"].fillna("")
    pam = np.flatnonzero(ctype.str.startswith("PAM").to_numpy())
    sugar = np.flatnonzero(n["Function"].fillna("").str.contains("sugar").to_numpy())
    sign = presynaptic_signs(n)
    print(f"sugar-taste neurons {len(sugar)} -> reward dopamine neurons (PAM) {len(pam)}")
    for label, min_syn in (("pairs >= 5 synapses", 5), ("pairs >= 1 synapse", 1)):
        sel = pairs[(pairs["syn"] >= min_syn) & (sign[pairs["pre"].to_numpy()] > 0)]
        A = sparse.csr_matrix((np.ones(len(sel)), (sel["pre"], sel["post"])), shape=(cn.n, cn.n))
        d = dijkstra(A, indices=sugar, unweighted=True, min_only=True)
        print(f"  excitatory hops ({label}): min {d[pam].min():.0f}, PAM within 3 hops "
              f"{(d[pam] <= 3).sum()}/{len(pam)}, unreachable {np.isinf(d[pam]).sum()}")
    for label, stim in (("awake, no sugar", None), ("awake + sugar 200 Hz", sugar)):
        net = LIFNetwork(cn.W, LIFParams(dt=0.5, da=1.0), seed=seed)
        net.run(800.0, background_hz=5.0, background_weight=90.0)
        _, i = net.run(1500.0, stim, 200.0, 1000.0, background_hz=5.0, background_weight=90.0)
        cnt = np.bincount(i, minlength=cn.n) / 1.5
        print(f"  {label:22s}: PAM {cnt[pam].mean():.2f} Hz, brain mean {len(i) / cn.n / 1.5:.2f} Hz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--glomeruli", type=int, default=5)
    a = ap.parse_args()
    cn = load_cached(a.data / "raw", a.data / "processed")
    pairs = load_all_pairs(a.data / "raw", a.data / "processed", cn.neurons)
    pops = populations(cn.neurons)
    print("== 1. odour code")
    odour_code(cn, pairs, pops, a.seed, a.glomeruli)
    print("== 2. reward path")
    reward_path(cn, pairs, a.seed)


if __name__ == "__main__":
    main()

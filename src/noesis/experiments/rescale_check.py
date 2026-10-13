"""Does restoring FAFB-like synapse counts make the BANC brain behave like Shiu et al.'s model?

Builds the BANC connectome twice -- as detected, and with every synapse multiplied by its brain region's
FAFB/BANC ratio (``neuropil_scale_factors``) -- and compares, with the same LIF parameters:

1. feeding: sugar-taste neurons at 50/100/200 Hz -> MN9 (Shiu et al. used ~100 Hz)
2. reach: how many neurons the sugar signal activates, and the reward dopamine neurons (PAM)
3. stability: the brain with no input, and with weak spontaneous background, must not run away
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from noesis.connectome import load_cached, neuropil_scale_factors
from noesis.lif import LIFNetwork, LIFParams


def _probe(cn, label: str) -> None:
    n = cn.neurons
    ctype = n["Primary Cell Type"].fillna("")
    sugar = np.flatnonzero(n["Function"].fillna("").str.contains("sugar").to_numpy())
    mn9 = np.flatnonzero(ctype.eq("MN9").to_numpy())
    pam = np.flatnonzero(ctype.str.startswith("PAM").to_numpy())
    print(f"--- {label}: {cn.W.nnz:,} synaptic pairs")
    for rate in (50.0, 100.0, 200.0):
        net = LIFNetwork(cn.W, LIFParams(dt=0.5, da=1.0))
        _, i = net.run(1000.0, sugar, rate, 400.0)
        cnt = np.bincount(i, minlength=cn.n)
        print(f"  sugar {rate:5.0f} Hz: MN9 {cnt[mn9].mean():6.1f} Hz | active neurons {np.unique(i).size:7,} | "
              f"PAM {cnt[pam].mean():5.2f} Hz ({(cnt[pam] > 5).sum()} cells > 5 Hz)", flush=True)
    for bg_hz, bg_w in ((0.0, 0.0), (1.0, 60.0), (5.0, 60.0)):
        net = LIFNetwork(cn.W, LIFParams(dt=0.5, da=1.0))
        net.run(500.0, background_hz=bg_hz, background_weight=bg_w)
        _, i = net.run(1000.0, background_hz=bg_hz, background_weight=bg_w)
        print(f"  background {bg_hz:3.0f} Hz x {bg_w:3.0f}: mean rate {len(i) / cn.n:6.2f} Hz, "
              f"neurons firing {np.unique(i).size:7,}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--reference", type=Path, help="FAFB connections table (default data/fafb/...)")
    a = ap.parse_args()
    ref = a.reference or a.data / "fafb" / "connections_princeton.csv.gz"
    raw, cache = a.data / "raw", a.data / "processed"

    factors = neuropil_scale_factors(raw / "connections_princeton.csv", ref)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "region_scale.json").write_text(json.dumps(factors, indent=1), encoding="utf-8")
    vals = np.array(list(factors.values()))
    print(f"{len(factors)} brain regions scaled; factor median {np.median(vals):.2f}, "
          f"range {vals.min():.2f}-{vals.max():.2f}; nerve cord unscaled")
    _probe(load_cached(raw, cache), "BANC as detected")
    _probe(load_cached(raw, cache, region_scale=factors), "BANC scaled to FAFB counts")


if __name__ == "__main__":
    main()

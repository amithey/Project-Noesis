"""Replicate the core Shiu et al. 2024 result: sugar neurons drive proboscis motor neuron MN9."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from noesis.connectome import load_cached
from noesis.lif import simulate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--ms", type=float, default=500.0)
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--weight", type=float, default=400.0)
    a = ap.parse_args()

    cn = load_cached(a.data / "raw", a.data / "processed")
    nr = cn.neurons
    sugar = cn.select(nr["Function"].fillna("").str.contains("sugar"))
    mn9 = cn.select(nr["Primary Cell Type"].eq("MN9"))
    print(f"{cn.n} neurons, {cn.W.nnz} synaptic pairs, {len(sugar)} sugar GRNs, {len(mn9)} MN9")

    for label, stim in (("control (no stimulus)", None), (f"sugar GRNs @ {a.rate:g} Hz", sugar)):
        t0 = time.time()
        _, i = simulate(cn.W, a.ms, stim, a.rate, a.weight)
        sec = a.ms / 1000
        active = np.unique(i).size
        mn9_hz = np.isin(i, mn9).sum() / max(len(mn9), 1) / sec
        print(f"{label:28s} active neurons: {active:6d}  MN9 mean rate: {mn9_hz:6.1f} Hz  "
              f"({time.time() - t0:.1f}s wall)")


if __name__ == "__main__":
    main()

"""Live brain-only demo for the brain map: sugar on, sugar off, and what the brain does with it.

No body, so it runs near real time. Every ``--period`` seconds the sugar-taste neurons switch on or off;
the map shows which neurons fire, and its status line shows MN9 (the feeding motor neuron) and how many
reward dopamine neurons (PAM) respond. ``--scaled`` uses BANC synapses corrected to FAFB counts.

    python -m noesis.experiments.live_brain --scaled --brain-map <file>
    python -m noesis.viz.brain_map --watch <file>          (in a second window)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from noesis.connectome import load_cached, neuropil_scale_factors
from noesis.lif import LIFNetwork, LIFParams
from noesis.motor import RateTracker
from noesis.viz.brain_map import write_activity


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--scaled", action="store_true", help="BANC synapses corrected to FAFB counts")
    ap.add_argument("--reference", type=Path, help="FAFB connections (default data/fafb/...)")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--period", type=float, default=2.0, help="seconds of sugar off, then on")
    ap.add_argument("--sugar-hz", type=float, default=100.0)
    ap.add_argument("--brain-map", type=Path, required=True)
    ap.add_argument("--realtime", action="store_true", help="never run faster than real time")
    a = ap.parse_args()

    raw, cache = a.data / "raw", a.data / "processed"
    kw = {}
    if a.scaled:
        ref = a.reference or a.data / "fafb" / "connections_princeton.csv.gz"
        kw["region_scale"] = neuropil_scale_factors(raw / "connections_princeton.csv", ref)
    cn = load_cached(raw, cache, **kw)
    n = cn.neurons
    ctype = n["Primary Cell Type"].fillna("")
    sugar = np.flatnonzero(n["Function"].fillna("").str.contains("sugar").to_numpy())
    mn9 = np.flatnonzero(ctype.eq("MN9").to_numpy())
    pam = np.flatnonzero(ctype.str.startswith("PAM").to_numpy())
    brain = "BANC scaled to FAFB counts" if a.scaled else "BANC as detected"
    print(f"{brain}: {cn.n:,} neurons, {cn.W.nnz:,} pairs; writing {a.brain_map}", flush=True)

    net = LIFNetwork(cn.W, LIFParams(dt=0.5, da=1.0))
    rates = RateTracker(cn.n, tau_ms=100.0)
    chunk_ms = 20.0
    t0 = time.time()
    for k in range(round(a.seconds * 1000 / chunk_ms)):
        t = k * chunk_ms / 1000
        on = int(t / a.period) % 2 == 1
        _, spikes = net.run(chunk_ms, sugar if on else None, a.sugar_hz, 400.0)
        r = rates.update(spikes, chunk_ms)
        note = (f"{brain} | sugar {'ON ' if on else 'off'} ({a.sugar_hz:.0f} Hz) | "
                f"MN9 {r[mn9].mean():5.1f} Hz | PAM > 5 Hz: {(r[pam] > 5).sum():3d} of {len(pam)}")
        write_activity(a.brain_map, r, t, awake=True, note=note)
        if k % 25 == 0:
            print(f"t={t:5.1f}s  {note}", flush=True)
        if a.realtime:
            time.sleep(max(0.0, t - (time.time() - t0)))


if __name__ == "__main__":
    main()

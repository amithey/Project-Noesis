"""Compare raw and FAFB/MANC-corrected gait timing (3 seconds, three seeds).

Run with PYTHONPATH=src and python -m noesis.experiments.gait_rhythm --data PATH.
All source data/caches are read-only. Missing caches are built under --cache-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from noesis.connectome import VNC_GROUPS, corrected_scale, load_cached
from noesis.gait import LEGS, antagonist_alternation, interleg_phase, rhythm_spectrum
from noesis.lif import LIFNetwork, LIFParams
from noesis.motor import leg_motor_table

EXPECTED = (
    "Before running: synthetic 10 Hz tripod should pass the shuffled spectral null, "
    "have negative antagonist correlation and tripod score near 1. Tonic Poisson "
    "should fail rhythmicity; co-contraction should correlate positively. "
    "Brain hypothesis: tonic/coactive motor output, weak or absent coordinated tripod. "
    "Correction may increase firing without restoring gait. Rejecting the stationary "
    "Poisson null alone is not evidence of walking."
)


def read_brain(data, cache, scale):
    """Use shared caches only when complete, avoiding load_cached's write fallback."""
    suffix = ""
    if scale:
        digest = hashlib.md5(json.dumps(sorted(scale.items())).encode()).hexdigest()[:8]
        suffix = f"_scaled_{digest}"
    shared = data / "processed"
    complete = (shared / "neurons.parquet").is_file() and (
        shared / f"W_min5{suffix}.npz").is_file()
    return load_cached(data / "raw", shared if complete else cache, region_scale=scale)


def measure(times, idx, table, duration, bin_ms, shuffles, seed):
    # Filter early: simulation output can contain millions of unrelated spikes.
    keep = np.isin(idx, table.idx.to_numpy())
    times, idx = times[keep], idx[keep]
    pooled, groups = {}, {}
    for leg in LEGS:
        members = table.loc[table.leg.eq(leg), "idx"].to_numpy()
        pooled[leg] = rhythm_spectrum(times, idx, members, duration, bin_ms,
                                     n_shuffles=shuffles, seed=seed)
        # A homologous muscle is necessary: pooling antagonists can cancel rhythm.
        groups[leg] = table.loc[table.leg.eq(leg) & table.muscle.isin(
            ["tibia_flexor", "accessory_tibia_flexor"]), "idx"].to_numpy()
    antagonists = antagonist_alternation(times, idx, table, duration, bin_ms)
    correlations = [r["correlation"] for r in antagonists.values()
                    if r["correlation"] is not None]
    return {"motor_spikes": len(times), "motor_neurons": len(table),
            "motor_rate_windows_hz": (np.histogram(times, bins=np.linspace(
                0, duration, 7))[0] / (len(table) * duration * .001 / 6)).tolist(),
            "mean_motor_hz": len(times) / (len(table) * duration * .001),
            "pooled_leg_spectra": pooled,
            "antagonists": antagonists,
            "antagonist_summary": {"valid_joints": len(correlations),
                                   "median_correlation": float(np.median(correlations))
                                   if correlations else None,
                                   "positive_joints": sum(c > 0 for c in correlations)},
            "tibia_flexor_phase": interleg_phase(times, idx, groups, duration, bin_ms,
                                                n_shuffles=shuffles, seed=seed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".gait-cache"))
    parser.add_argument("--output", type=Path, help="Optional new JSON results file")
    parser.add_argument("--duration-ms", type=float, default=3000)
    parser.add_argument("--bin-ms", type=float, default=10)
    parser.add_argument("--shuffles", type=int, default=199)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    data = args.data.resolve()
    cache = args.cache_dir.resolve()
    if cache == data or data in cache.parents:
        parser.error("cache-dir must be outside read-only data")
    if args.output and (args.output.exists() or data in args.output.resolve().parents):
        parser.error("output must be a new file outside read-only data")
    manc = [data / "manc" / "connections_princeton.csv",
            data.parent / "MANC" / "connections_princeton.csv"]
    if not any(p.is_file() for p in manc):
        parser.error("MANC reference missing; refusing brain-only correction")
    print(EXPECTED, flush=True)
    scale = corrected_scale(data)
    vnc_names = {name for banc, _ in VNC_GROUPS.values() for name in banc}
    if not vnc_names.intersection(scale):
        parser.error("no MANC correction factors were computed")
    results = {"expected_before_run": EXPECTED,
               "settings": {"duration_ms": args.duration_ms, "bin_ms": args.bin_ms,
                            "shuffles": args.shuffles, "seeds": args.seeds,
                            "dt_ms": .5, "da": 1.0, "background_hz": 3.0,
                            "background_weight": 60.0, "burn_in_ms": 0},
               "scale": scale, "runs": []}
    for label, factors in (("raw", None), ("corrected", scale)):
        cn = read_brain(data, cache, factors)
        table = leg_motor_table(cn.neurons)
        for seed in args.seeds:
            print(f"Running {label}, seed={seed}, n={cn.n}, edges={cn.W.nnz}", flush=True)
            net = LIFNetwork(cn.W, LIFParams(dt=.5, da=1.0), seed=seed)
            times, idx = net.run(args.duration_ms, background_hz=3.0, background_weight=60.0)
            run = {"brain": label, "seed": seed, "total_spikes": len(times),
                   **measure(times, idx, table, args.duration_ms, args.bin_ms,
                             args.shuffles, seed)}
            results["runs"].append(run)
            print(json.dumps(run, allow_nan=False), flush=True)
            # Preserve completed runs if a later simulation is interrupted.
            if args.output:
                args.output.write_text(json.dumps(results, indent=2, allow_nan=False),
                                       encoding="utf-8")
            del net, times, idx
        del cn


if __name__ == "__main__":
    main()

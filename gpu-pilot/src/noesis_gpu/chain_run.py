"""Heavy-run driver for the fixed-weight chain diagnostic (NOT started by its author; needs the shared host lease).

One process per seed runs the three arms (off, zero, active) one after another on a freshly reset body, writes one
``chain-<arm>.npz`` per arm and a ``report.json`` with the registered invariance verdict. It never trains, never writes
weights, and stops at the first technical failure. See docs/workstreams/claude/chain-diagnostic/protocol.md.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

from .chain_diagnostic import ARMS, ChainBrain, invariance_report, run_arm
from .run import file_digest, source_digest, timing_steps


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="cpu")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--control-ms", type=float, default=5.0)
    ap.add_argument("--physics-dt", type=float, default=0.0001)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--exploration-mv-ms", type=float, default=1.5)
    ap.add_argument("--wall-seconds", type=float, default=900)
    ap.add_argument("--noesis-source", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="new directory; never overwrites")
    return ap


def run(args):
    n_chunks, physics_steps = timing_steps(args.seconds, args.control_ms, args.physics_dt)
    source = args.noesis_source.resolve()
    if not (source / "noesis" / "lif.py").is_file():
        raise ValueError("--noesis-source must point to the intended src directory")
    sys.path.insert(0, str(source))
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "initializing", "learning": False, "arms": list(ARMS), "seed": args.seed,
              "seconds": args.seconds, "control_ms": args.control_ms, "exploration_mv_ms": args.exploration_mv_ms,
              "noesis_source_sha256": source_digest(source), "checkpoint_sha256": file_digest(args.checkpoint),
              "model_promoted": False, "walking_proven": False, "completed_arms": []}
    started, arrays = time.perf_counter(), {}
    try:
        from noesis.embodiment import load_training_brain

        from .backend import FlyGymBatch

        cn, _ = load_training_brain(args.data.resolve(), args.output / "brain-cache")
        for arm in ARMS:
            if time.perf_counter() - started >= args.wall_seconds:
                report["status"] = "incomplete_wall_budget"
                break
            body = FlyGymBatch(1, args.backend, args.physics_dt)
            brain = ChainBrain(cn, body.rest, body.lo, body.hi, checkpoint=args.checkpoint, arm=arm, seed=args.seed,
                               exploration_mv_ms=args.exploration_mv_ms)
            arrays[arm] = run_arm(brain, body, n_chunks, physics_steps, args.control_ms)
            report.setdefault("initial_weights_sha256", {})[arm] = brain.initial_weights_sha256
            np.savez_compressed(args.output / f"chain-{arm}.npz", **arrays[arm])
            report["completed_arms"].append(arm)
            del brain, body
            gc.collect()
        else:
            report["invariance_off_vs_zero"] = invariance_report(arrays["off"], arrays["zero"])
            report["weights_identical_across_arms"] = len(set(report["initial_weights_sha256"].values())) == 1
            report["status"] = ("completed_chain_diagnostic" if report["invariance_off_vs_zero"]["identical"]
                                and report["weights_identical_across_arms"] else "invariance_failed")
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["total_wall_seconds"] = time.perf_counter() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                                                  encoding="utf-8")
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        report = run(args)
    except (ValueError, RuntimeError, MemoryError, ImportError, FileExistsError) as error:
        print(f"chain diagnostic: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "completed_chain_diagnostic" else 3


if __name__ == "__main__":
    raise SystemExit(main())

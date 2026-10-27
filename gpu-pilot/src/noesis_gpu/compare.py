"""Compare completed one-world CPU/GPU pilots without promoting a training model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(cpu, gpu):
    cpu, gpu = Path(cpu), Path(gpu)
    a, b = (json.loads((directory / "report.json").read_text(encoding="utf-8"))
            for directory in (cpu, gpu))
    for report, backend in ((a, "cpu"), (b, "gpu")):
        if (report["status"] != "completed_transport_pilot" or report["worlds"] != 1
                or report["backend"] != backend or report["learning"]
                or report["motor_exploration"] != 0):
            raise ValueError("Comparison requires completed one-world frozen CPU and GPU pilots")
        required = {"upstream_commit", "requested_simulated_seconds_per_world", "control_ms",
                    "physics_dt", "completed_control_chunks", "seeds", "bg_hz", "bg_weight",
                    "senses", "rng_schedule", "body", "packages"}
        if not required.issubset(report):
            raise ValueError("Pilot report is missing provenance or settings")
    for key in ("upstream_commit", "controller", "requested_simulated_seconds_per_world",
                "control_ms", "physics_dt", "completed_control_chunks", "seeds", "bg_hz",
                "bg_weight", "senses", "rng_schedule", "body", "noesis_source_sha256",
                "checkpoint_sha256", "neurons", "edges", "region_scale"):
        if a.get(key) != b.get(key):
            raise ValueError(f"Pilot settings do not match: {key}")
    for package in ("flygym", "mujoco", "numba", "numpy"):
        if a["packages"].get(package) != b["packages"].get(package):
            raise ValueError(f"Runtime versions do not match: {package}")
    if a["controller"] == "brain" and not (a["weights_unchanged"] and b["weights_unchanged"]):
        raise ValueError("Frozen weights changed")
    metrics = {}
    with np.load(cpu / "observations.npz", allow_pickle=False) as left, np.load(
            gpu / "observations.npz", allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise ValueError("Observation channels do not match")
        for key in left.files:
            x, y = left[key], right[key]
            if (x.shape != y.shape or x.ndim < 2 or x.shape[0] != a["completed_control_chunks"]
                    or x.shape[1] != 1 or not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise ValueError(f"Invalid observation arrays: {key}")
            difference = np.abs(x.astype(float) - y.astype(float))
            metrics[key] = {"mean_absolute_difference": float(difference.mean()),
                            "max_absolute_difference": float(difference.max()),
                            "shape": list(x.shape)}
    return {"status": "paired_transport_comparison", "cpu": str(cpu.resolve()),
            "gpu": str(gpu.resolve()), "metrics": metrics, "model_promoted": False,
            "walking_proven": False, "v1_body_equivalence_verified": False,
            "note": "Descriptive numerical differences, not a gait or learning acceptance gate"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", type=Path, required=True)
    ap.add_argument("--gpu", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = compare(args.cpu, args.gpu)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

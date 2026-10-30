"""Audit completed frozen CPU/GPU observations; never selects deployment weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .gait_metrics import SCHEMA_VERSION, compute_gait_summary, extract_leg_step_counts
from .learning import standing_mask

MATCHED_SETTINGS = (
    "backend", "controller", "worlds", "control_ms", "physics_dt", "bg_hz", "bg_weight",
    "upstream_commit", "packages", "body", "noesis_source_sha256", "senses",
    "external_vision_and_chemical_senses", "rng_schedule",
)


def summarize(directory, *, warmup_ms=50):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8-sig"))
    if (report.get("status") != "completed_transport_pilot"
            or report.get("controller") != "brain" or report.get("worlds") != 1
            or report.get("learning") is not False or report.get("motor_exploration") != 0
            or report.get("weights_unchanged") is not True):
        raise ValueError("Require one completed frozen brain world with unchanged weights")
    dt_ms = report["control_ms"]
    if not np.isfinite(dt_ms) or dt_ms <= 0 or not np.isfinite(warmup_ms) or warmup_ms <= 0:
        raise ValueError("Require positive finite control and warmup durations")
    start = round(warmup_ms / dt_ms)
    total = round(5000 / dt_ms)
    if (not np.isclose(start * dt_ms, warmup_ms) or not 1 <= start < total
            or not np.isclose(total * dt_ms, 5000)
            or report.get("completed_control_chunks") != total
            or report.get("requested_simulated_seconds_per_world") != 5
            or report.get("simulated_seconds_per_world") != 5):
        raise ValueError("Require full five-second horizon with integral warmup and no cutoff")
    shapes = {"position": (3,), "up_body": (3,), "heading": (3,),
              "feet_position": (6, 3), "foot_load": (6,)}
    with np.load(directory / "observations.npz", allow_pickle=False) as archive:
        observation = {key: archive[key].copy() for key in shapes}
    for key, shape in shapes.items():
        value = observation[key]
        if value.shape != (total, 1, *shape) or not np.isfinite(value).all():
            raise ValueError(f"Invalid or truncated observation: {key}")
        observation[key] = value[:, 0]
    if np.any(observation["foot_load"] < 0):
        raise ValueError("Foot loads cannot be negative")
    standing = standing_mask(observation)
    support = observation["foot_load"] > 1e-5
    speed = np.linalg.norm(np.diff(observation["feet_position"], axis=0)[..., :2], axis=-1)
    speed /= dt_ms / 1000
    counts = support[1:].sum(axis=-1)
    slip = np.divide((speed * support[1:]).sum(axis=-1), counts,
                     out=np.zeros(total - 1), where=counts > 0)
    measured = slice(start, None)
    gait = compute_gait_summary(observation, dt_ms, warmup_samples=start)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": report["seeds"][0], "seconds": 5.0, "warmup_ms": warmup_ms,
        "warmup_samples": start, "measured_samples": total - start,
        "warmup_standing_fraction": float(standing[:start].mean()),
        "standing_fraction": float(standing[measured].mean()),
        "mean_stance_slip_mm_s": float(slip[start - 1:].mean()),
        "mean_supporting_feet": float(support[measured].sum(axis=-1).mean()),
        "mean_uprightness": float(observation["up_body"][measured, 2].mean()),
        "single_probe_standing_check": bool(standing[measured].mean() >= .9),
        "heading_forward_distance_mm": gait["heading_forward_distance_mm"],
        "net_displacement_mm": gait["net_displacement_mm"],
        "net_forward_displacement_mm": gait["net_forward_displacement_mm"],
        "total_leg_steps": gait["total_leg_steps"],
        "mean_steps_per_leg": gait["mean_steps_per_leg"],
        "leg_step_cycles": gait["leg_step_cycles"],
        "leg_step_counts": gait["leg_step_counts"],
        "tripod_coordination": gait["tripod_coordination"],
        "locomotion_regime": gait["locomotion_regime"],
        "support_measurement": "tarsus5-to-floor-only",
        "slip_approximation": "horizontal tarsus5 body-center finite differences at end-of-step contact",
        "weights_unchanged": True, "learning": False, "motor_exploration": 0,
        "walking_proven": False, "model_promoted": False,
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "report_sha256": hashlib.sha256((directory / "report.json").read_bytes()).hexdigest(),
        "observations_sha256": hashlib.sha256((directory / "observations.npz").read_bytes()).hexdigest(),
    }
    return report, summary


def compare(parent, candidate):
    a, before = summarize(parent)
    b, after = summarize(candidate)
    keys = (*MATCHED_SETTINGS, "seeds")
    if any(key not in a or key not in b or a[key] != b[key] for key in keys):
        raise ValueError("Frozen comparison requires matched source, body, senses, seeds and settings")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_matched_frozen_pair", "cases": [before, after],
        "standing_change_percentage_points": 100 * (after["standing_fraction"] - before["standing_fraction"]),
        "slip_change_percent": (100 * (after["mean_stance_slip_mm_s"] / before["mean_stance_slip_mm_s"] - 1)
                                if before["mean_stance_slip_mm_s"] > 0 else None),
        "heading_forward_change_mm": after["heading_forward_distance_mm"] - before["heading_forward_distance_mm"],
        "total_steps_change": after["total_leg_steps"] - before["total_leg_steps"],
        "locomotion_regime_before": before["locomotion_regime"],
        "locomotion_regime_after": after["locomotion_regime"],
        "same_conditions_verified": True, "only_one_frozen_seed_pair": True,
        "stage_advanced": False, "model_promoted": False, "walking_proven": False,
        "control_is_starting_training_bank_not_untrained_original": True,
    }


def standing_gate(pairs):
    """Two distinct training probes certify standing only, never deployment or gait."""
    if len(pairs) != 2:
        raise ValueError("Standing gate requires two matched frozen pairs")
    results = [compare(parent, candidate) for parent, candidate in pairs]
    seeds = [result["cases"][0]["seed"] for result in results]
    if len(set(seeds)) != 2 or set(seeds) & {10, 11, 200, 201, 202}:
        raise ValueError("Require distinct training seeds outside held-out final validation")
    reports = [summarize(directory)[0] for pair in pairs for directory in pair]
    if any(any(report[key] != reports[0][key] for key in MATCHED_SETTINGS)
           for report in reports[1:]):
        raise ValueError("All standing probes must use the same body and source")
    for bank in (0, 1):
        hashes = [result["cases"][bank]["checkpoint_sha256"] for result in results]
        if any(not digest for digest in hashes) or len(set(hashes)) != 1:
            raise ValueError("Same parent and same candidate required across both seeds")
    return {
        "standing_training_gate_passed": all(r["cases"][1]["single_probe_standing_check"]
                                             for r in results),
        "seeds": seeds, "pairs": results, "training_only": True,
        "stage_advanced": False, "model_promoted": False, "walking_proven": False,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = compare(args.parent, args.candidate)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Recompute EC1 lineage, learning accounting, frozen metrics and retention."""
import argparse
from pathlib import Path
import json

import numpy as np
import run_ec1_learning as run
from ec1_contract import TRAIN, BANKS, lineage, jobs, decision


def audit(prepared):
    plan = run.read(prepared)
    out = Path(plan["out"])
    result = run.read(out / "result.json")
    if (result["status"] != "completed_EC1" or result["seconds"] > run.LIMIT or
            [item["job"] for item in result["jobs"]] != jobs() or not all(result[key] for key in
                ("source_inputs_created_unchanged", "tree_empty", "lease_released", "awake_released"))):
        raise ValueError("EC1 allocation not complete and clean")
    if (Path(plan["runtime"]) / "work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock").exists():
        raise ValueError("Another heavy allocation owns the host")
    pins = {**plan["pins"], **result["created_hashes"], str(prepared): run.sha(prepared)}
    run.base.ss1.base.check_pins(pins)
    _, old, _ = run.base.ss1.base.legacy(Path(plan["runtime"]))
    banks = {"parent": {}, **{bank: {} for bank in BANKS}}
    for seed in TRAIN:
        original = run.cell(run.base.ss1.reference(plan["prior"], seed), old)
        if original != plan["parents"][str(seed)]:
            raise ValueError("Reused parent metric changed")
        banks["parent"][seed] = original
    for item in result["jobs"]:
        job = item["job"]
        run.checked_receipt(out, item, 600 if job["phase"] == "train" else 300)
        measured = run.validate_worker(plan, job, old)
        if job["phase"] == "frozen":
            banks[job["bank"]][job["seed"]] = measured
    for bank in BANKS:
        source_bank, seed = lineage(bank)
        reference = (run.base.ss1.reference(plan["prior"], seed) if source_bank == "parent"
                     else out / f"frozen-{source_bank}-{seed}")
        frozen = run.base.ss1.arrays(reference / "observations.npz")
        training = run.base.ss1.arrays(out / f"train-{bank}/training-observations.npz")
        for key in set(frozen) & set(training):
            np.testing.assert_array_equal(training[key][:10], frozen[key][:10, 0], err_msg=f"{bank}:{key}")
        flags = training["learning_active"]
        np.testing.assert_array_equal(flags, np.arange(5000) >= 10)
        packet = run.base.ss1.arrays(out / f"train-{bank}/exploration-log.npz")
        np.testing.assert_array_equal(packet["control_chunk_index"], np.arange(10, 5000))
        if len(packet["drive"]) != 4990 or len(packet["credit"]) != 4990 or np.max(np.abs(packet["drive"])) > 1.5:
            raise ValueError("Incorrect bounded native exploration history")
    science = decision(banks)
    saved = run.read(out / "science-private.json")
    actual = json.loads(json.dumps(dict(banks=banks, decision=science)))
    if saved != actual or any(result[key] != value for key, value in science.items()):
        raise ValueError("Final science differs from raw replay")
    run.base.ss1.base.check_pins(pins)
    public = dict(status="completed_EC1_audit", complete_workers=22, complete_training_births=3,
                  actual_weight_updates_audited=14970, all_checkpoint_transfers_verified=True,
                  episode_trace_reset_verified=True, warmup_replay_exact=True,
                  frozen_contexts=6, **science,
                  limits="TRAIN-context weight carry with unequal previous exposure of reset control. Independent physical validation remains required.")
    target = out.with_name(out.name + "-minimal-result.json")
    if target.exists():
        raise ValueError("Final audit result exists")
    run.write(target, public)
    print(json.dumps(public))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    args = parser.parse_args()
    audit(args.prepared.resolve())

"""Run E3: three v9 weight-0.5 births and six frozen development probes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from compare_exploration import digest, tree_digest, write
from run_e2_reproduction import EXPECTED, PARENT

CORE = "42c20769376bda5694b4c1c66ca75e52cbd9f1f4"


def audit_bank(path, parent):
    with np.load(path, allow_pickle=False) as bank, np.load(parent, allow_pickle=False) as original:
        for key in ("topology", "motor_indices", "initial_weights", "parameters", "rule"):
            if not np.array_equal(bank[key], original[key]):
                raise ValueError(f"Checkpoint invariant changed: {key}")
        weights, reference = bank["weights"], bank["initial_weights"]
        if (not np.isfinite(weights).all() or not np.array_equal(np.sign(weights), np.sign(reference))
                or np.any(np.abs(weights) > 1.5 * np.abs(reference) + 1e-12)
                or np.any(np.abs(weights) < .5 * np.abs(reference) - 1e-12)):
            raise ValueError("Checkpoint violates original-reference bounds or signs")
    return digest(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shared-root", type=Path, required=True)
    ap.add_argument("--core-source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root, source, output = args.shared_root.resolve(), args.core_source.resolve(), args.output.resolve()
    checkout = Path(__file__).resolve().parents[1]
    pilot = checkout / "gpu-pilot/src"
    parent = root / "gpu-pilot/outputs/standing-001-train-11000/weights-last.npz"
    j0_path = root / "gpu-pilot/outputs/e2-reproduction-20261028-1001/result.json"
    j0 = json.loads(j0_path.read_text())
    if j0["status"] != "reproduction_passed" or j0["checkpoint_sha256"] != EXPECTED:
        raise ValueError("J0 reproduction prerequisite failed")
    if digest(parent) != PARENT:
        raise ValueError("Parent hash mismatch")
    pin = subprocess.check_output(["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip()
    if pin != CORE:
        raise ValueError("Wrong experimental core")
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    hashes = {"core": tree_digest(source), "pilot": tree_digest(pilot)}
    remaining_budget = 3300
    token = uuid.uuid4().hex
    lease = {"owner": "codex", "host": os.environ.get("COMPUTERNAME"), "pid": os.getpid(),
             "token": token, "commit": commit, "core_commit": pin, "source_hashes": hashes,
             "output": str(output), "start_unix": time.time(),
             "deadline_unix": time.time() + remaining_budget, "budget_seconds": remaining_budget,
             "parent_sha256": PARENT, "j0_result_sha256": digest(j0_path),
             "training_seeds": [1001, 1000, 1002], "probe_seeds": [10006, 10007],
             "simulation_seconds": 5, "job_wall_seconds": 300, "process_wall_seconds": 330,
             "objective": "curriculum-gripped-cycle-v9", "exploration_mv_ms": 1.5,
             "cycle_weight": 0.5, "actual_operator": "codex", "experiment": "E3",
             "primary_progress_metric": "heading_forward_distance_mm",
             "sensitivity_progress_metric": "net_forward_displacement_mm",
             "model_promoted": False, "walking_proven": False}
    lock = root / "gpu-pilot/outputs/codex-training-host.lock"
    with os.fdopen(os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "w") as stream:
        json.dump(lease, stream)
    started, created, receipts = time.monotonic(), False, []
    outcome = {"status": "running", "model_promoted": False, "walking_proven": False}
    try:
        raw = subprocess.check_output(["powershell", "-NoProfile", "-Command",
              "$ErrorActionPreference='Stop'; Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"], text=True)
        inventory = json.loads(raw) if raw.strip() else []
        if isinstance(inventory, dict):
            inventory = [inventory]
        if any(item["ProcessId"] != os.getpid() for item in inventory):
            raise RuntimeError("Another Python worker is active")
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write(output / "protocol.json", lease)
        common = ["--backend", "cpu", "--seconds", "5", "--wall-seconds", "300",
                  "--noesis-source", str(source), "--data", str(root / "data")]

        def verify_sources():
            if digest(parent) != PARENT or tree_digest(source) != hashes["core"] or tree_digest(pilot) != hashes["pilot"]:
                raise RuntimeError("Parent or source modified during experiment")

        def execute(name, module, options, expected_status):
            verify_sources()
            remaining = remaining_budget - (time.monotonic() - started)
            if remaining < 330:
                raise TimeoutError("Insufficient remaining budget for another full job")
            command = [sys.executable, "-m", module, *common, *options, "--output", str(output / name)]
            print(f"Starting {name}", flush=True)
            with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=330, check=False)
            report = json.loads((output / name / "report.json").read_text())
            receipt = {"name": name, "exit_code": child.returncode, "status": report["status"],
                       "elapsed_seconds": time.monotonic() - started}
            receipts.append(receipt)
            write(output / "receipts.json", receipts)
            if child.returncode or report["status"] != expected_status:
                raise RuntimeError(f"Job did not complete: {name}")
            verify_sources()
            return report, receipt

        for seed in (1001, 1000, 1002):
            name = f"train-v9-{seed}"
            report, receipt = execute(name, "noesis_gpu.train",
                ["--checkpoint", str(parent), "--profile", "gripped-cycle", "--seed", str(seed),
                 "--exploration-mv-ms", "1.5", "--cycle-weight", "0.5"], "completed_training_birth")
            if (report["learning_updates"] != 990 or report["reward_objective"] != lease["objective"]
                    or report["cycle_scoring"]["steps"] != 1000
                    or report["cycle_scoring"]["cycle_weight"] != .5):
                raise ValueError("Training objective/updates/tracker mismatch")
            checkpoint = output / name / "weights-last.npz"
            receipt["checkpoint_sha256"] = audit_bank(checkpoint, parent)
            receipt["cycle_scoring"] = report["cycle_scoring"]
            write(output / "receipts.json", receipts)
            for probe in (10006, 10007):
                pname = f"frozen-v9-{seed}-{probe}"
                frozen, _ = execute(pname, "noesis_gpu.run", ["--checkpoint", str(checkpoint),
                        "--seed", str(probe), "--worlds", "1"], "completed_transport_pilot")
                if (frozen.get("learning") is not False or frozen.get("motor_exploration") != 0
                        or frozen.get("weights_unchanged") is not True
                        or frozen.get("completed_control_chunks") != 1000):
                    raise ValueError("Frozen probe integrity or duration mismatch")
                from noesis_gpu.evaluate import summarize
                _, summary = summarize(output / pname)
                write(output / f"{pname}-summary.json", summary)
                if digest(checkpoint) != receipt["checkpoint_sha256"]:
                    raise ValueError("Frozen probe modified checkpoint")
        outcome.update(status="completed_e3_jobs_pending_independent_review")
    except BaseException as error:
        outcome.update(status="stopped", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if created:
            write(output / "result.json", {**outcome, "jobs_recorded": len(receipts),
                  "wall_seconds": time.monotonic() - started,
                  "j0_wall_seconds": j0["wall_seconds"]})
        if json.loads(lock.read_text()).get("token") == token:
            lock.unlink()


if __name__ == "__main__":
    main()

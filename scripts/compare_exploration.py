"""Run one preregistered matched exploration pilot; never promotes a checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shared-root", type=Path, required=True)
    ap.add_argument("--core-source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from noesis_gpu.run import source_digest

    root, source, output = args.shared_root.resolve(), args.core_source.resolve(), args.output.resolve()
    parent = root / "gpu-pilot/outputs/gripped-v8-003-train-1001/weights-last.npz"
    expected = "626bde8dce2f996ef181650ba4c3812abdc0dfdd3a969bbfc1e0e1ed57b95ff1"
    if digest(parent) != expected:
        raise ValueError("Parent checkpoint hash mismatch")
    pin = subprocess.check_output(["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip()
    if pin != "41b5fe6534ed4c8f0c21a2e123c8d518b3bf5953":
        raise ValueError("Unexpected core pin")
    core_digest = source_digest(source)
    pilot_source = Path(__file__).resolve().parents[1] / "gpu-pilot/src"
    pilot_digest = source_digest(pilot_source)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    lock = root / "gpu-pilot/outputs/codex-training-host.lock"
    token = uuid.uuid4().hex
    # O_EXCL gives cooperating callers one host-wide lease; stale leases require inspection.
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    lease = {"owner": "Codex", "pid": os.getpid(), "token": token,
             "host": os.environ.get("COMPUTERNAME"), "commit": commit,
             "parent_sha256": expected, "output": str(output), "budget_seconds": 3600}
    with os.fdopen(descriptor, "w") as handle:
        json.dump(lease, handle)
    started = time.monotonic()
    receipts = []
    try:
        processes = subprocess.check_output([
            "powershell", "-NoProfile", "-Command",
            ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
             "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
        ], text=True)
        inventory = json.loads(processes) if processes.strip() else []
        if isinstance(inventory, dict):
            inventory = [inventory]
        if any(p["ProcessId"] != os.getpid() for p in inventory):
            raise RuntimeError("Another Python process is active; inspect before training")
        output.mkdir(parents=True, exist_ok=False)
        write(output / "protocol.json", {**lease, "core_commit": pin,
              "core_source_sha256": core_digest, "pilot_source_sha256": pilot_digest,
              "training_seed": 2100, "probe_seeds": [10008, 10009],
              "objective": "standing-gate-smooth-grip-v6", "amplitudes": [1.5, 1.0],
              "simulation_seconds": 5, "job_wall_seconds": 400,
              "purpose": "new matched method pilot, not a fourth curriculum round",
              "model_promoted": False, "walking_proven": False})
        common = ["--backend", "cpu", "--seconds", "5", "--wall-seconds", "400",
                  "--noesis-source", str(source), "--data", str(root / "data")]

        def execute(name, module, options):
            remaining = 3600 - (time.monotonic() - started)
            if remaining < 5:
                raise TimeoutError("Aggregate experiment wall budget exhausted")
            command = [sys.executable, "-m", module, *common, *options,
                       "--output", str(output / name)]
            print(f"Starting {name}", flush=True)
            with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=min(400, remaining), check=False)
            report = json.loads((output / name / "report.json").read_text())
            receipts.append({"name": name, "returncode": result.returncode,
                             "status": report["status"], "elapsed_seconds": time.monotonic()-started})
            write(output / "receipts.json", receipts)
            if result.returncode or report["status"] not in {"completed_training_birth", "completed_transport_pilot"}:
                raise RuntimeError(f"Incomplete or failed job: {name}")
            if digest(parent) != expected or source_digest(source) != core_digest or source_digest(pilot_source) != pilot_digest:
                raise RuntimeError("Parent or source changed during experiment")
            print(f"Completed {name}", flush=True)

        for arm, amplitude in (("control", "1.5"), ("lower", "1.0")):
            execute(f"train-{arm}", "noesis_gpu.train", ["--checkpoint", str(parent),
                    "--profile", "continuous-walking", "--seed", "2100",
                    "--exploration-mv-ms", amplitude])
        for bank in ("parent", "control", "lower"):
            checkpoint = parent if bank == "parent" else output / f"train-{bank}/weights-last.npz"
            for seed in (10008, 10009):
                execute(f"frozen-{bank}-{seed}", "noesis_gpu.run",
                        ["--checkpoint", str(checkpoint), "--seed", str(seed), "--worlds", "1"])
        from noesis_gpu.evaluate import compare

        comparisons = {arm: [compare(output / f"frozen-parent-{seed}",
                                    output / f"frozen-{arm}-{seed}")
                             for seed in (10008, 10009)] for arm in ("control", "lower")}
        write(output / "comparison.json", {"status": "completed_matched_method_pilot",
              "comparisons": comparisons, "model_promoted": False, "walking_proven": False,
              "parent_unchanged": digest(parent) == expected,
              "wall_seconds": time.monotonic() - started})
    finally:
        if output.is_dir():
            write(output / "worker-exit.json", {"pid": os.getpid(), "wall_seconds": time.monotonic()-started,
                  "jobs_completed": len(receipts), "comparison_complete": (output / "comparison.json").is_file()})
        if json.loads(lock.read_text()).get("token") == token:
            lock.unlink()


if __name__ == "__main__":
    main()

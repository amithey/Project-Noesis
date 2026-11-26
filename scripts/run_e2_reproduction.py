"""Bounded E2 J0 reproduction gate; never starts v9 or promotes a checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from compare_exploration import digest, tree_digest, write

PARENT = "fb80cfcc3018329209784751570d7ffb1ed33aa9ef48fceaa85588862f7626a8"
EXPECTED = "626bde8dce2f996ef181650ba4c3812abdc0dfdd3a969bbfc1e0e1ed57b95ff1"
CORE = "54828ca7e47d0c987e7324d5b4ec36fda2f37aff"


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
    pin = subprocess.check_output(["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip()
    if pin != CORE or digest(parent) != PARENT:
        raise ValueError("Core or parent does not match registered protocol")
    hashes = {"core_source_sha256": tree_digest(source), "pilot_source_sha256": tree_digest(pilot)}
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    token = uuid.uuid4().hex
    lock = root / "gpu-pilot/outputs/codex-training-host.lock"
    lease = {"owner": "codex", "host": os.environ.get("COMPUTERNAME"), "pid": os.getpid(),
             "token": token, "commit": commit, "core_commit": pin, "output": str(output),
             "start_unix": time.time(), "deadline_unix": time.time() + 330,
             "parent_sha256": PARENT, "training_seed": 1001,
             "objective": "curriculum-gripped-step-v8", "job_wall_seconds": 300,
             "process_wall_seconds": 330, "purpose": "E2 J0 reproduction prerequisite only",
             "expected_checkpoint_sha256": EXPECTED, "model_promoted": False, **hashes}
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(lease, stream)
    created = False
    result = {"status": "failed_before_start", "model_promoted": False, "walking_proven": False}
    started = time.monotonic()
    try:
        raw = subprocess.check_output(["powershell", "-NoProfile", "-Command",
              "$ErrorActionPreference='Stop'; Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"], text=True)
        inventory = json.loads(raw) if raw.strip() else []
        if isinstance(inventory, dict):
            inventory = [inventory]
        if any(item["ProcessId"] != os.getpid() for item in inventory):
            raise RuntimeError("Another Python process is active; host not exclusively available")
        output.mkdir(parents=True, exist_ok=False)
        created = True
        write(output / "protocol.json", lease)
        command = [sys.executable, "-m", "noesis_gpu.train", "--backend", "cpu",
                   "--seconds", "5", "--wall-seconds", "300", "--seed", "1001",
                   "--profile", "gripped-steps", "--exploration-mv-ms", "1.5",
                   "--noesis-source", str(source), "--data", str(root / "data"),
                   "--checkpoint", str(parent), "--output", str(output / "j0-v8-1001")]
        print("Starting E2 J0 v8 reproduction", flush=True)
        with (output / "j0.log").open("w", encoding="utf-8") as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=330, check=False)
        report = json.loads((output / "j0-v8-1001/report.json").read_text())
        if child.returncode or report["status"] != "completed_training_birth":
            raise RuntimeError(f"Incomplete J0: exit={child.returncode}, status={report['status']}")
        actual = digest(output / "j0-v8-1001/weights-last.npz")
        unchanged = (digest(parent) == PARENT and tree_digest(source) == hashes["core_source_sha256"]
                     and tree_digest(pilot) == hashes["pilot_source_sha256"])
        result.update(status="reproduction_passed" if actual == EXPECTED and unchanged else "reproduction_failed",
                      checkpoint_sha256=actual, expected_checkpoint_sha256=EXPECTED,
                      parent_and_sources_unchanged=unchanged, learning_updates=report["learning_updates"],
                      simulated_seconds=report["simulated_seconds"],
                      next_step="P2 integration and tests required before v9; no automatic continuation")
        print(json.dumps(result), flush=True)
        return 0 if result["status"] == "reproduction_passed" else 4
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if created:
            write(output / "result.json", {**result, "wall_seconds": time.monotonic() - started})
        if json.loads(lock.read_text()).get("token") == token:
            lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())

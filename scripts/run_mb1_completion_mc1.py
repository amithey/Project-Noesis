"""Complete the seven missing MB1 cells while preserving the interrupted bank."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import run_mb1_muscle_balance as mb1

ROOT, HERE = mb1.ROOT, mb1.HERE
read, write, sha = mb1.read, mb1.write, mb1.sha
ORIGINAL = ROOT / "outputs/muscle-balance-mb1-20261117-prepared-private.json"
LIMIT = 3000


def schedule(saved_names):
    expected = list(mb1.jobs())
    if saved_names != [job[0] for job in expected[:18]]:
        raise ValueError("Require exactly the registered 18-cell prefix")
    return [expected[0], *expected[18:]]


def checked_receipt(out, cell):
    receipt = read(out / (cell["name"] + "-receipt.json"))
    if (receipt != cell["receipt"] or receipt["exit_code"] != 0 or receipt["timed_out"] or
            not receipt["tree_empty"] or receipt["seconds"] > 300):
        raise ValueError("Worker receipt failed")


def prefix_values(original, old):
    out = Path(original["out"])
    result = read(out / "result.json")
    schedule([cell["name"] for cell in result["jobs"]])
    if result["status"] != "running":
        raise ValueError("Original interruption state changed")
    values = {}
    for job, cell in zip(mb1.jobs(), result["jobs"]):
        checked_receipt(out, cell)
        values[job[0]] = mb1.inspect_cell(original, out, job, old)
    return values


def prepare(out, old):
    prepared = out.with_name(out.name + "-prepared-private.json")
    if out.exists() or prepared.exists():
        raise ValueError("Completion output exists")
    original = read(ORIGINAL)
    old_out = Path(original["out"])
    old_result = read(old_out / "result.json")
    pins = {**original["pins"], **old_result["created_hashes"], str(ORIGINAL): sha(ORIGINAL)}
    mb1.ss1.base.check_pins(pins)
    prefix_values(original, old)
    for path in old_out.rglob("*"):
        if path.is_file():
            pins[str(path)] = sha(path)
    for name in ("run_mb1_completion_mc1.py", "test_mb1_completion_mc1.py"):
        pins[str(HERE / name)] = sha(HERE / name)
    plan = dict(original=original, out=str(out), pins=pins,
                jobs=schedule([cell["name"] for cell in old_result["jobs"]]),
                maximum_workers=8, worker_seconds=300, aggregate_seconds=LIMIT)
    mb1.ss1.base.check_pins(pins)
    write(prepared, plan)
    print("MC1 prepared: 18 preserved cells, 7 missing cells and 1 exact baseline")


def run(prepared, old, awake_type):
    plan = read(prepared)
    original, out = plan["original"], Path(plan["out"])
    runtime = Path(original["runtime"])
    if out.exists():
        raise ValueError("Completion allocation already exists")
    if plan["jobs"] != [list(job) for job in schedule([cell["name"] for cell in
                                                       read(Path(original["out"]) / "result.json")["jobs"]])]:
        raise ValueError("Completion schedule changed")
    pins = {**plan["pins"], str(prepared): sha(prepared)}
    mb1.ss1.base.check_pins(pins)
    values = prefix_values(original, old)
    mb1.ss1.base.resources()
    lock = runtime / "work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock"
    started, token = time.monotonic(), uuid.uuid4().hex
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w") as handle:
        json.dump(dict(owner="codex-MC1", pid=os.getpid(), token=token,
                       deadline_unix=time.time() + LIMIT), handle)
    awake, empty = awake_type(), True
    result = dict(status="running_MC1", jobs=[], created_hashes={})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               MPLCONFIGDIR=str(out.parent / "matplotlib-cache"), NUMBA_CACHE_DIR=str(out.parent / "numba-cache"))
    try:
        out.mkdir()
        with awake:
            for job in plan["jobs"]:
                if time.monotonic() - started + 300 > LIMIT:
                    raise TimeoutError("No full completion worker budget")
                mb1.ss1.base.resources()
                mb1.ss1.base.check_pins({**pins, **result["created_hashes"]})
                name, label, sign, seed = job
                checkpoint = original["parent"] if label is None else original["checkpoints"][f"{label}/{sign:+}"]
                command = [sys.executable, str(HERE / "ss1_worker.py"), "--runtime-root", str(runtime),
                           "--output", str(out / name), "--checkpoint", checkpoint, "--seed", str(seed),
                           "--phase", "baseline" if label is None else "trial",
                           "--bank", "parent" if label is None else "probe"]
                if label is not None:
                    command += ["--direction", "0", "--sign", str(sign), "--amplitude", str(mb1.AMPLITUDE)]
                print("START", name, flush=True)
                empty = False
                try:
                    with (out / (name + ".log")).open("w", encoding="utf-8") as log:
                        receipt = old.native.run_windows_job(command, timeout=300, stdout=log, cwd=ROOT, env=env)
                    empty = receipt["tree_empty"]
                except old.native.WatchdogError as error:
                    empty = error.cleanup_verified
                    raise
                write(out / (name + "-receipt.json"), receipt)
                cell = dict(name=name, receipt=receipt)
                result["jobs"].append(cell)
                write(out / "result.json", result)
                checked_receipt(out, cell)
                measured = mb1.inspect_cell(original, out, job, old)
                if label is None and measured != values[name]:
                    raise ValueError("Completion baseline metrics differ")
                values[name] = measured
                for path in [*(out / name).iterdir(), out / (name + ".log"),
                             out / (name + "-receipt.json"), out / (name + "-ss1-context.json")]:
                    if path.is_file():
                        result["created_hashes"][str(path)] = sha(path)
                print("COMPLETE", name, flush=True)
            result.update(mb1.summarize(original, values))
            mb1.ss1.base.check_pins({**pins, **result["created_hashes"]})
            if time.monotonic() - started > LIMIT or len(values) != 25:
                raise ValueError("Completion budget or bank count failed")
            result["status"] = "completed_MC1"
    except BaseException as error:
        result.update(status="failed_MC1", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        guards = True
        try:
            mb1.ss1.base.check_pins({**pins, **result["created_hashes"]})
        except (OSError, ValueError):
            guards = False
        if empty and lock.exists() and read(lock).get("token") == token:
            lock.unlink()
        result.update(seconds=time.monotonic() - started, source_inputs_created_unchanged=guards,
                      tree_empty=empty, lease_released=not lock.exists(), awake_released=awake.released)
        if out.exists():
            write(out / "result.json", result)


def audit(prepared, old):
    plan = read(prepared)
    out, original = Path(plan["out"]), plan["original"]
    result = read(out / "result.json")
    if result["status"] != "completed_MC1" or result["seconds"] > LIMIT or not all(
        result[key] for key in ("source_inputs_created_unchanged", "tree_empty", "lease_released", "awake_released")
    ):
        raise ValueError("Completion terminal guards failed")
    if [cell["name"] for cell in result["jobs"]] != [job[0] for job in plan["jobs"]]:
        raise ValueError("Incomplete or reordered completion")
    pins = {**plan["pins"], **result["created_hashes"], str(prepared): sha(prepared)}
    mb1.ss1.base.check_pins(pins)
    values = prefix_values(original, old)
    for job, cell in zip(plan["jobs"], result["jobs"]):
        checked_receipt(out, cell)
        actual = mb1.inspect_cell(original, out, job, old)
        if job[1] is None and actual != values[job[0]]:
            raise ValueError("Baseline metrics mismatch")
        values[job[0]] = actual
    summary = mb1.summarize(original, values)
    if any(result[key] != value for key, value in summary.items()) or len(values) != 25:
        raise ValueError("Combined selection differs from saved result")
    mb1.ss1.base.check_pins(pins)
    public = dict(status="completed_MB1_MC1_combined_audit", **summary, complete_distinct_cells=25,
                  original_complete_cells=18, new_complete_workers=8, allocations=2,
                  attempts_including_partial=27, original_interruption_preserved=True,
                  baseline_repeat_exact=True, raw_returns_and_selection_reproduced=True,
                  model_promoted=False, walking_proven=False,
                  limits="Four structured trial policies in six TRAIN contexts; independent retention required.")
    target = out.with_name(out.name + "-minimal-result.json")
    if target.exists():
        raise ValueError("Audit result exists")
    write(target, public)
    print(json.dumps(public))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    original = read(ORIGINAL)
    _, old, awake = mb1.ss1.base.legacy(Path(original["runtime"]))
    if args.mode == "prepare":
        prepare(args.path.resolve(), old)
    elif args.mode == "run":
        run(args.path.resolve(), old, awake)
    else:
        audit(args.path.resolve(), old)

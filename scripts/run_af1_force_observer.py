"""Bounded AF1 allocation using the existing native lease and watchdog."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from af1_force_observer import summarize
import run_mapped_target_search as base

ROOT, HERE = base.ROOT, base.HERE
read, write, sha = base.read, base.write, base.sha
LIMIT = 900


def prepare(out):
    prepared = out.with_name(out.name + "-prepared-private.json")
    if out.exists() or prepared.exists():
        raise ValueError("Existing allocation")
    prior_path = ROOT / "outputs/mapped-target-search-20261114-prepared-private.json"
    prior = read(prior_path)
    old_result = read(Path(prior["out"]) / "result.json")
    if old_result["status"] != "completed" or not old_result["lease_released"]:
        raise ValueError("Prior trial not terminal")
    pins = {**prior["pins"], str(prior_path): sha(prior_path)}
    for name in ("af1_force_observer.py", "run_af1_force_observer.py", "test_af1_force_observer.py"):
        pins[str(HERE / name)] = sha(HERE / name)
    base.ss1.base.check_pins(pins)
    write(prepared, dict(runtime=prior["runtime"], parent=prior["parent"], pins=pins,
                         prior=prior["prior"], out=str(out), seeds=[1010, 1011],
                         limit=LIMIT, worker_limit=300, sample_count=50000,
                         sampling_interval=.0001, force_tolerance_relative=1e-8))


def inspect(plan, seed, old):
    folder = Path(plan["out"]) / f"parent-{seed}"
    report = read(folder / "report.json")
    context = dict(phase="trial", bank="probe", seed=seed, direction=0, sign=1, amplitude=.005)
    base.ss1.validate_report(report, context, sha(Path(plan["parent"])),
                             plan["pins"][str(HERE / "ss1_worker.py")])
    if report["ss1_context"] != read(folder.with_name(folder.name + "-ss1-context.json")):
        raise ValueError("Delegated native worker context mismatch")
    reference = base.ss1.reference(plan["prior"], seed)
    old.native.compare(reference, folder)
    for name in ("observations.npz", "adhesion-observations.npz", "anatomy-observations.npz"):
        old.native.exact(reference / name, folder / name)
    if report["spikes_per_world"] != read(reference / "report.json")["spikes_per_world"]:
        raise ValueError("Exact spike-count replay failed")
    data = base.ss1.arrays(folder / "joint-actuator-forces.npz")
    metadata = read(folder / "joint-actuator-metadata.json")
    if data["force"].shape != (50000, 42) or data["control"].shape != (50000, 42):
        raise ValueError("Incomplete force/control horizon")
    if not np.isfinite(data["control"]).all() or len(set(metadata["actuator_ids"])) != 42:
        raise ValueError("Invalid actuator layout or controls")
    np.testing.assert_allclose(data["time"], np.arange(1, 50001) * .0001, rtol=0, atol=1e-10)
    if metadata["timestep"] != .0001 or metadata["kp"] != [45.] * 42:
        raise ValueError("Actuator parameters changed")
    return summarize(data["force"], metadata["limits"], metadata["enabled"])


def run(prepared, old, awake_type):
    plan = read(prepared)
    out, runtime = Path(plan["out"]), Path(plan["runtime"])
    if out.exists():
        raise ValueError("No allocation retry")
    pins = {**plan["pins"], str(prepared): sha(prepared)}
    base.ss1.base.check_pins(pins)
    base.ss1.base.resources()
    lock = runtime / "work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock"
    token = uuid.uuid4().hex
    started = time.monotonic()
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w") as handle:
        json.dump(dict(owner="codex-AF1", pid=os.getpid(), token=token,
                       deadline_unix=time.time() + LIMIT), handle)
    awake, tree_empty = awake_type(), True
    result = dict(status="running", jobs=[], summaries={}, created_hashes={})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               MPLCONFIGDIR=str(out.parent / "matplotlib-cache"), NUMBA_CACHE_DIR=str(out.parent / "numba-cache"))
    try:
        out.mkdir()
        with awake:
            for seed in plan["seeds"]:
                if time.monotonic() - started + 300 > LIMIT:
                    raise TimeoutError("No full worker budget")
                base.ss1.base.resources()
                base.ss1.base.check_pins({**pins, **result["created_hashes"]})
                folder = out / f"parent-{seed}"
                print("START", seed, flush=True)
                tree_empty = False
                try:
                    with (out / f"{seed}.log").open("w", encoding="utf-8") as log:
                        receipt = old.native.run_windows_job(
                            [sys.executable, str(HERE / "af1_force_observer.py"), "--runtime", str(runtime),
                             "--checkpoint", plan["parent"], "--output", str(folder), "--seed", str(seed)],
                            timeout=300, stdout=log, cwd=ROOT, env=env)
                    tree_empty = receipt["tree_empty"]
                except old.native.WatchdogError as error:
                    tree_empty = error.cleanup_verified
                    raise
                write(out / f"{seed}-receipt.json", receipt)
                result["jobs"].append(dict(seed=seed, receipt=receipt))
                if receipt["exit_code"] or receipt["timed_out"] or not tree_empty:
                    raise RuntimeError("AF1 worker failed")
                result["summaries"][str(seed)] = inspect(plan, seed, old)
                for path in [*folder.iterdir(), out / f"{seed}.log", out / f"{seed}-receipt.json",
                             folder.with_name(folder.name + "-ss1-context.json")]:
                    if path.is_file():
                        result["created_hashes"][str(path)] = sha(path)
                write(out / "result.json", result)
                print("COMPLETE", seed, flush=True)
            base.ss1.base.check_pins({**pins, **result["created_hashes"]})
            if time.monotonic() - started > LIMIT:
                raise TimeoutError("Aggregate budget exceeded")
            result["status"] = "completed_AF1"
    except BaseException as error:
        result.update(status="failed_AF1", error=str(error), error_type=type(error).__name__)
        raise
    finally:
        guards = True
        try:
            base.ss1.base.check_pins({**pins, **result["created_hashes"]})
        except (OSError, ValueError):
            guards = False
        if tree_empty and lock.exists() and read(lock).get("token") == token:
            lock.unlink()
        result.update(seconds=time.monotonic() - started, input_source_created_unchanged=guards,
                      tree_empty=tree_empty, lease_released=not lock.exists(), awake_released=awake.released)
        if out.exists():
            write(out / "result.json", result)


def audit(prepared, old):
    plan = read(prepared)
    out = Path(plan["out"])
    result = read(out / "result.json")
    if result["status"] != "completed_AF1" or result["seconds"] > LIMIT or not all(
        result[k] for k in ("input_source_created_unchanged", "tree_empty", "lease_released", "awake_released")
    ):
        raise ValueError("Failed terminal allocation guards")
    if [job["seed"] for job in result["jobs"]] != [1010, 1011]:
        raise ValueError("Incorrect allocation sequence")
    pins = {**plan["pins"], **result["created_hashes"], str(prepared): sha(prepared)}
    base.ss1.base.check_pins(pins)
    summaries = {}
    for job in result["jobs"]:
        seed = job["seed"]
        receipt = read(out / f"{seed}-receipt.json")
        if receipt != job["receipt"] or receipt["exit_code"] or receipt["timed_out"] or not receipt["tree_empty"]:
            raise ValueError("Invalid worker receipt")
        summaries[str(seed)] = inspect(plan, seed, old)
    if summaries != result["summaries"]:
        raise ValueError("Force summary replay mismatch")
    base.ss1.base.check_pins(pins)
    public = dict(status="completed_AF1_audit", replay_exact=True, complete_workers=2,
                  summaries=summaries, source_input_created_guards=True,
                  walking_proven=False, model_promoted=False,
                  limits="Two frozen TRAIN replays, observed actuation-space forces. No causal intervention.")
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
    prior = read(ROOT / "outputs/mapped-target-search-20261114-prepared-private.json")
    _, old, awake = base.ss1.base.legacy(Path(prior["runtime"]))
    if args.mode == "prepare":
        prepare(args.path.resolve())
    elif args.mode == "run":
        run(args.path.resolve(), old, awake)
    else:
        audit(args.path.resolve(), old)

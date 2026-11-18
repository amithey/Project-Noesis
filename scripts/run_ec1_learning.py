"""Bounded EC1 learning allocation and raw-evidence validation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from ec1_contract import TRAIN, BANKS, lineage, jobs, decision
import run_mb1_completion_mc1 as previous

ROOT, HERE = previous.ROOT, previous.HERE
base = previous.mb1
read, write, sha = previous.read, previous.write, previous.sha
LIMIT = 9000
SOURCE = ("ec1_contract.py", "ec1_worker.py", "run_ec1_learning.py",
          "audit_ec1_learning.py", "test_ec1_contract.py")


def checkpoint(plan, job):
    if job["phase"] == "train":
        parent_bank = lineage(job["bank"])[0]
    else:
        parent_bank = job["bank"]
    return Path(plan["parent"]) if parent_bank == "parent" else Path(plan["out"]) / f"train-{parent_bank}/weights-last.npz"


def cell(folder, old):
    _, summary = old.native.summarize(folder)
    history = old.previous.previous.history
    physical = history.physical_summary(folder)
    anatomy = history.prior.previous.contact_data(folder)
    observations = base.ss1.arrays(folder / "observations.npz")
    anatomical = old.native.anatomy(folder, "frozen")
    reward = old.episode_rewards(observations, anatomical, read(folder / "anatomy-model.json"))
    return dict(summary=summary, physical=physical, anatomical=anatomy, **{"return": float(reward[10:].mean())})


def prepare(out, old):
    prepared = out.with_name(out.name + "-prepared-private.json")
    if out.exists() or prepared.exists():
        raise ValueError("EC1 allocation already exists")
    prior_path = ROOT / "outputs/muscle-balance-mb1-20261117-prepared-private.json"
    prior = read(prior_path)
    completion_path = ROOT / "outputs/mb1-completion-mc1-20261118-prepared-private.json"
    completion = read(completion_path)
    result = read(Path(completion["out"]) / "result.json")
    if result["status"] != "completed_MC1" or not result["lease_released"]:
        raise ValueError("Previous allocation not terminal")
    pins = {**completion["pins"], **result["created_hashes"], str(prior_path): sha(prior_path),
            str(completion_path): sha(completion_path)}
    for name in SOURCE:
        pins[str(HERE / name)] = sha(HERE / name)
    runtime = Path(prior["runtime"])
    # Earlier experiments pin these native helpers in their own prospective record.
    ed2 = read(runtime / "outputs/eligibility-replication-ed2-20261107/protocol.json")
    for group in ("source_hashes", "prior_hashes"):
        for name, digest in ed2[group].items():
            path = runtime / name
            if path.suffix == ".py":
                if str(path) in pins and pins[str(path)] != digest:
                    raise ValueError("Historical source pins conflict")
                pins[str(path)] = digest
    base.ss1.base.check_pins(pins)
    parents = {}
    for seed in TRAIN:
        folder = base.ss1.reference(prior["prior"], seed)
        parents[str(seed)] = cell(folder, old)
        for path in folder.iterdir():
            if path.is_file():
                actual = sha(path)
                if str(path) in pins and pins[str(path)] != actual:
                    raise ValueError("Parent evidence changed")
                pins[str(path)] = actual
    base.ss1.base.check_pins(pins)
    write(prepared, dict(experiment="EC1", out=str(out), runtime=str(runtime), parent=prior["parent"],
                         parents=parents, prior=prior["prior"], pins=pins, jobs=jobs(), limit=LIMIT))
    print("EC1 prepared: common parent, three native training births, nineteen frozen probes")


def validate_worker(plan, job, old):
    folder = Path(plan["out"]) / job["name"]
    report = read(folder / "report.json")
    context = report["ec1_context"]
    expected = dict(experiment="EC1", phase=job["phase"], bank=job["bank"], seed=job["seed"],
                    checkpoint_sha256=sha(checkpoint(plan, job)),
                    worker_sha256=plan["pins"][str(HERE / "ec1_worker.py")],
                    objective="physical-standing-ps1-v1", delay_chunks=0, worker_completed=True)
    if any(context.get(key) != value for key, value in expected.items()) or context != read(
        folder.with_name(folder.name + "-ec1-context.json")
    ) or not report["ec1_worker_completed"]:
        raise ValueError("Wrong worker provenance")
    if len(context["import_records"]) != 1 or not all(context["import_records"][0].get(key) is True for key in
        ("actual_import_exact", "topology_unchanged", "nonselected_weights_unchanged")):
        raise ValueError("Native checkpoint import failed")
    if job["phase"] == "frozen":
        expected = dict(status="completed_transport_pilot", controller="brain", learning=False,
                        motor_exploration=0, worlds=1, weights_unchanged=True,
                        completed_control_chunks=1000, simulated_seconds_per_world=5,
                        control_ms=5., physics_dt=.0001)
        if any(report.get(key) != value for key, value in expected.items()):
            raise ValueError("Incomplete or altered frozen worker")
        old.native.compare(base.ss1.reference(plan["prior"], job["seed"]), folder)
        if job["bank"] == "parent":
            reference = base.ss1.reference(plan["prior"], job["seed"])
            for name in ("observations.npz", "anatomy-observations.npz", "adhesion-observations.npz"):
                old.native.exact(reference / name, folder / name)
            if report["spikes_per_world"] != read(reference / "report.json")["spikes_per_world"]:
                raise ValueError("Frozen baseline spikes differ")
        return cell(folder, old)
    from s2_contract import validate_training_report
    from run_eligibility_ed1 import eligibility_checks
    from ps1_objective import physical_score
    validate_training_report({**report, "reward_objective": "curriculum-standing-v7"})
    if (report["reward_objective"] != "physical-standing-ps1-v1" or report["eligibility_audit_rows"] != 4990
            or report["seed"] != job["seed"] or not context["import_records"][0]["episode_traces_reset"]):
        raise ValueError("Training method changed")
    saved = folder / "weights-last.npz"
    if report["saved_checkpoint_sha256"] != sha(saved) or report["import_checkpoint_sha256"] != sha(checkpoint(plan, job)):
        raise ValueError("Checkpoint lineage digest mismatch")
    comparison = old.native.compare_checkpoints(Path(plan["parent"]), saved)
    if not comparison["compatible"] or not comparison["rule"]["match"] or not comparison["parameters"]["match"]:
        raise ValueError("Original-reference checkpoint invariants failed")
    checked = eligibility_checks(folder, 0)
    anatomy = old.native.anatomy(folder, "train")
    observations = base.ss1.arrays(folder / "training-observations.npz")
    accounting = base.ss1.arrays(folder / "physical-reward-accounting.npz")
    if any(len(value) != 5000 or not np.isfinite(value).all() for value in accounting.values()):
        raise ValueError("Incomplete reward accounting")
    np.testing.assert_array_equal(accounting["control_chunk_index"], np.arange(5000))
    np.testing.assert_array_equal(accounting["time"], anatomy["time"][1:])
    np.testing.assert_array_equal(accounting["applied_reward"], observations["reward"])
    model = read(folder / "anatomy-model.json")
    components = physical_score(anatomy["region_vertical_reaction"][1:], anatomy["com_velocity"][1:],
                                observations["omega"], observations["up_body"][:, 2], observations["foot_load"],
                                model["total_mass"] * abs(model["gravity"][2]))
    for key, value in components.items():
        np.testing.assert_allclose(accounting["PS1_" + key], value, rtol=1e-14, atol=1e-15)
    return dict(checkpoint_sha256=sha(saved), actual_import_and_episode_reset=True, eligibility=checked)


def checked_receipt(folder, item, timeout):
    receipt = read(folder / (item["job"]["name"] + "-receipt.json"))
    if (receipt != item["receipt"] or receipt["exit_code"] != 0 or receipt["timed_out"] or
            not receipt["tree_empty"] or receipt["seconds"] > timeout):
        raise ValueError("Worker receipt failed")


def run(prepared, old, awake_type):
    plan = read(prepared)
    out, runtime = Path(plan["out"]), Path(plan["runtime"])
    if out.exists() or plan["jobs"] != jobs():
        raise ValueError("Existing allocation or modified schedule")
    pins = {**plan["pins"], str(prepared): sha(prepared)}
    base.ss1.base.check_pins(pins)
    base.ss1.base.resources()
    lock = runtime / "work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock"
    started, token = time.monotonic(), uuid.uuid4().hex
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w") as handle:
        json.dump(dict(owner="codex-EC1", pid=os.getpid(), token=token,
                       deadline_unix=time.time() + LIMIT), handle)
    awake, empty = awake_type(), True
    result = dict(status="running_EC1", jobs=[], created_hashes={})
    banks = {"parent": {int(seed): value for seed, value in plan["parents"].items()},
             **{bank: {} for bank in BANKS}}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               MPLCONFIGDIR=str(out.parent / "matplotlib-cache"), NUMBA_CACHE_DIR=str(out.parent / "numba-cache"))
    try:
        out.mkdir()
        with awake:
            for job in jobs():
                timeout = 600 if job["phase"] == "train" else 300
                if time.monotonic() - started + timeout > LIMIT:
                    raise TimeoutError("Insufficient complete worker budget")
                base.ss1.base.resources()
                base.ss1.base.check_pins({**pins, **result["created_hashes"]})
                folder = out / job["name"]
                print("START", job["name"], flush=True)
                empty = False
                try:
                    with (out / (job["name"] + ".log")).open("w", encoding="utf-8") as log:
                        receipt = old.native.run_windows_job(
                            [sys.executable, str(HERE / "ec1_worker.py"), "--root", str(runtime),
                             "--output", str(folder), "--checkpoint", str(checkpoint(plan, job)),
                             "--seed", str(job["seed"]), "--phase", job["phase"], "--bank", job["bank"]],
                            timeout=timeout, stdout=log, cwd=ROOT, env=env)
                    empty = receipt["tree_empty"]
                except old.native.WatchdogError as error:
                    empty = error.cleanup_verified
                    raise
                item = dict(job=job, receipt=receipt)
                write(out / (job["name"] + "-receipt.json"), receipt)
                result["jobs"].append(item)
                write(out / "result.json", result)
                checked_receipt(out, item, timeout)
                measured = validate_worker(plan, job, old)
                if job["phase"] == "frozen":
                    banks[job["bank"]][job["seed"]] = measured
                for path in [*folder.iterdir(), out / (job["name"] + ".log"),
                             out / (job["name"] + "-receipt.json"), out / (job["name"] + "-ec1-context.json")]:
                    if path.is_file():
                        result["created_hashes"][str(path)] = sha(path)
                print("COMPLETE", job["name"], flush=True)
            science = decision(banks)
            write(out / "science-private.json", dict(banks=banks, decision=science))
            result["created_hashes"][str(out / "science-private.json")] = sha(out / "science-private.json")
            result.update(science)
            base.ss1.base.check_pins({**pins, **result["created_hashes"]})
            if time.monotonic() - started > LIMIT:
                raise TimeoutError("Aggregate budget exceeded")
            result["status"] = "completed_EC1"
    except BaseException as error:
        result.update(status="failed_EC1", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        guards = True
        try:
            base.ss1.base.check_pins({**pins, **result["created_hashes"]})
        except (OSError, ValueError):
            guards = False
        if empty and lock.exists() and read(lock).get("token") == token:
            lock.unlink()
        result.update(seconds=time.monotonic() - started, source_inputs_created_unchanged=guards,
                      tree_empty=empty, lease_released=not lock.exists(), awake_released=awake.released)
        if out.exists():
            write(out / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    prior = read(ROOT / "outputs/muscle-balance-mb1-20261117-prepared-private.json")
    _, old, awake = base.ss1.base.legacy(Path(prior["runtime"]))
    if args.mode == "prepare":
        prepare(args.path.resolve(), old)
    else:
        run(args.path.resolve(), old, awake)

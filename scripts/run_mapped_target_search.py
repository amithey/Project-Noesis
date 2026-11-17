"""Prepare, run and replay the registered matched target-set experiment."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from mapped_target_search import AMPLITUDE, SETS, SIGNS, SEED, directions, selection
import run_scale_ss1 as ss1

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
BASELINE = ROOT / "outputs/probe-scale-ss1-20261113"
INDEX = ROOT / "outputs/om1-input-index-private-20261113.json"
LIMIT, TIMEOUT = 7800, 300
sha, read, write = ss1.sha, ss1.read, ss1.write


def jobs():
    yield "baseline-1010", None, None, 1010
    for seed in ss1.TRAIN:
        for label in SETS:
            for sign in SIGNS:
                yield f"{label}-{seed}-{sign:+}", label, sign, seed


def prepare(out, old):
    inputs = out.with_name(out.name + "-inputs")
    prepared = out.with_name(out.name + "-prepared-private.json")
    if any(x.exists() for x in (out, inputs, prepared)):
        raise ValueError("Require fresh output and inputs")
    index = json.loads(INDEX.read_text(encoding="utf-8-sig"))
    pins = {v["path"]: v["sha256"] for v in index.values()}
    prior = read(BASELINE / "protocol.json")
    result = read(BASELINE / "result.json")
    if result["status"] != "completed_SS1" or not result["baseline_exact"]:
        raise ValueError("Prior scale allocation incomplete")
    pins.update(prior["prior_hashes"])
    sources = (*ss1.SOURCE, "run_training_context_mp2.py", "mp2_select.py",
               "mapped_target_search.py", "run_mapped_target_search.py", "test_mapped_target_search.py")
    pins.update({str(HERE / name): sha(HERE / name) for name in sources})
    for name in ("protocol.json", "result.json", "training-returns.json"):
        pins[str(BASELINE / name)] = sha(BASELINE / name)
    ss1.base.check_pins(pins)
    parent_path = Path(index["parent"]["path"])
    parent = ss1.arrays(parent_path)
    runtime = Path(prior["runtime_root"])
    sys.path[:0] = [str(Path(index[r]["path"]).parents[1]) for r in ("motor", "bridge")]
    from noesis.embodiment import load_training_brain
    from noesis.motor import build_muscle_matrices
    from noesis_gpu.bridge import LEGACY_JOINTS

    cache = Path(index["cache"]["path"]).parent
    wfile = cache / "W_min5_scaled_a3e600af.npz"
    pins[str(wfile)] = "7bc39c8c85882386c6dcac9fbdc9f3825a3fbb3ecfd03b5961e59711daafb5b7"
    ss1.base.check_pins(pins)
    connectome, _ = load_training_brain(runtime / "work/NoesisData", cache)
    positive, negative = build_muscle_matrices(connectome.neurons, list(LEGACY_JOINTS))
    selected = np.isin(connectome.W.indices, parent["motor_indices"])
    np.testing.assert_array_equal(connectome.W.data[selected].astype(float), parent["initial_weights"])
    targets = connectome.W.indices[selected]
    mapped = np.asarray((positive + negative).getnnz(axis=0) > 0)
    offsets, eligible = directions(parent["weights"], parent["initial_weights"], mapped[targets])
    sealed = {x["context"]["seed"]: x for x in read(BASELINE / "training-returns.json")
              if x["context"]["bank"] == "parent"}
    parent_values = {}
    for seed in ss1.TRAIN:
        folder = ss1.reference(prior, seed)
        record = ss1.base.measured_record(folder, sealed[seed]["context"], old)
        if record != sealed[seed] or record["checkpoint_sha256"] != index["parent"]["sha256"]:
            raise ValueError("Sealed parent replay mismatch")
        pins.update({str(folder / n): h for n, h in record["source_hashes"].items()})
        parent_values[str(seed)] = metrics(folder, record)
    inputs.mkdir()
    representation = {}
    checkpoints = {}
    for label, delta in offsets.items():
        influenced = np.unique(targets[delta != 0])
        representation[label] = {
            "eligible_edges": eligible[label], "perturbed_edges": int(np.count_nonzero(delta)),
            "target_count": int(influenced.size),
            "positive_pools": int(np.count_nonzero(positive[:, influenced].getnnz(axis=1))),
            "negative_pools": int(np.count_nonzero(negative[:, influenced].getnnz(axis=1))),
        }
        for sign in SIGNS:
            path = inputs / f"{label}-{sign:+}.npz"
            probe = old.apply_delta(parent, sign * delta)
            if np.count_nonzero(probe["weights"] != parent["weights"]) != np.count_nonzero(delta):
                raise ValueError("Unexpected rounded or changed perturbation")
            old.save_new_checkpoint(path, probe)
            pins[str(path)] = sha(path)
            checkpoints[f"{label}/{sign:+}"] = str(path)
    ss1.base.check_pins(pins)
    plan = dict(runtime=str(runtime), out=str(out), parent=str(parent_path), parent_values=parent_values,
                prior=prior, pins=pins, checkpoints=checkpoints, representation=representation)
    write(prepared, plan)
    print(json.dumps(representation))


def metrics(folder, record):
    omega = ss1.arrays(folder / "observations.npz")["omega"]
    if omega.shape != (1000, 1, 3) or not np.isfinite(omega).all():
        raise ValueError("Missing or invalid registered angular telemetry")
    return [record["return"], float(np.linalg.norm(omega[10:, 0], axis=1).mean())]


def inspect_cell(plan, out, job, old):
    name, label, sign, seed = job
    folder = out / name
    path = Path(plan["parent"] if label is None else plan["checkpoints"][f"{label}/{sign:+}"])
    report = read(folder / "report.json")
    context = dict(phase="baseline" if label is None else "trial", bank="parent" if label is None else "probe",
                   seed=seed, direction=None if label is None else 0, sign=sign,
                   amplitude=None if label is None else AMPLITUDE)
    ss1.validate_report(report, context, sha(path), plan["pins"][str(HERE / "ss1_worker.py")])
    if report["ss1_context"] != read(out / (name + "-ss1-context.json")):
        raise ValueError("Worker context mismatch")
    reference = ss1.reference(plan["prior"], seed)
    old.native.compare(reference, folder)
    if label is None:
        for n in ("observations.npz", "adhesion-observations.npz", "anatomy-observations.npz"):
            old.native.exact(reference / n, folder / n)
        if report["spikes_per_world"] != read(reference / "report.json")["spikes_per_world"]:
            raise ValueError("Baseline spikes differ")
    return metrics(folder, ss1.base.measured_record(folder, context, old))


def summarize(plan, values):
    return selection([plan["parent_values"][str(seed)] for seed in ss1.TRAIN],
                     {(label, sign): [values[f"{label}-{seed}-{sign:+}"] for seed in ss1.TRAIN]
                      for label in SETS for sign in SIGNS})


def run(prepared, old, awake_type):
    plan = read(prepared)
    out, runtime = Path(plan["out"]), Path(plan["runtime"])
    if out.exists():
        raise ValueError("Allocation already exists")
    ss1.base.resources()
    pins = {**plan["pins"], str(prepared): sha(prepared)}
    ss1.base.check_pins(pins)
    lock = runtime / "work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock"
    token = uuid.uuid4().hex
    started = time.monotonic()
    awake = awake_type()
    tree_empty = True
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w") as handle:
        json.dump(dict(owner="codex-mapped-target-search", pid=os.getpid(), token=token,
                       deadline_unix=time.time() + LIMIT), handle)
    result = {"status": "running", "jobs": [], "created_hashes": {}}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1",
               MPLCONFIGDIR=str(out.parent / "matplotlib-cache"),
               NUMBA_CACHE_DIR=str(out.parent / "numba-cache"))
    values = {}
    try:
        out.mkdir()
        with awake:
            for job in jobs():
                if time.monotonic() - started + TIMEOUT > LIMIT:
                    raise TimeoutError("Insufficient complete worker budget")
                ss1.base.resources()
                ss1.base.check_pins({**pins, **result["created_hashes"]})
                name, label, sign, seed = job
                cp = plan["parent"] if label is None else plan["checkpoints"][f"{label}/{sign:+}"]
                command = [sys.executable, str(HERE / "ss1_worker.py"), "--runtime-root", str(runtime),
                           "--output", str(out / name), "--checkpoint", cp, "--seed", str(seed),
                           "--phase", "baseline" if label is None else "trial",
                           "--bank", "parent" if label is None else "probe"]
                if label is not None:
                    command += ["--direction", "0", "--sign", str(sign), "--amplitude", str(AMPLITUDE)]
                print("START", name, flush=True)
                tree_empty = False
                try:
                    with (out / (name + ".log")).open("w", encoding="utf-8") as log:
                        receipt = old.native.run_windows_job(command, timeout=TIMEOUT, stdout=log, cwd=ROOT, env=env)
                    tree_empty = receipt["tree_empty"]
                except old.native.WatchdogError as error:
                    tree_empty = error.cleanup_verified
                    raise
                write(out / (name + "-receipt.json"), receipt)
                result["jobs"].append({"name": name, "receipt": receipt})
                write(out / "result.json", result)
                if receipt["exit_code"] != 0 or receipt["timed_out"] or not tree_empty:
                    raise RuntimeError("Worker failed")
                values[name] = inspect_cell(plan, out, job, old)
                for path in [*(out / name).iterdir(), *(out.glob(name + ".*")),
                             out / (name + "-receipt.json"), out / (name + "-ss1-context.json")]:
                    if path.is_file():
                        result["created_hashes"][str(path)] = sha(path)
                print("COMPLETE", name, flush=True)
            result.update(summarize(plan, values), status="completed")
            ss1.base.check_pins({**pins, **result["created_hashes"]})
            if time.monotonic() - started > LIMIT:
                raise TimeoutError("Allocation exceeded budget")
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if tree_empty and lock.exists() and read(lock).get("token") == token:
                lock.unlink()
        finally:
            result.update(seconds=time.monotonic() - started, tree_empty=tree_empty,
                          lease_released=not lock.exists(), awake_released=awake.released)
            if out.exists():
                write(out / "result.json", result)


def audit(prepared, old):
    plan = read(prepared)
    out = Path(plan["out"])
    result = read(out / "result.json")
    if result["status"] != "completed" or result["seconds"] > LIMIT or not all(
        result[k] for k in ("tree_empty", "lease_released", "awake_released")
    ):
        raise ValueError("Allocation did not finish with all terminal guards")
    expected = list(jobs())
    if [cell["name"] for cell in result["jobs"]] != [job[0] for job in expected]:
        raise ValueError("Incomplete or misordered registered cells")
    pins = {**plan["pins"], **result["created_hashes"]}
    ss1.base.check_pins(pins)
    values = {}
    for job, cell in zip(expected, result["jobs"]):
        receipt = read(out / (job[0] + "-receipt.json"))
        if receipt != cell["receipt"] or receipt["exit_code"] or receipt["timed_out"] or not receipt["tree_empty"]:
            raise ValueError("Worker receipt failed")
        values[job[0]] = inspect_cell(plan, out, job, old)
    summary = summarize(plan, values)
    if any(result[key] != value for key, value in summary.items()):
        raise ValueError("Saved summary differs from replay")
    ss1.base.check_pins(pins)
    print(json.dumps(dict(summary, completed_workers=len(expected), independent_replay=True,
                          model_promoted=False, walking_proven=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "audit"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path.resolve()
    prior = read(BASELINE / "protocol.json")
    _, old, awake = ss1.base.legacy(Path(prior["runtime_root"]))
    if args.mode == "prepare":
        prepare(path, old)
    elif args.mode == "run":
        run(path, old, awake)
    else:
        audit(path, old)

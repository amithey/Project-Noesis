"""F10: frozen physical interface parity and extended-horizon stability."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
import psutil

from compare_exploration import digest, tree_digest, write
from noesis_gpu.evaluate import MATCHED_SETTINGS
from noesis_gpu.learning import standing_mask
from run_duration_study import resource_decision
from run_resource_envelope import SOURCES, monitored_run
from runtime_diagnostic import CORE, PARENT, inventory
from training_power import keep_system_awake
from training_resources import sample_quiescence

SEEDS = (10006, 10010)
BUDGET = 750
BASE = "c084e639f6eb1be19e33848c91e220d5e0283c8c"
MANIFEST = "bb3b95ef79fef2b534bc15c6502f1f2808988a4d830113000c39e95115e33738"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_trace(run, ref, seed):
    report, historical = read(run / "report.json"), read(ref / "report.json")
    expected = dict(status="completed_transport_pilot", weights_unchanged=True,
                    learning=False, motor_exploration=0, completed_control_chunks=2000,
                    simulated_seconds_per_world=10, requested_simulated_seconds_per_world=10,
                    checkpoint_sha256=PARENT, seeds=[seed], backend="cpu",
                    worlds=1, controller="brain", control_ms=5)
    old_expected = dict(expected, completed_control_chunks=1000, simulated_seconds_per_world=5,
                        requested_simulated_seconds_per_world=5)
    if any(historical.get(k) != v for k, v in old_expected.items()):
        raise ValueError("Invalid 5s reference report")
    for name, value in expected.items():
        if report.get(name) != value:
            raise ValueError(f"Invalid 10s report field: {name}")
    if any(report[k] != historical[k] for k in MATCHED_SETTINGS):
        raise ValueError("Historical context mismatch")

    with np.load(run / "observations.npz", allow_pickle=False) as new, np.load(
        ref / "observations.npz", allow_pickle=False
    ) as old:
        if not old.files or set(old.files) != set(new.files):
            raise ValueError("Trace field mismatch")
        for name in old.files:
            x, y = new[name], old[name]
            if (x.shape != (2000, *y.shape[1:]) or y.shape[0] != 1000
                    or x.dtype != y.dtype or not np.isfinite(x).all()
                    or not np.array_equal(x[:1000], y)):
                raise ValueError(f"Nonfinite, incomplete or divergent prefix: {name}")
        obs = {k: new[k][:, 0] for k in ("position", "up_body", "foot_load", "feet_position")}
        fields = sorted(new.files)

    standing = standing_mask(obs)
    support = obs["foot_load"][1:] > 1e-5
    speeds = np.linalg.norm(np.diff(obs["feet_position"][..., :2], axis=0), axis=-1) / .005
    counts = support.sum(axis=1)
    slip = np.divide((speeds * support).sum(axis=1), counts,
                     out=np.zeros(len(counts)), where=counts > 0)
    windows = []
    for start, stop in ((10, 1000), (1000, 2000)):
        windows.append(dict(start_chunk=start, stop_chunk=stop,
                            standing_fraction=float(standing[start:stop].mean()),
                            mean_stance_slip_mm_s=float(slip[start - 1:stop - 1].mean())))
    return dict(seed=seed, exact_prefix_fields=fields, windows=windows,
                observed_10s_sha256=digest(run / "observations.npz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shared-root", "core-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, core = args.shared_root.resolve(), args.core_source.resolve()
    pilot = Path(__file__).resolve().parents[1]
    outputs = root / "gpu-pilot/outputs"
    out = outputs / "interface-f10-20261105"
    bank = outputs / "e4-exploration-20261029/train-control-1000/weights-last.npz"
    manifest = root / "outputs/s1-evidence-archive-20261101/standing-profile-s1-manifest.json"
    if digest(manifest) != MANIFEST:
        raise ValueError("Wrong S1 evidence manifest")
    refs = {s: outputs / f"standing-profile-s1-20261101/frozen-parent-{s}" for s in SEEDS}
    hashes = {entry["path"]: entry["sha256"] for entry in read(manifest)["files"]}
    pinned = {bank: PARENT}
    for ref in refs.values():
        for name in ("observations.npz", "report.json"):
            p = ref / name
            pinned[p] = hashes[p.relative_to(root).as_posix()]
    controller = pilot / "gpu-pilot/src/noesis_gpu/controller.py"
    expected = subprocess.check_output(
        ["git", "-C", str(pilot), "show", BASE + ":gpu-pilot/src/noesis_gpu/controller.py"]
    )
    if controller.read_bytes().replace(b"\r\n", b"\n") != expected:
        raise ValueError("Unexpected interface source")
    source = tree_digest(pilot / "gpu-pilot/src")
    if subprocess.check_output(["git", "-C", str(pilot), "diff", "--name-only", BASE,
                                "--", "gpu-pilot/src"], text=True).strip():
        raise ValueError("Pilot runtime differs from interface commit")

    def verify():
        if (tree_digest(core) != SOURCES["core"] or tree_digest(pilot / "gpu-pilot/src") != source
                or any(digest(p) != sha for p, sha in pinned.items())):
            raise ValueError("Source or reference hash mismatch")

    verify()
    lock = outputs / "codex-training-host.lock"
    if list(outputs.glob("*host.lock")):
        raise RuntimeError("Host lease already exists")
    token = uuid.uuid4().hex
    lease = dict(experiment="F10", owner="codex", pid=os.getpid(), token=token,
                 deadline_unix=time.time() + BUDGET)
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w") as f:
        json.dump(lease, f)
    started = time.monotonic()
    created = False
    result = dict(status="running", jobs=[], walking_proven=False, model_promoted=False,
                  core_commit=CORE, core_source_sha256=SOURCES["core"], interface_commit=BASE,
                  pilot_source_sha256=source, parent_sha256=PARENT, budget_seconds=BUDGET,
                  s1_manifest_sha256=MANIFEST, seeds=list(SEEDS))
    try:
        inventory()
        out.mkdir(exist_ok=False)
        created = True
        write(out / "result.json", result)
        for seed in SEEDS:
            verify()
            inventory()
            samples = sample_quiescence()
            resources = resource_decision(samples, True)
            write(out / f"preflight-{seed}.json", dict(samples=samples, decision=resources))
            if not resources["passed"] or psutil.virtual_memory().available / 2**30 < 2:
                raise RuntimeError("Fresh resource gate refused")
            if BUDGET - (time.monotonic() - started) < 335:
                raise TimeoutError("Insufficient next-job budget")
            run = out / f"frozen-{seed}"
            command = [sys.executable, "-m", "noesis_gpu.run", "--backend", "cpu", "--worlds", "1",
                       "--noesis-source", str(core), "--data", str(root / "data"), "--seconds", "10",
                       "--wall-seconds", "300", "--seed", str(seed), "--checkpoint", str(bank),
                       "--output", str(run)]
            print(f"F10 starting {seed}", flush=True)
            job = monitored_run(command, out / f"frozen-{seed}.log")
            result["jobs"].append(dict(seed=seed, exit_code=job["exit_code"],
                                       process_wall_seconds=job["process_wall_seconds"]))
            if job["exit_code"] != 0:
                raise RuntimeError("Incomplete physical run")
            result["jobs"][-1].update(verify_trace(run, refs[seed], seed))
            verify()
            write(out / "result.json", result)
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="stopped", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            if created:
                result["wall_seconds"] = time.monotonic() - started
                write(out / "result.json", result)
        finally:
            if read(lock)["token"] == token:
                lock.unlink()


if __name__ == "__main__":
    with keep_system_awake():
        main()

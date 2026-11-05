"""Execute the registered LR4 frozen load-scale comparison."""

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
from run_interface_f10 import read
from run_resource_envelope import SOURCES, monitored_run
from runtime_diagnostic import CORE, PARENT, inventory
from training_power import keep_system_awake
from training_resources import sample_quiescence

SEEDS = (10006, 10010)
BUDGET = 1450
REPORT_HASHES = (
    "501c089d10cb40ad9702c18c7aba61af8d2ebab927b60052c0516817190fbe30",
    "02c7772367451cbd371d3d8663bb7bae499a67ce82896d227b57ec5e3f31c7da",
)
OBS_HASHES = (
    "38f5147135b2eca5d86941c84ff43b6170d422117729e14762f74233c0f95532",
    "94bcb1417e1967ad45891d90dec7525f2a00d790958023de2a22f741b9a124e4",
)


def summarize(run, reference, seed, scale):
    report, historical = read(run / "report.json"), read(reference / "report.json")
    expected = dict(status="completed_transport_pilot", weights_unchanged=True,
                    learning=False, motor_exploration=0, completed_control_chunks=2000,
                    requested_simulated_seconds_per_world=10, simulated_seconds_per_world=10,
                    checkpoint_sha256=PARENT, seeds=[seed], backend="cpu",
                    controller="brain", worlds=1, control_ms=5)
    if any(report.get(k) != v or historical.get(k) != v for k, v in expected.items()):
        raise ValueError("Incomplete or incompatible frozen receipt")
    settings = {k: historical[k] for k in MATCHED_SETTINGS}
    if scale == 4:
        settings["senses"] = [
            "ground_load_scale_4.0" if x == "ground_load" else x for x in settings["senses"]
        ]
    if any(report[k] != v for k, v in settings.items()) or report.get("leg_load_scale") != (
        4 if scale == 4 else None
    ):
        raise ValueError("Unexpected change in frozen settings")

    with np.load(run / "observations.npz", allow_pickle=False) as data, np.load(
        reference / "observations.npz", allow_pickle=False
    ) as old:
        if set(data.files) != set(old.files) or not old.files:
            raise ValueError("Observation field mismatch")
        for name in old.files:
            x, y = data[name], old[name]
            if x.shape != y.shape or x.shape[0] != 2000 or x.dtype != y.dtype or not np.isfinite(x).all():
                raise ValueError(f"Invalid trace: {name}")
            if scale == 2 and not np.array_equal(x, y):
                raise ValueError(f"Control did not replay F10: {name}")
        obs = {k: data[k][:, 0] for k in ("position", "up_body", "foot_load", "feet_position")}
    standing = standing_mask(obs)
    support = obs["foot_load"][1:] > 1e-5
    speed = np.linalg.norm(np.diff(obs["feet_position"][..., :2], axis=0), axis=-1) / .005
    counts = support.sum(axis=1)
    slip = np.divide((speed * support).sum(axis=1), counts,
                     out=np.zeros(len(counts)), where=counts > 0)
    return dict(scale=scale, replayed_f10=(scale == 2),
                observation_sha256=digest(run / "observations.npz"),
                windows=[dict(standing=float(standing[a:b].mean()),
                              slip_mm_s=float(slip[a - 1:b - 1].mean()),
                              upper_height_samples=int((obs["position"][a:b, 2] > 1.5).sum()))
                         for a, b in ((10, 1000), (1000, 2000))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shared-root", "core-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, core = args.shared_root.resolve(), args.core_source.resolve()
    pilot = Path(__file__).resolve().parents[1]
    source_dir = pilot / "gpu-pilot/src"
    outputs = root / "gpu-pilot/outputs"
    out = outputs / "load-range-lr4-20261105"
    bank = outputs / "e4-exploration-20261029/train-control-1000/weights-last.npz"
    refs = {seed: outputs / f"interface-f10-20261105/frozen-{seed}" for seed in SEEDS}
    pinned = {bank: PARENT}
    for i, seed in enumerate(SEEDS):
        pinned[refs[seed] / "report.json"] = REPORT_HASHES[i]
        pinned[refs[seed] / "observations.npz"] = OBS_HASHES[i]
    head = subprocess.check_output(["git", "-C", str(pilot), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(pilot), "status", "--porcelain",
                                "--", "gpu-pilot/src", "scripts/run_load_range_lr4.py"], text=True).strip():
        raise ValueError("Runner and source must be committed")
    source = tree_digest(source_dir)

    def verify():
        if (tree_digest(core) != SOURCES["core"] or tree_digest(source_dir) != source
                or any(digest(p) != sha for p, sha in pinned.items())):
            raise ValueError("Source, parent or archived F10 input changed")

    verify()
    if list(outputs.glob("*host.lock")):
        raise RuntimeError("Host lease already exists")
    lock = outputs / "codex-training-host.lock"
    token = uuid.uuid4().hex
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w") as f:
        json.dump(dict(owner="codex", experiment="LR4", pid=os.getpid(), token=token,
                       deadline_unix=time.time() + BUDGET), f)
    started = time.monotonic()
    created = False
    result = dict(status="running", jobs=[], comparisons=[], seeds=list(SEEDS), budget_seconds=BUDGET,
                  pilot_commit=head, pilot_source_sha256=source, core_commit=CORE,
                  core_source_sha256=SOURCES["core"], parent_sha256=PARENT,
                  f10_report_sha256=list(REPORT_HASHES), f10_observation_sha256=list(OBS_HASHES),
                  walking_proven=False, model_promoted=False)
    try:
        inventory()
        out.mkdir(exist_ok=False)
        created = True
        write(out / "result.json", result)
        for seed in SEEDS:
            windows = {}
            for scale in (2, 4):
                verify()
                inventory()
                name = f"scale-{scale}-{seed}"
                samples = sample_quiescence()
                decision = resource_decision(samples, True)
                write(out / f"{name}-preflight.json", dict(samples=samples, decision=decision))
                if not decision["passed"] or psutil.virtual_memory().available / 2**30 < 2:
                    raise RuntimeError("Fresh resource gate refused")
                if BUDGET - (time.monotonic() - started) < 335:
                    raise TimeoutError("Insufficient next-job budget")
                run = out / name
                command = [sys.executable, "-m", "noesis_gpu.run", "--backend", "cpu", "--worlds", "1",
                           "--noesis-source", str(core), "--data", str(root / "data"), "--seconds", "10",
                           "--wall-seconds", "300", "--seed", str(seed), "--checkpoint", str(bank),
                           "--output", str(run)]
                if scale == 4:
                    command += ["--leg-load-scale", "4"]
                print(f"LR4 starting {name}", flush=True)
                job = monitored_run(command, out / f"{name}.log")
                row = dict(seed=seed, scale=scale, exit_code=job["exit_code"],
                           process_wall_seconds=job["process_wall_seconds"])
                result["jobs"].append(row)
                if job["exit_code"] != 0:
                    raise RuntimeError("Incomplete physical run")
                row.update(summarize(run, refs[seed], seed, scale))
                windows[scale] = row["windows"]
                verify()
                write(out / "result.json", result)
            early, late = [
                dict(standing_delta=b["standing"] - a["standing"],
                     slip_delta_mm_s=b["slip_mm_s"] - a["slip_mm_s"])
                for a, b in zip(windows[2], windows[4], strict=True)
            ]
            passed = (early["standing_delta"] >= 0 and late["standing_delta"] >= .05
                      and early["slip_delta_mm_s"] <= 0 and late["slip_delta_mm_s"] <= 0)
            result["comparisons"].append(dict(seed=seed, early=early, late=late, passed=passed))
            write(out / "result.json", result)
        result.update(status="completed", exploratory_pass=all(r["passed"] for r in result["comparisons"]))
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

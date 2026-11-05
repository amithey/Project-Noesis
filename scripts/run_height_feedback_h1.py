"""H1: one bounded native v7 birth with anticipatory height feedback."""

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
from run_duration_study import CORE, LOG_FIELDS, PARENT, audit_bank, inventory, resource_decision
from run_resource_envelope import SOURCES, monitored_run
from training_power import keep_system_awake
from training_resources import sample_quiescence

BUDGET = 1000
BASE = "24b69a1a9de17a7ee4abb12cdbb22295ad710371"
MANIFEST = "bb3b95ef79fef2b534bc15c6502f1f2808988a4d830113000c39e95115e33738"
FEEDBACK = dict(ramp_start_mm=1.2, ramp_end_mm=1.5, maximum_reward_cost=.1)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def audit(child, reference, parent):
    report, control = read(child / "report.json"), read(reference / "report.json")
    context = ("backend", "worlds", "seed", "upstream_commit", "noesis_source_sha256",
               "import_checkpoint_sha256", "reward_objective", "learning_rate",
               "eligibility_ms", "motor_exploration_mv_ms", "original_weight_bound",
               "control_ms", "physics_dt", "birth_ms", "neurons", "edges", "plasticity_targets",
               "plasticity_edges", "requested_seconds", "support_measurement")
    if any(report[k] != control[k] for k in context):
        raise ValueError("Unmatched training context")
    expected = dict(status="completed_training_birth", learning=True,
                    completed_chunks=5000, simulated_seconds=25, learning_updates=4990,
                    upper_height_feedback=FEEDBACK, model_promoted=False, walking_proven=False)
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("Incomplete candidate birth")
    bank = child / "weights-last.npz"
    bank_hash = audit_bank(bank, parent)
    if report["saved_checkpoint_sha256"] != bank_hash or bank_hash == PARENT:
        raise ValueError("Unexpected checkpoint receipt")
    sidecar = read(bank.with_suffix(".model.json"))
    if any(sidecar.get(k) != v for k, v in expected.items()):
        raise ValueError("Inconsistent checkpoint context")
    with np.load(bank, allow_pickle=False) as a, np.load(parent, allow_pickle=False) as b:
        for k in ("version", "normalize_reward_error", "bias_correct_reward_error"):
            if not np.array_equal(a[k], b[k]):
                raise ValueError(f"Learning invariant changed: {k}")

    with np.load(child / "training-observations.npz", allow_pickle=False) as a, np.load(
        reference / "training-observations.npz", allow_pickle=False
    ) as b:
        if set(a.files) != set(b.files) | {"native_reward", "upper_height_penalty"}:
            raise ValueError("Unexpected training trace schema")
        for k in a.files:
            if a[k].shape[0] != 5000 or not np.isfinite(a[k]).all():
                raise ValueError(f"Incomplete or nonfinite trace: {k}")
        if any(not np.array_equal(a[k][:10], b[k][:10]) for k in b.files if k != "reward"):
            raise ValueError("Physical prelearning prefix changed")
        if not np.array_equal(a["native_reward"][:10], b["reward"][:10]):
            raise ValueError("Native prelearning reward changed")
        if not np.array_equal(a["learning_active"], np.arange(5000) >= 10):
            raise ValueError("Learning window changed")
        penalty = .1 * np.clip((a["position"][:, 2] - 1.2) / .3, 0, 1)
        if not np.array_equal(a["upper_height_penalty"], penalty):
            raise ValueError("Height penalty receipt mismatch")
        if not np.array_equal(a["reward"], np.clip(a["native_reward"] - penalty, -1, 1)):
            raise ValueError("Reward receipt mismatch")
        changed = int(np.sum(a["reward"][10:] != a["native_reward"][10:]))
    if changed == 0:
        raise ValueError("No active feedback exposure")
    with np.load(child / "exploration-log.npz", allow_pickle=False) as a, np.load(
        reference / "exploration-log.npz", allow_pickle=False
    ) as b:
        if set(a.files) != LOG_FIELDS or set(b.files) != LOG_FIELDS:
            raise ValueError("Exploration schema changed")
        if any(not np.isfinite(a[k]).all() or not np.array_equal(a[k], b[k]) for k in LOG_FIELDS):
            raise ValueError("Exploration schedule changed")
        if not np.array_equal(a["control_chunk_index"], np.arange(10, 5000)):
            raise ValueError("Exploration alignment changed")
    return dict(candidate_sha256=bank_hash, completed_chunks=5000, learning_updates=4990,
                changed_active_rewards=changed, training_standing_fraction=report["standing_fraction"],
                training_stance_slip_mm_s=report["mean_stance_slip_mm_s"],
                physical_prelearning_exact=True, exploration_exact=True, bank_audited=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shared-root", "core-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root, core = args.shared_root.resolve(), args.core_source.resolve()
    pilot = Path(__file__).resolve().parents[1]
    source_dir = pilot / "gpu-pilot/src"
    outputs = root / "gpu-pilot/outputs"
    out = outputs / "height-feedback-h1-20261105"
    parent = outputs / "e4-exploration-20261029/train-control-1000/weights-last.npz"
    reference = outputs / "standing-profile-s1-20261101/train25"
    manifest = root / "outputs/s1-evidence-archive-20261101/standing-profile-s1-manifest.json"
    if digest(manifest) != MANIFEST:
        raise ValueError("Wrong S1 evidence manifest")
    pins = {parent: PARENT}
    for item in read(manifest)["files"]:
        p = root / item["path"]
        if p.parent == reference:
            pins[p] = item["sha256"]
    if len(pins) != 6:
        raise ValueError("Incomplete S1 reference manifest")
    source = tree_digest(source_dir)
    head = subprocess.check_output(["git", "-C", str(pilot), "rev-parse", "HEAD"], text=True).strip()
    runtime_changes = subprocess.check_output(
        ["git", "-C", str(pilot), "diff", "--name-only", BASE, "--", "gpu-pilot/src"], text=True
    ).splitlines()
    if runtime_changes != ["gpu-pilot/src/noesis_gpu/train.py"]:
        raise ValueError("Only train.py may differ from native v7")
    if subprocess.check_output(["git", "-C", str(pilot), "status", "--porcelain", "--",
                                "gpu-pilot/src", "scripts/run_height_feedback_h1.py"], text=True).strip():
        raise ValueError("Runner and source must be committed")

    def verify():
        if (tree_digest(core) != SOURCES["core"] or tree_digest(source_dir) != source
                or any(digest(p) != sha for p, sha in pins.items())):
            raise ValueError("Source, parent or reference hash mismatch")

    verify()
    if list(outputs.glob("*host.lock")):
        raise RuntimeError("Host lease already exists")
    lock = outputs / "codex-training-host.lock"
    token = uuid.uuid4().hex
    with os.fdopen(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY), "w") as f:
        json.dump(dict(owner="codex", experiment="H1", pid=os.getpid(), token=token,
                       deadline_unix=time.time() + BUDGET), f)
    started = time.monotonic()
    created = False
    result = dict(status="running", pilot_commit=head, pilot_source_sha256=source, core_commit=CORE,
                  core_source_sha256=SOURCES["core"], parent_sha256=PARENT,
                  s1_manifest_sha256=MANIFEST, budget_seconds=BUDGET, model_promoted=False,
                  walking_proven=False, upper_height_feedback=FEEDBACK)
    try:
        inventory()
        out.mkdir(exist_ok=False)
        created = True
        write(out / "result.json", result)
        samples = sample_quiescence()
        decision = resource_decision(samples, True)
        write(out / "preflight.json", dict(samples=samples, decision=decision))
        if not decision["passed"] or psutil.virtual_memory().available / 2**30 < 2:
            raise RuntimeError("Fresh resource gate refused")
        if BUDGET - (time.monotonic() - started) < 935:
            raise TimeoutError("Insufficient job budget")
        train = out / "train25"
        command = [sys.executable, "-m", "noesis_gpu.train", "--backend", "cpu",
                   "--noesis-source", str(core), "--data", str(root / "data"),
                   "--checkpoint", str(parent), "--output", str(train), "--seconds", "25",
                   "--wall-seconds", "900", "--seed", "1000", "--profile", "standing",
                   "--learning-rate", ".2", "--exploration-mv-ms", "1.5", "--log-exploration",
                   "--upper-height-cost"]
        print("H1 starting one 25-second training birth", flush=True)
        job = monitored_run(command, out / "train25.log", process_seconds=930)
        result.update(exit_code=job["exit_code"], process_wall_seconds=job["process_wall_seconds"])
        if job["exit_code"] != 0:
            raise RuntimeError("Training process incomplete")
        verify()
        result.update(status="completed_birth_pending_frozen_evaluation",
                      **audit(train, reference, parent))
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

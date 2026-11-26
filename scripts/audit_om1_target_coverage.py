"""Read the registered inputs and verify native muscle-pool coverage offline."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from om1_target_coverage import counts


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(index):
    for role, record in index.items():
        if sha256(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"Digest mismatch: {role}")


def verify_metadata(raw_path, cache_path):
    raw = pd.read_csv(raw_path, low_memory=False).rename(columns={"Root ID": "root_id"})
    cached = pd.read_parquet(cache_path)
    if raw.root_id.isna().any() or not raw.root_id.is_unique:
        raise ValueError("Raw root identifiers must be complete and unique")
    pd.testing.assert_frame_equal(raw, cached, check_dtype=False, check_exact=True)
    return cached


def verify_counts(targets, positive, negative, observed):
    chosen = set(map(int, targets))
    mapped_by_sign = []
    for name, matrix in (("positive", positive), ("negative", negative)):
        coordinates = matrix.tocoo(copy=True)
        coordinates.sum_duplicates()
        entries = [(int(row), int(column)) for row, column, value in
                   zip(coordinates.row, coordinates.col, coordinates.data) if value]
        selected = [(row, column) for row, column in entries if column in chosen]
        mapped = {column for _, column in selected}
        expected = {
            f"{name}_nonempty_pools": len({row for row, _ in entries}),
            f"{name}_covered_pools": len({row for row, _ in selected}),
            f"{name}_selected_targets": len(mapped),
        }
        if any(observed[key] != value for key, value in expected.items()):
            raise ValueError(f"Independent pool count mismatch: {name}")
        mapped_by_sign.append(mapped)
    positive_targets, negative_targets = mapped_by_sign
    union = positive_targets | negative_targets
    expected = {
        "selected_targets": len(chosen),
        "joint_rows": positive.shape[0],
        "selected_targets_both_signs": len(positive_targets & negative_targets),
        "selected_targets_mapped": len(union),
        "selected_targets_unmapped": len(chosen - union),
    }
    if any(observed[key] != value for key, value in expected.items()):
        raise ValueError("Independent target count mismatch")


def audit(manifest):
    started = time.monotonic()
    index = json.loads(manifest.read_text(encoding="utf-8-sig"))
    verify(index)
    paths = {role: Path(record["path"]).resolve() for role, record in index.items()}
    if paths["audit"] != Path(__file__).resolve():
        raise ValueError("Audit source does not match input index")
    if paths["coverage"] != Path(__file__).with_name("om1_target_coverage.py").resolve():
        raise ValueError("Coverage source does not match input index")
    report = json.loads(paths["baseline_report"].read_text(encoding="utf-8"))
    context = report["ss1_context"]
    if (context["experiment"], context["phase"], context["bank"], context["seed"]) != (
        "SS1", "baseline", "parent", 1010
    ) or context["checkpoint_sha256"] != index["parent"]["sha256"]:
        raise ValueError("Incorrect baseline checkpoint context")
    if (paths["baseline_report"].parent / "brain-cache" / "neurons.parquet") != paths["cache"]:
        raise ValueError("Incorrect baseline cache location")
    if not report["ss1_worker_completed"] or report["completed_control_chunks"] != 1000:
        raise ValueError("Incomplete baseline report")
    neurons = verify_metadata(paths["raw"], paths["cache"])
    sys.path[:0] = [str(paths["motor"].parents[1]), str(paths["bridge"].parents[1])]
    from noesis import motor
    from noesis_gpu import bridge

    if Path(motor.__file__).resolve() != paths["motor"] or Path(bridge.__file__).resolve() != paths["bridge"]:
        raise ValueError("Incorrect native imports")
    with np.load(paths["parent"], allow_pickle=False) as checkpoint:
        targets = checkpoint["motor_indices"]
    if len(bridge.LEGACY_JOINTS) != 42 or len(set(bridge.LEGACY_JOINTS)) != 42:
        raise ValueError("Unexpected native joint ordering")
    positive, negative = motor.build_muscle_matrices(neurons, list(bridge.LEGACY_JOINTS))
    result = counts(targets, positive, negative)
    verify_counts(targets, positive, negative, result)
    verify(index)
    if time.monotonic() - started > 180:
        raise TimeoutError("OM1 offline budget exceeded")
    return {
        "status": "completed_om1_offline_audit",
        "metadata_verified_rowwise": True,
        "input_and_source_digests_unchanged": True,
        "independent_count_verified": True,
        "simulation_workers": 0,
        "counts": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

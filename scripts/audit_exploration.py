"""Audit completed exploration pilot banks and summarize frozen outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    args = ap.parse_args()
    output = args.output
    protocol = json.loads((output / "protocol.json").read_text())
    assert sha(args.parent) == protocol["parent_sha256"]
    with np.load(args.parent, allow_pickle=False) as data:
        parent = {key: data[key].copy() for key in data.files}
    banks = {}
    for arm in ("control", "lower"):
        path = output / f"train-{arm}/weights-last.npz"
        report = json.loads((path.parent / "report.json").read_text())
        assert report["status"] == "completed_training_birth" and report["learning_updates"] == 990
        assert report["import_checkpoint_sha256"] == protocol["parent_sha256"]
        assert report["reward_objective"] == protocol["objective"]
        assert report["motor_exploration_mv_ms"] == (1.5 if arm == "control" else 1.)
        assert report["seed"] == protocol["training_seed"]
        with np.load(path, allow_pickle=False) as data:
            for key in ("topology", "motor_indices", "initial_weights", "parameters", "rule"):
                assert np.array_equal(data[key], parent[key]), key
            weights, reference = data["weights"], data["initial_weights"]
            assert np.isfinite(weights).all()
            assert np.array_equal(np.sign(weights), np.sign(reference))
            assert (np.abs(weights) <= 1.5 * np.abs(reference) + 1e-12).all()
            assert (np.abs(weights) >= .5 * np.abs(reference) - 1e-12).all()
            changed = int(np.count_nonzero(weights != parent["weights"]))
        banks[arm] = {"sha256": sha(path), "changed_edges_from_parent": changed,
                      "original_reference_invariants": "passed"}
    source = Path(__file__).resolve().parents[1] / "gpu-pilot/src/noesis_gpu"
    initial = json.loads((output / "pilot-source-files.initial.json").read_text(encoding="utf-8-sig"))
    assert {p.name for p in source.glob("*.py")} == {item["name"] for item in initial}
    for item in initial:
        assert sha(source / item["name"]).lower() == item["Hash"].lower()
    comparison = json.loads((output / "comparison.json").read_text())
    rows = []
    for arm, pairs in comparison["comparisons"].items():
        for pair in pairs:
            for label, case in zip(("parent", arm), pair["cases"]):
                row = {"bank": label, "seed": case["seed"],
                       **{k: case[k] for k in ("standing_fraction", "heading_forward_distance_mm",
                                              "mean_stance_slip_mm_s", "leg_step_counts")}}
                if row not in rows:
                    rows.append(row)
    lower = {r["seed"]: r for r in rows if r["bank"] == "lower"}
    control = {r["seed"]: r for r in rows if r["bank"] == "control"}
    promising = all(low["standing_fraction"] >= .9 and low["heading_forward_distance_mm"] > 0
                     and low["standing_fraction"] >= control[seed]["standing_fraction"]
                     and low["heading_forward_distance_mm"] >= control[seed]["heading_forward_distance_mm"]
                     for seed, low in lower.items())
    result = {"banks": banks, "rows": rows, "lower_exploration_promising_by_preregistered_rule": promising,
              "parent_unchanged": True, "pilot_source_file_hashes_unchanged": True,
              "provenance_note": "Initial protocol pilot_source_sha256 used the core-only helper and is empty-tree hash; per-file pilot hashes and clean git tree provide the independent pilot-source audit.",
              "model_promoted": False, "walking_proven": False}
    (output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

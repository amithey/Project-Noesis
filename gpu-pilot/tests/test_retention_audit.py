"""Offline retention-audit tests on tiny synthetic NPZ checkpoints.

These require no connectome, no physics and no pinned Noesis core: they only
build small pickle-free NPZ files that mimic ``MotorPlasticity.save`` and check
that the audit retains/rejects them correctly and never mutates a file.
"""

import json

import numpy as np
import pytest

from noesis_gpu.retention_audit import DEFAULT_RULE, compare_checkpoints

# A tiny but meaningful synthetic checkpoint: mixed signs and an explicit-zero
# edge so sign preservation and zero handling are both exercised.
INITIAL = np.array([1.0, -2.0, 0.5, 0.0], dtype=float)


def write(path, **overrides):
    """Write a valid checkpoint, with ``overrides`` replacing/removing arrays.

    A value of ``None`` drops the key (to simulate a missing array).
    """
    fields = {
        "version": np.array(1),
        "topology": np.array("topology-hash-abc"),
        "motor_indices": np.array([0, 1, 2, 3], dtype=np.int64),
        "initial_weights": INITIAL.copy(),
        "parameters": np.array([0.2, 100.0, 0.5]),
        "weights": INITIAL.copy(),
        "activity": np.zeros(4),
        "eligibility": np.zeros(4),
        "reward_baseline": np.array(0.0),
        "rule": np.array("perturbation"),
    }
    fields.update(overrides)
    fields = {key: value for key, value in fields.items() if value is not None}
    with open(path, "wb") as handle:
        np.savez(handle, **fields)
    return path


def test_identical_child_reports_full_retention(tmp_path):
    parent = write(tmp_path / "parent.npz")
    child = write(tmp_path / "child.npz")
    report = compare_checkpoints(parent, child)

    assert report["compatible"] is True
    assert report["weights"]["unchanged_fraction"] == 1.0
    assert report["weights"]["changed_count"] == 0
    assert report["weights"]["difference_norm"] == 0.0
    assert report["topology"]["match"] and report["motor_indices"]["match"]
    assert report["initial_weights"]["match"] and report["rule"]["match"]
    # The whole report must round-trip through JSON unchanged in type.
    assert json.loads(json.dumps(report)) == report


def test_learned_change_within_bounds_is_retained_and_quantified(tmp_path):
    parent = write(tmp_path / "parent.npz")
    # Learn: push magnitudes by +20% (well inside the +/-50% bounds), signs kept.
    child = write(tmp_path / "child.npz", weights=INITIAL * 1.2)
    report = compare_checkpoints(parent, child)

    weights = report["weights"]
    # The zero edge stays zero, so only 3 of 4 edges change.
    assert weights["changed_count"] == 3
    assert 0.0 < weights["unchanged_fraction"] < 1.0
    assert weights["within_reference_bounds"] and weights["sign_preserved"]
    assert np.isfinite(weights["difference_norm"]) and weights["difference_norm"] > 0
    assert weights["relative_change"]["child_vs_parent_max"] == pytest.approx(0.2)


def test_reset_state_differences_do_not_break_retention(tmp_path):
    # Eligibility/activity/baseline are learning scratch reset on load; differing
    # here must NOT change the retention verdict for identical learned weights.
    parent = write(tmp_path / "parent.npz")
    child = write(tmp_path / "child.npz", activity=np.full(4, 0.4),
                  eligibility=np.full(4, 0.3), reward_baseline=np.array(0.5))
    report = compare_checkpoints(parent, child)

    assert report["weights"]["unchanged_fraction"] == 1.0
    assert report["keys"]["reset_state_ignored"] is True
    assert "eligibility" in report["keys"]["reset_state_present"]


def test_boundary_change_at_exactly_fifty_percent_is_accepted(tmp_path):
    parent = write(tmp_path / "parent.npz")
    child = write(tmp_path / "child.npz", weights=INITIAL * 1.5)
    report = compare_checkpoints(parent, child)
    assert report["weights"]["within_reference_bounds"] is True


def test_missing_required_array_is_rejected(tmp_path):
    parent = write(tmp_path / "parent.npz")
    child = write(tmp_path / "child.npz", weights=None)
    with pytest.raises(ValueError, match="missing required array"):
        compare_checkpoints(parent, child)


def test_nonfinite_weights_are_rejected(tmp_path):
    parent = write(tmp_path / "parent.npz")
    bad = INITIAL.copy()
    bad[0] = np.inf
    child = write(tmp_path / "child.npz", weights=bad)
    with pytest.raises(ValueError, match="non-finite"):
        compare_checkpoints(parent, child)


def test_sign_flip_is_rejected(tmp_path):
    parent = write(tmp_path / "parent.npz")
    # Same magnitudes (so bounds pass) but every sign flipped.
    child = write(tmp_path / "child.npz", weights=-INITIAL)
    with pytest.raises(ValueError, match="sign"):
        compare_checkpoints(parent, child)


def test_out_of_bounds_weights_are_rejected(tmp_path):
    parent = write(tmp_path / "parent.npz")
    # Double the magnitudes: +100% exceeds the +/-50% reference bound.
    child = write(tmp_path / "child.npz", weights=INITIAL * 2.0)
    with pytest.raises(ValueError, match="magnitude bounds"):
        compare_checkpoints(parent, child)


def test_incompatible_topology_is_rejected(tmp_path):
    parent = write(tmp_path / "parent.npz")
    child = write(tmp_path / "child.npz", topology=np.array("different-network"))
    with pytest.raises(ValueError, match="incompatible topology"):
        compare_checkpoints(parent, child)


def test_incompatible_reference_is_rejected(tmp_path):
    parent = write(tmp_path / "parent.npz")
    # A different frozen reference: we cannot call the child's weights "retained".
    other = np.array([1.0, -2.0, 0.5, 1.0], dtype=float)
    child = write(tmp_path / "child.npz", initial_weights=other, weights=other)
    with pytest.raises(ValueError, match="incompatible reference"):
        compare_checkpoints(parent, child)


def test_incompatible_motor_indices_are_rejected(tmp_path):
    parent = write(tmp_path / "parent.npz")
    child = write(tmp_path / "child.npz", motor_indices=np.array([0, 1, 2, 9], dtype=np.int64))
    with pytest.raises(ValueError, match="incompatible motor_indices"):
        compare_checkpoints(parent, child)


def test_unreadable_file_raises_valueerror(tmp_path):
    parent = write(tmp_path / "parent.npz")
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"this is not an npz archive")
    with pytest.raises(ValueError, match="child checkpoint"):
        compare_checkpoints(parent, junk)


def test_missing_rule_defaults_and_is_reported(tmp_path):
    parent = write(tmp_path / "parent.npz", rule=None)
    child = write(tmp_path / "child.npz")  # keeps "perturbation"
    report = compare_checkpoints(parent, child)
    assert report["rule"]["parent"] == DEFAULT_RULE
    assert report["rule"]["child"] == "perturbation"
    assert report["rule"]["match"] is False


def test_audit_does_not_mutate_the_checkpoint_files(tmp_path):
    parent = write(tmp_path / "parent.npz")
    child = write(tmp_path / "child.npz", weights=INITIAL * 1.1)
    before = (parent.read_bytes(), child.read_bytes())
    compare_checkpoints(parent, child)
    assert (parent.read_bytes(), child.read_bytes()) == before

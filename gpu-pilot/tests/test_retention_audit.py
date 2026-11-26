"""Offline retention-audit tests on tiny synthetic NPZ checkpoints.

These require no connectome, no physics and no pinned Noesis core: they only
build small pickle-free NPZ files that mimic ``MotorPlasticity.save`` and check
that the audit retains/rejects them correctly and never mutates a file.
"""

import json

import numpy as np
import pytest

from noesis_gpu.retention_audit import (
    DEFAULT_RULE,
    compare_checkpoints,
    compare_rollout_summaries,
)

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


# --------------------------------------------------------------------------- #
# compare_rollout_summaries                                                    #
# --------------------------------------------------------------------------- #

def rollout(**overrides):
    """A passing six-leg rollout summary with matched context and provenance.

    ``overrides`` replace fields; a value of ``None`` drops the key entirely.
    """
    summary = {
        # matched evaluation context (strict equality between parent/candidate)
        "seed": 7,
        "duration_s": 5.0,
        "body": "fly-v1",
        "control_ms": 20.0,
        # complete frozen evaluation metadata (exact values required in both)
        "status": "completed_transport_pilot",
        "learning": False,
        "motor_exploration": 0,
        "weights_unchanged": True,
        # provenance: source must match, weight may differ (both supplied)
        "source_sha256": "src-aaa",
        "weight_sha256": "wt-parent",
        # behavioural metrics
        "standing_fraction": 0.95,
        "heading_forward_distance_mm": 3.0,
        "net_forward_displacement_mm": 2.0,
        "mean_stance_slip_mm_s": 4.0,
        # exactly the six canonical legs LF LM LH RF RM RH, integer counts
        "leg_step_counts": {"LF": 3, "LM": 2, "LH": 2, "RF": 3, "RM": 2, "RH": 2},
    }
    summary.update(overrides)
    return {key: value for key, value in summary.items() if value is not None}


def test_rollout_stable_candidate_passes_all_thresholds():
    before = rollout()
    after = rollout(weight_sha256="wt-candidate",
                    standing_fraction=0.94, net_forward_displacement_mm=2.3)
    report = compare_rollout_summaries(before, after)

    assert report["context"]["matched"] is True
    assert report["provenance"]["source_sha256_match"] is True
    # The weight hash legitimately differs but is only reported, never fatal.
    assert report["provenance"]["weight_sha256_match"] is False
    assert report["candidate_meets_all_thresholds"] is True
    assert report["regressed_gates"] == []
    assert report["standing"]["delta"] == pytest.approx(-0.01)
    assert report["net_forward_displacement"]["delta"] == pytest.approx(0.3)
    assert json.loads(json.dumps(report)) == report


def test_rollout_regressed_candidate_fails_and_is_described():
    before = rollout()
    # Candidate falls over: low standing, no progress, high slip, few leg cycles.
    after = rollout(weight_sha256="wt-candidate",
                    standing_fraction=0.40, net_forward_displacement_mm=0.10,
                    mean_stance_slip_mm_s=25.0,
                    leg_step_counts={"LF": 1, "LM": 0, "LH": 1,
                                     "RF": 0, "RM": 1, "RH": 0})
    report = compare_rollout_summaries(before, after)

    assert report["parent_met_all_thresholds"] is True
    assert report["candidate_meets_all_thresholds"] is False
    assert set(report["regressed_gates"]) == {
        "standing_fraction", "net_forward_displacement_mm",
        "mean_stance_slip_mm_s", "leg_step_cycles"}
    assert report["standing"]["delta"] < 0
    checks = report["threshold_checks"]
    assert checks["leg_step_cycles"]["candidate_legs_meeting"] == 0
    assert checks["mean_stance_slip_mm_s"]["candidate_pass"] is False


def test_rollout_primary_heading_passes_while_sensitivity_net_fails():
    # The preregistered PRIMARY forward gate is heading_forward_distance_mm.
    # A candidate that keeps heading progress but loses net displacement passes
    # the primary gate; the net gate is sensitivity-only and only flags here.
    before = rollout()
    after = rollout(weight_sha256="wt-candidate",
                    heading_forward_distance_mm=2.0,  # >= 0.25, primary pass
                    net_forward_displacement_mm=0.10)  # < 0.25, sensitivity fail
    report = compare_rollout_summaries(before, after)

    checks = report["threshold_checks"]
    assert checks["heading_forward_distance_mm"]["role"] == "primary"
    assert checks["heading_forward_distance_mm"]["candidate_pass"] is True
    assert checks["net_forward_displacement_mm"]["role"] == "sensitivity"
    assert checks["net_forward_displacement_mm"]["candidate_pass"] is False
    # Net failing does NOT fail the primary verdict; it only flags sensitivity.
    assert report["candidate_meets_all_thresholds"] is True
    assert report["candidate_meets_sensitivity_thresholds"] is False
    assert report["regressed_primary_gates"] == []
    assert report["regressed_gates"] == ["net_forward_displacement_mm"]


def test_rollout_sensitivity_net_passes_while_primary_heading_fails():
    # The inverse: net displacement looks fine but the PRIMARY heading gate
    # fails. We must NOT silently pass on net; the primary verdict is a fail.
    before = rollout()
    after = rollout(weight_sha256="wt-candidate",
                    heading_forward_distance_mm=0.10,  # < 0.25, primary fail
                    net_forward_displacement_mm=2.0)   # >= 0.25, sensitivity pass
    report = compare_rollout_summaries(before, after)

    checks = report["threshold_checks"]
    assert checks["heading_forward_distance_mm"]["candidate_pass"] is False
    assert checks["net_forward_displacement_mm"]["candidate_pass"] is True
    assert report["candidate_meets_all_thresholds"] is False
    assert report["candidate_meets_sensitivity_thresholds"] is True
    assert report["regressed_primary_gates"] == ["heading_forward_distance_mm"]
    assert "net_forward_displacement_mm" not in report["regressed_gates"]


def test_rollout_missing_heading_metric_raises():
    before = rollout()
    after = rollout(heading_forward_distance_mm=None)  # primary field is required
    with pytest.raises(ValueError, match="missing rollout metric data"):
        compare_rollout_summaries(before, after)


def test_rollout_incompatible_context_raises():
    before = rollout()
    after = rollout(seed=99)  # different seed -> not comparable
    with pytest.raises(ValueError, match="mismatched comparison metadata"):
        compare_rollout_summaries(before, after)

    body_mismatch = rollout(body="mouse-v2")
    with pytest.raises(ValueError, match="mismatched comparison metadata"):
        compare_rollout_summaries(before, body_mismatch)


def test_rollout_missing_context_metadata_raises():
    before = rollout()
    after = rollout(duration_s=None)  # drop a required context key
    with pytest.raises(ValueError, match="missing comparison metadata"):
        compare_rollout_summaries(before, after)


def test_rollout_missing_provenance_raises():
    before = rollout()
    after = rollout(weight_sha256=None)  # provenance must be supplied
    with pytest.raises(ValueError, match="missing provenance"):
        compare_rollout_summaries(before, after)


def test_rollout_missing_metric_data_raises():
    before = rollout()
    after = rollout(net_forward_displacement_mm=None)
    with pytest.raises(ValueError, match="missing rollout metric data"):
        compare_rollout_summaries(before, after)


def test_rollout_nonfinite_metric_raises():
    before = rollout()
    after = rollout(mean_stance_slip_mm_s=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        compare_rollout_summaries(before, after)


def test_rollout_does_not_mutate_inputs():
    before = rollout()
    after = rollout(weight_sha256="wt-candidate", standing_fraction=0.91)
    before_copy = json.loads(json.dumps(before))
    after_copy = json.loads(json.dumps(after))
    compare_rollout_summaries(before, after)
    assert before == before_copy and after == after_copy


# --- focused repair regressions -------------------------------------------- #

def test_rollout_differing_source_sha256_is_rejected():
    # Reproduced bug: sourceA vs sourceB must NOT silently compare as a match.
    before = rollout(source_sha256="sourceA")
    after = rollout(source_sha256="sourceB", weight_sha256="wt-candidate")
    with pytest.raises(ValueError, match="source_sha256"):
        compare_rollout_summaries(before, after)


def test_rollout_differing_weight_sha256_is_permitted_and_reported():
    before = rollout(weight_sha256="wt-parent")
    after = rollout(weight_sha256="wt-candidate")
    report = compare_rollout_summaries(before, after)
    assert report["provenance"]["source_sha256_match"] is True
    assert report["provenance"]["weight_sha256_match"] is False


def test_rollout_fabricated_legs_are_rejected():
    # Reproduced bug: fake1/fake2/fake3/fake4 = 2 must not satisfy the gate.
    before = rollout()
    after = rollout(weight_sha256="wt-candidate",
                    leg_step_counts={"fake1": 2, "fake2": 2, "fake3": 2, "fake4": 2})
    with pytest.raises(ValueError, match="six\\s+canonical legs"):
        compare_rollout_summaries(before, after)


def test_rollout_partial_leg_set_is_rejected():
    before = rollout()
    after = rollout(weight_sha256="wt-candidate",
                    leg_step_counts={"LF": 2, "LM": 2, "LH": 2, "RF": 2})
    with pytest.raises(ValueError, match="six\\s+canonical legs"):
        compare_rollout_summaries(before, after)


def test_rollout_bool_leg_count_is_rejected():
    before = rollout()
    after = rollout(weight_sha256="wt-candidate",
                    leg_step_counts={"LF": True, "LM": 2, "LH": 2,
                                     "RF": 3, "RM": 2, "RH": 2})
    with pytest.raises(ValueError, match="bool"):
        compare_rollout_summaries(before, after)


def test_rollout_fractional_leg_count_is_rejected():
    before = rollout()
    after = rollout(weight_sha256="wt-candidate",
                    leg_step_counts={"LF": 2.5, "LM": 2, "LH": 2,
                                     "RF": 3, "RM": 2, "RH": 2})
    with pytest.raises(ValueError, match="whole integer"):
        compare_rollout_summaries(before, after)


def test_rollout_missing_frozen_metadata_is_rejected():
    before = rollout()
    after = rollout(weight_sha256="wt-candidate", status=None)
    with pytest.raises(ValueError, match="missing frozen evaluation metadata"):
        compare_rollout_summaries(before, after)


def test_rollout_wrong_frozen_metadata_value_is_rejected():
    before = rollout()
    after = rollout(weight_sha256="wt-candidate", learning=True)
    with pytest.raises(ValueError, match="frozen evaluation metadata"):
        compare_rollout_summaries(before, after)


def test_rollout_motor_exploration_bool_is_rejected():
    # motor_exploration == 0 must not be satisfied by False (False == 0).
    before = rollout()
    after = rollout(weight_sha256="wt-candidate", motor_exploration=False)
    with pytest.raises(ValueError, match="motor_exploration"):
        compare_rollout_summaries(before, after)


def test_rollout_timing_context_requires_strict_equality():
    before = rollout()
    after = rollout(weight_sha256="wt-candidate", control_ms=20.0001)
    with pytest.raises(ValueError, match="mismatched comparison metadata"):
        compare_rollout_summaries(before, after)


def test_rollout_negative_seed_is_rejected():
    before = rollout(seed=-1)
    after = rollout(seed=-1, weight_sha256="wt-candidate")
    with pytest.raises(ValueError, match="seed"):
        compare_rollout_summaries(before, after)


def test_rollout_nonpositive_duration_is_rejected():
    before = rollout(duration_s=0.0)
    after = rollout(duration_s=0.0, weight_sha256="wt-candidate")
    with pytest.raises(ValueError, match="duration_s"):
        compare_rollout_summaries(before, after)

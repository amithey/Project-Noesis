"""Offline, non-mutating retention audit for Noesis plasticity NPZ checkpoints.

This module *reads* two checkpoints written by
``noesis.plasticity.MotorPlasticity.save`` and reports whether a ``child``
checkpoint plausibly RETAINED the learned motor synapses of a ``parent``
checkpoint. It never mutates a weight bank, never loads a connectome, and never
runs physics or an LIF network. It is an audit, not a resume: it cannot and
does not assert that the child reproduces the parent's behaviour, that any
success threshold was crossed, or that the animal learned to walk or fly.

The checkpoint layout audited here is the one defined by
``MotorPlasticity.save`` (see ``src/noesis/plasticity.py``):

    version, topology, motor_indices, initial_weights, parameters, weights,
    activity, eligibility, reward_baseline

Of these, ``activity``, ``eligibility`` and ``reward_baseline`` are learning
*scratch*: ``MotorPlasticity.load`` resets them for a fresh brain/body episode
unless ``restore_traces=True``. They are therefore NOT retained synaptic memory
and retention is judged independently of them. The retained reference is
``topology`` + ``motor_indices`` + ``initial_weights`` (the frozen signed
connectome edges the learner was built from); the retained *learned* state is
``weights``, which must stay within +/-``max_change`` of ``|initial_weights|``
with every sign preserved, exactly as the core loader enforces.

Standard library and numpy only. Loads use ``allow_pickle=False``.

See ``docs/workstreams/claude/retention-contract.md`` for the full contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

# Compatibility metadata that MUST be present to establish that two checkpoints
# describe the same frozen reference network and the same selected synapses.
# Without these we cannot distinguish a retained weight from a weight that
# simply belongs to a different network, so a missing one is a hard error.
RETAINED_KEYS = ("topology", "motor_indices", "initial_weights", "parameters", "weights")

# Keys ``save`` writes but ``load`` resets for a fresh episode. They are
# learning scratch, not retained synaptic memory; mismatches here are expected
# and never fail the audit.
RESET_STATE_KEYS = ("activity", "eligibility", "reward_baseline")

# ``rule`` is optional in older checkpoints; the core loader defaults it to
# "hebbian" when absent (see noesis_gpu/controller.import_checkpoint).
DEFAULT_RULE = "hebbian"

# Position of ``max_change`` inside the saved ``parameters`` triple
# [learning_rate, tau_eligibility_ms, max_change].
_MAX_CHANGE_INDEX = 2

# Floating-point slack when testing the magnitude bounds, so a weight that sits
# exactly on the +/-max_change edge is accepted rather than rejected by noise.
_BOUND_RTOL = 1e-9


def _read(path, label):
    """Read every array of a pickle-free NPZ into memory, or raise a clear error."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as handle:
            # Copy out of the zip-backed lazy store so nothing is held open and
            # the original file is never touched again.
            return {key: np.array(handle[key]) for key in handle.files}
    except FileNotFoundError:
        raise ValueError(f"{label} checkpoint not found: {path}")
    except ValueError as error:
        # np.load raises ValueError for pickled/object arrays under
        # allow_pickle=False and for malformed archives.
        raise ValueError(f"{label} checkpoint is not a readable pickle-free NPZ: {error}")
    except Exception as error:  # BadZipFile, OSError, non-npz payloads, ...
        raise ValueError(f"{label} checkpoint could not be read as an NPZ: {error}")


def _validate_structure(data, label):
    """Validate the shapes/dtypes of the retained arrays and normalise them."""
    missing = [key for key in RETAINED_KEYS if key not in data]
    if missing:
        raise ValueError(
            f"{label} checkpoint is missing required array(s): {', '.join(missing)}")

    topology = data["topology"]
    if topology.ndim != 0 or topology.dtype.kind not in "US":
        raise ValueError(f"{label} checkpoint 'topology' must be a scalar string")

    motor = data["motor_indices"]
    if motor.ndim != 1 or (motor.size and motor.dtype.kind not in "iu"):
        raise ValueError(f"{label} checkpoint 'motor_indices' must be a 1-D integer array")

    initial = data["initial_weights"]
    weights = data["weights"]
    for name, array in (("initial_weights", initial), ("weights", weights)):
        if array.ndim != 1 or array.dtype.kind != "f":
            raise ValueError(
                f"{label} checkpoint '{name}' must be a 1-D floating-point array")
        if not np.isfinite(array).all():
            raise ValueError(f"{label} checkpoint '{name}' contains non-finite values")
    if weights.shape != initial.shape:
        raise ValueError(
            f"{label} checkpoint 'weights' shape {tuple(weights.shape)} does not match "
            f"'initial_weights' shape {tuple(initial.shape)}")

    parameters = data["parameters"]
    if (parameters.shape != (3,) or parameters.dtype.kind not in "fiu"
            or not np.isfinite(parameters).all()):
        raise ValueError(
            f"{label} checkpoint 'parameters' must be 3 finite values "
            "[learning_rate, tau_eligibility_ms, max_change]")
    max_change = float(parameters[_MAX_CHANGE_INDEX])
    if not 0 <= max_change < 1:
        raise ValueError(
            f"{label} checkpoint max_change must satisfy 0 <= max_change < 1, got {max_change}")

    return {
        "topology": str(topology),
        "motor_indices": motor.astype(np.int64, copy=False),
        "initial_weights": initial.astype(float, copy=False),
        "weights": weights.astype(float, copy=False),
        "parameters": parameters.astype(float, copy=False),
        "max_change": max_change,
        "rule": str(data["rule"]) if "rule" in data else DEFAULT_RULE,
        "keys": sorted(data.keys()),
    }


def _validate_learned(initial, weights, max_change, label):
    """Reject sign flips and out-of-original-reference magnitude excursions.

    Mirrors the invariants the core ``MotorPlasticity.load`` enforces: every
    learned magnitude stays within ``max_change`` (50% for a pilot checkpoint)
    of its original reference magnitude and no sign is allowed to change.
    Explicit-zero reference edges must remain exactly zero.
    """
    if np.any(np.sign(weights) != np.sign(initial)):
        raise ValueError(
            f"{label} checkpoint flips the sign of at least one learned synapse")
    magnitude = np.abs(initial)
    lower = magnitude * (1 - max_change) * (1 - _BOUND_RTOL)
    upper = magnitude * (1 + max_change) * (1 + _BOUND_RTOL)
    abs_weights = np.abs(weights)
    if np.any(abs_weights < lower) or np.any(abs_weights > upper):
        raise ValueError(
            f"{label} checkpoint has weights outside the original-reference "
            f"+/-{max_change * 100:.0f}% magnitude bounds")


def _relative_change(initial, weights, reference):
    """Per-edge |weights - reference| / |initial|; zero where the reference edge is zero."""
    magnitude = np.abs(initial)
    difference = np.abs(weights - reference)
    return np.divide(difference, magnitude, out=np.zeros_like(magnitude), where=magnitude > 0)


def _finite_float(value):
    """JSON-safe float: numpy scalars are plain floats; guaranteed finite here."""
    return float(value)


def compare_checkpoints(parent_path, child_path):
    """Audit whether ``child`` retains ``parent``'s learned motor synapses.

    Reads both NPZ checkpoints without mutation, validates each independently,
    confirms they share the same frozen reference network, and returns a
    JSON-serializable report. Any structural problem, non-finite weight, sign
    flip, out-of-bounds weight, or incompatible topology/reference raises a
    ``ValueError`` with a message naming the offending checkpoint and reason.

    The report quantifies how much of the learned state was retained (unchanged
    fraction, changed count) and the finite norms of the differences. It makes
    no success-threshold or walking/locomotion claim and does not assert that
    the child resumes the parent exactly.
    """
    parent = _validate_structure(_read(parent_path, "parent"), "parent")
    child = _validate_structure(_read(child_path, "child"), "child")

    # Each checkpoint must individually respect the learning-rule invariants.
    _validate_learned(parent["initial_weights"], parent["weights"],
                       parent["max_change"], "parent")
    _validate_learned(child["initial_weights"], child["weights"],
                      child["max_change"], "child")

    # Compatibility: the same frozen reference network and the same synapses.
    if parent["topology"] != child["topology"]:
        raise ValueError(
            "incompatible topology: parent and child describe different networks")
    if not np.array_equal(parent["motor_indices"], child["motor_indices"]):
        raise ValueError(
            "incompatible motor_indices: parent and child select different synapses")
    if parent["initial_weights"].shape != child["initial_weights"].shape:
        raise ValueError(
            "incompatible reference: initial_weights have different shapes")
    if not np.array_equal(parent["initial_weights"], child["initial_weights"]):
        raise ValueError(
            "incompatible reference: initial_weights differ between the checkpoints")

    initial = parent["initial_weights"]
    parent_w = parent["weights"]
    child_w = child["weights"]

    unchanged_mask = parent_w == child_w
    count = int(parent_w.size)
    changed_count = int(np.count_nonzero(~unchanged_mask))
    unchanged_fraction = float(np.mean(unchanged_mask)) if count else 1.0

    child_vs_parent = _relative_change(initial, child_w, parent_w)
    parent_vs_initial = _relative_change(initial, parent_w, initial)
    child_vs_initial = _relative_change(initial, child_w, initial)

    parent_keys = set(parent["keys"])
    child_keys = set(child["keys"])

    return {
        "audit": "noesis-retention",
        "compatible": True,
        "keys": {
            "parent": parent["keys"],
            "child": child["keys"],
            "retained_required": list(RETAINED_KEYS),
            "parent_only": sorted(parent_keys - child_keys),
            "child_only": sorted(child_keys - parent_keys),
            "reset_state_present": sorted(
                key for key in RESET_STATE_KEYS if key in parent_keys or key in child_keys),
            "reset_state_ignored": True,
        },
        "shapes": {
            "motor_indices": list(parent["motor_indices"].shape),
            "initial_weights": list(initial.shape),
            "weights": list(parent_w.shape),
            "match": True,
        },
        "topology": {"value": parent["topology"], "match": True},
        "motor_indices": {"count": int(parent["motor_indices"].size), "match": True},
        "initial_weights": {
            "count": int(initial.size),
            "match": True,
            "norm": _finite_float(np.linalg.norm(initial)),
        },
        "parameters": {
            "parent": parent["parameters"].tolist(),
            "child": child["parameters"].tolist(),
            "learning_rate": _finite_float(parent["parameters"][0]),
            "tau_eligibility_ms": _finite_float(parent["parameters"][1]),
            "max_change": _finite_float(parent["parameters"][2]),
            "match": bool(np.array_equal(parent["parameters"], child["parameters"])),
        },
        "rule": {
            "parent": parent["rule"],
            "child": child["rule"],
            "match": parent["rule"] == child["rule"],
        },
        "weights": {
            "count": count,
            "unchanged_fraction": unchanged_fraction,
            "changed_count": changed_count,
            "sign_preserved": True,
            "within_reference_bounds": True,
            "max_change_fraction": _finite_float(parent["max_change"]),
            "parent_norm": _finite_float(np.linalg.norm(parent_w)),
            "child_norm": _finite_float(np.linalg.norm(child_w)),
            "difference_norm": _finite_float(np.linalg.norm(child_w - parent_w)),
            "parent_deviation_from_initial_norm":
                _finite_float(np.linalg.norm(parent_w - initial)),
            "child_deviation_from_initial_norm":
                _finite_float(np.linalg.norm(child_w - initial)),
            "relative_change": {
                "parent_vs_initial_max":
                    _finite_float(np.max(parent_vs_initial)) if count else 0.0,
                "child_vs_initial_max":
                    _finite_float(np.max(child_vs_initial)) if count else 0.0,
                "child_vs_parent_mean":
                    _finite_float(np.mean(child_vs_parent)) if count else 0.0,
                "child_vs_parent_max":
                    _finite_float(np.max(child_vs_parent)) if count else 0.0,
            },
        },
        "notes": (
            "Non-mutating offline audit. Reports structural compatibility and the "
            "magnitude of retained-versus-changed learned motor synapses relative to "
            "the shared original reference. It does NOT assert exact resume, any "
            "success threshold, or any walking/locomotion claim."
        ),
    }


# ---------------------------------------------------------------------------
# Rollout-summary retention comparison
# ---------------------------------------------------------------------------
#
# ``compare_checkpoints`` audits the *stored weights*. This complementary audit
# compares two *rollout evaluation summaries* (a ``before`` parent rollout and
# an ``after`` candidate rollout) to describe how locomotion behaviour moved,
# but ONLY when the two rollouts were produced under a matched evaluation
# context. It never runs physics; it reads already-computed summary dicts of the
# shape produced by ``noesis_gpu.evaluate.summarize`` (plus explicit context and
# provenance fields the caller must attach -- see the contract doc).

# Evaluation-context metadata that MUST be present in BOTH summaries and MUST
# match by STRICT equality. Without a matched seed, duration and body/control
# setup the two rollouts are not comparable, so we refuse rather than silently
# compare. Timing context is compared by strict equality (no float slack): a
# matched behavioural comparison must share the exact same evaluation timing.
ROLLOUT_CONTEXT_KEYS = ("seed", "duration_s", "body", "control_ms")

# Frozen evaluation metadata that MUST be present in BOTH summaries and MUST
# hold these exact values. A matched behavioural comparison is only meaningful
# across a completed, non-learning, non-exploring transport pilot whose weights
# were held fixed during the rollout; anything else is not a frozen evaluation.
FROZEN_EVAL_METADATA = {
    "status": "completed_transport_pilot",
    "learning": False,
    "motor_exploration": 0,
    "weights_unchanged": True,
}

# Provenance that must be SUPPLIED in both summaries. The two are treated
# DIFFERENTLY. ``source_sha256`` identifies the evaluation *source* (the fixed
# body/control/eval code path); a matched behavioural comparison REQUIRES the
# same source, so a differing ``source_sha256`` is fatal. ``weight_sha256`` is
# EXPECTED to differ between a parent and a candidate (that is the point of the
# comparison), so a mismatch there is reported but never fatal. A *missing* one
# is always fatal, because an unattributed rollout cannot be compared.
ROLLOUT_PROVENANCE_KEYS = ("source_sha256", "weight_sha256")

# The six canonical insect legs: Left/Right x Front/Mid/Hind. A matched
# behavioural comparison must report step cycles for exactly these legs, so a
# fabricated or partial leg set cannot inflate the >= 4-legs gate.
CANONICAL_LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")

# Behavioural metrics required in both summaries to form deltas and gates.
ROLLOUT_METRIC_KEYS = (
    "standing_fraction",
    "heading_forward_distance_mm",
    "net_forward_displacement_mm",
    "mean_stance_slip_mm_s",
    "leg_step_counts",
)

# Engineering thresholds. These are per-rollout gates, NOT a retention score and
# NOT a statistical test: standing fraction >= 0.90, forward progress >= 0.25 mm,
# mean stance slip <= 10 mm/s, and >= 2 step cycles on >= 4 legs.
#
# Forward progress has TWO metrics and they are NOT interchangeable. The
# preregistered PRIMARY forward gate is ``heading_forward_distance_mm`` (distance
# travelled along the commanded heading). ``net_forward_displacement_mm`` (net
# straight-line displacement) is a SENSITIVITY metric only: it is gated and
# reported, but it never decides the primary pass/fail. Both fields are required
# so the primary gate is never silently substituted by the sensitivity metric.
STANDING_FRACTION_MIN = 0.90
HEADING_FORWARD_DISTANCE_MIN_MM = 0.25  # primary forward gate
NET_FORWARD_DISPLACEMENT_MIN_MM = 0.25  # sensitivity forward gate
STANCE_SLIP_MAX_MM_S = 10.0
LEG_STEP_CYCLES_MIN = 2
MIN_LEGS_MEETING_CYCLES = 4

# Explicit gate roles. The primary set alone decides
# ``candidate_meets_all_thresholds``; the sensitivity set is reported separately.
PRIMARY_GATES = (
    "standing_fraction",
    "heading_forward_distance_mm",
    "mean_stance_slip_mm_s",
    "leg_step_cycles",
)
SENSITIVITY_GATES = ("net_forward_displacement_mm",)


def _finite_number(value, label):
    """Coerce ``value`` to a finite Python float or raise a clear ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label} must be a finite real number, got {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _positive_number(value, label):
    """Finite and strictly positive float, or raise."""
    number = _finite_number(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be strictly positive, got {value!r}")
    return number


def _nonnegative_int(value, label):
    """Coerce ``value`` to a non-negative Python int, rejecting bools/fractions."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not a bool")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, (float, np.floating)):
        as_float = float(value)
        if not np.isfinite(as_float) or as_float != int(as_float):
            raise ValueError(f"{label} must be a whole integer, got {value!r}")
        number = int(as_float)
    else:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if number < 0:
        raise ValueError(f"{label} must be non-negative, got {value!r}")
    return number


def _check_frozen_metadata(summary, label):
    """Require the complete frozen-evaluation metadata with its exact values."""
    for key, expected in FROZEN_EVAL_METADATA.items():
        if key not in summary:
            raise ValueError(
                f"missing frozen evaluation metadata: {label} rollout requires "
                f"'{key}' == {expected!r}")
        actual = summary[key]
        # ``learning``/``weights_unchanged`` are genuine bools; ``motor_exploration``
        # is an integer and must NOT be satisfied by a bool (True == 1).
        if isinstance(expected, bool):
            if not (isinstance(actual, (bool, np.bool_)) and bool(actual) == expected):
                raise ValueError(
                    f"frozen evaluation metadata: {label} rollout '{key}' must be "
                    f"{expected!r}, got {actual!r}")
        else:
            if isinstance(actual, bool) or actual != expected:
                raise ValueError(
                    f"frozen evaluation metadata: {label} rollout '{key}' must be "
                    f"{expected!r}, got {actual!r}")


def _check_rollout_context(before, after):
    """Require matched seed/duration/body/control metadata (strict) or raise."""
    for key in ROLLOUT_CONTEXT_KEYS:
        if key not in before or key not in after:
            raise ValueError(
                f"missing comparison metadata: both rollouts require '{key}'")

    # Validate and normalise each side independently, then compare by STRICT
    # equality so a matched behavioural comparison shares the exact same timing.
    def normalise(summary, side):
        return {
            "seed": _nonnegative_int(summary["seed"], f"{side} 'seed'"),
            "duration_s": _positive_number(summary["duration_s"], f"{side} 'duration_s'"),
            "body": str(summary["body"]),
            "control_ms": _positive_number(summary["control_ms"], f"{side} 'control_ms'"),
        }

    before_ctx = normalise(before, "before")
    after_ctx = normalise(after, "after")
    for key in ROLLOUT_CONTEXT_KEYS:
        if before_ctx[key] != after_ctx[key]:
            raise ValueError(
                f"mismatched comparison metadata: '{key}' differs between rollouts "
                f"({before[key]!r} vs {after[key]!r}); contexts are not comparable")

    context = dict(before_ctx)
    context["matched"] = True
    return context


def _check_rollout_provenance(before, after):
    """Require supplied hashes; require matching source, permit differing weight."""
    for key in ROLLOUT_PROVENANCE_KEYS:
        if key not in before or key not in after:
            raise ValueError(
                f"missing provenance: both rollouts must supply '{key}' to "
                "attribute parent vs candidate")
    parent = {key: str(before[key]) for key in ROLLOUT_PROVENANCE_KEYS}
    candidate = {key: str(after[key]) for key in ROLLOUT_PROVENANCE_KEYS}
    # A matched behavioural comparison must share the same evaluation source;
    # a differing source_sha256 is a different experiment and is fatal.
    if parent["source_sha256"] != candidate["source_sha256"]:
        raise ValueError(
            "mismatched provenance: 'source_sha256' differs between rollouts "
            f"({parent['source_sha256']!r} vs {candidate['source_sha256']!r}); "
            "a matched behavioural comparison requires the same evaluation source")
    return {
        "parent": parent,
        "candidate": candidate,
        "source_sha256_match": True,
        "weight_sha256_match": parent["weight_sha256"] == candidate["weight_sha256"],
        "note": (
            "source_sha256 must match (same evaluation source) and is required "
            "to be equal; weight_sha256 is expected to differ between parent and "
            "candidate and is reported but never required to match."
        ),
    }


def _legs_meeting_cycles(counts, label):
    """Return (legs with >= LEG_STEP_CYCLES_MIN cycles, total legs) for the six legs.

    Requires exactly the six canonical legs (LF LM LH RF RM RH), each with an
    integer non-negative step-cycle count (no bools, no fractional counts), so a
    fabricated or partial leg set cannot inflate the >= 4-legs gate.
    """
    if not isinstance(counts, Mapping):
        raise ValueError(
            f"{label} rollout 'leg_step_counts' must be a leg->count mapping")
    legs = set(counts.keys())
    expected = set(CANONICAL_LEGS)
    if legs != expected:
        raise ValueError(
            f"{label} rollout 'leg_step_counts' must have exactly the six "
            f"canonical legs {list(CANONICAL_LEGS)}, got {sorted(legs)}")
    meeting = 0
    for leg in CANONICAL_LEGS:
        number = _nonnegative_int(
            counts[leg], f"{label} rollout leg_step_counts[{leg!r}]")
        if number >= LEG_STEP_CYCLES_MIN:
            meeting += 1
    return meeting, len(CANONICAL_LEGS)


def _read_rollout_metrics(summary, label):
    """Validate and extract the behavioural metrics needed for the comparison."""
    missing = [key for key in ROLLOUT_METRIC_KEYS if key not in summary]
    if missing:
        raise ValueError(
            f"missing rollout metric data: {label} rollout lacks {', '.join(missing)}")
    metrics = {}
    for key in ("standing_fraction", "heading_forward_distance_mm",
                "net_forward_displacement_mm", "mean_stance_slip_mm_s"):
        metrics[key] = _finite_number(summary[key], f"{label} rollout '{key}'")
    if not 0.0 <= metrics["standing_fraction"] <= 1.0:
        raise ValueError(f"{label} rollout 'standing_fraction' must lie in [0, 1]")
    if metrics["mean_stance_slip_mm_s"] < 0:
        raise ValueError(
            f"{label} rollout 'mean_stance_slip_mm_s' must be non-negative")
    metrics["legs_meeting"], metrics["legs_total"] = _legs_meeting_cycles(
        summary["leg_step_counts"], label)
    return metrics


def _gate(parent_value, candidate_value, threshold, direction):
    """Pass/fail of a single threshold for parent and candidate (no significance)."""
    if direction == "min":
        parent_pass = parent_value >= threshold
        candidate_pass = candidate_value >= threshold
    else:  # "max"
        parent_pass = parent_value <= threshold
        candidate_pass = candidate_value <= threshold
    return {
        "threshold": threshold,
        "direction": direction,
        "parent_value": parent_value,
        "candidate_value": candidate_value,
        "parent_pass": bool(parent_pass),
        "candidate_pass": bool(candidate_pass),
    }


def compare_rollout_summaries(before, after):
    """Compare two matched-context rollout summaries (parent vs candidate).

    ``before`` is the parent rollout summary and ``after`` the candidate. Both
    must carry a matched evaluation context (``seed``, ``duration_s``, ``body``,
    ``control_ms`` -- compared by STRICT equality), complete frozen evaluation
    metadata (``status == "completed_transport_pilot"``, ``learning is False``,
    ``motor_exploration == 0``, ``weights_unchanged is True``) and supplied
    provenance (``source_sha256``, ``weight_sha256``). Missing or mismatched
    *context* metadata, incomplete/wrong frozen metadata, or a differing
    ``source_sha256`` each raise a ``ValueError`` rather than silently comparing
    incomparable rollouts. ``weight_sha256`` is expected to differ between parent
    and candidate and is reported explicitly, never required to match.

    Returns a JSON-serializable report with descriptive standing, heading and
    net-forward deltas plus per-threshold pass/fail. The PRIMARY gates are
    standing (>= 0.90), heading_forward_distance_mm (>= 0.25 mm), stance slip
    (<= 10 mm/s) and leg step cycles (>= 2 cycles on >= 4 legs);
    net_forward_displacement_mm (>= 0.25 mm) is a SENSITIVITY gate, reported but
    never deciding ``candidate_meets_all_thresholds``. Both forward fields are
    required, so the primary heading gate is never silently replaced by the net
    metric. It does NOT mutate its inputs and makes
    NO claim of memory retention, NO claim of forgetting, and NO claim of
    statistical significance -- a single matched pair proves none of those.
    """
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise ValueError(
            "rollout summaries must be mappings (parent 'before', candidate 'after')")

    _check_frozen_metadata(before, "parent")
    _check_frozen_metadata(after, "candidate")
    context = _check_rollout_context(before, after)
    provenance = _check_rollout_provenance(before, after)
    parent = _read_rollout_metrics(before, "parent")
    candidate = _read_rollout_metrics(after, "candidate")

    def delta_block(key):
        return {
            "before": parent[key],
            "after": candidate[key],
            "delta": candidate[key] - parent[key],
        }

    checks = {
        "standing_fraction": {
            "role": "primary",
            **_gate(parent["standing_fraction"], candidate["standing_fraction"],
                    STANDING_FRACTION_MIN, "min"),
        },
        # PRIMARY forward gate: distance travelled along the commanded heading.
        "heading_forward_distance_mm": {
            "role": "primary",
            **_gate(parent["heading_forward_distance_mm"],
                    candidate["heading_forward_distance_mm"],
                    HEADING_FORWARD_DISTANCE_MIN_MM, "min"),
        },
        # SENSITIVITY forward gate: reported and gated, never decides the primary.
        "net_forward_displacement_mm": {
            "role": "sensitivity",
            **_gate(parent["net_forward_displacement_mm"],
                    candidate["net_forward_displacement_mm"],
                    NET_FORWARD_DISPLACEMENT_MIN_MM, "min"),
        },
        "mean_stance_slip_mm_s": {
            "role": "primary",
            **_gate(parent["mean_stance_slip_mm_s"], candidate["mean_stance_slip_mm_s"],
                    STANCE_SLIP_MAX_MM_S, "max"),
        },
        "leg_step_cycles": {
            "role": "primary",
            "threshold_cycles": LEG_STEP_CYCLES_MIN,
            "min_legs": MIN_LEGS_MEETING_CYCLES,
            "parent_legs_meeting": parent["legs_meeting"],
            "candidate_legs_meeting": candidate["legs_meeting"],
            "parent_legs_total": parent["legs_total"],
            "candidate_legs_total": candidate["legs_total"],
            "parent_pass": bool(parent["legs_meeting"] >= MIN_LEGS_MEETING_CYCLES),
            "candidate_pass": bool(candidate["legs_meeting"] >= MIN_LEGS_MEETING_CYCLES),
        },
    }

    # The PRIMARY set alone decides the headline pass/fail; the SENSITIVITY set
    # (net displacement) is summarised separately so the primary heading gate is
    # never silently replaced by the net metric.
    candidate_meets_all = all(checks[g]["candidate_pass"] for g in PRIMARY_GATES)
    parent_met_all = all(checks[g]["parent_pass"] for g in PRIMARY_GATES)
    candidate_meets_sensitivity = all(
        checks[g]["candidate_pass"] for g in SENSITIVITY_GATES)
    parent_met_sensitivity = all(checks[g]["parent_pass"] for g in SENSITIVITY_GATES)
    regressed_gates = sorted(
        name for name, check in checks.items()
        if check["parent_pass"] and not check["candidate_pass"])
    regressed_primary_gates = sorted(
        g for g in PRIMARY_GATES
        if checks[g]["parent_pass"] and not checks[g]["candidate_pass"])

    return {
        "audit": "noesis-rollout-retention",
        "context": context,
        "frozen_evaluation": dict(FROZEN_EVAL_METADATA),
        "provenance": provenance,
        "standing": delta_block("standing_fraction"),
        "heading": delta_block("heading_forward_distance_mm"),
        "net_forward_displacement": delta_block("net_forward_displacement_mm"),
        "stance_slip": delta_block("mean_stance_slip_mm_s"),
        "thresholds": {
            "standing_fraction_min": STANDING_FRACTION_MIN,
            "heading_forward_distance_mm_min": HEADING_FORWARD_DISTANCE_MIN_MM,
            "net_forward_displacement_mm_min": NET_FORWARD_DISPLACEMENT_MIN_MM,
            "mean_stance_slip_mm_s_max": STANCE_SLIP_MAX_MM_S,
            "leg_step_cycles_min": LEG_STEP_CYCLES_MIN,
            "min_legs_meeting_cycles": MIN_LEGS_MEETING_CYCLES,
        },
        "primary_gate_names": list(PRIMARY_GATES),
        "sensitivity_gate_names": list(SENSITIVITY_GATES),
        "threshold_checks": checks,
        "candidate_meets_all_thresholds": bool(candidate_meets_all),
        "parent_met_all_thresholds": bool(parent_met_all),
        "candidate_meets_sensitivity_thresholds": bool(candidate_meets_sensitivity),
        "parent_met_sensitivity_thresholds": bool(parent_met_sensitivity),
        "regressed_gates": regressed_gates,
        "regressed_primary_gates": regressed_primary_gates,
        "notes": (
            "Descriptive comparison of two rollouts under a matched evaluation "
            "context. The PRIMARY forward gate is heading_forward_distance_mm; "
            "net_forward_displacement_mm is a sensitivity metric only and never "
            "decides the primary pass/fail. Threshold pass/fail is a per-rollout "
            "engineering gate, NOT a retention score. It does NOT prove memory "
            "retention, does NOT prove forgetting, and carries NO claim of "
            "statistical significance (a single matched pair). Inputs are not "
            "mutated."
        ),
    }

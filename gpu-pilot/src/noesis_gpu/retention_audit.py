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

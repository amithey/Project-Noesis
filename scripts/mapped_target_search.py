"""Matched perturbations and preregistered TRAIN selection."""
import numpy as np

AMPLITUDE = .005
SETS = ("mapped", "remaining")
SIGNS = (-1, 1)
SEED = 2026111401


def directions(weights, original, mapped):
    weights, original, mapped = map(np.asarray, (weights, original, mapped))
    if weights.ndim != 1 or weights.shape != original.shape or mapped.shape != weights.shape or mapped.dtype.kind != "b":
        raise ValueError("Expected aligned checkpoint vectors and Boolean mask")
    if not np.isfinite(weights).all() or not np.isfinite(original).all():
        raise ValueError("Nonfinite weights")
    reference = np.abs(original)
    step = AMPLITUDE * reference
    eligible = (reference > 0) & (np.abs(weights) - step >= .5 * reference) & (np.abs(weights) + step <= 1.5 * reference)
    groups = (np.flatnonzero(eligible & mapped), np.flatnonzero(eligible & ~mapped))
    size = min(map(len, groups))
    size -= size % 2
    if size < 2:
        raise ValueError("Both sets need headroom for matched perturbations")
    rng = np.random.default_rng(SEED)
    result = {}
    for label, group in zip(SETS, groups):
        edges = rng.choice(group, size, replace=False)
        direction = np.zeros(weights.shape)
        direction[edges[:size // 2]] = AMPLITUDE
        direction[edges[size // 2:]] = -AMPLITUDE
        result[label] = direction
    return result, dict(zip(SETS, map(len, groups)))


def selection(parent, arms):
    parent = np.asarray(parent, dtype=float)
    if parent.shape != (6, 2) or not np.isfinite(parent).all():
        raise ValueError("Require six finite parent reward/angular-speed pairs")
    if set(arms) != {(label, sign) for label in SETS for sign in SIGNS}:
        raise ValueError("Incomplete matched panel")
    results = {}
    winners = {}
    for label in SETS:
        accepted = []
        for sign in SIGNS:
            values = np.asarray(arms[label, sign], dtype=float)
            if values.shape != parent.shape or not np.isfinite(values).all():
                raise ValueError("Require complete finite trial panel")
            delta = values - parent
            gain = float(delta[:, 0].min())
            regressions = int(np.count_nonzero(delta[:, 1] > 0))
            passed = gain > 1e-6 and regressions == 0
            results[f"{label}/{sign:+}"] = {
                "contexts_reward_improved": int(np.count_nonzero(delta[:, 0] > 1e-6)),
                "contexts_angular_speed_increased": regressions,
                "passes_training_gate": passed,
            }
            if passed:
                accepted.append((gain, sign))
        winners[label] = max(accepted, key=lambda pair: pair[0])[1] if accepted else None
    return {"arms": results, "retained_signs": winners}

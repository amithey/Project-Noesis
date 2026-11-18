"""MB1 fixed muscle-pool loading axes and registered training selection."""
import numpy as np
from scipy import sparse

AMPLITUDE = .005
SETS = ("Femur", "Tibia")
SIGNS = (-1, 1)


def directions(weights, original, targets, positive, negative, joint_names):
    weights, original, targets = map(np.asarray, (weights, original, targets))
    if weights.ndim != 1 or original.shape != weights.shape or targets.shape != weights.shape:
        raise ValueError("Expected aligned incoming-edge vectors")
    if targets.dtype.kind not in "iu" or not np.isfinite(weights).all() or not np.isfinite(original).all():
        raise ValueError("Invalid checkpoint indices or weights")
    if not sparse.issparse(positive) or not sparse.issparse(negative) or positive.shape != negative.shape:
        raise ValueError("Require aligned native sparse muscle matrices")
    if len(joint_names) != positive.shape[0] or len(set(joint_names)) != len(joint_names):
        raise ValueError("Invalid native joint ordering")
    if np.any(targets < 0) or np.any(targets >= positive.shape[1]):
        raise ValueError("Target index outside native matrix")
    for matrix in (positive, negative):
        if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
            raise ValueError("Invalid muscle pool")
    magnitude = np.abs(original)
    if (np.any(np.sign(weights) != np.sign(original)) or
            np.any(np.abs(weights) < .5 * magnitude) or np.any(np.abs(weights) > 1.5 * magnitude)):
        raise ValueError("Invalid signed original-reference parent")
    result, eligible = {}, {}
    for label in SETS:
        rows = [i for i, name in enumerate(joint_names) if name.endswith(label)]
        if len(rows) != 6:
            raise ValueError("Require six native rows for each axis")
        loading = np.asarray((positive[rows] - negative[rows]).sum(axis=0)).ravel()
        largest = float(np.max(np.abs(loading)))
        if largest == 0:
            raise ValueError("Empty muscle axis")
        delta = AMPLITUDE * loading[targets] / largest
        step = np.abs(delta) * magnitude
        headroom = ((np.abs(weights) - step >= .5 * magnitude) &
                    (np.abs(weights) + step <= 1.5 * magnitude) & (magnitude > 0))
        delta[~headroom] = 0
        if not np.any(delta):
            raise ValueError("Axis has no eligible selected incoming edges")
        result[label] = delta
        eligible[label] = int(np.count_nonzero(delta))
    return result, eligible


def selection(parent, arms):
    parent = np.asarray(parent, dtype=float)
    if parent.shape != (6, 2) or not np.isfinite(parent).all():
        raise ValueError("Require six finite parent reward/angular-speed pairs")
    if set(arms) != {(label, sign) for label in SETS for sign in SIGNS}:
        raise ValueError("Incomplete registered panel")
    summary, winner, best = {}, None, -np.inf
    for label in SETS:
        for sign in SIGNS:
            values = np.asarray(arms[label, sign], dtype=float)
            if values.shape != parent.shape or not np.isfinite(values).all():
                raise ValueError("Incomplete or invalid trial panel")
            delta = values - parent
            gain = float(delta[:, 0].min())
            regressions = int(np.count_nonzero(delta[:, 1] > 0))
            passed = gain > 1e-6 and regressions == 0
            summary[f"{label}/{sign:+}"] = dict(
                contexts_reward_improved=int(np.count_nonzero(delta[:, 0] > 1e-6)),
                contexts_angular_speed_increased=regressions, passes_training_gate=passed)
            if passed and gain > best:
                best, winner = gain, f"{label}/{sign:+}"
    return dict(arms=summary, retained_candidate=winner)

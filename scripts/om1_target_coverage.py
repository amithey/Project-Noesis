"""Count selected checkpoint targets in the native signed muscle pools."""

import numpy as np
from scipy import sparse


def validate_inputs(targets, positive, negative):
    if not all(sparse.issparse(matrix) and matrix.ndim == 2 for matrix in (positive, negative)):
        raise ValueError("Muscle pools must be sparse matrices")
    if positive.shape != negative.shape or min(positive.shape) < 1:
        raise ValueError("Muscle pool shapes do not agree")
    targets = np.asarray(targets)
    if targets.ndim != 1 or targets.dtype.kind not in "iu" or not targets.size:
        raise ValueError("Expected a nonempty vector of integer target indices")
    if np.any(targets < 0) or np.any(targets >= positive.shape[1]):
        raise ValueError("Target index outside the muscle matrices")
    if np.unique(targets).size != targets.size:
        raise ValueError("Duplicate target index")
    matrices = []
    for matrix in (positive, negative):
        canonical = matrix.tocsr(copy=True)
        canonical.sum_duplicates()
        canonical.eliminate_zeros()
        if not np.all(np.isfinite(canonical.data)) or np.any(canonical.data < 0):
            raise ValueError("Muscle pool entries must be finite and nonnegative")
        matrices.append(canonical)
    return targets, *matrices


def counts(targets, positive, negative):
    targets, positive, negative = validate_inputs(targets, positive, negative)
    result = {"selected_targets": int(targets.size), "joint_rows": positive.shape[0]}
    selected_masks = []
    for sign, matrix in (("positive", positive), ("negative", negative)):
        subset = matrix[:, targets]
        selected_masks.append(np.asarray(subset.getnnz(axis=0)) > 0)
        result[f"{sign}_nonempty_pools"] = int(np.count_nonzero(matrix.getnnz(axis=1)))
        result[f"{sign}_covered_pools"] = int(np.count_nonzero(subset.getnnz(axis=1)))
        result[f"{sign}_selected_targets"] = int(np.count_nonzero(selected_masks[-1]))
    pos, neg = selected_masks
    result["selected_targets_both_signs"] = int(np.count_nonzero(pos & neg))
    result["selected_targets_mapped"] = int(np.count_nonzero(pos | neg))
    result["selected_targets_unmapped"] = int(targets.size - result["selected_targets_mapped"])
    return result

"""Verify the real frozen controller import without altering neural execution."""
import hashlib
import numpy as np


def checked_import(native_import, net, path):
    before = net.data.copy()
    indices, indptr = net.indices.copy(), net.indptr.copy()
    with np.load(path, allow_pickle=False) as cp:
        selected = np.flatnonzero(np.isin(indices, cp['motor_indices']))
        original, weights = cp['initial_weights'].copy(), cp['weights'].copy()
    if not len(selected) or len(selected) != len(weights):
        raise ValueError('Checkpoint selected-edge count mismatch')
    np.testing.assert_array_equal(before[selected], original)
    expected = before.copy()
    expected[selected] = weights
    native_import(net, path)
    np.testing.assert_array_equal(net.indices, indices)
    np.testing.assert_array_equal(net.indptr, indptr)
    np.testing.assert_array_equal(net.data, expected)
    return dict(actual_import_exact=True, existing_selected_edges=len(selected),
                changed_selected_edges=int(np.count_nonzero(weights != original)),
                full_weights_sha256=hashlib.sha256(memoryview(net.data)).hexdigest(),
                nonselected_weights_unchanged=True, topology_unchanged=True)

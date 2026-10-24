import numpy as np
import pytest
from scipy import sparse

from noesis.pool_genome import LIMIT, apply, gains


def test_zero_genome_is_identity():
    gp = sparse.csr_matrix(np.array([[1.0, 2.0], [0.0, 3.0]]))
    gn = sparse.csr_matrix(np.array([[0.5, 0.0], [4.0, 1.0]]))
    p, n = apply(gp, gn, np.zeros(4))
    assert (p != gp).nnz == 0 and (n != gn).nnz == 0


def test_rows_scale_independently_for_agonist_and_antagonist():
    gp = sparse.csr_matrix(np.array([[1.0, 1.0], [2.0, 2.0]]))
    gn = sparse.csr_matrix(np.array([[1.0, 1.0], [2.0, 2.0]]))
    genome = np.array([np.log(2.0), 0.0, 0.0, np.log(3.0)])   # agonist: row0 x2 ; antagonist: row1 x3
    p, n = apply(gp, gn, genome)
    assert np.allclose(p.toarray(), [[2.0, 2.0], [2.0, 2.0]])
    assert np.allclose(n.toarray(), [[1.0, 1.0], [6.0, 6.0]])


def test_gains_are_clipped_and_validated():
    a, _ = gains(np.array([10.0, -10.0, 0.0, 0.0]), 2)
    assert a[0] == pytest.approx(np.exp(LIMIT)) and a[1] == pytest.approx(np.exp(-LIMIT))
    with pytest.raises(ValueError):
        gains(np.zeros(3), 2)
    with pytest.raises(ValueError):
        gains(np.array([np.nan, 0, 0, 0]), 2)

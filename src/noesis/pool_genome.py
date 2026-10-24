"""A small genome of muscle-pool gains: one log-gain per leg joint for its agonist pool and one for its antagonist pool.

The connectome and every synaptic weight are untouched; the genome rescales the *pooled motor rate* that the muscle model
turns into activation (``a = r / (r + r_half)``), i.e. a per-joint, per-direction motor gain and nothing else.
Layout of a genome vector of length ``2 * n_joints``: ``[agonist log-gains..., antagonist log-gains...]`` in the order of the
leg joints passed to the muscle model. Gains are exp(clip(log-gain, -LIMIT, +LIMIT)).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

LIMIT = 2.0  # |log gain| <= 2  ->  factor between 0.135 and 7.4


def gains(genome, n_joints: int) -> tuple[np.ndarray, np.ndarray]:
    """(agonist factors, antagonist factors) from a genome; validates length and finiteness."""
    g = np.asarray(genome, float)
    if g.shape != (2 * n_joints,) or not np.all(np.isfinite(g)):
        raise ValueError(f"genome must be a finite vector of length {2 * n_joints}")
    g = np.clip(g, -LIMIT, LIMIT)
    return np.exp(g[:n_joints]), np.exp(g[n_joints:])


def apply(g_pos: sparse.csr_matrix, g_neg: sparse.csr_matrix, genome) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Row-scaled copies of the agonist/antagonist pool matrices (rows = leg joints)."""
    if g_pos.shape[0] != g_neg.shape[0]:
        raise ValueError("pool matrices must have the same number of joints")
    a, b = gains(genome, g_pos.shape[0])
    return (sparse.diags(a) @ g_pos).tocsr(), (sparse.diags(b) @ g_neg).tocsr()

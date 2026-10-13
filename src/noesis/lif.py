"""Event-driven leaky integrate-and-fire network, after Shiu et al., Nature 2024.

State per neuron: membrane potential ``v`` and synaptic input ``g`` (both mV).
    dv/dt = (v_rest - v + g) / tau_m        dg/dt = -g / tau_syn
A presynaptic spike adds ``w_syn * W[i, j]`` mV to ``g_j`` after ``delay``.
Poisson stimulation adds ``stim_weight`` synapse-equivalents per event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class LIFParams:
    v_rest: float = -52.0   # mV (also reset potential)
    v_th: float = -45.0     # mV
    tau_m: float = 20.0     # ms
    tau_syn: float = 5.0    # ms
    t_ref: float = 2.2      # ms
    delay: float = 1.8      # ms
    w_syn: float = 0.275    # mV per synapse
    dt: float = 0.1         # ms


@nb.njit(cache=True)
def _run(indptr, indices, data, n_steps, n, stim_idx, stim_p, stim_w, seed,
         v_rest, v_th, dv_k, dg_k, ref_steps, delay_steps, w_syn):
    np.random.seed(seed)
    v = np.full(n, v_rest)
    g = np.zeros(n)
    refr = np.zeros(n, np.int32)
    ring = np.empty((delay_steps, n), np.int32)
    ring_n = np.zeros(delay_steps, np.int64)
    out_t = np.empty(1 << 20, np.int32)
    out_i = np.empty(1 << 20, np.int32)
    n_out = 0
    for step in range(n_steps):
        slot = step % delay_steps
        # deliver spikes emitted `delay_steps` ago (slot holds exactly those)
        for k in range(ring_n[slot]):
            pre = ring[slot, k]
            for e in range(indptr[pre], indptr[pre + 1]):
                g[indices[e]] += data[e] * w_syn
        ring_n[slot] = 0
        for k in range(stim_idx.shape[0]):
            if np.random.random() < stim_p:
                g[stim_idx[k]] += stim_w * w_syn
        for i in range(n):
            if refr[i] == 0 and g[i] == 0.0 and v[i] == v_rest:
                continue  # quiescent: nothing to integrate (dominant case in sparse activity)
            if refr[i] > 0:
                refr[i] -= 1
                g[i] *= dg_k
                continue
            v[i] += (v_rest - v[i] + g[i]) * dv_k
            g[i] *= dg_k
            if abs(g[i]) < 1e-9:
                g[i] = 0.0
            if g[i] == 0.0 and abs(v[i] - v_rest) < 1e-9:
                v[i] = v_rest
            if v[i] >= v_th:
                v[i] = v_rest
                refr[i] = ref_steps
                ring[slot, ring_n[slot]] = i
                ring_n[slot] += 1
                if n_out == out_t.shape[0]:
                    out_t = np.concatenate((out_t, np.empty_like(out_t)))
                    out_i = np.concatenate((out_i, np.empty_like(out_i)))
                out_t[n_out] = step
                out_i[n_out] = i
                n_out += 1
    return out_t[:n_out], out_i[:n_out]


def simulate(
    W: sparse.csr_matrix,
    duration_ms: float,
    stim_idx: np.ndarray | None = None,
    stim_rate_hz: float = 100.0,
    stim_weight: float = 50.0,
    params: LIFParams | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the network; return (spike_times_ms, neuron_indices)."""
    p = params or LIFParams()
    stim = np.zeros(0, np.int64) if stim_idx is None else np.asarray(stim_idx, np.int64)
    t, i = _run(
        W.indptr.astype(np.int64), W.indices.astype(np.int64), W.data.astype(np.float64),
        round(duration_ms / p.dt), W.shape[0], stim,
        stim_rate_hz * p.dt * 1e-3, stim_weight, seed,
        p.v_rest, p.v_th, p.dt / p.tau_m, float(np.exp(-p.dt / p.tau_syn)),
        round(p.t_ref / p.dt), max(1, round(p.delay / p.dt)), p.w_syn,
    )
    return t * p.dt, i

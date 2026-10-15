"""Event-driven leaky integrate-and-fire network, after Shiu et al., Nature 2024.

State per neuron: membrane potential ``v`` and synaptic input ``g`` (both mV).
    dv/dt = (v_rest - v + g - a) / tau_m    dg/dt = -g / tau_syn    da/dt = -a / tau_a
(``a`` is optional spike-frequency adaptation, off by default.)
A presynaptic spike adds ``w_syn * W[i, j]`` mV to ``g_j`` after ``delay``.
External drive adds ``weight`` synapse-equivalents per Poisson event.
Optional short-term depression scales recurrent transmission by a recovering
presynaptic resource (``depression_u=0`` disables it). External drive is unaffected.

``LIFNetwork`` keeps its state between ``run`` calls so a body simulation can advance it
in lockstep; ``simulate`` is the one-shot convenience wrapper.
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
    # Spike-frequency adaptation: each spike adds ``da`` mV to a slow hyperpolarising term
    # that decays with ``tau_a``; steady-state it is da * rate * tau_a. 0 disables it.
    da: float = 0.0         # mV per spike
    tau_a: float = 200.0    # ms
    # Opt-in normalized presynaptic resource depression; U=0 preserves legacy dynamics.
    depression_u: float = 0.0
    depression_tau_ms: float = 300.0
    depression_excitatory_only: bool = True


@nb.njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@nb.njit(cache=True, inline="always")
def _add(g, in_act, act, nact, j, val):
    g[j] += val
    if in_act[j] == 0:
        in_act[j] = 1
        act[nact[0]] = j
        nact[0] += 1


@nb.njit(cache=True)
def depression_event(resource, last_step, step, dt, utilization, tau_ms):
    """Exact inter-event recovery; return transmission fraction and depleted state.

    Normalized resting efficacy remains one: baseline weights already include
    release probability. Fly ORN->PN depression: Kazama & Wilson 2008,
    https://doi.org/10.1016/j.neuron.2008.02.030. Whole-brain use is an assumption.
    """
    recovered = 1.0 - (1.0 - resource) * np.exp(-(step - last_step) * dt / tau_ms)
    return recovered, recovered * (1.0 - utilization)


@nb.njit(cache=True)
def _run(indptr, indices, data, v, g, adapt, refr, ring, ring_n, act, in_act, nact, step0, n_steps,
         stim_idx, stim_p, stim_w, bg_mean, bg_w, cur_idx, cur_val,
         v_rest, v_th, dv_k, dg_k, ref_steps, w_syn, da_inc, da_k,
         resources, last_release, dep_u, dep_tau_ms, dt, excitatory_only):
    n = v.shape[0]
    delay_steps = ring.shape[0]
    out_t = np.empty(1 << 16, np.int64)
    out_i = np.empty(1 << 16, np.int32)
    n_out = 0
    # Neurons whose g and v are this close to rest are treated as quiescent (<1% of the 7 mV
    # firing gap): keeps the active set small without changing spiking behaviour noticeably.
    eps_g = 5e-3
    eps_a = 5e-2
    eps_v = 5e-2
    for s in range(n_steps):
        step = step0 + s
        slot = step % delay_steps
        # deliver spikes emitted `delay_steps` ago (slot holds exactly those)
        for k in range(ring_n[slot]):
            pre = ring[slot, k]
            efficacy = 1.0
            if dep_u > 0.0:
                efficacy, resources[pre] = depression_event(
                    resources[pre], last_release[pre], step, dt, dep_u, dep_tau_ms)
                last_release[pre] = step
            for e in range(indptr[pre], indptr[pre + 1]):
                _add(g, in_act, act, nact, indices[e], data[e] * w_syn * (efficacy if (not excitatory_only or data[e] > 0) else 1.0))
        ring_n[slot] = 0
        for k in range(cur_idx.shape[0]):  # continuous (graded) input, mV per step
            _add(g, in_act, act, nact, cur_idx[k], cur_val[k])
        for k in range(stim_idx.shape[0]):
            if np.random.random() < stim_p[k]:
                _add(g, in_act, act, nact, stim_idx[k], stim_w[k] * w_syn)
        if bg_mean > 0.0:  # spontaneous background events on random neurons
            for _ in range(np.random.poisson(bg_mean)):
                _add(g, in_act, act, nact, np.random.randint(0, n), bg_w * w_syn)
        keep = 0
        for a in range(nact[0]):  # only neurons with something to integrate
            i = act[a]
            adapt[i] *= da_k
            if refr[i] > 0:
                refr[i] -= 1
                g[i] *= dg_k
            else:
                v[i] += (v_rest - v[i] + g[i] - adapt[i]) * dv_k
                g[i] *= dg_k
                if v[i] >= v_th:
                    v[i] = v_rest
                    adapt[i] += da_inc
                    refr[i] = ref_steps
                    ring[slot, ring_n[slot]] = i
                    ring_n[slot] += 1
                    if n_out == out_t.shape[0]:
                        out_t = np.concatenate((out_t, np.empty_like(out_t)))
                        out_i = np.concatenate((out_i, np.empty_like(out_i)))
                    out_t[n_out] = step
                    out_i[n_out] = i
                    n_out += 1
            if abs(g[i]) < eps_g:
                g[i] = 0.0
            if abs(adapt[i]) < eps_a:
                adapt[i] = 0.0
            if g[i] == 0.0 and adapt[i] == 0.0 and refr[i] == 0 and abs(v[i] - v_rest) < eps_v:
                v[i] = v_rest
                in_act[i] = 0
            else:
                act[keep] = i
                keep += 1
        nact[0] = keep
    return out_t[:n_out], out_i[:n_out]


class LIFNetwork:
    """Stateful network. ``run(ms)`` advances time and returns (spike_times_ms, neuron_idx)."""

    def __init__(self, W: sparse.csr_matrix, params: LIFParams | None = None, seed: int = 0):
        self.p = params or LIFParams()
        if not np.isfinite(self.p.depression_u) or not 0 <= self.p.depression_u <= 1:
            raise ValueError('depression_u must be finite and within [0, 1]')
        if not np.isfinite(self.p.depression_tau_ms) or self.p.depression_tau_ms <= 0:
            raise ValueError('depression_tau_ms must be finite and positive')
        self.indptr = W.indptr.astype(np.int64)
        self.indices = W.indices.astype(np.int64)
        self.data = W.data.astype(np.float64)
        self.n = W.shape[0]
        self.v = np.full(self.n, self.p.v_rest)
        self.g = np.zeros(self.n)
        # Lazy recovery at delivery events; preserve state across run() boundaries.
        self.resources = np.ones(self.n)
        self.last_release = np.zeros(self.n, np.int64)
        self.adapt = np.zeros(self.n)
        self.refr = np.zeros(self.n, np.int32)
        delay_steps = max(1, round(self.p.delay / self.p.dt))
        self.ring = np.empty((delay_steps, self.n), np.int32)
        self.ring_n = np.zeros(delay_steps, np.int64)
        self.act = np.empty(self.n, np.int64)  # active-neuron list
        self.in_act = np.zeros(self.n, np.uint8)
        self.nact = np.zeros(1, np.int64)
        self.step = 0
        _seed(seed)

    @property
    def time_ms(self) -> float:
        return self.step * self.p.dt

    def run(
        self,
        ms: float,
        drive_idx: np.ndarray | None = None,
        drive_rate_hz: float | np.ndarray = 100.0,
        drive_weight: float | np.ndarray = 50.0,
        background_hz: float = 0.0,
        background_weight: float = 50.0,
        current_idx: np.ndarray | None = None,
        current_mv_per_ms: float | np.ndarray = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance ``ms``. Poisson drive onto ``drive_idx`` (per-neuron rate/weight allowed);
        ``background_hz`` is the mean event rate *per neuron* on random neurons;
        ``current_mv_per_ms`` is a continuous input to ``g`` of ``current_idx`` (negative inhibits)."""
        p = self.p
        idx = np.zeros(0, np.int64) if drive_idx is None else np.asarray(drive_idx, np.int64)
        prob = np.broadcast_to(np.asarray(drive_rate_hz, float) * p.dt * 1e-3, idx.shape).copy()
        wts = np.broadcast_to(np.asarray(drive_weight, float), idx.shape).copy()
        n_steps = round(ms / p.dt)
        cidx = np.zeros(0, np.int64) if current_idx is None else np.asarray(current_idx, np.int64)
        cval = np.broadcast_to(np.asarray(current_mv_per_ms, float) * p.dt, cidx.shape).copy()
        t, i = _run(
            self.indptr, self.indices, self.data, self.v, self.g, self.adapt, self.refr, self.ring,
            self.ring_n, self.act, self.in_act, self.nact, self.step, n_steps, idx, prob, wts,
            background_hz * self.n * p.dt * 1e-3, background_weight, cidx, cval,
            p.v_rest, p.v_th, p.dt / p.tau_m, float(np.exp(-p.dt / p.tau_syn)),
            round(p.t_ref / p.dt), p.w_syn, p.da, float(np.exp(-p.dt / p.tau_a)),
            self.resources, self.last_release, p.depression_u, p.depression_tau_ms, p.dt, p.depression_excitatory_only,
        )
        self.step += n_steps
        return t * p.dt, i


def simulate(
    W: sparse.csr_matrix,
    duration_ms: float,
    stim_idx: np.ndarray | None = None,
    stim_rate_hz: float = 100.0,
    stim_weight: float = 50.0,
    params: LIFParams | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot run; return (spike_times_ms, neuron_indices)."""
    net = LIFNetwork(W, params, seed)
    return net.run(duration_ms, stim_idx, stim_rate_hz, stim_weight)

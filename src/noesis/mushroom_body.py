"""Intrinsic, dopamine-gated learning in the mushroom body: nothing outside the brain teaches.

In the fly, associative learning is written at the synapses from Kenyon cells (KC, which encode
the odour) onto mushroom-body output neurons (MBON), gated by dopaminergic neurons (DAN) that are
themselves driven by the world (taste, pain, ...) *through the connectome*. This module adds only
the local plasticity rule at those synapses; no reward function or trainer is supplied:

    dw(KC -> MBON) = -eta * trace(KC) * gate * w          (dopamine-paired depression)
    w recovers toward its connectome value with time constant ``tau_recover``

``gate`` is how far the smoothed spike rate of the brain's own DAN population exceeds its slow
baseline plus ``gate_hz`` (background fluctuations are not reward).
The rule (sign, time constants, a single global dopamine signal instead of per-compartment DANs)
is an assumption; the connectome supplies the topology and the initial weights only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from noesis.connectome import Connectome, presynaptic_signs


@dataclass(frozen=True)
class MBPopulations:
    orn: np.ndarray
    pn: np.ndarray
    kc: np.ndarray
    mbon: np.ndarray
    dan: np.ndarray
    apl: np.ndarray = field(default_factory=lambda: np.zeros(0, int))


def populations(neurons: pd.DataFrame) -> MBPopulations:
    sub = neurons["Sub Class"].fillna("")
    cls = neurons["Class"].fillna("")
    ctype = neurons["Primary Cell Type"].fillna("")
    where = lambda mask: np.flatnonzero(np.asarray(mask))
    return MBPopulations(
        orn=where(sub.str.contains("olfactory_receptor")),
        pn=where(cls.eq("antennal_lobe_projection_neuron")),
        kc=where(sub.str.startswith("KC")),
        mbon=where(ctype.str.startswith("MBON")),
        dan=where(ctype.str.startswith(("PAM", "PPL", "PPM")) | cls.eq("mushroom_body_dopaminergic_neuron")),
        apl=where(ctype.eq("APL")),
    )


def with_weak_pathway_edges(cn: Connectome, pairs: pd.DataFrame, pops: MBPopulations,
                            min_syn: int = 5) -> sparse.csr_matrix:
    """``cn.W`` plus the sub-threshold (1..min_syn-1 synapse) edges of the olfactory/MB pathway."""
    n = cn.n
    pre_set = np.zeros(n, bool)
    post_set = np.zeros(n, bool)
    for idx in (pops.orn, pops.pn, pops.kc, pops.dan, pops.mbon, pops.apl):
        pre_set[idx] = True
    for idx in (pops.pn, pops.kc, pops.mbon, pops.dan, pops.apl):
        post_set[idx] = True
    weak = pairs[(pairs["syn"] < min_syn) & pre_set[pairs["pre"]] & post_set[pairs["post"]]]
    sign = presynaptic_signs(cn.neurons)
    extra = sparse.csr_matrix(
        (weak["syn"].to_numpy(np.float32) * sign[weak["pre"].to_numpy()],
         (weak["pre"].to_numpy(), weak["post"].to_numpy())), shape=(n, n))
    out = (cn.W + extra).tocsr()
    out.sort_indices()
    return out


class DopamineGatedPlasticity:
    """Local three-factor plasticity on the KC -> MBON synapses of a running ``LIFNetwork``."""

    def __init__(self, net, pops: MBPopulations, *, eta: float = 5e-4, tau_trace_ms: float = 500.0,
                 tau_dopamine_ms: float = 200.0, tau_baseline_ms: float = 20000.0,
                 tau_recover_ms: float = 60000.0, gate_hz: float = 5.0):
        self.net, self.pops = net, pops
        self.eta, self.tau_trace, self.tau_d = eta, tau_trace_ms, tau_dopamine_ms
        self.tau_base, self.tau_rec = tau_baseline_ms, tau_recover_ms
        n = net.n
        rows = np.repeat(np.arange(n), np.diff(net.indptr))
        is_kc = np.zeros(n, bool)
        is_kc[pops.kc] = True
        is_mbon = np.zeros(n, bool)
        is_mbon[pops.mbon] = True
        mask = is_kc[rows] & is_mbon[net.indices]
        self.pos = np.flatnonzero(mask)            # positions of KC->MBON entries in net.data
        self.pre = rows[self.pos]
        self.w0 = net.data[self.pos].copy()
        self.trace = np.zeros(n)
        self.dopamine = 0.0
        self.baseline: float | None = None  # slow running level of dopamine, set on first step
        self.gate_hz = gate_hz  # dopamine must exceed its baseline by this much to count as reward
        self.enabled = True

    def step(self, spike_idx: np.ndarray, window_ms: float) -> None:
        """Advance traces by ``window_ms`` of spikes and update the plastic weights."""
        counts = np.bincount(spike_idx, minlength=self.net.n)
        self.trace *= np.exp(-window_ms / self.tau_trace)
        self.trace[self.pops.kc] += counts[self.pops.kc]
        rate = counts[self.pops.dan].sum() / max(len(self.pops.dan), 1) / (window_ms * 1e-3)  # Hz
        self.dopamine += (rate - self.dopamine) * (1 - np.exp(-window_ms / self.tau_d))
        if self.baseline is None:
            self.baseline = self.dopamine
        self.baseline += (self.dopamine - self.baseline) * (1 - np.exp(-window_ms / self.tau_base))
        if not self.enabled or len(self.pos) == 0:
            return
        gate = max(0.0, self.dopamine - self.baseline - self.gate_hz)
        w = self.net.data[self.pos]
        w = w - self.eta * (window_ms * 1e-3) * self.trace[self.pre] * gate * w
        w = w + (self.w0 - w) * (1 - np.exp(-window_ms / self.tau_rec))
        # depression only: never stronger than the connectome value, never sign-flipped
        self.net.data[self.pos] = np.where(self.w0 >= 0, np.clip(w, 0.0, self.w0), np.clip(w, self.w0, 0.0))

    @property
    def mean_weight_fraction(self) -> float:
        """Mean current / connectome weight of the plastic synapses (1.0 = untouched)."""
        return float(np.mean(self.net.data[self.pos] / self.w0)) if len(self.pos) else 1.0

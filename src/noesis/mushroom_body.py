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
        live = self.w0 != 0  # zero-weight entries (e.g. purely axonal pairs) carry no information
        return float(np.mean(self.net.data[self.pos][live] / self.w0[live])) if live.any() else 1.0


class GradedAPL:
    """The APL neuron as a non-spiking, graded inhibitor of all Kenyon cells.

    The real APL does not fire action potentials: it depolarises with total Kenyon-cell input and
    releases GABA continuously. A spiking APL saturates (~400 Hz in this model) and cannot scale its
    inhibition, so its spiking output is removed and replaced by a continuous inhibitory input:

        a      <- low-pass( sum_k  w(KC_k -> APL) * rate_k  /  sum_k w(KC_k -> APL) )      [Hz]
        I_KC_i  = -gain * w_syn * syn(APL -> KC_i) * a                                    [mV/ms]

    i.e. APL behaves like a synapse that "fires" at ``gain`` x the input-weighted Kenyon-cell rate.
    ``gain`` is the one free parameter; it is calibrated to the Kenyon-cell sparseness of real flies.
    """

    def __init__(self, net, pops: MBPopulations, gain: float = 1.0, tau_ms: float = 10.0,
                 uniform: bool = False):
        self.net, self.pops, self.gain, self.tau = net, pops, gain, tau_ms
        n = net.n
        rows = np.repeat(np.arange(n), np.diff(net.indptr))
        is_apl = np.zeros(n, bool)
        is_apl[pops.apl] = True
        is_kc = np.zeros(n, bool)
        is_kc[pops.kc] = True
        # KC -> APL input weights and APL -> KC output synapse counts, read from the connectome
        into = is_kc[rows] & is_apl[net.indices]
        self.w_in = np.bincount(rows[into], weights=np.abs(net.data[into]), minlength=n)[pops.kc]
        out = is_apl[rows] & is_kc[net.indices]
        self.w_out = np.bincount(net.indices[out], weights=np.abs(net.data[out]), minlength=n)[pops.kc]
        if uniform:  # APL contacts every KC; BANC misses the synapses of ~24% of them
            self.w_in = np.full(len(pops.kc), self.w_in.mean())
            self.w_out = np.full(len(pops.kc), self.w_out.mean())
        net.data[is_apl[rows]] = 0.0  # non-spiking: no spike-triggered output at all
        self.level = 0.0  # APL activation, Hz-equivalent

    def current(self, spike_idx: np.ndarray, window_ms: float) -> dict:
        """Update APL from the last window's spikes; return ``LIFNetwork.run`` current arguments."""
        counts = np.bincount(spike_idx, minlength=self.net.n)[self.pops.kc]
        rate = counts / (window_ms * 1e-3)
        drive = float(self.w_in @ rate / max(self.w_in.sum(), 1e-9))
        self.level += (drive - self.level) * (1 - np.exp(-window_ms / self.tau))
        mv_per_ms = -self.gain * self.net.p.w_syn * self.w_out * self.level * 1e-3
        return {"current_idx": self.pops.kc, "current_mv_per_ms": mv_per_ms}


def mbon_input_compartments(raw_dir, cache_dir, neurons: pd.DataFrame, mbon: np.ndarray) -> pd.DataFrame:
    """Synapses onto each MBON split by location: ``dend`` inside the mushroom body (MB_* neuropils,
    where MBON dendrites are) and ``axon`` elsewhere. Columns: pre, post (matrix indices), dend, axon."""
    from pathlib import Path

    cache = Path(cache_dir) / "mbon_input_compartments.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rid = neurons["root_id"].to_numpy()
    index = pd.Series(np.arange(len(neurons)), index=rid)
    conn = pd.read_csv(Path(raw_dir) / "connections_princeton.csv",
                       usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"])
    conn = conn[conn["post_root_id"].isin(set(rid[mbon]))]
    inside = conn["neuropil"].astype(str).str.startswith("MB_")
    conn = conn.assign(dend=np.where(inside, conn["syn_count"], 0), axon=np.where(inside, 0, conn["syn_count"]))
    out = conn.groupby(["pre_root_id", "post_root_id"], as_index=False)[["dend", "axon"]].sum()
    out = pd.DataFrame({"pre": index.reindex(out["pre_root_id"]).to_numpy(),
                        "post": index.reindex(out["post_root_id"]).to_numpy(),
                        "dend": out["dend"].to_numpy(np.float32), "axon": out["axon"].to_numpy(np.float32)})
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def compartmentalize_mbons(W: sparse.csr_matrix, comp: pd.DataFrame, neurons: pd.DataFrame,
                           axonal_weight: float) -> sparse.csr_matrix:
    """Re-weight every existing synapse onto an MBON as ``sign * (dend + axonal_weight * axon)``.

    A point neuron cannot tell where a synapse lands; MBONs receive 80% of their synapses outside the
    mushroom body (on their axons in CRE/SMP/LAL), which a point model would weigh like dendritic
    input. ``axonal_weight`` < 1 approximates their small effect on spike initiation (an assumption).
    The connectivity pattern is unchanged; only existing entries are re-weighted.
    """
    from noesis.connectome import presynaptic_signs

    n = W.shape[0]
    coo = W.tocoo()
    key = coo.row.astype(np.int64) * n + coo.col
    ckey = comp["pre"].to_numpy(np.int64) * n + comp["post"].to_numpy(np.int64)
    order = np.argsort(ckey)
    ckey = ckey[order]
    eff = (comp["dend"].to_numpy() + axonal_weight * comp["axon"].to_numpy())[order]
    pos = np.searchsorted(ckey, key)
    hit = (pos < len(ckey)) & (ckey[np.minimum(pos, len(ckey) - 1)] == key)
    sign = presynaptic_signs(neurons)
    data = coo.data.copy()
    data[hit] = sign[coo.row[hit]] * eff[pos[hit]]
    out = sparse.csr_matrix((data, (coo.row, coo.col)), shape=W.shape)
    out.sort_indices()
    return out


def mb_class(cell_type: str) -> str:
    """Coarse mushroom-body class of a FlyWire cell-type name (same rule for BANC and FAFB)."""
    t = str(cell_type)
    if t.startswith("KC"):
        return "KC"
    if t.startswith("MBON"):
        return "MBON"
    if t == "APL":
        return "APL"
    if t.startswith(("PAM", "PPL", "PPM")):
        return "DAN"
    if "PN" in t:
        return "PN"
    return "other"


def kc_rescale_factors(banc_pairs: pd.DataFrame, banc_types: np.ndarray, fafb_dir,
                       min_synapses: int = 1000) -> dict[tuple[str, str], float]:
    """FAFB / BANC synapse ratio for every (pre class, post class) that involves Kenyon cells.

    BANC under-counts Kenyon-cell synapses (e.g. KC -> MBON 18x fewer than FAFB). Classes with fewer
    than ``min_synapses`` in either dataset are left out (no rescaling) rather than trusted.
    """
    from pathlib import Path

    fafb_dir = Path(fafb_dir)
    to_class = np.vectorize(mb_class)
    bc = to_class(banc_types)
    b = pd.DataFrame({"pre": bc[banc_pairs["pre"].to_numpy()], "post": bc[banc_pairs["post"].to_numpy()],
                      "syn": banc_pairs["syn"].to_numpy()})
    ft = pd.read_csv(fafb_dir / "consolidated_cell_types.csv.gz").set_index("root_id")["primary_type"]
    fc = pd.read_csv(fafb_dir / "connections_princeton.csv.gz",
                     usecols=["pre_root_id", "post_root_id", "syn_count"])
    f = pd.DataFrame({"pre": to_class(ft.reindex(fc["pre_root_id"]).fillna("?").to_numpy()),
                      "post": to_class(ft.reindex(fc["post_root_id"]).fillna("?").to_numpy()),
                      "syn": fc["syn_count"].to_numpy()})
    out = {}
    for d in (b, f):
        d.drop(d.index[(d["pre"] != "KC") & (d["post"] != "KC")], inplace=True)
    tb, tf = b.groupby(["pre", "post"])["syn"].sum(), f.groupby(["pre", "post"])["syn"].sum()
    for key in tb.index.intersection(tf.index):
        if tb[key] >= min_synapses and tf[key] >= min_synapses:
            out[key] = float(tf[key] / tb[key])
    return out


def rescale_kc_edges(W: sparse.csr_matrix, cell_types: np.ndarray,
                     factors: dict[tuple[str, str], float]) -> sparse.csr_matrix:
    """Multiply every edge touching a Kenyon cell by the FAFB/BANC factor of its class pair.

    The connectivity pattern stays BANC's (no edges are invented); only the synapse counts of
    existing KC edges are brought to FAFB totals per class pair. Concentrating the missing synapses
    on the pairs BANC did detect is an assumption.
    """
    cls = np.vectorize(mb_class)(cell_types)
    coo = W.tocoo()
    pre_c, post_c = cls[coo.row], cls[coo.col]
    data = coo.data.astype(np.float64).copy()
    for (a, b), factor in factors.items():
        data[(pre_c == a) & (post_c == b)] *= factor
    out = sparse.csr_matrix((data, (coo.row, coo.col)), shape=W.shape)
    out.sort_indices()
    return out

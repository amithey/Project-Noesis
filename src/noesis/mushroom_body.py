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

    def __init__(self, net, pops: MBPopulations, gain: float = 1.0, tau_ms: float = 10.0):
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


# --- mushroom-body circuit from FAFB ------------------------------------------------------------

FAFB_MB_KINDS = ("pn", "kc", "apl", "mbon", "dan")


def _kc_coarse(cell_type: pd.Series) -> pd.Series:
    """Kenyon-cell lobe class from a type name: KCg / KCab / KCapbp (None for an unresolved 'KC')."""
    t = cell_type.fillna("").astype(str)
    coarse = np.select([t.str.startswith("KCapbp"), t.str.startswith("KCab"), t.str.startswith("KCg")],
                       ["KCapbp", "KCab", "KCg"], default="")
    return pd.Series(coarse, index=cell_type.index, dtype=object).replace("", None)


def fafb_populations(fafb_neurons: pd.DataFrame) -> dict[str, np.ndarray]:
    """FAFB root ids per MB population, from ``primary_type`` and FlyWire's ``class`` column."""
    t = fafb_neurons["primary_type"].fillna("").astype(str)
    cls = fafb_neurons["class"].fillna("").astype(str)
    rid = fafb_neurons["root_id"].to_numpy()
    masks = {
        "pn": cls.eq("ALPN"),
        "kc": cls.eq("Kenyon_Cell") | t.str.startswith("KC"),
        "apl": t.eq("APL"),
        "mbon": t.str.startswith("MBON"),
        "dan": t.str.startswith(("PAM", "PPL", "PPM")) | cls.eq("DAN"),
    }
    taken = np.zeros(len(rid), bool)
    out = {}
    for kind in FAFB_MB_KINDS:  # each neuron in one population only
        m = masks[kind].to_numpy() & ~taken
        taken |= m
        out[kind] = rid[m]
    return out


def _match(banc: pd.DataFrame, fafb: pd.DataFrame, levels: list[tuple[str, ...]],
           rng: np.random.Generator) -> dict:
    """One-to-one random matching of BANC rows (index = matrix index) to FAFB rows (index = root id).

    For each level (a tuple of key columns, most specific first) the still-unmatched neurons of each
    key present in both datasets are shuffled and paired up to the smaller count; the rest go on to
    the next, coarser level. Rows with a missing key at a level are skipped there.
    """
    pairs: dict = {}
    for cols in levels:
        b = banc[~banc.index.isin(list(pairs))].dropna(subset=list(cols))
        f = fafb[~fafb.index.isin(list(pairs.values()))].dropna(subset=list(cols))
        if b.empty or f.empty:
            continue
        fgroups = f.groupby(list(cols)).groups
        for key, bidx in sorted(b.groupby(list(cols)).groups.items()):
            if key not in fgroups:
                continue
            bi = rng.permutation(np.asarray(bidx))
            fi = rng.permutation(np.asarray(fgroups[key]))
            k = min(len(bi), len(fi))
            pairs.update(zip(bi[:k].tolist(), fi[:k].tolist()))
    return pairs


@dataclass
class FAFBMushroomBody:
    """Result of ``with_fafb_mushroom_body``."""

    W: sparse.csr_matrix
    banc_to_fafb: pd.Series      # FAFB root id per matched BANC matrix index
    mapped: np.ndarray           # bool per BANC neuron: its MB-internal edges now come from FAFB
    compartments: pd.DataFrame   # FAFB edges onto MBONs: pre, post (BANC indices), dend, axon
    report: pd.DataFrame         # per population: neurons matched, edges / synapses mapped
    unmatched_banc: dict = field(default_factory=dict)  # population -> type counts
    unmatched_fafb: dict = field(default_factory=dict)


def with_fafb_mushroom_body(W: sparse.csr_matrix, neurons: pd.DataFrame, pops: MBPopulations,
                            fafb_neurons: pd.DataFrame, fafb_pairs: pd.DataFrame,
                            seed: int = 0) -> FAFBMushroomBody:
    """Replace the BANC edges among {PN, KC, APL, MBON, DAN} by the FAFB ones, mapped by cell type.

    BANC under-counts Kenyon-cell synapses ~15-20x, so the mushroom-body circuit is taken from FAFB:
    projection neurons, APL, MBONs and DANs are matched by primary type and side (then by type
    alone); Kenyon cells by a seeded random one-to-one assignment within each KC type and side, then
    within each lobe class (KCg / KCab / KCapbp) and side, then within the lobe class. FAFB has more
    Kenyon cells than BANC, so they are subsampled. Every BANC edge between two *matched* neurons of
    these populations is removed and every FAFB edge between their partners is added, signed by the
    BANC presynaptic neuron (``presynaptic_signs``). Unmatched BANC neurons keep their BANC edges;
    everything outside the mushroom-body circuit (ORN -> PN, inputs from and outputs to the rest of
    the brain and nerve cord) stays BANC.

    ``fafb_neurons``: root_id, primary_type, class, side. ``fafb_pairs``: pre_root_id, post_root_id,
    syn (all neuropils) and optionally ``mb`` (synapses inside MB_* neuropils, for MBON compartments).
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    ctype = neurons["Primary Cell Type"].fillna("").astype(str).replace("", None)
    side = neurons["Soma side"] if "Soma side" in neurons else pd.Series(None, index=neurons.index)
    sub = neurons["Sub Class"] if "Sub Class" in neurons else pd.Series(None, index=neurons.index)
    fpops = fafb_populations(fafb_neurons)
    fn = fafb_neurons.set_index("root_id")
    banc_to_fafb: dict = {}
    rows, unmatched_banc, unmatched_fafb = [], {}, {}
    for kind in FAFB_MB_KINDS:
        bidx = getattr(pops, kind)
        bidx = bidx[~np.isin(bidx, list(banc_to_fafb))]
        b = pd.DataFrame({"type": ctype.iloc[bidx].to_numpy(), "side": side.iloc[bidx].to_numpy()},
                         index=bidx)
        f = pd.DataFrame({"type": fn.loc[fpops[kind], "primary_type"].to_numpy(),
                          "side": fn.loc[fpops[kind], "side"].to_numpy()}, index=fpops[kind])
        if kind == "kc":
            coarse = _kc_coarse(ctype.iloc[bidx]).to_numpy()
            b["coarse"] = np.where(pd.isna(coarse), _kc_coarse(sub.iloc[bidx]).to_numpy(), coarse)
            b["coarse"] = b["coarse"].replace({np.nan: None})
            f["coarse"] = _kc_coarse(f["type"]).to_numpy()
            levels = [("type", "side"), ("coarse", "side"), ("coarse",)]
        else:
            levels = [("type", "side"), ("type",)]
        pairs = _match(b, f, levels, rng)
        banc_to_fafb.update(pairs)
        unmatched_banc[kind] = b.loc[~b.index.isin(list(pairs)), "type"].fillna("?").value_counts()
        unmatched_fafb[kind] = f.loc[~f.index.isin(list(pairs.values())), "type"].fillna("?").value_counts()
        rows.append({"population": kind, "BANC neurons": len(b), "FAFB neurons": len(f),
                     "matched": len(pairs)})

    b2f = pd.Series(banc_to_fafb, dtype="int64").sort_index()
    mapped = np.zeros(n, bool)
    mapped[b2f.index.to_numpy()] = True
    f2b = pd.Series(b2f.index.to_numpy(), index=b2f.to_numpy())
    kind_of = {rid: kind for kind in FAFB_MB_KINDS for rid in fpops[kind].tolist()}

    # FAFB edges inside the circuit, and among them those whose two ends were both matched
    internal = fafb_pairs[fafb_pairs["pre_root_id"].isin(list(kind_of))
                          & fafb_pairs["post_root_id"].isin(list(kind_of))]
    keep = (internal["pre_root_id"].isin(f2b.index) & internal["post_root_id"].isin(f2b.index)).to_numpy()
    edges = internal[keep]
    pre = f2b.reindex(edges["pre_root_id"]).to_numpy(np.int64)
    post = f2b.reindex(edges["post_root_id"]).to_numpy(np.int64)
    sign = presynaptic_signs(neurons)
    syn = edges["syn"].to_numpy(np.float32)
    add = sparse.csr_matrix((syn * sign[pre], (pre, post)), shape=(n, n))

    coo = W.tocoo()
    drop = mapped[coo.row] & mapped[coo.col]
    kept = sparse.csr_matrix((coo.data[~drop], (coo.row[~drop], coo.col[~drop])), shape=(n, n))
    out = (kept + add).tocsr()
    out.sort_indices()

    # bookkeeping per presynaptic population
    pre_kind = internal["pre_root_id"].map(kind_of).to_numpy()
    for r in rows:
        k = r["population"]
        m = pre_kind == k
        r["FAFB edges"] = int(m.sum())
        r["FAFB edges mapped"] = int((m & keep).sum())
        r["FAFB syn"] = int(internal["syn"].to_numpy()[m].sum())
        r["FAFB syn mapped"] = int(internal["syn"].to_numpy()[m & keep].sum())
        r["BANC syn replaced"] = int(np.abs(coo.data[drop][np.isin(coo.row[drop], getattr(pops, k))]).sum())

    is_mbon = np.zeros(n, bool)
    is_mbon[pops.mbon] = True
    onto = is_mbon[post]
    mb_in = edges["mb"].to_numpy(np.float32) if "mb" in edges else syn
    comp = pd.DataFrame({"pre": pre[onto], "post": post[onto], "dend": mb_in[onto],
                         "axon": (syn - mb_in)[onto]})
    return FAFBMushroomBody(W=out, banc_to_fafb=b2f, mapped=mapped, compartments=comp,
                            report=pd.DataFrame(rows).set_index("population"),
                            unmatched_banc=unmatched_banc, unmatched_fafb=unmatched_fafb)


def load_fafb_mushroom_body(connections_csv, cell_types_csv,
                            classification_csv) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FAFB neuron table (root_id, primary_type, class, side) and per-pair synapse counts among the
    mushroom-body populations (pre_root_id, post_root_id, syn, mb = synapses in MB_* neuropils)."""
    types = pd.read_csv(cell_types_csv, usecols=["root_id", "primary_type"])
    cls = pd.read_csv(classification_csv, usecols=["root_id", "class", "side"])
    fafb_neurons = cls.merge(types, on="root_id", how="outer")
    ids = np.concatenate(list(fafb_populations(fafb_neurons).values()))
    conn = pd.read_csv(connections_csv, usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"])
    conn = conn[conn["pre_root_id"].isin(ids) & conn["post_root_id"].isin(ids)]
    in_mb = conn["neuropil"].astype(str).str.startswith("MB_")
    conn = conn.assign(mb=np.where(in_mb, conn["syn_count"], 0))
    pairs = (conn.groupby(["pre_root_id", "post_root_id"], as_index=False)[["syn_count", "mb"]].sum()
             .rename(columns={"syn_count": "syn"}))
    return fafb_neurons, pairs

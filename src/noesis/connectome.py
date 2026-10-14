"""Load the FlyWire connectome tables into a signed sparse weight matrix."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# Sign of a presynaptic neuron's effect, keyed by FlyWire's predicted neurotransmitter.
# ACh is excitatory; GABA/Glu/histamine are inhibitory in the fly CNS. Monoamines
# (DA, 5-HT, OA, TA) act modulatory and are treated as excitatory here -- an
# explicit modelling assumption, overridable via ``nt_sign``.
DEFAULT_NT_SIGN: dict[str, int] = {
    "ACH": 1, "GABA": -1, "GLUT": -1, "HIST": -1,
    "DA": 1, "SER": 1, "OCT": 1, "TYR": 1,
}
UNKNOWN_NT_SIGN = 1

# Predicted neurotransmitters that contradict established biology, by cell type. They override the
# table's prediction everywhere. APL, the mushroom body's feedback interneuron, is GABAergic (Liu &
# Davis 2009) but predicted cholinergic here, which turned its global feedback into excitation.
NT_OVERRIDES: dict[str, str] = {
    "APL": "GABA",
    # il3LN6 is GABA-immunopositive and ChAT/vGlut-negative: Salman et al. 2026,
    # Fig. 2I-L, https://doi.org/10.1152/jn.00571.2025. The left BANC cell's
    # predicted ACh contradicts this evidence and its own verified GABA label.
    "il3LN6": "GABA",
}


def presynaptic_signs(neurons: pd.DataFrame, sign_map: dict[str, int] | None = None,
                      unknown_sign: int = UNKNOWN_NT_SIGN) -> np.ndarray:
    """+1 / -1 / 0 per neuron from its predicted neurotransmitter, with ``NT_OVERRIDES`` applied."""
    sign_map = DEFAULT_NT_SIGN if sign_map is None else sign_map
    nt = neurons["Predicted NT type"].copy()
    ctype = neurons.get("Primary Cell Type")
    if ctype is not None:
        for cell_type, override in NT_OVERRIDES.items():
            nt[ctype.eq(cell_type)] = override
    return nt.map(sign_map).fillna(unknown_sign).to_numpy(np.float32)


@dataclass(frozen=True)
class Connectome:
    """Neuron table plus signed synapse counts. ``W[i, j]`` = i -> j (row = presynaptic)."""

    neurons: pd.DataFrame  # one row per neuron, positional index == matrix index
    W: sparse.csr_matrix

    @property
    def n(self) -> int:
        return len(self.neurons)

    def select(self, mask: pd.Series | np.ndarray) -> np.ndarray:
        """Matrix indices of neurons where ``mask`` is true."""
        return np.flatnonzero(np.asarray(mask))


def _region(neuropil: pd.Series) -> pd.Series:
    """Neuropil name without its side suffix (``MB_ML_L`` -> ``MB_ML``)."""
    return neuropil.astype(str).str.replace(r"_[LR]$", "", regex=True)


def neuropil_scale_factors(banc_connections: Path, reference_connections: Path,
                           clip: tuple[float, float] = (1.0, 20.0)) -> dict[str, float]:
    """Reference / BANC synapse ratio per brain region (both sides pooled), clipped to ``clip``.

    BANC detects 20-60% of FAFB's synapses per brain region. Weights calibrated on FAFB counts (Shiu et
    al.) are therefore too weak on BANC; multiplying each synapse by its region's ratio restores
    FAFB-like totals. Pooling sides and clipping damp differences in where the two datasets draw region
    boundaries. Regions absent from the reference (the nerve cord) get no factor, i.e. stay at 1.
    """
    b = pd.read_csv(banc_connections, usecols=["neuropil", "syn_count"])
    f = pd.read_csv(reference_connections, usecols=["neuropil", "syn_count"])
    tb = b.groupby(_region(b["neuropil"]))["syn_count"].sum()
    tf = f.groupby(_region(f["neuropil"]))["syn_count"].sum()
    common = tb.index.intersection(tf.index)
    ratio = (tf[common] / tb[common]).clip(*clip)
    return {str(k): float(v) for k, v in ratio.items()}


def build_connectome(
    neurons_csv: Path,
    connections_csv: Path,
    min_syn: int = 5,
    nt_sign: dict[str, int] | None = None,
    unknown_sign: int = UNKNOWN_NT_SIGN,
    region_scale: dict[str, float] | None = None,
) -> Connectome:
    """Aggregate per-neuropil rows to neuron pairs, drop weak pairs, sign by presynaptic NT.

    ``region_scale`` multiplies each row's synapse count by its region's factor (default 1) before
    aggregation, so ``min_syn`` applies to the corrected counts."""
    sign_map = DEFAULT_NT_SIGN if nt_sign is None else nt_sign
    neurons = pd.read_csv(neurons_csv, low_memory=False).rename(columns={"Root ID": "root_id"})
    if not neurons["root_id"].is_unique:
        raise ValueError("neurons table has duplicate root ids")
    index = pd.Series(np.arange(len(neurons), dtype=np.int64), index=neurons["root_id"])

    cols = ["pre_root_id", "post_root_id", "syn_count"] + (["neuropil"] if region_scale else [])
    conn = pd.read_csv(connections_csv, usecols=cols)
    if region_scale:
        factor = _region(conn["neuropil"]).map(region_scale).fillna(1.0)
        conn["syn_count"] = conn["syn_count"] * factor.to_numpy()
    pairs = conn.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    pairs = pairs[pairs["syn_count"] >= min_syn]

    pre = index.reindex(pairs["pre_root_id"]).to_numpy()
    post = index.reindex(pairs["post_root_id"]).to_numpy()
    if np.isnan(pre.astype(float)).any() or np.isnan(post.astype(float)).any():
        raise ValueError("connections reference root ids missing from the neuron table")

    sign = presynaptic_signs(neurons, sign_map, unknown_sign)
    weight = pairs["syn_count"].to_numpy(np.float32) * sign[pre]
    W = sparse.csr_matrix((weight, (pre, post)), shape=(len(neurons), len(neurons)))
    return Connectome(neurons=neurons.reset_index(drop=True), W=W)


def load_cached(raw_dir: Path, cache_dir: Path, **kwargs) -> Connectome:
    """Build once, then reuse ``cache_dir`` (Parquet + npz) on later runs.

    With ``region_scale`` the weights are cached separately (``..._scaled.npz``)."""
    nfile = cache_dir / "neurons.parquet"
    suffix = "_scaled" if kwargs.get("region_scale") else ""
    wfile = cache_dir / f"W_min{kwargs.get('min_syn', 5)}{suffix}.npz"
    if nfile.exists() and wfile.exists():
        neurons, W = pd.read_parquet(nfile), sparse.load_npz(wfile).tocsr()
        sign = presynaptic_signs(neurons)  # re-sign cached rows (the cache predates overrides)
        rows = np.repeat(np.arange(W.shape[0]), np.diff(W.indptr))
        W.data = np.abs(W.data) * sign[rows]
        return Connectome(neurons, W)
    cn = build_connectome(raw_dir / "neurons.csv", raw_dir / "connections_princeton.csv", **kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cn.neurons.to_parquet(nfile)
    sparse.save_npz(wfile, cn.W)
    return cn


def load_all_pairs(raw_dir: Path, cache_dir: Path, neurons: pd.DataFrame) -> pd.DataFrame:
    """Every neuron pair with >= 1 synapse as (pre, post, syn) matrix indices, cached as Parquet.

    ``build_connectome`` drops pairs below ``min_syn`` for reliability; the olfactory/mushroom-body
    pathway is made of many 1-4 synapse pairs, so it needs this unfiltered table.
    """
    cache = cache_dir / "pairs_all.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    index = pd.Series(np.arange(len(neurons), dtype=np.int64), index=neurons["root_id"])
    conn = pd.read_csv(raw_dir / "connections_princeton.csv",
                       usecols=["pre_root_id", "post_root_id", "syn_count"])
    pairs = conn.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    out = pd.DataFrame({
        "pre": index.reindex(pairs["pre_root_id"]).to_numpy(),
        "post": index.reindex(pairs["post_root_id"]).to_numpy(),
        "syn": pairs["syn_count"].to_numpy(np.int32),
    })
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out

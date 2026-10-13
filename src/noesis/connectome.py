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


def build_connectome(
    neurons_csv: Path,
    connections_csv: Path,
    min_syn: int = 5,
    nt_sign: dict[str, int] | None = None,
) -> Connectome:
    """Aggregate per-neuropil rows to neuron pairs, drop weak pairs, sign by presynaptic NT."""
    sign_map = DEFAULT_NT_SIGN if nt_sign is None else nt_sign
    neurons = pd.read_csv(neurons_csv, low_memory=False).rename(columns={"Root ID": "root_id"})
    if not neurons["root_id"].is_unique:
        raise ValueError("neurons table has duplicate root ids")
    index = pd.Series(np.arange(len(neurons), dtype=np.int64), index=neurons["root_id"])

    conn = pd.read_csv(connections_csv, usecols=["pre_root_id", "post_root_id", "syn_count"])
    pairs = conn.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    pairs = pairs[pairs["syn_count"] >= min_syn]

    pre = index.reindex(pairs["pre_root_id"]).to_numpy()
    post = index.reindex(pairs["post_root_id"]).to_numpy()
    if np.isnan(pre.astype(float)).any() or np.isnan(post.astype(float)).any():
        raise ValueError("connections reference root ids missing from the neuron table")

    sign = (
        neurons["Predicted NT type"].map(sign_map).fillna(UNKNOWN_NT_SIGN).to_numpy(np.float32)
    )
    weight = pairs["syn_count"].to_numpy(np.float32) * sign[pre]
    W = sparse.csr_matrix((weight, (pre, post)), shape=(len(neurons), len(neurons)))
    return Connectome(neurons=neurons.reset_index(drop=True), W=W)


def load_cached(raw_dir: Path, cache_dir: Path, **kwargs) -> Connectome:
    """Build once, then reuse ``cache_dir`` (Parquet + npz) on later runs."""
    nfile = cache_dir / "neurons.parquet"
    wfile = cache_dir / f"W_min{kwargs.get('min_syn', 5)}.npz"
    if nfile.exists() and wfile.exists():
        return Connectome(pd.read_parquet(nfile), sparse.load_npz(wfile).tocsr())
    cn = build_connectome(raw_dir / "neurons.csv", raw_dir / "connections_princeton.csv", **kwargs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cn.neurons.to_parquet(nfile)
    sparse.save_npz(wfile, cn.W)
    return cn

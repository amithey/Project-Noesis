"""Graph-level audit of the connectome used (BANC) against FAFB on the pathways that matter for the fly's senses.

No simulation. Both datasets are reduced to the same coarse classes (ORN, olfactory PN, Kenyon cell, MBON, PAM and PPL1
dopamine cells, APL, descending neurons) from the cell-type names that each table provides, neuron pairs are aggregated over
neuropils and thresholded at >= 5 synapses (the threshold the simulation uses), and the following are compared:
population sizes; for selected class pairs the number of connected neuron pairs, the synapses per presynaptic neuron, and the
share of the postsynaptic class's input synapses (within the audited classes) that comes from the presynaptic class.
Class membership rules are heuristics from type names and are printed so they can be checked.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("ORN", "PN", "KC", "MBON", "PAM", "PPL1", "APL", "DN")
PAIRS = [("ORN", "PN"), ("PN", "KC"), ("KC", "KC"), ("KC", "APL"), ("APL", "KC"), ("KC", "MBON"), ("PN", "MBON"),
         ("MBON", "MBON"), ("MBON", "PAM"), ("MBON", "PPL1"), ("PAM", "KC"), ("PPL1", "KC"), ("PAM", "MBON"),
         ("PPL1", "MBON"), ("MBON", "DN"), ("PN", "DN"), ("KC", "DN"), ("PAM", "DN")]
PN_PATTERN = re.compile(r"(?:^|_)(?:l|v|ad|vP|il|m)?PN[a-z0-9]*$|^M_.*PN")


def classify_fafb(types: pd.Series) -> pd.Series:
    t = types.fillna("").astype(str)
    out = pd.Series("", index=t.index)
    out[t.str.startswith("ORN_")] = "ORN"
    out[t.str.match(PN_PATTERN) & ~t.str.startswith("ORN_")] = "PN"
    out[t.str.startswith("KC")] = "KC"
    out[t.str.startswith("MBON")] = "MBON"
    out[t.str.startswith("PAM")] = "PAM"
    out[t.str.startswith("PPL1")] = "PPL1"
    out[t.str.match(r"^APL$")] = "APL"
    out[t.str.match(r"^DN[a-z]")] = "DN"
    return out


def classify_banc(neurons: pd.DataFrame) -> pd.Series:
    ct = neurons["Primary Cell Type"].fillna("")
    out = pd.Series("", index=neurons.index)
    out[neurons["Class"].eq("olfactory_receptor_neuron")] = "ORN"
    out[neurons["Class"].eq("antennal_lobe_projection_neuron")] = "PN"
    out[neurons["Sub Class"].fillna("").str.startswith("KC")] = "KC"
    out[ct.str.startswith("MBON")] = "MBON"
    out[ct.str.startswith("PAM")] = "PAM"
    out[ct.str.startswith("PPL1")] = "PPL1"
    out[ct.eq("APL")] = "APL"
    out[neurons["Super Class"].eq("descending")] = "DN"
    return out


def audit(pairs: pd.DataFrame, cls: pd.Series) -> dict:
    """pairs: pre, post, syn (>= 5 per pair, aggregated). cls: root id -> class ('' = not audited)."""
    pre_c, post_c = pairs["pre"].map(cls), pairs["post"].map(cls)
    keep = pre_c.isin(CLASSES) & post_c.isin(CLASSES)
    p = pairs[keep].assign(pre_c=pre_c[keep], post_c=post_c[keep])
    sizes = cls[cls.isin(CLASSES)].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()
    in_total = p.groupby("post_c").syn.sum()
    out = {"sizes": sizes, "pairs": {}}
    for a, b in PAIRS:
        sel = p[(p.pre_c == a) & (p.post_c == b)]
        out["pairs"][f"{a}->{b}"] = {
            "connected_pairs": len(sel), "synapses": int(sel.syn.sum()),
            "synapses_per_presyn_neuron": float(sel.syn.sum() / max(sizes[a], 1)),
            "share_of_postsyn_class_input": float(sel.syn.sum() / max(in_total.get(b, 0), 1))}
    return out


def load_fafb(data: Path, min_syn: int) -> tuple[pd.DataFrame, pd.Series]:
    ct = pd.read_csv(data / "fafb" / "consolidated_cell_types.csv.gz")
    cls = classify_fafb(ct["primary_type"])
    cls.index = ct["root_id"].to_numpy()
    cls = cls[~cls.index.duplicated()]
    ids = np.array(cls[cls.isin(CLASSES)].index)
    conn = pd.read_csv(data / "fafb" / "connections_princeton.csv.gz", usecols=["pre_root_id", "post_root_id", "syn_count"])
    conn = conn[conn.pre_root_id.isin(ids) & conn.post_root_id.isin(ids)]
    pairs = conn.groupby(["pre_root_id", "post_root_id"], as_index=False).syn_count.sum()
    pairs.columns = ["pre", "post", "syn"]
    return pairs[pairs.syn >= min_syn], cls


def load_banc(data: Path, min_syn: int) -> tuple[pd.DataFrame, pd.Series]:
    neurons = pd.read_csv(data / "raw" / "neurons.csv", low_memory=False).rename(columns={"Root ID": "root_id"})
    cls = classify_banc(neurons)
    cls.index = neurons["root_id"].to_numpy()
    ids = np.array(cls[cls.isin(CLASSES)].index)
    chunks = []
    for chunk in pd.read_csv(data / "raw" / "connections_princeton.csv", usecols=["pre_root_id", "post_root_id", "syn_count"],
                             chunksize=2_000_000):
        chunks.append(chunk[chunk.pre_root_id.isin(ids) & chunk.post_root_id.isin(ids)])
    conn = pd.concat(chunks)
    pairs = conn.groupby(["pre_root_id", "post_root_id"], as_index=False).syn_count.sum()
    pairs.columns = ["pre", "post", "syn"]
    return pairs[pairs.syn >= min_syn], cls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/connectome_audit.json"))
    ap.add_argument("--min-syn", type=int, default=5)
    a = ap.parse_args()
    res = {}
    for name, loader in (("BANC", load_banc), ("FAFB", load_fafb)):
        pairs, cls = loader(a.data, a.min_syn)
        res[name] = audit(pairs, cls)
        print(f"{name}: sizes {res[name]['sizes']}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n{'pair':14s} | {'BANC pairs':>10s} {'BANC syn/pre':>12s} {'BANC share':>10s} | {'FAFB pairs':>10s} {'FAFB syn/pre':>12s} {'FAFB share':>10s}")
    for key in res["BANC"]["pairs"]:
        b, f = res["BANC"]["pairs"][key], res["FAFB"]["pairs"][key]
        print(f"{key:14s} | {b['connected_pairs']:10d} {b['synapses_per_presyn_neuron']:12.1f} {b['share_of_postsyn_class_input']:10.3f} | "
              f"{f['connected_pairs']:10d} {f['synapses_per_presyn_neuron']:12.1f} {f['share_of_postsyn_class_input']:10.3f}")


if __name__ == "__main__":
    main()

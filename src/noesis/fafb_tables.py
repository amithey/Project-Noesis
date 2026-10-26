"""FAFB neuron table for the mushroom-body transplant when FlyWire's ``classification`` export is not available.

``noesis.mushroom_body.load_fafb_mushroom_body`` needs ``class`` and ``side`` per FAFB neuron from ``classification.csv.gz``, which is not on this
machine. This module rebuilds an equivalent table from the two files that are:
  * class from the consolidated cell type (Kenyon cells ``KC*``, APL, MBON, dopamine cells PAM/PPL/PPM), and for antennal-lobe projection neurons
    (ALPN) from connectivity: at least half of the neuron's input synapses are in antennal-lobe neuropils (``AL_*``) and at most half of its output
    synapses are, which excludes receptor neurons (outputs in the AL) and local neurons (inputs and outputs in the AL);
  * side from the neuropil suffix (``_L`` / ``_R``) that holds the larger share of the neuron's INPUT synapses (dendrites): a proxy for soma
    side that is reliable for neurons with ipsilateral dendrites (projection neurons, Kenyon cells, MBONs, dopamine cells).
The table is an approximation of the FlyWire annotation; counts per population are printed so they can be compared with the earlier ones
(ALPN 685, Kenyon 5,177, MBON 96, DAN 347, APL 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TYPE_RULES = {
    "Kenyon_Cell": lambda t: t.str.startswith("KC"),
    "APL": lambda t: t.eq("APL"),
    "MBON": lambda t: t.str.startswith("MBON"),
    "DAN": lambda t: t.str.startswith(("PAM", "PPL", "PPM")),
}


def neuron_table(types: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """types: root_id, primary_type. stats (index root_id): in_total, in_al, out_total, out_al, in_left, in_right (synapse counts)."""
    t = types.drop_duplicates("root_id").set_index("root_id")["primary_type"]
    out = pd.DataFrame({"root_id": stats.index.union(t.index)})
    out["primary_type"] = out["root_id"].map(t)
    ptype = out["primary_type"].fillna("").astype(str)
    cls = pd.Series("", index=out.index)
    for name, rule in TYPE_RULES.items():
        cls[rule(ptype)] = name
    s = stats.reindex(out["root_id"]).fillna(0).reset_index(drop=True)
    in_al = s["in_al"] / s["in_total"].clip(lower=1)
    out_al = s["out_al"] / s["out_total"].clip(lower=1)
    cls[(cls == "") & (s["in_total"] >= 20) & (in_al >= 0.5) & (s["out_total"] >= 20) & (out_al <= 0.5)] = "ALPN"
    out["class"] = cls.replace("", np.nan)
    side = np.where(s["in_left"] > s["in_right"], "left", np.where(s["in_right"] > s["in_left"], "right", None))
    out["side"] = pd.Series(side, dtype=object)
    return out


def neuron_stats(connections_csv, chunksize: int = 4_000_000) -> pd.DataFrame:
    """Per-neuron synapse counts (input/output, in the antennal lobe, on the left/right) from the FAFB connection table."""
    acc: dict[str, pd.Series] = {}

    def add(name: str, series: pd.Series) -> None:
        acc[name] = series if name not in acc else acc[name].add(series, fill_value=0)

    for chunk in pd.read_csv(connections_csv, usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"],
                             chunksize=chunksize):
        npil = chunk["neuropil"].astype(str)
        syn = chunk["syn_count"]
        is_al = npil.str.startswith("AL_")
        left, right = npil.str.endswith("_L"), npil.str.endswith("_R")
        post, pre = chunk["post_root_id"], chunk["pre_root_id"]
        add("in_total", syn.groupby(post).sum())
        add("in_al", syn[is_al].groupby(post[is_al]).sum())
        add("in_left", syn[left].groupby(post[left]).sum())
        add("in_right", syn[right].groupby(post[right]).sum())
        add("out_total", syn.groupby(pre).sum())
        add("out_al", syn[is_al].groupby(pre[is_al]).sum())
    return pd.DataFrame(acc).fillna(0)


def load_fafb_mushroom_body_from_types(connections_csv, cell_types_csv) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same output as ``load_fafb_mushroom_body`` (neuron table, mushroom-body pair table) without ``classification.csv.gz``."""
    from noesis.mushroom_body import fafb_populations

    types = pd.read_csv(cell_types_csv, usecols=["root_id", "primary_type"])
    neurons = neuron_table(types, neuron_stats(connections_csv))
    ids = np.concatenate(list(fafb_populations(neurons).values()))
    conn = pd.read_csv(connections_csv, usecols=["pre_root_id", "post_root_id", "neuropil", "syn_count"])
    conn = conn[conn["pre_root_id"].isin(ids) & conn["post_root_id"].isin(ids)]
    in_mb = conn["neuropil"].astype(str).str.startswith("MB_")
    conn = conn.assign(mb=np.where(in_mb, conn["syn_count"], 0))
    pairs = (conn.groupby(["pre_root_id", "post_root_id"], as_index=False)[["syn_count", "mb"]].sum()
             .rename(columns={"syn_count": "syn"}))
    return neurons, pairs


def transplant_fafb_mushroom_body(W, neurons, pops, fafb_dir, seed: int = 0, classification=None):
    """Replace the PN/KC/APL/MBON/DAN edges of ``W`` with the FAFB ones (``with_fafb_mushroom_body``) and print the match report.

    Uses ``classification`` (FlyWire ``classification.csv.gz``) when that file exists, otherwise the table rebuilt here."""
    from pathlib import Path

    from noesis.mushroom_body import load_fafb_mushroom_body, with_fafb_mushroom_body

    fdir = Path(fafb_dir)
    conn, types = fdir / "connections_princeton.csv.gz", fdir / "consolidated_cell_types.csv.gz"
    if classification is not None and Path(classification).exists():
        fneurons, fpairs = load_fafb_mushroom_body(conn, types, Path(classification))
    else:
        print("no FAFB classification table found; class and side rebuilt from cell types and connectivity", flush=True)
        fneurons, fpairs = load_fafb_mushroom_body_from_types(conn, types)
    fmb = with_fafb_mushroom_body(W, neurons, pops, fneurons, fpairs, seed=seed)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("mushroom body from FAFB (matched by cell type):")
        print(fmb.report.to_string(), flush=True)
    return fmb

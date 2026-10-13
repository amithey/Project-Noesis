"""Compare mushroom-body synapse coverage between connectome datasets (e.g. BANC vs FAFB).

For each dataset directory it needs two CSV tables (``.csv`` or ``.csv.gz``), found by their columns:

* connections: ``pre_root_id, post_root_id, neuropil, syn_count``
* cell types:  ``root_id`` (or ``Root ID``) and any column whose values include ``MBON...`` and
  ``KC...`` type names (FlyWire's primary/cell type columns qualify)

and reports, per dataset: output synapses per Kenyon cell, KC -> MBON synapses per KC and per MBON,
and the KC share of MBON input inside and outside the mushroom body. If one dataset has far fewer KC
synapses than another, the shortfall is in the data, not in the model.

    python -m noesis.experiments.compare_mb_coverage data/raw "downloaded fafb"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CONN_COLS = {"pre_root_id", "post_root_id", "neuropil", "syn_count"}


def _tables(folder: Path) -> tuple[pd.DataFrame, pd.Series]:
    conn, types = None, None
    for f in sorted(folder.glob("*.csv*")):
        head = pd.read_csv(f, nrows=200, low_memory=False)
        cols = set(head.columns)
        if CONN_COLS <= cols and conn is None:
            conn = pd.read_csv(f, usecols=list(CONN_COLS))
            continue
        id_col = "root_id" if "root_id" in cols else ("Root ID" if "Root ID" in cols else None)
        if id_col is None or types is not None:
            continue
        full = pd.read_csv(f, low_memory=False)
        best, score = None, 0
        for col in full.columns:  # the column naming the most Kenyon cells and MBONs
            vals = full[col].astype(str)
            n_kc, n_mbon = vals.str.startswith("KC").sum(), vals.str.startswith("MBON").sum()
            if n_kc and n_mbon and n_kc + n_mbon > score:
                best, score = col, n_kc + n_mbon
        if best is not None:
            types = full.set_index(id_col)[best].astype(str)
    if conn is None or types is None:
        raise SystemExit(f"{folder}: need a connections table {sorted(CONN_COLS)} and a cell-type table")
    return conn, types


def coverage(folder: Path) -> dict:
    conn, types = _tables(folder)
    kc = set(types.index[types.str.startswith("KC")])
    mbon = set(types.index[types.str.startswith("MBON")])
    from_kc = conn["pre_root_id"].isin(kc)
    to_mbon = conn["post_root_id"].isin(mbon)
    in_mb = conn["neuropil"].astype(str).str.startswith("MB_")
    syn = conn["syn_count"]
    kc_mbon = syn[from_kc & to_mbon].sum()
    inside = syn[to_mbon & in_mb].sum()
    return {
        "Kenyon cells": len(kc),
        "MBONs": len(mbon),
        "output synapses per KC": syn[from_kc].sum() / max(len(kc), 1),
        "KC->MBON synapses per KC": kc_mbon / max(len(kc), 1),
        "KC->MBON synapses per MBON": kc_mbon / max(len(mbon), 1),
        "MBON input inside the MB": f"{inside / max(syn[to_mbon].sum(), 1):.0%}",
        "KC share of MBON input inside the MB": f"{syn[from_kc & to_mbon & in_mb].sum() / max(inside, 1):.0%}",
        "KC share of all MBON input": f"{kc_mbon / max(syn[to_mbon].sum(), 1):.0%}",
    }


REGIONS = {
    "mushroom body (MB_*)": r"^MB_",
    "antennal lobe (AL)": r"^AL_",
    "lateral horn (LH)": r"^LH_",
    "central complex (FB/EB/PB/NO)": r"^(FB|EB|PB|NO)",
    "SMP/SIP/SLP/CRE": r"^(SMP|SIP|SLP|CRE)_",
    "optic lobe (ME/LO/LOP)": r"^(ME|LO|LOP)_",
    "GNG": r"^GNG$",
}


def region_totals(folder: Path) -> dict:
    """Total synapses per brain region (to tell a mushroom-body-specific shortfall from a global one)."""
    conn, _ = _tables(folder)
    region = conn["neuropil"].astype(str)
    out = {name: conn.loc[region.str.match(rx), "syn_count"].sum() for name, rx in REGIONS.items()}
    out["(all, incl. nerve cord if present)"] = conn["syn_count"].sum()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--regions", action="store_true", help="also compare synapse totals per region")
    a = ap.parse_args()
    rows = {f.name or str(f): coverage(f) for f in a.folders}
    table = pd.DataFrame(rows)
    with pd.option_context("display.float_format", "{:,.1f}".format, "display.width", 140):
        print(table.to_string())
    if a.regions:
        reg = pd.DataFrame({f.name or str(f): region_totals(f) for f in a.folders})
        base = reg.columns[0]
        for col in reg.columns[1:]:
            reg[f"{base} / {col}"] = (reg[base] / reg[col]).round(2)
        print()
        print(reg.to_string(float_format="{:,.2f}".format))


if __name__ == "__main__":
    main()

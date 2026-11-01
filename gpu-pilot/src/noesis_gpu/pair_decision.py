"""Explicit real-vs-control decision rule for one training pair (replaces the ambiguous counts of the first replication script).

Per probe, "non-inferior" is a conjunction on the SAME probe: standing_fraction >= the other arm's AND mean stance slip <= the other arm's.
Ties count as non-inferior in both directions, and the reverse direction is counted from the reverse inequalities directly
(never as ``not`` of the forward count). REAL_BETTER needs median gains >= 0.25 mm in heading and net AND non-inferiority on >= 3
of the 4 probes; CONTROL_BETTER is the mirror image. Anything else is INDISTINGUISHABLE, which means neither superiority rule
passed; it is not evidence of equivalence. The first three pair verdicts (docs/workstreams/claude/credit-noise-control) were produced by the
earlier separate-count rule and are NOT rewritten; ``strict-recomputation.json`` there shows what this rule gives on the same data.
"""

from __future__ import annotations

import numpy as np

GAIN_MM = 0.25
MIN_PROBES = 3


def classify_pair(real, control):
    """``real`` and ``control`` map probe -> summary dict (standing_fraction, mean_stance_slip_mm_s, heading_forward_distance_mm,
    net_forward_displacement_mm). Returns the decision dict."""
    if set(real) != set(control) or not real:
        raise ValueError("real and control must cover the same, non-empty probe set")
    probes = sorted(real)
    dh = [real[p]["heading_forward_distance_mm"] - control[p]["heading_forward_distance_mm"] for p in probes]
    dn = [real[p]["net_forward_displacement_mm"] - control[p]["net_forward_displacement_mm"] for p in probes]
    forward = [real[p]["standing_fraction"] >= control[p]["standing_fraction"]
               and real[p]["mean_stance_slip_mm_s"] <= control[p]["mean_stance_slip_mm_s"] for p in probes]
    reverse = [control[p]["standing_fraction"] >= real[p]["standing_fraction"]
               and control[p]["mean_stance_slip_mm_s"] <= real[p]["mean_stance_slip_mm_s"] for p in probes]
    h, n = float(np.median(dh)), float(np.median(dn))
    better = h >= GAIN_MM and n >= GAIN_MM and sum(forward) >= MIN_PROBES
    worse = h <= -GAIN_MM and n <= -GAIN_MM and sum(reverse) >= MIN_PROBES
    return {"verdict": "REAL_BETTER" if better else "CONTROL_BETTER" if worse else "INDISTINGUISHABLE",
            "median_dheading": h, "median_dnet": n, "real_noninferior_probes": int(sum(forward)),
            "control_noninferior_probes": int(sum(reverse)), "probes": probes}

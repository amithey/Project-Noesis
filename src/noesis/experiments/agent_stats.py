"""Statistics for the decoder-steered agent runs (JSON lines written by decoder_forage / virtual_forage).

For each world file: reached-food counts per condition; one-sided Fisher exact test of 'hungry decoder reached
more often than control' for every control, Holm-corrected within the world; paired final-distance differences
(hungry decoder minus control) with an exact one-sided sign test over start poses. Poses missing a condition are
dropped from that paired comparison. Nothing here assumes the start poses are independent beyond what is stated.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

MAIN = "decoder_hungry"


def fisher_greater(a: int, n_a: int, b: int, n_b: int) -> float:
    """One-sided Fisher exact p for 'group A has more successes than B' (hypergeometric tail)."""
    if min(n_a, n_b) <= 0 or a > n_a or b > n_b:
        raise ValueError("invalid counts")
    total, succ = n_a + n_b, a + b
    return float(sum(comb(succ, k) * comb(total - succ, n_a - k)
                     for k in range(a, min(n_a, succ) + 1)) / comb(total, n_a))


def sign_test_less(diffs: list[float]) -> float:
    """Exact one-sided sign test for 'median difference < 0' (ties dropped)."""
    nonzero = [d for d in diffs if d != 0]
    n, k = len(nonzero), sum(d < 0 for d in nonzero)
    return 1.0 if n == 0 else float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[key]))
        out[key] = running
    return out


def analyse(rows: list[dict]) -> dict:
    conds = sorted({r["condition"] for r in rows})
    by = {c: {r["seed"]: r for r in rows if r["condition"] == c} for c in conds}
    main = by[MAIN]
    report = {"n_poses": len(main), "conditions": {}}
    p_fisher = {}
    for c in conds:
        entry = {"n": len(by[c]), "reached": int(sum(r["reached"] for r in by[c].values()))}
        if c != MAIN:
            seeds = sorted(set(main) & set(by[c]))
            diffs = [main[s]["final_dist_mm"] - by[c][s]["final_dist_mm"] for s in seeds]
            entry["paired_final_mm_mean"] = float(np.mean(diffs)) if diffs else None
            entry["paired_final_mm_sd"] = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None
            entry["closer_in"] = int(sum(d < 0 for d in diffs))
            entry["n_paired"] = len(diffs)
            entry["sign_test_p"] = sign_test_less(diffs)
            p_fisher[c] = fisher_greater(sum(main[s]["reached"] for s in main), len(main),
                                         entry["reached"], entry["n"])
            entry["fisher_p"] = p_fisher[c]
        report["conditions"][c] = entry
    for c, p in holm(p_fisher).items():
        report["conditions"][c]["fisher_p_holm"] = p
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    for f in a.files:
        rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]
        rep = analyse(rows)
        print(f"== {f.name}: {rep['n_poses']} poses for {MAIN}")
        for c, e in rep["conditions"].items():
            if c == MAIN:
                print(f"{c:18s} reached {e['reached']}/{e['n']}")
            else:
                print(f"{c:18s} reached {e['reached']}/{e['n']}  Fisher p {e['fisher_p']:.3f} (Holm {e['fisher_p_holm']:.3f})  "
                      f"final-dist diff {e['paired_final_mm_mean']:+.1f} ± {e['paired_final_mm_sd'] or 0:.1f} mm, "
                      f"closer in {e['closer_in']}/{e['n_paired']}, sign-test p {e['sign_test_p']:.3f}")


if __name__ == "__main__":
    main()

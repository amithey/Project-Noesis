"""Side decoding versus contrast and integration time, on descending (DN) and projection (PN) neurons.

Calibrated network (all inhibitory weights x ``--g-inh``, graded APL). Blocks of ``--block-s`` seconds present
left/right concentrations (level x left/right ratio, ratios from 0.33 to 3); every 50 ms the causally smoothed
log(1 + rate) of the DN and PN populations is stored at two time constants (250 ms and 1 s) together with the
ratio, level, block and the time elapsed in the block. Analysis uses only windows from the LAST second of each
block, so a 1 s average is not contaminated by the previous block and all feature sets see the same windows.

Evaluation: ridge decoder (strengths fixed in advance), leave-one-concentration-level-out, so the test level was
never in training; block-level label shuffle null (blocks permuted within level); accuracy is broken down by
contrast |ln(ratio)| at window level and at block level (sign of the block-mean score).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from noesis.experiments.decoder_forage import (
    CHUNK_MS,
    LEVELS,
    STEP_MS,
    antenna_rates,
    fit_decoder,
    shuffle_block_labels,
)
from noesis.experiments.virtual_forage import Brain

RATIOS = (0.33, 0.5, 0.67, 0.8, 1.25, 1.5, 2.0, 3.0)
TAUS_S = (0.25, 1.0)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (treats trials as independent)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def loso_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray, lam_per_feature: float) -> np.ndarray:
    """Cross-validated decision scores: each window is scored by a decoder that never saw its group."""
    scores = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        dec = fit_decoder(x[~te], y[~te], lam_per_feature)
        z = (x[te] - dec["mu"]) / dec["sd"]
        scores[te] = z @ dec["w"] + dec["b"]
    return scores


def contrast_table(scores, y, block, ratio) -> list[dict]:
    """Accuracy by |ln ratio|: per window, and per block using the sign of the block-mean score."""
    contrast = np.round(np.abs(np.log(ratio)), 2)
    rows = []
    for c in np.unique(contrast):
        sel = contrast == c
        k, n = int((np.sign(scores[sel]) == y[sel]).sum()), int(sel.sum())
        blocks = np.unique(block[sel])
        bk = sum(int(np.sign(scores[block == b].mean()) == y[block == b][0]) for b in blocks)
        lo, hi = wilson(bk, len(blocks))
        rows.append({"abs_log_ratio": float(c), "window_accuracy": k / n, "n_windows": n,
                     "block_accuracy": bk / len(blocks), "n_blocks": len(blocks),
                     "block_ci95": [lo, hi]})
    return rows


def collect(a, brain: Brain) -> Path:
    n = brain.cn.neurons
    dn = np.flatnonzero(n["Super Class"].eq("descending").to_numpy())
    pn = brain.pops.pn
    rng = np.random.default_rng(21)
    combos = list(itertools.product(range(len(LEVELS)), range(len(RATIOS))))
    orn_l, orn_r = brain.orn["left"], brain.orn["right"]
    drive_idx = np.concatenate([orn_l, orn_r])
    steps_per_block = round(a.block_s * 1000 / STEP_MS)
    keep = {"dn250": [], "dn1000": [], "pn250": [], "pn1000": []}
    meta = []
    block_id = 0
    for run in range(a.runs):
        net, apl = brain.new_net(7000 + run)
        cur = {}
        ema = {t: {"dn": np.zeros(len(dn)), "pn": np.zeros(len(pn))} for t in TAUS_S}
        picks = [combos[i] for i in rng.permutation(len(combos))[:a.blocks]]
        plan = [(0.15, 1.0, None, 0)] * 10
        for li, ri in picks:
            plan += [(LEVELS[li], RATIOS[ri], (block_id, li, ri), s) for s in range(steps_per_block)]
            block_id += 1
        for level, ratio, tag, s in plan:
            r_l, r_r = antenna_rates(level, ratio)
            rates = np.concatenate([np.full(len(orn_l), r_l), np.full(len(orn_r), r_r)])
            counts = np.zeros(net.n)
            for _ in range(round(STEP_MS / CHUNK_MS)):
                _, sp = net.run(CHUNK_MS, drive_idx=drive_idx, drive_rate_hz=rates, drive_weight=a.weight, **cur)
                cur = apl.current(sp, CHUNK_MS)
                counts += np.bincount(sp, minlength=net.n)
            hz = counts / (STEP_MS * 1e-3)
            for t in TAUS_S:
                k = 1 - np.exp(-STEP_MS * 1e-3 / t)
                ema[t]["dn"] += (hz[dn] - ema[t]["dn"]) * k
                ema[t]["pn"] += (hz[pn] - ema[t]["pn"]) * k
            if tag is not None:
                for t, key in ((0.25, "250"), (1.0, "1000")):
                    keep["dn" + key].append(np.log1p(ema[t]["dn"]).astype(np.float32))
                    keep["pn" + key].append(np.log1p(ema[t]["pn"]).astype(np.float32))
                meta.append((1 if RATIOS[tag[2]] > 1 else -1, tag[1], tag[0], RATIOS[tag[2]], (s + 1) * STEP_MS / 1000,
                             run))
        print(f"collected run {run + 1}/{a.runs} ({len(meta)} windows)", flush=True)
    meta = np.array(meta, dtype=float)
    a.windows.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.windows, y=meta[:, 0].astype(int), level=meta[:, 1].astype(int),
                        block=meta[:, 2].astype(int), ratio=meta[:, 3], tin=meta[:, 4], run=meta[:, 5].astype(int),
                        **{k: np.stack(v) for k, v in keep.items()})
    return a.windows


def analyse(path: Path, lams, n_perm: int, min_tin: float) -> dict:
    d = np.load(path)
    sel = d["tin"] >= min_tin - 1e-9
    y, level, block, ratio = d["y"][sel], d["level"][sel], d["block"][sel], d["ratio"][sel]
    report = {"n_windows": int(sel.sum()), "n_blocks": len(np.unique(block)), "min_time_in_block_s": min_tin,
              "sets": {}}
    for name in ("dn250", "dn1000", "pn250", "pn1000"):
        x = d[name][sel]
        entry = {"by_lambda": {}}
        for lam in lams:
            scores = loso_scores(x, y, level, lam)
            acc = float((np.sign(scores) == y).mean())
            entry["by_lambda"][str(lam)] = {"accuracy_unseen_level": acc}
            print(f"{name:7s} lam {lam:g}: accuracy on unseen level {acc:.3f}", flush=True)
        lam0 = lams[len(lams) // 2]
        scores = loso_scores(x, y, level, lam0)
        entry["contrast_table"] = contrast_table(scores, y, block, ratio)
        rng = np.random.default_rng(0)
        null = np.array([(np.sign(loso_scores(x, shuffle_block_labels(y, block, level, int(rng.integers(1 << 30))),
                                              level, lam0)) == y).mean() for _ in range(n_perm)])
        acc0 = entry["by_lambda"][str(lam0)]["accuracy_unseen_level"]
        entry["null"] = {"lam": lam0, "mean": float(null.mean()), "sd": float(null.std()),
                         "p_value": float((1 + np.count_nonzero(null >= acc0 - 1e-12)) / (n_perm + 1))}
        print(f"{name:7s} lam {lam0:g}: null {null.mean():.3f} ± {null.std():.3f}, p={entry['null']['p_value']:.3f}",
              flush=True)
        for r in entry["contrast_table"]:
            print(f"   |ln ratio| {r['abs_log_ratio']:.2f}: window acc {r['window_accuracy']:.2f} "
                  f"(n={r['n_windows']}), block acc {r['block_accuracy']:.2f} of {r['n_blocks']} "
                  f"[{r['block_ci95'][0]:.2f}-{r['block_ci95'][1]:.2f}]", flush=True)
        report["sets"][name] = entry
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/selectivity-cache"))
    ap.add_argument("--windows", type=Path, default=Path("outputs/decoder_contrast_windows.npz"))
    ap.add_argument("--out", type=Path, default=Path("outputs/decoder_contrast.json"))
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--g-inh", type=float, default=12.0)
    ap.add_argument("--apl-gain", type=float, default=1024.0)
    ap.add_argument("--weight", type=float, default=400.0)
    ap.add_argument("--runs", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=7)
    ap.add_argument("--block-s", type=float, default=2.0)
    ap.add_argument("--min-tin", type=float, default=1.0)
    ap.add_argument("--lambdas", default="0.01,0.1,1")
    ap.add_argument("--n-perm", type=int, default=60)
    a = ap.parse_args()
    if not a.analyse_only:
        collect(a, Brain(a.data, a.cache, a.apl_gain, a.g_inh))
    report = analyse(a.windows, [float(x) for x in a.lambdas.split(",")], a.n_perm, a.min_tin)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

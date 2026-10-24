"""Frozen development-seed evaluation of a checkpoint against the original weights, with paired statistics.

Runs ``scripts/frozen_probe.py`` (the locomotion protocol's frozen 5 s quiet job) for every (model, seed), at most
``--workers`` at a time, caches each result under ``--out-dir/<model>-s<seed>/result.json`` (existing results are
reused, so interrupted runs resume), then prints a per-seed table and the pre-registered paired comparison
(``noesis.experiments.paired_stats``). Seeds should be development seeds only; never the held-out ones.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from noesis.experiments.paired_stats import compare

HELD_OUT = {200, 201, 202}


def run_job(a, model: str, seed: int) -> dict | None:
    out = a.out_dir / f"{model}-s{seed}"
    cache = out / "result.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    cmd = [sys.executable, "-u", str(Path(__file__).with_name("frozen_probe.py")), "--data", str(a.data),
           "--output", str(out / "run"), "--seed", str(seed)]
    cmd += a.probe_args.split()
    if model == "candidate":
        cmd += ["--checkpoint", str(a.checkpoint)]
    log = out.with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": "src"}, check=False)
    for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[7:])
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result), encoding="utf-8")
            return result
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, help="candidate weights (required unless --models original)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    ap.add_argument("--models", default="candidate,original")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--original-dir", type=Path, help="reuse cached original results from this directory")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--probe-args", default="", help="extra arguments for frozen_probe, e.g. \"--birth 0.5 --seconds 5.5\"")
    a = ap.parse_args()
    if set(a.seeds) & HELD_OUT:
        ap.error("held-out seeds must not be used for development evaluation")
    models = a.models.split(",")
    if "candidate" in models and a.checkpoint is None:
        ap.error("--checkpoint is required for the candidate")
    if a.original_dir is not None and "original" in models:
        for seed in a.seeds:
            src = a.original_dir / f"original-s{seed}" / "result.json"
            dst = a.out_dir / f"original-s{seed}" / "result.json"
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    jobs = [(m, s) for s in a.seeds for m in models]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = dict(zip(jobs, pool.map(lambda j: run_job(a, *j), jobs), strict=True))
    table = {m: {s: results[(m, s)] for s in a.seeds if results.get((m, s))} for m in models}
    print("seed | " + " | ".join(f"{m} standing/slip/fwd" for m in models))
    for s in a.seeds:
        cells = []
        for m in models:
            r = table[m].get(s)
            cells.append(f"{r['standing_fraction']:.3f}/{r['mean_stance_slip_mm_s']:.1f}/{r['heading_forward_distance_mm']:+.1f}" if r else "missing")
        print(f"{s:4d} | " + " | ".join(cells))
    if {"candidate", "original"} <= set(models):
        comp = compare({s: r["standing_fraction"] for s, r in table["candidate"].items()},
                       {s: r["standing_fraction"] for s, r in table["original"].items()})
        print(json.dumps(comp, indent=1))
        (a.out_dir / "comparison.json").write_text(json.dumps(comp, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

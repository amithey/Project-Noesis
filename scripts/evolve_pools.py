"""Evolve an 84-parameter set of leg muscle-pool gains (separable CMA-ES) for robust frozen standing.

Fitness of an individual = mean standing fraction over ``--seeds-per-gen`` fresh seeds (drawn per generation from a pool
disjoint from the development, validation, held-out and training seeds; common random numbers inside a generation) in frozen
2 s episodes. Each generation also scores the distribution mean and the original weights (zero genome) on the same seeds,
which is the progress curve. Resumable: state and every task result are cached under ``--out``. The candidate to evaluate
afterwards is the distribution mean after the last generation (``mean-final.json``), never the best sample.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from noesis.experiments.evo_worker import init, run_task
from noesis.sep_cma import SepCMA

N_JOINTS = 42
SEED_POOL = range(1000, 2000)       # disjoint from 10-19 (dev), 80-111 (training), 200-202 (held-out)


def generation_seeds(gen: int, k: int) -> list[int]:
    return [int(s) for s in np.random.default_rng(10_000 + gen).choice(list(SEED_POOL), size=k, replace=False)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--seeds-per-gen", type=int, default=3)
    ap.add_argument("--awake-seconds", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.cache or a.out / "cache"
    state_path, log_path, results_path = a.out / "state.json", a.out / "evolution.json", a.out / "results.jsonl"
    done = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["key"]] = r
    n = 2 * N_JOINTS
    es = (SepCMA.from_state(json.loads(state_path.read_text(encoding="utf-8")), a.popsize)
          if state_path.exists() else SepCMA(np.zeros(n), a.sigma, a.popsize, a.seed))
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=init, initargs=(str(a.data), str(cache))) as pool:
        while es.gen < a.generations:
            g, t0 = es.gen, time.time()
            seeds = generation_seeds(g, a.seeds_per_gen)
            xs = es.ask()
            people = {f"g{g}-i{i}": xs[i] for i in range(a.popsize)}
            people[f"g{g}-mean"] = es.m.copy()
            people[f"g{g}-orig"] = None
            tasks = [{"key": f"{k}-s{s}", "seed": s, "genome": None if v is None else v.tolist(),
                      "awake_seconds": a.awake_seconds, "workdir": str(a.out / "work" / f"{k}-s{s}")}
                     for k, v in people.items() for s in seeds if f"{k}-s{s}" not in done]
            for res in pool.imap_unordered(run_task, tasks):
                done[res["key"]] = res
                with results_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(res) + "\n")

            def fit(k: str, seeds=seeds) -> float:
                return float(np.mean([done[f"{k}-s{s}"]["standing_fraction"] for s in seeds]))

            fits = np.array([fit(f"g{g}-i{i}") for i in range(a.popsize)])
            entry = {"gen": g, "seeds": seeds, "fitness": fits.tolist(), "best": float(fits.max()),
                     "pop_mean": float(fits.mean()), "mean_vector_fitness": fit(f"g{g}-mean"),
                     "original_fitness": fit(f"g{g}-orig"), "sigma": es.sigma, "wall_s": time.time() - t0,
                     "failed_tasks": int(sum(not done[f"{k}-s{s}"].get("ok", False) for k in people for s in seeds))}
            (a.out / f"mean-g{g:02d}.json").write_text(json.dumps(es.m.tolist()), encoding="utf-8")
            es.tell(xs, fits)
            state_path.write_text(json.dumps(es.state()), encoding="utf-8")
            log.append(entry)
            log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
            print(f"gen {g}: best {entry['best']:.3f} pop-mean {entry['pop_mean']:.3f} | mean-vector "
                  f"{entry['mean_vector_fitness']:.3f} | original {entry['original_fitness']:.3f} | sigma "
                  f"{entry['sigma']:.3f} | {entry['wall_s'] / 60:.1f} min | failed {entry['failed_tasks']}", flush=True)
    (a.out / "mean-final.json").write_text(json.dumps(es.m.tolist()), encoding="utf-8")


if __name__ == "__main__":
    main()

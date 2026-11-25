"""Train the fly's connectome synapse gains with OpenAI-ES or (sep-)CMA-ES.

Every run writes a self-contained, documented directory:
  config.json        exact settings, source/checkpoint hashes, group table
  generations.jsonl  one line per generation (population stats, best, wall time)
  best-gains.npy     best mean-gain vector so far (by re-evaluated centre score)
  weights-best.npz   MotorPlasticity-compatible checkpoint of that vector
  report.json        final status

Example (Windows PowerShell, from gpu-pilot):
  python es_train.py --method es --stage stand --generations 20 --workers 6 `
     --output outputs/claude-es/R1-es-stand
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path(os.environ.get(
    "NOESIS_PROJECT", r"C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis"))
DEFAULT_CORE = Path(os.environ.get(
    "NOESIS_CORE", r"C:\Users\heyma\.codex\worktrees\brain-body-feedback\Project Noesis\src"))
DEFAULT_PARENT = DEFAULT_PROJECT / "gpu-pilot/outputs/standing-001-train-11000/weights-last.npz"

_EVAL = None


def _paths(core):
    for p in (str(HERE / "src"), str(core)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _init(core, data, cache, checkpoint, grouping, threads):
    global _EVAL
    os.environ.setdefault("NUMBA_NUM_THREADS", str(threads))
    _paths(core)
    from noesis_gpu.es_brain import Evaluator

    _EVAL = Evaluator(data, cache, checkpoint, grouping)


def _job(args):
    gains, seed, seconds, stage = args
    from noesis_gpu.es_brain import STAGE_OBJECTIVE, fitness

    try:
        result = _EVAL.episode(gains, seed, seconds, STAGE_OBJECTIVE[stage])
        result["fitness"] = fitness(result, stage)
    except Exception as error:  # a physics blow-up is a bad candidate, not a crashed run
        result = {"seed": int(seed), "error": f"{type(error).__name__}: {error}", "fitness": -1.0}
    return result


def ranks(values):
    """Centered rank transform in [-0.5, 0.5] (OpenAI-ES fitness shaping)."""
    order = np.argsort(np.argsort(values))
    return order / max(len(values) - 1, 1) - 0.5


class OpenAIES:
    def __init__(self, n, sigma, lr, pairs, rng):
        self.mean, self.sigma, self.lr, self.pairs, self.rng = np.ones(n), sigma, lr, pairs, rng
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0

    def ask(self):
        self.eps = self.rng.standard_normal((self.pairs, len(self.mean)))
        eps = np.concatenate([self.eps, -self.eps])
        return np.clip(self.mean + self.sigma * eps, 0.5, 1.5)

    def tell(self, fit):
        shaped = ranks(np.asarray(fit))
        p = self.pairs
        grad = ((shaped[:p] - shaped[p:]) @ self.eps) / (2 * p * self.sigma)
        self.t += 1  # Adam ascent
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad ** 2
        step = self.lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)
        self.mean = np.clip(self.mean + step, 0.5, 1.5)


class SepCMA:
    """Separable (diagonal) CMA-ES, Ros & Hansen 2008; suited to ~100-300 dims."""

    def __init__(self, n, sigma, popsize, rng):
        self.n, self.rng, self.sigma = n, rng, sigma
        self.lam = popsize
        self.mu = popsize // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1 / (self.w ** 2).sum()
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = 4 / (n + 4)
        c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        cmu = min(1 - c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.c1, self.cmu = c1 * (n + 2) / 3, cmu * (n + 2) / 3  # separable speed-up
        self.chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.mean, self.d = np.ones(n), np.ones(n)
        self.ps, self.pc = np.zeros(n), np.zeros(n)
        self.pairs = None

    def ask(self):
        self.z = self.rng.standard_normal((self.lam, self.n))
        return np.clip(self.mean + self.sigma * self.d * self.z, 0.5, 1.5)

    def tell(self, fit):
        idx = np.argsort(-np.asarray(fit))[: self.mu]
        zw = self.w @ self.z[idx]
        self.mean = np.clip(self.mean + self.sigma * self.d * zw, 0.5, 1.5)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zw
        hs = np.linalg.norm(self.ps) / self.chi < 1.4 + 2 / (self.n + 1)
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * self.d * zw
        y = self.d * self.z[idx]
        c = (1 - self.c1 - self.cmu) * self.d ** 2 + self.c1 * self.pc ** 2 + self.cmu * (self.w @ y ** 2)
        self.d = np.sqrt(np.maximum(c, 1e-12))
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi - 1))
        self.sigma = float(np.clip(self.sigma, 1e-3, 0.5))


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=("es", "cma"), required=True)
    ap.add_argument("--stage", choices=("stand", "walk"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=6, help="ES antithetic pairs (population = 2*pairs)")
    ap.add_argument("--popsize", type=int, default=12, help="CMA population")
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seconds", type=float, default=1.5)
    ap.add_argument("--seeds-per-gen", type=int, default=1)
    ap.add_argument("--train-seed-base", type=int, default=30000)
    ap.add_argument("--grouping", choices=("fine", "coarse"), default="fine")
    ap.add_argument("--init-gains", type=Path, default=None, help="start from a previous best-gains.npy")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--threads-per-worker", type=int, default=1)
    ap.add_argument("--wall-hours", type=float, default=6.0)
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--core", type=Path, default=DEFAULT_CORE)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_PARENT)
    ap.add_argument("--rng-seed", type=int, default=0)
    return ap


def main():
    args = parser().parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = (args.project / "data").resolve()
    cache = (args.project / "gpu-pilot/outputs/claude-es/brain-cache").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    _paths(args.core)
    from noesis_gpu.es_brain import Evaluator, dump
    from noesis_gpu.run import file_digest, source_digest

    started = time.time()
    local = Evaluator(data, cache, args.checkpoint, args.grouping)  # for groups + checkpoint writing
    n = local.groups.n_groups
    rng = np.random.default_rng(args.rng_seed)
    opt = (OpenAIES(n, args.sigma, args.lr, args.pairs, rng) if args.method == "es"
           else SepCMA(n, args.sigma, args.popsize, rng))
    if args.init_gains:
        opt.mean = np.clip(np.load(args.init_gains), 0.5, 1.5)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "method_description": {
                  "es": "OpenAI-ES, antithetic pairs, centred-rank shaping, Adam on group gains",
                  "cma": "separable CMA-ES on group gains"}[args.method],
              "n_groups": n, "plastic_edges": int(len(local.groups.edge_indices)),
              "neurons": int(local.cn.n), "edges": int(local.cn.W.nnz),
              "checkpoint_sha256": file_digest(args.checkpoint),
              "noesis_source_sha256": source_digest(args.core),
              "constraints": "existing edges only; signs fixed; |w| within original +-50%; frozen evaluation",
              "groups": local.groups.describe(), "started_unix": started}
    dump(out / "config.json", config)
    print(json.dumps({"event": "config", "n_groups": n, "plastic_edges": config["plastic_edges"]}), flush=True)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_init,
                    initargs=(args.core, data, cache, args.checkpoint, args.grouping, args.threads_per_worker))
    best, best_gains, status = -np.inf, opt.mean.copy(), "completed"
    log = (out / "generations.jsonl").open("a", encoding="utf-8")
    try:
        for gen in range(args.generations):
            if time.time() - started > args.wall_hours * 3600:
                status = "stopped_wall_budget"
                break
            t0 = time.time()
            cands = opt.ask()
            seeds = [args.train_seed_base + gen * 100 + k for k in range(args.seeds_per_gen)]
            jobs = [(c, s, args.seconds, args.stage) for c in cands for s in seeds]
            jobs += [(opt.mean.copy(), s, args.seconds, args.stage) for s in seeds]  # centre
            results = pool.map(_job, jobs, chunksize=1)
            k = args.seeds_per_gen
            fit = np.array([np.mean([r["fitness"] for r in results[i * k:(i + 1) * k]])
                            for i in range(len(cands))])
            centre = results[-k:]
            centre_fit = float(np.mean([r["fitness"] for r in centre]))
            centre_gains = opt.mean.copy()
            opt.tell(fit)
            improved = centre_fit > best
            if improved:
                best, best_gains = centre_fit, centre_gains
                np.save(out / "best-gains.npy", best_gains)
                sha = local.save_checkpoint(best_gains, out / "weights-best.npz")
            row = {"generation": gen, "seeds": seeds, "pop_fitness_mean": float(fit.mean()),
                   "pop_fitness_max": float(fit.max()), "pop_fitness_min": float(fit.min()),
                   "centre_fitness": centre_fit, "best_centre_fitness": float(best), "improved": bool(improved),
                   "centre_metrics": [{k2: v for k2, v in r.items() if k2 != "touchdowns_per_leg"} | {
                       "touchdowns_per_leg": r.get("touchdowns_per_leg")} for r in centre],
                   "errors": sum("error" in r for r in results),
                   "gain_mean": float(np.mean(opt.mean)), "gain_min": float(np.min(opt.mean)),
                   "gain_max": float(np.max(opt.mean)),
                   "sigma": float(getattr(opt, "sigma", args.sigma)),
                   "wall_seconds": time.time() - t0}
            if improved:
                row["best_checkpoint_sha256"] = sha
            log.write(json.dumps(row) + "\n")
            log.flush()
            np.save(out / "last-mean-gains.npy", opt.mean)
            print(json.dumps({"event": "generation", **{k2: row[k2] for k2 in (
                "generation", "pop_fitness_mean", "pop_fitness_max", "centre_fitness",
                "best_centre_fitness", "errors", "wall_seconds")}}), flush=True)
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        log.close()
        pool.terminate()
        dump(out / "report.json", {"status": status, "best_centre_fitness": float(best) if np.isfinite(best) else None,
                                   "wall_hours": (time.time() - started) / 3600,
                                   "model_promoted": False, "walking_proven": False,
                                   "note": "training score only; acceptance requires es_evaluate.py on held-out seeds"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Minimal separable CMA-ES (diagonal covariance; Ros & Hansen 2008) for noisy maximisation, with resumable state.

Written here because the shared environment has no ``cma`` package and should not be modified. ``ask`` returns a population
around the distribution mean, ``tell`` takes the fitnesses (higher is better) and updates mean, step size, evolution paths and
the diagonal covariance. The distribution mean, not any single sample, is the candidate to evaluate at the end.
"""

from __future__ import annotations

import numpy as np


class SepCMA:
    def __init__(self, mean, sigma: float, popsize: int, seed: int = 0):
        self.m = np.array(mean, float)
        n = self.n = len(self.m)
        if n < 2 or popsize < 4 or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("need n >= 2, popsize >= 4 and a positive finite sigma")
        self.sigma, self.lam = float(sigma), int(popsize)
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        mueff = self.mueff
        self.cs = (mueff + 2) / (n + mueff + 5)
        self.ds = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        self.c1, self.cmu = c1 * (n + 2) / 3, cmu * (n + 2) / 3     # separable-CMA learning-rate boost
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.C = np.ones(n)
        self.ps, self.pc = np.zeros(n), np.zeros(n)
        self.gen = 0
        self.rng = np.random.default_rng(seed)
        self._z = None

    def ask(self) -> np.ndarray:
        self._z = self.rng.standard_normal((self.lam, self.n))
        return self.m + self.sigma * np.sqrt(self.C) * self._z

    def tell(self, xs, fitness) -> None:
        xs, f = np.asarray(xs, float), np.asarray(fitness, float)
        if xs.shape != (self.lam, self.n) or f.shape != (self.lam,) or not np.all(np.isfinite(f)):
            raise ValueError("xs/fitness shape mismatch or non-finite fitness")
        order = np.argsort(-f)[: self.mu]
        y = (xs[order] - self.m) / self.sigma
        yw = self.w @ y
        self.m = self.m + self.sigma * yw
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * yw / np.sqrt(self.C)
        hs = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chi_n
              < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        delta = (1 - hs) * self.cc * (2 - self.cc)
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (self.pc ** 2 + delta * self.C)
                  + self.cmu * (self.w @ (y ** 2)))
        self.C = np.maximum(self.C, 1e-12)
        self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        self.gen += 1

    def state(self) -> dict:
        return {"m": self.m.tolist(), "sigma": self.sigma, "C": self.C.tolist(), "ps": self.ps.tolist(),
                "pc": self.pc.tolist(), "gen": self.gen, "rng": self.rng.bit_generator.state}

    @classmethod
    def from_state(cls, state: dict, popsize: int) -> SepCMA:
        obj = cls(state["m"], state["sigma"], popsize)
        obj.C, obj.ps, obj.pc = (np.array(state[k]) for k in ("C", "ps", "pc"))
        obj.gen = state["gen"]
        obj.rng.bit_generator.state = state["rng"]
        return obj

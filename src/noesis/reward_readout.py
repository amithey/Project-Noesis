"""A linear read-out of projection-neuron activity trained by reward only (no side labels), on recorded brain responses.

The agent is the leg-free point agent of ``noesis.virtual_agent``. Its steering score is ``clip(w . z + noise, -1, 1)`` where ``z`` is
the standardised log(1 + 250 ms average rate) vector of the projection neurons. Training uses a SURROGATE of the brain in the loop:
each control step the two antenna concentrations are converted to an effective level and left/right ratio, and ``z`` is drawn from
a library of PN feature vectors that were RECORDED from the real network under constant inputs of that level and ratio
(`FeatureLibrary`). Weights are learned with REINFORCE from a scalar episode return that is multiplied by a hunger factor.
Nothing here uses which side was stronger as a label. The learned weights are then tested with the real network in the loop
(`experiments/reward_readout_run.py`); the surrogate lacks the network's temporal dynamics, so training results are not evidence by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np

from noesis.virtual_agent import AgentParams, SteeringAgent, plume

STEP_S = 0.05
FOOD = (0.0, 0.0)


class FeatureLibrary:
    """Recorded PN feature vectors grouped by block (constant level and left/right ratio); nearest-block sampling in log space."""

    def __init__(self, x: np.ndarray, level: np.ndarray, ratio: np.ndarray, block: np.ndarray, floor_sd: float = 1e-6,
                 n_components: int | None = None):
        x = np.asarray(x, np.float64)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("x must be a non-empty 2-D array")
        self.mu = x.mean(axis=0)
        self.sd = np.where(x.std(axis=0) < floor_sd, 1.0, x.std(axis=0))
        z = (x - self.mu) / self.sd
        # Optional unsupervised dimensionality reduction (no labels, no reward): top principal components of the recorded
        # features, each scaled to unit variance; the read-out then has ``n_components`` weights instead of one per neuron.
        self.basis = None
        if n_components is not None:
            _, sv, vt = np.linalg.svd(z, full_matrices=False)
            k = min(int(n_components), len(sv))
            scale = sv[:k] / np.sqrt(len(z))
            self.basis = (vt[:k].T / scale)                      # (d, k): z_pc = z @ basis has unit variance per component
            z = z @ self.basis
        self.z = z.astype(np.float32)
        ids = np.unique(block)
        self.rows = [np.flatnonzero(block == b) for b in ids]
        self.ln_level = np.array([np.log(level[r[0]]) for r in self.rows])
        self.ln_ratio = np.array([np.log(ratio[r[0]]) for r in self.rows])
        if not np.all(np.isfinite(self.ln_level)) or not np.all(np.isfinite(self.ln_ratio)):
            raise ValueError("levels and ratios must be positive and finite")
        self.n_features = self.z.shape[1]

    def full_weights(self, w: np.ndarray) -> np.ndarray:
        """Weights on the standardised neuron features equivalent to weights ``w`` on the library's own feature space."""
        return w if self.basis is None else self.basis @ w

    def nearest_block(self, level: float, ratio: float) -> int:
        ll = np.clip(np.log(max(level, 1e-9)), self.ln_level.min(), self.ln_level.max())
        lr = np.clip(np.log(max(ratio, 1e-9)), self.ln_ratio.min(), self.ln_ratio.max())
        return int(np.argmin((self.ln_level - ll) ** 2 + (self.ln_ratio - lr) ** 2))

    def sample(self, level: float, ratio: float, rng: np.random.Generator) -> np.ndarray:
        rows = self.rows[self.nearest_block(level, ratio)]
        return self.z[rows[rng.integers(len(rows))]]


@dataclass(frozen=True)
class EpisodeSpec:
    duration_s: float = 12.0
    start_mm: float = 20.0
    sigma_mm: float = 10.0
    gain: float = 1.0            # receptor gain from hunger (rate ~ gain * sqrt(c))


def random_pose(rng: np.random.Generator, start_mm: float):
    angle = rng.uniform(0, 2 * pi)
    start = (start_mm * cos(angle), start_mm * sin(angle))
    heading = atan2(FOOD[1] - start[1], FOOD[0] - start[0]) + rng.uniform(-pi, pi)
    return start, heading


def episode_return(d0: float, d_min: float, t_reach: float | None, duration_s: float) -> float:
    """Shaped scalar return in [0, 2.5]: approach fraction + 1 for reaching the food + a bonus for reaching it early."""
    approach = max(0.0, (d0 - d_min) / d0)
    if t_reach is None:
        return approach
    return approach + 1.0 + 0.5 * (1.0 - t_reach / duration_s)


def run_episode(lib: FeatureLibrary, w: np.ndarray, b: float, spec: EpisodeSpec, params: AgentParams,
                rng: np.random.Generator, noise_sigma: float = 0.0, steering_sign: float = 1.0,
                no_gradient: bool = False, trace: bool = False) -> dict:
    """One surrogate episode. With ``noise_sigma`` > 0 the score gets Gaussian exploration noise and the REINFORCE statistic is returned."""
    start, heading = random_pose(rng, spec.start_mm)
    agent = SteeringAgent(start, heading, params, seed=int(rng.integers(1 << 31)), sign=steering_sign)
    d0 = agent.distance_to(FOOD)
    d_min, t_reach, steps = d0, None, 0
    stat = np.zeros(lib.n_features)
    g2 = spec.gain ** 2
    for k in range(round(spec.duration_s / STEP_S)):
        lp, rp = agent.antenna_positions()
        if no_gradient:
            c_l = c_r = plume(agent.x, agent.y, FOOD, spec.sigma_mm)
        else:
            c_l, c_r = plume(*lp, FOOD, spec.sigma_mm), plume(*rp, FOOD, spec.sigma_mm)
        c_l, c_r = max(c_l, 1e-9), max(c_r, 1e-9)
        z = lib.sample(np.sqrt(c_l * c_r) * g2, c_l / c_r, rng)
        mean = float(w @ z + b)
        eps = rng.normal(0.0, noise_sigma) if noise_sigma > 0 else 0.0
        agent.step_with_signal(STEP_S, float(np.clip(mean + eps, -1.0, 1.0)))
        stat += eps * z
        steps += 1
        d = agent.distance_to(FOOD)
        d_min = min(d_min, d)
        if d <= params.food_radius_mm:
            t_reach = (k + 1) * STEP_S
            break
        if d > params.arena_radius_mm:
            break
    out = {"d0": d0, "d_min": d_min, "final": agent.distance_to(FOOD), "t_reach": t_reach, "reached": t_reach is not None,
           "stat": stat / max(steps, 1), "steps": steps, "ret": episode_return(d0, d_min, t_reach, spec.duration_s)}
    if trace:
        out["path"] = np.array(agent.path)
    return out


def evaluate(lib, w, b, spec, params, n: int, seed: int, **kw) -> dict:
    rng = np.random.default_rng(seed)
    res = [run_episode(lib, w, b, spec, params, rng, 0.0, **kw) for _ in range(n)]
    return {"success": float(np.mean([r["reached"] for r in res])), "mean_return": float(np.mean([r["ret"] for r in res])),
            "median_final_mm": float(np.median([r["final"] for r in res]))}


def train(lib: FeatureLibrary, spec: EpisodeSpec, params: AgentParams, *, iters: int, batch: int, lr: float, sigma: float,
          reward_multiplier: float = 1.0, shuffle_reward: bool = False, seed: int = 0, l2: float = 1e-4,
          momentum: float = 0.9, eval_every: int = 0, eval_n: int = 100) -> dict:
    """REINFORCE with a batch-mean baseline, SGD with momentum. The hunger factor multiplies the return (not the learning rate)."""
    rng = np.random.default_rng(seed)
    w, vel, b = np.zeros(lib.n_features), np.zeros(lib.n_features), 0.0
    curve = []
    for it in range(iters):
        eps_stats, rets = [], []
        for _ in range(batch):
            r = run_episode(lib, w, b, spec, params, rng, sigma)
            eps_stats.append(r["stat"])
            rets.append(r["ret"])
        rets = np.array(rets) * reward_multiplier
        if shuffle_reward:
            rets = rng.permutation(rets)              # contingency broken: each episode gets another episode's return
        adv = rets - rets.mean()
        grad = (adv[:, None] * np.array(eps_stats)).mean(axis=0) / (sigma ** 2) - l2 * w
        vel = momentum * vel + lr * grad
        w = w + vel
        if eval_every and (it + 1) % eval_every == 0:
            curve.append({"iter": it + 1, **evaluate(lib, w, b, spec, params, eval_n, seed=900_000 + it)})
    return {"w": w, "b": b, "curve": curve}

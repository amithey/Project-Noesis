"""Leg-free virtual agent steered by the DNa left/right balance of the real connectome network.

Each 50 ms step: antenna concentrations (Gaussian plume) set graded firing of the left/right food-odour
receptor neurons (scaled by the hunger gain); the corrected brain with graded APL runs; the firing of the
left- and right-soma DNa (numbered DNa family) neurons gives a steering signal that turns the agent.
Claims are limited to ODOUR-GRADIENT SIDE STEERING: no odour identification, no learning. The mapping
"more left DNa activity turns left" is an assumption (reversed-sign control included). A per-episode
1 s calibration with symmetric input estimates the network's intrinsic left/right bias, which is subtracted.

Conditions (same start poses per seed):
  brain_hungry / brain_fed   DNa steering with hunger gain from energy 0.3 / 1.0
  shuffled_sides             receptor left/right labels randomly permuted (hungry)
  reversed_sign              steering sign flipped (hungry): should repel
  random_dn                  16+16 random non-DNa descending neurons as the steering groups (hungry)
  no_gradient                both antennae sample the body-centre concentration (hungry)
  diffusion_only             no steering input, only rotational diffusion
"""

from __future__ import annotations

import argparse
import json
import re
from math import atan2, cos, pi, sin
from pathlib import Path

import numpy as np

from noesis.connectome import load_all_pairs
from noesis.drives import FOOD_CHANNELS, HomeostaticState
from noesis.embodiment import load_training_brain
from noesis.experiments.ei_balance_scan import scale_inhibition
from noesis.experiments.sense_audit import receptor_sides
from noesis.lif import LIFNetwork, LIFParams
from noesis.mushroom_body import GradedAPL, populations, with_weak_pathway_edges
from noesis.virtual_agent import AgentParams, SteeringAgent, path_efficiency, plume, steering_signal

CHUNK_MS = 5.0
CONDITIONS = {
    "brain_hungry": {"energy": 0.3},
    "brain_fed": {"energy": 1.0},
    "shuffled_sides": {"energy": 0.3, "shuffle": True},
    "reversed_sign": {"energy": 0.3, "sign": -1.0},
    "random_dn": {"energy": 0.3, "random_dn": True},
    "no_gradient": {"energy": 0.3, "no_gradient": True},
    "diffusion_only": {"energy": 0.3, "steer": False},
}
START_OFFSETS_DEG = (90.0, -90.0, 150.0, -150.0, 120.0, -120.0)   # heading relative to the food bearing


def summarise(rows: list[dict]) -> dict:
    """Per-condition aggregates from episode result dicts."""
    out = {}
    for name in sorted({r["condition"] for r in rows}):
        sel = [r for r in rows if r["condition"] == name]
        out[name] = {"n": len(sel), "reached": int(sum(r["reached"] for r in sel)),
                     "median_final_mm": float(np.median([r["final_dist_mm"] for r in sel])),
                     "mean_min_mm": float(np.mean([r["min_dist_mm"] for r in sel])),
                     "median_efficiency": float(np.median([r["efficiency"] for r in sel]))}
    return out


def paired_diff(rows: list[dict], a: str, b: str, key: str = "final_dist_mm") -> dict:
    """Mean/sd of (a - b) over seeds present in both conditions; negative = a ended closer to the food."""
    by = {(r["condition"], r["seed"]): r[key] for r in rows}
    diffs = [by[(a, s)] - by[(b, s)] for s in sorted({r["seed"] for r in rows})
             if (a, s) in by and (b, s) in by]
    return {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs else None,
            "sd": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
            "negative_in": int(sum(d < 0 for d in diffs))}


class Brain:
    """Wiring of the shared network to receptor groups and steering groups."""

    def __init__(self, data: Path, cache: Path, apl_gain: float, g_inh: float = 1.0):
        cn, _ = load_training_brain(data, cache, "corrected")
        self.cn, self.apl_gain = cn, apl_gain
        self.pops = populations(cn.neurons)
        self.W = scale_inhibition(with_weak_pathway_edges(
            cn, load_all_pairs(data / "raw", data / "processed", cn.neurons), self.pops), g_inh)
        n = cn.neurons
        function = n["Function"].fillna("")
        orn = n["Flow"].eq("afferent") & n["Class"].eq("olfactory_receptor_neuron")
        food = np.zeros(len(n), bool)
        for ch in FOOD_CHANNELS:
            food |= function.str.contains(rf"(?:^|,){ch}(?:,|$)", regex=True).to_numpy()
        side = receptor_sides(n).to_numpy()
        self.orn = {s: np.flatnonzero(orn.to_numpy() & food & (side == s)) for s in ("left", "right")}
        ctype = n["Primary Cell Type"].fillna("").to_numpy()
        dn = n["Super Class"].eq("descending").to_numpy()
        soma = n["Soma side"].to_numpy()
        is_dna = np.array([bool(re.match(r"^DNa\d\d$", t)) for t in ctype])
        self.dna = {s: np.flatnonzero(dn & is_dna & (soma == s)) for s in ("left", "right")}
        rng = np.random.default_rng(0)
        pool = {s: np.flatnonzero(dn & ~is_dna & (soma == s)) for s in ("left", "right")}
        self.random_dn = {s: rng.choice(pool[s], size=len(self.dna[s]), replace=False) for s in pool}

    def new_net(self, seed: int):
        net = LIFNetwork(self.W, LIFParams(dt=0.5, da=1.0), seed=seed)
        return net, GradedAPL(net, self.pops, gain=self.apl_gain)


def run_episode(brain: Brain, name: str, cond: dict, seed: int, a, params: AgentParams) -> dict:
    rng = np.random.default_rng(seed)
    food = (0.0, 0.0)
    offset = START_OFFSETS_DEG[seed % len(START_OFFSETS_DEG)] * pi / 180
    angle = rng.uniform(0, 2 * pi)
    start = (a.start_mm * cos(angle), a.start_mm * sin(angle))
    heading = atan2(food[1] - start[1], food[0] - start[0]) + offset
    state = HomeostaticState(energy=cond["energy"])
    gain = state.sensory_gain("fruity_volatile")
    orn_l, orn_r = brain.orn["left"], brain.orn["right"]
    if cond.get("shuffle"):
        mixed = rng.permutation(np.concatenate([orn_l, orn_r]))
        orn_l, orn_r = mixed[:len(orn_l)], mixed[len(orn_l):]
    drive_idx = np.concatenate([orn_l, orn_r])
    groups = brain.random_dn if cond.get("random_dn") else brain.dna
    net, apl = brain.new_net(seed + 7000)
    cur = {}

    def run_for(ms, c_left, c_right, count_groups=True):
        nonlocal cur
        rates = np.concatenate([np.full(len(orn_l), min(a.rate_max * np.sqrt(c_left) * gain, 200.0)),
                                np.full(len(orn_r), min(a.rate_max * np.sqrt(c_right) * gain, 200.0))])
        counts = np.zeros(net.n)
        for _ in range(round(ms / CHUNK_MS)):
            _, sp = net.run(CHUNK_MS, drive_idx=drive_idx, drive_rate_hz=rates, drive_weight=a.weight, **cur)
            cur = apl.current(sp, CHUNK_MS)
            if count_groups:
                counts += np.bincount(sp, minlength=net.n)
        sec = ms * 1e-3
        return (counts[groups["left"]].sum() / max(len(groups["left"]), 1) / sec,
                counts[groups["right"]].sum() / max(len(groups["right"]), 1) / sec)

    steer = cond.get("steer", True)
    bias = 0.0
    if steer:   # calibration: symmetric input at the start concentration
        c0 = plume(*start, food, a.sigma_mm)
        run_for(400.0, c0, c0, count_groups=False)
        bias = steering_signal(*run_for(600.0, c0, c0))
    agent = SteeringAgent(start, heading, params, seed=seed, bias=bias, sign=cond.get("sign", 1.0))
    dt = a.step_ms * 1e-3
    t_reach, raw_signals, dists = None, [], [agent.distance_to(food)]
    for k in range(round(a.duration_s / dt)):
        if steer:
            left_pos, right_pos = agent.antenna_positions()
            if cond.get("no_gradient"):
                c_l = c_r = plume(agent.x, agent.y, food, a.sigma_mm)
            else:
                c_l, c_r = plume(*left_pos, food, a.sigma_mm), plume(*right_pos, food, a.sigma_mm)
            hz_l, hz_r = run_for(a.step_ms, c_l, c_r)
            raw_signals.append(steering_signal(hz_l, hz_r) - bias)
            agent.step(dt, hz_l, hz_r)
        else:
            agent.step(dt, None, None)
        dists.append(agent.distance_to(food))
        if dists[-1] <= params.food_radius_mm:
            t_reach = (k + 1) * dt
            break
        if dists[-1] > params.arena_radius_mm:
            break
    return {"condition": name, "seed": seed, "reached": t_reach is not None, "t_reach_s": t_reach,
            "final_dist_mm": float(dists[-1]), "min_dist_mm": float(min(dists)), "start_dist_mm": float(dists[0]),
            "efficiency": path_efficiency(agent.path), "bias": float(bias),
            "signal_mean": float(np.mean(raw_signals)) if raw_signals else 0.0,
            "signal_sd": float(np.std(raw_signals)) if raw_signals else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/selectivity-cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/virtual_forage.jsonl"))
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--step-ms", type=float, default=50.0)
    ap.add_argument("--start-mm", type=float, default=20.0)
    ap.add_argument("--sigma-mm", type=float, default=10.0)
    ap.add_argument("--rate-max", type=float, default=100.0)
    ap.add_argument("--weight", type=float, default=400.0)
    ap.add_argument("--apl-gain", type=float, default=1024.0)
    ap.add_argument("--gain-rad-s", type=float, default=10.0, help="chosen on held-out pilot seed 99")
    a = ap.parse_args()

    params = AgentParams(gain_rad_s=a.gain_rad_s)
    brain = Brain(a.data, a.cache, a.apl_gain)
    print(f"receptors L/R {len(brain.orn['left'])}/{len(brain.orn['right'])}; DNa L/R "
          f"{len(brain.dna['left'])}/{len(brain.dna['right'])}", flush=True)
    done = set()
    rows = []
    if a.out.exists():
        rows = [json.loads(line) for line in a.out.read_text(encoding="utf-8").splitlines() if line]
        done = {(r["condition"], r["seed"]) for r in rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in a.seeds.split(",")):
        for name in a.conditions.split(","):
            if (name, seed) in done:
                continue
            r = run_episode(brain, name, CONDITIONS[name], seed, a, params)
            rows.append(r)
            with a.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"seed {seed} {name:15s} reached={r['reached']!s:5} final {r['final_dist_mm']:5.1f} mm "
                  f"(start {r['start_dist_mm']:.1f}, min {r['min_dist_mm']:.1f}) eff {r['efficiency']:.2f} "
                  f"bias {r['bias']:+.3f} signal {r['signal_mean']:+.3f}±{r['signal_sd']:.3f}", flush=True)
    summary = summarise(rows)
    print(json.dumps(summary, indent=1))
    for control in ("shuffled_sides", "reversed_sign", "random_dn", "no_gradient", "diffusion_only", "brain_fed"):
        if any(r["condition"] == control for r in rows):
            print(f"brain_hungry - {control}: final distance {json.dumps(paired_diff(rows, 'brain_hungry', control))}")


if __name__ == "__main__":
    main()

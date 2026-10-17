"""Leg-free virtual agent steered by a LEARNED linear read-out of descending-neuron rates (calibrated network).

Stage 'collect': on the calibrated network (all inhibitory weights x ``--g-inh``, graded APL) present blocks of
1 s with left/right antenna concentrations (level x left/right ratio); record causally smoothed (250 ms)
descending-neuron rates every 50 ms with the true side as label. Stage 'cv': leave-one-concentration-level-out
accuracy with a block-label permutation null. Stage 'agent': fit the ridge decoder on all training windows
(plus a decoder fitted on block-shuffled labels as a control) and steer the agent with clip(score, -1, 1).

Claims are limited to: the network carries side information that a trained linear read-out can use in real
time. The decoder is supervised on side labels, so this is a trained interface to the network; no odour
identification and no learning inside the network is claimed. 'Left activity turns left' is a mapping
assumption through the label definition itself (left-stronger input = +1 = turn left); a reversed-sign
control is included.
"""

from __future__ import annotations

import argparse
import itertools
import json
from math import atan2, cos, pi, sin
from pathlib import Path

import numpy as np

from noesis.drives import HomeostaticState
from noesis.experiments.dn_decoder import loso_accuracy
from noesis.experiments.virtual_forage import START_OFFSETS_DEG, Brain, paired_diff, summarise
from noesis.virtual_agent import AgentParams, SteeringAgent, path_efficiency, plume

CHUNK_MS = 5.0
LEVELS = (0.02, 0.05, 0.15, 0.4, 1.0, 2.9)       # effective concentration (2.9 ~ hungry gain 1.7 at c = 1)
RATIOS = (0.67, 0.8, 0.9, 1.11, 1.25, 1.5)       # left / right concentration; >1 = left stronger
STEP_MS = 50.0
TAU_FEATURE_S = 0.25
CONDITIONS = {
    "decoder_hungry": {"energy": 0.3},
    "decoder_fed": {"energy": 1.0},
    "shuffled_decoder": {"energy": 0.3, "decoder": "shuffled"},
    "reversed_sign": {"energy": 0.3, "sign": -1.0},
    "shuffled_sides": {"energy": 0.3, "shuffle": True},
    "no_gradient": {"energy": 0.3, "no_gradient": True},
    "diffusion_only": {"energy": 0.3, "steer": False},
}


def antenna_rates(level: float, ratio: float, rate_max: float = 100.0, gain: float = 1.0) -> tuple[float, float]:
    """Left/right receptor firing (Hz, capped at 200) for effective concentration ``level`` and left/right ``ratio``."""
    if level < 0 or ratio <= 0:
        raise ValueError("level must be >= 0 and ratio > 0")
    c_l, c_r = level * ratio ** 0.5, level / ratio ** 0.5
    return (min(rate_max * gain * c_l ** 0.5, 200.0), min(rate_max * gain * c_r ** 0.5, 200.0))


def fit_decoder(x: np.ndarray, y: np.ndarray, lam_per_feature: float) -> dict:
    """Dual-form ridge on standardised features; returns weights usable on new rows."""
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd
    lam = lam_per_feature * z.shape[1]
    alpha = np.linalg.solve(z @ z.T + lam * np.eye(len(z)), y - y.mean())
    return {"mu": mu, "sd": sd, "w": z.T @ alpha, "b": float(y.mean())}


def decode(dec: dict, feature: np.ndarray) -> float:
    return float(((feature - dec["mu"]) / dec["sd"]) @ dec["w"] + dec["b"])


def shuffle_block_labels(y: np.ndarray, block: np.ndarray, level: np.ndarray, seed: int) -> np.ndarray:
    """Permute block-level labels among blocks of the same concentration level (keeps windows within a block together)."""
    rng = np.random.default_rng(seed)
    out = y.copy()
    for lv in np.unique(level):
        blocks = np.unique(block[level == lv])
        labels = np.array([y[block == b][0] for b in blocks])
        for b, lab in zip(blocks, rng.permutation(labels), strict=True):
            out[block == b] = lab
    return out


def collect(a, brain: Brain) -> Path:
    dn = np.flatnonzero(brain.cn.neurons["Super Class"].eq("descending").to_numpy())
    rng = np.random.default_rng(11)
    combos = list(itertools.product(range(len(LEVELS)), range(len(RATIOS))))
    orn_l, orn_r = brain.orn["left"], brain.orn["right"]
    drive_idx = np.concatenate([orn_l, orn_r])
    feats, labels, levels, blocks, runs = [], [], [], [], []
    block_id = 0
    for run in range(a.runs):
        net, apl = brain.new_net(5000 + run)
        cur, ema = {}, np.zeros(len(dn))
        picks = [combos[i] for i in rng.permutation(len(combos))[:a.blocks]]
        plan = [(0.15, 1.0, None)] * 10
        for li, ri in picks:
            plan += [(LEVELS[li], RATIOS[ri], (block_id, li, ri))] * 20
            block_id += 1
        for level, ratio, tag in plan:
            r_l, r_r = antenna_rates(level, ratio)
            rates = np.concatenate([np.full(len(orn_l), r_l), np.full(len(orn_r), r_r)])
            counts = np.zeros(net.n)
            for _ in range(round(STEP_MS / CHUNK_MS)):
                _, sp = net.run(CHUNK_MS, drive_idx=drive_idx, drive_rate_hz=rates, drive_weight=a.weight, **cur)
                cur = apl.current(sp, CHUNK_MS)
                counts += np.bincount(sp, minlength=net.n)
            ema += (counts[dn] / (STEP_MS * 1e-3) - ema) * (1 - np.exp(-STEP_MS * 1e-3 / TAU_FEATURE_S))
            if tag is not None:
                feats.append(np.log1p(ema).astype(np.float32))
                labels.append(1 if RATIOS[tag[2]] > 1 else -1)
                levels.append(tag[1])
                blocks.append(tag[0])
                runs.append(run)
        print(f"collected run {run + 1}/{a.runs} ({len(feats)} windows)", flush=True)
    a.windows.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.windows, x=np.stack(feats), y=np.array(labels), level=np.array(levels),
                        block=np.array(blocks), run=np.array(runs), dn=dn)
    return a.windows


def cross_validate(path: Path, lams, n_perm: int) -> dict:
    d = np.load(path)
    x, y, level, block = d["x"], d["y"], d["level"], d["block"]
    out = {"n_windows": len(y), "n_blocks": len(np.unique(block)), "by_lambda": {}}
    for lam in lams:
        acc = loso_accuracy(x, y, level, lam)
        rng = np.random.default_rng(0)
        null = np.array([loso_accuracy(x, shuffle_block_labels(y, block, level, int(rng.integers(1 << 30))),
                                       level, lam)["accuracy"] for _ in range(n_perm)])
        p = float((1 + np.count_nonzero(null >= acc["accuracy"] - 1e-12)) / (n_perm + 1))
        out["by_lambda"][str(lam)] = {"accuracy_unseen_level": acc["accuracy"], "per_level": acc["per_set"],
                                      "null_mean": float(null.mean()), "null_sd": float(null.std()), "p_value": p}
        print(f"decoder lam {lam:g}/feat: accuracy on unseen concentration level {acc['accuracy']:.3f} "
              f"(block-shuffle null {null.mean():.3f} ± {null.std():.3f}) p={p:.4f} per level "
              f"{ {LEVELS[k]: round(v, 2) for k, v in acc['per_set'].items()} }", flush=True)
    return out


def run_episode(brain: Brain, name: str, cond: dict, seed: int, a, params: AgentParams, decoders: dict) -> dict:
    rng = np.random.default_rng(seed)
    food = (0.0, 0.0)
    offset = START_OFFSETS_DEG[seed % len(START_OFFSETS_DEG)] * pi / 180
    angle = rng.uniform(0, 2 * pi)
    start = (a.start_mm * cos(angle), a.start_mm * sin(angle))
    heading = atan2(food[1] - start[1], food[0] - start[0]) + offset
    gain = HomeostaticState(energy=cond["energy"]).sensory_gain("fruity_volatile")
    orn_l, orn_r = brain.orn["left"], brain.orn["right"]
    if cond.get("shuffle"):
        mixed = rng.permutation(np.concatenate([orn_l, orn_r]))
        orn_l, orn_r = mixed[:len(orn_l)], mixed[len(orn_l):]
    drive_idx = np.concatenate([orn_l, orn_r])
    dn = decoders["idx"]
    dec = decoders[cond.get("decoder", "real")]
    net, apl = brain.new_net(seed + 9000)
    cur, ema = {}, np.zeros(len(dn))
    steer = cond.get("steer", True)

    def step_brain(c_left, c_right):
        nonlocal cur, ema
        rates = np.concatenate([np.full(len(orn_l), min(a.rate_max * gain * c_left ** 0.5, 200.0)),
                                np.full(len(orn_r), min(a.rate_max * gain * c_right ** 0.5, 200.0))])
        counts = np.zeros(net.n)
        for _ in range(round(STEP_MS / CHUNK_MS)):
            _, sp = net.run(CHUNK_MS, drive_idx=drive_idx, drive_rate_hz=rates, drive_weight=a.weight, **cur)
            cur = apl.current(sp, CHUNK_MS)
            counts += np.bincount(sp, minlength=net.n)
        ema += (counts[dn] / (STEP_MS * 1e-3) - ema) * (1 - np.exp(-STEP_MS * 1e-3 / TAU_FEATURE_S))
        return decode(dec, np.log1p(ema))

    if steer:
        c0 = plume(*start, food, a.sigma_mm)
        for _ in range(10):
            step_brain(c0, c0)
    agent = SteeringAgent(start, heading, params, seed=seed, sign=cond.get("sign", 1.0))
    dt = STEP_MS * 1e-3
    t_reach, dists, scores = None, [agent.distance_to(food)], []
    for k in range(round(a.duration_s / dt)):
        if steer:
            lp, rp = agent.antenna_positions()
            if cond.get("no_gradient"):
                c_l = c_r = plume(agent.x, agent.y, food, a.sigma_mm)
            else:
                c_l, c_r = plume(*lp, food, a.sigma_mm), plume(*rp, food, a.sigma_mm)
            score = float(np.clip(step_brain(c_l, c_r), -1.0, 1.0))
            scores.append(score)
            agent.step_with_signal(dt, score)
        else:
            agent.step_with_signal(dt, None)
        dists.append(agent.distance_to(food))
        if dists[-1] <= params.food_radius_mm:
            t_reach = (k + 1) * dt
            break
        if dists[-1] > params.arena_radius_mm:
            break
    return {"condition": name, "seed": seed, "reached": t_reach is not None, "t_reach_s": t_reach,
            "final_dist_mm": float(dists[-1]), "min_dist_mm": float(min(dists)), "start_dist_mm": float(dists[0]),
            "efficiency": path_efficiency(agent.path),
            "score_mean": float(np.mean(scores)) if scores else 0.0,
            "score_sd": float(np.std(scores)) if scores else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/selectivity-cache"))
    ap.add_argument("--stage", choices=("collect", "cv", "agent"), required=True)
    ap.add_argument("--windows", type=Path, default=Path("outputs/decoder_windows.npz"))
    ap.add_argument("--out", type=Path, default=Path("outputs/decoder_forage.jsonl"))
    ap.add_argument("--g-inh", type=float, default=12.0)
    ap.add_argument("--apl-gain", type=float, default=1024.0)
    ap.add_argument("--weight", type=float, default=400.0)
    ap.add_argument("--rate-max", type=float, default=100.0)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=11)
    ap.add_argument("--lambdas", default="0.01,0.1,1")
    ap.add_argument("--lam", type=float, default=0.1, help="ridge strength used by the agent (fixed in advance)")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--start-mm", type=float, default=20.0)
    ap.add_argument("--sigma-mm", type=float, default=10.0)
    ap.add_argument("--gain-rad-s", type=float, default=1.5)
    ap.add_argument("--readout", choices=("dn", "pn"), default="dn")
    ap.add_argument("--antenna-mm", type=float, default=1.0)
    a = ap.parse_args()

    if a.stage == "cv":
        cross_validate(a.windows, [float(x) for x in a.lambdas.split(",")], a.n_perm)
        return
    brain = Brain(a.data, a.cache, a.apl_gain, a.g_inh)
    if a.stage == "collect":
        collect(a, brain)
        return
    d = np.load(a.windows)
    if a.readout == "pn":     # windows from decoder_contrast.py: PN features, 250 ms average, last second of blocks
        keep = d["tin"] >= 1.0 - 1e-9
        x, y, level, block = d["pn250"][keep], d["y"][keep], d["level"][keep], d["block"][keep]
        idx = brain.pops.pn
    else:
        x, y, level, block, idx = d["x"], d["y"], d["level"], d["block"], d["dn"]
    decoders = {"idx": idx, "real": fit_decoder(x, y, a.lam),
                "shuffled": fit_decoder(x, shuffle_block_labels(y, block, level, 123), a.lam)}
    params = AgentParams(gain_rad_s=a.gain_rad_s, smooth_s=0.5, antenna_mm=a.antenna_mm)
    rows, done = [], set()
    if a.out.exists():
        rows = [json.loads(line) for line in a.out.read_text(encoding="utf-8").splitlines() if line]
        done = {(r["condition"], r["seed"]) for r in rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in (int(s) for s in a.seeds.split(",")):
        for name in a.conditions.split(","):
            if (name, seed) in done:
                continue
            r = run_episode(brain, name, CONDITIONS[name], seed, a, params, decoders)
            rows.append(r)
            with a.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"seed {seed} {name:16s} reached={r['reached']!s:5} final {r['final_dist_mm']:5.1f} mm "
                  f"(start {r['start_dist_mm']:.1f}, min {r['min_dist_mm']:.1f}) eff {r['efficiency']:.2f} "
                  f"score {r['score_mean']:+.2f}±{r['score_sd']:.2f}", flush=True)
    print(json.dumps(summarise(rows), indent=1))
    for control in CONDITIONS:
        if control != "decoder_hungry" and any(r["condition"] == control for r in rows):
            print(f"decoder_hungry - {control}: final distance {json.dumps(paired_diff(rows, 'decoder_hungry', control))}")


if __name__ == "__main__":
    main()

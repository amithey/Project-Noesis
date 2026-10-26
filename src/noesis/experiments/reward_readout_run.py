"""Reward-only training of the PN read-out, then a closed-loop test with the real network.

Stages:
  collect-near1  record extra PN feature blocks at left/right ratios 0.9, 1.0, 1.11 (the existing library has none near 1)
  train          REINFORCE on the surrogate (recorded PN features) for hungry / fed / shuffled-reward; random read-out control
  eval           the learned weights, frozen, steer the agent with the REAL network in the loop; same poses, fixed conditions
Conditions at evaluation (hungry unless noted): reward_hungry, reward_fed (fed receptor gain), shuffled_reward, random_readout,
reversed_sign (reward_hungry weights, sign flipped), no_gradient (reward_hungry weights, both antennae sample the body centre),
diffusion_only (no steering) and label_decoder (the earlier side-label-trained read-out, a reference, not a control).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from noesis.experiments import decoder_contrast as dc
from noesis.experiments.agent_stats import fisher_greater, holm, sign_test_less
from noesis.experiments.decoder_forage import LEVELS, fit_decoder, run_episode
from noesis.experiments.virtual_forage import Brain
from noesis.reward_readout import EpisodeSpec, FeatureLibrary, evaluate, train
from noesis.virtual_agent import AgentParams

HUNGRY_GAIN = 1.7
CONDITIONS = {
    "reward_hungry": {"energy": 0.3, "decoder": "rw_hungry"},
    "reward_fed": {"energy": 1.0, "decoder": "rw_fed"},
    "shuffled_reward": {"energy": 0.3, "decoder": "rw_shuffled"},
    "random_readout": {"energy": 0.3, "decoder": "rw_random"},
    "reversed_sign": {"energy": 0.3, "decoder": "rw_hungry", "sign": -1.0},
    "no_gradient": {"energy": 0.3, "decoder": "rw_hungry", "no_gradient": True},
    "diffusion_only": {"energy": 0.3, "steer": False},
    "label_decoder": {"energy": 0.3, "decoder": "label"},
}


def load_library(paths: list[Path], n_components: int | None = None) -> tuple[FeatureLibrary, np.ndarray]:
    """Steady-state windows (>= 1 s into a block) of PN features from the given recordings."""
    xs, level, ratio, block, off = [], [], [], [], 0
    for p in paths:
        d = np.load(p)
        keep = d["tin"] >= 1.0 - 1e-9
        xs.append(d["pn250"][keep])
        level.append(np.array(LEVELS)[d["level"][keep]])
        ratio.append(d["ratio"][keep])
        block.append(d["block"][keep] + off)
        off += int(d["block"].max()) + 1
    x = np.concatenate(xs)
    return FeatureLibrary(x, np.concatenate(level), np.concatenate(ratio), np.concatenate(block),
                          n_components=n_components), x


def cmd_collect(a) -> None:
    dc.RATIOS = (0.9, 1.0, 1.11)          # read at call time by decoder_contrast.collect
    brain = Brain(a.data, a.cache, a.apl_gain, a.g_inh)
    ns = SimpleNamespace(runs=a.runs, blocks=a.blocks, block_s=2.0, weight=400.0, windows=a.near1)
    dc.collect(ns, brain)


def cmd_train(a) -> None:
    lib, x = load_library([a.contrast, a.near1] if a.near1.exists() else [a.contrast], a.components)
    params = AgentParams(gain_rad_s=1.5, smooth_s=0.5, antenna_mm=a.antenna_mm)
    print(f"library: {len(lib.rows)} blocks, {lib.n_features} features, near-1 blocks: {a.near1.exists()}", flush=True)
    settings = {"hungry": {"gain": HUNGRY_GAIN, "mult": HUNGRY_GAIN, "shuffle": False},
                "fed": {"gain": 1.0, "mult": 1.0, "shuffle": False},
                "shuffled": {"gain": HUNGRY_GAIN, "mult": HUNGRY_GAIN, "shuffle": True}}
    out = {"n_blocks": len(lib.rows), "lr": a.lr, "sigma": a.sigma, "iters": a.iters, "batch": a.batch, "models": {}}
    for name, s in settings.items():
        spec = EpisodeSpec(gain=s["gain"])
        res = train(lib, spec, params, iters=a.iters, batch=a.batch, lr=a.lr, sigma=a.sigma, reward_multiplier=s["mult"],
                    shuffle_reward=s["shuffle"], seed=a.seed, l2=a.l2, eval_every=a.eval_every, eval_n=100)
        final = evaluate(lib, res["w"], res["b"], spec, params, 300, seed=777_000)
        out["models"][name] = {"w": res["w"].tolist(), "w_full": lib.full_weights(res["w"]).tolist(), "b": res["b"],
                               "curve": res["curve"], "surrogate_final": final, "w_norm": float(np.linalg.norm(res["w"]))}
        print(f"{name:9s}: surrogate success {final['success']:.2f} return {final['mean_return']:.2f} "
              f"median final {final['median_final_mm']:.1f} mm | |w| {np.linalg.norm(res['w']):.2f}", flush=True)
    rng = np.random.default_rng(a.seed + 1)
    w_h = np.array(out["models"]["hungry"]["w"])
    w_rand = rng.normal(0.0, w_h.std(), size=w_h.shape)       # same number of weights and spread, no structure
    spec = EpisodeSpec(gain=HUNGRY_GAIN)
    out["models"]["random"] = {"w": w_rand.tolist(), "w_full": lib.full_weights(w_rand).tolist(), "b": 0.0, "surrogate_final": evaluate(lib, w_rand, 0.0, spec, params, 300, 777_000)}
    print("random readout surrogate:", out["models"]["random"]["surrogate_final"], flush=True)
    out["mu"], out["sd"] = lib.mu.tolist(), lib.sd.tolist()
    a.weights.parent.mkdir(parents=True, exist_ok=True)
    a.weights.write_text(json.dumps(out), encoding="utf-8")


def summarise_closed_loop(rows: list[dict]) -> dict:
    main = {r["seed"]: r for r in rows if r["condition"] == "reward_hungry"}
    out = {"n_poses": len(main), "conditions": {}}
    p_f = {}
    for c in sorted({r["condition"] for r in rows}):
        sel = {r["seed"]: r for r in rows if r["condition"] == c}
        e = {"n": len(sel), "reached": int(sum(r["reached"] for r in sel.values()))}
        if c != "reward_hungry":
            seeds = sorted(set(main) & set(sel))
            diffs = [main[s]["final_dist_mm"] - sel[s]["final_dist_mm"] for s in seeds]
            e.update(n_paired=len(seeds), closer_in=int(sum(d < 0 for d in diffs)), sign_test_p=sign_test_less(diffs),
                     mean_final_diff_mm=float(np.mean(diffs)) if diffs else None)
            p_f[c] = fisher_greater(sum(r["reached"] for r in main.values()), len(main), e["reached"], e["n"])
            e["fisher_p"] = p_f[c]
        out["conditions"][c] = e
    for c, p in holm({k: v for k, v in p_f.items() if k != "label_decoder"}).items():
        out["conditions"][c]["fisher_p_holm"] = p
    return out


def cmd_eval(a) -> None:
    w = json.loads(a.weights.read_text(encoding="utf-8"))
    brain = Brain(a.data, a.cache, a.apl_gain, a.g_inh)
    mu, sd = np.array(w["mu"]), np.array(w["sd"])
    decoders = {"idx": brain.pops.pn}
    for key, model in (("rw_hungry", "hungry"), ("rw_fed", "fed"), ("rw_shuffled", "shuffled"), ("rw_random", "random")):
        decoders[key] = {"mu": mu, "sd": sd, "w": np.array(w["models"][model]["w_full"]), "b": w["models"][model]["b"]}
    cd = np.load(a.contrast)
    keep = cd["tin"] >= 1.0 - 1e-9
    decoders["label"] = fit_decoder(cd["pn250"][keep], cd["y"][keep], 0.1)
    params = AgentParams(gain_rad_s=1.5, smooth_s=0.5, antenna_mm=a.antenna_mm)
    args = SimpleNamespace(start_mm=20.0, sigma_mm=10.0, rate_max=100.0, weight=400.0, duration_s=a.duration_s)
    rows, done = [], set()
    if a.out.exists():
        rows = [json.loads(line) for line in a.out.read_text(encoding="utf-8").splitlines() if line]
        done = {(r["condition"], r["seed"]) for r in rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    conditions = a.conditions.split(",")
    for seed in (int(s) for s in a.seeds.split(",")):
        for name in conditions:
            if (name, seed) in done:
                continue
            r = run_episode(brain, name, CONDITIONS[name], seed, args, params, decoders)
            rows.append(r)
            with a.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"seed {seed} {name:16s} reached={r['reached']!s:5} final {r['final_dist_mm']:5.1f} mm eff {r['efficiency']:.2f} "
                  f"score {r['score_mean']:+.2f}±{r['score_sd']:.2f}", flush=True)
    print(json.dumps(summarise_closed_loop(rows), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=("collect-near1", "train", "eval"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/selectivity-cache"))
    ap.add_argument("--contrast", type=Path, default=Path("outputs/decoder_contrast_windows.npz"))
    ap.add_argument("--near1", type=Path, default=Path("outputs/library_near1.npz"))
    ap.add_argument("--weights", type=Path, default=Path("outputs/reward_readout_weights.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs/reward_readout_eval.jsonl"))
    ap.add_argument("--g-inh", type=float, default=12.0)
    ap.add_argument("--apl-gain", type=float, default=1024.0)
    ap.add_argument("--antenna-mm", type=float, default=3.0)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=7)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--components", type=int, default=20, help="unsupervised PCA components of the recorded PN features")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(12)))
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--duration-s", type=float, default=12.0)
    a = ap.parse_args()
    {"collect-near1": cmd_collect, "train": cmd_train, "eval": cmd_eval}[a.stage](a)


if __name__ == "__main__":
    main()

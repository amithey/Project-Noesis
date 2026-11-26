"""Why is a leg not used? Per-leg / per-joint audit of the motor path for one gain vector.

Reports, for each of the 42 joints (6 legs x 7 DOF):
  pool_pos / pool_neg   number of motor neurons in the agonist / antagonist pool
  rate_pos / rate_neg   mean pool firing rate (Hz) during the episode
  cmd_std               std of the muscle command (rad) -> is the brain moving it?
  angle_std             std of the measured joint angle (rad) -> does it actually move?
  cmd_at_limit          share of steps the command sits at the +-0.7 rad range edge
plus per-leg claw-load share and debounced steps.

  python leg_usage_audit.py --gains <run>/best-gains.npy --seconds 3 --output <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from es_train import DEFAULT_CORE, DEFAULT_PARENT, DEFAULT_PROJECT, _paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains", type=Path, default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20001)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--core", type=Path, default=DEFAULT_CORE)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_PARENT)
    ap.add_argument("--infer-mn-sides", action="store_true")
    args = ap.parse_args()
    _paths(args.core)
    from noesis_gpu.backend import FlyGymBatch
    from noesis_gpu.bridge import LEGACY_JOINTS
    from noesis_gpu.controller import FrozenBrain
    from noesis_gpu.es_brain import CONTROL_MS, SUBSTEPS, Evaluator, _WeightHolder, debounced_steps, dump

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ev = Evaluator((args.project / "data").resolve(),
                   (args.project / "gpu-pilot/outputs/claude-es/brain-cache").resolve(), args.checkpoint,
                   infer_sides=args.infer_mn_sides)
    from es_train import load_gains
    gains = load_gains(args.gains, ev.groups.names) if args.gains else np.ones(ev.groups.n_groups)
    body = FlyGymBatch(1, "cpu")
    brain = FrozenBrain(ev.cn, body.rest, body.lo, body.hi, seed=args.seed,
                        shared_weights=_WeightHolder(ev.indptr, ev.indices, ev.data_for(gains)))
    from noesis.motor import build_muscle_matrices
    pos, neg = build_muscle_matrices(ev.cn.neurons, list(LEGACY_JOINTS))
    pool_pos, pool_neg = np.diff(pos.tocsr().indptr), np.diff(neg.tocsr().indptr)
    obs = {k: v[0] for k, v in body.observe().items()}
    cmds, angles, rpos, rneg, loads = [], [], [], [], []
    for _ in range(int(round(args.seconds * 1000 / CONTROL_MS))):
        cmd, pads = brain.step(obs, CONTROL_MS)
        body.advance(cmd[None], pads[None], SUBSTEPS)
        obs = {k: v[0] for k, v in body.observe().items()}
        rates = brain.tracker.rate
        rpos.append(pos @ rates)
        rneg.append(neg @ rates)
        cmds.append(cmd)
        angles.append(obs["angles"])
        loads.append(obs["foot_load"] > 1e-5)
    cmds, angles, loads = np.array(cmds)[10:], np.array(angles)[10:], np.array(loads)[10:]
    rpos, rneg = np.array(rpos)[10:], np.array(rneg)[10:]
    edge = (np.abs(cmds - body.lo) < 1e-3) | (np.abs(cmds - body.hi) < 1e-3)
    joints = []
    for j, name in enumerate(LEGACY_JOINTS):
        joints.append({"joint": name, "pool_pos": int(pool_pos[j]), "pool_neg": int(pool_neg[j]),
                       "rate_pos_hz": float(rpos[:, j].mean()), "rate_neg_hz": float(rneg[:, j].mean()),
                       "cmd_mean_minus_rest": float(cmds[:, j].mean() - body.rest[j]),
                       "cmd_std": float(cmds[:, j].std()), "angle_std": float(angles[:, j].std()),
                       "cmd_at_limit": float(edge[:, j].mean())})
    legs = {}
    steps = debounced_steps(loads)
    for i, leg in enumerate(("LF", "LM", "LH", "RF", "RM", "RH")):
        rows = joints[7 * i:7 * i + 7]
        legs[leg] = {"claw_load_share": float(loads[:, i].mean()), "debounced_steps": steps[i],
                     "motor_neurons": int(sum(r["pool_pos"] + r["pool_neg"] for r in rows)),
                     "empty_pools": [r["joint"] for r in rows if r["pool_pos"] + r["pool_neg"] == 0],
                     "mean_cmd_std": float(np.mean([r["cmd_std"] for r in rows])),
                     "mean_angle_std": float(np.mean([r["angle_std"] for r in rows]))}
    report = {"motor_side_inference": ev.side_audit, "gains": str(args.gains), "seed": args.seed, "seconds": args.seconds, "legs": legs,
              "joints": joints}
    dump(out / "leg-usage.json", report)
    print(f"{'leg':4} {'MNs':>4} {'cmd_std':>8} {'ang_std':>8} {'load':>5} {'steps':>5}  empty pools")
    for leg, r in legs.items():
        print(f"{leg:4} {r['motor_neurons']:4d} {r['mean_cmd_std']:8.4f} {r['mean_angle_std']:8.4f} "
              f"{r['claw_load_share']:5.2f} {r['debounced_steps']:5d}  {', '.join(r['empty_pools'])}")


if __name__ == "__main__":
    main()

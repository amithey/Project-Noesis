"""Offline audit of the two ACTIVE senses (SensoryMap, BalanceMap) on existing frozen-probe observations.

Descriptive only: how much of each organ class's 0..1 activation range is used (dead < .05, saturated > .95) with the
current transfer constants, from the same body signals the controller passes (controller.py step()).
usage: active_sense_audit.py <core src> <out.json> <run dir> [<run dir> ...]   (dirs hold report.json + observations.npz)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
from noesis.balance import UPRIGHT  # noqa: E402
from noesis.sensory import LegState, _signal  # noqa: E402

out, dirs = sys.argv[2], [Path(d) for d in sys.argv[3:]]
VEL0, LOAD0, TILT, SPIN = 2.0, 2.0, .3, 3.0  # current controller constants (SensoryMap / BalanceMap defaults)
KINDS = ("claw", "hook", "club", "hairplate", "campaniform")  # campaniform and bristle share one signal


def stats(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)), "dead_share": float((x < .05).mean()), "saturated_share": float((x > .95).mean())}


acts = {k: [] for k in KINDS}
raw = {"load": [], "tibia_speed": [], "tilt_norm": [], "omega_norm": []}
for d in dirs:
    rest = np.array(json.loads((d / "report.json").read_text(encoding="utf-8-sig"))["body"]["neutral_pose"])
    z = np.load(d / "observations.npz")
    ang = (z["angles"][:, 0] - rest).reshape(-1, 6, 7)
    vel = z["velocity"][:, 0].reshape(-1, 6, 7)
    load = z["load"][:, 0]
    for t in range(10, ang.shape[0]):  # skip the 10 warmup samples
        for i in range(6):
            s = LegState(ang[t, i, 5], vel[t, i, 5], ang[t, i, 0], ang[t, i, 3], load[t, i])
            for k in KINDS:
                acts[k].append(_signal(k, s, VEL0, LOAD0))
            raw["load"].append(load[t, i])
            raw["tibia_speed"].append(abs(vel[t, i, 5]))
    raw["tilt_norm"] += list(np.linalg.norm(z["up_body"][10:, 0] - UPRIGHT, axis=1))
    raw["omega_norm"] += list(np.linalg.norm(z["omega"][10:, 0], axis=1))
res = {"constants": {"vel0": VEL0, "load0": LOAD0, "tilt_scale": TILT, "spin_scale": SPIN}, "runs": [str(d) for d in dirs],
       "organ_activation": {k: stats(v) for k, v in acts.items()},
       "raw_signal_quantiles": {k: {q: float(np.percentile(v, q)) for q in (10, 50, 90, 99)} for k, v in raw.items()},
       "note": "campaniform stats also describe bristle (same signal). Descriptive; no claim about better standing."}
Path(out).write_text(json.dumps(res, indent=1), encoding="utf-8")
for k, v in res["organ_activation"].items():
    print(k, {a: round(b, 3) for a, b in v.items()})
print(res["raw_signal_quantiles"])

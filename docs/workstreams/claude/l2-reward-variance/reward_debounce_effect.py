"""Effect of debouncing the v8 'standing' boolean (persistence filter, same state machine as the evaluation) on the reward. Offline.

usage: reward_debounce_effect.py <core src> <run dir> [<run dir> ...]
"""
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from noesis.embodied_metrics import debounce_contact, locomotion_reward  # noqa: E402

DT, KEYS = 5.0, ("position", "up_body", "velocity", "heading", "feet_position", "foot_load")


def ema(x, tau_ms):
    a = 1 - np.exp(-DT / tau_ms)
    out, s = np.empty_like(x), x[0]
    for i, v in enumerate(x):
        s += a * (v - s)
        out[i] = s
    return out


def per_run(d):
    z = np.load(f"{d}/training-observations.npz")
    obs = {k: (z[k][:, 0] if z[k].ndim > {"position": 2, "up_body": 2, "velocity": 2, "heading": 2, "feet_position": 3, "foot_load": 2}[k] else z[k]) for k in KEYS}
    rest, standing, sq = [], [], []
    for i in range(10, len(obs["position"])):
        b = {k: obs[k][i - 1] for k in KEYS}
        a = {k: obs[k][i] for k in KEYS}
        dt = DT / 1000
        r = locomotion_reward(b["position"], a["position"], float(np.clip(a["up_body"][2], -1, 1)), a["velocity"], dt, b["heading"],
                              a["foot_load"], (a["feet_position"] - b["feet_position"]) / dt, objective="curriculum-gripped-step-v8")
        s = bool(r["supporting_feet"] >= 1 and .3 <= r["height_mm"] <= 1.5 and r["uprightness"] >= .7)
        standing.append(s)
        sq.append(min(r["supporting_feet"] / 3, 1))
        rest.append(.6 * r["grounded_progress"] - .05 * r["effort"] - .03 * r["stance_slip_mm_s"] - .2 * r["obstacle_cost"])
    return np.array(standing), np.array(sq), np.array(rest)


def total(standing, sq, rest):
    return np.clip(.4 * standing - .8 * (1 - standing) + .1 * standing * sq + rest, -1, 1)


orig, clean, flips_o, flips_c, low_o, low_c = [], [], 0, 0, [], []
for d in sys.argv[2:]:
    st, sq, rest = per_run(d)
    deb = debounce_contact(st, DT, 15.0)
    o, c = total(st, sq, rest), total(deb, sq, rest)
    orig.append(o)
    clean.append(c)
    flips_o += int(np.abs(np.diff(st.astype(int))).sum())
    flips_c += int(np.abs(np.diff(deb.astype(int))).sum())
    low_o.append(ema(o, 1000.))
    low_c.append(ema(c, 1000.))
O, C = np.concatenate(orig), np.concatenate(clean)
fast = lambda xs: np.concatenate([x - ema(x, 200.) for x in xs])  # noqa: E731
print(f"steps {len(O)}; standing flips: original {flips_o}, debounced {flips_c} ({flips_c / flips_o:.0%})")
print(f"reward variance: original {O.var():.4f} -> debounced {C.var():.4f} ({C.var() / O.var():.0%})")
print(f"fast (>5 Hz) variance: original {fast(orig).var():.4f} ({fast(orig).var() / O.var():.0%} of its total) -> debounced {fast(clean).var():.4f} ({fast(clean).var() / C.var():.0%})")
print(f"correlation original vs debounced reward: {np.corrcoef(O, C)[0, 1]:.3f}; after 1 s smoothing: {np.corrcoef(np.concatenate(low_o), np.concatenate(low_c))[0, 1]:.4f}")
print(f"mean reward original {O.mean():.4f}, debounced {C.mean():.4f}; share of steps whose value differs {np.mean(O != C):.1%}")

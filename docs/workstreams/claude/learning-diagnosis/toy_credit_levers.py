"""Which lever raises the toy rule's signal-to-noise: more steps, less fast reward noise, or eligibility matched to the lag?

Same synthetic positive control as toy_credit_snr.py (imports it). kappa = 0.05 (weak but real effect), learning rate 0.2.
usage: toy_credit_levers.py <core src dir>
"""
import sys

import numpy as np

sys.argv = [sys.argv[0], sys.argv[1], "quick"]
import toy_credit_snr as toy  # noqa: E402

KAPPA, LR = 0.05, 0.2
CONFIGS = [
    ("baseline 990 steps (5 s)", dict(steps=990)),
    ("5x longer (25 s)", dict(steps=4950)),
    ("10x longer (50 s)", dict(steps=9900)),
    ("fast reward noise 0.30 -> 0.10", dict(steps=990, sig_fast=0.10)),
    ("fast reward noise 0.30 -> 0.10 and slow 0.2 -> 0.05", dict(steps=990, sig_fast=0.10, sig_slow=0.05)),
    ("credit lag 150 ms, eligibility 100 ms", dict(steps=990, lag=150)),
    ("credit lag 150 ms, eligibility 200 ms", dict(steps=990, lag=150, tau_elig=200.)),
    ("credit lag 150 ms, eligibility 300 ms", dict(steps=990, lag=150, tau_elig=300.)),
]
print(f"kappa {KAPPA}, lr {LR}, {toy.SEEDS} seeds")
print(f"{'configuration':58s} {'good change':>12s} {'z':>6s} {'right way':>10s}")
for name, cfg in CONFIGS:
    lag = cfg.pop("lag", 0)
    good, ctrl = [], []
    for s in range(toy.SEEDS):
        g, c, _ = toy.one_birth(s, KAPPA, lag, LR, **cfg)
        good.append(g)
        ctrl.append(c)
    diff = np.array(good) - np.array(ctrl)
    z = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)) + 1e-12)
    print(f"{name:58s} {np.mean(good):+12.5f} {z:6.2f} {(diff > 0).mean():10.0%}", flush=True)

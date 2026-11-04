"""Review-only counterexamples for the delivered chain statistic; no fly or learning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--core", type=Path, required=True)
ap.add_argument("--diagnostic", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
started = time.perf_counter()
sys.path[:0] = [str(args.core / "src"), str(args.diagnostic / "gpu-pilot/src")]
from noesis_gpu import chain_diagnostic as cd

def git_bytes(path):
    return subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(args.diagnostic.resolve()),
         "show", "aa7ecdbc03c3e064ff616c905b1b8016c337a238:" + path],
        cwd=args.diagnostic,
    )

sources = {}
for path in ["gpu-pilot/src/noesis_gpu/chain_diagnostic.py",
             "gpu-pilot/src/noesis_gpu/chain_run.py",
             "docs/workstreams/claude/chain-diagnostic/analyse_chain.py"]:
    canonical = git_bytes(path)
    local = (args.diagnostic / path).read_bytes().replace(b"\r\n", b"\n")
    assert local == canonical
    sources[path] = hashlib.sha256(canonical).hexdigest()

rng = np.random.default_rng(20261104)
steps, lag = 1000, 7
base = rng.normal(size=(steps, 4))
x = np.repeat(base, 2, axis=1)
signs = np.tile([1.0, -1.0], 4)
uniform_y = np.roll(x, lag, axis=0)
opposed_y = uniform_y * signs
options = dict(max_lag=40, permutations=200, min_shift=200, seed=0)
uniform = cd.coupling(x, uniform_y, **options)
opposed = cd.coupling(x, opposed_y, **options)
per_joint = [float(np.corrcoef(x[:, j], np.roll(opposed_y[:, j], -lag))[0, 1])
             for j in range(x.shape[1])]
assert uniform["observed_r2"] > .999 and uniform["best_lag_steps"] == lag
assert opposed["observed_r2"] < 1e-25
assert all(abs(v) > .999 for v in per_joint)

# A failed invariance check must prevent scientific reading. The delivered
# chain_report nevertheless returns ordinary coupling fields.
joints = 4
z = rng.uniform(-1, 1, size=(steps, joints))
drive = np.zeros((steps, 2 * joints))
drive[:, ::2] = z
pos = np.zeros((joints, 2 * joints), bool)
neg = np.zeros_like(pos)
for j in range(joints):
    pos[j, 2 * j] = True
    neg[j, 2 * j + 1] = True
active = {
    "drive": drive,
    "circuit_spikes": np.repeat(np.full((steps, joints), 20), 2, axis=1),
    "total_spikes": np.full(steps, 100),
    "a_pos": .5 + .1 * z,
    "a_neg": np.full((steps, joints), .5),
    "command": z.copy(),
    "adhesion": np.zeros((steps, 6)),
    "obs_angles": z.copy(),
    "obs_velocity": z.copy(),
    "obs_position": np.zeros((steps, 3)),
    "obs_up_body": np.tile([0., 0., 1.], (steps, 1)),
    "obs_load": np.zeros((steps, 6)),
    "rest": np.zeros(joints),
}
# Nonnegative integer-valued counts with a clear input relationship.
active["circuit_spikes"][:, ::2] += (5 * z).astype(int)
off = copy.deepcopy(active)
off["drive"][:] = 0
off["circuit_spikes"][:] = 20
for key in ["command", "obs_angles", "obs_velocity"]:
    off[key] = rng.normal(size=(steps, joints))
off["a_pos"] = rng.uniform(.4, .6, size=(steps, joints))
zero = copy.deepcopy(off)
zero["command"][0, 0] += 1
gate = cd.chain_report({"off": off, "zero": zero, "active": active}, pos, neg, **options)
assert not gate["invariance_off_vs_zero"]["identical"]
assert gate["coupling"]["active"]["command"]["observed_r2"] > .999
assert gate["coupling"]["active"]["command"]["reference_rank"] < .05

result = {
    "scope": "Synthetic review counterexamples only; no physical/network simulation, no training or gait evidence.",
    "diagnostic_commit": "aa7ecdbc03c3e064ff616c905b1b8016c337a238",
    "canonical_git_source_sha256": sources,
    "same_sign_control": uniform,
    "opposed_sign_counterexample": opposed,
    "individual_signed_correlations_at_planted_lag": per_joint,
    "invariance_gate_counterexample": {
        "invariance_identical": gate["invariance_off_vs_zero"]["identical"],
        "coupling_still_returned": True,
        "active_command": gate["coupling"]["active"]["command"],
    },
    "interpretation": [
        "The shared signed coefficient can cancel perfect jointwise coupling. This does not show that real fly data have this structure.",
        "chain_report computes coupling after an invariance failure; analyse_chain.py does not gate reading on invariance or driver completion.",
        "Historical real runs report successful invariance; these counterexamples do not invalidate those runs or prove a broken physical chain.",
    ],
    "elapsed_seconds": time.perf_counter() - started,
}
assert result["elapsed_seconds"] < 60
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))


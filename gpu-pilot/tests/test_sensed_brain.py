"""The hook is a Codex-file patch that is NOT applied in this branch: these tests apply it to a temporary copy of the package and run a fresh
interpreter against that copy (no import collision), so the patch file itself is what is verified."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from noesis_gpu.sensed_brain import GradientField, sensed_class, thermo_signals

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "docs/workstreams/claude/senses/hook-proposal/controller-extra-drive-parts.patch"
needs_core = pytest.mark.skipif(importlib.util.find_spec("noesis.lif") is None, reason="Requires pinned Noesis core")

SCRIPT = r'''
import hashlib, json
import numpy as np, pandas as pd
from scipy import sparse
from noesis.connectome import Connectome
from noesis_gpu.bridge import LEGS
from noesis_gpu.controller import FrozenBrain
from noesis_gpu.sensed_brain import GradientField, sensed_class

rows = []
for leg in LEGS:
    side = "left" if leg[0] == "L" else "right"
    pos = {"F": "front", "M": "middle", "H": "hind"}[leg[1]]
    for cls, sub, primary, flow in (("leg_motor_neuron", f"{pos}_leg_motor_neuron", "tibia_flexor", "efferent"),
                                    ("leg_motor_neuron", f"{pos}_leg_motor_neuron", "tibia_extensor", "efferent"),
                                    ("sensory_neuron", f"{pos}_leg_claw_chordotonal_organ_neuron", "claw", "afferent")):
        rows.append({"Class": cls, "Sub Class": sub, "Primary Cell Type": primary, "Flow": flow, "Nerve": f"{side}_leg", "Soma side": side, "Function": ""})
for fn in ("heating", "heating", "cooling", "cooling"):
    rows.append({"Class": "sensory_neuron", "Sub Class": "thermosensory_receptor", "Primary Cell Type": "thermo", "Flow": "afferent",
                 "Nerve": "", "Soma side": "left", "Function": fn})
table = pd.DataFrame(rows)
n = len(table)
rng = np.random.default_rng(0)
w = (rng.random((n, n)) * (rng.random((n, n)) < .6)).astype(np.float32) * 3
cn = Connectome(table, sparse.csr_matrix(w))
REST, LO, HI = np.zeros(42), np.full(42, -.7), np.full(42, .7)

def state():
    return {"position": np.array([0., 0., 1.]), "up_body": np.array([0., 0., 1.]), "omega": np.zeros(3), "angles": np.zeros(42),
            "velocity": np.zeros(42), "heading": np.array([1., 0., 0.]), "feet_position": np.zeros((6, 3)), "load": np.ones(6), "foot_load": np.zeros(6)}

def run(brain, steps=60):
    out, spikes = [], 0
    for _ in range(steps):
        c, _ = brain.step(state(), 5.0)
        out.append(c.copy()); spikes += len(brain.last_spikes)
    return hashlib.sha256(np.stack(out).tobytes()).hexdigest(), spikes

field = GradientField(axis=0, origin=-1.0, length=2.0)   # u = 0.5 at position 0
Sensed = sensed_class(FrozenBrain)
result = {"plain": run(FrozenBrain(cn, REST, LO, HI, seed=3))}
for mode in ("off", "zero", "constant", "active"):
    b = Sensed(cn, REST, LO, HI, seed=3, field=field, gain=800.0, mode=mode, level=0.5)
    result[mode] = run(b)
    result[mode + "_weights_unchanged"] = b.weights_unchanged()
result["disjoint"] = bool(len(np.intersect1d(b.thermo_idx, np.union1d(b.senses.idx, b.balance.idx))) == 0)
result["thermo_n"] = int(len(b.thermo_idx))
print(json.dumps(result))
'''


@pytest.fixture(scope="module")
def patched_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("patched")
    package = tmp / "gpu-pilot/src/noesis_gpu"
    shutil.copytree(ROOT / "gpu-pilot/src/noesis_gpu", package, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "apply", str(PATCH)], check=True, cwd=tmp)  # tmp mirrors gpu-pilot/src/noesis_gpu, outside any repository
    script = tmp / "run.py"
    script.write_text(SCRIPT, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp / "gpu-pilot/src"), os.environ.get("NOESIS_SOURCE", str(ROOT / "src"))]),
               PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path[:0]=sys.argv[1].split({os.pathsep!r}); exec(open(sys.argv[2]).read())",
                           env["PYTHONPATH"], str(script)], capture_output=True, text=True, env=env, timeout=240, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@needs_core
def test_disabled_and_zero_signal_are_bit_identical_to_the_plain_brain(patched_run):
    assert patched_run["off"] == patched_run["plain"] and patched_run["zero"] == patched_run["plain"]


@needs_core
def test_a_real_signal_adds_afferent_spiking_and_the_constant_control_is_matched_in_level(patched_run):
    # In this tiny random network the thermo afferents reach no muscle pool, so the command hash is unchanged; the observable effect is spiking.
    assert patched_run["active"][1] > patched_run["plain"][1] and patched_run["constant"][1] > patched_run["plain"][1]
    assert patched_run["active"] == patched_run["constant"]  # field value 0.5 at the fixed test position: the controls are matched in level


@needs_core
def test_selected_afferents_are_disjoint_from_wired_ones_and_weights_never_change(patched_run):
    assert patched_run["disjoint"] and patched_run["thermo_n"] == 4
    assert all(patched_run[f"{m}_weights_unchanged"] for m in ("off", "zero", "constant", "active"))


def test_class_factory_refuses_a_controller_without_the_hook():
    class Plain:
        pass
    with pytest.raises(TypeError, match="hook"):
        sensed_class(Plain)


def test_field_and_signal_split_are_bounded_and_opponent():
    f = GradientField(axis=0, origin=0.0, length=10.0)
    assert f({"position": np.array([-30., 0, 0])}) == -1 and f({"position": np.array([5., 0, 0])}) == .5
    assert thermo_signals(.5) == {"thermo_warm": .5, "thermo_cool": 0.0} and thermo_signals(-.25) == {"thermo_warm": 0.0, "thermo_cool": .25}
    for bad in ({"axis": 3}, {"length": 0}, {"origin": float("nan")}):
        with pytest.raises(ValueError):
            GradientField(**bad)

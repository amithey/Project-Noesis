"""Compare the sensory interface to its pristine predecessor on a tiny graph."""

import hashlib
import json
import subprocess
import tempfile
import types
from pathlib import Path

import numpy as np
from noesis_gpu import controller
from test_learning import state
from test_weight_integrity import build

BASE = "24b69a1a9de17a7ee4abb12cdbb22295ad710371"
PATH = "gpu-pilot/src/noesis_gpu/controller.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    pristine = types.ModuleType("noesis_gpu.pristine_controller")
    source = subprocess.check_output(["git", "show", f"{BASE}:{PATH}"])
    exec(compile(source, PATH, "exec"), pristine.__dict__)
    with tempfile.TemporaryDirectory() as tmp:
        cn, parent = build(Path(tmp))
        bank_hash = sha(parent.read_bytes())
        rest, lo, hi = np.zeros(42), np.full(42, -0.7), np.full(42, 0.7)
        steps, total_spikes = 120, 0
        for seed in (10006, 10010):
            original = pristine.FrozenBrain(cn, rest, lo, hi, seed=seed, checkpoint=parent)
            candidate = controller.FrozenBrain(cn, rest, lo, hi, seed=seed, checkpoint=parent)
            for step in range(steps):
                obs = state()
                obs["angles"] = np.full(42, 0.2 * np.sin(step / 7))
                obs["velocity"] = np.full(42, 0.2 * np.cos(step / 7))
                obs["load"] = np.arange(6) * (1 + step % 3)
                expected = original.step(obs, 5.0)
                actual = candidate.step(obs, 5.0)
                for a, b in zip(expected, actual, strict=True):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(candidate.last_spikes, original.last_spikes)
                total_spikes += len(candidate.last_spikes)
            assert original.weights_unchanged() and candidate.weights_unchanged()
            assert original.initial_weights_sha256 == candidate.initial_weights_sha256
            for name in ("indptr", "indices", "data"):
                a, b = getattr(original.net, name), getattr(candidate.net, name)
                np.testing.assert_array_equal(a, b)
                assert not a.flags.writeable and not b.flags.writeable

        class Duplicate(controller.FrozenBrain):
            def _extra_drive_parts(self, observation):
                return [(self.senses.idx[:1], np.array([1.0]), self.sensory_weight)]

        duplicate = Duplicate(cn, rest, lo, hi, checkpoint=parent)
        assert len(duplicate.senses.idx)
        try:
            duplicate.step(state(), 5.0)
        except ValueError as exc:
            assert "overlap" in str(exc).lower(), str(exc)
        else:
            raise AssertionError("Existing afferent was accepted as a duplicate")
        assert sha(parent.read_bytes()) == bank_hash
    return {
        "pristine_commit": BASE,
        "pristine_controller_sha256": sha(source),
        "candidate_controller_sha256": sha(Path(controller.__file__).read_bytes().replace(b"\r\n", b"\n")),
        "synthetic_neurons": cn.n,
        "seeds": [10006, 10010],
        "steps_per_seed": steps,
        "observed_spikes": total_spikes,
        "exact_commands_adhesion_and_per_step_spikes": True,
        "exact_readonly_topology_and_weights": True,
        "checkpoint_unchanged": True,
        "duplicate_input_rejected": True,
        "physical_simulation": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))

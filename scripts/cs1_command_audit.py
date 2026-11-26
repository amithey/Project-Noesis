"""CS1 offline reconstruction of the native command slew limiter."""
import argparse
import ast
import json
from pathlib import Path
import time

import numpy as np
import run_mapped_target_search as trial


def reconstruct(desired, rest, limit=.06):
    desired, rest = np.asarray(desired, dtype=float), np.asarray(rest, dtype=float)
    if desired.ndim != 2 or rest.shape != (desired.shape[1],) or not len(desired):
        raise ValueError("Invalid command shapes")
    if not np.isfinite(desired).all() or not np.isfinite(rest).all() or not 0 < limit < np.inf:
        raise ValueError("Invalid command values")
    command = rest.copy()
    commands, active = [], []
    for target in desired:
        delta = target - command
        active.append(np.abs(delta) > limit)
        command += np.clip(delta, -limit, limit)
        commands.append(command.copy())
    return np.asarray(commands), np.asarray(active)


def native_reconstruction(source, desired, rest):
    tree = ast.parse(source)
    brain = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FrozenBrain")
    step = next(n for n in brain.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    update = [n for n in step.body if isinstance(n, ast.AugAssign) and
              isinstance(n.target, ast.Attribute) and n.target.attr == "command"]
    if len(update) != 1:
        raise ValueError("Unexpected native command update")
    expression = ast.unparse(update[0])
    if expression != "self.command += np.clip(desired - self.command, -0.06, 0.06)":
        raise ValueError("Native limiter changed")
    from types import SimpleNamespace
    state = SimpleNamespace(command=rest.copy())
    code = compile(ast.Module(body=update, type_ignores=[]), "<pinned-native-command>", "exec")
    values = []
    for row in desired:
        exec(code, {"np": np, "self": state, "desired": row})
        values.append(state.command.copy())
    return np.asarray(values)


def prepare(manifest):
    if manifest.exists():
        raise ValueError("Manifest exists")
    plan_path = trial.ROOT / "outputs/mapped-target-search-20261114-prepared-private.json"
    plan = trial.read(plan_path)
    out = Path(plan["out"])
    result_path = out / "result.json"
    result = trial.read(result_path)
    if result["status"] != "completed" or len(result["jobs"]) != 25:
        raise ValueError("Incomplete source experiment")
    pins = {**plan["pins"], **result["created_hashes"],
            str(plan_path): trial.sha(plan_path), str(result_path): trial.sha(result_path)}
    mp2 = trial.ROOT / "outputs/training-context-mp2-20261113/result.json"
    prior_created = trial.read(mp2)["created_hashes"]
    records = []
    for seed in trial.ss1.TRAIN:
        parent = trial.ss1.reference(plan["prior"], seed)
        for label, folder in [("parent", parent), *[(f"{group}/{sign:+}", out / f"{group}-{seed}-{sign:+}")
                                                   for group in trial.SETS for sign in trial.SIGNS]]:
            for name in ("report.json", "adhesion-observations.npz"):
                path = folder / name
                expected = pins.get(str(path), prior_created.get(str(path)))
                if expected is None or trial.sha(path) != expected:
                    raise ValueError("Missing or changed sealed command evidence")
                pins[str(path)] = expected
            records.append({"label": label, "seed": seed, "folder": str(folder)})
    for name in ("cs1_command_audit.py", "test_cs1_command_audit.py"):
        path = trial.HERE / name
        pins[str(path)] = trial.sha(path)
    controller = Path(plan["runtime"]) / "work/Noesis-Sensory/gpu-pilot/src/noesis_gpu/controller.py"
    if str(controller) not in pins:
        raise ValueError("Native controller lacks signed provenance")
    trial.ss1.base.check_pins(pins)
    trial.write(manifest, {"records": records, "pins": pins, "controller": str(controller)})


def audit(manifest):
    started = time.monotonic()
    config = trial.read(manifest)
    pins = {**config["pins"], str(manifest): trial.sha(manifest)}
    trial.ss1.base.check_pins(pins)
    source = Path(config["controller"]).read_text()
    rows, commands = [], {}
    for entry in config["records"]:
        folder = Path(entry["folder"])
        report = trial.read(folder / "report.json")
        if report["status"] != "completed_transport_pilot" or report["completed_control_chunks"] != 1000:
            raise ValueError("Incomplete source horizon")
        body = report["body"]
        rest, lo, hi = [np.asarray(body[key], dtype=float) for key in
                        ("neutral_pose", "lower_limits", "upper_limits")]
        recording = trial.ss1.arrays(folder / "adhesion-observations.npz")
        desired = recording["desired"]
        if desired.shape != (1000, 42) or rest.shape != (42,) or lo.shape != (42,) or hi.shape != (42,):
            raise ValueError("Incorrect full single-world horizon")
        if not np.all(np.isfinite([rest, lo, hi])) or np.any(hi <= lo) or np.any(rest < lo) or np.any(rest > hi):
            raise ValueError("Incorrect joint limits")
        if np.any(desired < lo - 1e-12) or np.any(desired > hi + 1e-12):
            raise ValueError("Desired angles outside native joint limits")
        command, active = reconstruct(desired, rest)
        np.testing.assert_array_equal(command, native_reconstruction(source, desired, rest))
        femur = (desired - rest).reshape(1000, 6, 7)[:, :, 3]
        np.testing.assert_array_equal(recording["Femur_delta"], femur)
        np.testing.assert_array_equal(recording["pads"], (femur <= .08).astype(float))
        normalized = (command - rest) / (hi - lo)
        rows.append(dict(label=entry["label"], seed=entry["seed"],
                         limiter_fraction=float(active[10:].mean()),
                         normalized_excursion_rms=float(np.sqrt(np.mean(normalized[10:] ** 2)))))
        commands[entry["label"], entry["seed"]] = (desired, command)
    summaries = {}
    for label in ("parent", *[f"{g}/{s:+}" for g in trial.SETS for s in trial.SIGNS]):
        selected = [r for r in rows if r["label"] == label]
        summaries[label] = {name: float(np.mean([r[name] for r in selected])) for name in
                            ("limiter_fraction", "normalized_excursion_rms")}
        if label != "parent":
            differences = []
            for seed in trial.ss1.TRAIN:
                desire, command = commands[label, seed]
                parent_desire, parent_command = commands["parent", seed]
                differences.append([np.mean((desire[10:] - parent_desire[10:]) ** 2),
                                    np.mean((command[10:] - parent_command[10:]) ** 2)])
            summaries[label]["desired_difference_rms_rad"] = float(np.sqrt(np.mean(np.asarray(differences)[:, 0])))
            summaries[label]["command_difference_rms_rad"] = float(np.sqrt(np.mean(np.asarray(differences)[:, 1])))
    trial.ss1.base.check_pins(pins)
    if time.monotonic() - started > 180:
        raise TimeoutError("Offline budget exceeded")
    return dict(status="completed_CS1",recordings=len(rows),summaries=summaries,
                native_recurrence_exact=True,adhesion_exact=True,input_source_hashes_unchanged=True,
                new_simulation_workers=0,walking_proven=False,
                limits="Reconstructed controller commands. Activation states and actuator forces are unobserved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.manifest.resolve())
    else:
        print(json.dumps(audit(args.manifest.resolve()), indent=2, allow_nan=False))

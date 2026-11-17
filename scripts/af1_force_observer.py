"""Observe existing native physics calls and summarize actuator force bounds."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def summarize(force, limits, enabled, dt=.0001, warmup=.05):
    force, limits, enabled = np.asarray(force), np.asarray(limits), np.asarray(enabled, dtype=bool)
    if (force.ndim != 2 or limits.shape != (force.shape[1], 2) or
            enabled.shape != (force.shape[1],) or not len(force)):
        raise ValueError("Invalid force-array shapes")
    if not np.isfinite(force).all() or not np.isfinite(limits).all() or not 0 < dt < np.inf:
        raise ValueError("Nonfinite force input or invalid timestep")
    if np.any(limits[:, 0] >= 0) or np.any(limits[:, 1] <= 0) or not enabled.all():
        raise ValueError("Require enabled force limits spanning zero")
    if not np.isfinite(warmup) or warmup < 0:
        raise ValueError("Invalid warmup")
    values = force[int(round(warmup / dt)):]
    if not len(values):
        raise ValueError("Empty post-warmup horizon")
    tolerance = 1e-8 * np.maximum(1., np.abs(limits))
    if np.any(values < limits[:, 0] - tolerance[:, 0]) or np.any(values > limits[:, 1] + tolerance[:, 1]):
        raise ValueError("Recorded force exceeds configured limits")
    hit = (np.abs(values - limits[:, 0]) <= tolerance[:, 0]) | (np.abs(values - limits[:, 1]) <= tolerance[:, 1])
    streak, longest = np.zeros(force.shape[1], dtype=int), 0
    for row in hit:
        streak = np.where(row, streak + 1, 0)
        longest = max(longest, int(streak.max()))
    normalized = np.where(values >= 0, values / limits[:, 1], values / limits[:, 0])
    return dict(bound_hit_fraction=float(hit.mean()), peak_normalized_force=float(normalized.max()),
                longest_bound_hit_seconds=longest * dt, samples=int(len(values)),
                actuators=force.shape[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1010, 1011), required=True)
    args = parser.parse_args()
    root, out = args.runtime.resolve(), args.output.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError("Require fresh output outside source runtime")
    sys.path[:0] = [str(root / "work"), str(root / "work/Noesis-Sensory/gpu-pilot/src"),
                   str(root / "work/Noesis-Core/src")]
    import ap1c_body
    import ss1_worker
    import mujoco

    original_observer = ap1c_body.observing_body
    instances = []

    def observer(original, anatomy_instances):
        anatomy = original_observer(original, anatomy_instances)

        class ForceBody(anatomy):
            def __init__(self, *values, **options):
                super().__init__(*values, **options)
                model = self.sim.mj_model
                joints = self.sim._internal_jointids_by_fly[self.fly.name][self.read_order]
                actuator_ids = []
                for joint in joints:
                    candidates = np.flatnonzero((model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT)) &
                                               (model.actuator_trnid[:, 0] == joint))
                    if len(candidates) != 1:
                        raise ValueError("Expected exactly one position actuator per active joint")
                    actuator_ids.append(int(candidates[0]))
                if len(set(actuator_ids)) != 42:
                    raise ValueError("Incorrect active actuator layout")
                actuator_ids = np.asarray(actuator_ids)
                self.force_metadata = dict(
                    actuator_ids=actuator_ids.tolist(), joint_ids=joints.tolist(),
                    actuator_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in actuator_ids],
                    limits=model.actuator_forcerange[actuator_ids].tolist(),
                    enabled=model.actuator_forcelimited[actuator_ids].tolist(),
                    timestep=float(model.opt.timestep), sampling="after-existing-native-sim-step",
                    force_semantics="actuation-space force from the last solver stage",
                    kp=model.actuator_gainprm[actuator_ids, 0].tolist(),
                )
                if not np.all(model.actuator_forcelimited[actuator_ids]) or not np.all(
                    model.actuator_gainprm[actuator_ids, 0] == 45
                ):
                    raise ValueError("Unexpected position actuator parameters")
                np.testing.assert_array_equal(model.actuator_forcerange[actuator_ids],
                                              np.tile([-65., 65.], (42, 1)))
                self.force_samples = np.empty((50000, 42))
                self.control_samples = np.empty((50000, 42))
                self.sample_time = np.empty(50000)
                self.force_count = 0
                native_step = self.sim.step

                def observed_step(*step_args, **step_options):
                    returned = native_step(*step_args, **step_options)
                    index = self.force_count
                    if index >= 50000:
                        raise ValueError("Unexpected additional native step")
                    data = self.sim.mj_data
                    self.force_samples[index] = data.actuator_force[actuator_ids]
                    self.control_samples[index] = data.ctrl[actuator_ids]
                    self.sample_time[index] = data.time
                    self.force_count += 1
                    return returned

                self.sim.step = observed_step
                instances.append(self)

        return ForceBody

    ap1c_body.observing_body = observer
    argv = sys.argv
    try:
        sys.argv = ["ss1_worker", "--runtime-root", str(root), "--output", str(out),
                    "--checkpoint", str(args.checkpoint.resolve()), "--seed", str(args.seed),
                    "--phase", "trial", "--bank", "probe", "--direction", "0", "--sign", "1",
                    "--amplitude", ".005"]
        ss1_worker.main()
        if len(instances) != 1 or instances[0].force_count != 50000:
            raise ValueError("Incomplete native physics telemetry")
        body = instances[0]
        np.testing.assert_allclose(body.sample_time, np.arange(1, 50001) * .0001, rtol=0, atol=1e-10)
        if not np.isfinite(body.control_samples).all():
            raise ValueError("Nonfinite recorded actuator input")
        np.savez_compressed(out / "joint-actuator-forces.npz", time=body.sample_time,
                            force=body.force_samples, control=body.control_samples)
        (out / "joint-actuator-metadata.json").write_text(json.dumps(body.force_metadata, indent=2))
    finally:
        sys.argv = argv
        ap1c_body.observing_body = original_observer


if __name__ == "__main__":
    main()

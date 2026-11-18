"""EC1 native learning episodes and frozen evaluation with read-only audits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from ec1_contract import BANKS, TRAIN, lineage


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "frozen"), required=True)
    parser.add_argument("--bank", choices=("parent", *BANKS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root, out, checkpoint = args.root.resolve(), args.output.resolve(), args.checkpoint.resolve()
    if out.exists() or out.is_relative_to(root) or args.seed not in TRAIN:
        raise ValueError("Require registered TRAIN context and fresh output")
    if args.phase == "train" and lineage(args.bank)[1] != args.seed:
        raise ValueError("Wrong training lineage")
    sys.path[:0] = [str(root / "work"), str(root / "work/Noesis-Sensory/gpu-pilot/src"),
                   str(root / "work/Noesis-Core/src")]
    from noesis_gpu import controller, run, train
    from noesis import plasticity
    from ap1c_body import observing_body
    from ed1_contract import audited_learner
    from ps1_objective import physical_score, OBJECTIVE
    from ws1_import import checked_import

    module = train if args.phase == "train" else run
    original_body, original_scorer = module.FlyGymBatch, train.episode_scorer
    original_pad, original_import = controller.adhesion_from_command, controller.import_checkpoint
    original_learner = plasticity.MotorPlasticity
    bodies, learners, scores, pads, imports = [], [], [], [], []
    audited = audited_learner(original_learner, 0, learners)

    class CarryAudit(audited):
        def load(self, path, restore_traces=False, *, weights_only=False):
            if not weights_only or restore_traces or Path(path).resolve() != checkpoint:
                raise ValueError("EC1 uses fresh native weights-only episodes")
            before = self.net.data.copy()
            indices, indptr = self.net.indices.copy(), self.net.indptr.copy()
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["initial_weights"], self.initial_weights)
                wanted = saved["weights"].copy()
            super().load(path, restore_traces=False, weights_only=True)
            expected = before.copy()
            expected[self.edge_indices] = wanted
            np.testing.assert_array_equal(self.net.data, expected)
            np.testing.assert_array_equal(self.net.indices, indices)
            np.testing.assert_array_equal(self.net.indptr, indptr)
            if (np.any(self.activity) or np.any(self.eligibility) or self.reward_baseline != 0 or
                    self.reward_variance != 0 or self.reward_normalization_mass != 0):
                raise ValueError("Episode learning state did not reset")
            imports.append(dict(actual_import_exact=True, topology_unchanged=True,
                                nonselected_weights_unchanged=True, episode_traces_reset=True))

    def imported(net, path):
        if Path(path).resolve() != checkpoint:
            raise ValueError("Wrong frozen checkpoint")
        imports.append(checked_import(original_import, net, path))

    def scorer_factory(objective, cycle_weight=None):
        if objective != "curriculum-standing-v7" or cycle_weight is not None:
            raise ValueError("Unexpected native objective")
        native = original_scorer(objective, cycle_weight)

        def scorer(before, after, dt_ms):
            if dt_ms != 5 or len(bodies) != 1 or len(bodies[0].ap1_records) != len(scores) + 2:
                raise ValueError("Misaligned reward transition")
            body = bodies[0]
            np.testing.assert_array_equal(after["position"], body.ap1_latest_position)
            row = body.ap1_records[-1]
            result = native(before, after, dt_ms)
            components = physical_score(row["region_vertical_reaction"], row["com_velocity"],
                                        after["omega"], after["up_body"][2], after["foot_load"],
                                        body.ap1_metadata["total_mass"] * abs(body.ap1_metadata["gravity"][2]))
            scores.append(dict(control_chunk_index=np.array(len(scores)), time=row["time"].copy(),
                               native_reward=np.array(result["reward"]),
                               applied_reward=np.array(components["reward"]),
                               **{"PS1_" + key: np.array(value) for key, value in components.items()}))
            return {**result, "reward": components["reward"]}
        return scorer

    def pad(desired, rest):
        value = original_pad(desired, rest)
        pads.append((desired.copy(), (desired - rest).reshape(6, 7)[:, 3].copy(), value.copy()))
        return value

    context = dict(experiment="EC1", phase=args.phase, bank=args.bank, seed=args.seed,
                   checkpoint_sha256=sha(checkpoint), worker_sha256=sha(__file__),
                   objective=OBJECTIVE, delay_chunks=0)
    command = ["--backend", "cpu", "--seed", str(args.seed), "--noesis-source", str(root / "work/Noesis-Core/src"),
               "--data", str(root / "work/NoesisData"), "--checkpoint", str(checkpoint), "--output", str(out)]
    command += (["--seconds", "25", "--wall-seconds", "600", "--profile", "standing", "--learning-rate", ".2",
                 "--exploration-mv-ms", "1.5", "--log-exploration"] if args.phase == "train" else
                ["--worlds", "1", "--seconds", "5", "--wall-seconds", "300"])
    module.FlyGymBatch = observing_body(original_body, bodies)
    controller.adhesion_from_command = pad
    controller.import_checkpoint = imported
    if args.phase == "train":
        train.episode_scorer = scorer_factory
        plasticity.MotorPlasticity = CarryAudit
    success = False
    try:
        report = module.run(module.parser().parse_args(command))
        count = 5000 if args.phase == "train" else 1000
        if len(bodies) != 1 or len(bodies[0].ap1_records) != count + 1 or len(pads) != count or len(imports) != 1:
            raise ValueError("Incomplete native EC1 worker")
        if args.phase == "train":
            if len(learners) != 1 or len(learners[0].ed1_audit) != 4990 or len(scores) != 5000:
                raise ValueError("Incomplete audited learning")
        elif report["status"] != "completed_transport_pilot" or not report["weights_unchanged"]:
            raise ValueError("Frozen native episode failed")
        success = True
    finally:
        module.FlyGymBatch = original_body
        train.episode_scorer = original_scorer
        controller.adhesion_from_command, controller.import_checkpoint = original_pad, original_import
        plasticity.MotorPlasticity = original_learner
        if bodies and out.exists():
            body = bodies[0]
            if body.ap1_records:
                np.savez_compressed(out / "anatomy-observations.npz",
                                    **{key: np.stack([row[key] for row in body.ap1_records]) for key in body.ap1_records[0]})
            (out / "anatomy-model.json").write_text(json.dumps(body.ap1_metadata, indent=2))
            if pads:
                np.savez_compressed(out / "adhesion-observations.npz",
                                    **{key: np.stack([row[i] for row in pads]) for i, key in
                                       enumerate(("desired", "Femur_delta", "pads"))})
            if scores:
                np.savez_compressed(out / "physical-reward-accounting.npz",
                                    **{key: np.asarray([row[key] for row in scores]) for key in scores[0]})
            if learners and learners[0].ed1_audit:
                rows = learners[0].ed1_audit
                np.savez_compressed(out / "eligibility-audit.npz",
                                    **{key: np.asarray([row[key] for row in rows]) for key in rows[0]})
        context.update(worker_completed=success, import_records=imports)
        (out.parent / (out.name + "-ec1-context.json")).write_text(json.dumps(context, indent=2))
        report_path = out / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report.update(ec1_context=context, ec1_worker_completed=success, adhesion_probe_mode="legacy",
                          neural_desired_angles_drive_pads=True,
                          anatomical_measurement_not_balance_validation=True)
            if bodies:
                body = bodies[0]
                report["body"] = dict(neutral_pose=body.rest.tolist(), lower_limits=body.lo.tolist(),
                                      upper_limits=body.hi.tolist(), servo_kp=45., force_range=[-65., 65.],
                                      damping=1., noslip_iterations=0, axis_order="yaw_pitch_roll",
                                      adhesion_mapping="legacy-positive-Femur")
            if args.phase == "train":
                report.update(reward_objective=OBJECTIVE, base_reward_objective="curriculum-standing-v7",
                              eligibility_audit_rows=len(learners[0].ed1_audit) if learners else 0)
            report_path.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

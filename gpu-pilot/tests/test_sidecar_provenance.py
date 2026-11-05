import json
from types import SimpleNamespace

from noesis_gpu import train
from noesis_gpu.profile_run import SIDECAR, main
from test_cycle_scoring import COMPLETED, argv, needs_v9


def native_train(native):
    def run(args):
        args.output.mkdir(parents=True)
        report = {"status": COMPLETED, "reward_objective": "curriculum-standing-v7", "seed": args.seed}
        if native is not None:
            report["cycle_scoring"] = native
        (args.output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report
    return SimpleNamespace(parser=train.parser, run=run)


@needs_v9
def test_sidecar_takes_the_cycle_numbers_from_the_native_report_when_the_wrapper_scorer_was_unused(tmp_path):
    native = {"cycle_weight": .1, "steps": 5000, "credit_step_share": .2}
    assert main(argv(tmp_path, "--profile", "gripped-cycle"), native_train(native)) == 0
    sidecar = json.loads((tmp_path / "out" / SIDECAR).read_text(encoding="utf-8"))
    assert sidecar["cycle_scoring"]["steps"] == 5000 and sidecar["cycle_scoring_source"] == "native_report"
    assert sidecar["wrapper_scorer_steps"] == 0


@needs_v9
def test_sidecar_keeps_the_wrapper_scorer_and_labels_it_when_no_native_numbers_exist(tmp_path):
    assert main(argv(tmp_path, "--profile", "gripped-cycle"), native_train(None)) == 0
    sidecar = json.loads((tmp_path / "out" / SIDECAR).read_text(encoding="utf-8"))
    assert sidecar["cycle_scoring_source"] == "wrapper_scorer" and sidecar["cycle_scoring"]["steps"] == 0

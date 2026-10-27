"""Close the second preregistered pair after its native process has exited."""
import json
from datetime import UTC, datetime
from pathlib import Path

from noesis_gpu.evaluate import standing_gate
from noesis_gpu.run import file_digest, source_digest

root = Path(__file__).resolve().parent
outputs = root / "outputs"
protocol = json.loads((outputs / "standing-probe-10007.protocol.json").read_text(encoding="utf-8-sig"))
receipts = json.loads((outputs / "standing-probe-10007-exit.json").read_text(encoding="utf-8-sig"))
assert len(receipts) == 2 and all(r["exit_code"] == 0 for r in receipts)
assert [r["case"] for r in receipts] == ["parent", "candidate"]
for pin in protocol["package_source_pins"]:
    assert file_digest(pin["Path"]) == pin["Hash"].lower(), pin["Path"]
source = Path("C:/Users/heyma/.codex/worktrees/brain-body-feedback/Project Noesis/src")
assert source_digest(source) == protocol["noesis_source_sha256"]
pairs = [(outputs / f"standing-001-parent-{seed}", outputs / f"standing-001-candidate-{seed}")
         for seed in (10006, 10007)]
for pair in pairs:
    for bank, directory in zip(("parent", "candidate"), pair, strict=True):
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        expected = protocol[f"expected_{bank}_sha256"]
        assert report["checkpoint_sha256"] == expected
        assert file_digest(report["checkpoint"]) == expected
        assert report["noesis_source_sha256"] == protocol["noesis_source_sha256"]
result = standing_gate(pairs)
result.update(source_pin_verified_unchanged=True, checkpoint_files_verified=True,
              native_exit_codes=[r["exit_code"] for r in receipts],
              completed_at_utc=datetime.now(UTC).isoformat(),
              tests="31 passed, 1 NVIDIA-only skipped; Ruff passed")
with (outputs / "standing-two-seed-gate.json").open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, allow_nan=False)
status_path = root / "continuation-status.json"
status = json.loads(status_path.read_text(encoding="utf-8-sig"))
status.update(status="one_training_birth_and_two_frozen_pairs_complete", active_workers=False,
              app_goal_objective_updated=True, app_goal_status_observed="active",
              standing_training_gate_passed=result["standing_training_gate_passed"],
              standing_training_probe_seeds=[10006, 10007],
              last_frozen_seed=10007, two_seed_gate="outputs/standing-two-seed-gate.json",
              stage_advanced=False, source_pin_verified_unchanged=True)
status["next_steps"][0] = (
    "Standing training gate passed; port and test curriculum manager before gripped-step training"
    if result["standing_training_gate_passed"] else
    "Standing training gate failed; retain current experimental bank and test the next standing round through the curriculum manager"
)
status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
plan_path = root / "training-plan.json"
plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
plan.update(next_training_probe_seed=None, app_goal_objective_updated=True,
            app_goal_status_observed="active", standing_training_gate_passed=result["standing_training_gate_passed"],
            standing_training_probe_seeds=[10006, 10007])
plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({"standing_training_gate_passed": result["standing_training_gate_passed"],
                  "pairs": [{"seed": r["cases"][0]["seed"],
                             "parent_standing": r["cases"][0]["standing_fraction"],
                             "candidate_standing": r["cases"][1]["standing_fraction"],
                             "slip_change_percent": r["slip_change_percent"]} for r in result["pairs"]]}, indent=2))

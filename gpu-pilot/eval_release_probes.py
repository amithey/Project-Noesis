"""Reproduce and extend evaluation of Release standing probes (seeds 10006 and 10007)."""
import json
import sys
from pathlib import Path

# Add pilot src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from noesis_gpu.evaluate import compare, standing_gate, summarize

base = Path(r"c:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\gpu-pilot\outputs")
cases = {
    "parent_10006": base / "standing-001-parent-10006",
    "candidate_10006": base / "standing-001-candidate-10006",
    "parent_10007": base / "standing-001-parent-10007",
    "candidate_10007": base / "standing-001-candidate-10007",
}

results = {}
for name, path in cases.items():
    rep, summ = summarize(path)
    results[name] = summ
    print(f"=== {name} ===")
    print(f"  standing_fraction: {summ['standing_fraction']:.6f}")
    print(f"  mean_stance_slip_mm_s: {summ['mean_stance_slip_mm_s']:.6f}")
    print(f"  heading_forward_distance_mm: {summ['heading_forward_distance_mm']:.6f}")
    print(f"  net_displacement_mm: {summ['net_displacement_mm']:.6f}")
    print(f"  total_leg_steps: {summ['total_leg_steps']}")
    print(f"  mean_steps_per_leg: {summ['mean_steps_per_leg']:.2f}")
    print(f"  locomotion_regime: {summ['locomotion_regime']}")
    print(f"  tripod_detected: {summ['tripod_coordination']['tripod_gait_detected']}")
    print(f"  tripod_score: {summ['tripod_coordination']['tripod_coordination_score']:.4f}")
    for leg, ldata in summ["leg_step_cycles"].items():
        duty_str = f"{ldata['mean_duty_factor']:.2f}" if ldata['mean_duty_factor'] is not None else "None"
        print(f"    {leg}: steps={ldata['step_count']}, freq={ldata['step_frequency_hz']:.2f}Hz, duty={duty_str}, contact_frac={ldata['debounced_contact_fraction']:.2f}")

comp10006 = compare(cases["parent_10006"], cases["candidate_10006"])
comp10007 = compare(cases["parent_10007"], cases["candidate_10007"])

print("\n=== Matched Comparison Seed 10006 ===")
print(f"  standing change pp: {comp10006['standing_change_percentage_points']:.4f}")
print(f"  slip change %: {comp10006['slip_change_percent']:.4f}")
print(f"  heading forward change mm: {comp10006['heading_forward_change_mm']:.6f}")
print(f"  total steps change: {comp10006['total_steps_change']}")
print(f"  regime: {comp10006['locomotion_regime_before']} -> {comp10006['locomotion_regime_after']}")

print("\n=== Matched Comparison Seed 10007 ===")
print(f"  standing change pp: {comp10007['standing_change_percentage_points']:.4f}")
print(f"  slip change %: {comp10007['slip_change_percent']:.4f}")
print(f"  heading forward change mm: {comp10007['heading_forward_change_mm']:.6f}")
print(f"  total steps change: {comp10007['total_steps_change']}")
print(f"  regime: {comp10007['locomotion_regime_before']} -> {comp10007['locomotion_regime_after']}")

gate = standing_gate([(cases["parent_10006"], cases["candidate_10006"]), (cases["parent_10007"], cases["candidate_10007"])])
print("\n=== Standing Gate Verification ===")
print(f"  standing_training_gate_passed: {gate['standing_training_gate_passed']}")
print(f"  walking_proven: {gate['walking_proven']}")
print(f"  stage_advanced: {gate['stage_advanced']}")
print(f"  model_promoted: {gate['model_promoted']}")

# Save full output for documentation
out_path = Path(__file__).resolve().parent / "release_probes_gait_evaluation.json"
dump_data = {
    "schema_version": "gait-metrics-v1.0",
    "probes": results,
    "comparisons": {
        "10006": comp10006,
        "10007": comp10007,
    },
    "gate": gate,
}
out_path.write_text(json.dumps(dump_data, indent=2), encoding="utf-8")
print(f"\nSaved detailed evaluation to {out_path}")

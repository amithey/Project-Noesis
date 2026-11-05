"""Evaluate Stage 2 candidate (from train-11001) under matched frozen seeds 10006 and 10007."""
import json
import sys
from pathlib import Path

# Add pilot src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from noesis_gpu.evaluate import compare, standing_gate, summarize

base = Path(r"C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\gpu-pilot\outputs")

parent_10006 = base / "standing-001-parent-10006"
parent_10007 = base / "standing-001-parent-10007"
cand_10006 = base / "gripped-001-candidate-10006"
cand_10007 = base / "gripped-001-candidate-10007"

print("Evaluating Seed 10006...")
rep_p6, summ_p6 = summarize(parent_10006)
rep_c6, summ_c6 = summarize(cand_10006)
comp_10006 = compare(parent_10006, cand_10006)

print(f"10006 Parent Standing: {summ_p6['standing_fraction']:.4f}, Slip: {summ_p6['mean_stance_slip_mm_s']:.4f} mm/s, Fwd: {summ_p6['heading_forward_distance_mm']:.4f} mm")
print(f"10006 Cand   Standing: {summ_c6['standing_fraction']:.4f}, Slip: {summ_c6['mean_stance_slip_mm_s']:.4f} mm/s, Fwd: {summ_c6['heading_forward_distance_mm']:.4f} mm")
print(f"  Standing diff: {comp_10006['standing_change_percentage_points']:.4f} pp")
print(f"  Slip change:   {comp_10006['slip_change_percent']:.4f} %")
print(f"  Forward diff:  {comp_10006['heading_forward_change_mm']:.4f} mm")
print(f"  Regime:        {comp_10006['locomotion_regime_before']} -> {comp_10006['locomotion_regime_after']}")
print(f"  Tripod detected: {summ_c6['tripod_coordination']['tripod_gait_detected']}, score: {summ_c6['tripod_coordination']['tripod_coordination_score']:.4f}")

if cand_10007.exists() and (cand_10007 / "report.json").exists():
    print("\nEvaluating Seed 10007...")
    rep_p7, summ_p7 = summarize(parent_10007)
    rep_c7, summ_c7 = summarize(cand_10007)
    comp_10007 = compare(parent_10007, cand_10007)
    print(f"10007 Parent Standing: {summ_p7['standing_fraction']:.4f}, Slip: {summ_p7['mean_stance_slip_mm_s']:.4f} mm/s, Fwd: {summ_p7['heading_forward_distance_mm']:.4f} mm")
    print(f"10007 Cand   Standing: {summ_c7['standing_fraction']:.4f}, Slip: {summ_c7['mean_stance_slip_mm_s']:.4f} mm/s, Fwd: {summ_c7['heading_forward_distance_mm']:.4f} mm")
    print(f"  Standing diff: {comp_10007['standing_change_percentage_points']:.4f} pp")
    print(f"  Slip change:   {comp_10007['slip_change_percent']:.4f} %")
    print(f"  Forward diff:  {comp_10007['heading_forward_change_mm']:.4f} mm")
    print(f"  Regime:        {comp_10007['locomotion_regime_before']} -> {comp_10007['locomotion_regime_after']}")
    print(f"  Tripod detected: {summ_c7['tripod_coordination']['tripod_gait_detected']}, score: {summ_c7['tripod_coordination']['tripod_coordination_score']:.4f}")

    gate = standing_gate([(parent_10006, cand_10006), (parent_10007, cand_10007)])
    print("\nStanding Gate Result:", gate["standing_training_gate_passed"])

    out_file = base / "gripped-001-evaluation.json"
    out_file.write_text(json.dumps({
        "schema_version": "gait-metrics-v1.0",
        "cand_10006": summ_c6,
        "cand_10007": summ_c7,
        "comp_10006": comp_10006,
        "comp_10007": comp_10007,
        "gate": gate,
    }, indent=2), encoding="utf-8")
    print("Saved full gate report successfully to output folder.")

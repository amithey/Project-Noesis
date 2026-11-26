"""Recompute audited gait metrics across all probes with the horizontal projection fix."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\dev\noesis-gemini-gait\gpu-pilot\src")))
from noesis_gpu.evaluate import compare, standing_gate, summarize

base = Path(r"C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\gpu-pilot\outputs")

def audit_pair(name, parent_prefix, cand_prefix):
    p6 = base / f"{parent_prefix}-10006"
    c6 = base / f"{cand_prefix}-10006"
    p7 = base / f"{parent_prefix}-10007"
    c7 = base / f"{cand_prefix}-10007"
    
    if not (p6.exists() and c6.exists() and p7.exists() and c7.exists()):
        print(f"Skipping {name}: directories not found")
        return
    
    _, s_p6 = summarize(p6)
    _, s_c6 = summarize(c6)
    comp_6 = compare(p6, c6)

    _, s_p7 = summarize(p7)
    _, s_c7 = summarize(c7)
    comp_7 = compare(p7, c7)

    gate = standing_gate([(p6, c6), (p7, c7)])

    print(f"\n==================== {name} ====================")
    print(f"10006 Forward (Horizontal): Parent={s_p6['heading_forward_distance_mm']:.4f} mm -> Cand={s_c6['heading_forward_distance_mm']:.4f} mm (diff: {comp_6['heading_forward_change_mm']:+.4f} mm)")
    print(f"10006 Standing: {s_c6['standing_fraction']*100:.2f}%, Slip: {s_c6['mean_stance_slip_mm_s']:.4f} mm/s, Tripod: {s_c6['tripod_coordination']['tripod_coordination_score']:.4f}")
    print(f"10007 Forward (Horizontal): Parent={s_p7['heading_forward_distance_mm']:.4f} mm -> Cand={s_c7['heading_forward_distance_mm']:.4f} mm (diff: {comp_7['heading_forward_change_mm']:+.4f} mm)")
    print(f"10007 Standing: {s_c7['standing_fraction']*100:.2f}%, Slip: {s_c7['mean_stance_slip_mm_s']:.4f} mm/s, Tripod: {s_c7['tripod_coordination']['tripod_coordination_score']:.4f}")
    print(f"Standing Gate (>=90%): {gate['standing_training_gate_passed']}")

print("Auditing historical and recent candidates...")
audit_pair("Historical Release Probes (standing-001)", "standing-001-parent", "standing-001-candidate")
audit_pair("Stage 2 Birth 1 (gripped-v8-002, seed 1000)", "standing-001-parent", "gripped-v8-002-candidate")
audit_pair("Stage 2 Birth 2 (gripped-v8-003, seed 1001)", "standing-001-parent", "gripped-v8-003-candidate")
audit_pair("Stage 2 Birth 3 (gripped-v8-004, seed 1002)", "standing-001-parent", "gripped-v8-004-candidate")
audit_pair("Stage 3 Birth 1 (walking-v6-001, seed 2000)", "gripped-v8-003-candidate", "walking-v6-001-candidate")
audit_pair("Stage 3 Birth 2 (walking-v6-002, seed 2001)", "gripped-v8-003-candidate", "walking-v6-002-candidate")
audit_pair("Stage 3 Birth 3 (walking-v6-003, seed 2002)", "gripped-v8-003-candidate", "walking-v6-003-candidate")

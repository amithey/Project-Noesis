# V24: stronger measured-slip credit with preserved standing reward

V23 frozen seed10 completed5seconds without learning/exploration. Standing
92.0202%, cycles12/11/12/15/16/4, heading advance6.0807mm and reward passed,
but stance slip16.7551mm/s failed against original9.3666 (required<=7.4933).
The candidate was rejected;9cases are audited,22required cases unrun. Final
training bank2f3400da5a50e03584ba84346cac453eab093a192a5532855538e69db6029cfc
is retained, best validatednull. All V23 source/data/failures remain archived.

The new opt-in objective standing-gate-linear-slip-v5 changes only the scalar
slip coefficient:0.006->0.03per measuredmm/s (5x), before total reward clipping.
The explicit v4standing credit, support/progress, effort and obstacle terms stay
unchanged. Hard standing credit is retained because prior soft-posture/strong-slip
experiments failed to preserve stance. This is a moderate intermediate penalty,
not evidence that stronger punishment will improve physical walking.

Only embodied_metrics.py changes. Existing v3default and v4 remain compatible;
all physical measurements are exact on randomized states. Targeted tests cover
linear slip slope, standing/fall preference, real posture/contact requirements,
measured-slip preservation under total clipping, explicit v4->v5 weight import,
zero frozen learning statistics and every train/control/ablation job using the
same selected reward. The full staged suite passed338tests in70.26seconds;
Ruff passes. All31targeted installed checks also pass in4.11seconds; the installed
source hash exactly matches the fully tested staged source.

Full-connectome preflight passes158262neurons,73594selected edges/498readouts,
exact imported weights/original references/topology/motor IDs, signs and±50%
bounds, with all birth statistics reset. Max original deviation22.685582%.
Checkpoint format stays4; schema8model context records the new objective name,
so ordinary old-context loading rejects and explicit weight import is required.
Neural dynamics, anatomy, contacts, proprioception, vision and body control do
not change; the optional camera-cadence prototype is not installed or used.

Fixed recorded V23 states were re-scored independently after matching every old
reward. On the unchanged failed frozen candidate, v5mean reward would0.151144
instead of0.526444, with supported-standing mean0.262926. Raw body standing and
slip remain92.0202%/16.7551mm/s. This counterfactual is not a neural/body run or
proof of learned improvement; details/hashes are in fixed-v23-reward-rescore.json.

Prospective training: six quiet births80..85 from the retained final V23 bank,
5seconds each with training-only sustained-fall termination0.1seconds. Learning
rate0.2, eligibility100ms, bounded OU motor current1.5mV/ms, corrected reward
normalization, background3Hz x40, depressionU0, legacy proprioception. Original
connectome signs/topology/bounds stay fixed. No programmed leg order or gait
clock is added. Training/evaluation use the existing no-video mode to reduce
observer overhead; prior V23 failure videos remain available with their original
provenance. No new V24 video exists at this registration point.

Each worker is capped at3600seconds and supports exact --resume. Interrupted
cases restart from their episode beginning; partials never count as controls.
All frozen cases remain5seconds with learning/exploration/fall cutoff disabled:
validation10/11 and heldout200/201/202, matched originals, quiet/obstacle cases,
and external-sense ablation preserving joint/balance feedback. Physical standing,
cycle, forward/slip and obstacle response gates remain unchanged. Reward means
are compared only within the same objective. Early quiet rejection does not
claim unrun cases complete. Latest training stays separate from best validated.
A full frozen video of any behaviorally accepted model remains mandatory before
normal startup/goal-completion claims. No model has been promoted.

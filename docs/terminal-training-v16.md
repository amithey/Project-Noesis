# Terminal-fall training: V16

The completed V15 sensory screen rejected the conservative encoding. All six
original-weight cases and four exact old controls are preserved. V16 returns to
the legacy sensory reference and carries the last complete V14 learned weights;
no experience bank or original connectome is discarded.

New optional `--training-fall-seconds` (PowerShell `-TrainingFallSeconds`) ends a
training birth after continuously low uprightness (<0.5) or height (<0.3 mm).
The default 0 preserves full horizons. V16 uses 0.1 seconds. Cold birth time is
excluded and a recovered posture resets the timer. Reward and eligibility are
updated for the actual failure transition before saving the checkpoint. The next
birth creates fresh neural/body state and retains completed learned weights.

A terminal training fall is `status=completed, termination_reason=training_fall`,
with actual `simulated_seconds` and requested `episode_horizon_seconds` recorded.
Wall-budget, physics-error and viewer interruptions remain distinct and do not
advance the last-training checkpoint. Resume still replays interrupted episodes
from birth and the last complete weights. Short training standing fractions must
not be compared with full five-second evaluation results or presented as success.

Frozen cases force termination off, learning off, exploration zero and five
seconds. Passing a nonzero training-fall duration to evaluation is rejected before
creating the body. No neural dynamics, connectome edge, motor mapping, reward,
body parameter, leg trajectory, gait oscillator or behavioral threshold changed.

Evidence for testing this rule: the V14 audit found 11.67/29.70 simulated awake
seconds (39.293%) after a sustained fall, with no recovery in those four affected
births and continuing learning/weight changes. This measures exposure, not causal
harm or a measured wall-time saving. Whether termination improves behavior is
unproven.

Verification: staged and installed complete suites each passed 280 tests; Ruff
passed in both locations. Tests verify failure credit, weight carryover/fresh
births, transient/cold losses, low height, invalid durations, full fallen frozen
evaluation, every pipeline control's termination-off setting and distinction from
wall-budget interruption. Longer tiny-body tests exposed incomplete test-neuron
metadata; that fixture was repaired and the original failure log retained. The
initial test append had an incorrect nested path, corrected before the actual
new tests were run. Source was installed only after V15 exited and the staged
suite passed. Active source is
`a9cc082094e8234d4749c29fbd05635d0ef3c5dceec45c5b14144f6f6afaa735`.

Predeclared training: 18 fresh seeds 25..42, quiet floor, maximum five seconds per
birth, sustained-fall terminal 100 ms, latest complete V14 seed-22 weights imported
explicitly across sensory context. Learning rate 0.2, eligibility 100 ms, bounded
exploration 1.5 mV/ms, signed existing weights within 50%, normalized reward error,
U=0, neural wings and geometric adhesion remain unchanged. Each invocation has a
one-hour wall limit and strict source/settings resume guards; incomplete training
or controls never permit promotion. Videos retain failures and actual durations.

This protocol also changes seed diversity, birth count and reverts the sensory
encoding relative to V14; an improvement would not isolate termination as the
sole cause. It must still pass matched original-weight validation 10/11 and
held-out 200/201/202: >=90% standing, >=1 mm heading progress, >=2 support/lift
cycles in >=4 legs, >=20% reduced slip, the existing reward guard, obstacle
response within one second and reduced sustained contact versus external-senses
ablation retaining joint/balance feedback. All evaluations remain five seconds.
Latest-training and best-validated weights stay separate. Normal launch remains
unchanged until every required control and video review passes.

## First actual terminal birth verified

V16 seed 25 ended after 3.615 simulated seconds with termination_reason=training_fall.
The read-only runtime audit reconstructed the sustained-fall timer from every
saved awake observation: the first 100 ms threshold occurs on the final row.
The failure-transition reward is negative and is included before checkpoint save.
The next seed-26 attempt starts from exactly that completed checkpoint.

The 723 body rows contain 713 awake chunks; motor-spike duration is exactly
3,565 ms, matching the awake chunks. Original weight references, topology and
connection signs are unchanged. Maximum weight change from original is 15.289%,
mean fractional change 0.00682346; 35,241 selected edges changed since the imported
V14 checkpoint. All remain inside the 50% bound. No partial state was imported.

Observed training-only metrics over this shortened episode: standing fraction
87.38%, mean supporting feet 2.288, stance slip 16.753 mm/s, cycles LF 11 / LM 11 /
LH 9 / RF 6 / RM 6 / RH 1. These are not five-second standing or walking results:
exploration was active, the birth ended in failure and slip remains substantial.
No behavioral success is claimed and best-validated weights remain absent.

The runtime auditor checks exact source/protocol, checkpoint start/end chain,
context encoding, original references/signs/bounds, actual body/neural durations,
every terminal event and full-horizon frozen cases when they exist. It does not
edit source, brain state, checkpoints, acceptance thresholds or promotion state.
Evidence: outputs/protocol-v16-terminal-training-candidate/audit_runtime.py and
runtime-audit.json; full first episode CSV/checkpoint/video are preserved under
outputs/birth-cycle-v16 and the main checkout's young-fly-v16-progress folder.

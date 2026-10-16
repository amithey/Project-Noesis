# Terminal-fall training: V16

Current audited progress: [dated snapshot](<C:/Users/heyma/OneDrive/Desktop/מדעי המחשב/פרוייקטים/Project Noesis/outputs/young-fly-v16-progress/current-progress.md>). The sections below record the protocol and earlier training milestones.

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

## Second terminal birth verified; third learning birth running

Seed 26 ended with a training terminal fall at 2.255 simulated seconds (441 awake
chunks). The runtime audit confirms the first continuous 100 ms failure threshold
occurs on the final saved observation, preserves the full checkpoint chain and
original topology/signs/bounds, and verifies the next seed-27 start checkpoint.
The second birth's 90.476% standing fraction applies only to this shortened,
exploratory training episode; it is not a five-second standing success. Mean
stance slip remains 13.204 mm/s and RH has one physical cycle. No frozen V16
candidate evaluation has run and no best-validated checkpoint exists.

## First full five-second V16 training birth

Seed 27 completed its full five-second horizon, with 990 awake samples and
95.556% standing. The runtime audit confirms no sustained-fall terminal event
occurred, every checkpoint link matches and topology/original references/signs
remain fixed. Maximum weight change is 15.601%, within the 50% bound; 36,265
selected edges changed since the previous birth.

Physical cycles are LF 12 / LM 15 / LH 12 / RF 14 / RM 14 / RH 4. Mean supporting
feet is 2.523, mean stance slip 15.754 mm/s, mean reward +0.25518 and heading-forward
translation 4.612 mm. These are exploratory training observations with learning
and 1.5 mV/ms exploration active. Slip remains substantial, tripod score is absent
and no matched original-weight or frozen candidate comparison has been made for
this checkpoint. The full-duration standing result does not establish a learning
improvement or basic walking. The full video is preserved without truncation.

Three of 18 planned training births are complete. Seed 28 is learning from the
seed-27 checkpoint; best-validated remains absent. Source and criteria are fixed.

## Second consecutive full-horizon training birth

Seed 28 completed five seconds with 94.242% standing, 16.211 mm/s stance slip,
2.354 mean supporting feet and cycles LF 13 / LM 17 / LH 16 / RF 8 / RM 11 / RH 4.
The two full-horizon births (27/28) exceed 90% standing during exploratory
training; both retain substantial slip and neither is frozen evidence of learned
walking. Different seeds and changing weights do not isolate a learning benefit.
The runtime audit confirms the fourth completed checkpoint, its original
references/topology/signs/bounds and full-horizon termination. Seed 29 is learning
from that checkpoint. Four of 18 births are complete; no candidate has been
validated or promoted and the source/criteria remain unchanged.

## Third consecutive full-horizon training birth

Seed 29 completed five seconds with 91.919% standing and 15.358 mm/s mean stance
slip. Mean supporting feet is 2.394; physical cycles are LF 13 / LM 14 / LH 10 /
RF 8 / RM 14 / RH 2. The full-horizon training seeds 27/28/29 stand 95.556% /
94.242% / 91.919%, with substantial slip in each. Learning/exploration are active,
seeds differ and there is no matched original/frozen comparison; these observations
do not prove learned walking or a causal improvement from termination.

Before this progress snapshot was finalized, seed 30 also completed with a
terminal fall at 0.885 s. The runtime audit verifies all six completed checkpoints,
actual termination/duration, original references, topology, signs and magnitude
bounds. Maximum deviation is now 16.863%, within the 50% bound. Seed 31 is learning
from the seed-30 checkpoint. Six of 18 births are complete: three full horizons
and three terminal falls. This early new failure shows stability is not yet
consistent. Best-validated remains absent and all acceptance gates stay unchanged.
Full videos/CSVs/checkpoints, including the early failure, are retained.

## Two bounded invocations preserved; fifteen births complete

The first invocation ended normally after 3595.796 wall seconds with nine complete
births; seed 34 stopped at 0.170 s with wall_budget. Its replay completed a
terminal training fall at 0.355 s. All 34 recorded observation rows and 1,051
motor-spike events from the partial prefix match exactly; neural durations are
120/305 ms and both attempts start from the same complete seed-33 checkpoint.
This proves recorded-prefix replay, not internal weight-vector equality at the
prefix boundary or behavioral success.

The second invocation ended normally after 3597.172 wall seconds with fifteen
complete births and seed 40 partial at 2.410 s. Five births reached their full
horizon and ten ended in terminal falls. All fifteen checkpoint chains, original
references/topology/signs/bounds and actual terminal conditions passed the runtime
audit. No frozen candidate test has run and no model has been promoted.

A temporary automatic approval review failure due to usage limits prevented the
additional replay-audit write from executing. After ordinary usage was restored,
that action was approved normally and completed successfully; no approval bypass
or account reset credit was used. The exact protocol was resumed from the last
complete seed-39 weights, restarting seed 40 from birth. Its partial second-wall
evidence is preserved separately and is not counted complete. The new worker was
verified live. Source and all training/evaluation parameters remain unchanged.

The dated current-progress.md snapshot is generated from audited cases and copies
all complete artifacts with checkpoint SHA256 verification. It distinguishes full
horizons from terminal training failures and states that training is not frozen
walking evidence; chronological counts above may lag the live run.

The first attempt to add this section used an implicit Windows text encoding. It
failed on the Hebrew path and truncated this report. The previous report was
restored from its parent Git commit, this section added, and the result written
atomically as explicit UTF-8. Neural source and training checkpoints were not
modified by that documentation failure.

## Longer second-hour replay verified

The seed-40 replay completed a terminal fall at 2.915 simulated seconds. Every
recorded field in all 482 partial CSV rows and all 21,297 partial motor-spike
events matches the completed prefix exactly. Partial/full neural durations are
2,360/2,865 ms, matching awake sample counts. Both attempts start from the same
complete seed-39 checkpoint; the partial attempt remains wall_budget and is not
counted complete. No internal weight-vector equality at the prefix boundary is
asserted and this terminal training failure is not a five-second success.

The following audited snapshot contains seventeen completed births: five full
horizons and twelve terminal falls. Seed 42, the final planned training birth,
was running. All seventeen checkpoint chains and actual terminal conditions pass
the runtime audit. Frozen comparison and obstacle/held-out verification remain
required; no candidate is validated or promoted. Exact replay proofs, reports,
videos and checkpoints are preserved with the dated progress snapshot.

# V29: retain directional credit during partial slip

V28's full frozen candidate10 falls:12.22% standing versus required90%, despite
lower mean slip while collapsed. Its failure and exact interrupted-prefix replay
are retained in premotor-credit-v28.md and immutable boundary002 artifacts.
Latest training97weights remain the parent, not a validated deployment model.

Training-only diagnostics on all six V28births find positive progress credit in
1.05–3.55% of awake frames.87.17–95.39% of upright, height-valid forward-moving
samples with three loaded feet receive zero progress because slip exceeds body
speed. This is sparse scalar feedback, not proof of the cause of failed gait.
A fixed-data counterfactual retains signed progress in those eligible frames;
it cannot predict the behavior of a newly trained fly.

The opt-in objective standing-gate-smooth-grip-v6 changes only grip to
1/(1+stance_slip/(abs(heading_speed)+.1)). It retains the same posture/height,
three-foot and speed gates, standing/support/effort/obstacle terms, linear
.03per-mm/s slip cost and final reward clipping. The default v3 and old v4/v5
outputs stay exact. Physical standing, forward distance, lift cycles, slip
reduction and obstacle acceptance are unchanged. No motor action, timing,
external gait clock, prescribed leg order, topology or sign is introduced.

Full staged source digest:
b190fadaadd43ff2bf7945819125e0b65ed2d7fa0794ddfa087e812e48f747e8.
Only embodied_metrics.py differs from CF8515; installed bytes match tested
source.384full tests pass286.86s, Ruff passes. Tests include real MuJoCo body,
eye, contact and collision checks, plus small synthetic/fake-body pipelines.
The suite runs alongside frozen V28without a second full-connectome body worker;
its wall duration is not a performance benchmark. Full log SHA256:
ca5cc3a18667c29c237eda342eafdec8d95136e57026b716c588a3cd517c34b4.
Thirty installed targeted tests also pass5.52s. Coverage includes360random
old-objective golden outputs, physical/speed/support guards, reward monotonicity
outside clipping, V5-toV6weight-only circuit import, no frozen injected currents,
all25controls, retained proprio/balance in ablation and incompatible resume
rejected before report mutation. The v5choice remains in its original third
position; its test now permits later opt-in choices to be appended.

Actual full-connectome preflight passes without neural/body integration:
158262neurons,562plasticity targets including64existing CNS targets,
85390existing plastic edges. All parent weights, original references, target
selection, topology/signs and limits are exact; maximum deviation25.480914%
from original. All learning traces/baselines/normalization reset to zero;
full-network save/load is exact and frozen exploration is completely off.
Imported-reset checkpoint has SHA256f28efa79a9c80da2acc0ebeda0350d4d648daeed21e1d89f761fd589e2ccaf72;
its new model context records only the explicit scalar objective change.

Protocol registered before body results: six serial quiet births98..103from
retained V28training97/A2E5, lr.2, eligibility100ms, bounded OU1.5, circuit scope,
original sign/50% bounds, BG40,U0. Five-second training horizon, training-only
sustained-fall cutoff.1s. Validation10/11and untouched heldout200/201/202retain
all25full5s frozen cases with learning/exploration/fall cutoff off. Matched
original/trained/externally-ablated controls retain proprioception and balance.
A failed quiet candidate is rejected early without claiming skipped controls.
Each invocation is limited to3600wall seconds; partial episodes replay from
their original start. Latest training remains separate from best validated.

This is a serial continuation with new training seeds, not a causal comparison
isolating reward from every historical protocol difference. It is a learning
hypothesis. No walking result is claimed. No normal-startup update is allowed
without all behavioral gates, actual frozen video evidence and startup check.
The pilot omits recording to reduce overhead; a passing bank still needs video.

## Registered live V29 continuation

Code commit77d2ad2 installs only the tested optional scalar objective and tests
in the existing isolated worktree. No primary/normal-startup source changes.
Actual live training session46310 is verified by elevated CIM launcher16768/
interpreter36120, with no duplicate task worker. Exact CLI uses output
outputs/birth-cycle-v29, imported V28training97/A2E5, six quiet births98..103,
BG40,U0, circuit scope, normalized/bias-corrected lr.2, smooth-grip-v6,
5s horizon/.1s training fall cutoff, validation10/11, heldout200..202,
early-reject and wall3600. Native stdout and exit records are separate under
protocol-v29-smooth-grip-candidate/worker-000-*.

Producer stdout confirms seed98/learningTrue, observed through.6s with upright
.84 and reward.615. This is explored training telemetry, not a committed birth,
frozen success, or evidence of walking. All old V28negative results and raw
partial/full replay proof remain preserved; no V28candidate is accepted.

The file-only V29case auditor is prepared with an explicit worktree_root
(the inherited checkpoint is nested deeper than V28's earlier parent). It
independently reconstructs smooth grip and signed grounded progress as well as
scalar reward/normalized learning moments, original refs/signs/bounds, actual
ground support and fall timing. Only producer-committed cases will be opened;
actual start-pointer binding waits for a stopped-worker ledger. The live
ledger is not read. No V29committed birth or validated bank exists yet.

## First two V29births: failures retained, actual directional coverage audited

Seed98 ends1.355s/261awake with training_fall, standing82.758621% in the short
window, slip17.976421mm/s, reward.036048338, heading advance1.145624mm and
cycles3/4/3/2/3/2. Checkpoint9d821b19c26364dd961a4e11147da6cc8c1446c98bb93b29eeecc51901ba38ca.
7315CNSinput and32193motor-input edges changed, all64CNSaffected; max original
deviation25.462017%. Seed99ends1.505s/291awake with training_fall, standing
87.285223%, slip18.621498mm/s, reward.080024778, heading advance2.419292mm,
cycles6/5/8/4/3/0.7246CNSinput and31849motor-input edges change, all64CNSaffected,
max original deviation25.622401%. Cropped standing/cycle counts are not full
five-second success. Two completed births total2.860body seconds, zero full
horizons and two falls. Latest training99is retained separately from bestnull.

Both immutable-case audits pass original refs/topology/signs/bounds, exact
first sustained-fall termination, actual tarsal-ground support/body measures,
independent smooth grip and signed progress reconstruction, scalar reward and
normalized/bias-corrected learning moments (max scalar error2.22e-16).
All raw artifacts are copied with verified hashes. Actual starting-pointer
binding still waits for the terminal worker ledger. No live ledger was read.
The audit helper now refreshes checkpoint roles automatically after every
completed training birth, including failures, without touching production code.

Hash-verified actual training coverage finds64/261and70/291positive progress
frames (24.52% and24.05%). Applying only the old v5grip formula to those same
fixed V29traces would leave25/261and14/291positive frames (9.58%,4.81%). Every
eligible positive-forward/three-support/posture-valid frame receives current
progress credit. This confirms the intended scalar-credit change operates in
actual training, not improved learned behavior or an old-policy comparison.
No frozen or heldout data was used in the diagnostic. SourceB190, reward and
physical acceptance remain unchanged through the trial. Worker46310has begun
training100, last observed through.4s; no frozen V29result or promoted model.

## Training-only load feedback diagnostic, no active parameter change

On the two committed98/99traces, loaded pre-step tarsal forces have medians
42.185988and42.194190in model force units. The legacy SensoryMap transfer uses
load0=2, base2Hz and max150Hz. Its inferred rates reach>=149Hz in89.0244%and
91.2134% of loaded leg samples; median rate derivative is numerically zero.
Thus graded force information may be compressed into nearly binary contact
input. This does not show that binary contact causes failed gait or that a
less saturated encoding would help. Actual sensory spike trains were not
measured by this diagnostic; these are rate-curve reconstructions.

The input timing is matched correctly: each awake chunk uses the preceding
CSV row's post-step ground loads, as confirmed against learning_lab's order
(ground_contacts before sensory drive, then post-step ground_contacts before
CSV). Raw CSV hashes match the committed-case index. Only training98/99data
are used, no validation or heldout traces. Fixed-input scale2/10/20/40/80curves
are retained in load-feedback-diagnostic.json. At40the inferred median rate
is117.97–117.99Hz and slope about1.428Hz/model-force-unit, preserving a graded
range; this is a possible later hypothesis, not a chosen behavior improvement.
No sensory encoding, neural dynamics, weight bounds, reward or acceptance gate
is changed during V29. The active worker46310continues training100, observed
through1.8s upright.89. No completed third birth or frozen result is claimed.

## Third V29birth retained: support/lift cycles, terminal fall

Seed100ends3.005s/591awake with training_fall. Standing88.494078% applies only
to this shortened window; mean stance slip17.720150mm/s, reward.101892180,
heading advance4.523187mm and cycles10/10/8/6/4/4. Tripod remainsnull. Neither
forward movement nor those cycles prove walking while the body slips/falls.
Checkpoint SHA2563ceb32d5faef541991d115dacae3ac428ee9f46e143bde1525bef4426732b7a2.

The committed-case audit passes independent smooth grip/progress, scalar reward
and normalized/bias-corrected learning moments (max scalar error2.22e-16),
actual ground support/body metrics, exact sustained-fall termination, original
refs/topology/signs and50%bounds.8117CNSinput and35818motor-input edges change,
all64CNSaffected, maximum original deviation25.105438%. All raw artifacts are
copied with hashes; latest-training roles refresh automatically to seed100,
best validatednull. Actual checkpoint starting pointers await the stopped
ledger. Three completed births total5.865body seconds, zero full horizons and
three falls. No frozen V29case is complete. Worker46310continues training101,
observed through.4s/upright.92; sourceB190and all trial criteria stay unchanged.

## Fourth V29birth: terminal fall retained

Seed101ends1.565s/303awake with training_fall, standing89.768977% in the
short window, slip12.201230mm/s, reward.277536373, heading advance3.537906mm
and cycles5/5/4/3/2/1. No full-horizon pass or improvement is inferred from
these cropped/different-seed values. Four completed births total7.430body
seconds, zero full horizons and four falls.

The immutable-case audit passes independent smooth grip/progress, scalar
reward/normalized learning moments (max scalar error1.11e-16), real ground
support/body metrics, exact first sustained-fall termination, original refs/
topology/signs and50%bounds.7101CNSinput and31408motor-input edges change,
all64CNSaffected, max original deviation24.676016%. Raw artifacts are copied
with hashes and latest-training roles update to101, best validatednull.
Starting-pointer binding remains pending until worker termination. SourceB190,
reward and physical gates stay fixed. Worker46310continues102, observed through
.8s/upright.87. No frozen V29result, accepted bank or startup update exists.

Read-only helper session98042is collecting a pre-control input/runtime manifest
for the already-tested continue_young_fly helper. It has not changed the live
ledger or reused a control. The separate helper is still pending; a timeout is
not treated as process termination. Original-control reuse, if later used,
still requires a terminal rejected worker, full immutable originals and exact
source/context/input/runtime checks; failed candidate and heldout cases are
never counted as newly evaluated through reuse.

## Optional manifest scan cancelled; active training continues

Read-only helper98042remained live but produced no manifest; elevated CIM
identified its exact stdinPython parent chain17840/1808/26236 by the unique
preregistration-command token. At the last sample it used339.625CPU seconds
and112513024bytes working set. Its exact slow phase was not instrumented and
is unknown; observation timeout was never treated as process death.

The optional helper was deliberately cancelled after identity verification,
with terminal session exit-1. Full-training PIDs16768/36120were not stopped;
session46310subsequently produced further seed102telemetry. Neither a raw
manifest nor its preregistration proof exists, and no partial fingerprint is
usable. No live report/source/training parameter changed and no controls were
reused. Mandatory originals will run fresh through the existing driver.

Seed102remains active, observed through3.2s/upright.89. At2.8s uprightness.66
was followed by.89at3.0s; this explored training transient is not a learned
recovery claim or a complete standing/walking evaluation. Latest committed
training remains101, best validatednull. No fifth birth is yet committed.

## Fifth V29birth: first full horizon in this cohort

Seed102completes5s/990awake with horizon, standing90.909091%, mean stance
slip14.845907mm/s, reward.226571938, heading advance4.665878mm and support/lift
cycles17/16/13/8/10/6. Tripodnull is reported separately. This is explored
training (OU1.5), not a frozen walking result; slip has not been compared to a
matched seed102original. No promotion follows a noisy training-only threshold.
Checkpoint SHA256d83ae0777bfbe20d5b2af79f505064aa6cd790e153e9fce0e18e3115695d1245.

Independent committed-case audit passes smooth grip/signed progress, scalar
reward and normalized/bias-corrected credit moments (max scalar error2.22e-16),
actual ground support/body metrics, no sustained terminal fall and full horizon,
original references/topology/signs and50%bounds.8199CNSinput and36246motor-input
edges change, all64CNSaffected; max original deviation25.985439%. Raw artifacts
are copied with hashes; latest-training roles now point to102, bestnull.
Five completed births total12.430body seconds: one full horizon and four falls.
Actual starting-pointer ledger audit waits for termination; no live report is
read. Worker46310has begun103, observed through.2s/upright.90, learningTrue.
SourceB190, reward, body/sensory settings and physical acceptance stay fixed.

Read-only hardware inventory finds Intel i7-1165G7,4physical/8logical cores,
Intel Iris Xe graphics. A separate standard-library timing observer prototype
under primary outputs/runtime-profile-prototype passes7tests2.20s plus Ruff.
It profiles setup and an initial500ms neural prefix, forwards arguments/results
and restores methods after prefix/end/exception. Tests use tiny fake networks;
it has not been used on an actual body and real trajectory equivalence is
unproven. It changes no package source. Any later use must occur only after a
verified terminal boundary of the current worker and be explicitly registered;
no performance optimization or acceleration is claimed yet.

## Sixth V29birth complete; frozen originals start

Seed103completes5s/990awake, standing95.151515%, stance slip19.034420mm/s,
reward.166560322, heading advance13.037285mm and cycles18/13/12/10/14/4,
tripodnull. Exploration remains1.5; no frozen walking is inferred. Its high
slip is still a limitation, and heading-integrated advance alone is not gait.
The independent audit passes smooth grip/progress, reward/normalized learning
moments (max scalar error2.22e-16), real ground support/body metrics, full horizon
without sustained terminal fall, original refs/topology/signs/50%bounds and raw
artifact hash copies.8291CNSinput and36742motor-input edges change, all64CNS,
max original deviation25.270368%. Latest-training roles now point to103,
best validatednull. Six births total17.430body seconds: two full and four falls.

Worker46310has begun frozen quiet original10, observed through.4s learningFalse.
All actual start-pointer bindings remain pending until the terminal ledger.
The file-only audit_stopped_worker.py sidecar is prepared (Ruff/compile pass):
it requires prior terminal/elevated-OS verification, checks every completed
and incomplete attempt's start against the last completed training bank,
preserves incomplete raw artifacts without counting them, and snapshots the
stopped report/native exit. It has not opened the live ledger or run on a real
boundary yet. Model sourceB190and all physical/training/evaluation gates stay
unchanged. No validated candidate, video evidence or normal-startup update.

Latest completed training103checkpoint SHA256: f9f21933fbd47d00aa00c992fbec7473693fbce1e15b39de8f088ac328656036.

## First V29hour boundary verified; frozen original replay resumed

Session46310ends0/native0. Elevated CIM confirms no remaining V29Python
before the terminal ledger is read. Stopped-report SHA256:
19e8cd9b81d1f0732e001175699bfd0e95e58ef867b6aaf1435a989017e91031.
Wall duration3597.094s;6/31jobs complete, all six training births, zero frozen
controls complete. Actual start pointers for every completed/incomplete attempt
match the last completed training bank (or no bank for originals). No partial
checkpoint is used as a new training start. Latest training103SHA256:
f9f21933fbd47d00aa00c992fbec7473693fbce1e15b39de8f088ac328656036.
Best validated remainsnull. The boundary auditor passes on actual data.

Original10stops at.510s/92awake with wall_budget.102CSV rows,460neural ms and
4558motor events are preserved with raw file hashes under worker-boundary-000.
Its saved weights equal original references exactly and all learning statistics
are zero. Frozen partial checkpoint SHA256be34c92913c574e09e32d0a759ef7df66aca3deab10563fddbaae260a361ada0.
Its temporary100% standing in a short window is not a completed control.

Exact protocol/sourceB190resume is live as session50439, elevated CIM confirms
launcher30940/interpreter20892and no duplicate full-training worker. It restarts
original10from episode start, learning/exploration/fall cutoff0, no imported
bank. Native stdout/exit are separately logged as worker001. All CLI settings
remain equal, including the original V28A2E5import argument; actual candidate
evaluations use the retained V29last103bank through the ledger.

A registered timing-only observer profiles setup plus the first500ms neural
prefix of the first replayed episode, then restores the original methods.
Updated observer7tests pass1.13s/Ruff; script SHA256:
581161c0ed65dddab0422a37d9943bcff42bb700408ec8dcd9169f71a8abb9c7.
It emits a receipt after its stats files are written. No package source is
edited, but actual trajectory equivalence is still pending the immutable-case
CSV/motor-prefix comparison; software forwarding tests alone do not prove it.
No live ledger is read after resume, and no model or startup is promoted.

## First frozen V29original committed and replay equivalence proven

Original10completes5s/990awake: standing44.242424%, slip9.366594mm/s, reward
-.356606466, heading advance7.977423mm and cycles7/6/9/13/10/12. No learning,
exploration or fall cutoff. The independent audit passes original weights and
zero learning statistics, smooth grip/progress/reward reconstruction (max scalar
error2.22e-16), body/support measures and raw file hashes. This is the original
control, not trained gait. Seven protocol jobs are now committed (six births,
one frozen control); six start pointers are terminal-ledger-bound, the new
control awaits the next stopped-worker audit.

Same-source replay exactly matches all102saved CSV rows/all columns,92awake
rows,460ms/4558motor events and every checkpoint array from the prior cutoff.
Across the full1000CSV rows, every field outside reward/grip_factor/
grounded_progress exactly matches V28original10. All motor events, raw senses,
gait report, retinal PNG and original checkpoint arrays are exact. Model
contexts differ only in the explicitly changed scalar objective. Thus the
registered timing observer and new scalar reward have no observed body/neural/
sensory effect on this actual frozen original control. This is compatibility
for the measured original only, not trained walking or a general speedup proof.
Evidence is retained in original10-compatibility.json. Worker50439continues
original11, observed through2.0s; candidate validation remains pending.

The observed timing prefix includes cold setup,500neural ms and5450physical
steps. Total profiled108.166s; get_observation5451calls cumulative51.570s,
LIFkernel100calls cumulative14.719s, native mj_forward5456calls17.469s and
mj_step15450calls14.500s. These timings nest/overlap and include initialization;
they must not be added or projected as a whole-cycle benchmark. Local source
inspection finds per-step vision/adhesion/last-observation bookkeeping and
synchronized named-array reads with forward recalculation. Dropping observation
calls or physics stages is not justified; any later readout optimization must
preserve the actual data/timing and show equivalence. No such optimization is
implemented or used in V29; sourceB190and physical criteria remain unchanged.

Before seeing the trained V29validation result, an optional next-cohort plan is
registered under primary outputs/extended-training-v30-plan/plan.json: only if
V29quiet validation completes and rejects, retain last103/F9F and train30serial
quiet births104..133with the same source/learning/body/sensory/reward settings,
one-hour invocations/resume, fresh originals and all25frozen controls. No worker
is launched for that plan and no heldout data informs it. If quiet validation
passes, finish V29's remaining controls instead. This extends experience
(current V29has17.430training body seconds) before another mechanism change;
it does not promise improvement. No accepted model or startup update exists.

## Both frozen originals audited; trained candidate validation active

Original11completes5s/990awake with standing92.929293%, stance slip16.732374mm/s,
reward.195403376, heading advance12.691961mm and cycles12/15/12/6/12/3. Its
independent audit passes original weights/zero learning state, smooth progress/
grip/scalar reward (max scalar error2.22e-16), actual ground support/body metrics
and raw file hashes. All1000CSV rows outside the three explicit scalar fields,
motor events, raw senses, gait report, retinal PNG and checkpoint arrays exactly
match V28original11. Context differs only in the scalar objective. This control
is unprofiled; original10's registered timing observer was already restored.
Eight protocol jobs are now committed: six training births and two full frozen
originals;23frozen cases remain. Six prior start pointers are terminal-bound,
two current originals await the next stopped-worker ledger. No live ledger is
read. No candidate or validated walking result is inferred from originals.

Worker50439continues frozen candidate10from the retained F9Ftraining103bank.
Producer confirms learningFalse; latest observed through1.4s/upright.91 after
.65at.6s recovered to.85at.8s. This is partial telemetry only. The complete
five-second standing, foot cycles, forward advance, mean-slip and reward gates
remain mandatory; motor/CNS exploration and evaluation fall cutoff stay off.
SourceB190and all model/body/sensory/learning/acceptance settings are unchanged.
No bank or normal startup is promoted. The conditional30birth continuation is
still only a plan, not an additional running worker.

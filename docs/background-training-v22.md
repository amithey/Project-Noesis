# Prospective training at the measured lower background gain

V20 showed standing improvement at gain40 in all four fixed-weight contrasts,
but neither gain produced accepted walking. V21's earlier bank at seed66/gain40
stands94.8485% with repeated support/lift cycles in all six legs, but slip is
14.160074 versus original15.429347mm/s: only8.23% improvement. Its gain60 result
fails standing. Two seed67 comparisons are still running. No model promoted.

Prepare a six-birth quiet training protocol at gain40 using seeds68..73, with
the earlier bank7baaf2fabf0c238884f74260e2f3cfa420ce5df4880c6d47c60d62cea2f6247f.
Keep rate3Hz, learning rate0.2, eligibility100ms, bounded OU1.5, U0, v4 reward,
legacy proprioception and geometric adhesion. Training-only terminal fall0.1s;
frozen validation10/11 and heldout200..202 remain five seconds/no learning/no OU.
Obstacle and external-sense ablation controls retain joint/balance input. All
existing behavioral thresholds, signs/topology and50% weight bounds remain.
Each process is bounded to3600seconds and resumes incomplete episodes from
complete start weights. Latest training and best validated remain separate.

The staged change is only train_validate.py: expose --bg-weight (default60),
reject negative/nonfinite values and assign the value to the shared episode
template. Settings fingerprint includes the gain, so resume cannot change it.
The same template reaches training, original, candidate, obstacle and ablated
controls. No LIF, muscle, sensory, reward, connectome or gait implementation
changes. The active source remains839a0a8bd85a8ebca55048319dc95642413491a9a14c5d168fe24cc4d6e8d107;
staged5651f6dd207cd9ee7be6ea2ebcfb381cfe66d01e03422cda5a9ae70f3bf1032e.

Sixteen targeted tests pass against the actual staged source. New tests cover
all26 jobs including original/candidate/heldout/ablated controls, zero evaluation
exploration, interrupted replay at the same gain, byte-unchanged ledger on gain
mismatch and invalid gains before output creation. A source-location assertion
caught pytest initially preferring active src via its configured pythonpath;
the invocation now explicitly selects staged src. Those initial failures are
preserved in the plan, not counted as passing staged tests. Ruff passes.

Candidate files live in outputs/protocol-v22-background-training-candidate.
They are not installed or launched. All297 unit/integration tests now pass
on staged source in118.99seconds; Ruff passes across staged src and both test
sets. Complete V21 before installation and fresh bounded training. This
staging does not constitute neural/body improvement or verified walking.


The full-suite invocation explicitly selects staged src, and the new source-path
assertion passes. Logs and hashes are preserved in plan.json. This runs separately
from the active diagnostic and does not change its source, driver or parameters.
The installation gate remains completion/archival of all four V21 body cases.


## Installed after complete control archival

All four V21 cases completed and were audited before installation. The byte-
identical staged source is installed, SHA2565651f6dd207cd9ee7be6ea2ebcfb381cfe66d01e03422cda5a9ae70f3bf1032e.
All297 staged unit/integration checks and Ruff had passed; the five added
control/resume/gain checks also pass against the actual installed import path.
The new test file is tests/test_background_training.py. Source changes remain
limited to the shared CLI gain argument; normal startup and model promotion
remain unchanged. The prospective six-birth learning protocol is now ready.


## Fresh bounded training launched

After complete V21 archival, session75379 launched train_validate on a fresh
outputs/birth-cycle-v22 directory with six5s quiet births, seeds68..73, gain40
and the earlier parent7baaf2fabf0c238884f74260e2f3cfa420ce5df4880c6d47c60d62cea2f6247f.
First stdout confirms158262 entries, learning=True, seed68 and physical t=0.00s.
The worker has a3600-second absolute wall budget and records complete start
weights for replay; no completed training birth has yet been observed. The
unchanged v4 reward, rate0.2, eligibility100ms, OU1.5, terminal fall0.1s, legacy
encoding and original graph/sign/bound constraints remain. Original, candidate
and ablated controls use the same gain40. Every evaluation remains5s with no
learning/OU; validation10/11, heldout200..202, obstacle and sensory ablation all
remain required. No live ledger opened, no normal startup or model promotion.


## First complete learning birth verified

Seed68 completed five seconds, standing89.7980%, heading-relative distance
11.097045mm, cycles14/10/13/9/13/5 and mean stance slip15.934218mm/s. This is
training with learning/OU active, not frozen behavioral validation or proof of
improvement over a different seed. The second birth, seed69, is running.

The stdout-only completed-case observer verifies source565, background3Hz/40,
model context, parent original topology/references, unchanged signs and50%
weight bounds. Max original-weight deviation20.5009%, mean0.9900%;35668 edges
differ from the expected prior bank. It verifies every CSV scalar reward and
RPE/baseline/RMS learning signal, matching checkpoint trace statistics; maximum
scalar reward difference1.11e-16. It independently checks all1000 body rows,
990 awake samples,4950 neural ms, reported physical metrics and absence of the
training terminal-fall condition. These checks establish implementation and
recording consistency, not causal learning success or a policy gradient proof.

Checkpoint SHA256989f69981bdc8c6a4c3dba2db3e7e01a6eb7b3e8daba96a1bb5457d2b5629243.
All immutable completed files, including video, were copied with hashes.
The observer never opens the live report; its expected prior-bank ordering is
verified from committed receipts, while actual attempt start-pointer metadata
remains for the terminal-ledger audit. Partial/budget cases cannot be banked as
completed results. Frozen validation/heldout/obstacle gates remain outstanding;
best validated remains null and normal startup is unchanged.


## Video/motion limits and second training birth

The first complete video has1495frames at30fps. Frames at displayed physical
times1.52/3.02/4.52s were directly inspected: sampled bodies are upright and leg
configurations/headings differ. Sparse frames do not certify coordinated gait.
The first training trajectory has61.007439mm horizontal path and2.946193mm net
horizontal displacement. A separate immutable-CSV analysis compares matched
original/earlier-bank frozen cases at gain40, with training listed separately.
Standing-adjacent projected-heading angular rates remain large: original seeds
66/67 about333/329deg/s, earlier bank294/311deg/s, first new training332deg/s.
These conditional state samples and projected-heading estimates are descriptive,
not independent rigid-body rotation measurements or causal neural proof. They
show that translation/turning/slip must accompany leg-cycle reporting; no
acceptance criterion or active reward/controller was changed. Source CSV hashes,
interval definitions, script, sampled image and interpretation limits are saved.

Seed69 ended under the existing100ms sustained-fall rule at3.885body seconds,
standing90.2216% only over its shortened training window, slip16.276339mm/s.
It is a completed training birth with retained experience, not a five-second
behavioral success. Every terminal observation and scalar reward/RPE/baseline/
RMS signal matches the independent audit (reward max error1.11e-16). Topology,
original references, signs and50% bounds are preserved; max original-weight
deviation20.8956%, mean1.0060%,35242 edges differ from the expected prior bank.
Checkpoint SHA2565f6e1339845add1e2fe5749293162f75412d026981c3144c22b57c2dccf22a50.
The full short video, measurements and checkpoint are copied with hashes. Two
births are verified (one full horizon, one terminal fall); seed70 is now active.
No live ledger opened. Frozen validation and all promotion gates remain pending.

## Third birth retained after early terminal fall

Seed70 ended under the unchanged100ms sustained-fall rule at1.35s. Its86.9231%
standing is over the shortened training window, not a five-second result.
Slip11.796586mm/s and limited cycle counts cannot establish gait after this
fall. The complete short recording, checkpoint and all CSV data were preserved.
The observer verifies exact terminal timing, all scalar rewards/RPE/RMS traces,
original topology/references/signs and50% bounds; maximum scalar reward error
1.11e-16, maximum original-weight deviation20.3306%, mean1.0169%,31269 edges
differ from the expected prior. Three births are now verified: one full horizon
and two terminal falls. The fourth, seed71, is active. Different seeds and active
learning/OU prevent treating these fractions as a matched learning comparison.
No live ledger opened; validation and promotion remain outstanding.

## Fourth birth completes five seconds

Seed71 completed the full horizon, standing95.6566%, heading-relative distance
13.866417mm, cycles20/17/13/8/11/5 and mean stance slip14.526927mm/s. These
are training-with-exploration measurements, not a matched frozen improvement.
Every body row, reward/error/RMS signal and terminal condition passes the
independent observer; max reward error1.11e-16. Topology, original references,
signs and50% bounds are preserved; max original-weight deviation20.7866%,
mean1.0317%,35890 edges differ from the expected prior checkpoint.
Checkpoint SHA256d13832e1d1a5455ccaacea2f4328af2a694a68066cea8b90f465cf1639b351c8.
All complete artifacts and video are copied with hashes. Four births are now
verified: two full horizons and two terminal falls. Seed72 is active. All frozen
validation/heldout/obstacle gates remain pending; no live ledger opened and no
model promoted. Source and training configuration remain unchanged.


## Fifth birth is a short terminal fall

Seed72 ended at0.95s under the existing100ms sustained-fall rule. Its86.6667%
standing and9.890164mm/s slip describe a shortened training window, not a
five-second standing/slip success. Every terminal observation, scalar reward
and RPE/RMS trace passes the observer. Original references/topology/signs and
50% bounds remain preserved: max original-weight deviation21.3709%, mean1.0373%,
29924 edges differ from the expected prior checkpoint.
Checkpoint SHA25645a865b8fd3654a0a1efdf94b41ca09ebd2d8a5645f5900d5a4a047e48b3c8d7.
All short-run measurements/video and the complete weight bank are preserved
with hashes. Five births are verified: two full horizons, three terminal falls.
The sixth, seed73, is active; frozen matched evaluations follow all six births.
No live ledger opened, source/configuration unchanged, no model promoted.


## Six-birth training phase complete; frozen controls running

Seed73 ended under the unchanged terminal-fall rule at2.76s. Its92.8044%
standing and12.458617mm/s slip are shortened-window training measurements.
All six births are now audited/preserved: two full5s horizons, four terminal
falls, 18.945 total body seconds. No partial wall-budget episode was
banked. Max original-weight deviation in the final bank22.0710%, mean1.0588%;
33756 edges differ from the expected prior bank. Every scalar reward/RPE/RMS
trace and terminal condition matches the immutable measurements.
Final training checkpoint SHA25679ed5b2c3392ca082f8289ad8e43abbedd0162fcd01cb843e6f4987df471f418.

The worker automatically advanced to the original-weight validation control
seed10, with learning=False; all frozen cases use gain40, exploration0, full5s
and no fall cutoff. Candidate performance is not yet known. Validation10/11,
heldout200..202 and obstacle/ablation coverage remain mandatory. No candidate
promoted; normal startup unchanged.

A stopped-worker ledger auditor is prepared and Ruff passes. It will bind every
individually audited case to its actual attempt start checkpoint, verify latest
training/best-validated bank retention across cycles, independently reconstruct
complete quiet/obstacle gates, list unrun controls and preserve incomplete
attempts separately. It requires --worker-stopped and supports an explicit exact
archived source root. It has not opened the live report and will be executed
only after session/OS termination is verified. This additional audit does not
change source, training configuration, reward or controller.


## First frozen original control complete: lower-gain benefit is not universal

Original seed10 completed all five seconds at gain40 with learning/exploration
off: standing44.2424%, slip9.366594mm/s, mean reward-0.173475. Original weights,
zero learner traces, full horizon/no fall cutoff, scalar reward and physical
summary checks pass. The failure video is preserved.
Checkpoint SHA256bafea4a21f54e4611097d40dc3112909138b92f4f0e32a63c35f2a0b25b36e6e.
This is baseline behavior; the newly trained bank has not yet been evaluated.

The matched prior original seed10/gain60 control stood92.2222%, slip15.839122
and reward+0.519817. Saved weights, original references, topology, motor IDs and
parameters are exactly equal; both have zero traces. Every saved model-context
field is equal except bg_weight60/40. Lower gain improved original diagnostic
seeds66/67 but worsens original seed10 standing. Lower all-window slip accompanies
falling and cannot establish walking. This numerical-context contrast does not
prove a biological mechanism or robust parameter improvement. It is preserved
in original10-background-contrast.json; no source/reward/gate change is made.

The progress renderer now shows completed frozen controls separately from
training, and explicitly lists incomplete required coverage. Re-rendering the
already-verified control is idempotent; its checkpoint/audit records are unchanged.
Ruff passes. The live ledger remains unopened. The original seed11 control is
next; all candidate/obstacle/heldout gates remain outstanding.


## Both required original quiet controls complete

Original seed11 completed five seconds, standing92.9293%, slip16.732374mm/s
and reward0.551633. Exact original weights and zero learner traces/exploration,
all CSV/scalar reward checks and full-horizon/no-early-cutoff checks pass.
Checkpoint SHA256bafea4a21f54e4611097d40dc3112909138b92f4f0e32a63c35f2a0b25b36e6e.
Both original quiet controls10/11 are now complete; the trained candidate10
frozen case has started with learning=False. No completed candidate result yet.

The two-seed original gain contrast rechecks exact weights/references/topology,
parameters and all learner traces; all saved model-context fields match except
bg_weight60/40. Seed11 standing at60/40 is94.9495%/92.9293%, slip16.222091/
16.732374mm/s. Together with seed10 standing92.2222%/44.2424%, lower gain is
not a uniform original-weight improvement beyond diagnostic seeds66/67.
Per-seed comparisons and hashes are preserved without extrapolating a biological
mechanism or modifying the active protocol. No live ledger opened or model
promoted; all candidate/obstacle/heldout requirements remain outstanding.


## First one-hour boundary: complete controls retained, incomplete candidate replayed

The first worker exited normally after3599.25wall seconds, with six complete
training births and both original controls retained. Candidate10 reached3.545
body seconds and statuswall_budget; it is not a five-second evaluation. No
matching Python worker remained in the elevated OS inventory before ledger read.
The stopped-worker auditor passed all eight committed cases, actual attempt
start-checkpoint pointers, latest/best bank separation, original/frozen weights
and all copied artifact hashes. Report SHA256cf7d7191feb871ea30a9c00097bf5086767705c34a4cf055e21209c28645914b.
Best validated remains null;23required jobs remain. No partial result promoted.

The partial attempt0008 was preserved separately:709body rows,699awake samples,
3495neural ms and31643motor events. Frozen weights equal the complete final
training bank79ed5b2c3392ca082f8289ad8e43abbedd0162fcd01cb843e6f4987df471f418,
with zero learner traces. The first wall report/audit and all partial files have
separate immutable copies and hashes. Partial checkpoint76d8984d3b780c7051f540ff84f3ec4b427ea60471b6ff2e2ca73c1fdf528a5b
contains frozen weights/traces, not a resumable body/membrane state.

Exact --resume under source565 and unchanged settings started a fresh full
seed10 candidate episode in session32017, with another3600-second wall budget.
It loads the complete final training bank, skips all eight completed jobs and
replays from episode start. Stdout confirms learning=False; the first three
printed0.2/0.4/0.6s neural/body samples match. Full CSV/motor-event prefix
equivalence will be checked only after the replay completes. No live ledger
was opened after resume. All full-window, slip, obstacle, ablation and heldout
requirements remain unchanged; normal startup and model promotion unchanged.

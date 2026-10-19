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

# Standing-gate reward: installed for prospective brain-only training

V16 completed two batches of eighteen births while retaining learned weights.
The second frozen seed-10 candidate stood for only 45.455% of five seconds,
versus 92.222% for original weights, and remained inverted after its sustained
fall. All-window slip fell to 9.555 from 15.839 mm/s, but standing-conditioned
slip was 15.426 mm/s. Falling is not walking; the standing gate rejected it.
Reports, videos, both source snapshots and learned weights are preserved.

The earlier V16 candidate exposed a reward mismatch: its standing declined
from 92.222% to 84.444% while mean reward increased from 0.109431 to 0.146775.
Support-weighted posture could outweigh lost standing time. This observation
motivates a direct standing threshold, without claiming it caused every fall.

## Opt-in learning feedback

Use --grounded-objective standing-gate-linear-slip-v4 (PowerShell:
-GroundedObjective standing-gate-linear-slip-v4). The v3 objective remains the
default. V4 credits uprightness >= 0.7 and height 0.3..1.5 mm, with at least one
force-bearing foot; a smaller bonus rewards support. Progress still requires
three loaded feet and measured traction. The scalar reward is:

    clip(0.7*standing - 0.8*(1-standing)
         + 0.1*standing*min(supporting_feet/3,1)
         + 0.5*grounded_progress - 0.05*effort
         - 0.006*stance_slip_mm_s - 0.2*obstacle_cost, -1,1)

Motor actions continue to come from the brain and muscle model. Original
connectome topology, signed 50% weight bounds, learning rate 0.2, eligibility
100 ms and training exploration 1.5 mV/ms remain fixed. No neural dynamics,
physical mapping or sensor encoding changes accompany the new reward.

The existing schema-7 reward_objective checkpoint tag records the selected
objective. Normal loads reject changed objectives; explicit weight import
preserves weights and original references while resetting learner traces.
Default-v3 checkpoint contexts remain unchanged. Protocol resume still rejects
changed source/settings. The old V16 audit can verify its exact archived source
with --source-root outputs/protocol-v16-terminal-training-candidate/src/noesis.

## Software verification and source provenance

Fourteen offline prototype tests passed. All 292 staged tests passed in 79.46 s;
after exact-byte installation, all 292 tests passed again in 140.49 s. Ruff and
PowerShell parser checks pass. Tests cover 1,000 exact default-v3 comparisons,
unchanged non-reward measurements under v4, independent formula agreement,
standing-time ordering, checkpoint transfer/reset, unchanged tiny-body/neural
frozen observations, and all 25 frozen pipeline cases retaining five seconds,
zero exploration and zero training-fall termination. These are software checks;
full-connectome behavioral success is still unproven.

Installed source SHA256:
839a0a8bd85a8ebca55048319dc95642413491a9a14c5d168fe24cc4d6e8d107
Preserved V16 source SHA256:
a9cc082094e8234d4749c29fbd05635d0ef3c5dceec45c5b14144f6f6afaa735

Only embodied_metrics.py, learning_lab.py and train_validate.py change in source,
plus the launcher and tests. Staged code, exact installation manifest, previous
document, test logs and prospective plan remain in
outputs/protocol-v17-standing-gate-candidate.

## Prospective protocol

The preregistered transition condition is met: V16 terminated after a completed
quiet candidate failed standing. Explicitly import its final complete seed-42
training checkpoint (SHA256 8dcb55f82430c27fcc8ad3fa7477a66cc53e32a0e456b37df7ebf26de1e6c111).
Train eighteen births on seeds 43..60, max five seconds, quiet arena, v4 reward,
100 ms sustained-fall termination, normalized reward error, legacy sensory
encoding, U=0, neural wings and geometric adhesion. Each worker is capped at
3600 wall seconds; interrupted episodes replay from their complete start bank.

Frozen validation remains seeds 10/11, with held-out 200/201/202. All original
standing, forward-distance, leg-cycle, slip, obstacle/ablation and promotion
gates remain. The existing mean-reward guard also remains. Both original and
trained controls use the same new reward; reward scales across objectives are
not evidence of behavioral improvement. Tripod is reported separately, and
cycles or displacement alone do not establish walking. Ablation retains joint
and balance feedback. Reports and videos must include failures and limitations.

New training seeds and additional experience are explicit confounds for reward
causality. The protocol report is authoritative for launch/progress. Latest
training weights remain separate from best fully validated weights. Normal
startup is not pointed to an unvalidated candidate; learned walking remains
unproven until all behavioral requirements and video review pass.

## First completed birth and recorded learning-stream verification

The first v4 birth, seed 43, completed five seconds with 96.667% standing,
mean stance slip 15.650 mm/s and exploration active. This is a complete
training episode, not a frozen walking result; high slip remains unresolved.
The checkpoint retains its original connectome references, topology/signs
and bounds. It changes 36,099 incoming motor edges from the complete parent;
maximum deviation from original weights is 15.921% (mean 0.8874%).

The independent audit_reward_stream.py verifies the scalar feedback actually
recorded during this coupled run: all 1,000 rewards agree with the preserved
offline formula within 1.12e-16, and all 990 awake prediction errors, learning
signals and normalization scales agree with an independent reconstruction.
The saved final reward baseline and variance also match. Cold-frame weight
deviation matches the retained parent, supporting the explicit weight transfer.
This does not reconstruct presynaptic eligibility or prove causal credit for
each synapse, biological learning, or gait. Frozen behavioral tests remain
required. The next training birth has started.

## First bounded worker completed; interrupted seed 49 preserved

The first V17 worker ended at 3596.765 wall seconds with six complete births:
four horizons and two terminal training falls. Seed 47 completed five seconds
with standing 92.323% and slip 15.659 mm/s; seed 48 ended in a training fall
at 2.110 seconds. All six complete checkpoints and recorded scalar learning
streams pass their respective audits; this is still exploratory training.

Seed 49 was interrupted after 0.405 seconds: 81 body observations, 71 awake
updates and 355 neural ms. It remains wall_budget and never replaced the
complete seed-48 training bank. All partial artifacts and the terminal hour
report are preserved. The prior tool handle is unavailable, and a read-only
process check found no matching V17 worker before resume. Restart seed 49
from its complete start checkpoint under the same source and settings.

## Second bounded worker and verified seed-49 replay

The second bounded worker exited normally at its wall budget. Fifteen of
eighteen training births are complete: nine full five-second horizons and six
terminal falls. Seed 49 replayed from its complete seed-48 parent and completed
five seconds. All 81 body rows and 3,453 motor spike times/indices from its
interrupted 0.405-second attempt match the completed replay prefix exactly.
The complete run contains 1,000 body rows, 43,411 motor spikes and 4,950 neural
ms. This verifies recorded replay, not full membrane-vector equality or gait.

All fifteen complete checkpoint chains and scalar learning streams pass the
independent audits. Seed 58 stopped at 0.980 seconds and remains partial; it
did not replace the seed-57 training bank (SHA256
6137f46f487a9cd65037f91f98f2bd7016a94e4f501353fd01db0f70b9859756).
Its artifacts and the second wall report are preserved. Resume uses the same
source and settings, replaying seed 58 from the last complete bank. Frozen
validation has not yet completed, no best model exists, and normal startup
remains unchanged.

## Training phase complete; frozen controls started

All eighteen births (seeds 43..60) are complete: eleven five-second horizons
and seven terminal falls, totaling 64.225 body seconds. Seed 60 fell at 0.580
seconds; its failure feedback was retained before the final update. The last
training checkpoint SHA256 is
8755d9888de39337cf4286526a92ca6568d38bca8deb9e965f16cb0f5a1713fc.
All eighteen checkpoint chains and recorded scalar learning streams pass
their audits. Maximum weight deviation from original is 19.695%; mean absolute
fractional deviation is 1.145%. Original topology, signs and 50% bounds remain.

The interrupted seed-58 replay also passed exact prefix comparison: all 196
body rows and 7,854 motor events match its completed replay, from the same
complete seed-57 parent. The replay contains 1,000 body rows, 43,877 motor
events and 4,950 neural ms. As with seed 49, this is recorded prefix equality,
not equality of every membrane vector or evidence of walking.

A post-hoc reward-component diagnostic on the first sixteen complete births
shows that standing dominates seed 58 reward: mean pre-clip standing component
0.67667, extra support 0.07741, slip -0.08624, and gripped progress 0.000648.
Only 2.02% of awake samples have positive gripped progress; 93.13% have zero
grip factor. These scalar diagnostics neither isolate objective causality nor
certify gait. The contact-requiring reward standing gate is explicitly labeled
separately from the established posture/height behavioral standing fraction.
No dynamics, reward, criteria or protocol were changed in response.

The original-weight quiet control on seed 10 has started with learning off.
The trained candidate has not yet completed frozen validation, so learned
walking remains unproven, best_validated_checkpoint stays null, and normal
startup remains unchanged. Full controls and failure videos remain required.

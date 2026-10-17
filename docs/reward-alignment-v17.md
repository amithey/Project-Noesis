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

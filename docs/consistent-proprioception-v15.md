# Conservative organ-preserving proprioception: V15 development protocol

V14 was rejected after a full frozen candidate test: 45.76% standing and higher
stance slip than original weights. Its complete source, learned weights, controls,
failed video and official decision are preserved; no model was promoted.

The new opt-in `typed-consistent` encoding preserves documented organ subclass
when primary-type annotations conflict. Only primary/subclass-consistent hook and
claw cells receive the inferred directional tuning. All conflicting and unknown
rates remain exactly legacy. Actual coverage is 270 typed / 3,651 legacy among
3,921 mapped afferents; 36 conflicting annotations retain their legacy transfer.
Fifteen hooks remain without directional tuning. Signs are still inferred, not
directly measured physiological preferences. This does not solve that uncertainty.

No connectome edge, neural dynamics, body parameter, motor mapping, reward or
behavioral success threshold changed. Legacy remains default. Existing three
encodings reproduce exact rates/afferent identities across 100 independent random
six-leg states each against the V14 source archive. Tests cover organ conflicts,
load/touch/movement channels, identity/order, filtered organ ablation, missing
labels, context mismatch/explicit weight transfer, all frozen pipeline controls,
custom screen choices and immutable resume comparisons.

Verification: 269 complete unit/integration tests passed in 45.28 seconds; Ruff
and diff whitespace checks passed. Actual annotation coverage and exact old-input
compatibility are recorded in outputs/protocol-v15-consistent-candidate.

Before running: compare legacy, typed-resistance and typed-consistent on the same
development seeds 15 and 16, original weights, no learning or exploration, quiet
floor, full five-second episodes and video. Order reverses for seed 16. Reserved
validation seeds 10/11 and held-out 200/201/202 remain excluded from this screen.
Each invocation is limited to one hour; strict source/driver/settings/job checks
apply on resume, and unfinished trials restart from birth. The screen never
promotes or trains a model. All results, including failures, will be reported.

Expected control result: the legacy and typed-resistance trajectories should be
identical to the corresponding completed V13 screen cases. The new encoding may
improve or worsen stability; no behavioral success is assumed from unit tests.

Predeclared advancement to training requires all six complete cases. Compared to
both baseline encodings: new standing must be within two percentage points of the
better baseline on each seed, its worst-seed standing must exceed the better
baseline's worst-seed standing by at least five percentage points, and its mean
standing-conditioned stance slip must not exceed the lower baseline mean. Missing
standing samples, incomplete cases or nonfinite metrics fail advancement. This is
a development selection gate only. All original five-second standing/walking,
obstacle/ablation and held-out criteria are still required for any deployment.

Normal launch remains unchanged. Latest complete training and best-validated
weights remain separate, and the failed V14 learning experience is preserved.

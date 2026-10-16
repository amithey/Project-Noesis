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

## Read-only advancement gate and first control verified

The output-only analyzer `analyze_screen.py` reads the preregistered plan and
checks the exact source/driver fingerprint and six-job design. Every completed
case must be five seconds, original weights, learning/exploration off and zero
learner traces. It recomputes standing and standing-conditioned slip from all
awake CSV observations. Existing legacy/resistance controls must reproduce all
1,000 CSV rows and every motor-spike event/duration from V13 exactly.

Its separate selection gate cannot advance an incomplete screen. Seven boundary
and failure tests pass: incomplete/unverified controls, insufficient worst-seed
gain, per-seed regression, worse upright slip, missing/nonfinite metrics and
floating-point threshold boundaries. Even a passing selection result permits
training only and always retains promotion_allowed=false. These diagnostics do
not change live source, model weights, runner or behavioral acceptance gates.

The first new development case, legacy seed 15, completed five seconds with
90.303% standing and 19.156 mm/s mean stance slip (19.597 while standing). Its
weights are original and learner traces zero. All 1,000 observation rows and all
motor-spike arrays/duration match V13 exactly. This confirms the first old control
was preserved, not that the new encoding works. One of six cases is complete;
typed-resistance seed 15 is now running. Advancement remains false because the
comparison is incomplete. Evidence and seven-test log are under
outputs/protocol-v15-consistent-candidate. Copied analysis scripts in the main
output folder are reference snapshots; execute the originals in the worktree.

## Interrupted second control resumed without accepting partial data

During monitoring, the tool host timed out and the wait handle became stale.
The underlying process was checked separately: terminal session 49477 returned
exit 1 and no matching V15 screen Python worker remained. The log reached 4.0
simulated seconds of typed-resistance seed 15, but no completed CSV/checkpoint
was published. The process-termination cause is unknown; the observation timeout
alone was not treated as proof of termination.

The pre-resume report is preserved as interrupted-observation-report.json.
Attempt_0001 retains only an unfinalized partial movie and sensory-map metadata;
it is not a completed behavioral result and cannot supply a row-by-row replay
proof. The exact source/driver/settings were resumed with --resume. The report
now marks that attempt interrupted and starts typed-resistance seed 15 again
from birth, original weights, in attempt_0002. A matching worker was verified
live. The full legacy seed-15 case remains complete and is not repeated.

The source hash, preregistered plan and all six planned comparisons remain
unchanged. Advancement to training is still false; no learned weights or normal
launch were changed during the interruption.

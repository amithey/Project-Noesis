# L2: one-factor learning-rate contrast at matched25s duration

Preregistered issue18 before implementation, baseline5243aa4. T1B completed
25s and all12controls but failed retained behavior (2/4primary,1/4retention,
median gain rule false). This motivates testing a defined factor, not assuming
more identical training helps. Smaller LR may limit destructive updates; it
does not repair credit signal-to-noise and may prevent acquisition. No claim
that LR.2 caused the failure or LR.02 will cure it.

## Fixed scientific design

ONE full25s candidate birth from original COMPLETE parent9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c,
seed1000, native LR.02 versus completed25s T1B LR.2 control
ab4cf13f3a7908ae36e87826267ba8f9c13a2134562cdae1fd4d191c48db0641.
The control is not a training parent. Same v9cycleweight.1/exploration1.5,
eligibility100ms, body/senses/control/physics/neural steps, original-reference
bounds, core42c2076/f52c8d and pilot80556cd. Existing opt-in LR plumbing only;
no worker learning/core edits, default change, independent-weight merge,
external gait generator, scripted legs or brain bypass.

Before lease, re-audit actual COMPLETE T1B raw observations/reports/telemetry/
decision and D4M receipt. Require known control hash. Freeze and recheck its
JSON/NPZ receipt hashes before/after every job. Reject partial or mismatched
control; prior negative scientific outcome does not invalidate its matched role.

Candidate must complete5000chunks4990updates with the same19finite trace fields
and4990aligned exploration rows. EXACT first10prelearning trace rows vsT1B;
EXACT all6exploration arrays (drive,credit,indices,timing) across all4990rows.
Postlearning physics may differ. Candidate native report AND model sidecar must
record LR.02. All training context fields match control except explicit LR;
raw bank parameter vector permits ONLY first entry .2->.02. Topology, motor
indices, initial reference, rule, version, normalization and bias flags remain
exact. Signs/original-reference±50%bounds remain mandatory. No blanket metadata
exception; failed/partial bank never becomes curated training-last.

After complete audited acquisition:12NEW full5s frozen probes parent/control25/
child25 x10006/10007/10010/10011. Same source/packages/context, weights unchanged,
learning/exploration off. Fresh parent andcontrol25 observations must match all
prior T1B raw arrays exactly; mismatch stops as a technical finding. Final
seeds10/11/200/201/202 remain untouched. Save candidate hash before evaluation
and verify throughout. Report all paired metrics/regimes/cycles, including fails.

UNCHANGED decision: all4child probes standing>=.9, heading>=.25mm, slip<=10mm/s,
>=2cycles in>=4canonical legs. Every probe must retain standing(no decrease)
and slip(no increase) vs BOTH parent and matched25sLR.2 control. Median heading
AND net gains>=.25mm over BOTH. Scientific failure does not stop remaining
controls. N1training/four reused development probes is descriptive/exploratory,
not statistical generalization, behavioral retention across repeated training,
walking proof, validated-best or curriculum advancement. Even pass needs further
replication and sequential-retention tests; no automatic model promotion.

## Operational allocation

Explicit --guarded-mean-cpu --d4m-study COMPLETE_D4M
--low-rate-control-study COMPLETE_T1B. Original T1/T1B routes/defaults preserved.
Unique PRIMARY/gpu-pilot/outputs/learning-rate-l2-20261031. One heavy child under
shared exclusive lease and host-visible Python inventory, scoped keep-awake.
Fresh10sample windows before initial and EVERYjob: minavailableRAM>=2GiB,
meanhostCPU<=60%, record maxima/legacyT1decision; live2s telemetry immediately
stops/reaps ONLYowned child below1GiB or on unreadable telemetry/watchdog/interrupt.
No user applications closed or other full-connectome allocation.

3600s aggregate from lease,900inner/930process train,300inner/330process probes;
require fullnextprocess+5s after fresh admission. Stop first technical/refusal/
partial/mutation/mismatch/resource/budget failure. No retries/timeextensions,
same-turn fallback or automatic continuation. Preserve complete candidate if
later probes stop; generic originalT1 evaluation-resume does not support L2.
Prerequisites and focused tests precede separate allocation notice. Code/plans/
compact results in Git; evidence Releases/perfileSHA+ZIPCRC+remoteasset digest
and size verification. Exclude raw connectome/caches/envs/credentials.

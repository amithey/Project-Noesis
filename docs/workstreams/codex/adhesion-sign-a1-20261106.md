# A1 — frozen dynamic test of pad-release sign

Registered2026-11-06 after FK1 PR33. FK1 measured positiveFemur lowering the observed tarsus5 origin in126/126sampled leg/configuration combinations, while the existing rule releases for that direction. This supports an intended lift-release convention mismatch in sampled kinematics, not a proven cause of failed gait or a biologically validated adhesion model.

## Single intervention and ownership

Codex owns new scripts/adhesion_mapping.py, adhesion_ab_worker.py, run_adhesion_a1.py and tests/test_adhesion_mapping.py on codex/adhesion-sign-a1-20261106 fromFK1head0451f53. Native pilot bytes e0cbd10/core42c2076 stay unchanged. No shared bridge/controller edit, common pin/default activation or peer branch change. Learning/credit implementation remains Claude's responsibility.

LEGACY releases when neural desiredFemur-rest>.08rad. CORRECTED (geometry-consistent direction) releases when neural desiredFemur-rest<-.08rad. Threshold magnitude.08rad, per-leg dependence, neural desired-angle source and every other command/actuator/body/neural parameter remain unchanged. No clock, oscillator, leg ordering, extra current or joint command is introduced. The isolated worker binds only the controller's existing pad-function reference, records desired angles/delta/pads, and restores that reference on exit.

Both use completed experimental parent9fd7d411, read-only graph weights, no in-episode learning/exploration/external senses. Synapse signs/topology/original bounds, muscle-to-joint mapping and neuron-to-muscle activation stay fixed. A geometry-consistent mapping need not preserve behavior of a bank trained under the old convention; do not automatically adopt it or call regressions a successful improvement.

## Fixed physical design before outcomes

Eight sequential CPU one-world jobs: legacy then corrected for each reused development10006/10007/10010/10011. Full5s/1000chunks each, first50ms excluded from native metrics. No fresh or reserved final10/11/200/201/202 probes. Legacy native arrays must exactly replay ES1parent before the corresponding corrected job; any mismatch stops the allocation.

Require full horizon, finite arrays, unchanged weights/source/parent/data, exact1000x42desired and1000x6Femur-delta/pad observations. Verify legacy pads=(delta<=.08) and corrected pads=(-delta<=.08) at every call. Reports explicitly label body.adhesion_mapping; ordinary compare must not treat changed policies as identical settings. This A/B explicitly verifies ALL other matched settings/body fields and seeds. Desired angles differ from actual angles under actuator lag, and body feedback may change later neural activity.

Fixed physical improvement screen (all required):
- EACH corrected seed: standing>=.90 AND>=legacy; slip<=legacy.
- Median paired heading-forward AND net-forward gains>=.25mm; both corrected absolute forward medians positive.
- Median total cycles and tripod-coordination score not lower.
Report all native metrics/regimes and per-leg release fractions regardless of verdict. No retuning of threshold/seed/criterion and no automatic source, model or stage promotion. A failed screen can coexist with the measured static convention mismatch; it does not erase FK1 or prove which mechanism caused past training failures. A passing screen still does not prove stable fly-like gait, balance recovery, generalization or retained learning.

## Budgets, checks and evidence

1200s aggregate,300s hard worker-tree budget including startup, no new job with<300s remaining. Primary canonical lease, one heavy worker, fresh RAM>=2GiB/no competing Python. Source/parent AND all four licensed-data hashes checked before EVERY job. Suspended worker assigned to the tested Windows Job Object before resume; verify all descendants empty before releasing owned lease; retain it on uncertain cleanup. Stop on any technical/nonfinite/invariance/context/hash/resource/timeout failure, preserve partial evidence, no automatic retry.

Five focused locality/sign/boundary/invalid-input/strict-screen checks passed in.07s; worker/runner compile. Existing native watchdog is unchanged and previously passed real descendant cleanup tests. Software tests are not physical gait evidence. Final source commit and fresh allocation are recorded separately before execution.

Prepared layout: restore the three scripts from scripts/ to ROOT/work, preserve exact ROOT/work/Noesis-Sensory/core/environment/data and completed ES1outputs, then execute its Python on ROOT/work/run_adhesion_a1.py once with new outputs/adhesion-sign-a1-20261106. Compact code/protocol/results in Git; named numerical/checkpoint evidence with SHA/CRC manifest separately, excluding licensed raw graph/cache/private lease/environment. Release uploader remains unavailable under previously verified invalid CLI auth/logged-out browser; do not claim a local archive was uploaded.

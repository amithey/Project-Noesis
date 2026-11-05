# Proposed next acquisition test:25s versus5s before state reset

Proposal only; no run allocated by this document. C2 did not reproduce the C1
single-cell180ms association, so that lag must not be used to tune eligibility.
This is not evidence that no useful learning signal exists. A longer observation
window is a separate testable factor, not an assumption that more training works.

Hypothesis: keeping neural/body/eligibility/reward-baseline state continuous for
25s rather than resetting after5s gives the existing rule more within-episode
experience and yields a better retained checkpoint. The synthetic Claude study
motivates testing sample duration but does not establish a physical-fly effect.

One25s birth, same complete9fd7d411 parent, seed1000, core42c2076, LR.2,
exploration1.5, v9weight.1, body/senses/control/neural timestep unchanged.
Control is the D3 complete5s2074b1e2 child from the same parent/seed and verified
source. Require exact first1000training observation/reward/command rows versus
that control before interpreting the25s result. Do not call the5s prefix a
completed25s checkpoint. Final4990learning updates, no parent overwrite, source
and raw topology/sign/original-reference bound checks; no checkpoint averaging.

Only after full25s completion: evaluate frozen parent,5s control and25s child on
four development probes10006,10007,10010,10011 using identical current source,
5s duration, no exploration/learning. Check previous seed usage before calling
any seed new. These are development probes, not final held-out10,11,200,201,202.
Twelve evaluations provide matched behavioral retention, not statistical proof.

Prerecorded descriptive decision rule proposed for final protocol: all four
child probes satisfy existing primary standing>=.90, heading>=.25mm,slip<=10mm/s,
cycles>=2in>=4legs. Per-probe standing must not decrease and slip must not
increase versus either control. Median heading AND net progress must improve by
at least.25mm versus both controls. Report every probe/delta, tripod and failure;
no cherry-picking. This additional retention rule never weakens existing gates.
Even a pass is an exploratory candidate, not walking proof or validated-best.

Proposed compute envelope:3600s aggregate, one900s inner/930s process25s birth,
then300s inner/330s process per frozen probe with fresh>=2GiB resource checks,
existing exclusive host lease and scoped keep-awake. Stop first technical
failure/partial/mutation/prefix mismatch or inadequate remaining budget. No
automatic retry or time extension. Failure to finish all controls is incomplete
evidence, not success. No reward/sensory/calibration/eligibility change in this
test. Codex must implement/review the bounded runner and allocate exact source
hashes/unique output before execution; Claude/Gemini reviews remain coordinated.

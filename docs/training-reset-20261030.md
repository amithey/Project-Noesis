# Training review and proposed replacement method

User requested a pause for organization and a new training approach,2026-10-30.
No active training lease was present at inspection; no new run is authorized
by a historical running status. Existing failed/partial outputs remain intact.

## Observed evidence
- E3 weight.5: nine completed jobs,0/3paired heading and0/3net; no promotion.
- E4 lower exploration1.0: failed paired standing/progress; control1.5 passed
  heading but not net. Exact E2seed1000 control checkpoint reproduced.
- E5retry: nine completed jobs/844s;0/3paired heading,0/3net. Individual
  progress increases did not generalize across the two development probes.
- E6: stopped incomplete at316chunks/1.58simulated seconds and300.685s.
  The v8 arm did not run. No reward-ablation conclusion is justified.
- Pilot LearningBrain hardcodes learning_rate.2; pinned core MotorPlasticity
  default is.02. This does not show that.2 caused failure, only a testable
  hypothesis. Reused development probes do not support generalization claims.
- Provider code additions passed synthetic tests, but external sensing is not
  connected and structural checkpoint audits do not prove behavioral memory.

## Proposed method: conservative fine-tuning with a retention gate
Keep a frozen complete parent and compare two independent branches from it:
existing learning_rate.2 control versus.02 treatment. Keep reward/profile,
exploration,body,seed,duration and original-reference bounds identical.
No externally programmed gait, arbitrary weight averaging, topology/sign changes,
held-out seed use or relaxed physical acceptance criteria.

Start with one matched training seed and both reused development probes.
Require a complete matched control before interpreting treatment. In addition
to the existing standing/progress/slip/cycle gates, report per-probe changes
against the frozen parent; no apparent progress gain may hide standing loss.
Passing this small test is exploratory, not promotion or walking proof.
Do not overwrite parent or treat partial episodes as training-last. Rejected
children remain evidence; no automatic promotion of a new training parent.

## Prerequisites before a concrete run allocation
1. Codex profiles a fixed frozen baseline and repeat to explain throughput
   variability and demonstrate completion within a preregistered budget.
   Do not simply raise the wall budget or infer that sleep prevention fixed it.
2. Claude owns an opt-in learning-rate parameter with.2 default preserved,
   validates positive finite range, and tests native train/report propagation.
   Do not change reward formula in the same experiment.
3. Checkpoint auditing must distinguish expected LR metadata differences from
   forbidden topology/sign/reference/bound changes. Do not bypass all metadata
   checks or silently load a treatment checkpoint under a different LR.
4. Gemini supplies parent/child matched evaluation and validates physical
   cycle semantics; sensory development remains a separate branch.
5. Codex preregisters exact parent hash/source hashes/seed/output/budgets and
   retention criteria before training. Final validation seeds remain reserved.

This is a proposed replacement protocol, not an implemented new learning rule
or an active run. No additional training is launched during this review.

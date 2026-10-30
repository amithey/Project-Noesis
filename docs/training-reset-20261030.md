# Training review and proposed replacement method

## Current decision update, 2026-10-30

D2 completed a0.5s physical frozen prefix identical to all10D1 observation
fields. The profiled loop cost10.245s, dominated by LIF.run8.962s versus physics
mj_step0.756s; load_training_brain cost19.056s during setup. Available memory
stayed>=2.656GiB. Profiler overhead prevents a direct speedup claim; historical
stall causality remains unresolved. Compact evidence: runtime-profile-d2-result.json.

D3 is the next reliability gate: two full5s births (logging disabled/enabled)
must both reproduce historical2074b1e2 under unchanged conditions, including
every observation/reward array and explicitly aligned exploration rows. See
logging-parity-d3-20261030.md. No learning-method improvement follows merely
from reproducibility.

Claude's independent diagnosis2db4d11 supplies useful descriptive checkpoint
geometry and a synthetic credit experiment, plus optional learning/logging
plumbing. The initial LR.2vs.02 trial below is deferred as an acquisition method:
in the synthetic setup LR scales both signal and noise. It could test drift or
retention magnitude, not establish that lowering LR improves acquisition.

Before modifying reward or learning, prioritize L1: directly measure logged
exploration innovation versus subsequent reward at preregistered lags. Claude
owns the draft; Codex owns source review, physical logging parity and final
resource allocation. Preserve parent/reward/body/senses/LR/exploration in this
diagnostic. Temporal nulls must preserve autocorrelation and account for the
lag/channel search. Closed-loop association is not proof of causal credit.
Failure to detect dependence is inconclusive without adequate power, not proof
that learning is impossible.

Review limits of the initial diagnosis: parent-only sibling grouping can mix
different objectives/source/exploration; edge counts are not independent sample
counts because neural inputs/perturbations are shared; high same-seed cosine
under a tiny reward-bonus change does not remove the common base reward. Reward
spectral power alone cannot identify its causal source. Weights alone do not
prove noise, memory or forgetting. The E5 matched behavioral regressions remain
descriptive evidence on reused, parent-selected development probes.

Future L2 reward filtering and L3 consensus updates remain unallocated
hypotheses. Do not average independently trained checkpoints or weaken any
physical/invariant gate. Gemini's reported resting-sensory calibration issue
also remains a separate hypothesis pending exact source and physical controls;
76Hz at zero rest offset was already documented as a transfer-model assumption.

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

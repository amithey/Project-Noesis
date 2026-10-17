# Offline standing-gate reward prototype (not installed)

The completed V16 frozen candidate failed standing and slip gates. In seed 10,
standing declined from 92.222% to 84.444% while recorded mean reward increased
from 0.109431 to 0.146775. Exact reconstruction of every reward in all three
completed frozen trials (2,970 awake observations) identifies a mismatch:
support-weighted posture contributes 0.810265 in the original and 0.837697 in
the candidate. That increase and lower all-window slip outweigh standing loss.
This does not establish that the reward caused the late fall.

The isolated prototype directly credits the measured standing threshold
(uprightness >= 0.7, height 0.3..1.5 mm), additionally requiring at least one
force-bearing foot. Foot support is a smaller bonus. Grounded progress still
requires three feet and traction; slip, effort and obstacle costs remain.

    clip(0.7*standing - 0.8*(1-standing)
         + 0.1*standing*min(supporting_feet/3,1)
         + 0.5*grounded_progress - 0.05*effort
         - 0.006*stance_slip_mm_s - 0.2*obstacle_cost, -1,1)

Fourteen targeted tests pass: a synthetic 90%-standing two-foot trajectory
outranks an 80%-standing three-foot trajectory (the old objective reverses
this ordering), unsupported/low/tilted observations get no standing/progress
credit, measured slip/effort/contact reduce credit, useful grounded progress
increases credit, and invalid/bounded cases are checked. Ruff passes after
import/dictionary formatting fixes; the initial lint output is retained.

On identical recorded rows, prototype means are 0.519817 for original seed 10,
0.428728 for candidate seed 10 and 0.582407 for original seed 11. Absolute reward
scales differ and must not be compared as behavioral improvement. This formula
was designed after observing development seed 10; the retrospective ordering
is not independent validation or a counterfactual simulation.

The plan, formula, tests, hashes, raw logs and exact telemetry rescore are in
outputs/reward-alignment-v17-prototype. Run rescore.py only from the isolated
worktree against its original source and recorded artifacts; it verifies the
unchanged V16 source hash before reporting. The current V16 cycle remains on
source a9cc082094e8234d4749c29fbd05635d0ef3c5dceec45c5b14144f6f6afaa735.

No simulator source, dynamics, muscles, connectome, weights, learning settings,
five-second acceptance gates or normal launcher has changed for this prototype.
It is not installed or promoted. After the active cycle ends, any adoption
requires integration tests, explicit checkpoint-context versioning and a new
training protocol with full frozen controls. Learned walking remains unproven.

## Opt-in integration staged and full regression passed

The isolated integration accepts --grounded-objective standing-gate-linear-slip-v4
through the episode, train_validate and PowerShell launcher. V3 remains default.
The existing reward_objective checkpoint tag gates normal loads across objectives;
explicit import transfers weights while resetting traces. Schema stays 7 because
the existing tag already records the semantic version and default contexts are
unchanged. Resume with a different objective is rejected before any episode.

All 292 staged regression tests pass (79.46 s), Ruff passes and PowerShell syntax
is valid. Tests include 1,000 exact default-v3 physical observations versus the
archived implementation, unchanged non-reward measurements under v4, equivalence
to the independent offline formula, trajectory ordering, checkpoint transfer,
unchanged tiny-body/neural frozen observations and all 25 frozen pipeline cases
retaining zero exploration, zero training-fall termination and five seconds.
These are software checks, not full-brain walking proof.

Only embodied_metrics.py, learning_lab.py and train_validate.py differ in staged
source, plus the launcher and tests. The active V16 source is still exactly a9cc;
its preserved source archive also matches a9cc. Staged code and manifests are in
outputs/protocol-v17-standing-gate-candidate and copied for review under the
primary outputs/staged-standing-gate-v17 directory. Nothing is installed or
promoted while the active V16 cycle runs.

## Conditional prospective training protocol

conditional-launch-plan.json records the next experiment before its outcome.
Only after the active V16 worker terminates and its completed quiet validation
fails standing or slip: preserve the full cycle, install the reviewed source,
verify its hash and installed tests, then explicitly import the final complete
training weights. If quiet validation passes, finish V16 obstacle/held-out
evaluation rather than switching to the new reward.

The prospective run uses eighteen new training seeds 43..60, max five seconds
per birth, 100 ms sustained-fall termination, quiet arena and opt-in v4 reward.
Learning rate 0.2, eligibility 100 ms, exploration 1.5 mV/ms, normalized reward
error, legacy sensory encoding, U=0, neural wings and geometric adhesion remain
fixed. All original five-second frozen validation/held-out/obstacle/ablation
gates remain, with exploration and training-fall termination off. Each worker
is bounded to 3600 wall seconds, with strict source/settings resume and replay.

This is conditional preparation, not an installed or launched experiment. New
seeds and additional experience are explicit confounds for reward causality;
actual behavioral success still requires all the specified frozen evidence.

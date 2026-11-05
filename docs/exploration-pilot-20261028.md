# Bounded matched exploration pilot — 2026-10-28

User explicitly requested Codex to continue training after the integration review.
This is a new method-comparison pilot, not an additional round of the exhausted
three-round continuous-walking curriculum. No gate thresholds or core pin change.

Owner/operator: Codex, issue #18; shared resource coordination in #17. One CPU
world; sequential jobs only. Aggregate wall limit 3600 seconds, each of eight
jobs capped at 400 seconds including setup. Any failed/partial job stops the
experiment; only a completed birth produces a candidate for evaluation.

Parent: provisional experimental stage anchor (not validated-best),
`gpu-pilot/outputs/gripped-v8-003-train-1001/weights-last.npz`, SHA-256
`626bde8dce2f996ef181650ba4c3812abdc0dfdd3a969bbfc1e0e1ed57b95ff1`.
Core: `41b5fe6534ed4c8f0c21a2e123c8d518b3bf5953`; FlyGym upstream pin unchanged.

Two independent births from that exact same parent, training seed 2100, five
simulated seconds each: control exploration 1.5 mV/ms; experimental exploration
1.0 mV/ms. Both use `standing-gate-smooth-grip-v6`, learning rate .2, eligibility
100ms, original-reference bounds ±50%, initial 50ms with learning/exploration
disabled. Mutable neural state and weights are independent across births.
Shared seed provides a matched random schedule, not identical trajectories.

Evaluate the parent and both children in new frozen five-second runs on seeds
10008 and 10009, no exploration or learning, with corrected horizontal gait
metrics. These are method-pilot probes, not held-out final validation. They are
consumed for this comparison and cannot later be called unseen confirmation of
a method selected using these results. Held-out 10,11,200,201,202 remain untouched.

Primary descriptive outcomes, both seeds separately: standing fraction,
horizontal heading-forward distance, stance slip and per-leg complete cycles.
Report tripod as a secondary diagnostic. Lower exploration is promising only
if standing is >=90% in both probes, progress is positive in both, and standing
and progress do not regress against the matched 1.5 control in either probe.
Report all tradeoffs and all failed conditions; one birth per arm is not evidence
of generalization or statistical significance. No model or stage is promoted.
Continuous-walking and final gates retain their original stricter requirements.

`scripts/compare_exploration.py` acquires a shared O_EXCL lease, checks for other
Python processes, records source/parent hashes, runs bounded children and writes
receipts and comparison data. Existing leases and existing output are rejected.
An uncooperative external runner can still bypass the lease, so the live #17
allocation is also required. Interrupted leases require PID inspection before
manual recovery; never automatically remove another worker's lease.

Output: `gpu-pilot/outputs/exploration-comparison-20261028-2100` in the shared
project checkout. The parent and shared validated-best pointer remain unchanged.
Artifacts and results, including failures, will be published with SHA-256 hashes.

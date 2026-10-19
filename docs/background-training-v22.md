# Prospective training at the measured lower background gain

V20 showed standing improvement at gain40 in all four fixed-weight contrasts,
but neither gain produced accepted walking. V21's earlier bank at seed66/gain40
stands94.8485% with repeated support/lift cycles in all six legs, but slip is
14.160074 versus original15.429347mm/s: only8.23% improvement. Its gain60 result
fails standing. Two seed67 comparisons are still running. No model promoted.

Prepare a six-birth quiet training protocol at gain40 using seeds68..73, with
the earlier bank7baaf2fabf0c238884f74260e2f3cfa420ce5df4880c6d47c60d62cea2f6247f.
Keep rate3Hz, learning rate0.2, eligibility100ms, bounded OU1.5, U0, v4 reward,
legacy proprioception and geometric adhesion. Training-only terminal fall0.1s;
frozen validation10/11 and heldout200..202 remain five seconds/no learning/no OU.
Obstacle and external-sense ablation controls retain joint/balance input. All
existing behavioral thresholds, signs/topology and50% weight bounds remain.
Each process is bounded to3600seconds and resumes incomplete episodes from
complete start weights. Latest training and best validated remain separate.

The staged change is only train_validate.py: expose --bg-weight (default60),
reject negative/nonfinite values and assign the value to the shared episode
template. Settings fingerprint includes the gain, so resume cannot change it.
The same template reaches training, original, candidate, obstacle and ablated
controls. No LIF, muscle, sensory, reward, connectome or gait implementation
changes. The active source remains839a0a8bd85a8ebca55048319dc95642413491a9a14c5d168fe24cc4d6e8d107;
staged5651f6dd207cd9ee7be6ea2ebcfb381cfe66d01e03422cda5a9ae70f3bf1032e.

Sixteen targeted tests pass against the actual staged source. New tests cover
all26 jobs including original/candidate/heldout/ablated controls, zero evaluation
exploration, interrupted replay at the same gain, byte-unchanged ledger on gain
mismatch and invalid gains before output creation. A source-location assertion
caught pytest initially preferring active src via its configured pythonpath;
the invocation now explicitly selects staged src. Those initial failures are
preserved in the plan, not counted as passing staged tests. Ruff passes.

Candidate files live in outputs/protocol-v22-background-training-candidate.
They are not installed or launched. All297 unit/integration tests now pass
on staged source in118.99seconds; Ruff passes across staged src and both test
sets. Complete V21 before installation and fresh bounded training. This
staging does not constitute neural/body improvement or verified walking.


The full-suite invocation explicitly selects staged src, and the new source-path
assertion passes. Logs and hashes are preserved in plan.json. This runs separately
from the active diagnostic and does not change its source, driver or parameters.
The installation gate remains completion/archival of all four V21 body cases.


## Installed after complete control archival

All four V21 cases completed and were audited before installation. The byte-
identical staged source is installed, SHA2565651f6dd207cd9ee7be6ea2ebcfb381cfe66d01e03422cda5a9ae70f3bf1032e.
All297 staged unit/integration checks and Ruff had passed; the five added
control/resume/gain checks also pass against the actual installed import path.
The new test file is tests/test_background_training.py. Source changes remain
limited to the shared CLI gain argument; normal startup and model promotion
remain unchanged. The prospective six-birth learning protocol is now ready.

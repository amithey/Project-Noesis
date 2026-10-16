# Post-fall learning exposure in the completed V14 births

Read-only audit of all six completed five-second training births, seeds 17..22.
No simulation, checkpoint, reward, brain dynamics or active V15 source changed.
The hypothetical training terminal condition is uprightness below 0.5 or height
below 0.3 mm continuously for 100 ms after the brain wakes. The threshold is a
candidate training-episode rule, not a behavioral success criterion.

| Seed | First sustained fall (s) | Remaining awake time (s) | Fallen fraction afterward |
| --- | ---: | ---: | ---: |
| 17 | Not reached | 0 | n/a |
| 18 | 2.505 | 2.495 | 100% |
| 19 | Not reached | 0 | n/a |
| 20 | 0.865 | 4.135 | 100% |
| 21 | 2.710 | 2.290 | 100% |
| 22 | 2.250 | 2.750 | 100% |

Of 29.70 simulated awake training seconds, 11.67 seconds (39.293%) occurred after
this condition. All four affected births remained fallen for the rest of their
observed episodes. Learning signals continued and the mean absolute weight
deviation from original increased after the threshold in each case. This scalar
metric is not total weight-update magnitude or proof that those updates caused
the later behavioral failure. Simulated-time exposure is not a measured wall-time
saving, and lack of recovery in these six births does not establish impossibility
of recovery in another case.

Candidate next training change: make sustained falls terminal for training births
only, saving learned weights and starting the next seed/body/neural birth. The
learner would still receive the failure-transition reward before termination.
This resets a failed training environment; it does not inject a gait, fix body
posture during an episode or alter the connectome. At the time of this audit the implementation and tests were not present. V15
subsequently completed; see terminal-training-v16.md for the tested implementation.

If implemented, evaluation must always retain its complete five-second window,
learning/exploration off, all standing/cycle/heading/slip/obstacle gates and the
reserved validation/held-out seeds. The default should preserve existing behavior.
A training terminal fall must be reported explicitly and must not be conflated
with a wall-budget interruption or an incomplete control. Interrupted episodes
still replay from their start and the last complete checkpoint.

Evidence and reproducible audit: outputs/post-fall-learning-audit/analyze.py and
report.json. Latest complete V14 training weights, rejected candidate and exact
source remain preserved. No model has been promoted.

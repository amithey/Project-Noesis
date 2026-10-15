# Frozen standing/walking evaluation in progress

The requested goal is not complete. The corrected-body v6 ten-second training
episode completed and its weights were saved. Body uprightness/height criteria
held for 93.62% of training samples; heading-relative motion was 2.7803 mm and
all six legs had repeated physical stance/lift cycles. Mean stance slip remained
6.5476 mm/s. Exploration was active, so these figures do not prove frozen walking.

The first five-second original-weight frozen validation control (seed 10) completed.
It had standing fraction 0.8333, heading-relative motion 2.7147 mm and slip
5.4259 mm/s. It fails the 90% standing criterion. The original-seed-11 control and
subsequent trained-weight controls remain in the live official protocol. Trained
and original weights must be compared with the same seed and exploration off;
training-seed-0 numbers cannot establish improvement over validation-seed-10.

A separate five-second trained-seed-10 frozen diagnostic is running with one Numba
thread, the same candidate checkpoint and the original physics stepping path.
It will not promote a model or substitute for the matched validation/held-out gate.
The goal remains verified standing and basic walking, not merely moving joints.

## Runtime investigations

A 0.3-second profile attributed substantial cost to reconstructing the full FlyGym
observation at every 0.1 ms physics step. A prototype reading the full observation
only at the existing 5 ms brain/control interval first changed the trajectory: it
had omitted dm_control's per-step lazy forward-dynamics synchronization. That
prototype was rejected. Restoring the original forward synchronization made every
recorded CSV column exactly equal, but did not establish a speed gain (45.468 vs
43.0 wall seconds in these short runs), so the live stepping code was not changed.

One-thread retinal execution with the original stepping path also matched every
recorded CSV column, at 35.578 wall seconds. This is a short cold-start measurement,
not a reliable steady-state speedup estimate. The separate candidate diagnostic
uses one thread; the live protocol retains its original environment. No timestep,
physics, sensory pathway, learning rate or behavioral acceptance criterion was
weakened. The full physical protocol remains at five-second frozen trials.

Machine-readable snapshot: `docs/training-progress-v6.json`. Live authoritative
state: `outputs/birth-cycle-v6/report.json`; check the actual process/session before
resuming. The ongoing work uses the unchanged source fingerprint from commit
`3792528`, and the test result remains 165 passed plus Ruff.

# Anatomical motor-direction audit

The walking goal remains unverified. The v7 cycle completed and was rejected. The separately tested correction
has now been applied to the isolated worktree after confirming that run exited.

## Physical failure diagnosis

Completed five-second frozen quiet seed 10: original slip 5.425919 mm/s versus
candidate 6.411587 mm/s; original three-or-more leg support fraction 0.227273
versus candidate 0.165657. RF carried ground load in only 3.232% of original
samples and 0.909% of candidate samples. In the candidate, RF was physically
clear of the ground without support for 99.091% of awake samples. Body uprightness
improved, but that does not demonstrate useful leg support or low-slip walking.
These values come from complete recorded physical traces, not video estimates.
The video's 0.1x overlay means its approximately 50 seconds cover five simulated
seconds. A true midpoint frame was extracted and inspected.

## Direction error

At the physical standing pose, increasing the `Femur` joint coordinate increases
the interior Coxa-Femur-Tibia angle in every leg (derivative approximately +1).
Increasing `Tibia` decreases the Femur-Tibia-Tarsus angle (approximately -1).
The old motor table treats both positive coordinates as flexion, so the five
proximal groups are reversed: trochanter flexor, accessory trochanter flexor,
trochanter extensor, tergotrochanter and sternotrochanter.

The anatomical roles are independently specified in Azevedo et al.'s
[primary supplementary methods, Table A1 and Figures A5-A9](https://faculty.washington.edu/tuthill/docs/azevedo24_appendix.pdf).
The two flexor groups flex the coxa-trochanter joint; the three extensor groups
extend it. The correction changes only their joint-coordinate signs. It preserves
connectome topology, synaptic signs/weights, the existing mirrored right-leg roll
and yaw signs, learning dynamics, sensors, adhesion and every behavioral criterion.
It adds no gait sequence or oscillator. Biarticular actions beyond the current
one-joint muscle approximation remain a modeling limitation.

## Verification and deployment hold

The anatomical regression activates each named muscle pool and measures resulting
physical segment angles in all six legs through BOTH motor matrix APIs and the
muscle model. Before correction: five proximal cases fail and two tibial controls
pass. In the separate corrected source snapshot: all 16 motor/geometry tests pass,
and the full suite plus the new anatomical regression passes **197 tests**. Ruff
passes for the two changed files and the new test. Pytest's default `pythonpath=src`
must be explicitly overridden when testing the source snapshot; the authoritative
successful log is `outputs/contact-slip-audit-v7/full-regression-candidate.log`.

Only `motor.py` and the `learning_lab.py` motor-interface tag differ from the
archived current source. The new tag is `anatomical-flexion-v2`; normal checkpoint
loading must reject the old body context, while explicit weights-only import can
carry prior learning with reset traces. A new physical protocol must recompute its
original controls; v7 cached controls are not reusable across this correction.

No behavioral improvement is claimed from geometry tests. The v7 cycle is terminal; run a new protocol with fresh frozen physical controls
and brain-only training under the corrected motor interface. Preserve both earlier weights and failure videos, and keep
the regular launcher on the prior verified state until the full gate passes.

Machine-readable diagnostics and exact before/after test logs are in
`outputs/contact-slip-audit-v7`; the original source is preserved in
`outputs/protocol-v7-source/src/noesis`.


## Completed v7 cycle 1

Training seed 1 completed ten simulated seconds and saved its checkpoint.
The full five-second frozen seed-10 candidate was rejected: standing fraction
0.895960, heading-relative progress -0.866765 mm, stance slip 6.583347 mm/s.
All six legs had at least two physical stance/lift cycles (LF=12, LM=11, LH=15,
RF=7, RM=8, RH=22), but cycles alone do not demonstrate forward walking.
Standing, heading progress, slip and reward gates failed. Remaining controls
were not run and no model was promoted. The source snapshot and all completed
training/evaluation artifacts are preserved.

The corrected protocol has started with explicit import of only the last completed
v7 weights, reset learning traces, and training seed 2. Its original controls are newly
measured; previous-body controls cannot support the new body's acceptance decision.
Expected result before physical testing: named muscles move their anatomical
joints in the correct direction; stable posture, useful support and reduced slip
are hypotheses requiring the matched frozen trials, not conclusions of the tests.


## Active corrected protocol

`outputs/birth-cycle-v8/report.json` is authoritative. It starts from
`birth-cycle-v7/cycle_001/train-0/attempt_0005/episode_000.npz` through explicit
weights-only transfer. Source fingerprint:
`e781c5f9e9c00d318594f939922679b36f75897b3d53cdb574aaea9638957a66`.
Ten-second training episodes, five-second frozen evaluations, validation seeds
10/11, heldout seeds 200/201/202, normalized reward error, original LR 0.2,
100 ms eligibility and 1.5 mV/ms training exploration remain unchanged.
Both the separately staged source and the installed source passed all 197 tests;
Ruff passed on src, tests and scripts. No model is promoted.

Only resume after the actual process/session is terminal, using the matching
arguments (including the original import path and training seed 2):

```powershell
$env:PYTHONPATH='src'
python -u -m noesis.experiments.train_validate --data <existing-data> --output outputs/birth-cycle-v8 --episodes 1 --seconds 10 --seed 2 --wall-seconds 3600 --import-weights outputs/birth-cycle-v7/cycle_001/train-0/attempt_0005/episode_000.npz --normalize-reward-error --early-reject --video --resume
```

The normal launcher in the primary checkout has not been changed.

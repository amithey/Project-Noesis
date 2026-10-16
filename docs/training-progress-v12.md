# V12 standing-first training progress

Status: terminal rejected_validation. Both original controls and the frozen
candidate seed-10 trial completed; candidate standing and reward failed.
This is learning-on data, not a frozen acceptance result or deployment.

Seed 12 imported the complete final v11 training weights under the explicit
weights-only path, with fresh neural/body state and reset traces. The source
fingerprint remains 6f6b31dbe85d7ff21e00cdb5c6fbe10aadc417d8d3d45316fdff9107e88b5ef3.
Learning rate 0.2, eligibility 100 ms and motor exploration 1.5 mV/ms are unchanged.

The episode completed five seconds in 532.84 wall seconds: standing 91.72%,
heading-integrated motion 15.247 mm, stance slip 17.215 mm/s, mean reward +0.13511.
It fell near the end. The complete video/CSV retain that failure. Measured cycles
were LF14 LM13 LH13 RF12 RM11 RH5; tripod score remains undefined. Neither cycles
nor translation establishes coordinated walking. Reward is on a different scale
from v11, and the training seed differs, so these numbers cannot demonstrate a
causal improvement over the previous protocol. Frozen matched controls are pending.

The complete checkpoint was audited against imported weights: original topology,
motor identities, initial magnitudes, signs and the 50% bounds are preserved.
Numeric weight updates and physical support fractions are saved in the companion
JSON. No external joint trajectory, oscillator or leg sequence was introduced.
Best-validated weights remain null; the primary default model is unchanged.

Artifacts: outputs/birth-cycle-v12/cycle_000/train-0/attempt_0000 and
outputs/training-audit-v12/first-completed-training.json. A complete first-episode
copy and reproducible source are preserved in primary outputs/young-fly-v12-progress.

## Second completed birth: seed 13

Seed 13 completed five seconds in 337.84 wall seconds. Standing fraction was
18.48%, mean stance slip 7.951 mm/s, heading-integrated translation 11.303 mm and
mean reward -0.44426 under the new objective. It fell early and did not recover.
The low all-window slip reflects long fallen intervals and does not demonstrate
walking. The full video/CSV and a frame near three simulated seconds preserve
this failure. Together with the first birth's late fall, it shows that the changed
objective has not established consistent training stability. Different seeds and
evolving weights prevent a causal comparison of those two trials.

The complete second checkpoint passed sign and 50% bound checks and is now used
for seed 14. Frozen validation and held-out tests have not started. No model has
been promoted and the primary launch remains unchanged.

## Third completed birth and first frozen original control

Seed 14 completed five seconds with 91.92% standing and 16.185 mm/s mean stance
slip. The full trained checkpoint preserves signs and the 50% magnitude bounds.
Across training seeds 12/13/14, standing fractions were 91.72%, 18.48%, 91.92%;
this is inconsistent training stability, not matched evidence of learning.

Original seed 10 completed its frozen five-second control. Every field of all
1,000 CSV rows except the intentionally changed reward matches the V11 control
exactly; all motor-spike arrays also match. The reward change therefore leaves
this original brain/body trajectory unchanged. Seed 11 and trained candidates
are still pending, and no validated checkpoint exists.

## Final frozen failure

The five-second candidate seed-10 trial used learning=False and exploration=0.
It stood for only 6.67% versus original 92.22%, fell early and did not recover.
Heading-integrated translation was 7.586 mm and all-window stance slip 6.089 mm/s.
Those falling-motion metrics do not establish walking. The gate correctly failed
standing and reward and promoted no checkpoint. Candidate seed 11, obstacle and
held-out cases were never started and are not claimed complete.

The complete final failure video and CSV are preserved. After normal process exit
the exact 6f6b31db... source, tests and scripts were archived in
outputs/protocol-v12-standing-first-source. The original training experience is
still saved in train-2/attempt_0002 for a future explicit weights-only import.
The next work is a development-only sensory comparison on seeds 15/16, without
training or promotion. The normal primary-checkout launch remains unchanged.

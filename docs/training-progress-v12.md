# V12 standing-first training progress

Status: first five-second quiet-floor birth completed; second birth is starting.
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

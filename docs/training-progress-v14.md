# Brain-only young-fly training: V14 progress

Implementation is preserved in the isolated `codex/brain-body-feedback` worktree.
The active source hash is
`738bd3a8130fc9a37eeb4b2a1351735006232e65ca1c74e60dbbac630422477b`.

The implementation includes tarsal-ground-only support, geometric release in
response to muscle commands, support/slip/obstacle reward, motor-perturbation
learning (rate 0.2, eligibility 100 ms, exploration 1.5 mV/ms), signed existing
connections bounded to 50% of their original magnitude, and separate last-training
and best-validated checkpoints. Frozen evaluations disable learning and motor
exploration. External-senses ablation retains joint and balance feedback.
No external gait oscillator or programmed leg sequence was added.

Excitatory-only depression U=0.03/0.1, recovery 300 ms, failed its earlier neural
screen; U=0 remains active. The development-only sensory screen completed all six
cases, including an exact replay of the interrupted 4.9-second case. The inferred
resistance mapping was selected by worst-seed standing. Its biological direction
assignments and 36 annotation conflicts remain experimental assumptions. See
`proprioception-screen-v13.md` for all six results and failures.

V14 carries the last complete V12 training weights into six quiet-floor births,
seeds 17..22, five simulated seconds each. Birth resets body and neural state;
only complete learned weights carry forward. Each invocation is bounded to one
hour. `--resume` preserves completed jobs and restarts an interrupted episode
from its beginning and the last complete training checkpoint.

First completed birth, seed 17:

| Measurement | Result |
| --- | ---: |
| Standing fraction | 89.49% |
| Mean stance slip | 16.190 mm/s |
| Heading-forward translation | 6.197 mm |
| Physical cycles LF / LM / LH / RF / RM / RH | 16 / 21 / 12 / 10 / 10 / 5 |
| Mean supporting feet | 2.520 |
| Mean reward | +0.17382 |
| Largest weight change from original | 11.852% |

The completed-checkpoint audit confirms identical topology, motor-edge indices,
original weight references and connection signs; all 73,594 selected connections
remain within the 50% bound. 37,559 connections changed during this birth.
Training exploration was active. This is not proof of learned walking: standing
is below 90%, slip is substantial, tripod score is unavailable and matched frozen
evaluations have not yet run for V14. No candidate has been promoted.

The unchanged acceptance protocol reserves validation seeds 10/11 and held-out
seeds 200/201/202. Every evaluation lasts five seconds. It requires standing >=90%,
heading progress >=1 mm, at least two stance/lift cycles in four legs and >=20%
less slip than original weights. Obstacle response within one second and reduced
sustained contact versus external-senses ablation are also mandatory. Failed or
incomplete candidates cannot replace best-validated weights. Ordinary launch
remains unchanged pending all controls and video review.

Installed-source verification: 257 tests passed in 52.47 seconds and Ruff passed.
Logs are preserved with the source snapshot. Completed training video, CSV,
checkpoint and the read-only `audit_progress.py`/`progress.json` are in
`outputs/birth-cycle-v14`; user-accessible copies are under the main checkout's
`outputs/young-fly-v14-progress`. Live `report.json` is authoritative; copied
reports are dated snapshots and may lag the ongoing run.

## Ground-support diagnostic for the completed first birth

The CSV reproduces the reported 990 awake samples, standing fraction and mean
stance slip exactly. Standing with at least three loaded feet occupies 45.66%
of the measurement window; standing without any loaded feet occupies 1.72%.
Mean slip conditioned on standing is 15.901 mm/s, so substantial slip is present
while upright, not only during a fall. Stance fractions are LF 55.86%, LM 36.87%,
LH 58.18%, RF 23.43%, RM 65.25%, RH 12.42%; support is visibly uneven.
The posture trace also shows an abrupt loss of uprightness at the end of the
five-second episode. The earlier 4.02-second frame does not show that later
failure. The full video and support plot retain it. These are diagnostic
observations, not new acceptance criteria or evidence of an improvement over
matched original weights. `diagnose_support.py` regenerates the plot and JSON
without modifying the brain, reward, checkpoint or behavioral gates.

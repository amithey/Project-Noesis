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

## Interrupted second birth and strict continuation

The original terminal handle disappeared and a fresh process inventory confirmed
that no V14 train_validate worker remained. Its second-birth log reached 4.8 s,
but no completed CSV or checkpoint was published; only a partial movie and
mapping metadata remain in attempt_0001. The cause of process termination was
not established. This is not a completed five-second trial or a wall-budget
completion. The existing report retained the first complete birth and weights.

The exact source/settings were resumed with --resume and a new one-hour limit.
The report now marks attempt_0001 interrupted and starts train-1 again from the
completed train-0 checkpoint, in a fresh attempt directory. A matching worker
was verified live. No partial training weights were imported; no candidate was
promoted. Live report/log state, rather than this timestamped note, determines
subsequent progress.

## Five complete training births

The resumed seed-18 episode completed five seconds. Its first logged checkpoints
reproduced the interrupted trajectory; no partial CSV existed for a row-by-row
replay proof, so none is claimed. Seeds 19, 20 and 21 also completed.

| Seed | Standing | All-window slip mm/s | Slip while standing mm/s | Final uprightness |
| --- | ---: | ---: | ---: | ---: |
| 17 | 89.49% | 16.190 | 15.901 | 0.143 |
| 18 | 41.41% | 10.933 | 15.515 | -0.391 |
| 19 | 94.04% | 16.625 | 16.445 | 0.883 |
| 20 | 12.93% | 8.383 | 22.157 | -0.417 |
| 21 | 46.26% | 8.894 | 12.990 | -0.324 |

Only seed 19 meets the standing threshold in these training observations. This
is not a matched comparison or frozen evaluation: seeds differ, weights change
sequentially and motor exploration is active. The lower all-window slip in failed
births is misleading; seed 20's standing-conditioned slip is 22.157 mm/s.

Across these five births, right-front stance occupies only 11.6..26.1% of upright
samples, versus 44.6..88.3% for left-front. This is evidence of uneven support,
not a demonstrated cause or justification to change anatomical signs. The current
training series and source remain unchanged pending frozen validation.

Every complete checkpoint preserves original topology, motor-edge indices,
reference weights and signs. The latest complete seed-21 weights have mean
fractional change 0.0064424 and maximum 0.1260464 from original, inside the 0.5
bound. The sixth birth (seed 22) is running. Full per-leg/window measurements and
reproducible read-only audit are in `support-across-births.json` and
`audit_support_across_births.py`; no candidate is best-validated.

## Six training births completed; frozen control running

Seed 22 completed five seconds with 38.79% standing, 9.073 mm/s all-window
stance slip and 15.795 mm/s slip conditioned on standing. Its final uprightness
is -0.967: it did not recover from the fall. Physical cycle counts are
LF 5 / LM 10 / LH 7 / RF 8 / RM 10 / RH 8. Those cycles and the 9.319 mm heading
translation are not proof of walking because most of this birth was fallen.

All six complete checkpoints pass topology, motor-edge identity, original-weight
reference, sign and 50% bound audits. The latest seed-22 checkpoint has mean
fractional change 0.0067014 and maximum 0.1445601 from original; 37,204 connections
changed during the final birth. It is the last-training checkpoint only.

The protocol has now started validation-quiet-original-10 with learning=False
and start_checkpoint=null. The runner sets motor exploration to zero for every
nontraining job. Neither this control nor subsequent candidate, obstacle or
held-out trials are claimed complete yet. There is still no best-validated
checkpoint and the normal launcher remains unchanged.

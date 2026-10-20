# Dev-seed-robust training, stage 1 (branch `claude/locomotion-v25`) — 2026-10-20

Naming note: the locomotion track independently uses "V26/V27/V28" labels (e.g. `batch-credit-v26`, `birth-cycle-v28`,
`--learning-scope motor-circuit-v1`, running in its own worktree on 2026-10-20). Everything below is a separate line of work
from this branch and is called "claude dev-seed stage 1" here; its output directory is `outputs/birth-cycle-v26a` only for
historical reasons.

## What was run
Twelve quiet training births on 12 new seeds (100-111) from the V25 bank, 5 s each, training fall cutoff 0.1 s, exploration
decaying geometrically 1.0 → 0.1 (`--exploration-schedule`), V24/V25 registration otherwise unchanged (standing-gate-linear-slip-v5,
bg weight 40, legacy proprioception, normalised and bias-corrected reward error). Candidate = last training checkpoint (birth 11).
Evaluation per `training-progress-v25.md` plan: frozen 5 s quiet jobs (learning off, exploration off) on 10 development seeds
10-19 for the candidate and the original weights, paired by seed. Held-out seeds 200-202 were not used.

## Training births (learning and exploration on; not frozen results)
Births 1, 4, 5, 7 ran the full 5 s with standing 0.943, 0.953, 0.891, 0.962; birth 10 ran 4.0 s (0.915); seven others ended
early at the fall cutoff or stopped (0.4-2.6 s; standing 0.62-0.91). The protocol's own validation was cut off after its first
control (cycle status `pending`) and is not part of the criterion.

## Frozen development-seed result (standing / stance slip mm/s / heading-forward mm)
| Seed | candidate | original |
|---|---|---|
| 10 | 0.975 / 14.0 / +9.5 | 0.442 / 9.4 / +8.0 |
| 11 | 0.146 / 5.9 / +3.5 | 0.929 / 16.7 / +12.7 |
| 12 | 0.478 / 10.1 / +7.9 | 0.742 / 13.8 / +8.6 |
| 13 | 0.448 / 10.0 / +7.4 | 0.863 / 15.9 / +12.4 |
| 14 | 0.973 / 16.7 / +3.2 | 0.435 / 14.1 / +7.5 |
| 15 | 0.061 / 5.2 / +15.0 | 0.856 / 14.7 / +8.5 |
| 16 | 0.909 / 15.6 / +9.7 | 0.904 / 16.7 / +12.7 |
| 17 | 0.635 / 13.1 / +11.6 | 0.948 / 17.0 / +8.8 |
| 18 | 0.873 / 14.9 / +8.8 | 0.329 / 7.3 / +7.4 |
| 19 | 0.924 / 15.1 / +5.9 | 0.929 / 15.0 / +8.9 |
| mean (sd) | 0.642 (0.35) | 0.738 (0.24) |

Pre-registered comparison (paired by seed, 10,000-sample bootstrap):
- mean standing difference (candidate − original) **−0.096**, 95% interval **[−0.39, +0.21]**, sd of the paired differences 0.51;
- seeds with standing ≥ 0.9: candidate 4, original 4; flipped onto the back (mean uprightness < 0): candidate 2, original 1;
- candidate better on 4 of 10 seeds (one-sided sign test p = 0.83); verdict by the written rule: **no evidence of improvement**.
- Mean heading-forward: candidate 8.2 mm, original 9.5 mm. On the two seeds where both models stand ≥ 0.9 (16, 19) the stance
  slip is 15.6 vs 16.7 and 15.1 vs 15.0 mm/s: no slip reduction.
- Across seeds the standing of the two models is negatively related (correlation −0.51, n = 10): each fails on different seeds,
  consistent with a body-plus-network system whose outcome is dominated by seed-dependent dynamics, not a systematic skill difference.

## Decision
By the stop rule fixed beforehand (point difference ≤ +0.05 and lower bound < 0) stage 2 is **not** run. Nothing is promoted, no
walking is claimed, and the 95% interval is wide (it excludes neither a moderate gain nor a moderate loss with 10 seeds).
Twelve more births at the same objective would probably not change this; the evidence points to seed-dependent stability of the
body-network dynamics as the limiting factor rather than the amount of training.

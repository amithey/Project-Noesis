# Does a softer start reduce the seed-to-seed spread of the original weights? — 2026-10-24

Question: is the large spread of frozen standing across seeds (original weights 0.33-0.95) caused by the start-up transient
(the brain igniting from silence while the body is still landing)? Frozen jobs as in the locomotion protocol (quiet arena,
learning off, exploration off, standing-gate-linear-slip-v5, bg weight 40, legacy proprioception), original weights, seeds 10-19,
5 s of awake time in every condition. Opt-in changes only: `learning_lab --warmup-ms` (brain runs on background activity alone
for that long before first coupling; default 0) and `frozen_probe --birth/--warmup-ms`.

| Condition | mean standing | sd across seeds | seeds ≥ 0.9 | flipped (mean uprightness < 0) | heading-forward (mm) | stance slip, seeds standing ≥ 0.9 (mm/s) |
|---|---|---|---|---|---|---|
| baseline (birth 0.05 s) | 0.738 | 0.240 | 4 | 1 | 9.5 | 16.3 |
| B: body settles 0.5 s before coupling | 0.612 | 0.302 | 2 | 3 | 9.5 | 16.6 |
| C: brain warmed up 500 ms before coupling | 0.719 | 0.286 | 4 | 2 | 9.3 | 16.1 |
| D: both | 0.565 | 0.387 | 2 | 4 | 8.3 | 13.6 |

Per seed (10..19) standing: baseline [0.44 0.93 0.74 0.86 0.44 0.86 0.90 0.95 0.33 0.93]; B [0.28 0.88 0.19 0.89 0.41 0.56 0.74
0.92 0.29 0.96]; C [0.52 0.88 0.74 0.94 0.78 0.93 0.34 0.16 0.96 0.95]; D [0.89 0.04 0.92 0.97 0.89 0.86 0.21 0.53 0.02 0.32].

- **No variant reduces the spread** (sd 0.29, 0.30 and 0.39 against 0.24) or the number of falls (2, 3 and 4 against 1); the combined start
  is the worst. The expected effect (fewer falls, smaller spread, no loss of mean standing) is absent in all three.
- Differences between conditions (mean 0.57-0.74) are smaller than the spread inside each condition (sd 0.24-0.39) with 10 seeds, so
  none of the conditions should be read as better or worse than the baseline; the outcome per seed also changes in both directions
  (e.g. seed 11: 0.93 baseline, 0.88 B/C, 0.04 D).
- Consequence: the transient at the start does not explain the seed dependence. Whatever makes the outcome chaotic acts during the
  episode itself (network-state noise and the body's contact dynamics), so reducing it will need the objective, not the initial
  conditions, to prefer robustness. The planned next step is a small evolved parameter set scored on several seeds.

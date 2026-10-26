# Do falls have a detectable precursor? Analysis of 30 existing frozen episodes — 2026-10-26

No new simulation: 30 frozen 5 s episodes already recorded under the V24/V25 protocol settings (quiet arena, learning and exploration
off), seeds 10-19, for three weight sets: original (10), the twelve-birth "trained" candidate (10) and the evolved pool-gain genome (10).
Per-step CSV series (5 ms) were summarised in windows; outcome "failed" = standing fraction < 0.5 (9 of 30: original 3, trained 4,
evolved 2). Tables are descriptive; the permutation p-values are two-sided and uncorrected unless said otherwise.

## 1. When do the falls happen?
The first time uprightness drops below 0.3 in the nine failed episodes was at 0.40, 0.81, 0.95, 1.15, 1.79, 2.40, 2.46, 2.50 and 2.85 s
(median 1.8 s; five of nine after 1.5 s). Nine further episodes that ended up standing >= 0.5 also dipped below 0.3 at some point
(0.38 and 0.89 s early; 2.8, 2.8, 3.2, 3.4, 4.6, 4.7 and 4.7 s late), i.e. brief losses of balance with recovery are common.
This **corrects the brief's earlier statement that falls start in the first 1-2 s**: they are spread over the whole 5 s.

## 2. Do early windows predict failure?
Features of the first second (two windows, 0-0.5 s and 0.5-1 s: supporting feet, released feet, stance slip, uprightness, height,
motor rate, active motor neurons, joint speed, effort, forward speed, grip factor, yaw rate, minima) against failure, pooled over the
30 episodes, rank AUC with a 2,000-label permutation:
- Smallest p-values: released feet 0.5-1 s (AUC 0.24, p = 0.021), stance slip 0.5-1 s (0.25, p = 0.037), uprightness 0-0.5 s
  (0.26, p = 0.042). With 28 features the Bonferroni threshold is 0.0018: **none survives**.
- The group differences are small (e.g. uprightness 0.88 in failed against 0.91 in surviving episodes; slip 11.9 against 15.2 mm/s).
  Yaw rate is large in every episode (about 5-6 rad/s in the first second) and does not separate the groups (AUC 0.36, p = 0.25).
- Within the ten original-weight episodes alone the AUCs for the same features are unstable and contradict each other
  (released feet: 0.95 in the first half second, 0.00 in the next), as expected for n = 10 with three failures.

## 3. Is there a lead-in just before a fall?
Eight failed episodes fell after 0.4 s. For each, the window 0.30-0.05 s before the first uprightness < 0.3 was compared with the same
time window in the 12 episodes that never dropped below 0.3 (96 control windows):

| Measure | before a fall (n = 8) | control windows (n = 96) |
|---|---|---|
| supporting feet | 2.46 (sd 0.28) | 2.32 (sd 0.36) |
| released feet | 1.83 (sd 0.14) | 1.92 (sd 0.35) |
| stance slip, mm/s | 15.2 (3.5) | 15.1 (5.0) |
| uprightness | 0.85 (0.05) | 0.87 (0.05) |
| height, mm | 1.21 (0.11) | 1.19 (0.10) |
| joint speed, rad/s | 2.38 (0.15) | 2.33 (0.18) |
| motor rate, Hz | 31.0 (3.0) | 31.1 (2.7) |
| effort | 0.11 (0.01) | 0.10 (0.01) |
Every difference is a fraction of a standard deviation. At this resolution (window averages of the recorded signals) a fall has no
visible lead-in.

## What this does and does not show
- Shows: slowly changing aggregate signals (support, slip, posture, motor rate) in the first second or in the 250 ms before a fall do
  not distinguish episodes that fall from those that do not; a precursor model "it starts wrong in the first second" is not supported.
- Does not show: that no precursor exists. Not analysed: per-leg contact patterns at 5 ms, angular-velocity/roll-rate spikes, single
  foot-slip events, the actual force/contact state, or the neural state (motor-pool synchrony). n = 9 failures from 30 episodes is small.
- Consistent with earlier measurements: the outcome is dominated by seed-dependent, chaotic-looking dynamics during the episode.

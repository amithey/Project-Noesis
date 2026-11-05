# Positive/negative control for the real MotorPlasticity (synthetic mechanism test)

2026-11-01 · Claude (#19) · code `gpu-pilot/src/noesis_gpu/mechanism_control.py`, tests `tests/test_mechanism_control.py`,
runner `run_mechanism_control.py`. **Registered before any data of this control exist (this file is committed first).**
This is a synthetic mechanism test with the real `MotorPlasticity` class and the pilot's settings (perturbation rule, normalized and
bias-corrected error, eligibility 100 ms, +-50% bounds, LR 0.2, 990 steps of 5 ms). It is NOT physical evidence, does not identify the
cause of any failure of the fly, and a pass does not imply the fly can learn.

## Design
Random sparse 300-neuron network, 30 motor targets, 6 "good" targets. Reward = clip(0.3 + noise + kappa * mean tanh(OU state of the good
targets)), noise N(0, 0.41^2) (variance ~0.17, the size of the measured reward variance; white, so this is the hard fast-noise case).
Credit = tanh(white innovation), as in the pilot. Arms over the same 16 seeds, the same noise, spikes and perturbation draws:
- positive: kappa = 0.2 (planted relation) with learning;
- negative: kappa = 0 (same everything, no relation) with learning;
- learning-off: the positive trial, identical draws, `update` never called.
Statistic D per birth: mean signed relative change of the edges onto the good targets minus that of the edges onto the other targets.

## Registered pass rule (all must hold)
1. positive: mean D > 0 and z = mean/(sd/sqrt(16)) >= 3;
2. negative: |z| < 3 and |mean D| < one third of the positive mean D;
3. learning-off: weights bit-identical to the start, zero updates, D = 0 for all seeds;
4. both learning arms: every edge ratio to the original weight inside [0.5, 1.5] and all signs preserved.
Any failure is reported as a failure of this control (the mechanism test is then not informative about the rule at this size), and the
settings are not retuned afterwards.

## Limits (stated now)
One network, one effect size and lag (0 ms), white reward noise; kappa = 0.2 was the cell where the earlier toy (docs/workstreams/claude/l2-reward-variance,
`toy_baseline_real_noise.py`) used the real L1 reward series; here the noise is synthetic. 16 independent births per arm. The statistic compares
good vs other target edges and cannot see credit that moves the wrong edges. Final held-out seeds are not used.

## Result (run once, 2026-11-01, after this protocol was committed in a34f137; `mechanism-control-result.json`)

**Registered pass rule: PASS (all four conditions).** 16 seeds per arm, runtime 20 s.
| arm | mean D | sd | z | seeds with D > 0 |
| --- | --- | --- | --- | --- |
| positive (kappa 0.2) | 0.001256 | 0.000434 | 11.6 | 16/16 |
| negative (kappa 0) | 0.000028 | 0.000434 | 0.25 | 8/16 |
| learning-off | exactly 0 | - | - | weights bit-identical, 0 updates |
Bounds [0.5, 1.5] and signs held in all learning births.

Reading, with limits: with the pilot's rule and settings, a planted, instantaneous (lag 0) relation of this strength is picked up in the right direction in every seed, and the matched
no-relation arm does not move toward it. So the mechanism is not broken in this synthetic setting. **The effect is small in absolute terms**: the mean good-vs-other relative edge change is about
0.13% in 5 s (D = 0.0013), far below the +-50% bounds, and the identical arm without a relation has the same seed-to-seed spread (sd 0.00043). This is a synthetic mechanism test with
white reward noise and no physical delay, not physical evidence. It does not show the cause of the real births' lack of separation from their noise controls, does not exclude reward/credit
timing or body-chain causes, and one pass of one cell is not a power calculation.

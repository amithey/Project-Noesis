# Body A/B of the inhibition scale on the locomotion pipeline — 2026-10-18

Plan: `plan-body-ab-g-inh.md`. Pipeline: `noesis.experiments.learning_lab` (corrected connectome, muscle body,
quiet arena, background 3 Hz × 60), **original weights, frozen: no learning, no exploration**, 5 simulated
seconds, validation seeds 10 and 11 of the locomotion protocol (held-out seeds 200-202 untouched). New opt-in flags
`--g-inh` (all inhibitory weights × g_I; refused together with `--learn`) and `--r-half` (pool firing for half
muscle activation, default 25 Hz). With both at their defaults the pipeline reproduces the earlier baseline exactly
(same standing fraction, mean motor rate and reward to all printed digits) and all 319 tests pass.

## Calibration of r_half at g_I = 4 (seed 50, 1 s pilots; not an evaluation seed)
| Run | mean motor rate (Hz) | mean muscle activation, agonist / antagonist | standing |
|---|---|---|---|
| g_I = 1, r_half 25 | 29.8 | 0.233 / 0.226 | 1.00 |
| g_I = 4, r_half 25 | 4.8 | 0.058 / 0.100 | 1.00 |
| g_I = 4, r_half 4.04 (rule: scale by motor-rate ratio) | 5.8 | 0.144 / 0.185 | 0.82 |
| g_I = 4, r_half **2.68** (rule: match mean activation) | 5.5 | 0.162 / 0.226 | 0.94 |

The planned rate-ratio rule missed the activation target by 38% and was replaced by matching mean activation
(a deviation from the written plan). 2.68 Hz leaves the average activation 15% below the g_I = 1 level (agonist
−30%, antagonist matched), outside the 10% tolerance in the plan; calibration was stopped there rather than chasing
a single noisy 1 s pilot.

## Results (5 s, frozen original weights)
| Condition | seed | standing | stance slip (mm/s) | heading-forward (mm) | legs with ≥ 2 cycles | mean supporting feet | uprightness | mean reward | mean motor rate (Hz) | mean activation |
|---|---|---|---|---|---|---|---|---|---|---|
| C1: g_I 1, r_half 25 | 10 | **0.172** | 7.8 | +11.1 | 6 | 1.36 | −0.34 | −0.475 | 29.4 | 0.255 |
| C1 | 11 | 0.949 | 17.5 | +7.3 | 6 | 2.34 | 0.87 | +0.178 | 29.8 | 0.260 |
| C2: g_I 4, r_half 25 | 10 | 1.000 | 18.6 | **−49.6** | 5 | 3.65 | 0.96 | +0.583 | 5.3 | 0.096 |
| C2 | 11 | 1.000 | 16.8 | **−51.1** | 6 | 3.71 | 0.97 | +0.577 | 4.8 | 0.088 |
| C3: g_I 4, r_half 2.68 | 10 | 0.946 | 11.0 | −5.7 | 6 | 2.25 | 0.88 | +0.180 | 5.3 | 0.221 |
| C3 | 11 | 0.410 | 8.3 | −2.8 | 6 | 1.60 | 0.04 | −0.263 | 5.1 | 0.215 |

(Protocol gates: standing ≥ 90%, heading progress ≥ +1 mm, ≥ 2 stance/lift cycles in four legs, slip ≥ 20% below
the original weights; the last gate is not defined here because every condition uses original weights. Only C1 seed 11
passes the standing and heading gates.)

## Reading (pre-stated rule: differences smaller than the seed spread of C1 are not effects)
- **The baseline itself is unstable across seeds:** C1 stands 17% of the time on seed 10 (it falls over) and 95%
  on seed 11. Standing differences between conditions are therefore **not interpretable** with two seeds.
- **Lower activation made the body stand, but not walk.** With the unscaled actuator (C2) the legs sit near the
  passive standing pose: 100% standing on both seeds, 3.7 supporting feet, but the body slides **backward ≈ 50 mm**
  in 5 s in both episodes (the baseline moves +7 to +11 mm forward).
- **After matching activation (C3) the standing advantage disappears** (0.95 and 0.41) and the heading stays
  negative (−5.7, −2.8 mm).
- **Heading is the one comparison larger than the baseline's seed spread:** all four g_I = 4 episodes end behind
  their start (−2.8 to −51 mm) while both baseline episodes move forward (+7 to +11 mm). Two seeds per condition:
  a candidate effect, not an established one.
- No condition is a locomotion improvement; nothing here is learned walking.

## Consequence for sharing a network between the tracks
`g_I` > 1 is not a drop-in change for the body: the motor operating point drops by ≈ 6× in rate and the rate-to-
activation map and probably the learned weights and exploration amplitude would all have to be re-derived, and
the first backward-drift signal suggests the directional balance of the motor pools is also affected. With this
evidence the two tracks should stay on separate network settings (`g_I` = 1 for locomotion; the calibrated network
only for the leg-free sensory agent), and any joint use needs a locomotion retraining at the chosen `g_I`, not a
parameter swap. Limits: two seeds, one 5 s episode each, frozen original (not trained) weights, quiet arena, one
network seed per episode, an imperfect (−15%) activation calibration, `g_I` = 12 not tested on the body.

## Follow-up: how unstable is the g_I = 1 baseline? (4 more seeds + a reproducibility rerun)
Same pipeline, `g_I` = 1, `r_half` = 25, frozen original weights, quiet arena, 5 s; added seeds 12-15 and a rerun
of seed 10.

| Seed | standing | uprightness | heading-forward (mm) | supporting feet | stance slip (mm/s) |
|---|---|---|---|---|---|
| 10 | 0.172 | −0.34 | +11.1 | 1.36 | 7.8 |
| 10 (rerun) | 0.172 | −0.34 | +11.1 | 1.36 | 7.8 |
| 11 | 0.949 | +0.87 | +7.3 | 2.34 | 17.5 |
| 12 | 0.273 | −0.21 | +9.0 | 1.44 | 9.3 |
| 13 | 0.957 | +0.87 | +10.1 | 2.35 | 16.3 |
| 14 | 0.101 | −0.51 | +8.8 | 1.13 | 6.5 |
| 15 | 0.438 | +0.08 | +10.5 | 1.77 | 10.5 |

- **The run is exactly reproducible** (the seed-10 rerun matches to every printed digit), so the spread is not
  hidden nondeterminism: the seed sets the network noise and the body's response is highly sensitive to it.
- **Standing is strongly seed dependent:** 0.10-0.96 (mean 0.48, sd 0.38 over six seeds); only 2 of 6 seeds reach
  the 90% gate, and three seeds end on their back (negative mean uprightness). This is the frozen *original*
  weights; improving it is what the locomotion training does, so the spread is an expected property of the
  untrained network, not a new defect. The two unstable and stable episodes shown earlier have the same motor rates
  (28-32 Hz) and active motor-neuron counts (≈ 150-165); the unstable one turned ≈ 2 rad about the vertical axis
  within the first second and flipped between 0.5 and 1.5 s.
- **Heading-forward is the stable quantity:** +7.3 to +11.1 mm on every seed (mean 9.5, sd 1.4), including seeds that
  fell over.

## Revised reading of the A/B
- **Standing comparisons remain uninterpretable at this sample size:** the baseline spans 0.10-0.96; C2's 1.00 and
  1.00 and C3's 0.95 and 0.41 all fall inside that range (C2 is 2/2 at the gate versus 2/6 for the baseline, Fisher
  one-sided p ≈ 0.4).
- **The backward drift at `g_I` = 4 is now clearly outside the baseline distribution:** all four `g_I` = 4 episodes
  (−49.6, −51.1, −5.7, −2.8 mm) lie below the smallest of the six baseline values (+7.3 mm); the chance that four
  specified values are all below all six others under no difference is 1/C(10,4) = 1/210 ≈ 0.005 (one-sided
  permutation). Two seeds per `g_I` = 4 condition and a heading measure that is stable at `g_I` = 1; the cause
  (motor-pool directional balance, the reduced activation, or the body's passive pose) is not identified.

# Stage 4: leg-free virtual agent steered by DNa — 2026-10-16

`experiments/virtual_forage.py`, `virtual_agent.py`. Corrected brain, graded APL gain 1024, receptor drive
weight 400, a 2D point agent walking 8 mm/s, Gaussian odour plume (sigma 10 mm) with the food 20 mm away, two
antennae 1 mm either side of the body axis driving the left/right food-odour receptor neurons (867 / 787) with a
rate that grows with the square root of concentration and with the hunger gain. Steering signal =
(left-soma DNa rate − right-soma DNa rate) / (sum), calibrated against a 1 s symmetric-input period, low-passed
over 1 s, gain 10 rad/s per unit (chosen on held-out pilot seed 99, not on the evaluation seeds). 12 s episodes,
4 start poses (headings 90°, −90°, 150°, −150° off the food bearing). "More left DNa activity turns the agent
left" is an assumption (reversed-sign control included).

## Result: no evidence that the brain steers the agent to the food
| Condition (hungry unless noted) | Reached food (of 4) | Final distance, mm (per seed 0–3) |
|---|---|---|
| brain, DNa | 0 | 60.2, 14.2, 60.1, 60.0 |
| brain, fed | 0 | 40.1, 5.9, 60.1, 60.2 |
| shuffled receptor sides | 0 | 15.6, 37.0, 60.2, 17.7 |
| reversed sign | 0 | 60.1, 28.8, 30.6, 25.3 |
| random descending neurons | 1 (seed 0, by chance) | 3.0, 59.2, 60.1, 60.2 |
| no gradient (same odour at both antennae) | 0 | 60.1, 18.3, 60.1, 11.8 |
| diffusion only (no steering) | 0 | 60.1, 60.1, 60.4, 60.1 |

Paired differences in final distance (brain_hungry minus control; negative = brain closer), mean ± sd over 4
seeds: shuffled sides +16.0 ± 33.1, reversed sign +12.5 ± 23.6, random DN +3.0 ± 41.9, no gradient +11.0 ± 24.9,
diffusion only −11.5 ± 22.9, fed +7.1 ± 9.6. Every sd is larger than its mean; no comparison is
distinguishable from zero.

## Why this is not a success
- The agent never reached the food from the DNa signal. The only successful episode was a random-DN control.
- Distances below 60 mm come mostly from turning in circles (path efficiency 0.05–0.4), which keeps an agent
  near its start whatever drives the turning; diffusion-only simply walks out of the arena (60 mm).
- The steering signal is dominated by network-state noise: its per-step sd is ≈ 0.21 while its mean after
  bias subtraction is 0.02–0.18 and changes sign or magnitude between conditions that share the same input
  structure (e.g. no-gradient seed 1: +0.177). The 1 s calibration bias does not hold over the episode, so the
  network is not stationary enough for a fixed-bias read-out.
- Consistent with the earlier audits: the DNa side effect is a few percent of a large, noisy, nearly saturated
  response (parts 3–4 of `findings-sense-audit.md`).

## What is and is not shown
- Shown: in this model, with this read-out, a DNa left/right difference does not steer a point agent up an odour
  gradient better than controls; hunger as a receptor gain made no reliable difference.
- Not shown: that no read-out could work (more neurons, longer integration, different mapping, a
  better-calibrated network). Four poses, one set of agent parameters, one plume size. No odour identification
  and no learning were attempted or claimed.

## Where the evidence points
Every stage so far is limited by the same property of the network: it is either silent or saturated and has
no wide sparse operating regime, so sensory structure rides on large state noise
(`findings.md` stability table, `findings-sense-audit.md` parts 1–4, `operating_point.py`). Further agent
tuning on this network is unlikely to help; a calibration of excitation/inhibition balance that produces a
wide sparse regime, judged by the existing selectivity and operating-point scripts, is the first thing to try.
That touches the shared brain model, so it should be coordinated with the locomotion track.

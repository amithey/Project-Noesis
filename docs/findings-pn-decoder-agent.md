# Leg-free virtual agent steered by a trained projection-neuron read-out — 2026-10-17

`experiments/decoder_forage.py --readout pn`, `virtual_agent.py`, branch `feature/network-calibration`.
Network: corrected connectome, all inhibitory weights × 12, graded APL 1024. Read-out: ridge decoder (ridge 0.1 per
feature, fixed in advance) on log(1 + 250 ms causal average of the rates of the 699 projection neurons), trained
on `decoder_contrast` windows (last second of 2 s blocks, 112 blocks, left/right ratios 0.33-3, 6 concentration
levels). Its score, clipped to [-1, 1] and smoothed over 0.5 s, turns the agent (1.5 rad/s per unit); positive =
left stronger = turn left. Agent: 2D point, 8 mm/s, rotational diffusion 0.3 rad/√s, Gaussian plume (σ 10 mm),
food 20 mm away, 12 s episodes, 4 start poses (seed = pose; same pose and same network seed for every condition).
Hunger scales receptor drive (energy 0.3 hungry, 1.0 fed).

Two worlds: **standard** (antennae 1 mm either side of the body axis) and **wide** (3 mm): an explicit change
of the *world*, not of the network, that raises left/right contrast.

## Result
Food reached (of 4 start poses) and final distance in mm per pose (seeds 0-3):

| Condition | standard, reached | standard, final mm | wide, reached | wide, final mm |
|---|---|---|---|---|
| decoder, hungry | 2 | 3.0, 33.6, 3.0, 23.2 | **4** | 2.8, 2.8, 2.7, 2.9 |
| decoder, fed | 1 | 2.7, 22.2, 47.3, 18.3 | 3 | 2.9, 3.0, 43.7, 2.8 |
| decoder fitted on shuffled labels | 0 | 60.0, 6.9, 60.1, 17.6 | 0 | 60.2, 27.0, 31.6, 58.4 |
| reversed sign | 1 | 2.8, 60.0, 25.4, 60.1 | 0 | 30.9, 60.2, 60.1, 52.2 |
| receptor sides shuffled | 0 | 28.7, 43.4, 14.2, 38.5 | 0 | 36.9, 32.4, 15.6, 31.2 |
| no gradient (same odour at both antennae) | 0 | 14.7, 43.2, 49.2, 13.6 | 0 | 14.7, 43.2, 49.2, 13.6 |
| diffusion only (no steering) | 0 | 60.1, 60.1, 60.4, 60.1 | 0 | 60.1, 60.1, 60.4, 60.1 |

Paired difference in final distance, hungry decoder minus control (mean ± sd over 4 poses; negative = decoder
closer; poses where it was closer):

| Control | standard | wide |
|---|---|---|
| shuffled-label decoder | −20.5 ± 43.2 (2/4) | −41.5 ± 17.4 (4/4) |
| reversed sign | −21.4 ± 15.6 (3/4) | −48.1 ± 13.8 (4/4) |
| shuffled receptor sides | −15.5 ± 7.2 (4/4) | −26.2 ± 9.2 (4/4) |
| no gradient | −14.5 ± 23.2 (3/4) | −27.4 ± 18.7 (4/4) |
| diffusion only | −44.5 ± 15.4 (4/4) | −57.4 ± 0.2 (4/4) |
| fed | −6.9 ± 25.3 (1/4) | −10.3 ± 20.4 (3/4) |

(The no-gradient rows are identical across worlds because that control samples the body-centre concentration, so
antenna spacing does not enter.)

## What this shows and does not show
- **Wide world: 4 of 4 hungry-decoder episodes reach the food, versus 0 of 4 in each of the six controls**
  (one-sided Fisher exact p = 1/70 ≈ 0.014 per comparison, not corrected for the six comparisons or the two
  worlds). The reversed-sign agent, which uses the same decoder and brain, moves away, so the sign convention
  steers rather than merely rotates the agent.
- **Standard world: weak and not conclusive.** 2 of 4 reached versus 0-1 of 4 in controls (Fisher p ≈ 0.21-0.5);
  one reversed-sign episode also reached the food (seed 0), consistent with the decoder's modest accuracy (61-68%
  per window) at the lower contrast of 1 mm antennae.
- **Hunger made no reliable difference** (wide: 4 vs 3 reached; standard: 2 vs 1; fed-minus-hungry differences
  have sd larger than their means), so a hunger-dependent behaviour is **not** supported.
- This is **not** the brain steering by itself: the read-out is trained with side labels, the projection neurons
  are one synapse from the receptors, and the "world" had to be changed (3 mm antennae) to reach 4 of 4. It is
  a demonstration that the calibrated network preserves enough left/right information for a trained linear
  interface to steer a leg-free agent up an odour gradient, nothing about odour identity, memory or learning
  inside the network.
- Only 4 start poses; one network calibration (`g_I = 12`); the inhibition factor of 12 is a model assumption.

## Update: 12 start poses per world (`experiments/agent_stats.py`)
The four-pose result above was extended with eight more poses per world (seeds 4-11; seeds 0-3 unchanged; same
decoder, parameters and controls; nothing was tuned after the first results). Fisher exact is one-sided
("hungry decoder reached food more often than the control"), Holm-corrected over the six controls within a world;
the sign test is exact and one-sided on the paired final distance (hungry decoder minus control, negative =
decoder closer).

| Condition | **wide (3 mm)** reached / 12 | Holm Fisher p | final-distance difference, mm (closer in n/12) | **standard (1 mm)** reached / 12 | Holm Fisher p | final-distance difference, mm (closer in n/12) |
|---|---|---|---|---|---|---|
| decoder, hungry | **9** | | | 3 | | |
| decoder, fed | 5 | 0.107 | −10.2 ± 17.5 (9) | 4 | 1.00 | −0.4 ± 23.2 (4) |
| shuffled-label decoder | 0 | 0.001 | −41.6 ± 16.2 (12) | 1 | 1.00 | −15.5 ± 29.6 (5) |
| reversed sign | 0 | 0.001 | −49.3 ± 13.0 (12) | 1 | 1.00 | −15.7 ± 28.6 (7) |
| shuffled receptor sides | 0 | 0.001 | −27.5 ± 18.7 (11) | 1 | 1.00 | −5.1 ± 28.3 (7) |
| no gradient | 1 | 0.003 | −29.5 ± 21.1 (11) | 1 | 1.00 | −9.0 ± 20.1 (7) |
| diffusion only | 0 | 0.001 | −51.5 ± 13.6 (12) | 0 | 0.65 | −31.0 ± 23.2 (10) |

- **Wide-antenna world: the effect holds with more poses.** The trained projection-neuron read-out brought the
  agent to the food in 9 of 12 poses; no control exceeded 1 of 12 (Holm p ≤ 0.003 for all five controls other than
  fed; sign tests on distance p ≤ 0.003). The earlier "4 of 4" was an overestimate of the rate (≈ 75%, not 100%).
- **Standard world: no evidence of steering.** 3 of 12 reached versus 0-1 of 12 in the controls (Holm p 0.65-1.0).
  Its only nominally significant result (distance versus no steering, sign test p = 0.019) is not corrected and
  can come from turning in circles rather than approaching. The earlier statement that this world was "weak" is
  superseded: it is indistinguishable from the controls.
- **Hunger: no reliable effect.** Wide world 9 vs 5 of 12 (Holm-uncorrected Fisher p = 0.107, closer in 9/12
  with sign-test p = 0.073); standard world 3 vs 4 of 12. The hunger gain is therefore not supported as an
  effect on foraging in this model.
- All limits above remain: trained read-out, projection neurons one synapse from the receptors, changed world
  needed to see an effect, one calibration (`g_I = 12`, a model assumption), 12 poses per world with one
  network seed per pose (the pose and the network noise are not independent factors).

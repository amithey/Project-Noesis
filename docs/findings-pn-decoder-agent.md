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

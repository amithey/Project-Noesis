# Findings log

Dated, reproducible observations. Everything here is a single run unless stated.

## 2026-10-13 — brain stability (brain only, 8 s, dt 0.5 ms, default NT signs)

Spontaneous activity = Poisson events on random neurons (`background_hz` per neuron,
`background_weight` synapse-equivalents).

| background | state | spiking neurons | rate of those |
|---|---|---|---|
| 5 Hz, w=60 | self-sustained, stationary | 12.7 k | ~60 Hz (hyper-active core) |
| 3 Hz, w=60 / 5 Hz, w=50 | nearly dead | ~500 | -- |
| 5 Hz, w=90 | self-sustained, stationary | ~47 k | ~12 Hz |
| 10 Hz, w=60 | self-sustained, stationary | ~26 k | ~15 Hz |

- The network is **bistable**: below a threshold it is silent, above it it sustains itself.
  It does not run away; after ~1 s transient the rate is constant.
- Treating monoamine / unknown-NT neurons as silent or inhibitory instead of excitatory
  barely changes the state, so that assumption is not the driver.
- Default is now 5 Hz / w=90 (broad, ~12 Hz regime).
- Leg motor neurons sit at ~10 Hz on average, but agonist-antagonist differences reach
  hundreds of Hz for some joints (tonic, near the refractory limit): a likely
  non-physiological feature of the model (no adaptation, no neuromodulation).

## 2026-10-13 — embodied newborn, 3 s, birth at 0.4 s

| condition | mean joint offset | body height (mm) | notes |
|---|---|---|---|
| blind (`--no-senses`) | 0.14 rad | 0.51 | sinks low, stable |
| leg senses on (3,921 afferents) | 0.21 rad | 0.75 | rolls onto its side within ~1 s and stays there |

- With the first version (motor scale 5 Hz, no slew limit) the blind run crashed the physics
  (`mjWARN_BADQACC`) after being flung 2 mm; targets are now rate-limited (8 rad/s) and the scale
  is 100 Hz.
- One seed per condition: the blind/sensing difference is **not** statistically established.
- Not learned, not walking: no self-righting, no coordinated gait.
- *Correction (2026-10-13, later):* an earlier note said "on its back". The uprightness metric
  (thorax z-axis . world up) shows min +0.29, i.e. ~70 deg on its side, never inverted (-1).

## 2026-10-13 — spike-frequency adaptation (`LIFParams.da`, `tau_a` = 200 ms)

Brain only, 5 Hz / w=90 background, 2 s window, after a 1 s settle:

| da (mV/spike) | mean leg-MN rate | max leg-MN rate | p99 \|agonist-antagonist\| | sugar -> MN9 spikes (200 Hz drive, 0.5 s) |
|---|---|---|---|---|
| 0 | 8.8 Hz | 272 Hz | 347 Hz | 45 |
| 0.25 | 7.2 | 240 | 289 | 17 |
| 0.5 | 6.1 | 201 | 253 | 5 |
| 1.0 | 4.0 | 128 | 167 | 5 |

Adaptation is a uniform, assumed mechanism (same da for every neuron). At da=1 the extreme
motor rates fall into a biologically plausible range, but the sugar->MN9 response now needs a
stronger stimulus than the validation default.

Embodied, 3 s, birth at 0.4 s, one run each (da=1 unless stated):

| condition | mean joint offset | uprightness mean / min |
|---|---|---|
| senses, da=0 | 0.21 rad | +0.77 / +0.29 |
| senses, da=1 | 0.14 rad | +0.71 / +0.29 |
| blind, da=1 | 0.10 rad | +0.99 / +0.92 |

Adaptation did not remove the tilt when senses are on; the blind fly stays upright but barely
moves. Single runs: differences are suggestive only.

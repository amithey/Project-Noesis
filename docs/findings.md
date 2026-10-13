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
| leg senses on (3,921 afferents) | 0.21 rad | 0.75 | flips onto its back within ~1 s and stays there |

- With the first version (motor scale 5 Hz, no slew limit) the blind run crashed the physics
  (`mjWARN_BADQACC`) after being flung 2 mm; targets are now rate-limited (8 rad/s) and the scale
  is 100 Hz.
- One seed per condition: the blind/sensing difference is **not** statistically established.
- Not learned, not walking: no self-righting, no coordinated gait.

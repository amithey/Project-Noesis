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

## 2026-10-13 — gravity and rotation senses (`balance.py`)

1,053 afferents: Johnston's organ C/D/E/F (661, gravity/antennal position) driven by the
body-frame "up" vector, haltere campaniform/chordotonal/orphan (392) driven by body angular
velocity. Preferred directions are random per neuron (not in the data); rectified tuning.
3 s, birth at 0.4 s, da=1, one run each:

| condition | mean joint offset | uprightness mean / min |
|---|---|---|
| leg senses only | 0.14 rad | +0.71 / +0.29 |
| leg + balance senses | 0.14 rad | +0.71 / +0.30 |
| balance senses only | 0.10 rad | +0.98 / +0.89 |

- Balance senses did **not** change the outcome: no self-righting, no measurable effect.
- The tilt appears only when leg senses are on, so the leg afferent -> motor-neuron loop is the
  likelier cause (strong, asymmetric excitation), not missing gravity information.
- The balance-only fly stays upright but barely moves, like the blind one.

## 2026-10-13 — audit: why do leg senses coincide with a tilted fly? (`leg_sense_audit.py`)

**Structure** (row-normalised signed propagation, 1-3 hops, leg afferents -> leg motor neurons):
- Per sensory neuron, campaniform (load) sensilla influence leg motor neurons ~700x more than bristles
  (|influence| 0.065 vs 0.00009), hook 0.0076, claw 0.0047, hair plates 0.0034, club 0.0003.
- Campaniform cells in the table are almost all on the front legs (29 L, 25 R; only 2 per other leg).
- Most-driven muscles: tergotrochanter (the jump-muscle motor neurons), sternal posterior rotator,
  tibia flexor.
- Left/right: neuron inventories are symmetric (2,604 L / 2,616 R afferents; 195 L / 196 R motor
  neurons) but campaniform influence is 2x larger on the right (asym -0.34).

**Dynamics** (brain alone, resting-body sensory drive, 1.5 s, one seed): mean leg-MN rate 4.5 Hz
without leg senses vs 3.8 Hz with them; mean |joint drive| 14.0 vs 15.2 Hz. Left/right mirror
asymmetry of joint drive is already 0.31 **without any leg input** (0.29-0.41 with it): brain
noise, not the senses, makes the motor output asymmetric.

**Body ablation** (3 s, da=1, no balance senses, one run each; uprightness mean / min):

| leg senses | upright |
|---|---|
| none | +0.99 / +0.92 |
| all | +0.71 / +0.29 |
| all except campaniform | +0.87 / +0.39 |
| campaniform only (62 neurons) | +0.74 / +0.47 |

**Conclusion:** the hypothesis "campaniform load sensors cause the tilt" is **not supported**: almost any
sensory drive changes the trajectory. A single run per condition cannot separate causation from
chaotic divergence; multiple seeds are required. Also note that the resting "claw/hook" transfer
function puts those neurons at the sigmoid midpoint (~76 Hz) by construction.

## 2026-10-13 — dataset identity

`neurons.csv` / `connections_princeton.csv` are **BANC v888** (female adult brain + nerve cord, codex.flywire.ai):
158,262 neurons and 3,037,361 neuron-pair connections (our pair count after summing per-neuropil rows
is exactly 3,037,361). Brain-only FAFB v783 (139,255 neurons) is a subset, male MANC (VNC only) and MCNS
(male CNS, 166,700 neurons) are other specimens. No further download is needed for the body; MANC is
useful later to cross-check the leg motor-neuron -> muscle annotation.

## 2026-10-13 — body physics (`body.py`, `body_check.py`, `newborn_muscles.py`)

Goal: the body only supplies physics (skeleton, joint limits, muscles, tissue); the brain decides.

**Findings that changed the body**
- The `stretch` start posture lies outside the real stepping range of the joints (13/21 joints on the
  front/middle/hind left legs, 24/42 with a margin): tibia 0 rad vs 1.2-2.2 rad in real steps. A body in
  that pose lies on its belly (height 0.70 mm). The step-cycle **mean of the real stepping kinematics**
  makes the fly stand (1.06-1.13 mm, upright, all six legs loaded), so that is now the standing posture.
- Joints had **no range limits** (`jnt_range` 0,0). They now use the observed step range widened by
  0.25 x span + 0.35 rad (assumption: true ROM exceeds what walking uses).
- **Direct torque control is infeasible for this body.** Leg-segment inertia is ~6.5e-8 while torques that
  bear body weight are ~O(1): soft joint limits (stiffness scales with the tiny inertia) are overpowered
  (overshoot 1-7 rad) and strong torques blow up the integrator (`mjWARN_BADQACC`) for every
  strength/stiffness/damping/integrator combination tried (about 20 configurations).
- Chosen model: **equilibrium-point muscles**. Each joint has an agonist and an antagonist pool (from the
  FlyWire muscle annotations); activation = r / (r + 25 Hz) with a 30 ms time constant; the pair sets the
  joint's equilibrium angle between the standing posture and 90% of the way to a limit; tissue servo
  (kp 45, damping 1.0) supplies stiffness. No joint-target gain or slew limit remains.

**Acceptance checks** (`python -m noesis.experiments.body_check`, no brain)
- Stance, muscles relaxed: height 1.08 mm, upright +1.00, joint error 0.08 rad -> PASS.
- Floor: deepest penetration 20-50 um in every run (inside the 50 um contact skin of `world.py`).
- One leg at a time at the extreme of its range (damping 1.0, legs LF/LH): joints reach the extreme (7/7)
  and 3 of 4 cases overshoot the true limit by 0.000 rad; the loaded hind leg at full antagonist activation
  overshoots by **0.10 rad** (tolerance 0.05 -> FAIL, known limitation: body weight overpowers the soft stop).
  Damping 3.0 reduces it to 0.057 rad but makes the joint too slow (67 ms time constant).
- All 42 joints at their extremes at once (information only): overshoot 0.07-0.09 rad.

**Brain + muscle body** (`newborn_muscles`, 3 s, birth at 0.4 s, da=1, one run each)

| condition | mean joint offset | height (mm) | uprightness mean / min |
|---|---|---|---|
| blind | 0.12 rad | 0.89 | +0.96 / +0.85 |
| leg + balance senses | 0.16 rad | 0.80 | +0.99 / +0.93 |

The fly now stays upright (before: +0.71 / +0.29 with the old position-target mapping). After the brain
wakes it settles into a low crouch; it does not walk. Single runs: not statistically established.

## 2026-10-13 — leg through the body (the "leg in the eye" problem)

FlyGym collides legs only with other legs; the head, eyes, thorax and abdomen are not in the self-collision
set, so a leg passes through them.

- Measured (10 random extreme poses of both front legs, reached over 100 ms): **worst leg-to-body overlap
  520 um** (a front femur inside the eye/thorax ellipsoid). Same 3 poses: 464 um unguarded.
- **Dynamic solid-body contacts do not work here.** Collision pairs leg x (ellipsoid proxy of each body part)
  slowed the physics 24x (mesh bodies; 2x with proxies) and then **crashed the simulation in plain standing**
  (`BADQACC`) at two drop heights, while the same body without them stands (z 1.12 mm). Making the contacts
  stiffer (absolute spring/damper), softer or giving them a 0.2 mm skin did not reduce the overlap
  (350-600 um) and 5 of 8 poses blew up with the skin. Cause: the tissue servo (force up to 65) overpowers any
  contact, whose strength scales with the ~1e-8 segment inertia.
- **Shipped instead: `CollisionGuard`.** Massless ellipsoid proxies (sized from each mesh's bounding box,
  0.9x) exist only for exact distance queries (`mj_geomDistance`); the commanded equilibrium pose is shortened
  to the last pose that keeps a 0.1 mm clearance (or no closer than now). Result: **464 -> 271 um** on the
  same 3 poses. **This reduces, but does not eliminate, the overlap**: the leg can still enter the eye by up to
  ~0.27 mm in adversarial poses (the proxies already overlap the legs by ~80 um in the standing pose).
- A bug found on the way: the proxy geoms initially had default density and **added mass to the body**, which
  silently changed the whole physics (crouch at 0.6 mm instead of standing at 1.1 mm); they are massless now.

## 2026-10-13 — intrinsic learning in the mushroom body (`mushroom_body.py`, `mb_conditioning.py`)

Population in the BANC table: 4,139 Kenyon cells, 104 MBONs, 318 dopaminergic neurons, 3,007 olfactory
receptor neurons (56 glomerulus types), 699 antennal-lobe projection neurons. **Finding:** the global
`min_syn=5` filter keeps only 778 KC->MBON pairs; most are 1-4 synapses, so the learning substrate was
mostly deleted. `with_weak_pathway_edges` restores the sub-threshold pairs inside the olfactory/MB pathway
(+13,855 edges, 3,109 plastic KC->MBON synapses).

Rule (assumption): dopamine-paired depression of KC->MBON, `dw = -eta * trace(KC) * gate * w`, where the
gate is the brain's own DAN rate above its slow baseline plus 5 Hz; recovery toward the connectome value.
Nothing outside the brain supplies a reward.

Assay (brain only; odour = ORNs of 5 glomeruli, A and B disjoint; MBON response before/after, plasticity
frozen during tests):

| reward | DAN rate in training | plastic weights | MBON response A | MBON response B |
|---|---|---|---|---|
| direct DAN stimulation (60 Hz) | 73 Hz | 0.54 | -56% | -48% |
| none (control) | 1.2 Hz | 1.00 (untouched) | -15% | +0% |
| sugar-taste neurons (200 Hz) | 1.2 Hz | 1.00 (untouched) | -18% | -1% |

- The mechanism is dopamine-gated (control untouched). A first version was **not**: the dopamine baseline
  started at 0, so 1 Hz of DAN noise looked like reward and all three conditions gave the same ~-65%.
- **No odour specificity**: 77% of the Kenyon cells fire for either odour (overlap 95-97% of the union).
  The connectome network does not produce the sparse Kenyon-cell code that specific learning needs.
- **The natural reward path is broken**: sugar-taste neurons at 200 Hz leave the DANs at 1.2 Hz, so nothing in
  the world currently drives dopamine through the connectome. The -15% / -18% change for odour A with
  untouched weights is a repeated-exposure effect of the dynamics, not learning.
- So the fly cannot yet "learn by itself" from the world: the plasticity exists and is validated against
  its controls, but its two prerequisites (sparse odour code, world -> dopamine pathway) do not hold in this
  LIF model.

## 2026-10-13 — why learning from the world fails (`learning_prerequisites.py`)

**Fix applied: APL is inhibitory.** The table predicts the APL neuron (the mushroom body's global feedback
interneuron) as cholinergic; it is GABAergic in the literature. `connectome.NT_OVERRIDES` now corrects it
everywhere (cached weights are re-signed on load), so its KC feedback became inhibition instead of
excitation. Effect on the 5-glomerulus odour assay:

| | APL excitatory (table) | APL inhibitory (fix) |
|---|---|---|
| Kenyon-cell rate | 36 Hz | 21 Hz |
| Kenyon cells active per odour | 77% / 74% | 57% / 50% |
| A/B overlap (of the union) | 95% | 87% |
| conditioning, direct dopamine: MBON A / B | -56% / -48% | -49% / -43% |
| control, no reward: MBON A / B | -15% / 0% | -25% / -8% (weights untouched) |

Still far from the sparse code of real flies (~5-10% of Kenyon cells), still no odour-specific learning.

**Why the code stays dense** (one odour = receptor neurons of 5 glomeruli):
- Antennal lobe: the 26 projection neurons of the stimulated glomeruli fire 218 Hz, but **68% of the
  other projection neurons fire too (87 Hz)**. Local neurons give projection neurons 89,109 excitatory vs
  81,623 inhibitory synapses (by predicted NT) and fire at 142 Hz: lateral excitation spreads the odour across
  the lobe instead of keeping it specific.
- APL fires at ~390 Hz, the spiking ceiling of the LIF model at dt 0.5 ms. The real APL is non-spiking with
  graded inhibition, so the model cannot scale its inhibition with Kenyon-cell activity.
- Kenyon-cell to Kenyon-cell excitation is negligible (median 0 synapses per KC): not a cause.

**Why sugar does not reach dopamine:**
- Excitatory hops from the 541 sugar-taste neurons to the 281 reward dopamine neurons (PAM): with pairs >= 5
  synapses the shortest is 3, only 15 PAM are within 3 hops and 192 are unreachable; with every pair >= 1
  synapse the shortest is 2, 87 within 3 hops, 58 unreachable.
- In the network, activity from 200-400 Hz sugar input dies after hop 3 (22-35 active neurons at hop 3,
  none beyond). In the awake brain PAM fire 0.31 Hz without sugar and 0.30 Hz with it, also when every
  1-4 synapse pair in the whole brain is added. Sugar raises overall brain activity (2.3 -> 3.4 Hz) but not
  the reward neurons.
- Interpretation (not tested here): the real sweet-taste-to-PAM signal is believed to involve
  neuromodulatory (e.g. octopaminergic) and multi-hop routes that a fast, sign-only LIF network with uniform
  thresholds does not carry.

**What would be needed** (each is a modelling decision, not done yet): graded (non-spiking) APL inhibition,
a re-examined antennal-lobe local-neuron sign map, and a neuromodulatory (slow, receptor-type dependent)
channel for the taste-to-dopamine route.

## 2026-10-13 — graded (non-spiking) APL and what limits learning (`GradedAPL`)

The spiking APL saturated at ~390 Hz. `GradedAPL` removes its spike output and injects a continuous
inhibitory input into every Kenyon cell, proportional to that KC's APL synapse count and to the
input-weighted Kenyon-cell rate (new `current_idx` / `current_mv_per_ms` input of `LIFNetwork.run`).
Its one parameter (`gain`) was **calibrated to Kenyon-cell sparseness** (5-glomerulus odour, 600 ms):

| gain | KCs active (A / B) | KC rate | A/B overlap |
|---|---|---|---|
| spiking APL | 57% / 50% | 21 Hz | 87% |
| 1 | 67% / 63% | 28 Hz | 92% |
| 64 | 44% / 42% | 16 Hz | 88% |
| 256 | 31% / 28% | 10 Hz | 80% |
| **1024** (used) | **21% / 17%** | 6.4 Hz | **76%** |
| 4096 | 17% / 13% | 5.5 Hz | 74% |

Global inhibition cannot get below ~74% overlap: the input from the antennal lobe is itself non-specific.

**Conditioning with graded APL (gain 1024) -- the uncalibrated test:**

| | MBON response A | MBON response B | plastic weights |
|---|---|---|---|
| control, no reward | -25.5% | -2.2% | 1.00 |
| direct dopamine, eta 5e-4 | -24.1% | +11.6% | 0.91 |
| direct dopamine, eta 5e-3 | -27.3% | -6.5% | 0.61 |

Learning effect beyond the control: ~2 percentage points for A, ~4 for B. **No odour-specific learning**,
even though the plastic synapses were depressed by 39%.

**Why:** in this table Kenyon cells provide only **23% of all synapses onto MBONs (median 6% per MBON**,
10th-90th percentile 0-59%); the rest is 19,792 excitatory and 21,182 inhibitory synapses from other neurons.
As far as I know Kenyon cells are the dominant MBON input in the literature, so this looks like a property
of the data (possibly synapse detection in the dense mushroom-body lobes) rather than of the model; it is
not verified here. Changing KC->MBON weights therefore barely moves MBON output.

`newborn_muscles --plasticity` now uses graded APL (`--apl-gain`, default 1024).

## 2026-10-13 — brain map (`noesis.viz.brain_map`)

Every downloaded neuron as a dot that lights up when it fires, written live by
`newborn_muscles --brain-map FILE` and drawn by `python -m noesis.viz.brain_map --watch FILE`.
The tables have no neuron positions, so the layout is **schematic**: each neuron sits around a hand-placed
point for its main region (`Top in/out region`) on its own side, brain above, nerve cord below. A
`--coords` CSV (`root_id,x,y,z`) switches to real positions if one is downloaded. 9,477 neurons have no
synapses in the table (`NO_CONS`); they still fire from background noise and are drawn apart.

## 2026-10-13 — MBON dendrite/axon compartments (`compartmentalize_mbons`)

Synapses onto MBONs are re-weighted by location: inside the mushroom body (dendrites) x1, outside (axons)
x `axonal_weight` (an assumption: axonal input barely affects spike initiation). Conditioning with graded
APL (1024), eta 5e-3:

| axonal weight | MBON odour response before | after reward / after control |
|---|---|---|
| 1 (point neuron, earlier) | 3.5 Hz | -27% / -25% |
| 0.2 | 0.03 Hz | ~0 / ~0 |
| 0 | 0.03 Hz | ~0 / ~0 |

- With dendritic input only, **MBONs are essentially silent to odours (0.03 Hz)**: all of the earlier MBON
  odour response came from synapses on their axons, not from Kenyon cells. Plasticity at KC->MBON synapses
  cannot be read out from neurons that the KC input does not drive.
- Back-of-envelope: an MBON has ~120 KC synapses here; with 21% of KCs firing at ~6 Hz they add about
  120 x 0.21 x 0.275 mV x 6 Hz x 5 ms = 0.2 mV of drive, against a 7 mV threshold (>30x short).
- So either KC->MBON synapses are under-counted in BANC, or one detected synapse stands for much more than
  0.275 mV. The FAFB comparison (`compare_mb_coverage`, waiting for the download) decides between them.
- Fixed on the way: `mean_weight_fraction` returned NaN when a plastic synapse had zero initial weight.
## 2026-10-13 — where MBON input comes from (`compare_mb_coverage.py`)

Per-synapse neuropil labels in `connections_princeton.csv` (BANC, all pairs incl. < 5 synapses):

| | synapses onto MBONs | from Kenyon cells |
|---|---|---|
| inside the mushroom body (MB_*) | 10,799 (20%) | **71%** |
| outside (mostly CRE, SMP, LAL) | 42,676 (80%) | 11% |

- **Model limitation:** at their dendrites in the mushroom body, MBONs do get most input from Kenyon cells,
  as expected. But 80% of their synapses are outside the mushroom body (axonal regions). A point-neuron LIF
  gives those the same weight as dendritic input, which drowns the Kenyon-cell signal. This, not only the
  data, explains why KC->MBON plasticity barely moves MBON output.
- **Possible data limitation (unverified):** a Kenyon cell has a median of 9.6 output synapses in this table and
  2.8 onto MBONs; an MBON receives ~121 KC synapses. These look very low for Kenyon cells, which suggests
  under-detected synapses in the dense mushroom-body lobes of BANC. `compare_mb_coverage` reports the same
  numbers for any other dataset (e.g. FAFB) to settle this.

## 2026-10-13 — BANC vs FAFB: Kenyon-cell synapses are missing from BANC

`compare_mb_coverage` on the BANC tables used by the model and on two FAFB v783 exports from
codex.flywire.ai (same "princeton" table format; Buhmann = a different synapse detector, no threshold):

| | BANC (model) | FAFB princeton | FAFB Buhmann |
|---|---|---|---|
| Kenyon cells / MBONs | 4,554 / 104 | 5,177 / 96 | 5,177 / 96 |
| output synapses per KC | **9.6** | 129.5 | 190.2 |
| KC -> MBON synapses per KC | **2.8** | 43.8 | 49.6 |
| KC -> MBON synapses per MBON | **121** | 2,364 | 2,674 |
| MBON input inside the mushroom body | 20% | 59% | 61% |
| KC share of MBON input inside the MB | 71% | 84% | 88% |
| KC share of all MBON input | 23% | 63% | 67% |

- Two FAFB detectors agree with each other and both find **~15-20x more Kenyon-cell synapses** than BANC.
  The BANC tables under-count KC output (most likely in the dense mushroom-body lobes). This, not the
  learning rule, is why MBONs ignore Kenyon cells in the model (the earlier estimate: >30x short of threshold).
- In FAFB, Kenyon cells are the dominant MBON input (63% of all, 84% inside the MB), as expected.
- Consequence: learning in the mushroom body cannot work on BANC connectivity alone. Options: rescale BANC
  KC synapses per (KC type -> target type) to FAFB totals, or take the mushroom-body circuit from FAFB.
- The FAFB download (folder `FABA/`) also contains `coordinates.csv.gz` (positions of FAFB neurons; different
  neuron IDs from BANC) and several 0-byte files whose download failed; none of those were needed here.

## 2026-10-13 — mushroom body taken from FAFB (`with_fafb_mushroom_body`, `--mb-source fafb`)

The edges among projection neurons, Kenyon cells, APL, MBONs and DANs are replaced by FAFB's
(`connections_princeton`, summed over neuropils, signed by the BANC presynaptic neuron), mapped onto
BANC neurons by cell type: PN/APL/MBON/DAN by primary type + side, then type alone; Kenyon cells by a
seeded random one-to-one assignment within KC type + side, then lobe class (KCg / KCab / KCapbp) + side,
then lobe class. BANC edges between two matched neurons are removed; unmatched BANC neurons keep their
BANC edges; everything else (ORN -> PN, all inputs/outputs to the rest of the brain and VNC) stays BANC.

| population | BANC | FAFB | matched | FAFB syn (from this population, inside the circuit) | mapped | BANC syn replaced |
|---|---|---|---|---|---|---|
| PN | 699 | 685 | 636 | 461,232 | 390,118 | 186,107 |
| KC | 4,139 | 5,177 | 3,951 | 563,868 | 382,976 | 39,014 |
| APL | 2 | 2 | 2 | 156,926 | 116,549 | 21,296 |
| MBON | 104 | 96 | 93 | 36,358 | 34,317 | 10,942 |
| DAN | 318 | 347 | 279 | 44,062 | 29,530 | 1,222 |
| **total** | | | 4,961 | 1,262,446 (101,369 pairs) | **953,490 (72,739 pairs, 76%)** | 258,581 |

- **Dropped (unmatched):** FAFB KCs 1,226 (mostly KCapbp: BANC has only 122 alpha'beta' KCs vs 916 in
  FAFB, plus 432 surplus gamma KCs); BANC KCs 188 (alpha-beta KCs beyond FAFB's 1,771, and 17 untyped `KC`);
  BANC DANs 39 (18 untyped `PAM`, surplus PAM08/02/05); FAFB DANs 68 (PAM12/14/03...); MBONs 11 BANC / 3
  FAFB (`MBON15-like`, `MBON17-like`); PNs 63 BANC / 49 FAFB (types named differently, e.g. `CB1296`).
- KC -> MBON in the model goes from 121 to **1,523 synapses per MBON** (FAFB itself: 2,364; the gap is the
  unmatched alpha'beta' KCs and MBONs).
- Caveat: the FAFB princeton table only contains pairs with >= 5 synapses (the no-threshold export failed to
  download), so "pairs >= 1 synapse" is effectively >= 5 on the FAFB side, while the BANC side keeps 1-4.

**Protocol fix (applies to all graded-APL runs):** `mb_conditioning` ran the 300 ms settle between test odours
*without* the graded APL, so Kenyon cells were uninhibited while the antennal lobe was still active (PNs stay at
~80 Hz after an odour ends, in BANC too) and fired a synchronous burst at the next odour's onset. That made the
*second* test odour look much less sparse (FAFB, gain 1024: B 36% active, 1,212 KCs with exactly one spike;
BANC, gain 1024: B 23% -> 17%). The settle now keeps the graded APL on. Earlier BANC tables above were made
with the old protocol.

**APL gain recalibrated on the FAFB circuit** (only parameter calibrated; target 10-20% active KCs per odour,
same 5-glomerulus odours, 600 ms, seed 0, before training):

| gain | KCs active A / B | KC rate A | MBON A / B | A/B overlap |
|---|---|---|---|---|
| spiking APL | 69% / 57% | 34 Hz | 58 / 41 Hz | 81% |
| 128 | 30% / 25% | 7.7 Hz | 22 / 18 Hz | 67% |
| 256 | 20% / 20% | 4.6 Hz | 15 / 12 Hz | 64% |
| **512** (used) | **14% / 12%** | 2.8 Hz | 10.8 / 8.7 Hz | **60%** |
| 1024 | 11% / 8% | 1.9 Hz | 8.3 / 6.4 Hz | 59% |

With FAFB synapses the MBONs now follow Kenyon-cell activity (MBON rate falls with sparser KCs), which they did
not on BANC (3.5 Hz at 21% active KCs).

**Conditioning** (gain 512, seed 0, odour A + direct DAN stimulation 60 Hz for 2 s vs odour A alone; plasticity
frozen in tests). Success criterion fixed in advance: A drops clearly more than in the control *and* B changes
about as in the control.

| MBON model | eta | control A | control B | rewarded A | rewarded B | plastic weights |
|---|---|---|---|---|---|---|
| point neuron | 5e-4 (default) | -16.9% | -0.2% | **-48.4%** | **-33.5%** | 0.95 |
| point neuron | 5e-3 | -16.9% | -0.2% | -61.2% | -49.8% | 0.90 |
| axonal weight 0.2 | 5e-4 | -13.3% | -4.8% | -59.1% | -42.9% | 0.95 |
| axonal weight 0.2 | 5e-3 | -13.3% | -4.8% | -77.0% | -68.8% | 0.90 |

(MBON odour response before training: 10.8 / 8.7 Hz point neuron, 7.6 / 6.3 Hz with axonal weight 0.2;
on BANC with axonal weight 0.2 it was 0.03 Hz.)

- **Learning now moves MBON output** (it did not on BANC): rewarded A drops 31-64 points more than the control.
  Restoring the plastic weights to their connectome values after training puts the response back at control
  level (A 9.0, B 8.5 Hz), so the drop is the plasticity, not a lingering network state.
- **But it is not odour-specific: criterion not met.** B drops almost as much beyond the control as A
  (point neuron, default eta: A -31.5 points beyond control, B -33.3).
- **Why** (per-synapse depression, point neuron, default eta): synapses of KCs active for both odours fell to
  0.52 of their value, those of KCs active only for A (0.96) or only for B (0.95) barely changed. 98% of the KCs
  that respond to B were also active during A training. The KC code for the two odours overlaps 60% even at
  12-14% sparseness, and the depressed synapses are the shared, odour-independent ones -- most likely driven
  by the persistent (~80 Hz) antennal-lobe activity that outlasts each odour, which is a BANC/model property
  (ORN -> PN and the antennal lobe are unchanged), not a mushroom-body one.
- Single seed (0), one odour pair, one learning rule with a single global dopamine signal; no seed statistics.
  eta 5e-3 is the value used in the earlier BANC tables, not tuned here.

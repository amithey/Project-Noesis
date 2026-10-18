# Network calibration: inhibition scale and the sparse regime — 2026-10-16

Branch `feature/network-calibration` (from `feature/fly-survival-drive`). The connectome is **not edited**: all
inhibitory (negative) weights are multiplied by `g_I` when a network is built (`ei_balance_scan.scale_inhibition`,
`--g-inh` in the selectivity scripts). Inhibition is already 44% of total absolute weight at `g_I = 1`.
Motivation: with `g_I = 1` the network is silent or saturated (docs `findings.md`, `findings-sense-audit.md`).

## 1. Operating-point scan (`experiments/ei_balance_scan.py`, one seed, 5-12 points per g_I)
Per point: 0.5 s settle, 1 s baseline, 0.7 s of a moderate stimulus (300 random olfactory receptor neurons,
80 Hz, weight 100); background drive weight 60 at the listed rate. Response = PN rise: silent < 1 Hz,
graded 1-40 Hz with < 40% of neurons active, else ignites.

| g_I | PN level after stimulus | Background rates (Hz) with silent-or-sparse baseline AND graded response |
|---|---|---|
| 1 | ≈ 115 Hz (saturated) | none |
| 1.5 | ≈ 85 | none |
| 2 / 3 | ≈ 67 / 46 | 3 only |
| 4 / 5 / 6 / 8 | ≈ 35 / 26 / 20 / 12-18 | 1, 1.5, 2, 3 (a 3-fold range) |
| 12 | ≈ 6-17 | 1, 1.5, 2, 3, 5 (a 5-fold range) |

The strict criterion "baseline must be sparse (≥ 0.5% active)" gives widest runs of only 1-3 points; the
relaxed criterion above also counts a silent baseline with a graded response, which suits a sensory system.
Both are stated; the relaxed one was chosen after seeing the scan (a post-hoc definition). Single seed,
single stimulus, coarse grid: a window is indicated, not established. At background ≥ 5 Hz PN already
fire 20-45 Hz at baseline (background drive hits PN directly), so the stimulus adds nothing there.

## 2. Selectivity at g_I = 4 and 12 (`mb_selectivity.py`, `dn_selectivity.py`, bg 0, graded APL 1024)
Same protocol and nulls as `findings-sense-audit.md` parts 2-3. KC identity p = 0.013 is the smallest value
attainable with 8 codes and 1,000 permutations (105 distinct labelings), i.e. the observed code is the most
structured arrangement possible.

| Measure (ORN weight 400 / 100) | g_I = 1 (earlier) | g_I = 4 | g_I = 12 |
|---|---|---|---|
| PN own vs other glomeruli, Hz (typical) | 160-400 vs ≈ 120 | 57-259 vs ≈ 40 | 90-185 vs 8-22 |
| Glomeruli with own-PN rate above random sets (p < 0.05), of 10 | 7 / 6 | 5 / 3 | 6 / 5 |
| KC active fraction | 25.8% / 27.2% | 12.8% / 12.8% | 9.6% / 8.0% |
| KC within vs between-odour Jaccard | 0.89 vs 0.74 / 0.80 vs 0.72 | 0.88 vs 0.70 / 0.81 vs 0.69 | 0.83 vs 0.58 / 0.69 vs 0.57 |
| KC identity index (null sd) | +0.143 (0.037) / +0.084 | +0.180 (0.043) / +0.123 | +0.254 (0.062) / +0.119 |
| DN odour identity index (p) | +0.001 (0.16) / +0.001 (0.027) | +0.069 (0.074) / +0.010 (0.34) | +0.030 (0.31) / +0.187 (0.056) |
| DN side identity index (p) | −0.000 (0.60) / +0.000 (0.24) | +0.010 (0.28) / +0.086 (0.060) | +0.215 (0.023) / +0.218 (0.010) |
| DN population lateralisation S, Hz (p) | +0.75 (0.001) / +0.65 (0.004) | +0.12 (0.62) / +0.19 (0.65) | −0.19 (0.52) / 0.00 (0.99) |
| DN active fraction | 44% | 17.5% | 6-7% |

What changed:
- Non-targeted PNs drop from ≈ 120 Hz to ≈ 8-22 Hz at g_I = 12, so the olfactory response is far more local to the
  stimulated glomerulus (the *number* of glomeruli with a significant own-PN excess does not increase: 6 versus 7), and the Kenyon-cell code is sparser (≈ 9% active) with less overlap between odours
  (0.58 versus 0.74 between-odour Jaccard) and a larger identity index (+0.25 versus +0.14).
- Descending neurons now carry a side-specific *pattern* at g_I = 12 (side identity index ≈ +0.22, p = 0.01-0.02),
  which was absent at g_I = 1, and a trend toward odour identity at weight 100 (p = 0.056). The population mean
  left-right difference seen at g_I = 1 (S ≈ +0.7 Hz) disappears, so the earlier effect and this one are
  different phenomena.
- Several glomeruli lose their response at g_I = 12 (e.g. DL1, DP1m: own PN 1-5 Hz), so some channels are silenced.

## Limits
Uncontrolled or weak points: one network seed per trial, 10 glomeruli, 4 odours × 2 repeats
(3 for DN), no multiple-comparison correction across the table, a single stimulus protocol, and an
inhibition factor applied uniformly (it also scales the APL synapses that the graded APL reads, so
`apl_gain` was not re-calibrated). An inhibition factor of 12 is a large departure from the measured
connectome and is a model assumption, not a biological estimate; whether the real circuit is better described by
global inhibition, by weaker excitation, or by missing synapses (BANC detects 20-60% of FAFB's synapses) is not
tested here.

## 3. Steering-candidate DNa types on the calibrated networks (`dn_subtypes.py --g-inh`, ORN weight 400)
Same 8 pre-specified groups, 4 odour sets × 3 repeats, 5,000 random same-sized DN groups, Bonferroni alpha 0.0063.

| Group | g_I = 12: S Hz (p) | g_I = 12, shuffled sides: S (p) | g_I = 4: S Hz (p) | g_I = 4, shuffled sides: S (p) |
|---|---|---|---|---|
| DNa family (16 + 16) | +0.03 (0.91) | +0.36 (0.09) | −1.13 (0.21) | −0.21 (0.42) |
| DNa01-04 | +0.58 (0.21) | +0.75 (0.039) | −2.79 (0.086) | −0.75 (0.15) |
| DNa02 | +3.17 (0.041) | +2.33 (0.014) | −2.17 (0.11) | −0.33 (0.38) |
| DNb family | +0.54 (0.18) | +0.04 (0.78) | −1.05 (0.23) | −0.10 (0.70) |
| others (DNa01, 03, 04, DNp09) | none below 0.0298 | | DNa03 −7.83 (0.030) | |

- **No group passes the corrected threshold at either balance, and the unshuffled DNa lateralisation found at
  `g_I = 1` (+7.5 to +8.7 Hz, p = 0.0002) is gone.** Shuffled-side controls give values of the same size
  (e.g. DNa02 +2.3, p = 0.014), so the few nominal p < 0.05 entries are not distinguishable from noise.
- The DNa neurons themselves became nearly silent: the DNa family fires ≈ 0.9 Hz at `g_I = 12` (≈ 100 Hz at
  `g_I = 1`) and 2-4 Hz at `g_I = 4`. A 16-neuron group then yields about one spike per 50 ms control step, too
  little to steer a 20 Hz control loop.
- Together with part 2: stronger inhibition makes the olfactory pathway and descending population sparser and
  more pattern-specific (including a side-specific pattern across all DNs at `g_I = 12`), but the pre-chosen
  steering cells no longer carry it. Any steering read-out on the calibrated network would have to be
  *learned* from the data (e.g. a linear decoder over descending-neuron rates, fitted on some trials and tested on
  others), not taken from the literature.

## 4. A learned linear read-out of side on unseen odours (`experiments/dn_decoder.py`, `g_I = 12`)
60 trials: 10 disjoint odour sets (5 glomeruli each) × {left receptors only, right receptors only} × 3 networks
with different seeds; ORN weight 400, 500 ms windows. Dual-form ridge classifier on log(1 + rate), standardised
on training trials only, evaluated **leave-one-odour-set-out** (the test odour was never in training). Null:
side labels shuffled within each odour set, whole evaluation repeated 500 times (p floor = 0.002).

| Features | n features | held-out accuracy, ridge λ = 0.01 / 0.1 / 1 per feature | label-shuffle null | p |
|---|---|---|---|---|
| Descending neurons | 1,316 | 0.783 / 0.800 / 0.850 | 0.50 ± 0.07 | 0.002 |
| Projection neurons | 699 | 0.800 / 0.833 / 0.867 | 0.50 ± 0.06-0.07 | 0.002 |
| Random non-sensory, non-descending neurons | 1,316 | 0.900 / 0.883 / 0.833 | 0.50 ± 0.07 | 0.002 |

- **Which side an odour came from is linearly decodable from descending-neuron rates on odours never seen in
  training (78-85% versus 50% chance), far beyond the shuffle null.**
- **It is not special to descending neurons.** Random central neurons decode side equally well or better
  (83-90%), and so do projection neurons. The information is spread through the network (consistent with
  ipsilateral drive raising activity across one hemisphere), not concentrated in the output cells; the earlier
  literature-picked DNa group is therefore not the right read-out, but the network does preserve side.
- Limits: 60 trials, 10 odour sets, one weight, one network calibration; the three ridge strengths were fixed
  in advance and are all reported; no confidence interval on accuracy was computed (6 trials per held-out set,
  per-set accuracies are in `outputs/dn_decoder.json`); 500 ms windows, whereas the agent acts every 50 ms; the
  decoder is trained on labelled side, so it is a trained interface to the network, not the network steering
  by itself.

## 5. Real-time (50 ms) side decoding from graded bilateral input: not demonstrated (`experiments/decoder_forage.py`, stages collect + cv)
Calibrated network (`g_I = 12`, graded APL). 10 independent runs × 11 one-second blocks; in each block the two
antennae received receptor drive from concentrations (level × left/right ratio), 6 levels (0.02-2.9) and 6 ratios
(0.67-1.5 left/right concentration; rate ratio is its square root), label = which side was stronger. Features:
log(1 + 250 ms causal average of descending-neuron rates), sampled every 50 ms (2,200 windows, 110 blocks).
Ridge decoder, evaluated on a concentration level never used in training (leave-one-level-out); null = block-level
labels shuffled within level, 100 repeats.

| ridge per feature | accuracy on unseen level | null | p |
|---|---|---|---|
| 0.01 | 0.520 | 0.498 ± 0.039 | 0.33 |
| 0.1 (pre-specified for the agent) | 0.552 | 0.503 ± 0.043 | 0.14 |
| 1 | 0.558 | 0.507 ± 0.042 | 0.11 |

Per unseen level (λ = 0.1): 0.46, 0.48, 0.58, 0.71, 0.59, 0.53 for levels 0.02, 0.05, 0.15, 0.4, 1.0, 2.9.

- **The decoder does not beat chance on unseen concentration levels (55% versus 50%, p = 0.14).** By the
  rule fixed beforehand (run the agent only if the decoder generalises), the agent experiment with this decoder
  was **not run**.
- This is consistent with section 4 yet different from it: there, whole-side stimulation (receptors of one side
  only, 500 ms) was decodable at 78-85%; here the contrast is a graded 1.1-1.5-fold left/right difference
  (a 1.05-1.22-fold receptor-rate difference after the square-root compression), read in 50 ms windows. The
  side information that survives is large for one-sided input and, at this noise level, not usable for the
  gradients a plume provides.
- Not tried (so not excluded): longer integration windows, projection-neuron or all-neuron read-outs,
  larger antenna separation or stronger contrast, and training on more data. The per-window ratio was not saved,
  so accuracy as a function of contrast cannot be broken out without re-collecting.

## 6. Contrast and integration time (`experiments/decoder_contrast.py`, `g_I = 12`)
16 independent runs × 7 blocks of 2 s; 6 concentration levels × 8 left/right ratios (0.33-3); 112 blocks. Features:
log(1 + causally averaged rate) with time constants 250 ms and 1 s, for the 1,316 descending neurons (DN) and 699
projection neurons (PN). Only the last second of each block is analysed (2,240 windows, identical for all sets).
Ridge decoder, leave-one-concentration-level-out, block-shuffle null (60 repeats, p floor 0.016). Contrast =
|ln(left/right concentration)|; the ratio of receptor rates is the square root of the concentration ratio. Block
accuracy = sign of the block-mean score; intervals are Wilson 95% on blocks.

| Features | accuracy on unseen level (λ = 0.01 / 0.1 / 1) | null, p | contrast 0.2 | 0.4 | 0.7 | 1.1 |
|---|---|---|---|---|---|---|
| DN, 250 ms | 0.52 / 0.47 / 0.45 | 0.50 ± 0.04, p = 0.82 | 0.49 | 0.42 | 0.52 | 0.41 |
| DN, 1 s | 0.50 / 0.49 / 0.50 | 0.50 ± 0.04, p = 0.67 | 0.46 | 0.39 | 0.44 | 0.66 |
| PN, 250 ms | 0.72 / 0.72 / 0.68 | 0.50 ± 0.04, **p = 0.016** | **0.61** | **0.68** | **0.77** | **0.81** |
| PN, 1 s | 0.63 / 0.61 / 0.63 | 0.49 ± 0.04, p = 0.033 | 0.50 | 0.57 | 0.63 | 0.74 |

(window-level accuracy per contrast bin, λ = 0.1; block-level accuracy of PN 250 ms: 0.66, 0.65, 0.82, 0.85 of
29, 23, 34, 26 blocks, 95% intervals 0.47-0.80, 0.45-0.81, 0.66-0.92, 0.66-0.94.)

- **Descending neurons do not carry decodable side information at this calibration, at either integration time,
  at any contrast** (chance-level accuracy, null p = 0.67-0.82; the below-chance values at some contrasts are the
  usual leave-one-group-out artefact of a decoder that has learned nothing transferable).
- **Projection neurons do, in 250 ms windows:** 72% on unseen levels (null 50%, p = 0.016), rising with contrast to
  ≈ 80% at contrast ≥ 0.7. At contrasts a plume produces between two antennae 1 mm apart (≈ 0.2-0.4) it is
  61-68% per window (block-level intervals still include 50% for 0.2).
- **A longer average is worse, not better** (PN 1 s: 61-63%): the integration blurs a signal that changes when
  the block changes, and lags the label; note PN feeds are one synapse from the receptors, so this is largely
  periphery, not a computation by the circuit.
- The p = 0.016 for PN 250 ms is the smallest the 60-repeat null allows; there is no correction across the four
  feature sets.

## Consequence for the virtual agent
A side signal usable in real time exists only at the projection-neuron level and only weakly at plume-like
contrast. An agent steered from it would be an olfactory-periphery comparison with a trained read-out, not a brain
that steers; it is a defensible next experiment only if described as that. Not run yet.

## 7. Leg motor and proprioceptive pathway under inhibition scaling (`experiments/motor_g_inh.py`, brain only)
Locomotion-training conditions: corrected connectome without the weak-edge/APL additions, background 3 Hz × weight 60,
leg afferents (3,921) driven by `SensoryMap` at weight 400, 3 network seeds, 0.5 s settle + 1 s measured. 391 leg
motor neurons (left-front tibia: 17 flexor, 2 extensor).

| g_I | motor mean rate, background only → standing afferents (Hz) | motor neurons above 1 Hz, background → standing | whole brain active (standing) | left-front tibia reflex gain (Hz; 3 seeds) | response to +1.5 mV/ms into flexors (Hz) |
|---|---|---|---|---|---|
| 1 | 21.2 → 16.5 | 57% → 48% | 17.7% | −17.0 (−17.8, −13.4, −19.7) | +1.3 |
| 4 | 1.6 → 4.1 | 24% → 29% | 6.5% | −13.4 (−11.8, −12.5, −15.9) | +3.4 |
| 12 | 0.35 → 2.4 | 8% → 25% | 5.4% | −14.7 (−16.5, −11.8, −15.7) | +4.3 |

(Reflex gain = [flexor − extensor pool rate | tibia flexed 0.6 rad] − [same | extended −0.6 rad]; afferent firing is
unchanged by g_I, 48-52 Hz.)

- **Motor neurons are not silenced, but their operating point moves a lot.** At `g_I` = 1 they fire tonically
  (21 Hz mean, over half above 1 Hz even without sensory input, and *higher* without afferents than with them). At
  `g_I` = 4-12 they are near-silent without input (0.35-1.6 Hz) and rise 2.5-7-fold when leg afferents drive them,
  so motor output becomes input-driven rather than tonic.
- **The proprioceptive reflex survives**: the same sign in all 9 runs and a similar size (−13 to −17 Hz) at all three
  `g_I`; its sign means a flexed tibia lowers the flexor-minus-extensor difference (a resistance-type response). The
  extensor pool has only two neurons, so these differences are noisy.
- **Direct current is more effective at larger `g_I`** (+1.3 → +4.3 Hz for the same injected current): the motor pools are
  closer to threshold and less saturated.
- **Consequence for the body:** everything downstream of motor rates (the muscle adapter's agonist/antagonist
  equilibrium targets, the exploration amplitude, the reward-bound weight limits) was calibrated at `g_I` = 1 with
  tonic 10-20 Hz motor rates. At `g_I` > 1 the same pipeline would receive much lower, input-dependent rates, so a
  body test needs at least a re-scaling of the rate-to-activation map. Limits: brain only (no feedback loop), one
  proprioceptive test on one leg, 3 seeds, not the full training drive (no habitat, balance or exploration input).

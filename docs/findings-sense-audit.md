# Sense audit: does an odour reach steering-relevant output? — 2026-10-16

**Update (same day, see 'Part 2'):** the population-mean audit below was blind to odour-specific structure. A per-glomerulus / per-Kenyon-cell test with shuffle nulls finds a real but modest specific signal on top of a large nonspecific one.

## Part 1 — population means (this part alone is not evidence the brain cannot discriminate)

Brain only, corrected BANC brain (158,262 neurons), dt 0.5 ms, adaptation `da=1.0`. Scripts:
`experiments/sense_audit.py` (odour → mushroom body, dopamine cells, descending neurons) and
`experiments/operating_point.py` (spontaneous-activity regime). Every comparison has a control:
no stimulus, and the same number of receptor neurons drawn at random from other olfactory channels.

### Result of part 1: population means match the random-receptor control at every operating point tried
Stimulus-evoked changes are indistinguishable from the random-receptor control, and the left/right
asymmetry of descending-neuron responses does not depend on which side is stimulated.

| Setting (spontaneous background, odour weight) | What happens | Stimulus vs random control |
|---|---|---|
| none, odour w 20 | nothing responds (0 Hz everywhere) | identical (both zero) |
| none, odour w ≥ 60 (80 Hz) | whole brain ignites: PN +113, KC +48, MBON +52, DN +28 Hz | identical |
| none, w 400, 9 channels × 3 seeds | PN/KC/MBON/DN/PPL1 rise equally in all 9 channels, including a 10–26-receptor channel; DN asymmetry ≈ −0.10 for every channel and side | identical (e.g. fruity DN +28.6 vs +28.3 Hz) |
| 5 Hz / w 90 (previous default) | baseline already saturated: PN ≈ 115, KC ≈ 47, 35% of neurons active; odour adds ≤ +4 Hz | within noise of control |
| 1.5 Hz / w 60 (only sparse regime found) | baseline PN 13.6, KC 7.1 Hz; odour w 60/150 ignites PN ≈ +99–103 Hz; DN +5–7 Hz | identical to control (e.g. fruity both: PN +99.6 vs +98.8) |

Operating-point scan (no stimulus): ≤ 1 Hz background is silent; ≥ 2 Hz with weight ≥ 60 is saturated
(PN ≈ 115 Hz); weights ≤ 30 stay silent up to 40 Hz. The sparse window is only 1.25–1.75 Hz at weight 60,
next to the saturation cliff. This extends the bistability already logged in `findings.md`.

## What part 1 does and does not show
- Shows: with these drives, mean rates over whole populations carry no odour identity, valence or side
  (the ~90 Hz PPL1 rise in every channel is the same nonspecificity).
- Does not show: that the connectome cannot discriminate odours. Means over all projection neurons hide
  structure; part 2 looks for it directly.

## Part 2 — per-glomerulus and per-Kenyon-cell selectivity with GradedAPL (`experiments/mb_selectivity.py`)
Corrected brain + weak olfactory/MB edges, graded APL gain 1024, no background activity, 200 ms settle +
500 ms drive at 80 Hz, 51 glomeruli that have both receptors and uniglomerular PNs, 10 tested, each on a
fresh network. Nulls: random PN sets of equal size (1,000 draws); odour labels permuted over the 8 Kenyon-cell codes.

| ORN weight | Glomeruli with own-PN rate > random sets | Own PN rate vs other PN rate | KC active | KC within-odour / between-odour Jaccard | Identity index (null) | p |
|---|---|---|---|---|---|---|
| 400 | 6 of 10 at p < 0.01 (two more at p < 0.1) | 159–398 Hz vs ≈ 120 Hz | 25.8% | 0.89 / 0.74 | +0.143 (0.002 ± 0.037) | 0.013 |
| 100 | 5 of 10 at p < 0.05 (one is at 0.598) | 111–367 Hz vs ≈ 120 Hz | 27.2% | 0.80 / 0.72 | +0.084 (0.002 ± 0.030) | 0.023 |

Interpretation:
- Real wiring-specific signal exists: driving a glomerulus raises its own PNs well above random PN sets,
  and the Kenyon-cell code for the same odour is more similar across independent networks than codes of
  different odours.
- It sits on a large nonspecific component: non-targeted PNs fire at ≈ 120 Hz and ≈ 26% of Kenyon cells
  are active, so about 74% of two different odours' Kenyon-cell codes overlap (as found earlier,
  `findings.md`). Odour identity is a modest modulation of an ignited pathway, not a sparse code.
- Limits: 10 of 51 glomeruli; 4 odours × 2 repeats; one odour-set draw; no background activity (the
  silent-baseline regime, which part 1 showed is a cliff-edge); simulated receptor drive, not natural odour.
- Not yet tested: descending neurons. Whether any of this reaches output cells that could steer
  (left/right differences, odour identity in descending-neuron responses) is open.

## Part 3 — descending neurons: identity and side (`experiments/dn_selectivity.py`)
Same configuration as part 2 (graded APL gain 1024, no background, 200 ms settle + 500 ms at 80 Hz).
All 1,316 descending neurons (651 left-soma, 665 right-soma), full rate vectors, 1,000-permutation nulls.
Odour identity: 4 odours × 3 independently seeded repeats. Side: 3 odour sets driven through left-side or
right-side receptors only × 2 repeats.

| ORN weight | DN active | Odour identity index (null) | p | Side identity index | p | Lateralisation S (null sd) | p |
|---|---|---|---|---|---|---|---|
| 400 | 44.4% | +0.001 (0.000 ± 0.001) | 0.162 | −0.000 | 0.600 | +0.75 Hz (0.17) | 0.001 |
| 100 | 44.3% | +0.001 (0.000 ± 0.001) | 0.027 | +0.000 | 0.238 | +0.65 Hz (0.21) | 0.004 |

- **Odour identity is essentially absent at the DN level.** The index is ~0.001 correlation units: repeats
  of one odour are no more alike than different odours in any practical sense (p = 0.027 at one weight is
  statistically nominal, but the effect is negligible). The descending population is ≈ 44% active and its
  rate pattern is dominated by network-state noise, not by which odour was presented.
- **A small ipsilateral lateralisation exists.** Left-soma DNs respond ≈ 0.7 Hz more (relative to right-soma
  DNs) when left receptors are driven than when right receptors are driven, p ≤ 0.004 at both weights.
  That is ≈ 3% of the DN mean response (≈ 28 Hz); no side signal appears in the full-vector pattern test.
- Not established: that any DN subset steers, which direction a DN pattern turns the fly, or that a 0.7 Hz
  population difference is large enough to steer a body. All DNs were pooled; known steering types
  (e.g. DNa) were not analysed separately. Two weights, one odour-set draw, small repeat counts.

## Gate status
Stage 1: odour identity is represented at projection-neuron and Kenyon-cell level (part 2) but not in
descending-neuron output (part 3); a faint ipsilateral side bias exists. A virtual agent steered by pooled
descending neurons would be driven mostly by noise, so stage 4 stays **blocked** pending one of: a
steering-type subset analysis, a stronger/more natural graded bilateral input, or a better-calibrated
operating point.

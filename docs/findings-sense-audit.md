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

## Gate status
Stage 1 is **partly passed**: odour identity is represented at the projection-neuron and Kenyon-cell level.
It is **not** passed for steering. Stage 4 stays blocked until a per-neuron test on descending neurons
(odour identity, and left- versus right-sided stimulation) beats shuffle nulls.

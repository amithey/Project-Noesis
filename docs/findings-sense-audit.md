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
| 400 | 7 of 10 at p < 0.01 (one more, DM3, at p = 0.098) | 159–398 Hz vs ≈ 120 Hz | 25.8% | 0.89 / 0.74 | +0.143 (0.002 ± 0.037) | 0.013 |
| 100 | 6 of 10 at p < 0.05 (one, VA1v, at 0.598) | 111–367 Hz vs ≈ 120 Hz | 27.2% | 0.80 / 0.72 | +0.084 (0.002 ± 0.030) | 0.023 |

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

## Part 4 — steering-candidate DN types, with controls (`experiments/dn_subtypes.py`)
Groups fixed before looking (8): DNa01, DNa02, DNa03, DNa04, their union, the numbered DNa family (16 left + 16
right), the numbered DNb family (13 + 13), DNp09. That these types steer is taken from memory of the
literature and is not verified here. Statistic S = (left-soma minus right-soma rate | left receptors driven)
minus (same | right receptors driven), mean over 4 odour sets × 3 repeats; null = 5,000 random same-sized DN
groups; Bonferroni alpha = 0.0063.

| Group | w400 S (p) | w100 S (p) | w400, equal receptor counts S (p) | w400, shuffled sides S (p) |
|---|---|---|---|---|
| DNa family | +7.5 Hz (0.0002) | +8.7 Hz (0.0002) | +12.1 Hz (0.0002) | +0.1 Hz (0.79) |
| DNa01–04 | +11.5 (0.0018) | +10.7 (0.0056) | +17.2 (0.0014) | +0.6 (0.40) |
| DNa02 (1 pair) | +33 (0.003) | +31 (0.0046) | +59 (0.0002) | −1.7 (0.22) |
| DNb family | +3.4 (0.025) | +4.9 (0.0060) | +3.5 (0.088) | +0.6 (0.25) |
| DNa01, DNa03, DNa04 | not significant | not significant | DNa03 p = 0.035 only | not significant |
| DNp09 | silent (0 Hz) | silent | silent | silent |

- The DNa lateralisation replicates at two drive strengths, **survives equalising the number of left and right
  receptors** (it gets larger), and **disappears when receptor side is shuffled** (S ≈ 0 in every group). So it
  depends on which side's receptors are driven, not on total input or on the DN selection.
- The DNb result does not survive equalising counts, so it was a receptor-count effect.
- The effect is one-sided: it comes from right-soma DNs responding more to right-side drive (e.g. DNa family
  103 → 116 Hz); left-soma DNs show no change (98.6 vs 99.4 Hz). Causes not tested (data asymmetry, left-side
  ceiling at up to ~360 Hz in some DNs, or both).
- DNa rates of 100–360 Hz are near saturation; the lateralisation is ~8–12% of the mean response.
- Still not shown: that these cells steer a body in this model, the sign of any turn, or odour-specific
  steering. DNp09 never fires in this setup.

## Gate status
Stage 1: odour identity is represented at projection-neuron and Kenyon-cell level (part 2) but not in
descending-neuron output (part 3); the DNa family carries a real, control-surviving ipsilateral side signal
(part 4), one-sided and small. A virtual agent may use **side only** (not odour identity) through the DNa group
as a minimal steering signal, with these limits stated. Odour-specific steering is not supported by these data.

# Sense audit: does an odour reach steering-relevant output? — 2026-10-16

Brain only, corrected BANC brain (158,262 neurons), dt 0.5 ms, adaptation `da=1.0`. Scripts:
`experiments/sense_audit.py` (odour → mushroom body, dopamine cells, descending neurons) and
`experiments/operating_point.py` (spontaneous-activity regime). Every comparison has a control:
no stimulus, and the same number of receptor neurons drawn at random from other olfactory channels.

## Result: the gate failed at every operating point tried
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

## What this does and does not show
- Shows: with this model and these drives, any strong receptor input either does nothing or drives
  the whole olfactory pathway to its ceiling, so the means in this audit carry no odour identity,
  no valence (fruit vs aversive) and no side. The earlier note that PPL1 rises ~90 Hz in every channel
  is the same nonspecificity.
- Does not show: that the real connectome cannot discriminate odours. The audit averages all
  projection neurons and omits what limits Kenyon-cell activity in the earlier conditioning assay
  (`GradedAPL`); per-glomerulus and per-Kenyon-cell specificity were not examined here. Results are
  single-network runs, 1–3 seeds, 2 of 9 channels at the sparse operating point.
- Consequence: no descending-neuron steering signal is established, so stage 4 (virtual agent steered by
  descending neurons) is **not started**, and the stage 3 learning assay was not run on this basis.

## Next options (untested hypotheses)
1. Repeat the audit inside the mushroom-body assay with graded APL inhibition and report per-glomerulus
   projection-neuron and Kenyon-cell selectivity rather than population means.
2. Recalibrate the operating point (inhibition/excitation balance) so a wide sparse regime exists; this
   touches the shared brain model and should be coordinated with the locomotion track.

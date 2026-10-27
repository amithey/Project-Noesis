# Assumed reward on the FAFB mushroom body: does the learning machinery work when reward is handed to it? — 2026-10-28

**SHORTCUT, NOT A CONNECTOME RESULT.** Sugar-receptor drive does not raise dopamine-cell activity in this model (BANC or the FAFB mushroom body;
`findings-fafb-mushroom-body.md`). Here the missing route is replaced by an assumption, so nothing below says that a fly's sugar reward reaches these
cells this way. Two assumptions are built in: (1) the reward cells are the PAM types `PAM01`-`PAM08` (211 of the 318 dopamine cells; chosen by name, not
found by the model); (2) they are driven at 20 Hz x hunger gain while "sugar is touched" (weight 400), with the hunger gain taken from
`HomeostaticState.sensory_gain("sugar")` (fed 1.0, hungry 1.8), i.e. hunger scaling the reward is also assumed. The 20 Hz was fixed after one
calibration seed (dopamine rate 10 -> 30 Hz fed, 43 Hz hungry) and was not tuned on results.

`src/noesis/experiments/mb_assumed_reward.py`; brain = corrected BANC + FAFB mushroom body, MBON dendrite/axon compartments (axonal weight 0), graded APL 512,
`g_I` = 1. Fresh network per run; MBON response to odour A and B measured with plasticity frozen before and after; learning index = (dA - dB) / before A
(negative = learning). 6 seeds (each also draws a new odour pair), paired by seed. Odour B never appears in training.

Schedules (odour A exposure identical within a family; reward amount identical for paired and unpaired):
- **paired**: odour A + reward in blocks 0 and 2 of four 500 ms blocks; **unpaired**: reward in blocks 1 and 3 (odour absent); **none**.
- **paired_gap / unpaired_gap / none_gap** (eight 250 ms blocks): paired = A + reward in the first 500 ms; unpaired = A in the first 500 ms and the
  reward 1.25 s later. The gap was added after the first design: its unpaired reward arrives right after the odour, within the 500 ms Kenyon-cell
  eligibility trace, and the plastic weights came out equal in paired and unpaired runs (0.9872 vs 0.9885) — a leaky control.
- Hungry/none equals fed/none by construction, so fed/none is the no-reward control for both states.

## Results (paired - control learning index; negative = learning; mean ± sd over 6 seeds, seeds with negative difference, paired t)
| Comparison | eta 5e-4, adjacent blocks | eta 5e-4, gap | eta 5e-3 (10x), gap |
|---|---|---|---|
| fed: paired - none | -0.049 ± 0.128 (3/6, t -0.9) | +0.011 ± 0.035 (4/6, t 0.8) | +0.052 ± 0.137 (2/6, t 0.9) |
| fed: paired - unpaired | +0.001 ± 0.068 (3/6, t 0.1) | +0.029 ± 0.044 (1/6, t 1.6) | +0.038 ± 0.086 (1/6, t 1.1) |
| hungry: paired - none (fed/none) | -0.031 ± 0.141 (2/6, t -0.5) | -0.027 ± 0.062 (3/6, t -1.1) | +0.011 ± 0.084 (3/6, t 0.3) |
| hungry: paired - unpaired | -0.043 ± 0.055 (5/6, t -1.9) | +0.041 ± 0.045 (0/6, t 2.2) | **-0.077 ± 0.053 (6/6, t -3.6)** |
| hungry paired - fed paired | +0.018 ± 0.113 (3/6) | -0.038 ± 0.051 (4/6, t -1.8) | -0.041 ± 0.114 (3/6, t -0.9) |

Mean learning index per condition (eta 5e-4, adjacent blocks): fed paired -0.280, unpaired -0.281, none -0.231, hungry paired -0.262, unpaired -0.219.
Plastic weights relative to the connectome value: none 0.996, paired 0.987 (fed) / 0.982 (hungry), unpaired 0.9885 / 0.984; at 10x: none 0.967,
paired 0.948 / 0.940, unpaired 0.953 / 0.946. Dopamine rate during reward: ≈ 29 Hz fed, ≈ 42 Hz hungry (≈ 10 Hz otherwise).

## Reading
1. **The assumed reward does engage the plasticity**: dopamine cells go to 29-43 Hz and the KC→MBON weights depress further than without reward
   (0.987 vs 0.996), more when hungry (0.982) — the one place where hunger does something in the model, because it was put there.
2. **It does not produce odour-specific (associative) learning.** With the original plasticity rate, paired is not more negative than no-reward or than
   the unpaired control in either state (|t| < 2.3 everywhere except the one hit in point 3, and 4 of the 6 paired - unpaired differences are positive, the wrong direction). The index falls by about -0.2 even with no reward at all (the response to the repeatedly presented odour A adapts while B rises),
   which is larger than the 0.4-1.2% weight change dopamine adds.
3. **One nominal hit, which I do not count as learning:** hungry paired - hungry unpaired at 10x plasticity, 6 of 6 seeds negative (-0.077, t -3.6, exact sign test
   p = 0.031 two-sided). It arises because the *unpaired* hungry runs have the weakest index of all conditions (-0.164), while the hungry paired runs (-0.241)
   are no lower than no reward (-0.252); the fed comparison goes the other way (+0.038), and the same contrast with adjacent blocks at eta 5e-4 and with the
   gap at eta 5e-4 has 5 of 6 and 0 of 6 negative respectively. It is one of ≈ 15 comparisons in this document (Holm over the five of that run: p ≈ 0.08), so it
   is a lead to re-test with new seeds, not a result.
4. **Hunger**: hungry paired - fed paired is -0.04 to +0.02 with |t| < 1.8 in every design, i.e. no reliable effect of hunger on what is learned, even with the
   assumption that hunger multiplies the reward.

## What this does and does not show
- Shows: giving the model an explicit reward on a plausible set of reward cells does not, with this plasticity rule, 2 s of training and 6 seeds, yield a
  detectable odour-specific association. The Kenyon-cell code of two odours still overlaps (≈ 46% between-odour Jaccard on this brain), and the rule only
  depresses, so whatever is attached to the pre-synaptic KC trace of A also depresses B's shared KCs.
- Does not show: that the biology or a better-tuned model cannot learn. Not tried: a different reward subset, graded or lasting reward, longer training or repeated
  trials (a real conditioning protocol uses several pairings), plasticity of KC→MBON that can also potentiate, `g_I` > 1 for a sparser code, more seeds for the
  single nominal hit.
- Because the reward is assumed, a positive result here would have said only that the machinery can learn; the negative result says the bottleneck is also
  inside the machinery (plasticity size relative to adaptation, non-specific depression), not only in the missing sugar route.

## Reproduce
```
PYTHONPATH=src python -m noesis.experiments.mb_assumed_reward --data DATA --mb-source fafb --apl-gain 512 --mbon-axonal-weight 0 --seeds 6
... --conditions fed/paired_gap,fed/unpaired_gap,fed/none_gap,hungry/paired_gap,hungry/unpaired_gap [--eta 5e-3]
```
Result files: `docs/fafb-mb-results/ar_*.json`.

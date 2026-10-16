# Stage 3: odour + sugar conditioning, fed vs hungry — 2026-10-16

`experiments/mb_foraging_conditioning.py`. Corrected brain + weak olfactory/MB edges, graded APL gain 1024,
KC→MBON dopamine-gated depression (eta 5e-4), fresh network per run, 3 seeds (each seed also draws a new
odour pair), odour 80 Hz, 2 s training, MBON response to A and B measured with plasticity frozen before and
after. Hunger scales the sugar-receptor drive only (energy 0.2 → gain 1.8). Learning index =
(ΔA − ΔB) / baseline A; negative = paired odour weakened relative to the unpaired one.

## Result: no sugar- or hunger-dependent learning detected
| Paired comparison (mean ± sd over 3 seeds) | Difference in learning index | Seeds negative |
|---|---|---|
| fed: sugar − no reward | +0.009 ± 0.075 | 2 of 3 |
| hungry: sugar − no reward | −0.003 ± 0.064 | 1 of 3 |
| hungry sugar − fed sugar | −0.013 ± 0.033 | 2 of 3 |
| positive control, fed: direct dopamine drive − no reward | −0.043 ± 0.040 | 3 of 3 |

- The index also falls without any reward (−0.07 to −0.14): MBON responses to the paired odour drop on their
  own (adaptation/habituation-like), and that baseline swamps any sugar effect.
- **Sugar did not engage the dopamine cells.** Mean rate of the dopaminergic population during training was
  9.6–9.9 Hz with sugar versus 9.0–9.7 Hz without; plastic weights ended at 0.991–0.993 of the connectome value in
  both. Hunger therefore had nothing to act on (the hungry/no-reward runs are identical to fed/no-reward by
  construction, as hunger only scales sugar drive).
- **Even the positive control is weak.** Driving dopamine neurons directly (≈ 77 Hz) depressed the plastic
  weights to 0.94 and gave a consistent but small index shift (−0.04, 3 of 3 seeds, n = 3), and the unpaired odour
  B also dropped (e.g. 32.5 → 25.0 Hz), so the effect is not odour-specific. This matches the earlier finding in
  `findings.md` that Kenyon cells supply only ~23% of MBON input here, so changing KC→MBON weights barely moves
  MBON output.

## What this does and does not show
- Shows: in this model, sugar-receptor drive plus a paired odour does not produce measurable associative
  learning, and hunger as implemented has no effect on learning.
- Does not show: that real hunger-dependent learning cannot work with this connectome. Three seeds; one APL
  gain; one reward route (sugar-receptor neurons, not a specific PAM subset); one plasticity rate.

## Follow-up: which dopaminergic types does sugar reach? (`experiments/sugar_pathway.py`, `scripts/graph_reach.py`)
Sugar gustatory neurons (541) driven at 200 Hz, weight 400, graded APL gain 1024, 3 seeds, compared with bitter
(179) and water (24) gustatory neurons and three size-matched random afferent sets.

- **0 of 33 dopaminergic cell types are sugar-selective.** Sugar changes PAM types by 0.0–0.4 Hz and the
  controls (especially the random afferent sets) change them as much or more (e.g. PAM08: sugar +0.20 Hz,
  random sets +1.4 to +1.5 Hz; PAM01: +0.21 versus +0.3 to +0.5). The top row, PPM1201 (+1.5 Hz, 4 neurons),
  exceeds the best control by only 1.3 Hz, below the 2 Hz criterion.
- **Graph reach (excitatory edges, counts only):** from sugar neurons the feeding motor neuron MN9 is reached in
  2 hops and 87 of 281 PAM neurons in 3 hops (209 in 4), but random afferents reach 172 of 281 PAM in 3 hops,
  so the graph does not single out sugar → reward either.
- So in this model sugar does reach the feeding motor neuron (consistent with the earlier battery) but there is
  no sugar-specific route to a reward subset to rescue; trying PAM-subset learning on this evidence would mean
  injecting an assumed reward, not testing a connectome pathway.

## Possible fixes (untested)
1. Drive an assumed reward-coding PAM subset in proportion to sugar contact (an explicit shortcut, not a
   discovered pathway; must be labelled as such, like the body-reward in the locomotion track).
2. Make the KC share of MBON input larger via the existing `compartmentalize_mbons`.
3. Leave associative learning aside; use hunger only as a gain on innate responses and test it directly.

# Does a FAFB mushroom body make the fly brain learn? — 2026-10-26

Branch `feature/fafb-mushroom-body` = `feature/network-calibration` + merge of `feature/mb-from-fafb` (clean, 14 mushroom-body tests pass).
Question (from the connectome audit, `connectome-audit-banc-vs-fafb.md`): BANC under-counts the mushroom-body and dopamine pathways
10-300x; if the Kenyon-cell (KC) and learning results were limited by that, they should change when the mushroom-body circuit
(PN, KC, APL, MBON, dopamine edges) is replaced by the FAFB one, matched to BANC neurons by cell type (`with_fafb_mushroom_body`).
Everything else in the brain is unchanged (corrected BANC, weak olfactory/MB edges, `g_I` = 1, graded APL). Brain only, no body.

## Data caveat (my error, disclosed)
`FABA/classification.csv.gz` (FlyWire class and side per neuron) was deleted in the earlier clean-up. `src/noesis/fafb_tables.py` rebuilds an
equivalent table from the two files that remain: class from the cell-type name (KC, APL, MBON, PAM/PPL/PPM) and, for olfactory projection neurons
(PN), from connectivity (>= 50% of input synapses in antennal-lobe neuropils, < 50% of output synapses there); side from the L/R neuropil suffix
holding most of a neuron's input synapses (dendritic side, a proxy for soma side). Resulting populations: PN 570 (the FlyWire annotation gave 685),
KC 5,177, MBON 96, DAN 347, APL 2 — all but the PN count equal the earlier numbers. The PN set is therefore smaller (it excludes some
multiglomerular/atypical PNs); `load_fafb_mushroom_body` is still used when the original file is present. Unit test: `tests/test_fafb_tables.py`.
Matching report (printed by every run): PN 523 of 699 BANC neurons matched, KC 3,951 of 4,139 (1,226 FAFB KCs of subtypes `KCg-m`, `KCapbp-*`,
`KCg-d` have no BANC counterpart by name and are not used), MBON 93 of 104, DAN 279 of 318, APL 2 of 2.

## 1. Kenyon-cell selectivity: clearly better (`mb_selectivity --mb-source fafb`)
Same protocol and odour sets as `findings-sense-audit.md` (ORN weight 400, 80 Hz, 500 ms, 10 glomeruli, 4 odours x 2 repeats, 1,000 permutations).

| | BANC (graded APL 1024) | FAFB MB, APL 512 | FAFB MB, APL 1024 |
|---|---|---|---|
| Glomeruli with own-PN rate > random sets (p < 0.05) | 7 of 10 | 7 of 10 | 7 of 10 |
| Non-target PN rate | ≈ 120 Hz | ≈ 112 Hz | ≈ 112 Hz |
| KC active | 25.8% | 17.0% | 10.9% |
| KC Jaccard within-odour / between-odour | 0.89 / 0.74 | 0.79 / 0.46 | 0.78 / 0.40 |
| KC identity index (null sd) | +0.143 (0.037) | +0.328 (0.082) | +0.385 (0.095) |
| p (smallest attainable ≈ 0.01) | 0.013 | 0.013 | 0.013 |

- With the FAFB mushroom body the KC code is sparser and the overlap between two different odours drops from 74% to 40-46%, without scaling
  inhibition (`g_I` = 1). The identity index is 2.3-2.7x larger than on BANC and larger than the best BANC value (+0.25 at `g_I` = 12).
- The projection-neuron layer is unchanged (the transplant touches PN→KC, not the olfactory receptor→PN stage): the non-specific ≈ 112 Hz PN
  activity remains, so the KC improvement comes from the FAFB PN→KC / KC→APL wiring, not from sparser PNs.
- Limits: four odours, eight codes, one odour-set draw, silent baseline; p is at its attainable floor.

## 2. Associative learning: not demonstrated (`mb_foraging_conditioning --mb-source fafb`, 4 seeds, eta 5e-4, 2 s training)
Fresh network per run; odour A paired with sugar-receptor drive (reward reaches dopamine cells only through the connectome), odour B never paired;
MBON response to A and B measured with plasticity frozen before and after. Learning index = (dA − dB) / before A (negative = paired odour weakened);
a conditioned effect needs sugar < no-reward, paired by seed. Hunger scales the sugar drive only (`HomeostaticState`).

| Setting | fed: sugar − none | hungry: sugar − none | direct dopamine − none | dopamine rate, sugar vs none | MBON A at baseline |
|---|---|---|---|---|---|
| FAFB MB, point-neuron MBONs, APL 512 | +0.03 ± 0.09 (1 of 4 neg.) | +0.04 ± 0.05 (1 of 4) | +0.004 ± 0.08 | 9.0 vs 9.1 Hz | 32 Hz |
| FAFB MB, MBON dendrite/axon compartments | +0.06 ± 0.10 (1 of 4) | +0.14 ± 0.13 (1 of 4) | +0.011 ± 0.06 | 10.0 vs 10.1 Hz | 14.6 Hz |
| BANC, MBON compartments (control) | +0.09 ± 0.08 (0 of 4) | +0.10 ± 0.09 (0 of 4) | +0.018 ± 0.08 | 10.7 vs 11.1 Hz | 1.8 Hz |
| FAFB MB, compartments, plasticity 10x (eta 5e-3) | −0.02 ± 0.16 (3 of 4) | −0.04 ± 0.08 (3 of 4) | −0.08 ± 0.11 (3 of 4) | 10.1 vs 10.4 Hz | 14.6 Hz |

(Negative = learning in the expected direction. Means ± sd over 4 seeds; each seed also draws a new odour pair.)

- **Sugar still does not reach the dopamine cells** (rates equal with and without sugar in every setting). The transplant replaces only edges
  among PN/KC/APL/MBON/DAN; the sugar-receptor → dopamine route lives outside the mushroom body and is unchanged. With no reward signal, hunger
  has nothing to act on, as before (the hungry/none runs are identical to fed/none by construction).
- **A real functional gain from the transplant:** MBONs driven by Kenyon cells alone (compartment run) fire at 14.6 Hz versus 1.8 Hz on BANC
  (≈ 8x), i.e. the MBON output is now actually KC-driven, which was the reason learning could not move MBONs on BANC (KC supplied ≈ 23% of MBON input).
- **But learning is not detectable.** Differences are in the wrong direction at the original plasticity rate (sugar − none > 0) and
  consistent in sign (3 of 4 seeds) but not significant (t ≈ −1.5, n = 4) at 10x plasticity for direct dopamine. The no-reward control drifts by
  −0.2 to −0.4 on its own (MBON response to the repeated odour A falls and B rises), which is larger than anything dopamine adds.
- With the original plasticity rate, even 77 Hz direct dopamine moves the plastic KC→MBON weights only to 0.958 of their connectome value.

## 3. Direct-dopamine conditioning with the original assay (`mb_conditioning --mb-source fafb`, MBON compartments, APL 512, 2 s, seeds 0 and 1)
| Reward | MBON response A (change) | MBON response B (change) | plastic weights | dopamine rate |
|---|---|---|---|---|
| none (seed 0 / 1) | −17.2% / −19.0% | +4.6% / +2.7% | 1.000 | 4.6 / 4.4 Hz |
| sugar (seed 0 / 1) | −16.2% / −22.1% | +6.5% / +4.9% | 1.000 | 4.6 / 4.4 Hz |
| direct dopamine drive (seed 0 / 1) | −63.5% / −66.5% | −41.5% / −33.7% | 0.953 / 0.957 | 75 Hz |

- Direct dopamine drive depresses the MBON response strongly (−64 to −67% versus −17 to −19% without reward), so the FAFB mushroom body now carries
  a functional dopamine-gated depression. Sugar does nothing (weights exactly unchanged, dopamine rate identical to no reward).
- The depression is mostly **not odour-specific**: the unpaired odour B falls −34 to −42% (versus +3 to +5% without reward). The A − B gap is
  −22 points (seed 0) and −33 points (seed 1) under dopamine versus −22 points in both seeds of the no-reward control, so the paired-specific part is
  at most ≈ 0-11 points. This matches the KC overlap after training (A 469 and B 461 active KCs, 79% overlap) — the overlap between odours is
  what makes B follow A. Two seeds only; not a statistical test.

## What this supports and what it does not
- Supports: the BANC mushroom-body deficit was a real limit for **Kenyon-cell selectivity** (identity index +0.14 → +0.33/+0.39, between-odour
  overlap 74% → 46%/40%) and for **MBON output being driven by Kenyon cells** (1.8 → 14.6 Hz). Dopamine-gated plasticity now produces a large MBON depression when
  dopamine is driven directly.
- Does not support: any reward-driven or hunger-dependent associative learning. The missing piece is not the mushroom body but the
  **reward route**: sugar-receptor input does not raise dopamine cells in this model (also 0 of 33 types sugar-selective on BANC), and the
  depression that dopamine does cause is largely non-specific. A FAFB-level sugar → dopamine pathway cannot be built from the tables on disk
  by type matching: the FAFB cell-type table has no sugar-labelled receptor type (its only `GRN` names are 71 `claw_tpGRN`/`dorsal_tpGRN` neurons).
- Not tested: PAM-subset reward (an assumed, not discovered, route; would have to be labelled as such), other APL gains in the learning assay,
  longer training or more seeds, `g_I` > 1 combined with the FAFB mushroom body, KC subtype mapping for the 1,226 unmatched FAFB KCs.

## Reproduce
```
PYTHONPATH=src python -m noesis.experiments.mb_selectivity --data DATA --mb-source fafb --apl-gain 512 --weights 400
PYTHONPATH=src python -m noesis.experiments.mb_foraging_conditioning --data DATA --mb-source fafb --apl-gain 512 --mbon-axonal-weight 0 --seeds 4
PYTHONPATH=src python -m noesis.experiments.mb_conditioning --data DATA --mb-source fafb --reward direct --apl-gain 512 --mbon-axonal-weight 0 --seed 0
```
Result files: `docs/fafb-mb-results/`.

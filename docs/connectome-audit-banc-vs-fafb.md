# Graph audit: the connectome used (BANC) against FAFB on the olfactory, mushroom-body and dopamine pathways — 2026-10-26

`scripts/connectome_audit.py` (no simulation). Both datasets are reduced to the same coarse classes from their own cell-type names (ORN, olfactory PN,
Kenyon cell KC, MBON, PAM and PPL1 dopamine cells, APL, descending neurons DN); neuron pairs are summed over neuropils. Counts in the first table use pairs with
>= 5 synapses (the FAFB export only contains such pairs); the second table repeats BANC with every pair (>= 1). Class rules are heuristics from type names.

**This is not a new discovery: the Kenyon-cell under-count was found on 2026-10-13 in the unmerged branch `feature/mb-from-fafb`**
(`docs/findings.md` there: "BANC under-counts Kenyon-cell synapses ~15-20x") together with a transplant of the FAFB mushroom-body circuit
(`with_fafb_mushroom_body`, `--mb-source fafb`). The experiments in this repository's later branches (sense audits, odour/side decoding,
sugar-conditioning, calibration, reward-only read-out) were run on BANC connectivity **without** that correction. This audit independently
reproduces the finding with a different measure and extends it to the dopamine output side.

## Population sizes
| | ORN | PN | KC | MBON | PAM | PPL1 | APL | DN |
|---|---|---|---|---|---|---|---|---|
| BANC | 3,006 | 699 | 4,139 | 104 | 281 | 13 | 2 | 1,316 |
| FAFB | 2,279 | 264* | 5,177 | 96 | 307 | 16 | 2 | 1,299 |
\* the FAFB PN rule (type-name pattern) is narrower than BANC's class label; ORN->PN and PN->KC comparisons are therefore unreliable and are not used below.

## Mushroom-body and dopamine connectivity (pairs >= 5 synapses; "per pre" = synapses per presynaptic neuron of the class)
| Pathway | BANC pairs | BANC synapses per pre | FAFB pairs | FAFB synapses per pre | FAFB / BANC (per pre) |
|---|---|---|---|---|---|
| KC -> MBON | 776 | 1.1 | 26,937 | 43.8 | ~40x |
| KC -> KC | 441 | 0.7 | 24,169 | 29.1 | ~40x |
| KC -> APL | 1,886 | 3.2 | 5,179 | 28.9 | ~9x |
| APL -> KC | 2,178 | 8,700 | 5,178 | 73,511 | ~8x |
| MBON -> PAM | 57 | 4.1 | 327 | 31.8 | ~8x |
| PAM -> KC | **2** | 0.0 | **1,084** | 21.5 | — |
| PPL1 -> KC | **5** | 2.7 | **2,104** | 799 | ~300x |
| PAM -> MBON | 26 | 0.6 | 758 | 38.5 | ~60x |
| MBON -> DN | 102 | 17.8 | 130 | 35.7 | ~2x |
Including every BANC pair (>= 1 synapse): KC -> MBON 3,109 pairs (3.0 per KC), KC -> KC 2,670, PAM -> KC 17 pairs (0.2 per PAM), PPL1 -> KC 69 pairs
(18.5 per PPL1), APL -> KC 10,404 per APL. The gaps shrink but do not close, so the missing connections are not merely sub-threshold.
Kenyon cells supply 53% of the MBON input within the audited classes in BANC even with every pair counted, against 83% in FAFB (thresholded).

## What this explains, and what it does not
- **Explains** why dopamine-gated KC -> MBON plasticity, the graded APL and every learning experiment in this repository behaved so weakly: the synapses the rule
  modifies (KC -> MBON, 40x fewer per Kenyon cell) and the teaching synapses (PAM/PPL1 -> KC and -> MBON, 2 vs ~1,000 pairs) are nearly absent from the table.
  The stage-3 conditioning result (no sugar- or hunger-dependent learning) must therefore be read as "not testable on BANC", not as evidence about the fly.
- **Plausibly contributes to** the bistability (the APL feedback onto Kenyon cells is ~8x weaker than in FAFB; real mushroom bodies are sparse because of it).
- **Does not explain** sugar -> dopamine (sugar receptor neurons have no cell type in the FAFB table that is on disk, so that path cannot be audited here), nor the
  descending-output results, nor the body/locomotion instability.
- Limits: heuristic class rules; FAFB table thresholded at >= 5; one hemisphere-agnostic aggregate; the BANC raw table could differ from the one the project
  loads only through the min-synapse rule already applied.

## Consequence for the plan
The experiments that depend on the mushroom body should be repeated on a brain whose mushroom-body circuit comes from FAFB (the existing, untested-here
`feature/mb-from-fafb` implementation reports 1,523 KC -> MBON synapses per MBON instead of 121, MBONs that follow Kenyon-cell activity, and
odour-A conditioning of -48% against -17% in the control, but also -34% for the unrewarded odour B, i.e. not yet odour-specific). That branch has not been
merged into the current lineage.

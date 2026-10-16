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

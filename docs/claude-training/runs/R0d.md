# R0d — baseline for variant A (native legs, mean muscles), untrained parent

- **Date/time:** 2026-11-26, about 11:10–11:18 local; code `7cc8b67`
- **Command:** `python es_evaluate.py --output <out>\R0d-baseline --label "R0d baseline (variant A: native legs, mean muscles)" --workers 3`
- **Why:** per the legs-fix rule (pick the variant with the best six-leg usage among those with mean standing ≥ 0.88 and uprightness ≥ 0.85; otherwise A),
  A was chosen. Shortly afterwards the coordinator switched training to variant B on Amit's request (see [R0e](R0e.md)). R0d stays as the A reference.

## Leg-variant comparison (parent gains, held-out seeds 20001–20003, 5 s, no video)

| Variant | Seed | standing | upright | forward mm | v7 | extended support LF..RH | debounced steps | stride mm LF..RH |
|---|---|---|---|---|---|---|---|---|
| A native/mean | 20001 | 0.898 | 0.868 | +1.22 | 0.608 | [0.82 0.14 0.36 0.32 0.30 0.76] | [5, 5, 5, 4, 8, 4] | [0.95 0.74 1.26 0.44 0.65 0.74] |
| A native/mean | 20002 | 0.907 | 0.879 | +0.29 | 0.617 | [0.58 0.31 0.71 0.07 0.63 0.30] | [5, 7, 8, 2, 3, 4] | [0.82 1.05 1.77 0.39 1.26 0.49] |
| A native/mean | 20003 | 0.911 | 0.875 | +1.63 | 0.626 | [0.68 0.29 0.53 0.15 0.49 0.52] | [1, 4, 2, 2, 2, 4] | [1.07 1.06 1.46 0.44 1.22 0.56] |
| B real/mean | 20001 | 0.498 | 0.796 | -0.69 | -0.036 | [0.62 0.08 0.46 0.20 0.56 0.45] | [4, 2, 5, 2, 4, 6] | [1.46 1.16 2.03 0.66 1.92 0.94] |
| B real/mean | 20002 | 0.813 | 0.814 | +1.70 | 0.467 | [0.70 0.14 0.45 0.27 0.35 0.73] | [3, 5, 4, 5, 6, 2] | [1.50 0.93 1.62 0.60 1.28 0.92] |
| B real/mean | 20003 | 0.588 | 0.799 | -0.04 | 0.110 | [0.72 0.06 0.56 0.20 0.48 0.61] | [4, 1, 6, 2, 7, 5] | [1.80 1.20 2.12 0.65 1.92 0.98] |
| C real/size | 20001 | 0.320 | 0.751 | -1.28 | -0.319 | [0.54 0.06 0.61 0.13 0.69 0.17] | [2, 1, 3, 1, 5, 4] | [1.73 1.16 2.27 0.86 2.08 1.18] |
| C real/size | 20002 | 0.319 | 0.748 | -0.06 | -0.313 | [0.49 0.11 0.64 0.17 0.51 0.47] | [4, 3, 4, 6, 6, 5] | [1.82 1.28 2.19 0.69 2.18 1.25] |
| C real/size | 20003 | 0.369 | 0.771 | -0.40 | -0.236 | [0.57 0.03 0.55 0.23 0.64 0.55] | [2, 0, 3, 2, 7, 7] | [1.77 1.69 2.07 0.70 2.22 1.14] |

Summary: A mean standing 0.905 / uprightness 0.874 / min-leg extended support 0.07; B 0.633 / 0.803 / 0.06; C 0.336 / 0.757 / 0.03.
Only A qualifies, so the rule picks A. Files: [`legs-cmp/`](legs-cmp/).

## R0d gates (identical episodes to R0; new metrics added)
- Stand gate FAILED (standing ≥ 0.97, every-leg extended support ≥ 0.3, beats parent).
- Walk gate FAILED. Passes: stepping 4+ legs (raw and debounced), every-leg stride ≥ 0.3 mm. Fails: forward ≥ 2 mm, standing ≥ 0.90, all 6 legs ≥ 2 steps, every-leg extended support ≥ 0.2, beats parent.

Video: [`videos/R0d-baseline.mp4`](../videos/R0d-baseline.mp4) (seed 20001, 10 s; forward +1.49 mm, standing 0.896; same episode as R0).
File: [`R0d/evaluation.json`](R0d/evaluation.json)

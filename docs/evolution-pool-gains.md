# Evolution of 84 muscle-pool gains (separable CMA-ES) — 2026-10-26

Branch `claude/locomotion-v25`. Genome: one log-gain per leg joint for its agonist pool and one for its antagonist pool (42 leg joints,
84 parameters, factors limited to 0.14-7.4); it rescales only the pooled motor rate fed to the muscle model (`noesis.pool_genome`);
the connectome and all synapses are untouched. Optimiser: a minimal separable CMA-ES written for this work (`noesis.sep_cma`,
tested on toy problems). Fitness of an individual: mean standing fraction over 3 fresh seeds (pool 1000-1999, disjoint from
development, validation, held-out and training seeds; common random numbers within a generation) in frozen 2 s episodes
(quiet arena, learning and exploration off, same registration as V24/V25). Population 8, 8 generations, sigma 0.25 at the start.
Each generation also scored the distribution mean and the original weights on the same seeds. A check at generation 0: the mean
vector (all zeros) and the original weights gave identical results on every seed.

## Progress during the run (3 seeds x 2 s per generation; the seeds change each generation)
| Generation | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| best individual | 0.94 | 0.97 | 0.96 | 0.94 | 0.96 | 0.98 | 0.96 | 0.97 |
| population mean | 0.82 | 0.77 | 0.89 | 0.82 | 0.85 | 0.78 | 0.84 | 0.80 |
| distribution mean | 0.74 | 0.83 | 0.72 | 0.76 | 0.81 | 0.96 | 0.76 | 0.97 |
| original weights | 0.74 | 0.96 | 0.96 | 0.82 | 0.91 | 0.92 | 0.90 | 0.69 |
| sigma | 0.250 | 0.242 | 0.235 | 0.230 | 0.225 | 0.220 | 0.217 | 0.214 |
No generation failed an episode. The mean difference between distribution mean and original over the eight generations is about
-0.04; the two generations in which the mean exceeded the original (5 and 7) are the ones where the original happened to fall
(gen 7: 0.69). Sigma barely moved, so the search did not converge. Generation 6 took 651 minutes only because the machine was
interrupted. Final genome: mean |log-gain| 0.31 (about 1.36x), 35 of 84 parameters beyond 0.3, largest 1.01.

## Final evaluation of the evolved distribution mean (`mean-final.json`, the candidate fixed in advance)
Frozen 5 s episodes on development seeds 10-19, paired with the original weights, bootstrap 10,000 samples.

| Seed | evolved: standing / slip / forward mm | original: standing / slip / forward mm |
|---|---|---|
| 10 | 0.993 / 15.6 / +10.7 | 0.442 / 9.4 / +8.0 |
| 11 | 0.136 / 8.3 / +9.6 | 0.929 / 16.7 / +12.7 |
| 12 | 0.532 / 10.0 / +6.3 | 0.742 / 13.8 / +8.6 |
| 13 | 0.901 / 13.8 / +4.9 | 0.863 / 15.9 / +12.4 |
| 14 | 0.966 / 15.9 / +10.4 | 0.435 / 14.1 / +7.5 |
| 15 | 0.938 / 15.8 / +12.7 | 0.856 / 14.7 / +8.5 |
| 16 | 0.970 / 14.9 / +8.0 | 0.904 / 16.7 / +12.7 |
| 17 | 0.213 / 7.2 / +6.5 | 0.948 / 17.0 / +8.8 |
| 18 | 0.948 / 15.7 / +9.4 | 0.329 / 7.3 / +7.4 |
| 19 | 0.902 / 16.1 / +14.4 | 0.929 / 15.0 / +8.9 |
| mean (sd) | 0.750 (0.33) | 0.738 (0.24) |

- Difference in standing (evolved - original) **+0.012**, 95% interval **[-0.28, +0.29]**, sd of paired differences 0.49;
  better on 6 of 10 seeds (one-sided sign test p = 0.38). Verdict by the rule fixed beforehand (interval above zero and mean >= 0.75):
  **no evidence of improvement**.
- Secondary (not part of the criterion): the evolved genome stands >= 0.9 on 7 seeds against 4 for the original (one-sided Fisher
  p = 0.18); it flips onto its back on 2 seeds (11 and 17 stand 0.14 and 0.21) against 1 for the original; forward heading
  progress 9.3 vs 9.5 mm; on the two seeds where both stand >= 0.9 (16, 19) the stance slip is similar (14.9 vs 16.7, 16.1 vs 15.0 mm/s).
- Like every earlier candidate, the evolved genome stands very well on some seeds and falls on others, and the seeds on which it
  falls (11, 12, 17) are largely different from the original's (10, 14, 18). The seed spread is not reduced (sd 0.33 vs 0.24).

## Reading and limits
- Evolution on 84 pool gains did not produce a reliably more stable fly within 8 generations x 8 individuals x 3 seeds
  (about 240 two-second episodes, roughly 1.4 simulated seconds of evolution per parameter); the gain in the number of seeds above 0.9
  is a hint, not a result. The noise in an individual's fitness (the same individual changes by 0.3-0.5 between seeds) is as large as
  the differences evolution needs to resolve; the later generations' sigma shows no convergence.
- 2 s fitness episodes may not predict 5 s behaviour; the held-out seeds (200-202), obstacle controls and video were not used.

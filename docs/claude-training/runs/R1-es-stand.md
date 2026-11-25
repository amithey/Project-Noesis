# R1 — ES stand — **ABORTED: design flaw** (varying training seed)

- **Date/time:** 2026-11-25, 19:00–19:09 local; stopped by request after 3 of 20 generations (about 8 min)
- **Command** (from `gpu-pilot`, code at `7a71f53`):

```
python es_train.py --method es --stage stand --generations 20 --pairs 6 --seconds 1.5 --workers 3 --wall-hours 3 --output <project>\gpu-pilot\outputs\claude-es\R1-es-stand
```

- **Config:** OpenAI-ES, 6 antithetic pairs (12 candidates + centre), 1.5 s episodes, sigma 0.08, lr 0.03,
  **1 training seed per generation, a different seed each generation** (30000, 30100, 30200, …), 136 groups / 85,390 edges.
- **Wall time:** 136–188 s per generation with 3 workers.

### Training fitness

| Gen | seed | pop mean | pop max | pop min | centre | best centre | improved | errors | gain mean [min,max] | sigma | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 30000 | 0.6673 | 0.7696 | 0.3102 | 0.7386 | 0.7386 | yes | 0 | 1.000 [0.970,1.030] | 0.08 | 188 |
| 1 | 30100 | 0.5895 | 0.7688 | -0.0460 | 0.4419 | 0.7386 |  | 0 | 1.001 [0.940,1.060] | 0.08 | 142 |
| 2 | 30200 | 0.6436 | 0.7666 | 0.3489 | 0.3978 | 0.7386 |  | 0 | 1.002 [0.911,1.090] | 0.08 | 136 |

Centre standing fraction / uprightness: gen 0 0.979 / 0.878; gen 1 0.797 / 0.857; gen 2 0.772 / 0.884.

### Why it was aborted
- The "best" vector is chosen by the centre's fitness, but each generation scored the centre on a **new** seed.
  Gen 0's centre is the unmodified parent (all gains = 1.0), and it scored 0.739 on seed 30000. The gen 1 and 2 centres
  scored 0.442 and 0.398 on different seeds. Those numbers are not comparable, so "best" mostly tracks seed luck.
- As a result **`best-gains.npy` == parent (all ones)**. A held-out evaluation would have compared the parent with itself.
- The centre drop 0.739 → 0.44 → 0.40 does not separate "the mean got worse" from "harder seeds"
  (the population max stayed at about 0.77 every generation). The mean had moved only a little (gains 0.911–1.090, L2 distance from 1 = 0.73).
- Fix (cloud thread, `b45166f`): one fixed set of training seeds across all generations (2 per generation by default),
  sigma 0.05, lr 0.01; old behaviour available via `--vary-seeds`. Continued as **R1b**.

**Gates:** not evaluated (no eval or video, by decision). **Verdict:** aborted, no result.

Files: [`runs/R1-es-stand/`](R1-es-stand/) (config.json, generations.jsonl, best-gains.npy; no report.json because the process was stopped).

# R4 — ES stand, 4 training seeds — **INTERRUPTED** (legs-fix work took priority)

- **Date/time:** 2026-11-26, 09:09–10:44 local; stopped by request after 7 of 12 generations
- **Continues from:** `--init-gains` = [R1b](R1b-es-stand-fixedseeds.md) best-gains
- **Change vs previous round:** `--seeds-per-gen 4` (step a, against overfitting). Everything else as R1b/R3.
- **Command:**

```
python es_train.py --method es --stage stand --generations 12 --seconds 1.5 --workers 3 --wall-hours 3 --output <out>\R4-es-stand-4seeds --seeds-per-gen 4 --init-gains <out>\R1b-es-stand-fixedseeds\best-gains.npy
```

- **Config:** OpenAI-ES, 6 pairs, sigma 0.05, lr 0.01, fixed seeds 30000–30003, 1.5 s, fine grouping, mean muscles, native legs, 3 workers;
  52 episodes per generation, 573–1062 s per generation (gen 0 shared the CPU with leg audits).

### Training fitness (4-seed mean)

| Gen | pop mean | pop max | pop min | centre | best centre | improved | errors | gain mean [min,max] | sigma | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6659 | 0.7352 | 0.5930 | 0.6998 | 0.6998 | yes | 0 | 1.003 [0.898,1.125] | 0.0500 | 1062 |
| 1 | 0.6656 | 0.7249 | 0.6155 | 0.6634 | 0.6998 |  | 0 | 1.003 [0.905,1.119] | 0.0500 | 645 |
| 2 | 0.6726 | 0.7270 | 0.6175 | 0.6614 | 0.6998 |  | 0 | 1.003 [0.905,1.116] | 0.0500 | 725 |
| 3 | 0.6286 | 0.7063 | 0.5828 | 0.7004 | 0.7004 | yes | 0 | 1.004 [0.907,1.119] | 0.0500 | 712 |
| 4 | 0.6410 | 0.7178 | 0.5644 | 0.6513 | 0.7004 |  | 0 | 1.003 [0.904,1.123] | 0.0500 | 677 |
| 5 | 0.6609 | 0.7302 | 0.5432 | 0.5696 | 0.7004 |  | 0 | 1.003 [0.899,1.126] | 0.0500 | 657 |
| 6 | 0.6632 | 0.7273 | 0.5663 | 0.6558 | 0.7004 |  | 0 | 1.002 [0.899,1.126] | 0.0500 | 573 |

The R1b start scored 0.6998 on the 4 seeds. The best centre after 7 generations (0.7004, gen 3) is essentially unchanged.

**Gates:** not evaluated (interrupted). **Verdict:** interrupted, no measurable training gain in 7 generations.
With 4 seeds the training signal is less noisy but also flatter: no candidate around R1b scored clearly higher.

Files: [`R4-es-stand-4seeds/`](R4-es-stand-4seeds/) (config.json, generations.jsonl, best-gains.npy; no report.json because the process was stopped).

# R2b — sep-CMA-ES stand, fixed training seeds

- **Date/time:** 2026-11-25, 20:38–21:58 training, 21:58–22:05 evaluation (local)
- **Continues from:** parent (no `--init-gains`); replaces the planned R2
- **Code:** `a18e71d` (fixed seeds, all-6-legs walk criterion; stand fitness unchanged)
- **Command** (from `gpu-pilot`):

```
python es_train.py --method cma --stage stand --generations 15 --popsize 12 --seconds 1.5 --workers 3 --wall-hours 2.5 --output <project>\gpu-pilot\outputs\claude-es\R2b-cma-stand-fixedseeds
python es_evaluate.py --gains <run>\best-gains.npy --output <run>\eval --label "R2b-cma-stand-fixedseeds cma stand" --workers 3
```

- **Config:** separable CMA-ES, population 12 (+ centre), 1.5 s episodes, fixed training seeds 30000 and 30001, initial sigma 0.05,
  fine grouping (136 groups), 26 episodes per generation, 3 workers.
- **Wall time:** 80 min of training (257–318 s per generation; gen 2 took 849 s because the PC entered Modern Standby) plus 6.8 min of evaluation.
- **Best:** centre fitness 0.751 at gen 10 (parent 0.706). Best gains: mean 0.998, range 0.811–1.178, L2 distance from 1 = 0.90 (moved further than R1b's 0.53).

### Training fitness

| Gen | pop mean | pop max | pop min | centre | best centre | improved | errors | gain mean [min,max] | sigma | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6565 | 0.7684 | 0.5633 | 0.7063 | 0.7063 | yes | 0 | 1.001 [0.931,1.064] | 0.0486 | 302 |
| 1 | 0.6593 | 0.7300 | 0.5648 | 0.6696 | 0.7063 |  | 0 | 0.999 [0.934,1.080] | 0.0475 | 304 |
| 2 | 0.6625 | 0.7655 | 0.5126 | 0.7266 | 0.7266 | yes | 0 | 1.002 [0.911,1.103] | 0.0464 | 849 |
| 3 | 0.6560 | 0.7536 | 0.4954 | 0.5431 | 0.7266 |  | 0 | 1.001 [0.895,1.112] | 0.0455 | 285 |
| 4 | 0.6277 | 0.7096 | 0.4970 | 0.7485 | 0.7485 | yes | 0 | 1.002 [0.889,1.133] | 0.0447 | 279 |
| 5 | 0.6610 | 0.7378 | 0.5498 | 0.6473 | 0.7485 |  | 0 | 1.004 [0.879,1.141] | 0.0439 | 274 |
| 6 | 0.6333 | 0.7619 | 0.4311 | 0.6546 | 0.7485 |  | 0 | 1.002 [0.874,1.147] | 0.0433 | 283 |
| 7 | 0.6627 | 0.7388 | 0.5184 | 0.5761 | 0.7485 |  | 0 | 0.999 [0.862,1.177] | 0.0428 | 315 |
| 8 | 0.6404 | 0.7267 | 0.3711 | 0.6495 | 0.7485 |  | 0 | 1.000 [0.839,1.189] | 0.0424 | 318 |
| 9 | 0.6534 | 0.7495 | 0.4219 | 0.7150 | 0.7485 |  | 0 | 0.998 [0.811,1.178] | 0.0421 | 259 |
| 10 | 0.6349 | 0.7448 | 0.3995 | 0.7510 | 0.7510 | yes | 0 | 0.996 [0.805,1.165] | 0.0417 | 268 |
| 11 | 0.6825 | 0.7416 | 0.6009 | 0.6866 | 0.7510 |  | 0 | 0.999 [0.799,1.206] | 0.0414 | 257 |
| 12 | 0.6732 | 0.7716 | 0.5516 | 0.6622 | 0.7510 |  | 0 | 0.996 [0.790,1.191] | 0.0412 | 261 |
| 13 | 0.6845 | 0.7554 | 0.5653 | 0.5788 | 0.7510 |  | 0 | 0.996 [0.804,1.227] | 0.0409 | 264 |
| 14 | 0.6613 | 0.7569 | 0.5598 | 0.7205 | 0.7510 |  | 0 | 0.997 [0.796,1.224] | 0.0407 | 273 |

### Held-out evaluation (5 s frozen, seeds 20001-20003), candidate = best-gains (gen 10)

| Seed | who | standing frac | uprightness | slip mm/s | v7 reward | v8 reward | forward mm | height mm | raw touchdowns/leg | debounced steps/leg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20001 | candidate | 0.922 | 0.905 | 2.196 | 0.6469 | 0.3441 | +2.066 | 1.046 | [74, 39, 61, 38, 35, 56] | [4, 5, 7, 6, 9, 1] | 98 |
| 20001 | parent | 0.898 | 0.868 | 2.300 | 0.6077 | 0.3065 | +1.224 | 1.075 | [66, 29, 51, 43, 40, 54] | [5, 5, 5, 4, 8, 4] | 102 |
| 20002 | candidate | **0.581** | **0.213** | 1.778 | **0.1095** | -0.0820 | -2.292 | 0.897 | [40, 33, 58, 17, 39, 37] | [2, 2, 6, 0, 6, 2] | 114 |
| 20002 | parent | 0.907 | 0.879 | 2.693 | 0.6168 | 0.2824 | +0.292 | 1.256 | [56, 31, 55, 17, 62, 28] | [5, 7, 8, 2, 3, 4] | 106 |
| 20003 | candidate | 0.865 | 0.887 | 2.979 | 0.5528 | 0.2311 | -0.041 | 1.255 | [62, 38, 65, 11, 72, 37] | [6, 6, 3, 3, 2, 3] | 113 |
| 20003 | parent | 0.911 | 0.875 | 2.429 | 0.6264 | 0.2864 | +1.627 | 1.146 | [73, 41, 65, 12, 49, 50] | [1, 4, 2, 2, 2, 4] | 100 |

Means: candidate standing 0.789 vs parent 0.905; v7 0.436 vs 0.617.

### Gates

**Stand gate: FAILED**
- no_physics_errors: PASS
- standing_ge_0.97_every_seed: FAIL (0.922 / 0.581 / 0.865)
- uprightness_ge_0.85_every_seed: FAIL (20002: 0.213, the fly tips over)
- slip_not_worse_than_parent_mean: PASS (2.318 vs 2.474 mm/s)
- v7_reward_beats_parent_every_seed: FAIL (better only on 20001)

**Walk gate: FAILED**
- no_physics_errors: PASS
- forward_ge_2mm_every_seed: FAIL (+2.07 / −2.29 / −0.04)
- standing_ge_0.90_every_seed: FAIL
- stepping_4plus_legs_3plus_touchdowns: PASS
- real_steps_4plus_legs_2plus_debounced_steps: PASS
- all_6_legs_2plus_debounced_steps: FAIL
- forward_beats_parent_every_seed: FAIL

**Verdict:** no gate passed. Worse than the parent on held-out seeds; R1b remains the best stand result.

### Video
[`videos/R2b-cma-stand-fixedseeds-10s.mp4`](../videos/R2b-cma-stand-fixedseeds-10s.mp4): seed 20001, 10 s. Forward +3.79 mm, standing fraction 0.937.
This is the most forward motion seen so far, but on the one seed where this vector happens to work.

### Interpretation
CMA reached a slightly higher training score (0.751 vs R1b's 0.742) by moving the gains further (L2 0.90 vs 0.53), and it
overfit the two training seeds: on held-out seed 20002 the fly tips over (uprightness 0.21), and seed 20003 is also worse than the parent.
Training on 2 seeds is not enough to keep larger gain changes safe. Per the round rules the stand phase continues from
**R1b with ES** (R3). The `--seeds-per-gen 3` change is still queued as step (b), after (a) `--seconds 3.0`.

Files: [`runs/R2b-cma-stand-fixedseeds/`](R2b-cma-stand-fixedseeds/)

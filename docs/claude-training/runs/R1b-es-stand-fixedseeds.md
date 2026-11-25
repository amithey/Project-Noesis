# R1b — ES stand, fixed training seeds

- **Date/time:** 2026-11-25, 19:09–20:31 training, 20:31–20:38 evaluation (local)
- **Continues from:** parent `standing-001-train-11000/weights-last.npz` (no `--init-gains`); replaces aborted [R1](R1-es-stand.md)
- **Code:** training started at `b45166f` (fixed seeds); evaluation ran at `a18e71d` (adds debounced steps and the all-6-legs walk criterion; the stand fitness is unchanged)
- **Command** (from `gpu-pilot`):

```
python es_train.py --method es --stage stand --generations 15 --pairs 6 --seconds 1.5 --workers 3 --wall-hours 2.5 --output <project>\gpu-pilot\outputs\claude-es\R1b-es-stand-fixedseeds
python es_evaluate.py --gains <run>\best-gains.npy --output <run>\eval --label "R1b-es-stand-fixedseeds es stand" --workers 3
```

- **Config:** OpenAI-ES, 6 antithetic pairs (12 candidates + centre), 1.5 s episodes, **fixed training seeds 30000 and 30001 every generation**,
  sigma 0.05, lr 0.01 (Adam), fine grouping (136 groups, 85,390 edges), 26 episodes per generation, 3 workers.
- **Wall time:** 81 min of training (262–478 s per generation) plus 7.5 min of evaluation.
- **Best:** centre fitness 0.742 at gen 13 (parent on the same seeds: 0.706). Best gains: mean 1.003, range 0.904–1.115, L2 distance from 1 = 0.53.

### Training fitness

| Gen | pop mean | pop max | pop min | centre | best centre | improved | errors | gain mean [min,max] | sigma | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6595 | 0.7270 | 0.5633 | 0.7063 | 0.7063 | yes | 0 | 1.000 [0.990,1.010] | 0.0500 | 304 |
| 1 | 0.6472 | 0.7011 | 0.5174 | 0.6559 | 0.7063 |  | 0 | 1.000 [0.980,1.020] | 0.0500 | 276 |
| 2 | 0.6428 | 0.7067 | 0.4897 | 0.6573 | 0.7063 |  | 0 | 0.999 [0.970,1.030] | 0.0500 | 270 |
| 3 | 0.6612 | 0.7477 | 0.4565 | 0.6509 | 0.7063 |  | 0 | 1.000 [0.962,1.040] | 0.0500 | 281 |
| 4 | 0.6795 | 0.7420 | 0.6047 | 0.5958 | 0.7063 |  | 0 | 1.000 [0.954,1.049] | 0.0500 | 349 |
| 5 | 0.6266 | 0.7117 | 0.4273 | 0.5896 | 0.7063 |  | 0 | 1.000 [0.947,1.058] | 0.0500 | 409 |
| 6 | 0.6633 | 0.7428 | 0.5748 | 0.7097 | 0.7097 | yes | 0 | 1.000 [0.938,1.068] | 0.0500 | 304 |
| 7 | 0.6876 | 0.7595 | 0.5272 | 0.6667 | 0.7097 |  | 0 | 1.000 [0.932,1.076] | 0.0500 | 266 |
| 8 | 0.6693 | 0.7543 | 0.5175 | 0.7224 | 0.7224 | yes | 0 | 1.001 [0.925,1.085] | 0.0500 | 314 |
| 9 | 0.6430 | 0.7446 | 0.5207 | 0.7309 | 0.7309 | yes | 0 | 1.001 [0.917,1.093] | 0.0500 | 323 |
| 10 | 0.6872 | 0.7611 | 0.5350 | 0.6871 | 0.7309 |  | 0 | 1.001 [0.913,1.099] | 0.0500 | 277 |
| 11 | 0.6650 | 0.7616 | 0.5489 | 0.7113 | 0.7309 |  | 0 | 1.002 [0.910,1.107] | 0.0500 | 273 |
| 12 | 0.7009 | 0.7640 | 0.6489 | 0.7201 | 0.7309 |  | 0 | 1.003 [0.904,1.115] | 0.0500 | 262 |
| 13 | 0.6935 | 0.7560 | 0.5282 | 0.7422 | 0.7422 | yes | 0 | 1.003 [0.900,1.119] | 0.0500 | 458 |
| 14 | 0.6833 | 0.7437 | 0.6260 | 0.7142 | 0.7422 |  | 0 | 1.004 [0.897,1.122] | 0.0500 | 478 |

### Held-out evaluation (5 s frozen, seeds 20001-20003), candidate = best-gains (gen 13)

| Seed | who | standing frac | uprightness | slip mm/s | v7 reward | v8 reward | forward mm | height mm | raw touchdowns/leg | debounced steps/leg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20001 | candidate | **0.943** | 0.887 | 2.367 | **0.6757** | 0.3556 | +1.621 | 1.009 | [62, 39, 70, 30, 47, 64] | [8, 11, 3, 1, 4, 4] | 122 |
| 20001 | parent | 0.898 | 0.868 | 2.300 | 0.6077 | 0.3065 | +1.224 | 1.075 | [66, 29, 51, 43, 40, 54] | [5, 5, 5, 4, 8, 4] | 103 |
| 20002 | candidate | **0.980** | 0.905 | 2.826 | **0.7335** | 0.3793 | +0.556 | 1.210 | [81, 51, 78, 19, 63, 22] | [4, 4, 4, 5, 2, 2] | 133 |
| 20002 | parent | 0.907 | 0.879 | 2.693 | 0.6168 | 0.2824 | +0.292 | 1.256 | [56, 31, 55, 17, 62, 28] | [5, 7, 8, 2, 3, 4] | 110 |
| 20003 | candidate | 0.870 | 0.875 | 2.416 | 0.5632 | 0.2548 | -0.273 | 1.193 | [49, 45, 64, 30, 64, 33] | [10, 5, 3, 3, 4, 4] | 127 |
| 20003 | parent | 0.911 | 0.875 | 2.429 | 0.6264 | 0.2864 | +1.627 | 1.146 | [73, 41, 65, 12, 49, 50] | [1, 4, 2, 2, 2, 4] | 102 |

Means over the 3 seeds: candidate standing **0.931** vs parent 0.905; v7 **0.6575** vs 0.6170; slip 2.536 vs 2.474 mm/s; forward +0.63 vs +1.05 mm.

### Gates

**Stand gate: FAILED**
- no_physics_errors: PASS
- standing_ge_0.97_every_seed: FAIL (0.943 / 0.980 / 0.870)
- uprightness_ge_0.85_every_seed: PASS (0.887 / 0.905 / 0.875)
- slip_not_worse_than_parent_mean: FAIL (2.536 vs 2.474 mm/s)
- v7_reward_beats_parent_every_seed: FAIL (better on 20001 and 20002, worse on 20003)

**Walk gate: FAILED** (no walking objective in this run)
- no_physics_errors: PASS
- forward_ge_2mm_every_seed: FAIL
- standing_ge_0.90_every_seed: FAIL (20003: 0.870)
- stepping_4plus_legs_3plus_touchdowns: PASS
- real_steps_4plus_legs_2plus_debounced_steps: PASS
- all_6_legs_2plus_debounced_steps: FAIL (20001: leg 4 made 1 step)
- forward_beats_parent_every_seed: FAIL

**Verdict:** no gate passed. This is the first held-out improvement on average: standing +0.026 and v7 +0.041 over the parent.

### Video
[`videos/R1b-es-stand-fixedseeds-10s.mp4`](../videos/R1b-es-stand-fixedseeds-10s.mp4): seed 20001, 10 s. Standing fraction 0.956, forward +1.61 mm.

### Interpretation
Fixed training seeds made selection meaningful. The centre started below the parent (the fitness landscape is rugged,
and the first Adam steps hurt), then climbed to 0.742 against the parent's 0.706 on the training seeds. That gain partly
transfers to held-out seeds: large gains on 20001 and 20002 (standing 0.94 and 0.98), but a regression on 20003 (0.87 vs 0.91).
With only 2 training seeds the vector is somewhat overfit. The debounced step counts show that the parent already makes
a few real steps on most legs. Both parent and candidate fail "all six legs ≥2 steps" on at least one seed.

Files: [`runs/R1b-es-stand-fixedseeds/`](R1b-es-stand-fixedseeds/)

# R3 — ES stand, continued from R1b

- **Date/time:** training 2026-11-25 22:05 → 2026-11-26 09:01 local (10.9 h wall, of which ~10 h was the PC asleep overnight
  during gen 11); evaluation 2026-11-26 09:01–09:08
- **Continues from:** `--init-gains` = [R1b](R1b-es-stand-fixedseeds.md) best-gains (best held-out stand result so far)
- **Change vs previous round:** none (first continuation round; same settings as R1b)
- **Code:** `a18e71d` / `99bc903` (docs only since)
- **Command** (from `gpu-pilot`):

```
python es_train.py --method es --stage stand --generations 15 --seconds 1.5 --workers 3 --wall-hours 2.5 --output <out>\R3-es-stand-from-R1b --init-gains <out>\R1b-es-stand-fixedseeds\best-gains.npy
python es_evaluate.py --gains <run>\best-gains.npy --output <run>\eval --label "R3-es-stand-from-R1b es stand" --workers 3
```

- **Config:** OpenAI-ES, 6 antithetic pairs, sigma 0.05, lr 0.01, fixed seeds 30000 and 30001, 1.5 s, fine grouping (136 groups), 3 workers.
- **Status:** `stopped_wall_budget` after **12 of 15 generations**. The wall clock counted the overnight sleep, so the 2.5 h budget ran out.
- **Best:** centre 0.765 at gen 7 (R1b start: 0.742). Best gains: mean 1.005, range 0.848–1.123; L2 distance 0.61 from the parent and 0.38 from R1b.

### Training fitness

| Gen | pop mean | pop max | pop min | centre | best centre | improved | errors | gain mean [min,max] | sigma | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6748 | 0.7518 | 0.5437 | 0.7422 | 0.7422 | yes | 0 | 1.002 [0.894,1.125] | 0.0500 | 313 |
| 1 | 0.6591 | 0.7574 | 0.4603 | 0.6559 | 0.7422 |  | 0 | 1.002 [0.893,1.131] | 0.0500 | 285 |
| 2 | 0.6899 | 0.7637 | 0.5966 | 0.7291 | 0.7422 |  | 0 | 1.003 [0.883,1.126] | 0.0500 | 294 |
| 3 | 0.6457 | 0.7117 | 0.5354 | 0.6931 | 0.7422 |  | 0 | 1.004 [0.873,1.123] | 0.0500 | 293 |
| 4 | 0.6727 | 0.7563 | 0.5646 | 0.6635 | 0.7422 |  | 0 | 1.004 [0.865,1.123] | 0.0500 | 286 |
| 5 | 0.7024 | 0.7659 | 0.6429 | 0.6271 | 0.7422 |  | 0 | 1.004 [0.857,1.123] | 0.0500 | 286 |
| 6 | 0.6710 | 0.7644 | 0.5131 | 0.6878 | 0.7422 |  | 0 | 1.005 [0.848,1.123] | 0.0500 | 286 |
| 7 | 0.6920 | 0.7546 | 0.5807 | 0.7650 | 0.7650 | yes | 0 | 1.005 [0.840,1.129] | 0.0500 | 357 |
| 8 | 0.6724 | 0.7714 | 0.5470 | 0.7115 | 0.7650 |  | 0 | 1.005 [0.831,1.135] | 0.0500 | 365 |
| 9 | 0.6742 | 0.7562 | 0.5833 | 0.7619 | 0.7650 |  | 0 | 1.005 [0.824,1.138] | 0.0500 | 318 |
| 10 | 0.6644 | 0.7524 | 0.5716 | 0.7045 | 0.7650 |  | 0 | 1.006 [0.819,1.143] | 0.0500 | 293 |
| 11 | 0.6600 | 0.7598 | 0.4909 | 0.7437 | 0.7650 |  | 0 | 1.006 [0.816,1.145] | 0.0500 | 35965 (PC asleep) |

### Held-out evaluation (5 s frozen, seeds 20001-20003), candidate = best-gains (gen 7)

| Seed | who | standing frac | uprightness | slip mm/s | v7 reward | v8 reward | forward mm | height mm | raw touchdowns/leg | debounced steps/leg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20001 | candidate | 0.907 | 0.897 | 3.084 | 0.6159 | 0.2746 | +0.271 | 1.225 | [61, 34, 57, 26, 84, 30] | [3, 9, 5, 2, 2, 3] | 107 |
| 20001 | parent | 0.898 | 0.868 | 2.300 | 0.6077 | 0.3065 | +1.224 | 1.075 | [66, 29, 51, 43, 40, 54] | [5, 5, 5, 4, 8, 4] | 87 |
| 20002 | candidate | 0.917 | 0.904 | 2.747 | 0.6361 | 0.2952 | +0.328 | 1.183 | [68, 40, 66, 19, 68, 32] | [2, 4, 5, 3, 4, 2] | 103 |
| 20002 | parent | 0.907 | 0.879 | 2.693 | 0.6168 | 0.2824 | +0.292 | 1.256 | [56, 31, 55, 17, 62, 28] | [5, 7, 8, 2, 3, 4] | 94 |
| 20003 | candidate | 0.844 | 0.897 | 2.828 | 0.5203 | 0.1902 | -0.552 | 1.264 | [63, 46, 67, 13, 64, 35] | [6, 7, 6, 3, 2, 5] | 105 |
| 20003 | parent | 0.911 | 0.875 | 2.429 | 0.6264 | 0.2864 | +1.627 | 1.146 | [73, 41, 65, 12, 49, 50] | [1, 4, 2, 2, 2, 4] | 87 |

Means: candidate standing 0.889 (R1b 0.931, parent 0.905); v7 0.591 (R1b 0.658, parent 0.617); slip 2.886 vs 2.474 mm/s.

### Gates

**Stand gate: FAILED**
- no_physics_errors: PASS
- standing_ge_0.97_every_seed: FAIL (0.907 / 0.917 / 0.844)
- uprightness_ge_0.85_every_seed: PASS (0.897 / 0.904 / 0.897)
- slip_not_worse_than_parent_mean: FAIL (2.886 vs 2.474)
- v7_reward_beats_parent_every_seed: FAIL (better on 20001 and 20002, worse on 20003)

**Walk gate: FAILED**
- no_physics_errors: PASS
- forward_ge_2mm_every_seed: FAIL
- standing_ge_0.90_every_seed: FAIL (20003: 0.844)
- stepping_4plus_legs_3plus_touchdowns: PASS
- real_steps_4plus_legs_2plus_debounced_steps: PASS
- all_6_legs_2plus_debounced_steps: **PASS** (the first candidate to pass it: every leg made ≥2 real steps on every seed)
- forward_beats_parent_every_seed: FAIL

**Verdict:** no gate passed; **no held-out improvement over R1b** (counts as stand round 1 without improvement). R1b remains the best stand result.

### Video
[`videos/R3-es-stand-from-R1b-10s.mp4`](../videos/R3-es-stand-from-R1b-10s.mp4): seed 20001, 10 s. Forward −0.36 mm, standing fraction 0.891.

### Interpretation
Continuing from R1b raised the training score (0.742 → 0.765 on seeds 30000–30001), but the held-out result got worse.
This matches R2b: optimising on two seeds overfits them. Per the updated rule order, the next round (R4) restarts from R1b with
`--seeds-per-gen 4` to target overfitting.

Files: [`runs/R3-es-stand-from-R1b/`](R3-es-stand-from-R1b/)

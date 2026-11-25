# R0 — parent baseline (no training by this track)

- **Date/time:** 2026-11-25, 18:51–18:59 local (Windows PC, CPU only)
- **Machine:** Intel i7-1165G7, 4 physical / 8 logical cores, 15.7 GB RAM (about 2.5 GB free at idle)
- **Python:** 3.12.14 (Codex runtime) + `gpu-pilot/runtime` (FlyGym 2.1.0, MuJoCo 3.9.0, numba 0.68.0, imageio 2.38.1 / ffmpeg 7.1, Pillow 12.3.0). Nothing installed.
- **Command** (from `gpu-pilot` of this branch):

```
python es_evaluate.py --output <project>\gpu-pilot\outputs\claude-es\R0-parent-eval --label "R0 parent (untrained by this track)" --workers 3
```

- **Config:** no `--gains` (all 136 group gains = 1.0 = parent checkpoint
  `standing-001-train-11000/weights-last.npz`, sha256 `fb80cfcc…26a8`), held-out seeds 20001–20003, 5 s frozen each,
  objective `curriculum-gripped-step-v8` for the episode (the v7 and v8 rewards are both logged), 10 s video on seed 20001.
  `--workers 3` (not the default 6) because of RAM: see the smoke test notes.
- **Wall time:** 429 s for 6 episodes + the 10 s video (7 min 25 s total including startup).
  Per-seed wall times for a 5 s episode with 3 running in parallel: 117–146 s (table below).
- **Note:** the first attempt (18:42–18:51) finished all episodes and the video but crashed while writing
  `evaluation.json` (numpy bool not JSON-serializable). Fixed in commit `7a71f53`, then re-run. Results are deterministic, so nothing changed apart from the file being written.

### Held-out evaluation (5 s frozen, seeds 20001-20003)

"candidate" here is also the parent (gains = 1), so both rows match exactly; this confirms the evaluation is deterministic.

| Seed | who | standing frac | uprightness | slip mm/s | v7 reward | v8 reward | forward mm | height mm | touchdowns/leg | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 20001 | candidate | 0.898 | 0.868 | 2.300 | 0.6077 | 0.3065 | +1.224 | 1.075 | [66, 29, 51, 43, 40, 54] | 130 |
| 20001 | parent | 0.898 | 0.868 | 2.300 | 0.6077 | 0.3065 | +1.224 | 1.075 | [66, 29, 51, 43, 40, 54] | 117 |
| 20002 | candidate | 0.907 | 0.879 | 2.693 | 0.6168 | 0.2824 | +0.292 | 1.256 | [56, 31, 55, 17, 62, 28] | 146 |
| 20002 | parent | 0.907 | 0.879 | 2.693 | 0.6168 | 0.2824 | +0.292 | 1.256 | [56, 31, 55, 17, 62, 28] | 133 |
| 20003 | candidate | 0.911 | 0.875 | 2.429 | 0.6264 | 0.2864 | +1.627 | 1.146 | [73, 41, 65, 12, 49, 50] | 131 |
| 20003 | parent | 0.911 | 0.875 | 2.429 | 0.6264 | 0.2864 | +1.627 | 1.146 | [73, 41, 65, 12, 49, 50] | 116 |

### Gates (expected to fail: a model cannot beat itself)

**Stand gate: FAILED**
- no_physics_errors: PASS
- standing_ge_0.97_every_seed: FAIL (0.898 / 0.907 / 0.911)
- uprightness_ge_0.85_every_seed: PASS (0.868 / 0.879 / 0.875)
- slip_not_worse_than_parent_mean: PASS (equal)
- v7_reward_beats_parent_every_seed: FAIL (equal, by construction)

**Walk gate: FAILED**
- no_physics_errors: PASS
- forward_ge_2mm_every_seed: FAIL (+1.22 / +0.29 / +1.63 mm)
- standing_ge_0.90_every_seed: FAIL (seed 20001: 0.898)
- stepping_4plus_legs_3plus_touchdowns: PASS
- forward_beats_parent_every_seed: FAIL (equal, by construction)

**Verdict:** baseline recorded; no gate passed (as expected).

### Video
[`videos/R0-parent-eval-10s.mp4`](../videos/R0-parent-eval-10s.mp4): 301 frames, 10 s, seed 20001. Over 10 s: forward +1.49 mm, standing fraction 0.896.

### Interpretation
The parent stands about 90% of the time with uprightness about 0.87 and drifts forward only 0.3–1.6 mm in 5 s.
The touchdown counts are high (12–73 per leg in 5 s), but that reflects claw contact flicker rather than
stepping: there is no consistent forward displacement. The standing gate's two absolute requirements show the
gap to close: standing ≥ 0.97 (now about 0.90) and uprightness ≥ 0.85 (already met, with a small margin).

Files: [`runs/R0-parent-eval/evaluation.json`](R0-parent-eval/evaluation.json)

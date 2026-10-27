# Claude workstream, round 1: reward profiles and weight integrity

Branch `claude/flygym2-learning-rewards`, base `0f003b6`. Issue #19. Core pin (read-only):
`codex/brain-body-feedback` @ `41b5fe6534ed4c8f0c21a2e123c8d518b3bf5953`.

## Profile mapping (for the Codex runner)

`noesis_gpu.reward_profiles.PROFILES` mirrors `noesis.curriculum.STAGES`:

| profile | core objective |
| --- | --- |
| `standing` (default) | `curriculum-standing-v7` |
| `gripped-steps` | `curriculum-gripped-step-v8` |
| `continuous-walking` | `standing-gate-smooth-grip-v6` (the "v6" of the task) |
| `obstacles` | `standing-gate-smooth-grip-v6` |

`score_transition(before, after, dt_ms, objective=...)` is unchanged (same default, same
result fields); it now also accepts a profile name, resolved via `resolve_objective`.
Exact objective ids pass through. No reward formula, threshold or learning rule was changed.

## Tests (synthetic; run from `gpu-pilot` with `NOESIS_SOURCE=<core>/src`)

- `tests/test_reward_profiles.py`: names, default, ids exist in core `GROUNDED_OBJECTIVES`,
  result fields; slipping, fallen and airborne transitions get no/low `grounded_progress`
  and lower reward than a gripped step (v8 and v6).
- `tests/test_weight_integrity.py`: three load -> learn -> save cycles keep sparse topology,
  weight signs, original-reference bounds (0.5x..1.5x), and sibling brains stay independent.
- Existing `test_learning.py` already covers frozen == exact and exploration off when frozen.

Result: 35 passed, 3 skipped (pilot suite, Python 3.11 venv, pinned core). Synthetic only;
no training run, no real checkpoint loaded. Real-checkpoint load check is still open.

## Next
Gemini review of the slip/fall scenarios; Codex to select profile by stage name in the runner;
load the real parent checkpoint (`078cfe3c...`) once the shared data/release is available.

## Real-checkpoint integrity check (2026-10-28, read-only, no training)

`check_checkpoint.py <core/src> <gpu-pilot/src> <data> <weights.npz> <cache_dir>` loads the real
connectome and checkpoint through `LearningBrain` and compares to the original reference.
Checkpoint `standing-001-train-11000/weights-last.npz`, SHA-256
`fb80cfcc3018329209784751570d7ffb1ed33aa9ef48fceaa85588862f7626a8` (matches the recorded hash):

- 562 learning targets, 85,390 plastic edges; 56,305 edges differ from the original.
- Signs all preserved, no zero weights.
- Weight ratio to original min 0.776, max 1.253; 0 edges outside the 0.5x-1.5x bound.
- Loaded with learning inactive, 0 updates. Run used the pinned core `41b5fe6`, Python 3.11.

This verifies load integrity only; it is not validation of standing, stepping or walking.

## Training status

No training was started in this round. The gripped-step stage (v8) is the next stage in
`gpu-pilot/training-plan.md`, but the plan still lists the four-stage curriculum/restart
manager as not ported (Codex, #18), and run budget on the shared machine and the main
weights are Codex's. Gemini's evaluation (PR #22) shows the two standing probes are
stepping-in-place/uncoordinated, not walking.

## Gripped-step (v8) training birth (2026-10-28, exploratory)

- Finding: the existing `outputs/gripped-001-train-11001` was trained with
  `curriculum-standing-v7`, not v8. Recomputing its stored rewards from the stored observations
  matches v7 exactly (max diff 0.0) and v8 differs (0.80). `train.py` hard-codes v7. It is a second
  standing birth, so its failed gate (gripped-001-evaluation.json) is not evidence about v8.
- Run: `gpu-pilot/run_gripped_birth.py` (wrapper; `train.py` untouched) with `--profile gripped-steps`,
  seed 1000, 5 s, CPU, one world, parent `standing-001-train-11000/weights-last.npz`
  (SHA `fb80cfcc...26a8`, unchanged). Output `outputs/gripped-v8-002-train-1000` (not in Git).
- Verified: stored rewards match v8 exactly (max diff 0.0); 990 updates; exploration max 1.499 mV/ms.
- Result checkpoint SHA-256 `d7afb1bca9990e17cc2c3dd5bb9d73e2fa8692052d3eddb71e66687cea917daa`.
  Integrity: signs preserved, ratio 0.780-1.247, 0 edges outside 0.5x-1.5x, 56,347 edges changed.
- Training-time standing fraction 0.818 and slip 2.80 mm/s are under exploration and are NOT a
  frozen result. No frozen 10006/10007 evaluation yet; no stage advance, promotion or walking claim.

## Frozen evaluation of the v8 candidate (2026-10-28)

Candidate `d7afb1bc...daa` (seed-1000 v8 birth) vs its parent bank `fb80cfcc...26a8`
(`standing-001-train-11000`), frozen: no learning, no exploration, 5 s, CPU, one world, seeds 10006
and 10007, same `noesis_gpu.run` and pinned core. Metrics from Gemini's `gait-metrics-v1.0`
(branch `gemini/flygym2-gait-validation`, read-only copy; not part of this PR).
Raw output: `outputs/gripped-v8-002-evaluation.json` and `gripped-v8-002-candidate-1000{6,7}` (not in Git).

| seed | bank | standing | slip mm/s | forward mm | steps | regime | tripod |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10006 | parent | 0.9717 | 2.35 | 0.308 | 81 | in place | 0.052 |
| 10006 | v8 candidate | 0.9697 | 2.78 | -0.030 | 83 | in place | 0.157 |
| 10007 | parent | 0.9505 | 2.24 | 1.345 | 88 | uncoordinated | 0.041 |
| 10007 | v8 candidate | 0.9242 | 2.91 | 0.627 | 71 | in place | 0.207 |

Gripped-step criteria (>=0.25 mm forward, support/lift cycles, standing >=90%, slip <=10 mm/s):
10006 fails forward progress (-0.03 mm); 10007 meets forward progress but forward distance fell vs
parent; neither seed shows coordinated stepping. Slip rose 18.5% and 29.6% vs the parent. Result:
**no improvement from one v8 birth; stage not advanced; not promoted; walking not proven.**
Caveats: one birth, one training seed; tripod scores rose but remain far from coordinated; the
"standing gate" field in the output is the old standing-stage gate and says nothing about v8.

## Second v8 birth (seed 1001) and its frozen evaluation (2026-10-28)

Same parent (`fb80cfcc...26a8`), same wrapper/profile, CPU, one world, 5 s, 990 updates.
Training-seed 1001 checkpoint SHA-256 `626bde8dce2f996ef181650ba4c3812abdc0dfdd3a969bbfc1e0e1ed57b95ff1`
(`outputs/gripped-v8-003-train-1001`, not in Git). Stored rewards match v8 exactly (diff 0.0).
Integrity: signs preserved, ratio 0.772-1.257, 0 edges outside 0.5x-1.5x, 56,430 edges changed.
Frozen, same conditions, Gemini `gait-metrics-v1.0` (raw: `outputs/gripped-v8-003-evaluation.json`):

| seed | bank | standing | slip mm/s | forward mm | steps | regime | tripod |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10006 | parent | 0.9717 | 2.35 | 0.308 | 81 | in place | 0.052 |
| 10006 | v8 seed 1001 | 0.9687 | 1.92 | 0.989 | 84 | in place | 0.055 |
| 10007 | parent | 0.9505 | 2.24 | 1.345 | 88 | uncoordinated | 0.041 |
| 10007 | v8 seed 1001 | 0.9030 | 2.38 | 1.522 | 108 | uncoordinated | 0.067 |

Reading: the two births disagree (seed 1000 worsened both probes; seed 1001 improved forward
distance in both, slip -18% on 10006, +6% on 10007, standing 0.903 on 10007 only just above 0.90).
The numeric gripped-step thresholds (>=0.25 mm, standing >=90%, slip <=10 mm/s) are met by seed 1001
on both probes, but gait is still not coordinated (tripod 0.06-0.07, regimes in place/uncoordinated)
and the leg support/lift-cycle criterion has not been separately reviewed. Birth-to-birth variance is
large, so a single passing birth is not robust evidence. Not advanced, not promoted, walking not proven.
Stage advancement belongs to Codex after Gemini's independent review; round count for the stage: 2 of 3.

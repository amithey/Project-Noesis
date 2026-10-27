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

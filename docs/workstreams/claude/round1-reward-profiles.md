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

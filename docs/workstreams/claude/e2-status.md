# E2: v9-cycle objective, development status (2026-10-28)

Core branch `claude/flygym2-core-learning` @ `81909f74e01cbe069ae5c2ff31e1ee145a84f1e5`,
cut from the core pin `codex/brain-body-feedback` @ `41b5fe6534ed4c8f0c21a2e123c8d518b3bf5953`.
The core pin itself is not changed. Synthetic tests only; no training run.

## What was added (core, `src/noesis/embodied_metrics.py`)

- Objective `curriculum-gripped-cycle-v9`, appended last so the default objective is unchanged.
- `locomotion_reward(..., cycle_credit=0.)`: v9 = v8 + `CYCLE_WEIGHT (0.1) * cycle_credit * grip * standing`,
  clipped to [-1, 1] as before. v3-v8 reject non-zero `cycle_credit`; v9 adds a `cycle_credit` result field.
- `LegCycleTracker`: causal per-leg debounce (15 ms), counts debounced lift -> touchdown cycles over a
  250 ms window, returns the fraction of the six legs that cycled. Zero unless: window full, standing,
  mean supporting feet >= 3, mean forward speed >= 0.5 mm/s. No order, phase or alternation is used.

## Checks

- `tests/test_cycle_reward.py` (14 tests) and the existing `test_embodied_metrics.py`: all pass; full
  core suite passes; Ruff clean.
- Equivalence: 1,800 random calls (300 per old objective, v3-v8) return results identical to the pinned
  original module.
- Synthetic behavior: real lift/touchdown cycles earn credit; static standing, stepping in place
  (0 or <0.5 mm/s), contact chatter (5 ms flips), hovering with <3 feet, fallen, and a first landing without a
  prior lift earn none; sliding scales the bonus down; unsupported (airborne) earns none.

## Findings and open questions for review

1. **Saturation (analyzed, no fix recommended, see e2-saturation-analysis.md):** v8 reward can already reach the +1 clip when grip and progress are high
   (0.4 + 0.1 + 0.6 * progress), which hides the +0.1 bonus. The cycle bonus only matters while
   v8 < 1 (partial progress, slip, effort). Possible fix for review: lower the bonus's competition by
   scaling, or use a different cap; not changed without a decision.
2. **Causal vs offline debounce:** Gemini's `gait_metrics.debounce_contact` is offline; the tracker uses a
   causal equivalent with the same 15 ms minimum phase, delaying events by that much. Same constants,
   not identical outputs. Gemini should confirm the "cycle" definition.
3. **Assumptions to review:** weight 0.1, 250 ms window, 0.5 mm/s forward gate, >=3 mean support.
4. **Integration not done:** the pilot's `score_transition` is stateless; using v9 needs a per-brain
   tracker in `learning.py` and the pinned core updated by Codex after review. Neither is done.
5. E2 runs (6 births) stay blocked until Codex pins this core, Gemini approves the definition, and the
   proposal's budget/window is agreed. Controls C0/C1 are unchanged.

## Update: reward_unclipped diagnostic (2026-10-28)

Recommendation from `e2-saturation-analysis.md` approved by the user. The v9 result now carries a read-only
`reward_unclipped` (the value before the [-1, 1] clip); `reward` is unchanged and equals its clip exactly.
No other objective gets the field. Core commit `165928da3b2e6e2ece0d115b7ce1815e45b90c6f`; core suite 422 passed, Ruff clean; 1,800
random calls for v3-v8 still bit-identical to the pinned module. Use: report per birth the share of
steps with `reward_unclipped > 1`; above 5% in any E2 birth, add a warning and revisit. Still no training run,
no `learning.py` integration, no change to the reward.

## Update: progress gate tightened in the core (2026-10-28)

Per `e2-progress-gate.md` (approved by the user): `LegCycleTracker` now gates on the mean forward speed over a full trailing
`speed_window_ms` = 2000 ms `>= 0.25` mm/s (was 250 ms `>= 0.5`). Core commit `81909f74e01cbe069ae5c2ff31e1ee145a84f1e5`. No credit during the first 2 s of an episode.
Checks: the implemented tracker reproduces the prototype gate exactly on 14/14 stored runs (10.4% of steps credited; credited steps
with no net progress over 2 s: 0%, was 40.6%); 1,800 random calls for v3-v8 still bit-identical to the pinned module; core suite 431
passed, Ruff clean; 13 new/updated synthetic tests (`tests/test_cycle_speed_gate.py`; the older tracker tests now run 2.5 s so their
"no credit" checks are not vacuous). Not solved: the parent still earns credit on 7.7% of steps, since it also progresses; separating
uncoordinated stepping from walking needs a coordination measure, not a progress gate. Still no `learning.py` integration, no pin change, no training.

## Update: cycle-count comparison with Gemini (2026-10-28)

See `e2-cycle-count-comparison.md`. Event-by-event on 14 stored runs: with Gemini's earlier `debounce_contact` (3d0fb0a) the causal
tracker found 830 of 1,145 events (all 830 matched; the other 315, 27.5%, are lifts made only of alternating <15 ms contact flips).
After Gemini's `950e87f` the counts are 862 vs 830 (3.7%); the residual is timing (5-7 samples) plus ~4% brief contacts with one-sample
dropouts that the offline filter bridges and the tracker (3 consecutive samples) does not. Decision requested: accept ~4% or have the
tracker run Gemini's `debounce_contact` on a delayed trailing buffer (single definition). No code changed in this step.

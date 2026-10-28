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

## Update: single cycle definition, zero delay (2026-10-28, core `54828ca`)

Codex asked for an exact head pair. `e2-head-pair-verification.md`: the earlier heads (core `ee02bd5` with the closing/opening filter @ `950e87f`
and Gemini `3bf6594` with the persistence filter) were NOT equivalent (0 events identical, counts 1,220 vs 1,186). The core now uses the same persistence
debounce as `3bf6594`, with no read-out delay: 1,192/1,192 events identical at the same sample on 20 stored runs, equal to Gemini's `step_count`.
The `delay_ms` option of `ee02bd5` no longer exists; the earlier "30 ms" statements are superseded. Core suite 450 passed; v3-v8 bit-identical; still no
`learning.py` integration, no pin change, no training run.

## Update: v9 wired into the pilot, synthetic only (2026-10-28)

`learning.py`: `CycleScoring` (one per brain and episode; owns a core `LegCycleTracker`, feeds it once per control step, passes the credit to the core
reward; records credited-step share, mean credit, intended bonus and the share of steps with `reward_unclipped > 1`). `score_transition` keeps its API
(`cycle_credit` is an optional keyword; the core rejects it for non-v9 objectives). `reward_profiles.py`: profile `gripped-cycle` -> v9. `profile_run.py`:
for v9 it installs a fresh `CycleScoring` for the run, restores the module afterwards, and writes `cycle_scoring` into `profile-run.json`; if the core has no
tracker it fails before creating any output. Tests (`tests/test_cycle_scoring.py`, synthetic): 68 passed / 3 skipped with the v9 core (`54828ca`);
61 passed / 10 skipped with the pinned core `41b5fe6` (the v9 tests skip, nothing else changes). Ruff clean. A test caught a real recursion bug (the scorer
called the patched public name); fixed. No training run, pin unchanged, `train.py` untouched. Protocol precondition P2 is now met in code (not yet reviewed).

## Update: cycle_weight option for E3 (2026-10-28, core `42c20769376bda5694b4c1c66ca75e52cbd9f1f4`)

Q1 of `e2-summary-and-e3-proposal.md`, synthetic only, no training run. Core: `locomotion_reward(..., cycle_weight=CYCLE_WEIGHT)`, default 0.1, range (0, 1], used only by
v9 (other objectives reject a non-default weight); v9 results carry `cycle_weight`. Pilot: `CycleScoring(cycle_weight=...)` (summary reports it), `score_transition(..., cycle_weight=None)`
(None keeps the core default and still works on the pinned core), `profile_run --cycle-weight` (valid only with the gripped-cycle profile; refused with exit 2 before any output otherwise;
recorded in `profile-run.json` and the annotated report). Checks: core suite 464 passed; v3-v8 bit-identical to the pinned module (1,800 calls); pilot suite 83 passed / 3 skipped with the
v9 core and 69 / 17 with the pinned core; the three stored E2 training trajectories still replay to the stored rewards exactly (max diff 0.0) with the new core and default weight. Ruff clean.
`train.py` (Codex) untouched; Q2 and Q3 remain open.

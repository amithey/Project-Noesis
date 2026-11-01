# Sensory source and hook proposal (thermo first) — code, patch and tiny tests only

2026-11-01 · Claude (#19/#20, senses transferred by the user) · branch `claude/senses-hook` (from `claude/senses` 0597501). **No simulation of the fly was run; no Codex or Gemini file was edited.**

## What is proposed
1. **A 5-line patch to a Codex file** (`controller-extra-drive-parts.patch`, NOT applied on any branch): `FrozenBrain._extra_drive_parts(observation)` returns `[]` by default and its parts
   are appended to the existing `merge_drive` call. Default behaviour is unchanged; `merge_drive` already raises on overlap with the wired leg/balance afferents.
2. **No backend change.** The stimulus is a pure function of the EXISTING observation: `GradientField` maps `observation["position"]` along one axis to a signed deviation u in [-1, 1]
   (warm = max(u,0), cool = max(-u,0)). It is an experimenter-defined stimulus protocol, not physics; the unit of `position` and the gain (Hz per signal unit) are explicit modelling assumptions of the caller.
3. **`sensed_brain.py` (mine):** `sensed_class(FrozenBrain)` selects the 12 thermo afferents (6 warm, 6 cool on the real connectome) via `senses_bridge`, disjoint from the 4,974 wired afferents, and
   drives them through `ExternalSensoryAdapter`. It refuses a controller without the hook.

## One factor, matched controls (same code path, same neurons)
`off` adapter disabled · `zero` channels enabled, signal 0 · `constant` signal fixed at a level independent of position · `active` the field. Tiny tests (patch applied to a temporary copy, fresh interpreter):
off and zero are bit-identical to the plain brain in commands and spikes; a signal adds afferent spiking; constant and active are matched in level when the field takes that value; thermo afferents are disjoint
from the wired ones; weights never change; the factory raises without the hook. These do NOT test downstream muscle effects (the tiny network has none) or any behaviour.

## Not combined (as requested)
No foot_load change, no rectification, no per-leg reward, no reward change, no learning. Thermo is a single, new, otherwise-silent input.

## If Codex accepts the patch: proposed physical test (not started, not allocated)
Fixed weights, learning off, parent 9fd7d411, dev seeds 10006/10007/10010/10011, 5 s: arms off / zero / constant / active, gate 0 as in the chain diagnostic (off == zero == plain bitwise), then the same
descriptive chain statistic between the thermo signal and thermo-afferent firing, and for the unrestricted aggregate: change in total and motor-pool spiking, command and standing/slip against `off`. The registered rule and budget
(about 4 x 4 arms x 90 s ~ 1,700 s) would be written and committed BEFORE any run and sent to Codex for the lease. Success would be limited to "the new input reaches neural activity without breaking standing"; it is
not a behavioural benefit, not biological calibration and not walking. Changing senses changes the frozen-comparison identity (evaluate.py), so any later comparison needs its own matched parent run.

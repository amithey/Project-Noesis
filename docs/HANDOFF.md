# Project Noesis — handoff (2026-10-14)

State of the work as left by Claude, for whoever continues. Detailed numbers for every claim are in
`docs/findings.md` (dated sections). Read this first, then the findings.

## 1. Goal
A fruit fly whose body is driven only by its real connectome: skeleton, joints, muscles and senses connected
to the brain, nothing scripted, and eventually learning on its own from the world (no external reward).

## 2. Environment and rules (important)
- Main checkout: `C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis` (OneDrive, Hebrew path).
  **Do not edit there**; several agents share it. Use your own git worktree under `C:\dev\...`.
  Claude's worktrees: `C:\dev\noesis-claude2` (current), `C:\dev\noesis-claude` (taken over by Codex earlier).
- Python: `C:\dev\venvs\noesis\Scripts\python.exe`, always `PYTHONPATH=src` (`pip install -e` fails on the
  non-ASCII path). Windows PowerShell 5.1: no `&&`; scripts with Hebrew text need a UTF-8 BOM.
- Data (gitignored, never commit, CC-BY-NC): `<main>\data\raw` (BANC v888: `neurons.csv`,
  `connections_princeton.csv`), `<main>\data\fafb` (FAFB v783 connections + cell types), `<main>\data\fafb_buhmann`,
  `<main>\MANC` (male nerve cord). Pass `--data "<main>\data"` from a worktree. Caches in `data\processed`.
- Only `git add` your own files. Physics is ~50x slower than real time: long runs in the background.

## 3. Branches and PRs (nothing merged yet; stacked, merge in order)
`dev` <- #1 m1-connectome-sim <- #2 lif-adaptation <- #3 balance-senses <- #4 leg-sense-audit <- #5 body-physics
<- #6 apl-sign-and-reward-path <- #7 graded-apl <- #9 live-brain-map <- #10 mbon-compartments <- #11 brain-rescale
<- #13 vnc-rescale (latest Claude code). Side branches: #8 mb-coverage (on #7; conflicts with the chain only in
`docs/findings.md`), #12 codex/olfactory-specificity (on #11). Not pushed: `feature/gait-rhythm` (empty worktree
`C:\dev\noesis-codex-gait` prepared for the gait task, see section 7).
Codex-owned, uncommitted in the main checkout: `world.py`, `habitat.py`, `environment_senses.py`,
`learning_lab.py`, `plasticity.py`, `embodied_metrics.py`, `Start-Fly.ps1`.

## 4. What exists (code map)
| module | role |
|---|---|
| `connectome.py` | BANC -> signed sparse matrix; `NT_OVERRIDES` (APL = GABA); `corrected_scale(data)` = brain x FAFB/BANC + nerve cord x MANC/BANC per region; `load_cached` (cache key = factor hash, signs applied on load); `load_all_pairs` (1-4 synapse pairs) |
| `lif.py` | event-driven LIF (Shiu et al. parameters), adaptation `da`, Poisson drive, background noise, continuous current |
| `motor.py`, `body.py` | motor-neuron pools per muscle -> equilibrium-point muscles; standing pose and joint limits from real stepping kinematics; `CollisionGuard` |
| `sensory.py`, `balance.py` | leg afferents; Johnston's organ + halteres |
| `mushroom_body.py` | KC/MBON/DAN populations, weak MB edges, dopamine-gated KC->MBON plasticity, `GradedAPL`, MBON dendrite/axon compartments |
| `viz/brain_map.py` | live map of every neuron (schematic layout; the data have no positions) |
| experiments | `newborn_muscles` (fly + brain, `--scaled`, `--plasticity`, `--brain-map`), `body_check`, `feeding`, `response_battery`, `rescale_check`, `live_brain`, `mb_conditioning`, `learning_prerequisites`, `compare_mb_coverage` (on #8) |
| launchers | `Run-Fly.ps1` (fly + map, corrected brain), `Watch-Brain.ps1` (brain only, sugar on/off) |

## 5. What works (validated)
- BANC under-counts synapses (20-60% of FAFB per brain region, ~1/4 of MANC in the nerve cord, KC synapses 13-20x).
  Correcting counts per region restores Shiu et al.'s sugar->MN9 at 100 Hz (0 -> 14 Hz) with no tuning.
- Response battery (8 predictions fixed in advance), corrected brain: sugar and water drive MN9, bitter does not and
  suppresses sugar, odour drives its own glomeruli, sugar reaches reward dopamine neurons.
- Body: stands on its own in a real posture; muscles driven by motor-neuron pools; joint limits hold (one known
  case 0.10 rad over); floor contact within 50 um; fly stays upright with the brain running.
- Live views: fly (MuJoCo) + brain map.

## 6. What does not work yet (priority order)
1. **Locked brain (P8 fails).** The awake brain is bistable: any drive ignites one self-sustained state (~8 Hz mean,
   10-24k neurons) that never returns to rest. In it descending commands are contradictory (DNa02 313 Hz, MDN 62 Hz).
   Silencing monoamines halves it but removes sugar->PAM. **This blocks walking and clean learning trials.**
2. **No walking rhythm.** Leg motor neurons fire tonically; flexors and extensors co-contract (net drive ~0.3).
3. **Monoamines are modelled as fast excitatory synapses**; they should be slow and modulatory.
4. **Learning from the world** not shown: plasticity is dopamine-gated and validated on controls, but no
   odour-specific effect yet (dense Kenyon-cell code, PPL1/PAM not stimulus-specific). Re-test after 1 and 3.
5. Legs can still overlap the head/eye in extreme poses (guard reduces 464 -> 271 um, not eliminated).
6. Earlier calibrations were made on the weak raw brain (background drive, adaptation, APL gain 1024).
7. Habitat (vision/odour world) integration depends on Codex's uncommitted files.

## 7. Next steps (each with an acceptance test)
1. **Short-term synaptic depression** in `lif.py` (per presynaptic neuron resource x, use U, recovery tau ~0.3-1 s).
   Accept: `response_battery` P8 passes (recovery active < 2x rest + 50) while P1-P4, P7 still pass.
2. **Slow modulatory monoamines**: DA/5-HT/OA/TA change targets' excitability or synaptic gain over ~100s of ms
   instead of fast spikes. Accept: P5 and P6 pass with stimulus-specific dopamine (PAM sugar > odour).
3. **Recalibrate on the fully corrected brain**: background drive giving a graded (not bistable) awake state;
   APL gain for 5-10% Kenyon-cell sparseness.
4. **Gait rhythm** (task file `C:\dev\codex-tasks\gait-rhythm.md`, worktree `C:\dev\noesis-codex-gait`): measure
   rhythm, flexor/extensor alternation, tripod phase; then look for descending-neuron drive that produces it.
5. **Natural-reward conditioning**: odour paired with real sugar taste on the corrected brain, with no-reward control.
6. Merge the PR stack into `dev`; resolve `docs/findings.md` conflicts (#8 and #12 vs the chain).

## 8. Assumptions to keep in mind
Uniform LIF neurons (Shiu et al.); synapse count x 0.275 mV; NT sign from prediction (+ overrides); region factors
clipped 1-20, female BANC corrected with a male nerve cord (MANC); muscle -> joint table and sensory transfer
functions are anatomical judgement; the brain-map layout is schematic.

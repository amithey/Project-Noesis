# Plan: body A/B of the inhibition scale `g_I` on the locomotion pipeline (not yet run)

## Feasibility (checked 2026-10-18)
- The body stack runs in the shared environment (`C:/dev/venvs/noesis`: mujoco 3.2.7, flygym, dm_control, numba).
- A smoke run of the locomotion pipeline (`noesis.experiments.learning_lab`, frozen, quiet arena, seed 10, 0.6 s) completed:
  31 s wall for 0.6 simulated s including ≈ 5 s start-up, i.e. ≈ 45-50 s per simulated second; the locomotion
  documents report 340-820 s per 5 s episode, so one 5 s episode is roughly 5-14 minutes. Several processes can run
  in parallel (the brain step is single-threaded).
- In that smoke run at `g_I` = 1 the pipeline's mean motor rate was ≈ 30 Hz (brain-only tonic estimate: 16-21 Hz).

## Why a mapping correction is needed
The muscle adapter converts pooled motor firing into activation as `a = r / (r + r_half)` with `r_half = 25 Hz`
(`BodyParams.r_half`, `learning_lab` uses the default). At `g_I` = 4 the brain-only motor rates fall to 1.6-4.1 Hz
(`findings-network-calibration.md` section 7), so activation would drop from ≈ 0.4-0.5 to ≈ 0.06-0.14 and the body
would be tested mostly for a mis-scaled actuator, not for the network.

## Design
Code changes (defaults leave behaviour unchanged): `learning_lab.py` gets `--g-inh` (default 1.0, applies
`scale_inhibition` after the connectome is loaded) and `--r-half` (default 25.0, passed as `BodyParams(r_half=...)`
into `MuscleModel`). No other file in the locomotion branch changes.

Calibration (fixed before evaluation, one parameter, different seed): run frozen pilots of 1.0 s on seed 50 at
`g_I` = 1 and at the test `g_I`, read the pipeline's `mean_motor_hz`, and set
`r_half(g_I) = 25 × mean_motor_hz(g_I) / mean_motor_hz(1)`; repeat once with the updated `r_half` and keep it if
mean activation matches within 10%.

Evaluation (the locomotion branch's own validation protocol, frozen original weights, learning and exploration
off, quiet arena, 5 s, seeds 10 and 11 — the reserved held-out seeds 200-202 are not touched):

| Condition | `g_I` | `r_half` |
|---|---|---|
| C1 (locomotion baseline) | 1 | 25 |
| C2 (uncorrected) | 4 | 25 |
| C3 (corrected) | 4 | calibrated |
| C4 (optional) | 12 | calibrated |

Measured: standing fraction (gate ≥ 90%), mean stance slip, heading-forward distance (gate ≥ 1 mm), stance/lift
cycles per leg, mean supporting feet, mean reward, mean motor rate, active fraction of the brain.
Cost ≈ 8 episodes ≈ 1-2 h wall in 3-4 parallel processes.

## How to read it
- C3 ≈ C1 on standing and slip: `g_I` = 4 with a rescaled actuator does not hurt the body, so the tracks can share
  a network; the only remaining joint question is learning (weights trained at `g_I` = 1 are not transferable).
- C3 clearly worse than C1 and C2 not better: the two tracks need different calibrations (e.g. inhibition scaled
  only in olfactory/central populations).
- Two seeds and one seed of network noise per condition give a coarse comparison, not a ranking; a difference
  smaller than the seed-to-seed spread of C1 must not be read as an effect. All results go to a dated findings file.
- Out of scope here: trained weights, habitat/odour drive, exploration, the agent-to-body interface.

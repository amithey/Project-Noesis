# Claude training track: standing, then walking (from 2026-11-25)

Goal: train the fly's connectome "brain" (FlyWire/MANC LIF network in FlyGym 2)
until it passes fixed acceptance gates for **standing**, then for **walking**.
Every run is documented here (`runs/`), and every progress step has a saved
10-second video (`videos/`).

## What was learned from the previous ~40 experiments (E2 to EC1, Oct 28 - Nov 18)

* All used the same family: per-step reward-modulated perturbation plasticity
  on ~85k edges into leg motor/circuit neurons, with OU exploration currents.
* Weight changes stayed tiny (about 1e-5 relative, at most ~1%) and noisy.
  Training rewards rose in some contexts, but **frozen** evaluation never kept
  the gain (EC1: retention failed against parent, reset and stage-1 controls).
* Structured muscle-pool search (MB1/MC1), command limiter (CS1), force bounds
  (AF1), eligibility delay (ED1/ED2), episodic carry (EW1/EC1) all failed gates.
* The frozen parent already "stands" ~91-93% of the time (claw-only support),
  so the standing problem is mostly about stability and slip; walking was never
  observed (no tripod, no consistent forward displacement).

So repeating per-step plasticity with new hyperparameters is unlikely to help.

## Methods in this track

| ID | Method | Why it differs |
|---|---|---|
| ES | OpenAI evolution strategies, antithetic pairs, rank shaping, Adam | Episode-level credit; trains exactly what is evaluated (frozen brain, no exploration current) |
| CMA | Separable CMA-ES | Adapts per-group search scale; robust on noisy returns |
| (later) ES-coarse | ES with gains shared across the six legs (~30 groups) | Lower-dimensional, symmetric; tests if fewer parameters learn faster |
| (later) curriculum | Walk stage starts from the best stand-stage gains | Standing first, then walking |

Parameterisation (all methods): one multiplicative gain per *group* of existing
plastic edges. Group = postsynaptic population (each leg joint's agonist or
antagonist motor pool, or circuit class/side/neuromere) x presynaptic sign.
Topology and signs never change; every magnitude stays within the project's
original-reference +-50% bound. Checkpoints are MotorPlasticity-compatible.

Code: `gpu-pilot/es_train.py`, `gpu-pilot/es_evaluate.py`,
`gpu-pilot/src/noesis_gpu/es_brain.py`.

## Acceptance gates

See [GATES.md](GATES.md). Fixed before any run; never relaxed after seeing results.

## Run log

| Run | Date | Method | Stage | Gens x pop x sec | Best train fitness | Stand gate | Walk gate | Video |
|---|---|---|---|---|---|---|---|---|
| [R0](runs/R0-parent-eval.md) | 2026-11-25 | parent baseline (no training) | - | - | - (held-out v7 0.608/0.617/0.626) | FAIL (standing 0.90-0.91) | FAIL (fwd 0.3-1.6 mm) | [R0-parent-eval-10s.mp4](videos/R0-parent-eval-10s.mp4) |
| [smoke](runs/smoke.md) | 2026-11-25 | ES (pipeline check) | stand | 1 x 5 x 0.5 s | 0.652 (centre) | n/a | n/a | - |
| [R1](runs/R1-es-stand.md) | 2026-11-25 | ES | stand | 3/20 x 13 x 1.5 s | aborted (best = parent; varying seed) | not run | not run | - |
| [R1b](runs/R1b-es-stand-fixedseeds.md) | 2026-11-25 | ES, fixed seeds 30000-1 | stand | 15 x 13 x 2 seeds x 1.5 s | 0.742 (parent 0.706) | FAIL (standing 0.94/0.98/0.87; v7 mean 0.658 vs 0.617) | FAIL | [R1b-...-10s.mp4](videos/R1b-es-stand-fixedseeds-10s.mp4) |
| [R2b](runs/R2b-cma-stand-fixedseeds.md) | 2026-11-25 | sep-CMA, fixed seeds 30000-1 | stand | 15 x 13 x 2 seeds x 1.5 s | 0.751 (parent 0.706) | FAIL (falls on 20002: upright 0.21; v7 mean 0.436) | FAIL | [R2b-...-10s.mp4](videos/R2b-cma-stand-fixedseeds-10s.mp4) |
| [R3](runs/R3-es-stand-from-R1b.md) | 2026-11-26 | ES from R1b (no change) | stand | 12/15 x 13 x 2 seeds x 1.5 s | 0.765 (start 0.742) | FAIL (standing 0.91/0.92/0.84; v7 mean 0.591; no improvement) | FAIL (all-6-legs steps PASS) | [R3-...-10s.mp4](videos/R3-es-stand-from-R1b-10s.mp4) |
| [R0c](runs/R0c-parent-sizemuscle-eval.md) | 2026-11-26 | parent, `--muscle size` (no training) | - | - | - (held-out v7 0.407/0.437/0.557) | FAIL (standing 0.77-0.86; sits on abdomen) → size muscles not adopted | FAIL | stills in runs/audit-legs |
| [R4](runs/R4-es-stand-4seeds.md) | 2026-11-26 | ES from R1b, `--seeds-per-gen 4` | stand | 7/12 x 13 x 4 seeds x 1.5 s | interrupted; 0.7004 (start 0.6998) | not run | not run | - |
| [legs-fix](runs/legs-fix.md) | 2026-11-26 | `--legs real` (rest/range from real fly kinematics); A/B/C comparison | - | - | - | A .905 / B .633 / C .336 mean standing | - | - |
| [R0d](runs/R0d.md) | 2026-11-26 | baseline A: native legs, mean muscles (no training) | - | - | - | FAIL | FAIL (stride ≥0.3 PASS) | [R0d-baseline.mp4](videos/R0d-baseline.mp4) |

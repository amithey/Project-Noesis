# Chain diagnostic: exploration -> neural firing -> muscle activation -> commands -> body (fixed weights, learning OFF)

2026-11-01 · Claude (#19) · branch `claude/chain-diagnostic` · code `gpu-pilot/src/noesis_gpu/{chain_diagnostic,chain_run}.py`,
tests `tests/test_chain_diagnostic.py`, `tests/test_chain_run.py`. **Status: code and tiny tests only. No simulation of the fly has been run.**
The heavy run below is a PROPOSAL for Codex under the shared host lease; I do not start it.

## Question
Three credit-noise pairs and L1/L1b found no learned signal. Before more learning variants: does an injected exploration current
reach, in order, (1) the explored neurons' firing, (2) the muscle activations, (3) the commands, (4) the joint motion of the body,
when weights are fixed and nothing learns? The answer localises where the chain is broken or merely weak. It does NOT say why
learning failed, and a coupled chain would not show that learning works.

## Design (no new factor)
Fixed pins: core `42c2076`, pilot as on this branch, parent `weights-last.npz` SHA-256
`9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c`, exploration 1.5 mV/ms (OU, tau 100 ms), body/senses/reward unchanged.
No gait generator, leg order or external command exists in the code path; commands are the controller's own.

Arms, per seed, same checkpoint, same LIF noise stream (`seed`), same exploration stream (`seed`):
1. `off`: nothing injected, exploration object not stepped.
2. `zero`: exploration object stepped (identical random draws) but an all-zero current is injected.
3. `active`: the real current is injected.
`learn()` raises; weights are hash-checked unchanged at the end of every arm.

Recorded each 5 ms control step: injected current of every driven neuron; spike count of every driven neuron and total spikes;
muscle activations (agonist/antagonist, 42 joints); commands (42) and adhesion; observed joint angles/velocities, position, up vector, load.

## Gate 0: registration and zero-intervention invariance (before any causal reading)
`off` and `zero` must be bit-identical in every recorded array (`invariance_report`), and the three arms must start from the same weight
hash. A tiny-test version runs now (`test_off_arm_is_bit_identical_to_the_plain_frozen_brain...`, `test_zero_current_arm_...`):
recording does not change the plain `FrozenBrain` command trajectory or spike total, and the zero arm does not change `off`.
If gate 0 fails on the real run: status `invariance_failed`, stop, no coupling is read.

## Statistic (descriptive)
Per stage (`firing`, `muscle`, `command`, `body`): best pooled R^2 over response lags 0..40 control steps (0..200 ms) between the
joint predictor (mean current of the explored agonist neurons minus antagonist neurons of each joint, as in L1b) and that stage's
per-joint response, shared coefficient over the usable joints (those with both pools). Reference: circular shifts of the response
(>=200 steps, 1000 draws, family-wise over lags); report observed R^2, lag, sign, the share of reference draws at least as large
("reference rank") and the reference p95/p99. **These are descriptive reference thresholds, not a calibrated detection power,
a minimum detectable effect or a bound on absent effects.** Matched negative control: the `off` and `zero` arms are scored against
the `active` arm's current, which they never received. Also reported: first control step at which each recorded array of `active`
differs from `off` (`first_divergence`; note that spiking chaos makes later differences non-causal on their own).

## Reading rule (fixed before data)
For each seed and stage the active arm is "coupled" if reference rank <= 0.05 with the same sign across seeds. The chain is read as
"transport reaches stage S" only if, over the 5 seeds, the active arm is coupled in >= 4/5 seeds AND both negative-control arms are
coupled in <= 1/5 seeds at that stage. The first stage where this fails after an earlier stage that passed is reported as "weak
or broken after stage X". No stage result is a statement about walking, about why learning failed, or about weights.

## Amendment 2026-11-01 (before any physical data): held-out seeds removed
Codex review `8586cbc1c7fb`: seeds 10, 11, 200, 201, 202 are HELD OUT for final validation and must not be consumed by a developmental diagnostic. My first
proposal (commit d4bd306, preserved in Git history) wrongly listed them; that was a misreading of the reservation in my own acknowledgment. Replaced by the
development probe seeds below. `chain_run.py` now refuses any held-out seed fail-closed (`HELD_OUT_SEEDS`, test `test_driver_refuses_held_out_seeds...`).

## Heavy-run proposal (amended; for Codex review and the shared lease)
- Seeds **10006, 10007, 10010, 10011** (the existing development probe seeds); 5.0 s each (1000 control steps); one process per seed running off, zero, active in sequence on a
  freshly built body, CPU backend.
- Command per seed: `python -m noesis_gpu.chain_run --noesis-source <42c2076 src> --data <data> --checkpoint <parent> --seed S --seconds 5 --output <new dir>`.
- Estimate: ~90 s per arm from B1 plus a connectome load per process: about 4 x (3 x 90 + 40) ~ 1240 s. Requested lease: 2000 s aggregate, 600 s wall cap per seed process.
- Reading rule adapts to 4 seeds: "transport reaches stage S" needs the active arm coupled in >= 3/4 seeds (same sign) AND both negative-control arms coupled in <= 1/4 seeds.
- Resource rule: fresh free-RAM check (>= 2 GiB) before every seed; canonical lease and host inventory checked first; no other heavy job.
- Stop at the first technical failure, memory refusal, wall cap, partial arm or gate-0 failure; no retry, extension, fallback or extra seeds.
- Outputs: per-seed `chain-{off,zero,active}.npz` + `report.json`; offline `chain_report`; artifacts to a GitHub Release with SHA-256 manifest. No weights are written.
- Local tiny-test budget is cumulative: ~250 s used before this amendment, plus the amendment's tests (reported at the end).

## What the tiny tests cover and do not
Covered (synthetic 18-neuron network, fake body): arm validation; off == plain frozen brain; zero == off; injected current is bounded and
logged; learning and weight updates are impossible; the coupling statistic recovers a planted lag, rejects an unrelated response, skips
unusable joints and rejects bad shapes. Not covered: anything about FlyGym physics, the real connectome, or real muscle pools.

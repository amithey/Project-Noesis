# Young fly: brain-only learning protocol

Implemented on `codex/brain-body-feedback`, in the isolated managed worktree.
The connectome edge set and signs remain unchanged; selected incoming motor weights
can change by at most 50%. There is no gait clock, CPG, scripted tripod sequence,
or obstacle-coordinate input to the brain. Muscle equilibria and geometric adhesion
are explicit body-interface assumptions, not learned anatomy. PAM/PPL1/MBON/KC firing
is telemetry and does not establish emotion, thought or consciousness.

## Prediction fixed before the new trials

Previous pilots showed moving legs and substantial sliding, without demonstrated
walking. Correct adhesion geometry and a slip-aware reward may improve support,
but do not themselves imply repeated steps or useful learning. Uniform synaptic
depression previously harmed sensory responses. Weak excitatory-only depression
must earn passage through the unchanged sensory battery and recovery gate.

## Implementation

- Scratch MuJoCo kinematics predicts foot lift from muscle target angles without
  modifying the live body. Up to three lifted pads release; no leg order is imposed.
  The 20 micrometre threshold is a stated interface assumption. The earlier
  positive-Femur-angle release rule had the wrong geometric sign.
- Ground support sums positive normal forces only between tarsal geometries and
  the actual ground. Self contacts, thorax-ground contacts and obstacle contacts
  cannot create support. Obstacle contact is measured separately.
- Reward uses horizontal thorax heading, at least three supporting legs, stance
  slip, posture and effort. Progress is multiplied by
  `clip(1 - slip/(abs(forward_speed)+0.1), 0, 1)`; sustained obstacle contact costs
  up to 0.2 reward after one second. These scales are engineering assumptions.
- Motor perturbation learning uses rate-normalized presynaptic activity, signed
  exploration innovations, a 100 ms eligibility trace, learning rate 0.2, and
  bounded 1.5 mV/ms motor exploration. It is an action-credit heuristic, not a
  demonstrated biological learning rule or an exact policy gradient.
- A quiet arena precedes a visible, collidable box ahead of the initial body.
  Retinal observations, actual touch, joint feedback and balance drive the brain.
  Quiet temperature/humidity are constant and odor/taste/sound/wind fields zero.
  External sensory ablation disables the environment pathways while leaving
  joint feedback and balance enabled.
- Optional depression scales positive recurrent weights only; inhibitory output
  and external drive retain their amplitudes. Default U=0 remains unchanged.
  `excitatory_screen` compares U=0/.03/.1, recovery=300 ms, using the existing
  response-battery P1-P8 thresholds unchanged. Passage requires preserving every
  baseline-passing assay, passing P8, and at least 10% lower mean recovery firing.
  Nonzero depression in `train_validate` requires a matching completed screen.

## Acceptance and restart protocol

`train_validate` uses validation seeds 10,11 and held-out seeds 200,201,202, all
disjoint from training. Every frozen behavioral trial lasts five seconds with
learning and motor exploration off. Quiet trials compare original and trained
weights. Obstacle trials additionally compare trained weights with external senses
off. Validation decision is persisted before held-out trials are observed; final
promotion requires both stages to pass.

Every matched quiet trial requires standing fraction >=0.9 (uprightness >=0.7,
height 0.3..1.5 mm), heading-relative progress >=1 mm, two physical stance/lift/stance
cycles in at least four legs, at least 20% less slip than original, and reward gain
>0.001. States must persist for 20 ms; lift requires foot height >0.05 mm. This
rejects mere adhesion changes, chattering contacts and translating without steps.
Neural tripod score is reported independently.

Obstacle passage requires a >=30 degree turn or a 100 ms sustained stop relative
to nonzero pre-contact motion within one second of first contact, plus less contact
than the sensory-ablation control. A missing contact, already motionless fly, or
truncated observation window cannot pass.

The report stores independent `last_training_checkpoint` and
`best_validated_checkpoint` paths. Failure or incomplete controls never replaces
best. Each attempt has its seed, start checkpoint, output directory and completion
state. Resume skips completed jobs and reruns interrupted episodes from their start,
not from partial weights. Completed cycles retain history; a new cycle continues
from the last training weights. Reports are atomically replaced. Resume rejects
changed source/configuration to prevent mixing model versions.

`--wall-seconds` is positive and at most 3600. The episode checks its deadline at
5 ms control boundaries, reserving five wall seconds for artifacts. A process
watchdog also enforces the limit during native initialization and video encoding.
If it exits with code 124, the persisted running attempt is marked interrupted on
resume and its partial artifacts remain separate.

## Commands

From this worktree, with the existing body environment:

```powershell
./Train-YoungFly.ps1 -Video
./Train-YoungFly.ps1 -Resume -Video
```

Or explicitly:

```powershell
$env:PYTHONPATH='src'
python -m noesis.experiments.train_validate --data <existing-data> --output outputs/young-fly --wall-seconds 3600 --video
python -m noesis.experiments.train_validate --data <existing-data> --output outputs/young-fly --wall-seconds 3600 --resume --video
python -m noesis.experiments.excitatory_screen --data <existing-data> --output outputs/excitatory-screen --wall-seconds 3600
# Add --resume to continue the screen. A passing nonzero U also requires:
# train_validate ... --depression-u .03 --dynamics-report outputs/excitatory-screen/report.json
```

The primary checkout's regular `Start-Fly.ps1` launcher has not been updated.
Use the dedicated trainer in the isolated worktree for this experimental protocol.
No failed or incompletely evaluated candidate is made the primary default.

## Measurements

Real run results and outstanding controls are recorded below after completion of
the bounded cycles. Earlier subsecond pilots are separate from this protocol and
cannot establish five-second walking or obstacle behavior.


### Completed earlier pilots

v2's old validation rejected the candidate because slip increased by 1.230 mm/s.
Its held-out mean reward was -0.01961 original vs -0.02217 trained. The remaining
v3 held-out trials (original 201, candidate 200/201) were replayed from episode
start, with their original 0.6 s durations, all-leg-sensor contact definition,
additive slip reward and geometric adhesion. Additional telemetry does not change
that old objective. The adapter is preserved in `outputs/complete_legacy_pilot.py`.
v3's held-out reward worsened from -0.00538 to -0.03832, and slip increased from
7.289 to 8.018 mm/s. Its provisional validation approval did not generalize.
Both pilots retain their artifacts and selected checkpoint is null. Neither is
eligible for the new five-second protocol.

### Excitatory-only screen, seeds 10 and 11

All 36 matched trials completed (3 U values x 2 seeds x 6 stimulus conditions).

| U | Mean recovery firing, Hz | Sugar PAM, Hz | Original P8 | Accepted |
|---|---:|---:|---|---|
| 0 | 6.02459 | 0.98399 | fail | control |
| 0.03 | 0.42905 | 0.05872 | fail | no |
| 0.1 | 0.00607 | 0.00178 | fail | no |

Both candidates failed P5 (sugar -> PAM) and P8. Lower mean recovery activity
alone is insufficient: P8 counts every neuron active during the recovery window,
including transients. U=0.03's worst recovery had 13,446 active neurons; U=0.1's
worst had 1,239, compared with zero rest activity and the unchanged +50 threshold.
The physical training therefore stays at U=0. These are two-seed engineering
screens, not an empirical biological validation.

### New bounded physical run

`outputs/birth-cycle-v4`: 600 wall seconds, three 1.2 s training episodes,
followed by the fixed five-second frozen protocol. Exploration is active only in
training. The first quiet episode shows physical motion but no learning verdict.

| Seed | Scene | Standing fraction | Forward mm | Legs with >=2 cycles | Stance slip mm/s |
|---|---|---:|---:|---:|---:|
| 0 | quiet | 0.970 | 1.409 | 4 | 5.854 |
| 1 | obstacle | 0.961 | 1.559 | 3 | 5.698 |
| 2 | obstacle | 0.822 | -0.467 | 4 | 7.722 |

Seed 2 failed the 90% standing criterion and moved backward relative to its body
heading. No validated tripod rhythm was found in any training episode. Training
metrics cannot satisfy the frozen evaluation gate, even when an episode advances
more than one millimetre. The videos include this unstable episode as well as the
more stable first episode.

To continue this exact run, from the isolated worktree:

```powershell
./Train-YoungFly.ps1 -Resume -Output outputs/birth-cycle-v4 -Seconds 1.2 -WallSeconds 3600 -Video
```

Verification: 158 unit/integration tests passed, including real MuJoCo contact
forces and adhesion geometry, signed reward credit, delayed reward, old/new
checkpoint compatibility, frozen exploration, sensory ablation configuration,
partial episode replay, best-model preservation and a subprocess hard-deadline
check. `ruff check src tests` passed. The PowerShell trainer parsed successfully.


The 600-second watchdog stopped finalization of the first original-seed-10
validation attempt. Its episode artifact records `wall_budget` at 1.08 simulated
seconds. It is **not** a completed five-second control. A real `--resume` check
accepted the unchanged source/configuration and marked that attempt interrupted;
a deliberately four-second resume budget then ended with watchdog exit code 124
during loading. No episode was counted twice or promoted. All 25 frozen controls
remain pending, the last completed training checkpoint is retained, and
`best_validated_checkpoint` remains null. Resume requires enough time to complete
a five-second episode from its start (use the one-hour command above).

Machine-readable measurements: `docs/young-fly-results.json`. Live ledger:
`outputs/birth-cycle-v4/report.json`. Full sensory screen:
`outputs/excitatory-screen-v1/report.json`. Outputs were also copied to the primary
checkout under `outputs/young-fly-v4-artifacts` for preservation. Video framing is
fixed; the fly partly exits the left edge during the backward, unstable episode.

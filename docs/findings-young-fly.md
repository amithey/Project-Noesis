# Young fly: brain-only learning protocol

**Physical-pilot update:** a subsequent body-construction audit found that head/wing
extension mutated FlyGym's shared monitored-joint defaults. This changed servo gains
and damping in later episodes. The physical v2/v3/v4 measurements below are historical
and do not validate a trained model. See the repeatability correction and v5 run below.
The neural-only depression screen is unaffected by this body-construction defect.

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


## Continuation: stable body construction and longer training

A repeated original-weight, seed-10 episode first disagreed at the physical level
before the brain woke. Model inspection isolated the cause: identical recompilations
had a maximum damping difference of 0.98 and actuator-gain difference of 30.0.
`enable_wing_and_head_joints` appended to `fly.monitored_joints`, which referenced
FlyGym's shared `preprogrammed.all_leg_dofs` default list. Later Fly constructors
therefore treated the extra joints as default actuators and assigned different
head/wing servos. The controller now takes independent copies of both joint lists
before extending them. No learned or scripted leg sequence was added.

A new real-MuJoCo regression compares damping, gains, biases, stiffness and 200
pre-neural settling steps across independent bodies, and checks idempotent extension
and unmodified FlyGym defaults. Two actual 0.15-second brain/body episodes with seed
10 now match exactly for position, spikes, motor firing, vision and ground loads;
the corresponding v4 repetitions differed. These short repetitions verify this
construction fix, not five-second walking success. All 160 tests and Ruff passed.

The affected v4 continuation was stopped after preserving its artifacts. Its first
five-second original-weight control had standing fraction 0.9414, forward projection
6.6391 mm and stance slip 6.0081 mm/s. It is diagnostic evidence only: later bodies
in that process were different from first bodies in resumed processes. The ledger
is explicitly invalidated and no candidate is promoted from it.

Synaptic experience remains available. `train_validate --import-weights <checkpoint>`
starts a **new** protocol with compatible weights, recording the source checkpoint.
Topology, original signs, 50% bounds and learning-rule parameters remain checked.
Each newly saved checkpoint records body interface v2, body parameters and joint
names; ordinary loads reject mismatched model metadata. `learning_lab` requires an
explicit `--import-weights` switch for this deliberate model transfer.

`outputs/birth-cycle-v5` imports the last v4 training weights and trains one ten-second
quiet-floor episode, instead of resetting after 1.2 seconds. It then runs the same
frozen validation seeds 10/11 and held-out seeds 200/201/202, including obstacle
controls. Its current invocation has a 3,600-wall-second budget. This is ongoing
work toward verified standing and basic walking; no walking success is yet claimed.

To resume that exact protocol after its current process finishes:

```powershell
$env:PYTHONPATH='src'
python -u -m noesis.experiments.train_validate --data <existing-data> --output outputs/birth-cycle-v5 --episodes 1 --seconds 10 --wall-seconds 3600 --import-weights outputs/birth-cycle-v4/cycle_000/train-2/attempt_0002/episode_000.npz --video --resume
```

Do not launch a duplicate while the current invocation is live. The active process
and report, rather than a log timestamp alone, determine whether a resume is needed.


## Continuation: reflected right-leg rotational coordinates

The v5 quiet-floor training episode completed ten simulated seconds. Recorded
standing fraction was 0.9171, heading-relative progress 3.4333 mm, and physical
stance/lift/stance cycle counts LF=17, LM=26, LH=33, RF=11, RM=12, RH=33. Mean stance
slip remained high at 6.6259 mm/s. Mean absolute synaptic change was 0.00013797
(relative to original magnitudes), maximum 0.00323573, well within the 50% bound.
These training measurements are not a frozen walking verdict.

A further independent geometric audit found an output-coordinate defect. The
anatomical muscle table was applied with identical angular signs on both sides,
but the actual body and its reference kinematics reflect Coxa_roll, Coxa_yaw and
Femur_roll on the right. Equal homologous motor activation therefore produced
unmirrored physical feet:

| Muscle | Old mean left/right reflection error, mm | Corrected error, mm |
|---|---:|---:|
| sternal_anterior_rotator | 0.61364 | 0 |
| pleural_remotor_abductor | 0.75737 | 0 |
| sternal_adductor | 0.75737 | 0 |
| femur_reductor | 0.99698 | 0 |
| tibia_flexor (unchanged control) | 0 | 0 |

Both `build_motor_matrix` and `build_muscle_matrices` now reflect the right-side
roll/yaw coordinates. Coxa/Femur/Tibia/Tarsus1 flexion coordinates retain their
signs. This corrects the motor-neuron-to-muscle/body interpretation; connectome
synaptic signs, topology and learned magnitudes are not changed. Five new tests
activate homologous synthetic motor pools and check the actual MuJoCo skeleton's
foot positions. Before the correction, four tests failed and the flexor control
passed. After correction all five passed. Full suite: 165 passed; Ruff passed.

The v5 evaluation was stopped only after its ten-second training checkpoint was
complete. Its experience was explicitly imported into `outputs/birth-cycle-v6`,
which uses checkpoint model schema 5 and `motor_interface=mirrored-leg-rotations-v1`.
The v6 invocation trains ten quiet-floor seconds and then runs the unchanged
five-second frozen control protocol, with a 3,600-wall-second budget. It is live
at this snapshot; the requested standing/basic-walking goal is not complete.

A separate 0.3-second frozen audit measured real tangential contact-point velocity
using MuJoCo contact Jacobians. Mean contact-point slip was 4.9996 mm/s, compared
with 4.1712 mm/s for the existing distal-foot-origin proxy. Thus substantial slip
is real, not eliminated by changing the measurement point. No slip threshold or
acceptance criterion was weakened. The birth pose was also checked by forward
kinematics: all six distal feet are below the thorax and left/right symmetric.

Resume v6 only after its live invocation ends, retaining its import provenance:

```powershell
$env:PYTHONPATH='src'
python -u -m noesis.experiments.train_validate --data <existing-data> --output outputs/birth-cycle-v6 --episodes 1 --seconds 10 --wall-seconds 3600 --import-weights outputs/birth-cycle-v5/cycle_000/train-0/attempt_0000/episode_000.npz --video --resume
```

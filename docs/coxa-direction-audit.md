# Coxa direction audit and corrected-body training progress

The goal remains unverified. The v8 protocol completed and was rejected. It used the corrected proximal
flexion/extension interface; its archived source fingerprint is `e781c5f9e9c00d318594f939922679b36f75897b3d53cdb574aaea9638957a66`.

## Completed training, not acceptance

Training seed 2 completed ten simulated seconds with exploration active. Standing
fraction was 0.977387; mean supporting feet 2.839196; all six legs had repeated
physical cycles (LF=30, LM=30, LH=34, RF=25, RM=28, RH=26). Heading-relative forward
motion was only 0.272940 mm and mean stance slip 12.337649 mm/s. These measurements
show substantially more support than the preceding training episode, but they
are different seeds/body versions with exploration and do not establish a causal
learning gain, frozen walking, or acceptance. All completed weights and traces remain
in `outputs/birth-cycle-v8/cycle_000/train-0/attempt_0000`.

Slip remains high even without new foot contacts. New-contact frames account for
30.8% of frames and 31.6% of the slip total: mean 12.640627 mm/s on touchdown frames
versus 12.202772 otherwise. Touchdown transitions do not explain away this failure.
The slip reward and 20% improvement criterion are unchanged.

## Additional physical direction error

At the model's standing pose, small activation of sternal anterior rotator and
tergopleural promotor moves the distal coxa backward along the thorax's forward
axis in all six legs. Sternal posterior rotator and pleural remotor/abductor move
it forward. This conflicts with their annotated anterior/protraction and
posterior/retraction functions in the
[primary anatomical appendix, Table A1](https://faculty.washington.edu/tuthill/docs/azevedo24_appendix.pdf).
The proposed change reverses ONLY the `Coxa` component of these four groups.
All existing roll/yaw couplings and active degrees of freedom are retained.
It preserves graph topology, neural weight signs/bounds, learning, sensors, adhesion,
reward and acceptance requirements. The motor-interface tag becomes
`anatomical-coxa-flexion-v3` so normal loading cannot silently mix body contexts.

For the tarsus, unsigned interior angle alone is insufficient to identify its
functional direction at the rest pose. Direct physical foot displacement confirms
that the present depressor lowers the foot and levator raises it, in all six legs.
Those signs remain unchanged. The long tendon controls a claw in the anatomical
model ([primary motor-circuit paper](https://www.biorxiv.org/content/10.1101/2022.12.15.520299v2.full));
its existing single-joint proxy and the other multi-axis couplings remain modeling
assumptions, not newly validated anatomy. No new claw model or coupling is introduced.

## Tested, reviewable candidate

The new regression activates each named pool, evaluates both motor matrix APIs and
the muscle model, and measures physical coxa displacement in the thorax frame.
Before correction, four coxa cases fail while both tarsal direction controls pass.
The separately corrected snapshot passes **204 tests** and Ruff, including the
preceding proximal flexion, bilateral reflection, reward, sensing, checkpoint and
resumable training tests. Its source fingerprint is
`d6178c27c836871612233fa5ed7e88ebb6ec274050ce5eaf58cb54a061d8ddb7`.
These are interface tests, not behavioral evidence.

The portable patch is `docs/patches/coxa-direction.patch`; `git apply --check
--ignore-space-change` passed. It includes the motor correction, an external recording-camera helper, and
physical direction and real-rendering regression cases. The exact tested source is preserved separately in
`outputs/protocol-v9-candidate/src/noesis`, and current v8 source is archived in
`outputs/protocol-v8-source/src/noesis`. The patch was applied only after the v8 session exited with rejection.
The new protocol explicitly imports the last complete training weights with reset
traces and uses a new output directory. Previous-body controls cannot validate the new interface.
Original and candidate frozen tests remain five seconds, with the same reserved
validation and heldout seeds and all existing promotion gates.

Machine-readable evidence: `docs/coxa-direction-audit.json`. Full regression log,
physical audit script and raw diagnostics: `outputs/anatomical-audit-v8`.


## Recording repair

Visual inspection of the v8 training video's true midpoint found the body mostly
outside the frame. The enriched-world camera is fixed; completed original seed 10
also moves about 8.9 mm in world x and leaves that view. These clips are retained
as imperfect recordings, not presented as complete visual evidence of gait.

The staged candidate now follows fly x/y from the external recording camera,
keeps a fixed world orientation and floor-relative height, and includes simulated
time and 0.1x playback labels. It changes no eye camera or sensory channel.
The real MuJoCo regression checks visibility near the origin, at (15,-8) mm and
at (-20,10) mm with a flipped body; red/brown body pixels remain in view. The
full integration state, qpos/qvel/controls, accelerations and warm starts, eye
camera transforms, sensory arrays and last-observation arrays remain exactly
unchanged after rendering. The distant flipped frame was visually inspected.
The complete combined source passes 204 tests and Ruff with the staged source
root declared for import classification. The original motor-only candidate had
already passed 203 tests. The installed source is byte-identical to the complete tested candidate. After
installation, 25 focused motor, real-render and episode integration tests and
default project Ruff also passed.


## Completed frozen rejection and active v9 training

Frozen seed 10 under the v8 body: candidate standing fraction 0.986869 versus
original 0.864646; all six legs have at least two physical cycles. Candidate
heading-relative forward motion is -1.015620 mm and slip is 11.248535 mm/s,
versus original slip 12.099534 mm/s. Slip reduction is 7.033%, below
the required 20%; candidate slip must be <=9.679627 mm/s. Standing, cycles and
reward pass, while forward progression and slip fail. Candidate seed 11, obstacle
and heldout controls were not run and are not counted complete. No model was promoted.

The terminal v8 session and complete last training checkpoint were verified before
installing the tested correction. `outputs/birth-cycle-v9/report.json` is now
authoritative for the active model. Training seed 3 uses a twenty-second episode
and explicit weights-only import from
`birth-cycle-v8/cycle_000/train-0/attempt_0000/episode_000.npz`. The longer training
horizon accumulates more experience before each matched evaluation; evaluation
remains five seconds, with validation 10/11 and heldout 200/201/202. The existing
3600-second absolute budget, restart-from-episode-start semantics and full promotion
gates remain unchanged. Original controls will be recomputed for this body interface.

Resume only after its process is terminal, preserving all settings:

```powershell
$env:PYTHONPATH='src'
python -u -m noesis.experiments.train_validate --data <existing-data> --output outputs/birth-cycle-v9 --episodes 1 --seconds 20 --seed 3 --wall-seconds 3600 --import-weights outputs/birth-cycle-v8/cycle_000/train-0/attempt_0000/episode_000.npz --normalize-reward-error --early-reject --video --resume
```

The primary checkout's normal launcher remains unchanged. Interface tests establish
correct directions and faithful recording; only full frozen physical evaluations
can establish standing and useful forward walking.

# More upright experience before longer trials

This training-protocol option is now applied after v10 completed and was rejected.
It is not a behavioral success. V11 uses three five-second quiet-floor births with
seeds 5/6/7, importing the last complete v10 training checkpoint. The preceding
source is preserved in outputs/protocol-v10-streaming-source/src/noesis.

The preceding twenty-second episode fell after roughly ten seconds and did not
recover, standing during only 48.8% of its awake measurement window. Its full
five-second frozen candidate subsequently failed standing and slip. In the active
v10 episode, live observations also show a prolonged fall later in the episode.
Those failures remain in the evidence; no training clip is treated as acceptance.

The hypothesis is that several shorter, independently seeded births on the quiet
floor can provide more useful upright action/reward experience than spending much
of one long episode already fallen. Neural and body state reset only at ordinary
episode boundaries. Complete learned weights carry forward; an interrupted episode
replays from the preceding complete checkpoint. No joint target or leg sequence is
introduced by this curriculum.

The proposed `--training-arena quiet` allows three five-second floor episodes with
seeds 5/6/7. `graded` remains the backward-compatible default, and `obstacle` permits
a later obstacle-focused stage. The setting selects ONLY training scenes. Every
five-second original/candidate quiet test, obstacle/sensory-ablation control, and
heldout seed remains mandatory under the existing promotion gate. Exploration is
1.5 mV/ms only during learning, LR is 0.2, eligibility 100 ms, synaptic signs and
50% bounds are unchanged. Every frozen test still disables learning and exploration.

All 213 tests and Ruff pass in the separate snapshot, including a new integration
case that chains three quiet checkpoints, interrupts the third episode, replays it
from the preceding complete weights, and then executes all 25 frozen control cases.
The PowerShell launcher parses. Only the training job builder, CLI/launcher option
and regression tests change; neural, body, sensory, reward and acceptance code is
unchanged. The portable patch passes `git apply --check --ignore-space-change`.

Do not apply this patch or change the active run's settings until that process is
terminal. Complete its current frozen evaluation first. If quiet criteria still
fail, use the short quiet curriculum in a new output directory with explicit import
of the last complete training weights. Choose later training scenes from completed
results; quiet-only training does not waive obstacle validation or authorize promotion.

The streaming recorder is already active: six samples from 9.6 through 13.4
simulated seconds show learner private memory around 3.5 GB, varying by about
10 MB, rather than accumulating the RGB video arrays. This is actual memory
measurement, not proof of walking or a controlled wall-time speed comparison.

Evidence: `docs/quiet-training-curriculum.json`,
`outputs/curriculum-audit-v10/full-regression.log`, and
`outputs/protocol-v11-curriculum-candidate/src/noesis`.
Patch: `docs/patches/quiet-training-curriculum.patch`.

## Completed v10 and contact diagnostic

V10 frozen candidate seed 10 stood for 48.08% of its five-second window, below
the required 90%. Its 11.53 mm heading displacement and 10.26 mm/s legacy slip
do not establish walking while the standing gate fails. Remaining candidate,
obstacle and held-out cases were not run; no model was promoted.

A one-second frozen original seed-10 diagnostic measured actual tangent contact
velocity using MuJoCo geom velocity and the contact-point lever arm, independently
checked against the point Jacobian. Three real-physics tests passed, including
rolling without slip and excluding torso contacts. All 200 common connected-brain
rows remained exactly equal to the official original-control prefix and all
endpoint leg loads matched. Awake mean legacy slip was 11.96 mm/s; actual endpoint
contact slip was 14.36 mm/s and physics-step interval average was 14.60 mm/s.
This sample confirms genuine sliding; it does not support replacing the reward
on the assumption that the legacy proxy merely exaggerates slip. Reward and
acceptance criteria therefore remain unchanged in v11. Diagnostic-only columns
are not official validation results.

The installed snapshot exactly matches the source that passed 213 full tests.
The 11 focused curriculum/runner checks and Ruff also pass after installation.
V11 remains subject to the complete frozen acceptance matrix; early rejection
records missing cases as incomplete.

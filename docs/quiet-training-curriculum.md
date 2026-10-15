# More upright experience before longer trials

This is a tested training-protocol candidate, not an applied model change or a
behavioral success. The active v10 run continues under its original source.

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

# Existing motor-circuit synapses: optional learning prototype

V26 mean-bank frozen10 failed standing (86.77%) and slip (14.98mm/s versus
original9.37). It remains retained as latest training, not a validated model.
The present learner changes73594existing synapses entering498motor/readout
neurons. Inputs to local motor-circuit interneurons remain fixed. This is a
limitation of the current learning scope, not proof that expanding it will work.

The isolated outputs/premotor-credit-v28-prototype currently adds only a pure
selection function, learning_targets.py, to a copy of the currentcd369source.
Twelve focused tests and Ruff pass. Actual original corrected-connectome
preflight confirms selection of48local candidates (eight for each leg) and
16sequential-leg candidates. Those are anatomical cell labels plus existing
direct projections to mapped motor neurons, not physiological rhythm labels.
The resulting562target identities would make85390existing input edges eligible,
compared with73594now. No edge is added, removed, reordered or re-signed.
Full-connectome weights and the original motor reference arrays remain exact.

Selection uses original absolute projection weight, with index tie breaks.
Each leg has anatomical coverage, but there is no phase, gait clock, joint
trajectory or step sequence. The64selected identities are recorded in
full-connectome-selection.json. Parent remains retained mean bankce1e6ca0...
so prior motor-synapse experience can be transferred explicitly.

Before any installation or body training, complete these parts:

- Extend bounded exploratory current/innovation credit to these CNS targets,
  using a separate RNG so the existing motor-pool noise stream stays exact.
  Default behavior stays unchanged; evaluation has zero exploration everywhere.
- Add checked old-subset import into expanded plasticity. Construct expanded
  original references before applying old learned weights, validate topology,
  original references/signs/bounds and target inclusion, and reset all traces.
  Existing checkpoint/default learning behavior must remain compatible.
- Integrate the opt-in scope in learning_lab/train_validate with explicit model
  context. Keep rate.2, eligibility100ms, amplitude<=1.5mV/ms, original +/-50%
  bounds, depression0 and the same body/sensory/reward/physical gates.
- Run meaningful new unit/integration checks, full existing regression/Ruff,
  actual full-connectome import/save/load round trip and frozen-noise guards.
  Only then install and conduct bounded training with frozen matched controls.

This is selection-only staging: no new circuit has been trained, no installed
source or normal startup changed, and no walking success is claimed. Full frozen
validation/heldout/obstacle/ablation and real video remain required for promotion.

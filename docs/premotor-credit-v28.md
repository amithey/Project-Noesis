# Learning inside existing motor circuits

Opt-in `--learning-scope motor-circuit-v1` lets reward-modulated learning update
existing inputs of64anatomically selected CNS cells as well as existing motor
readouts. The default `motor` path and its checkpoint/model context remain
compatible. This extends a learning hypothesis; walking remains unverified.

V26 mean-bank frozen10 failed standing86.77% and slip14.98mm/s versus original
9.37. That bank is retained as the V28parent, preserving motor-synapse experience.
The new scope selects48local leg-neuromere cells (eight per leg) and16sequential
leg-neuromere cells by absolute original projections into mapped leg motor
neurons, with index tie breaks. The labels are anatomical, not rhythm labels.
562target identities make85390existing input edges eligible, compared with
73594in the motor-only scope. No connectome edge is added, removed or re-signed.

The new cells receive independently seeded bounded OUcurrents/innovation credit.
Their RNG is separate, so the old motor-pool stream is exact at every step.
Evaluation sends no exploratory currents to either population. Selection and
noise prescribe no leg phase, joint angle, gait clock or step sequence.

Explicit `--import-weights` accepts a validated old target subset into a fresh
expanded learner. Original expanded references are captured first, then old
learned motor weights are applied. Added target edges stay original. Topology,
identity, sign, bounds and subset inclusion are checked before mutation. All
traces/baseline/variance/normalization mass reset. Checkpoint format4 remains
valid; opt-in model context is schema9 with scope and selected CNSidentities.
Old scope retains schema7/8. The motor-Hz readout still measures actual motor
neurons, not the newly trainable interneurons.

All372staged tests and Ruff pass. The full-test wall time14848.61s includes long
host/tool/environment delays, not a speed benchmark.34installed targeted tests
pass in7.82s. Actual full-connectome preflight (158262neurons) verifies exact
transfer of73594old edges,11796new target edges unchanged, original-reference
bounds/signs, zero newborn statistics, full-network save/load identity, exact old
motor-noise stream and zero frozen exploration. Imported expanded bank SHA256
b688e73e725e7859c75a3eaa0dc5a10c9c30147615dd3401d4836b76862d2667.

Installed only in the isolated worktree, source SHA256
cf8515e2ad10da6a49fe3b87070ce62f1cdde41220482de59779af50af72d7c7.
Only new learning_targets.py/circuit_learning.py and optional plumbing in
learning_lab.py/train_validate.py differ. Body, sensory encodings, reward,
neural equations, depression0 and physical acceptance gates are unchanged.
The originalcd369source is preserved under the prototype's base-src/noesis.

The registered pilot uses serial persistent-weight train_validate births92..97,
quiet5s, sustained-fall termination.1s, rate.2, eligibility100ms, exploration
<=1.5mV/ms, original +/-50%bounds, background3Hz*40 and v5grounded reward with
normalized bias-corrected credit. This returns from the rejected independent
mean pilot to serial learning; it is not a causal experiment isolating scope.
Each wall cycle<=3600s supports exact resume. Validation10/11 and separate
heldout200/201/202 stay full5s with no exploration/fall cutoff. All25quiet,
obstacle and external-sense-ablation cases are required for promotion; early
quiet rejection leaves remaining controls explicitly uncompleted. Proprioception
and balance remain in external-sense ablation. No training video is requested;
a passing model still needs real frozen video and normal-startup verification.
No model is promoted and the normal startup default remains unchanged.

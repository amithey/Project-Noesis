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


The actual worker83373 (launcher25508/interpreter14992) began92, learningTrue,
observed through.2s. Its immutable target-map file, written before the episode
header and successful subset import, exactly matches the full-connectome
preflight64CNSidentities and389mapped leg motor neurons. No live ledger opened,
completed V28birth, frozen result or promotion yet. The active source/plan and
physical gates remain fixed during the batch.


## First runtime boundary and committed circuit updates

Initial83373terminated before a committed birth, leaving only target/sensory
maps. Outer shell exit1did not preserve native Python status, so the exact cause
is unresolved. No final trajectory/checkpoint exists and no walking result is
inferred. The archived audit explicitly withdraws an initial default-sandbox
process-absence claim: that query cannot see the known live elevated worker.
Subsequent elevated CIM confirms only resumed launcher15420/interpreter43696,
with no old worker. Exact --resume session32711 restarts92 from CE1E, unchanged
CF8515source/parameters, and records stdout plus explicit native exit JSON.

The original posthoc checker imported simulation dependencies unnecessarily;
only that checker was stopped, leaving the body worker active. The file-only
checker then passes actual saved cases without importing the simulation path.
Its edge basis separates11796CNSinput edges from73594motor-readout input edges.

Committed92 ends in a fall2.770s/544awake, standing90.992647% in the short
window, slip16.779237mm/s, reward.158356083.7667CNSinput synapses and34744motor
input synapses change, reaching all64selected CNS cells; max original deviation
23.044675%. Committed93 falls3.380s/666awake, standing72.672673%, slip16.452826,
reward−.052895283;8133CNSand36011motor input edges change, all64CNSaffected,
max original deviation24.016786%. Both pass original-reference/topology/sign
and bound checks, body/support metrics, scalar reward and bias-corrected credit
reconstruction (max scalar error2.22e-16), exact sustained-fall termination and
raw artifact hash copies. Actual start-pointer ledger audit awaits worker stop.

These are two retained failures totaling6.150completed body seconds. Circuit
weight changes demonstrate active new-scope learning, not useful learned gait.
No full V28training success, frozen outcome, validated checkpoint or startup
change exists yet. The same worker32711 is training94, observed through1.6s.


## Third birth: full horizon, stance slip still high

Seed94 completed5s/990awake with standing97.777778%, slip17.384094mm/s, reward
.226328355, heading advance6.848899mm and cycles13/17/13/15/13/1. RH has only
one support/lift cycle. Exploration remains on; this is not frozen walking.
The immutable-case audit passes v5scalar/corrected credit (max error2.22e-16),
ground-support counts, body/CSV measures, no sustained fall, original references,
topology/signs/bounds (max deviation24.599393%).8070CNSinput edges and35610motor
input edges changed, reaching all64selected CNS cells. Raw checkpoint, CSV,
motor/gait and sensory artifacts are copied and hash-verified to primary.
Three complete births total11.150body seconds, one full and two falls. Latest
training is retained separately from best validatednull; actual start-pointer
ledger audit waits for termination. Worker32711 has begun95, observed through.4s
with learningTrue. No source, reward, physical-gate, startup or promotion change.


## Fourth birth and directional-credit coverage

Seed95 ended in a fall2.700s/530awake, standing88.113208% in the short window,
slip14.989422mm/s, reward.164612931 and cycles10/6/7/7/7/3. The immutable-case
audit passes scalar/corrected learning moments, support/body metrics, exact
first sustained-fall termination, original references/topology/signs/bounds
(max deviation24.368958%).7975CNSinput and35162motor input edges changed, all
64CNSaffected. All raw artifacts copied with hashes. Four completed births
total13.850body seconds, one full and three falls; no frozen or promoted model.
Worker32711 has started96, observed through.8s with learningTrue.

A separate hash-verified coverage diagnostic on completed92/93/94 finds positive
progress credit in14/544,7/666and21/990awake frames (1.05–2.57%). Among valid
posture, positive heading speed<=30mm/s and>=3support frames,140/154,145/152and
214/235have zero grip credit (90.91–95.39%). This exposes a sparse directional
reward signal; it does not by itself prove the cause of failed walking. Different
seeds/lengths are not an improvement comparison. The active source, reward and
physical criteria remain unchanged through the registered trial.

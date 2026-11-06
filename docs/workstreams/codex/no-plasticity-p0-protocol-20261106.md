# P0: unchanged-weight, same-exploration physical controls

Prospective2026-11-06, issue18. Base4950ec75690396fda176504165ae8ec2fa382b38,
branch codex/no-plasticity-p0-20261106. Codex owns new p0_control.py,p0_worker.py,
run_no_plasticity_p0.py,test_p0_control.py and these docs; ownership recorded in
issue18 comment6688191439. Claude retains learner/reward ownership. No peer
module, native20/core40 source, S2worker/contract or adhesion helper modified.

S2 completed real weight changes but failed retained frozen standing under both
pad policies. Training behavior contains both exploration and plasticity. P0
isolates active learning versus no-plasticity exposure under the SAME exploration;
it does not itself establish retained walking or solve training-seed variability.

Fixed common parent SHA256:
9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c.
Core42c20769376bda5694b4c1c66ca75e52cbd9f1f4,
native runtime e0cbd109bb6e018bd253c9f1b5e1698e0421b276.
All jobs25s/5000chunks, seed1004, standing-v7, native exploration1.5mV/ms for
4990rows after50ms birth. Same neural/body inputs, passive mechanics and pad
threshold. No new/final seeds, external gait timing, reward edits or model selection.

Sequence, maximum3 jobs:

1. Active legacy25s bridge delegates unchanged S2worker. Require byte-exact saved
   checkpoint and every training/body/pad/exploration array equal to S2 before controls.
2. Legacy no-plasticity25s control.
3. Corrected no-plasticity25s control.

Control subclasses the existing LearningBrain only in the isolated worker process.
Inherited step() preserves neural firing, sensory feedback, exploration, muscles
and pad mapping. Network weights become read-only after parent loading. The learn()
entry logs the same innovation credit but NEVER calls learner.update or changes
eligibility/reward moments. Those traces have no other route to neural activity.
Keep updates0; verify4990control calls and whole-network unchanged-weight digest.
The control does not implement a competing learning rule. Effective settings and
partial context are labeled before execution. Native temporary checkpoint output
is renamed weights-control.npz; no completed learning birth/training-last or
validated-best pointer. Final report/sidecar explicitly say no-plasticity exposure.

Require full5000row finite outputs, corresponding complete exploration stream
equality with S2, and exact first11body/pad rows before learning can affect later
dynamics. Audit all selected checkpoint weights unchanged, original references,
topology, flags/rule/parameters and complete-network weight identity. No partial
exposures interpreted; no fresh frozen probes needed for unchanged weights.

Report active-minus-control mean standing, reward and stance slip over all4990
post-birth rows and the fixed final5s. These are descriptive within-training
effects of plasticity in ONE seed/policy pair. No significance threshold, candidate
acceptance or walking/absence-of-learning claim. Existing S2 frozen results remain
the retained-ability evidence; do not directly equate different horizons/exploration.

Budget2100s aggregate from lease,600s each worker, one canonical atomic host lease,
Windows Job Object, fresh >=2GiB RAM and no competing Python before each launch.
Require full worker budget remaining after hash/resource preflight. Stop on any
technical/hash/horizon/stream/prefix/context/invariant/resource/timeout failure;
no retry, retune, extension or partial analysis. Preserve source/parent/data/S2
artifacts and verify cleanup before releasing owned lease.

Checks17pass,0.12s; worker/runner compile. Publish source and this protocol BEFORE
allocation. Run prepared layout: work/noesis-env/Scripts/python.exe
work/run_no_plasticity_p0.py. Keep compact result/protocol in Git, selected artifacts
with CRC/per-file/archive SHA256; Release upload pending until actually successful.
Full project goal remains unachieved. No automatic merge/model promotion.

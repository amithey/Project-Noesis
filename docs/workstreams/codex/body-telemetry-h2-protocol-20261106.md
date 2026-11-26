# H2: read-only whole-body/ground-reaction telemetry on exact frozen replays

Prospective2026-11-06. Issue18 claim6690262486; Codex NEW h2_telemetry.py,
h2_worker.py,run_body_h2.py,test_h2_telemetry.py and H2docs on
codex/body-telemetry-h2-20261106, base48a3b01e6d95757545c94ac50e5efa053711807c.
Native20/core40, S2worker/helper/learner/reward/defaults/peer worktrees unchanged.
No training, learning updates, new policy/threshold, weight saves/merging/promotion.

H1confirmed supported upright samples beyond1.5mm, without enough evidence to
classify their balance/overextension. This diagnostic records missing physical
variables. It does not assume the height gate is wrong or score snapshots as success.

Fixed3full5s cases, legacy pad mapping:
1. Common parent9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c,
   seed10007, exact baseline A1legacy-10007.
2. SR1child1005 SHA46a6975dc7a1479f794034258abae67ff178d31772ac961d54617519a2b00853,
   seed10007, exact SR1child-1005-10007 baseline (supported high posture).
3. S2child1004 SHA07aa4f8198c20dae779d1a836fffb4bf6ddef76f3d62c0af55a050a2b2293e43,
   seed10006, exact S2child-legacy-10006 baseline (sustained upright/support failure).
These are selected known failure TYPES, not a representative sample, model search
or independent validation. No final seeds10,11,200,201,202.

Instrument the existing body class AFTER its normal observe()/mj_forward. Delegate
the unchanged S2frozen worker; no extra physics step, neural drive or random draw.
Record1001samples(initial+1000steps): full qpos/qvel including passive tarsi,
qacc/control/actuator generalized forces, body origins/inertial COM positions and
quaternions, mass-weighted whole-fly COM and subtree-COM-Jacobian*qvel world velocity,
signed world ground-reaction vectors, root translation constraint/actuator/applied
forces and all active fly-ground contact positions/normals/frames/wrenches/distances.
Separate ragged contacts by sample index; no object/pickle arrays. Joint/actuator
names/addresses, body IDs/masses and gravity provide interpretation metadata.

Mass checks: fly-body lowest-common-ancestor subtree must cover ALLpositive model
mass; one free root; weighted xiposCOM matches native subtreeCOM. Include zero-mass
intermediate bodies in the subtree rather than silently dropping massive links.
Contact signs: MuJoCo wrench is on geom2 in contact frame, transform by frame.T;
negate for fly geom1. Verify actual claw forces against unmodified upstream getter,
and whole-fly ground force against free-root qfrc_constraint translation. Record
root qfrc_actuator separately because adhesion applies external force; raw contact
force/mg alone can overstate effective support. Applied forces must be interpreted
from recorded arrays; joint servo actuation is internal to the whole fly.

Read-only checks: qpos/qvel/qacc/ctrl/sensordata/time identical before/after telemetry.
Before ANY scientific interpretation require every native body/pad array EXACT to
prior, unchanged total spike count, matched frozen source/body/packages/context and
checkpoint identity. Check full5s horizon1000steps,1001finite telemetry samples,
timestamps, mass closure and force closure. Whole ground-force closure tolerance:
maxabs error <=1e-6*weight+1e-10; claw upstream error <=1e-10.

Light measurement checks(<=180s, no neural/integration steps) passed4tests0.14s:
real static MuJoCo sphere ground-force sign/root closure/adhesion separation,
mass-offset rotating COM velocity, whole-mass coverage, readonly/time0. These
fixtures test measurement, not a positive organism/walking control. Worker/runner compile.

Budget1200s aggregate,300s each worker,3full5s jobs maximum. Canonical atomic host
lease/Windows Job Object/fresh>=2GiB RAM/no competing Python before each job. Source,
licensed data, parent/children and signed baseline hashes guarded before/finally.
Support both documented root-relative and outputs-relative older archive manifests,
checking exact signed per-file entries. Require full300s remaining after preflight.
Stop technical/hash/parity/horizon/context/mass/force/resource/timeout failure;
no retry/retune/partial interpretation or threshold relaxation. Preserve receipts
and release only owned lease after verified empty tree. Publish source/protocol
before allocation. Run: work/noesis-env/Scripts/python.exe work/run_body_h2.py.

Descriptive analysis: original high-only/standing/failure categories, whole-COM
motion, actual ground contact height/count, ground+root-actuator force/weight,
and contact distributions. Geometric COM/support relations with adhesion are not
certified balance or biological posture. No unmeasured muscle force inference,
no changed reward/acceptance gate, walking or retention claim. Compact results in
Git; signed selected artifacts locally, Release pending authenticated upload.
Full retained standing/balance/walking/environment-response goal active/unachieved.

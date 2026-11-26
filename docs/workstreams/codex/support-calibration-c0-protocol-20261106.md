# C0: anatomical-support descriptor calibration, not a standing gate

Prospective2026-11-06. Issue18 claim6691078415; Codex NEW support_descriptor_c0.py,
run_support_c0.py,test_support_c0.py and C0docs, branch
codex/support-calibration-c0-20261106, base0d1fd73b772548ef9957e79f03731c0b2ab7babe.
No existing evaluation/core/learner/reward/body/controller/default/peer edits.
Gemini independent physical interpretation and Claude reward review remain relevant.
Historical Nov1pause is superseded by active user continuation; no automatic merge.

H2Bmeasured central-body ground reaction inside some legacy-standing samples,
while child1005high-only samples had claw-only reaction. C0calibrates force-bearing
anatomical description, not equilibrium or learned standing/walking acceptance.
Keep ALLhistorical masks/verdicts. No retrospective threshold fitting or promotion.

Descriptor inputs: signed world-frame ground-force vectors per UNIQUEcompiled
body and explicit body names, flat horizontal plane only. Partition positive
vertical reaction into central body(including compiled thorax/fused head),
tarsus5claw, other tarsal1..4 and proximal leg(coxa/trochanter/femur/tibia).
Reject unknown LOADEDanatomy, nonfinite data, invalid shapes or significant
negative vertical force. Existing model-force threshold1e-5is explicit; roundoff
negatives no lower than-1e-9may be zeroed, with partition error checked. Do not
count multiple tarsal bodies as independent legs or infer biological muscle forces.

Contact modes: unsupported; central_body_contact(including mixed body/feet);
proximal_leg_contact; tarsal_only; indeterminate_low_force. Loaded-region amounts/
shares are always retained, so priority labels do not conceal mixed support.
No physical_standing_validated/dynamic_balance_validated flags can become true.
Height, uprightness, COMmotion and old standing mask are separate observations.

Six role-labelled STATICMuJoCo measurement fixtures with known contact geometry:
airborne, foot-only, body-only, mixed body+feet, proximal-leg, other-tarsal. Use
mj_forward, no integration or neural execution, verify time0/unchanged state/root
force closure. Sphere-role fixtures are NOT a fly, biological calibration or a
positive learned-standing control. Eight tests passed0.15s including all6fixtures,
conservative force partition/scale behavior and rejecting invalid anatomy/forces.

Apply the IDENTICALdescriptor to all1000samples of EACHof the3signed H2Bcases;
report after50ms warmup alongside original standing/high-only/upright-failure
masks and COMmotion. Require exact original standing score and central-body
contact counts against H2Bsummary, reaction conservation, all expected labels,
unchanged input/source hashes. Store per-frame codebook/region reactions, report
allcases without model selection. No new neural/physics steps or weights loaded.

Input archive SHA256f660b673da8d431873cd829a364d423633b01fe395ce971ecb6f3190984e3257
and selected files checked against manifest. Guard unchanged H2source and original
H1mask source. Publish code/protocol before analysis. Run prepared layout:
work/noesis-env/Scripts/python.exe work/run_support_c0.py.

Budget180s contained STATIC/OFFLINEworker under canonical lease, fresh>=2GiB RAM,
no competing Python. Stop technical/hash/nonfinite/shape/force/fixture-label/
partition/original-mask/count mismatch or timeout; no automatic retry/retune.
Preserve receipt/empty-tree cleanup and release only owned lease. No heavy training
allocation, new standing gate/reward/body model or promotion. Compact code/results
in Git; selected signed artifact locally, Release upload pending authenticated route.
Full retained standing/balance/walking/environment-response goal active/unachieved.

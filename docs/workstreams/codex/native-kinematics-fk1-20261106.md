# FK1 native geometry audit — registered2026-11-06

Question: does the positive Femur displacement used by the existing pad-release condition lift the observed tarsus5 body origin, and how do the encoded muscle signs relate to native geometry? Core42c2076 maps trochanter_flexor to negativeFemur and trochanter_extensor to positiveFemur; bridgee0cbd10 releases for positive desiredFemur>.08rad while its comment says flexor. That is a convention question, not yet a proven physical bug.

Codex owns new audit/wrapper/docs on codex/native-kinematics-fk1-20261106 fromeeea7d4. Native pilot/core/weights/controller remain unchanged. No peer branch edits, connectome load, learning, policy change, dynamics or walking claim. The earlier ES1 trial remains rejected.

## Before measurements

Instantiate the exact native CPU model and only call mj_forward. All42actuated hinge channels map through the existing native read order. Measure foot height at the same xpos tarsus5 body origin used by observations, in mm. Model introspection confirms these segment names and simulation clock0 without stepping.

Configurations: native neutral plus fixed frames10/250/500/750/999 from each ES1parent10006/10007/10010/10011 body trace, whose SHA must match ES1receipts.21active-joint configurations. Root/passive joints remain at native neutral: these are NOT recorded-state replays. No final physical10/11/200/201/202 data.

For all42channels and all6feet: central differences at+/-.001rad and+/-.0005rad, compare with mj_jacBody's matching vertical/dof columns. Require max absolute discrepancy and step-halving discrepancy<1e-5mm/rad; finite measurements, other-leg derivatives<1e-10, and rigid-translation invariance<1e-10mm. Flag derivative samples crossing native joint limits; these are infinitesimal mathematical samples, no constrained dynamic motion.

Primary observations by leg: positiveFemur own-foot height derivative at neutral and across all21configurations (raise>1e-4,lower<-1e-4,otherwise nearzero). Also record finite+/-.1rad neutralFemur own-foot height change and the actual unchanged adhesion output. Supplement: apparent coxa/femur and femur/tibia/tarsus1 interior-angle derivatives, with no biological torque interpretation. Verify elapsed model time remains0 and no neural execution.

No signal/pad sign flip or retuned diagnostic after results. Simultaneous multi-joint/body motion and actuator lag can alter actual foot velocity; isolated partial derivatives do not establish a cause of failed locomotion or validate biological muscle identity. Any resulting repair proposal requires separate source review and a one-factor dynamic comparison with a newly matched baseline.

## Budget and evidence

120s hard Windows Job Object worker budget plus bounded cleanup; primary canonical lease, fresh RAM>=2GiB/no competing Python. Start suspended/assign job before resume, verify all descendants empty before lease release, retain lease on uncertain cleanup. Output directory new/exclusive. No actual physical episode or generation is allocated by this geometry test.

Compile check passed. Mathematical crosschecks execute with the audit, not synthetic gait tests. Code/protocol/result compact JSON go to Git; signed numerical artifact excludes raw graph/cache/private lease/assets. No parent checkpoint is imported or written; its SHA is checked unchanged. Release upload remains pending auth.

Prepared layout: restore scripts/audit_native_kinematics_fk1.py and scripts/run_fk1_audit.py to ROOT/work; retain exact ROOT/work/Noesis-Sensory runtimee0cbd10, core42c2076, environment and completed ES1outputs. Execute its Python on ROOT/work/run_fk1_audit.py once. Publish final source SHA/fresh allocation before running.

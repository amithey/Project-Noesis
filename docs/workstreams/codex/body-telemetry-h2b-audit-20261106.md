# H2B: instrumented exact frozen replays

Three full5s jobs in178.844s. All native body/pad/spike replays exact. Whole mass/COM, root-force/upstream/raw-contact checks pass; source/prior/data unchanged, exit0/empty and lease released. A0zero-step failure remains archived.

|Case|Group|Samples|Median COM abs vz mm/s|Median contact Fz/weight|Median actuator Fz/weight|Median sum/weight|COM inside claw polygon %|
|---|---|---:|---:|---:|---:|---:|---:|
|parent-10007|standing|935|0.89792|1.12347|-0.09952|1.04874|55.37|
|parent-10007|high_only|39|0.57931|1.00588|-0.09952|0.94531|0.00|
|parent-10007|upright_failure|0|n/a|n/a|n/a|n/a|n/a|
|child1005-10007|standing|868|1.17864|1.13843|-0.09952|1.05144|64.48|
|child1005-10007|high_only|110|2.08224|1.15807|-0.09952|0.99154|98.67|
|child1005-10007|upright_failure|0|n/a|n/a|n/a|n/a|n/a|
|child1004-10006|standing|405|1.02261|1.13432|-0.09952|1.04735|71.68|
|child1004-10006|high_only|40|0.76414|1.00804|0.00000|1.00454|0.00|
|child1004-10006|upright_failure|544|0.08414|0.99962|0.00000|0.99904|50.00|

Polygon percentage is conditional on a defined noncollinear loaded-claw contact hull, not a balance gate. Median sum is computed per sample, not by adding separate medians.

Limits:

- Three known failure types selected for diagnosis, not representative validation or checkpoint selection.
- 5ms samples and geometric COM projection do not certify continuous/static balance; adhesion can support moments outside a claw-contact polygon.
- Contact+root-actuator force is a recorded contribution, not a fully audited Newton residual including every possible passive/aerodynamic term.
- Configured +/-0.7rad active ranges are pilot assumptions, not biologically validated extension limits.
- No threshold/reward/body/model changes, promotion, learned walking or retained-standing success claim.

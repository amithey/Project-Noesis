# C0: anatomical support description

Six static fixtures and all3H2Btraces checked in0.782s worker time; eight unit checks passed. Source/input/force/partition/original-mask/count checks pass; exit0/empty, lease released. No neural/physics integration, standing gate or reward change.

|Case|Group|Samples|Unsupported|Central-body contact|Proximal-leg contact|Tarsal-only|
|---|---|---:|---:|---:|---:|---:|
|parent-10007|all_measured|990|3|541|51|395|
|parent-10007|standing|935|0|505|47|383|
|parent-10007|high_only|39|0|28|0|11|
|child1005-10007|all_measured|990|1|367|65|557|
|child1005-10007|standing|868|0|358|63|447|
|child1005-10007|high_only|110|0|0|0|110|
|child1004-10006|all_measured|990|1|769|10|210|
|child1004-10006|standing|405|0|214|7|184|
|child1004-10006|high_only|40|0|37|0|3|

Contact modes are NOT standing/balance/learning validation. Tarsal-only inside old standing exists(383parent/447child1005samples); these are sampled states, not stable independent episodes. Alloldverdicts retained.

Limits:

- Contact mode describes force-bearing anatomy, not equilibrium, gait, continuous support or retained learning.
- Flat horizontal ground/world vertical force convention only; loaded unknown anatomy or negative vertical force rejected.
- Other tarsal segments are separately identified; multiple loaded body segments are not counts of independent legs.
- Central compiled thorax includes fused head; no anatomically separate head-force claim.
- Three H2Bdiagnostic cases are not representative validation or policy selection.

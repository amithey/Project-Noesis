# S2: paired standing reacquisition

Twelve full jobs completed in 1138.813s. Two exact parent bridges, two25s/4990update births, eight5s frozen probes. All technical gates passed; unchanged source/parent/data; all contained workers empty and lease released.

|Mapping|Seed|Parent standing %|Child standing %|Standing change pp|Slip gain mm/s|Heading gain mm|Child regime|
|---|---|---:|---:|---:|---:|---:|---|
|legacy|10006|96.2626|40.9091|-55.3535|-1.32791|-0.29606|FALL|
|legacy|10007|94.4444|94.1414|-0.3030|-0.23955|1.21476|STEPPING_UNCOORDINATED|
|legacy|10010|89.2929|96.7677|7.4747|0.44207|-1.00078|STEPPING_IN_PLACE|
|legacy|10011|91.9192|93.9394|2.0202|-0.32093|-0.05135|STEPPING_IN_PLACE|
|corrected|10006|92.6263|90.3030|-2.3232|-0.31251|-0.57742|STEPPING_IN_PLACE|
|corrected|10007|97.0707|87.7778|-9.2929|-0.05483|0.14103|STEPPING_IN_PLACE|
|corrected|10010|94.3434|86.1616|-8.1818|-0.21034|-0.98724|STEPPING_IN_PLACE|
|corrected|10011|89.0909|82.5253|-6.5657|0.49018|-0.47051|STEPPING_IN_PLACE|

Registered standing-stage screens:

```json
{
  "legacy": {
    "passed": false,
    "standing_retained": false,
    "slip_retained": false,
    "median_paired_gains": {
      "heading_forward_distance_mm": -0.173707954847333,
      "net_forward_displacement_mm": -0.8184306178682822,
      "total_leg_steps": -2.0
    },
    "median_tripod_gain": -0.0873007599144657,
    "standing_stage_only": true,
    "walking_proven": false,
    "model_promoted": false
  },
  "corrected": {
    "passed": false,
    "standing_retained": false,
    "slip_retained": false,
    "median_paired_gains": {
      "heading_forward_distance_mm": -0.5239649019687345,
      "net_forward_displacement_mm": -0.4339820388865674,
      "total_leg_steps": -9.0
    },
    "median_tripod_gain": -0.028944712795941602,
    "standing_stage_only": true,
    "walking_proven": false,
    "model_promoted": false
  }
}
```

Limits:

- One paired training birth, not four independent training replications.
- Four reused development seeds and five-second frozen horizon; final validation unused.
- Standing-stage screen is not walking, perturbation recovery, environmental response or lifetime retention.
- Pad mapping uses neural desired angles; biological adhesion-force control remains unvalidated.
- No automatic model promotion, weight merging, learner/reward change or duration/rate retune.
- A failed screen does not prove absence of learning or identify the cause of every earlier failure.

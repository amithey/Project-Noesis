# X1: fixed-weight current withdrawal

Nine full5s jobs complete in770.296s. All bridges/latent drive/seeds/pad formulas/source/data/weight checks pass; exit0, empty trees, lease released.

|Policy|Bank|Current|Standing %|Slip mm/s|Heading mm|Net-forward mm|Cycles|Regime|
|---|---|---|---:|---:|---:|---:|---:|---|
|legacy|parent|on|91.2121|2.44427|-0.35058|-0.17902|69|STEPPING_IN_PLACE|
|corrected|parent|on|85.2525|2.22626|0.25818|0.40074|59|STEPPING_IN_PLACE|
|legacy|parent|off|89.1919|2.94470|0.20373|-0.05538|46|STEPPING_IN_PLACE|
|corrected|parent|off|97.6768|2.23359|0.09855|-0.12143|48|STEPPING_IN_PLACE|
|legacy|child|on|92.2222|2.23860|0.71720|1.04088|63|STEPPING_IN_PLACE|
|legacy|child|off|98.8889|2.65277|0.88427|0.77841|58|STEPPING_IN_PLACE|
|corrected|child|on|89.6970|2.43282|1.02685|1.11518|55|STEPPING_UNCOORDINATED|
|corrected|child|off|89.4949|2.04744|0.74740|0.77033|54|STEPPING_IN_PLACE|

Descriptive paired interactions:

```json
{
  "legacy": {
    "child_minus_parent_on": {
      "standing_fraction": 0.010101010101010166,
      "mean_stance_slip_mm_s": -0.20566454075730523,
      "heading_forward_distance_mm": 1.0677769253331117,
      "net_forward_displacement_mm": 1.2198986596693717,
      "total_leg_steps": -6
    },
    "child_minus_parent_off": {
      "standing_fraction": 0.09696969696969704,
      "mean_stance_slip_mm_s": -0.29193240053951586,
      "heading_forward_distance_mm": 0.6805415772758718,
      "net_forward_displacement_mm": 0.8337857511987468,
      "total_leg_steps": 12
    },
    "difference_of_differences_on_minus_off": {
      "standing_fraction": -0.08686868686868687,
      "mean_stance_slip_mm_s": 0.08626785978221063,
      "heading_forward_distance_mm": 0.3872353480572399,
      "net_forward_displacement_mm": 0.38611290847062496,
      "total_leg_steps": -18
    },
    "one_seed_only": true,
    "model_promoted": false,
    "walking_proven": false
  },
  "corrected": {
    "child_minus_parent_on": {
      "standing_fraction": 0.0444444444444444,
      "mean_stance_slip_mm_s": 0.20656235837441583,
      "heading_forward_distance_mm": 0.7686676479850292,
      "net_forward_displacement_mm": 0.714440842755614,
      "total_leg_steps": -4
    },
    "child_minus_parent_off": {
      "standing_fraction": -0.08181818181818179,
      "mean_stance_slip_mm_s": -0.18615607609902396,
      "heading_forward_distance_mm": 0.6488469063284155,
      "net_forward_displacement_mm": 0.8917597376732955,
      "total_leg_steps": 6
    },
    "difference_of_differences_on_minus_off": {
      "standing_fraction": 0.1262626262626262,
      "mean_stance_slip_mm_s": 0.3927184344734398,
      "heading_forward_distance_mm": 0.11982074165661372,
      "net_forward_displacement_mm": -0.1773188949176815,
      "total_leg_steps": -10
    },
    "one_seed_only": true,
    "model_promoted": false,
    "walking_proven": false
  }
}
```

Limits:

- One reused development seed1004 and one S2training birth per mapping; no training-seed replication.
- Current-on is fixed-weight exposure, not an ordinary frozen/no-exploration validation.
- Interaction is descriptive; neither current benefit nor difference of differences proves learned walking or generalization.
- S2retention screens remain failed; no model promoted, weight merge, new training or rule change.
- All inputs/core/body matched within a mapping except the declared weight/current interventions.
- First divergence chunks are posthoc transport descriptions, not a fitted learning-credit delay.

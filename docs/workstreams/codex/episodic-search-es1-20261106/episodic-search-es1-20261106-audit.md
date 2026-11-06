# ES1 episodic synaptic search: observed results

Status:completed_no_candidate; 12full5s jobs, 710.609s. All four historical parent replays exact. Original-reference bounds/signs/topology/nonplastic weights preserved; all workers exit0/trees empty, source/parent unchanged, lease released.

One direction60061,plus/minus2%original magnitude. No within-episode plasticity, external gait, active senses or model promotion.

|Seed|Bank|Standing %|Slip mm/s|Heading mm|Net-forward mm|Regime|
|---|---|---:|---:|---:|---:|---|
|10006|parent|96.2626|3.05817|0.29836|0.02235|STEPPING_IN_PLACE|
|10006|plus|90.7071|2.74753|-0.87726|-1.34026|STEPPING_IN_PLACE|
|10006|minus|90.5051|2.67197|1.24659|1.10295|STEPPING_UNCOORDINATED|
|10007|parent|94.4444|2.49016|0.33179|0.38400|STEPPING_IN_PLACE|
|10007|plus|89.8990|2.57371|-0.13875|-0.12180|STEPPING_IN_PLACE|
|10007|minus|93.5354|2.17402|0.06310|0.43746|STEPPING_IN_PLACE|
|10010|parent|89.2929|2.24998|1.20188|1.46832|STEPPING_UNCOORDINATED|
|10010|plus|94.8485|2.45195|0.69688|0.50659|STEPPING_IN_PLACE|
|10010|minus|90.8081|2.45294|2.43673|2.11756|STEPPING_UNCOORDINATED|
|10011|parent|91.9192|2.93275|-0.05009|-0.04533|STEPPING_IN_PLACE|
|10011|plus|83.4343|2.63696|-0.30149|-0.21815|STEPPING_IN_PLACE|
|10011|minus|97.4747|2.15944|0.39239|0.47509|STEPPING_IN_PLACE|

Development gates:

```json
{
  "plus": {
    "passed": false,
    "standing_retained": false,
    "slip_retained": false,
    "cycles_retained": false,
    "tripod_retained": true,
    "positive_forward_progress": false,
    "median_heading_gain_mm": -0.4877723403497909,
    "median_net_forward_gain_mm": -0.7337675679564293,
    "ranking": -0.7337675679564293,
    "seeds": [
      10006,
      10007,
      10010,
      10011
    ],
    "walking_proven": false,
    "model_promoted": false
  },
  "minus": {
    "passed": false,
    "standing_retained": false,
    "slip_retained": false,
    "cycles_retained": true,
    "tripod_retained": false,
    "positive_forward_progress": true,
    "median_heading_gain_mm": 0.6953513591764493,
    "median_net_forward_gain_mm": 0.584830878994205,
    "ranking": 0.584830878994205,
    "seeds": [
      10006,
      10007,
      10010,
      10011
    ],
    "walking_proven": false,
    "model_promoted": false
  }
}
```

Limits:

- ONE parameter direction/antithetic pair; not an optimizer-efficacy estimate.
- Reused development probes select; conditional fresh development probes are not reserved final validation.
- No within-episode learning or new gait/sensory policy; proposals are not completed native training births.
- Strict screening does not establish stable walking/balance/environmental response or retained learning across generations.

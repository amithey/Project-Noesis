# TP2: RNG-matched constant thermo amplitude

Six full5s jobs completed in382.813s. All1000post-chunk full MT states and drive indices/rates match within each pair. Both plain probes replay B2, and both stimulus probes replay TP1 constant input exactly. All workers exit0 with entire Windows jobs empty; source/parent unchanged, lease released.

Registered neural screen:True; strict physical screen:False. No learning, checkpoint, model promotion or stable-walking claim.

|Seed|Mode|Warm Hz/cell|Standing %|Slip mm/s|Heading mm|Regime|
|---|---|---:|---:|---:|---:|---|
|10006|plain|13.2660|96.2626|3.05817|0.29836|STEPPING_IN_PLACE|
|10006|sham|13.6700|96.1616|2.99959|-0.06439|STEPPING_IN_PLACE|
|10006|stimulus|35.0505|88.6869|2.95833|0.27390|STEPPING_IN_PLACE|
|10007|plain|8.1818|94.4444|2.49016|0.33179|STEPPING_IN_PLACE|
|10007|sham|12.5589|87.3737|2.72462|0.52181|STEPPING_IN_PLACE|
|10007|stimulus|36.8350|89.2929|3.01029|0.76165|STEPPING_IN_PLACE|

Stimulus minus matched sham:

|Seed|Warm gain Hz/cell|Standing change pp|Slip change %|Neural pass|Physical pass|
|---|---:|---:|---:|---|---|
|10006|21.3805|-7.4747|-1.3755|True|False|
|10007|24.2761|1.9192|10.4848|True|False|

Interpretation limits:

- Two reused development seeds, frozen5s probes; no independent final validation or retained-learning evidence.
- Constant analytic warm input, not changing environmental temperature or thermotaxis.
- Sham retains zero-weight event/active-set processing; it is not a no-input deployment baseline.
- This amplitude contrast controls the identified RNG nuisance, not every possible biological modelling error.
- Strict physical screen is descriptive; no promotion, default activation or gain retuning.

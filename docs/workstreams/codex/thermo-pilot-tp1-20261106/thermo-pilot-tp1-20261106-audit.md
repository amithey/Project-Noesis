# TP1 frozen thermosensory integration: observed results

All ten full 5s/1000-chunk jobs completed in 612.062s. Plain exactly replayed B2 on both development seeds; off and zero exactly matched plain in every native and observer array. Source/parent unchanged; owned lease released.

Registered neural screen: True. Strict standing/slip retention screen: False. No learning, checkpoint, promotion, default activation or walking claim.

|Seed|Mode|Warm spikes Hz/cell|Requested warm Hz|Standing %|Slip mm/s|Heading mm|Regime|
|---|---|---:|---:|---:|---:|---:|---|
|10006|plain|13.2660|0.0000|96.2626|3.05817|0.29836|STEPPING_IN_PLACE|
|10006|off|13.2660|0.0000|96.2626|3.05817|0.29836|STEPPING_IN_PLACE|
|10006|zero|13.2660|0.0000|96.2626|3.05817|0.29836|STEPPING_IN_PLACE|
|10006|constant|35.0505|25.0000|88.6869|2.95833|0.27390|STEPPING_IN_PLACE|
|10006|active|47.5758|38.4935|66.6667|3.04403|0.28923|FALL|
|10007|plain|8.1818|0.0000|94.4444|2.49016|0.33179|STEPPING_IN_PLACE|
|10007|off|8.1818|0.0000|94.4444|2.49016|0.33179|STEPPING_IN_PLACE|
|10007|zero|8.1818|0.0000|94.4444|2.49016|0.33179|STEPPING_IN_PLACE|
|10007|constant|36.8350|25.0000|89.2929|3.01029|0.76165|STEPPING_IN_PLACE|
|10007|active|39.6296|27.3924|83.3333|3.13799|-0.05984|STEPPING_IN_PLACE|

Stimulus-arm deltas versus matched off:

|Seed|Mode|Warm gain Hz/cell|Standing change pp|Slip change %|Neural pass|Retention pass|
|---|---|---:|---:|---:|---|---|
|10006|constant|21.7845|-7.5758|-3.2647|True|False|
|10006|active|34.3098|-29.5960|-0.4623|True|False|
|10007|constant|28.6532|-5.1515|20.8874|True|False|
|10007|active|31.4478|-11.1111|26.0155|True|False|

Interpretation limits:

- Two reused development seeds; five seconds each, no independent final validation.
- Analytic normalized field; no calibrated thermal physics or learned thermotaxis.
- Constant matches active at x=0 only; differing exposure confounds pattern contrasts.
- Extra drive indices consume random numbers before background draws in the shared core stream. Shared seeds do not hold background realization fixed across stimulus/off.
- Afferent spikes are channel totals over six cells, not per-cell response estimates.
- Frozen descriptive retention screen cannot prove retention across training or stable walking.

Source/protocol: PR30, source137ca05, core42c2076, completed parent9fd7d411. Numerical artifact contains only explicit report/observer/body arrays and public scientific protocol/result; no raw connectome, derived graph cache, weights, private lease, token, PID or environment.

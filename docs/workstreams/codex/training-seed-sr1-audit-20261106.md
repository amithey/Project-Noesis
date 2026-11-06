# SR1: independent training-seed sensitivity

Fourteen new full jobs in1424.953s: two25s/4990update births plus twelve5s probes. All technical/hash/context/checkpoint gates pass; source/prior/data unchanged, exit0/empty, lease released. Existing1004birth included, no bank selection.

|Training seed|Own parent standing %|Own child standing %|Own gain pp|Development mean gain pp|Worst dev gain pp|Dev >=90%|Strict screen|
|---|---:|---:|---:|---:|---:|---:|---|
|1004|89.1919|98.8889|9.6970|-11.5404|-55.3535|3/4|False|
|1005|96.6667|94.9495|-1.7172|-0.3788|-6.7677|3/4|False|
|1006|94.0404|86.4646|-7.5758|-1.5152|-5.4545|3/4|False|

All development standing values:

|Training seed|10006 %|10007 %|10010 %|10011 %|
|---|---:|---:|---:|---:|
|1004|40.9091|94.1414|96.7677|93.9394|
|1005|97.6768|87.6768|94.2424|90.8081|
|1006|91.8182|94.6465|83.8384|95.5556|

Complete screens and metrics in audit JSON.

Limits:

- Three training births, not twelve independent training replications.
- Own-seed gains are training-seed checks; shared development seeds reused, final validation unused.
- Three births cannot fully estimate variability or establish overfitting, learning absence or causal mechanism.
- All banks reported, no selection/merge/default change/model promotion.
- Weight-delta cosine is posthoc descriptive, not proof of a correct/incorrect learning gradient.
- Standing/proxy retention screens do not establish balance recovery, stable walking or environmental responses.

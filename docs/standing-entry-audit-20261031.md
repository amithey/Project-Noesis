# Standing entry and reward audit — existing physical data only

No simulation or learning launched by this audit. Same coref52c8d/pilot80556cd,
standing thresholds and banks. [Reproducible script](../scripts/audit_standing_entry.py)
and [compact results/input hashes](standing-entry-audit-20261031.json).

## Which bank actually passed the old standing gate?

The original v7standing birth isfb80cfcc (seed11000,5s/990updates). Its historical
two-development-seed standing gate passed. Current common parent9fd7d411 is a
5s v9gripped-cycle descendant (seed1000) of that bank. Its failure in10010does
not establish that the ANCESTOR fails there. Four-probe ancestral standing under
the current pinned environment remains untested by this audit. Historical two-
seed qualification is not universal standing and is not invalidated retroactively.

## Current descendant: why standing fails

Warmup50ms excluded;990measured samples per5s frozen probe. Counts overlap:

| Seed | Standing % | Height>1.5mm | Upright<.7 | No tarsus support | Longest continuous failure |
|---|---|---|---|---|---|
|10006|96.263|35|4|2|135ms|
|10007|94.444|40|0|16|95ms|
|10010|89.293|60|27|20|150ms|
|10011|91.919|77|17|3|195ms|

No sample fails height<.3mm. The largest cause count is upper-height violation,
often while still upright/supported; that should not be called a ground-collapse
or unsupported fall without the other measurements. Tilt and no-support events
also exist. Thresholds stay unchanged: this diagnosis does not authorize relaxing
the height gate, smoothing measured failures away or counting them as standing.
100ms is a descriptive episode-length cutoff only, not a new success criterion.

## Does the recorded reward silently reward nonstanding?

For T1B and L2,4999consecutive transitions reconstructed EXACTLY (maxerror0),
using native reward code and recorded cycle credit. The initial transition is
excluded because its pre-step body state was not stored. All4990learning-active
rewards are covered. Reward component sums and clipping match the actual log.
Actual normalizer replay also matches the final bank's baseline/variance/mass.

- T1B:726nonstanding learning samples,0positive rewards,0positive normalized
  learning signals on them;702standing samples have negative relative signal.
- L2:751nonstanding samples,0positive rewards/signals on them;623standing
  samples have negative relative signal.

Negative relative signal during standing is ordinary comparison with a learned
baseline, not by itself a bug or a reason to clamp it. No evidence here supports
the proposed failure mode "positive reward for nonstanding" or a silent scorer
implementation mismatch. It also does not prove credit causes effective learning.

v9cycle term means are0.002290/0.002200; standing component means0.225411/0.219399
and std0.423114/0.429074. These are descriptive components, not independent
variance contributions or a calibrated causal decomposition.

Offline rescoring the SAME trajectories withv7 gives mean0.537196/0.528988,
versus actualv9means0.212874/0.212416. Different reward scales do NOT establish
better physical behavior or v7learning. No v7rollout or candidate was generated.

## Next falsifiable check

[B1](standing-baseline-b1-protocol.md): frozen original standing ancestor vs current
descendant on four matched development seeds, with fresh descendant replay controls.
This resolves baseline qualification before another training-method change. No
automatic parent switch, walking claim, stage advancement or model promotion.
29focused tests passed before allocation; native reward/normalizer reconstruction
was executed on the actual completed data. Inputs remain in the existing signed
T1B/L2 evidence releases; no raw connectome is copied to Git.

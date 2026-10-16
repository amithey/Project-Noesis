# Standing-first reward: applied after completed rejection

V11 completed and was rejected after the frozen candidate seed 10 stood for
36.87% versus original 92.22%. Its exact source and all failed artifacts were
preserved before installation. This is an applied training-objective hypothesis,
not a verified behavioral improvement or deployment.

This candidate addresses a concrete reward-ranking concern. A stationary fully
supported upright state with 16 mm/s slip receives -0.71 under the current
objective, while an unsupported motionless fall receives -0.6. The candidate
scores those states +0.654 and -0.6 respectively. This is an artificial training
objective, not a biological claim or evidence of learned walking.

The candidate changes only reward coefficients and its checkpoint context label:
standing quality coefficient 0.25 becomes 0.75; the binary -0.6 fall penalty
becomes -0.6*(1-standing_quality); the linear slip coefficient 0.3 becomes 0.03.
The original progress grip, heading, effort and obstacle terms remain unchanged.
The smaller slip coefficient prioritizes stable support during early training.
There is still a nonzero linear cost for greater slip, and equal-mean 3/7 and
5 mm/s examples keep equal mean reward rather than encouraging slip bursts.
Reward remains bounded to [-1,1]. This priority is a modelling hypothesis, and
a smaller penalty does not justify accepting a physically more slippery fly.

All physical acceptance requirements remain unchanged: >=90% standing, >=1 mm
heading progress, >=2 real stance/lift cycles in >=4 legs, >=20% lower mean stance
slip than matched original controls, obstacle response, ablation controls and
held-out seeds. Learning/exploration remain off in every frozen five-second test.
No joint commands, neural dynamics, connectome edges or sensory channels change.

Fourteen new regression cases include the explicit reward inversion, stationary
quiet-floor ordering for uprightness 0.7/0.8/1.0 and slip 0/16/60/100 mm/s, strict
slip ordering, equal-mean burst costs, and exact equality of all nonreward fields
on 1,000 random physical samples. Existing reward guards also pass. Full staged
regression: 227 passed; focused: 42 passed; Ruff passed.

A read-only replay reconstructs every saved v10/v11 reward from its original
components to 1e-12, then changes only the scalar reward formula. Original upright
floor fractions range from 24.6% to 35.2% across the included completed trials;
the candidate upright floor fraction is zero for each recording. This is a
calibration replay on the same observations, not a candidate behavior experiment.
It cannot prove that the changed objective will improve standing or reduce slip.

Application gate: first complete the current frozen candidate evaluation under
its existing source, resuming unfinished controls after the one-hour limit if
necessary. If it still rejects, preserve all failed artifacts and use this change
in a new output directory/source fingerprint with explicit complete-weight import.
Archive the v11 source before applying. Never waive resume source checks, reuse
old controls under a changed reward or deploy without the full frozen gates.

Portable patch: docs/patches/standing-first-reward.patch. Reproducible tests and
source snapshot: outputs/protocol-v12-posture-candidate; replay and source hashes
are recorded in docs/standing-first-reward-candidate.json.

## Frozen regression reference

The nonreward-field comparison now loads a frozen copy of the original v2 reward
module, carried with the portable patch as tests/support_slip_v2_reference.py.
This test-only fixture is never imported by production training. It keeps the
1,000-sample comparison meaningful after installation, instead of accidentally
comparing the candidate to itself through the current src path. Full regression
was repeated with the frozen fixture: 227 passed, Ruff passed. The candidate
model source and active v11 source fingerprints did not change.

## Installation and next training protocol

The application gate above has been satisfied: V11 is terminal and rejected.
The installed source is byte-identical to the staged source, fingerprint
6f6b31dbe85d7ff21e00cdb5c6fbe10aadc417d8d3d45316fdff9107e88b5ef3.
Installed full regression: 227 passed, Ruff passed.
New training uses unused seeds 12/13/14, disjoint from validation 10/11 and
held-out 200/201/202. Seeds 8/9 are skipped because a contiguous three-seed batch
would overlap validation. Complete V11 training weights are explicitly imported
with reset traces under a new reward context. No frozen controls or acceptance
results are transferred across objectives. The primary default is unchanged.

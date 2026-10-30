# M1 existing anatomical exploration-to-command audit

Codex issue18, starting97808a5. No simulation, trained model, rule or controller
change. Current resource gate failed(10samples,min1.010GiB,maxCPU99.8percent).
User prefers work whenever available; this authorizes opportunistic checks, not
app termination or relaxation of the scientific/resource gates.

Use only complete D3/C2 five-second physical logs,seeds1000/1001/1002, shared
9fd7d411 parent/core42c2076, same LR/reward/exploration and explicit row alignment.
Construct the exact existing anatomical Gpos/Gneg muscle matrices from local
neuron metadata. Project actual logged current and credit through Gpos-Gneg;
these projected inputs are NOT measured firing rates or a muscle simulation.
No anatomical mapping/labels are invented, altered or fitted to results.

Before analysis, fix corresponding-joint Pearson r at25ms(5chunks) as primary
descriptive readout;0..100ms(0..20chunks) secondary. The existing pipeline has a
50ms rate filter and30ms muscle filter;25ms is an inspection point, not a claim
of their peak latency. Compare projected current to actual command levels, and
projected innovation to actual command changes. Use the same central950rows
for every lag; no wrapped samples, cross-joint search or missing-as-zero results.
Constant channels produce undefined values explicitly. Report each joint/seed
and sign consistency; no hypothesis test, calibrated significance or causal
claim from closed-loop correlation. Confirm existing.06rad/chunk slew cap.

This checks whether a missing upstream motor association deserves investigation
before changing reward/credit timing. It cannot prove absence of coupling or
walking and is not a substitute for physical retention/ablation controls.
Offline budget60seconds,>=.5GiB available before reading metadata, no full graph,
no physics, no retry. Output unique primary:
gpu-pilot/outputs/exploration-motor-m1-20261030/result.json.
Keep raw neuron metadata local; only its hash and compact joint metrics in Git.

## Observed outcome

All42channels had varying projected input and commands. Fixed25ms current-level
to command-level correlations were positive in all3seeds for10channels; median
per-seed r=.0524/.0623/.0166. Innovation-to-command-change correlations were
positive in all3seeds for7channels; medians=.0081/.0112/.0042. These are DIFFERENT
signal/output pairs and their coefficient magnitudes cannot rank credit rules
or establish better learning SNR. No causal or no-coupling conclusion follows.

Command slew-cap hits were.147%/.089%/.260%; this gives no reason to relax the
existing cap. Historical metadata hashes are absent from the training reports;
this audit records the current cache hash rather than claiming verified historic
metadata identity. Complete joint rows and input hashes are archived; compact
result: exploration-motor-m1-result.json. No raw metadata is published.

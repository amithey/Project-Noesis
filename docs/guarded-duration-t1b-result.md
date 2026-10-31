# T1B: full duration completed; retained-behavior rule failed

2026-10-31. Registered protocol: [T1B](guarded-duration-t1b-protocol.md).
Allocation code4799b80/HEAD3a2128a, TEAM94b1aac79d6b, issues17/18.

One full25s physical training birth completed5000chunks/4990updates, followed
by ALL12fresh full5s frozen probes. Training inner393.279s/process397.813s;
aggregate1572.625s of3600. All13processes exited0, all telemetry terminal and
all prejob admission windows passed. Lease released; no next run allocated.

The exact19field first5s prefix matches D3, alignment passes, sources and parent
hashes remain unchanged, and the original-reference topology/sign/weight-bound
audit passes. Completed experimental child SHA256:
`ab4cf13f3a7908ae36e87826267ba8f9c13a2134562cdae1fd4d191c48db0641`.
It is not validated-best, a promoted parent or a new curated training-last.
Current curated C2d701 checkpoint remains unchanged; complete failed candidates
are kept separately from partial episodes.

## Frozen outcomes

| Seed | Parent standing % / net mm | 5s control standing % / net mm | 25s standing % / net mm | Child primary / retention vs both |
|---|---|---|---|---|
|10006|96.263 /0.0223|92.929 /-1.2360|98.081 /0.6701|pass /pass|
|10007|94.444 /0.3840|91.212 /1.0763|90.404 /-0.5524|fail /fail|
|10010|89.293 /1.4683|95.152 /1.4266|90.404 /-0.3810|fail /fail|
|10011|91.919 /-0.0453|98.889 /1.4410|91.616 /0.7935|pass /fail|

Primary requires standing>=.9, heading>=.25mm, slip<=10mm/s and>=2cycles in
>=4legs. Retention requires no standing decrease or slip increase vs EITHER
control on EVERYprobe; passing an absolute standing gate is insufficient.
Child standing decreases vs parent in2/4 and vs5s control in3/4. Slip increases
vs parent in2/4 and vs5s control in3/4. Only1/4probes retains both metrics vs both.
All four child regimes are STEPPING_IN_PLACE; no tripod walking is established.

| Median metric mm | Parent | 5s control | 25s child |
|---|---|---|---|
|Heading forward|0.315076|1.051333|0.059006|
|Net forward|0.203173|1.251473|0.144518|

The registered median heading AND net improvement>=.25mm vs BOTH controls fails.
Overall exploratory duration-method rule: **FAIL**. The good10006outcome alone
would have hidden regression in the other conditions. More continuous training
under this fixed method is not supported as a route to retained improvement.
This is N1training seed/four development probes; it does not establish universal
failure of longer training or identify LR, credit assignment or reward as a cause.
Held-out seeds10,11,200,201,202 remain unused. No synthetic test is gait evidence.

## Independent verification and resource limits

[Audit script](../scripts/audit_duration_result.py) rechecked D4M prerequisites,
source/parent/control/child hashes and raw-bank invariants, exact training prefix,
every native report/telemetry, finite1000row frozen arrays, matching settings,
and all12summaries recomputed from raw NPZ. It recomputed the unchanged decision
and matched the runner exactly. Full metrics/deltas/counts/regimes and resource
receipts: [compact JSON](guarded-duration-t1b-result.json).
This post-run audit uses the same versioned evaluator/decision implementation;
it checks raw-file consistency and completeness, not a second independent
definition of gait or an independent validation of the measurement model.

Observed minimum in-run available RAM3.338909GiB, maximum childRSS0.726673GiB,
maximum sampled hostCPU100%.8/13prejob windows also passed originalT1rule;
all13passed the separately registered meanCPU policy. This establishes this
completed study under observed conditions, not a causal resource-policy benefit,
2GiB-boundary validation or changed default policy. No user applications closed.

## Evidence archive

[Release](https://github.com/amithey/Project-Noesis/releases/tag/duration-t1b-2026-10-31).
91named evidence files plus internal manifest, including all new observations,
training bank/metadata/logs, reports, resources, decision and parent/D3 references.
No raw connectome, graph cache, environment or credentials. Every file hash and
ZIP CRC verified locally before publication; uploaded digest/size checked separately.

- duration-t1b-evidence.zip:31927502bytes;
  SHA256`0549351f09a2df4fc8fefcb5474c0790f5e3785086d16227a300c652dadadcb7`.
- duration-t1b-manifest.json:
  SHA256`126c55f6acf6444d2f43749c6643b26e4c804f8cdc3c036a3e7b02ab9b06b10c`.

## Next scientific decision

Do not repeat the same longer-duration intervention as an assumed cure. The
existing lower-LR proposal (0.02vs0.2) is a possible one-factor next experiment,
not an accepted fix: match duration, parent and settings against this25s control,
retain the original frozen gates, preregister budgets/stops and check all probes.
Another option is to review a delivered learning repair first; Claude's PCA/
coverage/null corrections are still pending and do not authorize tuning a lag.
Gemini sensory changes remain a separate candidate. No weight merging, automatic
stage advancement or new heavy allocation follows from this negative result.

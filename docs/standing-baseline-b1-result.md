# B1 completed: neither bank qualifies across all four standing probes

1November2026 local time; registered output suffix remains20261031.
Code81c255e, [protocol](standing-baseline-b1-protocol.md), TEAMaac165061b8c,
issues17/18. Eight NEWfull5s frozen probes completed,8exit0,747.984s of1200.
No learning, new weights, user app closure, retries or budget extension. Lease
released; no further simulation allocated. Sources remainf52c8d/80556cd.

## Full matched comparison

| Seed | Original standing anchor fb80: standing % / slip mm/s | Descendant/current parent9fd: standing % / slip mm/s | Standing check |
|---|---|---|---|
|10006|97.172 /2.3489|96.263 /3.0582|both pass|
|10007|95.051 /2.2429|94.444 /2.4902|both pass|
|10010|88.788 /2.5502|89.293 /2.2500|both FAIL|
|10011|90.404 /2.1390|91.919 /2.9328|both pass|

Unchanged native standing criterion:>=90%standing and mean slip<=10mm/s after
50mswarmup of a FULL5s probe. Both banks pass3/4, fail10010. The ancestor also
does not retain standing/slip vsdescendant on every probe. No automatic parent
switch, curriculum advancement, validated-best or model promotion.

The two historical standing seeds10006/10007pass again, consistent with the
earlier limited gate. The new result extends evidence to10010/10011; it does
not rewrite historical evidence or assert the old two-probe gate was never met.
The simple hypothesis "return to the old standing bank for an all-four baseline"
fails. Existing original fb80and descendant9fd banks remain intact.

Anchor10010has93upper-height failures and18no-support samples, with overlap;
NONEfails uprightness or lower height. Descendant10010has60upper-height,
27uprightness and20no-support violations. Thresholds stay unchanged. These are
measured criterion failures, not evidence that every event is a ground-collapse.
Detailed episodes, causes, gait cycles/regimes and progress:
[raw-recomputed result](standing-baseline-b1-result.json).

Anchor regimes are3STEPPING_IN_PLACE/1STEPPING_UNCOORDINATED; no walking proof.
Standing qualification itself was the prespecified question, not the walking
gate. Four development probes are not final validation or perturbation robustness;
held-outs10/11/200/201/202remain untouched.

## Runtime and raw verification

All four fresh descendant traces EXACTLY replay corresponding T1B observations
in all raw fields. [Post-run audit](../scripts/audit_standing_baseline.py) recomputed
all8native summaries and failure details, matched reports/settings/bank hashes,
finite1000row arrays, per-job terminal telemetry/exit0, original-reference bank
invariants and the complete decision. The versioned evaluator is reused; this is
raw consistency verification, not a second definition of standing.

All8fresh10sample admission windows passed the experimental meanCPU<=60/minRAM2
policy. Minimum observed in-run availableRAM2.548664GiB, peak childRSS0.750336GiB,
max sampled hostCPU60%. Live1GiB stop was not reached. No default admission or
scientific gate was changed.29focused software tests preceded the physical study;
their success was not counted as physical standing success.

## Signed evidence

[Release](https://github.com/amithey/Project-Noesis/releases/tag/standing-baseline-b1-2026-11-01):
69named files plus internal per-file manifest. Includes all8new observations/
reports, telemetry/preflights/logs/summaries, bank/report references and old
descendant replay inputs. Input SHA checked before/after copying, ZIPCRC and all
file hashes verified. No raw connectome/cache/env/credentials.
[Remote asset verification](standing-baseline-b1-artifact-verification.json)
confirms both uploaded hashes and sizes match the local signed artifacts.

- standing-baseline-b1-evidence.zip:13314291bytes,
  SHA256`677543a7553a11125a5f866e2e5944a327f51c3d51e00823c071de871f0d5445`.
- standing-baseline-b1-manifest.json:
  SHA256`619260a93b840c2c06217a1da7afcc369c4eb87f80c200b3c456aaf111d5118f`.

## Consequence for the next experiment

[Offline reward diagnosis](standing-entry-audit-20261031.md) found exact native
reward/normalizer replay and zero positive reward or positive learning signal on
nonstanding samples of T1B/L2. It did not identify a scorer implementation bug.
Returning to an old bank also does not establish the required expanded baseline.

Next design should target standing acquisition/retention explicitly, rather than
another duration/LR sweep or unsupported sensory calibration. One bounded
hypothesis to register separately is the EXISTING standing-v7 profile versus
gripped-cycle-v9 at matched parent/duration/rate. This is a profile replacement
(standing,progress,slip/cycle coefficients differ), not attribution to one term.
Retain frozen per-probe regression checks and disclose gait tradeoffs; a standing
result must never be relabeled walking. Any success remains exploratory until
replicated and tested over sequential births. Claude's credit-control corrections
and actual sensory-hook proposal remain separate reviewed workstreams. No new
training run follows automatically from this result.

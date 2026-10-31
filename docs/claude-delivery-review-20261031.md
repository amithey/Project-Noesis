# Independent review of new Claude deliveries — 31 October 2026

No new simulation was launched by Codex. Original pilot80556cd/coref52c8d,
curated C2d701 training-last, validated-best=null and scientific gates remain
unchanged. This review supersedes the older claim that no correction was delivered.

## Immutable sources and software checks

| Delivery | Reviewed commit | Independent checks |
|---|---|---|
|Credit-timing repair, PR25|77a3fa93ccba1d4d4f488bd8d6f3a4397b945629|16 passed|
|Independent-credit control|0f542eb1a6d4d6876a43ea70f66924ad5e2d430f|37 passed including native/LR regressions|
|Senses selection layer|059750103bce98748a6719e8ac12675dfa8f5499|26 passed|
|Reward-variance/L1b report|dc12cad5b0f35ce70018af287f08f57283c29ae9|Source/report read; full statistical runs not rerun|

All79tests used isolated copies of exact Git blobs and a pytest plugin asserting
import locations and recording module hashes. No owner working branch was edited,
reset or merged. Initial PowerShell argv construction failed before collection;
the corrected provenance-checked runs above are authoritative. Three additional
Codex tests reproduce scientific decision edge cases. [Hashes and acceptance](claude-delivery-review-20261031.json).

The credit repair actually pools a common predictor basis across births, uses
the full seven-joint response rather than per-birth PCA signs, explicitly counts
nerve/region assignments, unknowns and constant columns, and labels ranks as
uncalibrated. This repairs the previous concrete analysis defects for descriptive
offline use. Coverage372/453 refers to explored drive neurons; it is a different
denominator from the earlier motor-neuron annotation audit, not a contradiction.

## Recomputed physical credit-noise comparison

[Audit script](../scripts/review_claude_noise.py) checked six complete25s births
and all24full5s frozen probes from the actual NPZ/report files. All banks pass
original-reference topology/sign/bounds/parameter checks and version/normalization
checks. Parent9fd7d411 is shared. Within each training pair ALL19first10prelearning
trace fields and ALL6body-side exploration arrays match exactly. This closes the
previously unverified warmup comparison for seed1000 too. Independent applied
credit is separately logged and differs from real credit; correlations are
0.000232,0.002736,-0.000689 for1000/1001/1002.

Every frozen horizon/context/checkpoint and every published summary was recomputed
with the versioned evaluator. [Full paired metrics and input hashes](claude-noise-review-20261031.json).
The pooled real-minus-control median is-0.246100mm heading and-0.291082mm net.
All three published pair verdicts and the aggregate verdict reproduce:
**INDISTINGUISHABLE means neither registered superiority rule passed.** It does
not establish equivalence, absence of credit information or that learning is a
random walk. Three training pairs are the replication units;12probe differences
are not12independent training replicates. Final held-outs remain unused.

Terminal update norms from the shared parent are real/control253.6/177.6,
222.1/240.9 and219.1/198.1. Equal configured LR and broadly similar norms are not
exact norm matching. Mean deviation from ORIGINAL weights is dominated by the
already learned parent and is not a substitute for change from that parent.
Cosines0.088/0.109/-0.014 are descriptive direction comparisons, not calibrated
evidence that the real update contains no signal.

## Corrections requested from the owner

1. `analyse_replication.py` counts standing and slip non-regressions separately.
   Three passes on each can occur on only TWO common probes. If the intended
   condition is retention on three probes, count the conjunction per probe.
2. Its reverse condition `standing_ok<=1 and slip_ok<=1` is not symmetric with
   non-inferiority when ties exist. Count the reverse inequalities directly.
   Three Codex regression tests demonstrate both cases. Applying the stricter
   same-probe/symmetric interpretation leaves ALL THREE current pair verdicts
   unchanged. Do not rewrite a preregistered result after the fact; disclose the
   implementation ambiguity and fix future use with an explicit rule.
3. All five `profile-run.json` files have `cycle_scoring.steps=0` while native
   `report.json` has5000. The wrapper scorer is stale/unused on the native route;
   raw reward traces/native reports remain valid. Fix sidecar provenance instead
   of treating zero as evidence that the actual reward omitted the cycle term.
4. All25Claude process receipts omit exit codes. Complete native reports and
   full finite arrays verify simulated horizons, but do not retrospectively
   prove terminal exit status or Codex resource telemetry. Preserve that limit.
   Seed1000 also retains a disclosed wrapper/source-path difference despite
   now-verified raw warmup/exploration equality.
5. Null99th percentiles are descriptive reference thresholds. They are NOT
   calibrated detection power, a minimum detectable effect, or upper bounds on
   effects absent from these analyses. Those require power/calibration and
   assumptions not established by the current circular-shift reference. The
   reward-variance/L1/L1b statistical results were not independently rerun here.

These issues were sent in #19 and TEAM8d93684d710f. Native credit-control
integration is not accepted automatically and no new run is allocated. The data
do support testing a defined learning-mechanism hypothesis with an appropriate
control, rather than assuming more duration/LR tuning will yield retained skill;
they do not identify a proven causal fix.

## Senses ownership and remaining integration

Claude reports that the user transferred all senses to him (TEAM and #20 comment
6548732015). Coordination records this reported directive and keeps his
`claude/senses` branch separate. Gemini4a78e0b source/evidence and outstanding
review are preserved for handoff; no file ownership conflict is resolved by
overwriting either branch. Gemini can continue independent measurement review;
no new overlapping sensory implementation is assigned here.

The26senses tests establish annotation-derived, disjoint channel selection and
the existing adapter contract only. There is no backend input source or neural
step hook, so no new modality currently drives the physical pilot. No behavioral
benefit, biological calibration or walking success follows from these tests.
An actual proposed signal-source/controller patch and matched disabled-input
controls are required before integration review.

Separate Claude/Codex lock filenames can race. The review proposes one canonical
atomic host lease with an owner field; existing leases/processes must be checked
without deleting another agent's lease. No new heavy allocation occurs here.

## Signed evidence preservation

[Release](https://github.com/amithey/Project-Noesis/releases/tag/claude-credit-noise-review-2026-10-31)
contains88named raw input files plus an internal manifest, with all six banks/
training traces/logs/reports,24frozen reports/observations and process receipts.
The prior failed startup attempt remains locally preserved and disclosed in the
owner's protocol; it is not counted as a completed birth. No raw connectome,
graph cache, environment or credentials are archived. All88input hashes were
checked before/after copying, and ZIP CRC/per-file hashes verified.

- claude-credit-noise-evidence.zip:132016429bytes,
  SHA256`2ae35e4105765852261f55806035f76d933e713c34ac6d13a4baf1c03e0a036a`.
- claude-credit-noise-manifest.json:
  SHA256`de8346f7c44bcc95e7677a800f4d2c295b734c05e38050f4927569782b890cc4`.

Publishing preserves independently inspected negative results. No bank is merged
or promoted; stable standing, retained walking and sensory reactions remain goals
requiring further physical evidence.

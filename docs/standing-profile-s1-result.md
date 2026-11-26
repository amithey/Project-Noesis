# S1 complete: standing-profile acquisition did not retain standing

1November2026. [Preregistered protocol](standing-profile-s1-protocol.md),
codef29a77d, TEAM6dc657ef68f4, issues17/18. Existing nativev7standing profile
vscompleted25sv9gripped control; same original9fdparent/seed1000/LR.2/
exploration1.5/core42c2076/f52c8d/pilot80556cd. The profile replacement changes
several reward terms; no one-coefficient causal attribution.

Full25s/5000chunks4990updates completed, then ALL12fresh full5sfrozen probes.
13processes exit0,1660.610s of3600aggregate. Training413.760s inner/423.406s
process. Lease released; no next run allocated. Completed experimental child:
`5e91a54875de34c1e7c8f52734e16249ad66ee515d1416b3848e982303825f13`.

## Standing acquisition and retention

| Seed | Parent standing % / slip | v9control standing % / slip | v7child standing % / slip | Native standing / retention vs both |
|---|---|---|---|---|
|10006|96.263 /3.0582|98.081 /2.2273|93.535 /2.8961|pass /FAIL|
|10007|94.444 /2.4902|90.404 /2.5881|91.919 /2.3788|pass /FAIL|
|10010|89.293 /2.2500|90.404 /2.3861|89.899 /2.3925|FAIL /FAIL|
|10011|91.919 /2.9328|91.616 /2.3807|92.525 /2.5955|pass /FAIL|

Slip ismm/s. S1required ALL4standing>=.9/slip<=10 AND EACHprobe no standing
decrease/slip increase vs BOTHcontrols. Actual3/4absolute standing gates,
0/4retained standing+slip vsboth.10010has890/990standing samples, requiring891
to reach unchanged90%. That near miss does not justify moving the threshold;
retention independently fails even if it were rounded up. Standing falls vs
parent in2/4 andvscontrol in2/4. Slip increases vsparent in1/4,vscontrol in3/4.
**Standing-method rule: FAIL.** No parent switch/latest/stage/model promotion.

## Gait diagnostics were preserved

| Medianmm | Parent | v9control | v7child |
|---|---|---|---|
|Heading forward|0.315076|0.059006|0.800089|
|Net forward|0.203173|0.144518|0.809730|

Median headingANDnet gain>=.25mm vsboth passes, but the original per-probe
walking primary gates pass only2/4 and retention0/4. Original overall walking
decision therefore FAILS. Child regimes:2STEPPING_IN_PLACE/2STEPPING_UNCOORDINATED.
More displacement is not retained stability, coordinated walking or successful
standing acquisition. Full paired deltas, cycles and regimes are in
[the raw-recomputed JSON](standing-profile-s1-result.json).

N1training/four reused development probes, not sequential retention, population
generalization or final validation. Final10/11/200/201/202unused. All banks remain
separate; curated C2d701andvalidated-best=null unchanged. Native reward scores
across profiles were not used as success evidence.

## Verification and limits

121focused tests passed before allocation. Actual completed T1B control and its
70JSON/NPZreceipt hashes reverified before lease and checked throughout. S1has
16finite native trace fields; the3v9diagnostic fields are deliberately absent.
All15PHYSICALfirst10prelearning rows matchv9 exactly (reward intentionally may
differ), and ALL6exploration arrays match over4990learning rows. Native report
AND model sidecar identifyv7/LR.2. Original-reference topology/motor indices/
parameters/rule/sign/bounds audit passes. No core/controller/learning edits.

ALL8fresh parent/control raw frozen traces exactly replay T1B. The post-run
[audit](../scripts/audit_duration_result.py) recomputed12summaries and the complete
standing/walking decisions, matched contexts/banks/full finite horizons, and
checked all terminal telemetry/exit codes and budget. This uses the same
versioned evaluator, not an independent definition of gait.

Every initial/prejob10sample resource window passed the explicitly experimental
meanCPU<=60/minRAM2 policy. Observed in-run minimum1.952736GiB, peak childRSS
0.763535GiB, maximum sampled hostCPU100%. The live1GiB floor was not reached.
Launch and in-run limits are different; no general policy/boundary validation
or original-default change follows. No app closure/retry/extension occurred.

## Signed evidence

[Release](https://github.com/amithey/Project-Noesis/releases/tag/standing-profile-s1-2026-11-01):
91namedfiles plus internal perfileSHAmanifest, including candidate observations/
bank/metadata/logs, all frozen outputs, resource receipts and parent/v9control
references. ZIPCRC and every fileSHA verified; no connectome/cache/env/credentials.
[Uploaded asset verification](standing-profile-s1-artifact-verification.json)
confirms both GitHub digests and sizes match the local signed files.

- standing-profile-s1-evidence.zip:43598494bytes,
  SHA256`6548cf9b53258a41a5ab6428002b7a848682f4644a0ac38ee9aa225d7847d168`.
- standing-profile-s1-manifest.json:
  SHA256`bb3b95ef79fef2b534bc15c6502f1f2808988a4d830113000c39e95115e33738`.

## Next diagnostic, not another automatic sweep

Duration, LR and now existing profile interventions completed without retained
success in these specific tests. They do not establish universal method failure
or a proven cause. Before another training sweep, isolate the existing exploration
intervention with FIXED weights: matched exploration-on/off neural-input conditions,
logging actual neural activity, muscle commands and body response. Include a zero-
intervention path that exactly reproduces native behavior, and verify observation
does not alter default dynamics. Inputs must enter the brain; no scripted leg
sequence/external gait or commanded body motion. Perturbed episodes would be
explicit diagnostics, never ordinary frozen walking validation. This requires a
separate protocol/source review/budget/allocation; none is allocated here.

Claude's credit-control decision/provenance fixes and actual sensory input/hook
proposal remain separate workstreams; neither is silently integrated by S1.

# L2: lower LR improved median displacement, failed stability and retention

2026-10-31. [Registered protocol](lr-l2-protocol.md), code0c1fb63,
TEAM2071225ee8af and issues17/18. The sole training intervention is LR.02vs.2
at matched25s duration from original complete9fd7d411, seed1000. T1B ab4cf13
is a control, never the candidate's parent. Same core/pilot/body/reward/exploration.

All13jobs completed:25s/5000chunks4990updates plus12fresh full5s frozen probes.
Aggregate1530.453s of3600; training399.947s inner/404.328s process.13exit0,
lease released, no next heavy job allocated. Candidate checkpoint SHA256:
`6d63d3dc7d433aa3174946ec925edaea4b14bb6cfc8878a599b3df74d42cdb43`.

Candidate report, model sidecar and raw bank record LR.02. The ONLY permitted
raw parameter change is LR.2->.02; topology/signs/original-reference bounds,
eligibility, rule/version/normalization/bias flags remain checked. All19first10
prelearning trace rows and ALL6exploration arrays across4990learning rows match
T1B exactly. All8fresh parent/control frozen traces exactly replay T1B in every
field. These checks support the matched comparison; they are not gait proof.

## Frozen results

| Seed | Parent standing % / net mm | LR.2 standing % / net mm | LR.02 standing % / net mm | LR.02 primary / retention vs both |
|---|---|---|---|---|
|10006|96.263 /0.0223|98.081 /0.6701|96.263 /0.8481|pass /fail|
|10007|94.444 /0.3840|90.404 /-0.5524|83.333 /0.5994|fail /fail|
|10010|89.293 /1.4683|90.404 /-0.3810|90.707 /1.2953|pass /fail|
|10011|91.919 /-0.0453|91.616 /0.7935|97.475 /-0.3033|fail /pass|

2/4primary passes,1/4standing+slip retained vs BOTH controls, and **0/4passes
both requirements together**. Standing falls vs parent in1/4 and vs LR.2 in2/4;
slip increases vs parent in2/4 and vs LR.2 in3/4. The only retained probe moves
backward and fails heading. Regimes:3STEPPING_IN_PLACE,1STEPPING_UNCOORDINATED.
No stable or coordinated walking is established.

| Median mm | Parent | LR.2 | LR.02 |
|---|---|---|---|
|Heading forward|0.315076|0.059006|0.532381|
|Net forward|0.203173|0.144518|0.723737|

Median net improves by0.520564mm vs parent and0.579218mm vs LR.2. Heading
improves by0.473375mm vs LR.2 but only0.217305mm vs parent, below unchanged
0.25mm requirement. Median gain rule therefore fails; retention independently
fails in3probes. **Overall registered exploratory method rule: FAIL.** Do not
lower a threshold after seeing a near miss or promote from the favorable metric.

This is N1training seed/four reused development probes. It establishes this
specific comparison, not a universal LR effect, causal explanation of instability,
retention over sequential births or held-out generalization. Final10/11/200/201/202
unused. No curated training-last/validated-best/stage promotion; C2d701 unchanged.

## Post-hoc weight diagnostic

Measured after seeing the frozen results, and explicitly outside the decision
criterion: mean absolute net weight displacement from shared parent divided by
original-reference magnitude is0.002476596 for LR.2 and0.000242957 for LR.02
(0.24766%vs0.02430%, about10.2times smaller). Maximum displacement is5.603%vs
0.508%of original reference. Neither terminal bank has any edge exactly at an
original±50%bound. All85390plastic edges have nonzero original reference.

Thus this lower rate actually reduced terminal displacement, but that did not
produce behavioral retention. This is terminal displacement, not total update
path length or proof that LR, clipping, reward or credit timing causes the
failure. Further blind rate reduction is not justified as an assumed cure.

## Verification and artifacts

[Post-run audit](../scripts/audit_duration_result.py) reran native summary and
decision code from raw NPZ, checking every complete report/context/hash, full
finite arrays, frozen replays, LR metadata exception, exploration parity,
telemetry and budget. [Compact result](lr-l2-result.json) includes every paired
delta/cycle/regime and the clearly marked post-hoc diagnostic. This uses the
same versioned evaluator, not a second independent definition of gait.
103focused pre-run tests passed; original T1/T1B defaults and scientific gates
are preserved. Minimum observed in-runRAM2.451534GiB, peak childRSS0.752953GiB;
max hostCPU100%. All13mean-policy prejob windows passed;8also passed legacyT1.
No app closure or automatic resource-policy adoption.

[Evidence Release](https://github.com/amithey/Project-Noesis/releases/tag/learning-rate-l2-2026-10-31):
91namedfiles plus internal per-file manifest. Includes candidate training/frozen
data and matched parent/control25 references; excludes raw connectome/caches/envs.
Final archive (replaces unpublished provisional package before post-hoc audit):

- learning-rate-l2-evidence.zip:43649980bytes,
  SHA256`b06885c66f67772ad6dc41cc48886756ff1158ab02ceb6755731c4d61d4febf4`.
- learning-rate-l2-manifest.json:
  SHA256`c95eaea3aeeb3c18ba6490c062b6f1209265c0c1bc8e14fd0466936b0a5f9ad3`.

ZIP CRC/every file SHA verified locally; uploaded digests/sizes checked at
publication. Existing T1B/D4M releases retain their original artifacts/hashes.

## Next diagnosis

Duration extension and smaller LR have now completed matched physical tests
without retained success. Before another parameter sweep, inspect curriculum
entry and reward/retention behavior: the unchanged parent itself fails90%standing
in seed10010(89.293%). A universal stable standing anchor has not been established
by these4development probes. Determine whether a standing-stage intervention
under the existing curriculum is needed before further gait acquisition; preserve
and report the existing walking gates rather than reclassifying standing as walking.
Review Claude's delivered credit-tool repair and Gemini's sensory corrections
when available; no causal inference or lag tuning from the uncorrected tool.
No new mechanism/parameter/curriculum run is allocated by this result.

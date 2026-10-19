# Frozen test of an earlier V17 training checkpoint

The final training bank is a record of the latest learning state, not necessarily
the best bank encountered. Its complete frozen tests failed. Before altering
neural dynamics again, test whether the highest-reward complete training episode
left better frozen weights. This is checkpoint selection from training metrics;
training with ongoing learning/OU perturbations is not frozen performance.

Select the highest mean scalar training reward among the eleven complete
five-second V17 births, tie-breaking by ascending seed. Exclude truncated,
wall-budget, fall-terminated and evaluation episodes. All eighteen training
births remain recorded; this eligibility filter is for candidate ranking only.
The selected bank is training seed50, reward0.6681452789171789, SHA256
7baaf2fabf0c238884f74260e2f3cfa420ce5df4880c6d47c60d62cea2f6247f.
It is not declared best validated and does not replace the latest seed60 bank.

Prepare four frozen five-second quiet trials on diagnostic seeds66/67 at existing
background3Hz with event gains60/40. Reuse the matching fully completed V20 original controls,
whose summary/checkpoint hashes and original frozen weights are rechecked.
The original source839a0a8bd85a8ebca55048319dc95642413491a9a14c5d168fe24cc4d6e8d107,
body, senses, balance, U0, da1, legacy proprioception, geometry adhesion, neural
wings, v4 reward and muscle mapping remain unchanged. Import only selected
weights, preserving original topology/signs/50% bounds. No learning, motor
exploration or early fall termination is enabled during these trials.

The prepared runner uses the already tested V20 case/checkpoint verifier, whose
SHA256 is pinned. A strict source/configuration/parent/control manifest guards
resume; incomplete trials replay from the full selected start bank. Each worker
is limited to3600 wall seconds. Read only stdout while live, not the live ledger.
No simultaneous heavy workers will be started. Seven unit checks pass (partial
birth/evaluation exclusion, changed selected checkpoint/source and missing matched-control rejection),
Ruff passes, and the real checkpoint/control/argument preflight passes.

All eight V20 background trials completed and were independently audited.
The earlier-bank body diagnostic has now started; no new training is running. Regardless
of diagnostic comparison outcome, validation10/11, heldout200..202, obstacle and
external-sense ablation remain required for actual model promotion. Standing90%,
heading1mm, repeated cycles in4legs and20% slip reduction are unchanged. Report
standing-conditioned slip and falls as well as all-window scores. There is no
external gait clock, fixed leg sequence or promotion/startup modification.

Before any earlier-bank run, the initial two-case proposal was extended to both
background gains. The initial driver/plan are retained separately. This uses the
observed V20 original-control and trained seed66 standing improvement at40; it
does not use any result from the selected bank. All four original controls were
already complete. Candidate ranking is unchanged and training-only. Comparisons
remain separated by background gain. Source/model/acceptance are unchanged.


## Launch confirmed

After V20 worker termination was verified, session8749 started the four-case
frozen grid under a3600-second wall limit. The first case is seed66/background60;
stdout confirms learning=False and physical progress through0.40s. No full
case has completed yet. Driver SHA256fa6ebe0ab9e8c1ab8acd856f2c8fc7d53d659716694b2e11d902d271803d6174;
plan SHA2561061d042dfa6357b50d80f966dfb58eb3733ea05b4fff8168914a879c3bbeda8.
The source remains839a0a8bd85a8ebca55048319dc95642413491a9a14c5d168fe24cc4d6e8d107.
Observe stdout only while live. preserve_completed_case.py checks exact source,
plan, selected parent, frozen checkpoint and matched original-control coverage,
then copies only stdout-confirmed immutable completed artifacts. Ruff passes
for this helper. No live ledger was opened; no gait success or promotion claimed.

## First earlier-bank frozen case complete; failed standing

Seed66/background60 completed the full five seconds with learning and exploration
off. Standing is30.8081% versus original78.9899%, mean reward-0.364691 versus
+0.334575 (the exact matching summary is pinned in plan.json). All-window slip
is7.595033 versus12.620769mm/s, but standing-conditioned slip is14.533034 versus
13.998776mm/s: the lower all-window value is associated with extensive falling,
not improved standing gait. Standing-window heading progress is3.724743mm versus
6.321267mm original; all-window12.088498mm and cycle counts cannot overcome the
failed standing gate. Physical and comparison checks use the actual saved summaries.

The stdout-confirmed case was independently verified against selected parent
7baaf2fabf0c238884f74260e2f3cfa420ce5df4880c6d47c60d62cea2f6247f,
unchanged source/plan, original topology and zero learning traces. Every completed
artifact, including the full video, was copied with SHA256 checks. The reused
independent CSV auditor verifies all1000 rows/990 awake samples,5ms cadence,
standing, slip, scalar reward and heading progress. The live ledger was never
opened. The worker has advanced to seed66/background40; three cases remain.
No source or active driver changes, no model promotion or startup update.

## Earlier bank at reduced gain: standing passes, slip improvement fails

Seed66/background40 completed five frozen seconds, standing94.8485% (original
95.7576%), heading9.431014mm and cycles12/14/16/8/11/4. Mean slip14.160074
versus15.429347mm/s improves8.23%, below the required20%. Mean reward0.594951
versus0.586967 passes the existing reward comparison. Standing, cycle, progress
and reward gates pass for this single diagnostic case; slip fails and no model
is promoted. Standing-conditioned slip13.995344 versus15.415065mm/s is also
short of20%. The independent CSV auditor, frozen-parent/context/source checks
and all copied artifact hashes pass. The full video is preserved. Two cases
remain; the live worker has advanced to seed67/background60. The V21 source and
driver remain immutable. A separate staged CLI candidate will allow prospective
training at gain40 with fully matched controls; see background-training-v22.md.

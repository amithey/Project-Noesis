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

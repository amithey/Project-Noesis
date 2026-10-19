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

Prepare two frozen five-second quiet trials on diagnostic seeds66/67 at existing
background3Hz/60. Reuse the matching fully completed V20 original controls,
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
No simultaneous heavy workers will be started. Six unit checks pass (partial
birth/evaluation exclusion, changed selected checkpoint and source rejection),
Ruff passes, and the real checkpoint/control/argument preflight passes.

This is prepared only; V20's remaining background trials are still running.
No new body run or training was launched for this earlier bank yet. Regardless
of diagnostic comparison outcome, validation10/11, heldout200..202, obstacle and
external-sense ablation remain required for actual model promotion. Standing90%,
heading1mm, repeated cycles in4legs and20% slip reduction are unchanged. Report
standing-conditioned slip and falls as well as all-window scores. There is no
external gait clock, fixed leg sequence or promotion/startup modification.
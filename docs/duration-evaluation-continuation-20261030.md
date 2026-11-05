# Frozen evaluation continuation for T1

Implementation: `scripts/resume_duration_evaluation.py`, outside the pinned neural
and pilot source trees. This continues evaluation after a COMPLETE25s birth;
it does not restore within-episode neural, physics, eligibility or optimizer state.
No actual complete25s T1 child exists yet. Curriculum orchestration remains
incomplete and there is no new physical retention or walking evidence.

Before any simulation, require the original T1 protocol, completed25s/5000chunk/
4990update report, unchanged original parent/control/core/source hashes, saved
child hash, original-reference topology/sign/bound audit, exact5s trace prefix,
logged row alignment, and original final wall accounting. The real21.18s T1
partial was rejected by the CLI before output or lease creation. Renaming a
partial bank cannot satisfy the report/protocol/prefix checks.

Completed frozen observations are rescored with the current unchanged evaluator.
Reuse requires exact stored summary including report/observation hashes, expected
bank/seed, complete frozen horizon, and common source/body/senses/settings. Partial
frozen attempts are preserved and excluded; they can be rerun only in a new output
directory, from a complete bank, with remaining original budget. No checkpoint,
original observation, summary or protocol is overwritten.

The original study receives one atomic `evaluation-continuation.json` ledger,
under the existing exclusive shared host lease. New continuation invocations must
use that same study, ledger, banks and sources. A fresh sibling output is required.
Each physical job needs a new10sample >=3GiB/<=30percentCPU gate, live Python
inventory, and >=335s remaining. Same registered12controls, seeds,330s process and
300s inner limits,3600s aggregate active study budget and scientific gates apply.
Downtime is not active compute time; prior active segments are never reset.

Before launching a child, atomically reserve330s in the ledger. Normal terminal
cleanup records actual elapsed segment time, including killed-and-waited failed
children, and clears the reservation. A hard termination leaves its reservation
charged conservatively on restart. An orphaned lease requires live-process
inspection; this script never breaks or replaces a lease. Missing original final
wall accounting is refused, rather than treated as zero. No automatic retry,
training, promotion, stage advancement, checkpoint averaging or validated-best
update is performed.

After an explicit future allocation and fresh TEAM review, usage is:

```text
python scripts/resume_duration_evaluation.py --shared-root PRIMARY --core-source PINNED_CORE_SRC --study ORIGINAL_COMPLETE_T1_DIRECTORY --output NEW_SIBLING_DIRECTORY
```

Verification:24focused synthetic tests passed across duration continuation,
duration decision/prefix, resource assessment and Python inventory. Tests exercise
partial training, source/seed/budget changes, mutated frozen observations, wrong
bank/seed, partial probes, fixed job order, carried reservations, and fake process
failure/timeout with real temporary ledger/lease writes. These tests establish
software behavior only. Real full-model successful continuation remains untested.

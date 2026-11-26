# R4 — bounded frozen operating-envelope feasibility

Issue18; registered before new resource measurements or physical execution.
This is an operational experiment, not a T1 retry, training or gait validation.

Hypothesis: the3GiB/30percentCPU launch rule may be more restrictive than needed
for a single frozen5s world. D1 completed a full5s episode with peak childRSS
about.716GiB, although host pressure degraded runtime and its wrapper stopped
before the second repeat. That observation does not establish safe steady-state
resources, explain historical slowdown, or support changing training gates.

R4 uses a separate launch envelope:10one-second samples, each>=2GiB available
and<=60percenthostCPU. This leaves over1GiB beyond the historical childRSS at
launch, but that is only a bounded hypothesis. During the owned child, sample
host resources every2s and immediately stop/kill/wait that child if available
RAM falls below1GiB. Unreadable/nonfinite telemetry fails closed. No user
application is closed or suspended. The T1 launch rule remains3GiB/30percentCPU.

At most TWO fresh frozen5s repeats, original complete parent
9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c,
seed10006, one CPU world, unchanged42c2076 core/f52c8d source tree and
80556cd pilot tree. No learning/exploration, new checkpoint, model promotion,
weight composition, default sense change or held-out seed. Existing scientific
standing/slip/heading/cycle gates are unchanged.

Exclusive shared host lease, host-visible Python inventory, unique output and
scoped keep-awake. Default sandbox process visibility is insufficient for this
host-wide inventory; execute the registered runner on the host-visible tool
surface. Unknown Python fails closed; no blanket unit-worker exception.

720s aggregate;300s inner and330s process watchdog per job. Require330+5s
remaining after each fresh preflight. Stop on first technical failure, resource
refusal, mismatch, mutation or insufficient full-job budget; no retry/time
extension. Always reap the owned child and release only the matching lease.
Partial horizons never count as completed evidence.

Require both reports to describe full1000chunk/5s frozen horizons with unchanged
weights and matched source/body/senses/package context. Compare all10observation
fields against each other AND the existing D1 complete reference. Record input
hashes before execution and verify they remain unchanged. A mismatch is a
diagnostic failure; it is not a candidate benefit.

Even a pass is only N2 same-seed frozen feasibility under observed conditions.
It does not establish new training throughput, a generally validated resource
policy, causal workload effects, retained learning, or stable walking. Further
policy/training changes require a separate explicit registered decision.

Implementation: scripts/run_resource_envelope.py, outside pinned neural/pilot
source trees.20focused software tests pass across the R4 guard, inventory and
existing resource assessment. R4 tests use fake children for normal completion,
memory stop, timeout, unreadable telemetry and KeyboardInterrupt; cleanup and
terminal telemetry are verified. They are not physical evidence.

Planned unique output: PRIMARY/gpu-pilot/outputs/runtime-r4-operating-envelope-20261031.
A separate allocation notice, fresh TEAM read and actual guard are still required.

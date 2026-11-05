# Frozen runtime diagnostic D1

Codex issue #18; starting commit85666ca, branch codex/training-pilot-20261028.
Goal: determine whether fixed frozen physical episodes finish reliably before
changing learning. E6 raw report confirms316/1000chunks,1.58simulated seconds,
300.6847wall seconds; v8 never ran. Cause is unresolved, not evidence against v8.

Two sequential CPU worlds (one at a time), each5seconds/1000chunks, seed10006,
same complete exploratory9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c
parent, core42c20769376bda5694b4c1c66ca75e52cbd9f1f4. Existing brain controller,
no learning/exploration, no sensory additions, physics/control defaults unchanged.
Development seed reused; held-out seeds remain reserved. New output:
primary gpu-pilot/outputs/runtime-diagnostic-d1-20261030.

Budget720seconds aggregate;300seconds/job including setup,330seconds process
watchdog; reject late output after335seconds. Stop on first incomplete report,
nonzero exit, parent/source mutation, unknown Python worker, lease collision,
invalid observations or insufficient budget. No automatic retries or budget
increase. Exclusive existing host lock, scoped keep-awake; identified Streamlit
dashboards excluded explicitly. Parent checkpoint immutable. Telemetry every2s:
child cumulative CPU, RSS, available host memory, system CPU utilization.
No global environment/power/thread setting changes.

Require complete frozen integrity fields and exact observation-array equality
across repeats. Record per-field max differences, setup/loop/total timings and
process elapsed. A mismatch is diagnostic failure, never model validation.
Two completed repeats establish only local frozen reliability, not training
throughput, historical stall causality, walking, retention or generalization.
Persist compact results/code in Git, evidence with SHA256 manifest in Release.

Capability work stays separate: Claude learning/retention proposal (LR.2 control
versus.02 treatment, not yet implemented); Gemini external senses/composition
awaits semantic review. Do not combine independently trained weights. Future
learning trial requires matched training throughput and frozen-parent retention
controls before selecting a candidate; no automatic promotion.

## Observed result

One real frozen episode completed all1000chunks/5seconds: setup62.087s,
loop205.087s,total267.219s. The wrapper then rejected a nonexistent scalar
`seed` field; the frozen report actually supplies `seeds:[10006]`. This was a
Codex validator defect, corrected with tests after the run. The original stopped
result is preserved; the planned second fresh repeat was never launched.

Post-run offline validation confirms full frozen integrity and exact equality
of all10observation arrays against historical E4's same parent/core/seed episode.
The run/controller/backend/bridge files are unchanged since E4. The historical
episode took75.313s; this is useful retrospective reproduction evidence, not
completion of the preregistered two-fresh-repeat protocol.

Telemetry sampled132times: minimum host available memory0.483GiB, maximum
childRSS0.716GiB; final systemCPU84.7%, childCPU214.484s at268.734elapsed.
Host pressure is observed. These counters do not identify paging, thermal
throttling, suspension or another process as the cause. No unrelated process
was terminated or global power/thread setting changed. No further full
simulation allocated: first measure a resource envelope and profile setup versus
neural/physics CPU time, then demonstrate matched training completion. Merely
raising the wall limit would not resolve the reliability question.

Compact result: [runtime-diagnostic-d1-result.json](runtime-diagnostic-d1-result.json).
Evidence: [D1 prerelease](https://github.com/amithey/Project-Noesis/releases/tag/runtime-diagnostic-d1-2026-10-30),
archive SHA256 `e3bd201ea6493d5c6776f2afda46f071ef546ea370dd6faeee2b9b3a5e079483`.

Integration review also reproduced unsafe acceptance by managed R2 code:
retention comparison accepted sourceA/sourceB with four invented leg names;
sensory composition cast complex rate1+7j to1.0. Recorded in issues19/20 and
dispatched bounded file-only owner repairs. No claimed behavioral-memory or
external-sensory capability follows from those synthetic modules.

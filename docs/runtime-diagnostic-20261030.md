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

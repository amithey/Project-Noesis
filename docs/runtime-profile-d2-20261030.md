# D2 frozen CPU profile

Codex issue18, starting42d3244, codex/training-pilot-20261028. D1 completed5s
with exact historical trace but3.55x wall time and low available memory. This
profile identifies expensive code regions; it does not infer historical cause.

One frozen0.5s/100chunk CPU episode, same9fd7d411 parent, isolatedcore42c2076,
seed10006, unchanged body/brain/controller/defaults. cProfile records native
function own/cumulative time, including imports/setup/JIT. No learning, no
calibration changes, no held-out seeds or topology/sign/bound changes.

Require>=2GiB host available at launch, existing exclusive host lease and live
Python inventory. Latest preplanning snapshot3.28GiB available/12.7percentCPU
is not a guarantee. Budget180s total,120s inner budget,150s process watchdog;
reject late output after155s. Stop first failure/partial/source mutation/trace
mismatch/resource insufficiency. No automatic retry or budget extension.
Output primary gpu-pilot/outputs/runtime-profile-d2-20261030.

Require complete100chunks and exact equality of every observation array against
first100samples of completed D1. Save raw profile locally plus sanitized compact
top25 own/cumulative function costs, per2s CPU/RAM telemetry, hashes/provenance.
Profiler overhead is a confound: do not compare its wall time as an unprofiled
performance improvement. A short frozen completion does not establish full
training reliability, walking, behavioral retention or model promotion.

Next test depends on the dominant observed cost. Do not change physics timestep,
brain dynamics, JIT fastmath or sensory calibration merely to increase throughput.

## Observed outcome

Completed100chunks and exact equality of all10D1 prefix fields. Process39.125s;
setup26.811s, loop10.245s,total37.090s. LIF.run8.962s cumulative versus native
physics mj_step0.756s. Brain loading19.056s includes repeated CSV parsing/scaling.
Minimum available host memory2.656GiB. This local profile identifies neural and
setup costs; it does not prove the cause of D1/E6 slowdowns. Profiler overhead
precludes treating its time as a matched speedup. See runtime-profile-d2-result.json.

# S1 — existing standing profile versus gripped profile at matched25s

1November2026. Registered issue18 before implementation, baselineeb12f25.
B1 completed8frozen probes: both original standing anchor and descendant pass3/4,
fail10010. Exact native reward/normalizer replay found no positive reward or
signal on nonstanding T1B/L2 samples. Returning to an old bank or reducing LR is
not an established fix. Hypothesis: directing acquisition toward standing with
the EXISTING v7 profile may improve retained standing across these conditions.
It may instead worsen slip or gait; no success is assumed.

## Only training intervention: objective profile

ONE25s birth from complete9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c,
seed1000, LR.2, exploration1.5, eligibility100ms. Candidate native --profile standing
(curriculum-standing-v7), versus COMPLETE25sgripped-cycle-v9 control
ab4cf13f3a7908ae36e87826267ba8f9c13a2134562cdae1fd4d191c48db0641.
Control is NOT the training parent. Same core42c2076/f52c8d/pilot80556cd,
body/senses/timesteps/packages and original-reference bounds. No core, learner,
controller or worker-file edits; existing option only, defaults preserved.

v7 removes progress/cycle credit, increases standing coefficient.4->.7 and reduces
slip coefficient.03->.006. This is ONEprofile replacement with several terms;
outcomes cannot be attributed to a single coefficient. No external gait, leg
sequence, brain bypass, weight composition or hidden sensory intervention.

Before lease, re-audit actual complete T1B plus D4M prerequisites and source/bank
hashes. Freeze their receipt hashes throughout. Require5000chunks4990updates,
16finite nativev7trace fields (threev9diagnostics deliberately absent). First10
PHYSICAL trace rows (all except reward) EXACT vsT1B; reward may differ even during
warmup. ALL6exploration arrays must match the25scontrol across4990rows. Report
AND model sidecar must identify v7/LR.2; profile is the only intended context
difference. Raw bank topology/motor indices/initial reference/parameters/rule,
signs and original±50%bounds unchanged. Save complete child/hash before evaluation.

Then12NEWfull5sfrozen probes parent/control25/child25 x10006/10007/10010/10011.
Matched settings/source/body/seeds, learning/explorationoff, weights unchanged.
Fresh parent/control25 raw observations must EXACTLY replay corresponding T1B
files. All probes complete even after scientific failure; only technical failure
stops. Native full-horizon evaluator and independent raw recomputation required.

## Standing success and gait diagnostics remain distinct

Standing-method success requires ALL4native standing gates (standing>=.9,
slip<=10mm/s) AND on EACHprobe no standing decrease or slip increase versus
BOTH parent and25sv9control. An absolute standing pass cannot hide regression.
No relaxed standing threshold or averaging-away of a failed seed.

Also report the ORIGINAL walking/progress/cycle decision without modification:
standing>=.9,heading>=.25mm,slip<=10,>=2cycles in>=4legs onALL4, retention vsboth,
median heading AND net gains>=.25mm vsboth. These remain diagnostics here;
standing-only success never becomes walking proof or stage/model promotion.
Report all paired deltas/cycles/regimes, including gait tradeoffs. N1training/
four reused development probes is exploratory, not sequential-retention or final
generalization evidence. Held-outs10/11/200/201/202 remain untouched. Any success
needs later replication/continued-birth tests and appropriate noise controls.

## Operational budget

Explicit --guarded-mean-cpu --d4m-study COMPLETE_D4M
--standing-control-study COMPLETE_T1B; mutually exclusive with L2opt-in.
Unique PRIMARY/gpu-pilot/outputs/standing-profile-s1-20261101.3600saggregate,
900inner/930process train,300inner/330process perprobe, fullnextprocess+5remaining.
Fresh10samples beforeinitial and EVERYjob: minRAM2GiB/meanCPU<=60%; retain
legacyT1decision/maxima. Live2stelemetry/memoryfloor1GiB with ownedchild stop/reap.
Canonicalexclusivelease+host-visible Python inventory; refuseotherhostlease
pending authority check, neverdeleteit. No otherheavy/fullconnectome allocation.
Stopfirsttechnical/refusal/partial/mismatch/mutation/resource/budget failure;
no retry/extension/fallback/appclosure. Generic originalT1resume does not support
S1; preserve a complete child if later probes stop, never partial training asparent.

Tests, commit and separate allocation precede simulation. Compact code/settings/
results in Git; raw evidence/checkpoints in Releases with per-fileSHA/ZIPCRC and
remoteassetdigest/size verification. Exclude raw connectome/cache/env/credentials.

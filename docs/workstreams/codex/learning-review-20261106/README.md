# Learning control review — 2026-11-06

The full goal remains active and unachieved. This review changes no fly runtime, learning rule, connectome, parent bank or peer checkout. No physical simulation, checkpoint, walking/balance/retention validation or new heavy allocation occurred.

## What was verified

Claude aa7ecdb already delivers a positive/negative MotorPlasticity association control. Reproducing all48published rows under exact core42c2076 yields max absolute error0 and identical registered verdict. Positive meanD.00125592,z11.5678; negative meanD.00002752,z.25368; learning-off exact and original bounds/signs hold. A prospectively registered reversed planted effect kappa=-.2 over the same16synthetic draw IDs yields meanD-.00120591,z-8.35899, preserving signs/bounds. Worker exit0, entire Windows job empty,12.5s. The original0..15 identifiers are synthetic draws, not fly final-probe observations.

This closes independent reproduction of that delivered association control. It is not missing and should not be rerun as the next unresolved task. Its scope is exactly what Claude's source/protocol says: spikes are externally sampled Bernoulli values, reward is an externally planted OU relation, and learned weights do not feed back into either. The source never calls net.run during birth. Passing this control demonstrates an update-direction association under its assumptions, not improved closed-loop policy performance.

## Separate prospective CL1 review

Registered in issue18 comment6683379561 before data. Real LIF network with three neurons and two signed edges0->2=25,1->2=-10. Poisson input40/20Hz,eventweight400, background3Hz/40,dt.5ms/control5ms/adaptation1. Actual target spikes drive an EMA100ms readout, reward=tanh(rateHz/100). Target-only OU current1.5mV/ms,tau100ms, innovation credit and the native perturbation learner atLR.2/eligibility100ms with normalization/bias correction and original±50% bounds. Five-second trials,50ms no-learning warmup. Sixteen synthetic training seeds50000..50015; four two-second frozen synthetic probe streams60000..60003 averaged within each birth, first100ms excluded. These are no fly or final physical probes.

Arms: actual online reward; the same actual reward series shifted300activechunks as temporally yoked control; learning-off. The yoked reward marginal matches exactly. The fixed pass screen requires positive mean and z>=3 for online-minus-parent AND online-minus-yoked frozen reward. No tuning or rerun after results.

**Registered screen FAIL.** Every frozen probe return is exactly0 for parent, all16online children, all16yoked children and learning-off. Both mean differences0, standard error0,z undefined; do not turn this into significance or an absence-of-learning claim. Online weights change by up to4.2495% of original magnitude; yoked up to2.4780%. Signs/reference bounds hold; off weights and probe outputs are exact. Mean actual training reward online.0320745,yoked.0316898,off.0316900. Worker exit0/jobempty,12.328s; core files unchanged.

## Post-hoc diagnosis, not a rescued pass

The original frozen readout lies on a firing/reward floor. Reusing the identical probe function, seed streams and horizon, an extreme sign-preserving favorable boundary [37.5,-5] yields returns [.0157699,.0105117,.0105137,.0205774], while original [25,-10] and opposite boundary [12.5,-15] remain0. Runtime1.5s, no learning. These hand-set synthetic boundary weights are calibration only, not a trained candidate or a fly checkpoint.

The assay can detect a large within-bound change, but cannot resolve the changes produced by this fixed five-second experiment in the chosen frozen streams. The failed screen stays failed. It does not establish that the learner is incapable, that the fly has a firing floor, or that this is the cause of Noesis's physical retention failures. No plant/gain/reward/duration/threshold was retuned into a pass.

## Next decision

Do not confuse an association-control pass or changed weights with retained neural performance. A prospective closed-loop assay must first establish non-floor/non-saturated sensitivity to changes at the EXPECTED update scale, using calibration data separated from its learning evaluation. That prerequisite failed here; do not spend a large physical-training budget on this tiny plant.

For the actual fly, continue the distinct causal exploration-to-reward/credit and frozen behavioral-retention questions; the warm input response verified in PR30 does not resolve those. Claude retains mechanism/credit implementation ownership; Codex owns curriculum, integration and standing/walking comparisons. No independent banks are merged, original bounds/signs/topology remain fixed, final10/11/200/201/202 are reserved for physical validation, and no new simulation is allocated by this report.

## Reproduction and artifacts

Use the prepared second-host layout ROOT with work/Noesis-Core at42c2076, work/Noesis-Diagnostic ataa7ecdb and work/noesis-env. Restore the three scripts from this review's scripts/ into ROOT/work, then invoke its Python on each script. review_mechanism_control_20261106.py replays the existing association control; closed_loop_readout_review_20261106.py creates the fixed CL1 once in a new output directory; calibrate_cl1_posthoc_20261106.py only diagnoses that preserved result. Numerical JSON/receipts and named synthetic arrays carry hashes in the separate evidence archive. Release upload remains pending the previously verified invalid CLI authentication/browser logged-out condition; compact code/results are published through the connected GitHub tools.

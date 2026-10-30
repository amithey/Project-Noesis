# C1 offline development credit diagnostic

Codex issue18; baseline06bc732; own scripts/credit_diagnostic.py and tests. No
physical simulation, new checkpoint, source pin or learning change. Analyze the
already completed D3 logged seed1000 birth, parent9fd7d411/core42c2076. This seed
was used for development/reproduction and is not confirmatory or held out.

First replay active rewards through the actual pinned MotorPlasticity normalizer
on a tiny no-spike/no-perturbation reference network. Normalization is independent
of spikes and eligibility; test equality against returned metrics from a real
small LearningBrain. Require exact final baseline, variance and normalization
mass equality to saved D3 bank, plus all report/hash/column/row alignment checks.
Do not substitute a generic EMA or include the10warmup rewards in learning.

Before applying analysis: use every varying motor-credit column, Pearson r over
the common central790samples of990active rows; observed signed lags +/-100steps
(+/-500ms), positive-lag family0..100 primary, negatives descriptive. Each
observed lag excludes wrapped samples and uses identical sample count. Search
statistic is max absolute r across all varying channels and101positive lags.
Constant columns are excluded by zero variance, not association strength.

Reference499uniform circular shifts (seed271828), offset choices201..789,
same all-channel/lag maximum. This preserves the observed series' temporal and
cross-channel structure but assumes exchangeability of temporal alignment.
A short adapting episode is not established stationary; label the output a
reference rank, NOT a calibrated p-value or significance test. Report negative
lag maximum, effect size and full search family. No selection of a learning
method, causal credit-lag claim or absent-learning conclusion from this seed.

Offline budget120seconds maximum process time, no automatic retry. Small FFT
arrays, no full connectome/physics. Stop on invalid data, wrong core, partial
episode or state replay mismatch. Output unique primary:
gpu-pilot/outputs/credit-diagnostic-c1-20261030/result.json.

Claude owns the independent formal L1 draft; C1 supplies instrumentation parity
and development evidence for review. A future preregistered multi-seed experiment
must specify null validity/power and frozen behavioral retention independently.

## Observed C1 development outcome

Exact actual-core replay matched saved baseline.3431295071796494,
variance.18232238840335277 and mass.9158370097426899. All returned normalization
fields also match a real small LearningBrain test, not just a hand-coded EMA.

There were453varying and109constant credit columns. The searched positive-lag
maximum was r+.161698 at motor117575/180ms; negative-lag maxabsr.146616.499circular
reference maxima gave99th percentile.154905 and rank.002. These are development
descriptors, not calibrated significance or proof of a physical credit delay.
C2 preregisters this exact neuron/lag/sign for separate seed1001/1002 logs;
secondary peaks cannot replace that fixed-target replication result.

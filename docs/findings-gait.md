# Gait rhythm measurements

## Expectations recorded before simulation

A synthetic 10 Hz tripod should beat a shuffled-spike timing null, have negative
antagonist correlations, and score near 1 for tripod phase. Tonic independent
Poisson firing should not pass rhythmicity; co-contraction should correlate
positively. For the connectome, the hypothesis was tonic/coactive output with weak
or absent tripod coordination. Synapse correction might increase activity without
restoring gait. These expectations were printed before any simulation started.

## Measurement protocol

Only new analysis, experiment, test, and findings files were added; no model
parameters or implementation files were changed. Both raw BANC and corrected
FAFB-brain/MANC-cord weights use the existing loader and NT overrides. "Raw"
means no regional synapse scaling, not removal of other existing model assumptions.
Shared data, MANC references and processed caches were read without writing.

Each condition uses fresh networks with seeds 0, 1, 2, dt=0.5 ms, adaptation
increment da=1.0, background=3 Hz/neuron with weight=60, and duration=3000 ms.
There is no burn-in; all three seconds from network initialization are analyzed.
Leg motor annotations give 391 neurons. Rates use nonoverlapping 10 ms bins.
Frequency resolution is 1/3 Hz; the search covers positive FFT frequencies up to
50 Hz, without selecting a frequency band after seeing the results.

`rhythm_spectrum` reports the dominant frequency and periodogram power in Hz²
of the mean per-neuron pool rate. DC is removed. Each of 199 surrogates uniformly
redistributes spike times conditional on the original count, implemented exactly
as multinomial bin counts. The null statistic is the maximum non-DC power across
the entire frequency search. The rhythm flag requires exceeding its 95th
percentile and Monte Carlo p <= 0.05 (minimum p=0.005). Numerical ties are counted
conservatively, using a floating-point tolerance. This corrects for searching
frequencies within each pool, but does not adjust across pools or seeds.

`antagonist_alternation` uses MUSCLE_JOINT to construct positive and negative
pools per leg/joint and returns their Pearson rate correlation. Missing or
constant pools return null. Positive correlation means coincident fluctuations;
negative means opposing fluctuations. High constant activity can still coactivate
muscles without positive correlation, so this is a timing measure rather than a
muscle-force measurement.

`interleg_phase` uses homologous tibia-flexor pools, including accessory flexors.
Pooling antagonists together can hide a real oscillation, so spectra of all motor
neurons per leg are reported separately. Each leg's dominant phase is reported;
phase comparison uses a single shared frequency chosen by summed normalized leg
spectra. A tripod score is returned only if all six legs pass their own spectral
null and peak within one FFT bin of that frequency. The score is
abs(mean(exp(i*(observed phase - expected tripod phase)))), with LF/RM/LH at 0
and RF/LM/RH at pi. It is invariant to time origin. A score of 1 is ideal; a null
score means no valid common rhythm to compare. This validity gate is not a
statistical significance test of the phase score.

## Synthetic controls

Fixed independent Poisson draws around sinusoidal rates, 12 neurons per pool,
50 Hz baseline, modulation depth 0.95, 10 seconds, 10 Hz:

| Control | LF peak Hz | LF power / null95 (Hz²) | LF p | Antagonist correlation range | Tripod score |
|---|---:|---:|---:|---:|---:|
| Alternating tripod | 10.0 | 1132.269 / 7.449 | 0.005 | -0.729 to -0.712 | 0.999830 |
| Tonic Poisson | 24.2 | 4.440 / 7.540 | 0.885 | -0.043 to 0.022 | null |
| Co-contraction with tripod leg phases | 10.0 | 1132.269 / 7.449 | 0.005 | 0.700 to 0.739 | 0.999885 |
| All six legs synchronous, alternating antagonists | 10.0 | 1132.269 / 7.449 | 0.005 | -0.729 to -0.709 | 0.004567 |

Co-contraction can have a high leg-phase score: all three metrics must be considered
together. Poisson shot noise attenuates synthetic correlations below +/-1.
Additional controls cover missing/silent/constant pools, invalid inputs, a fixed
ensemble of stationary Poisson nulls, sparse FFT numerical ties, global time shifts,
and individually rhythmic legs with incompatible frequencies.

## Brain measurements

All six final simulations completed. Full per-joint correlations, firing rates,
spectral powers, null thresholds, p values, phases, and correction factors are in
[gait-rhythm-results.json](gait-rhythm-results.json).

| Brain | Seed | Total spikes | Motor spikes | Mean motor Hz | Median antagonist r | Positive / valid joints | Valid tripod |
|---|---:|---:|---:|---:|---:|---:|---|
| raw | 0 | 358,521 | 1,811 | 1.543905 | -0.031541 | 1 / 9 | No |
| raw | 1 | 1,634 | 11 | 0.009378 | null | 0 / 0 | No |
| raw | 2 | 1,642 | 6 | 0.005115 | null | 0 / 0 | No |
| corrected | 0 | 3,588,768 | 25,010 | 21.321398 | -0.195251 | 3 / 22 | No |
| corrected | 1 | 3,214,204 | 22,757 | 19.400682 | -0.189731 | 3 / 23 | No |
| corrected | 2 | 1,924,343 | 21,950 | 18.712702 | -0.134679 | 2 / 24 | No |

Pooled leg dominant frequencies in Hz (asterisk: passes the timing null):

| Brain | Seed | LF | LM | LH | RF | RM | RH |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0 | 0.333* | 0.333* | 0.333* | 0.333* | 0.333* | 0.333* |
| raw | 1 | 11.333 | null | null | 0.333 | 12.000 | 22.333 |
| raw | 2 | null | 0.667 | 7.333 | 19.667 | null | 4.333 |
| corrected | 0 | 2.333* | 1.000* | 0.667* | 1.000* | 1.667* | 2.000* |
| corrected | 1 | 2.000* | 1.333* | 0.333* | 1.333* | 1.333* | 2.333* |
| corrected | 2 | 2.000* | 2.000* | 1.333* | 2.000* | 1.667* | 3.000* |

Corrected tibia-flexor dominant frequencies (all six pools pass their null in
every corrected seed, p=0.005, but disagree in frequency):

| Seed | LF | LM | LH | RF | RM | RH |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.333 | 1.333 | 0.667 | 0.667 | 2.333 | 1.333 |
| 1 | 1.333 | 0.667 | 2.333 | 0.333 | 1.333 | 1.333 |
| 2 | 1.333 | 1.333 | 1.333 | 3.000 | 5.000 | 0.667 |

Mean motor rates (Hz/neuron) in successive 500 ms windows:

| Brain | Seed | 0–0.5 s | 0.5–1 s | 1–1.5 s | 1.5–2 s | 2–2.5 s | 2.5–3 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 0 | 0.010 | 0.000 | 0.010 | 2.609 | 3.412 | 3.223 |
| raw | 1 | 0.000 | 0.000 | 0.010 | 0.000 | 0.010 | 0.036 |
| raw | 2 | 0.010 | 0.000 | 0.005 | 0.000 | 0.015 | 0.000 |
| corrected | 0 | 18.747 | 27.499 | 20.077 | 19.719 | 24.916 | 16.972 |
| corrected | 1 | 12.752 | 17.381 | 20.046 | 21.944 | 23.514 | 20.767 |
| corrected | 2 | 16.235 | 20.972 | 21.069 | 18.890 | 15.013 | 20.097 |

The corrected network has substantial, non-Poisson motor timing but no valid
shared dominant tripod oscillation in these trials. Pooled leg peaks span
0.333–3 Hz, while homologous tibia-flexor peaks span 0.333–5 Hz across seeds.
All corrected pooled-leg spectral p values are 0.005. The raw seed-0 peak is
0.333 Hz in every leg (p=0.005), corresponding to just one period over the
recording. Activity turns on after roughly 1.5 s; this is evidence of strong
nonstationarity, not sufficient evidence of a recurring gait. Raw seeds 1 and 2
are almost silent and no pooled leg passes the null after the numerical-tie fix.

The anticipated predominantly positive antagonist correlation was not observed:
corrected medians are weakly negative, with only 2–3 positive joints per run.
This does not establish clean alternation or exclude coactivation. For example,
RM tibia flexor/extensor mean rates are 11.29/187.83, 12.18/174.00, and
7.92/124.83 Hz for seeds 0/1/2, with correlations +0.264, +0.094, and +0.171.
Strong firing imbalance and incompatible leg frequencies remain visible.

Validation: `pytest tests/test_gait.py -q` passed all 8 tests;
`ruff check src tests` passed. The final rerun reproduced all six preliminary
total and motor spike counts exactly. The sparse-timing numerical issue affected
null comparisons, not the simulated spikes, and is covered by a regression test.

## Reproduction

From the gait worktree in PowerShell:

```powershell
$env:PYTHONPATH='src'
& C:/dev/venvs/noesis/Scripts/python.exe -m pytest tests/test_gait.py -q
& C:/dev/venvs/noesis/Scripts/ruff.exe check src tests
& C:/dev/venvs/noesis/Scripts/python.exe -u -m noesis.experiments.gait_rhythm --data 'C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\data' --output docs/gait-rhythm-results.json
```

The output path must be new. The runner prints every metric and optionally writes
JSON, preserving completed runs if a later run fails. Missing processed caches
are built in `.gait-cache` in the worktree, never under source data. MANC must be
present and produce correction factors. Six equal-duration activity windows are
included to expose gross nonstationarity (500 ms windows at the default duration).

## Interpretation limits

A rejected uniform-time null means departure from stationary Poisson timing;
refractoriness, adaptation, common drive, or startup transients can all produce it.
It is not a test against shuffled interspike intervals and not evidence of walking
by itself. The phase analysis tests dominant tibia-flexor rhythms: it cannot exclude
a weaker shared frequency or coordination confined to other muscle pools.

The short initialization window and three seeds limit generalization to a
long-running awake brain. These simulations have no body or sensory feedback.
No force, actual steps, or behavioral gait is measured. Neither a weak negative
antagonist correlation nor a missing tripod score alone diagnoses the cause of
poor locomotion. Pool rates must be considered alongside the correlations: both
antagonists can be active on average while their fluctuations anticorrelate.

Source base: `a1b48c61b1ad9ad4885d7693b11363bd494608b6` on
`feature/gait-rhythm`. NumPy 2.0.2, SciPy 1.17.1. Corrected cache factor digest:
`a3e600af` (includes MANC). Raw/corrected matrices have 1,528,585 / 3,006,456
edges, respectively, over 158,262 neurons.

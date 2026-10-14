# Short-term depression on the fully corrected brain

## Predictions before runs (2026-10-14)

Base PR #13 (`feature/vnc-rescale`, 19190a9). Only synaptic depression changes;
NT signs, adaptation (1 mV), stimulus weights, region factors and the response
battery definitions remain fixed. PR #12 sign changes are not mixed into this test.

Hypothesis: finite presynaptic resources reduce sustained recurrent excitation,
allowing recovery after drive ends. Too-strong depression may also destroy taste
responses; silencing the network is not success. Depressing inhibitory output too
may oppose the intended effect. Uniform depression is a deliberately simplified
hypothesis, not a claim that all fly synapses have identical measured dynamics.

Mechanism: resource x starts at 1. Between events it recovers exactly as
x <- 1-(1-x)*exp(-elapsed/tau). Each recurrent transmission has amplitude x times
the original signed weight; afterwards x <- (1-U)*x. The first response retains
its original amplitude (weights already incorporate baseline release). Resources
are per presynaptic neuron, shared by its outgoing edges; external Poisson and
continuous currents are unchanged. U=0 disables the mechanism exactly.

Short-term depression is experimentally established at fly ORN->PN synapses
(Kazama & Wilson 2008, https://doi.org/10.1016/j.neuron.2008.02.030). Extrapolation
to the whole brain and the parameter grid below are modelling assumptions.

Predeclared screen: U in {0.1, 0.3, 0.6}, tau in {300, 1000} ms; unchanged U=0
control. Screen on seed 0; carry all candidates that preserve P1-P4/P7 and improve
P8 to seeds 1 and 2. If none qualifies, replicate the best recovery candidate that
preserves P1-P4/P7, or report no feasible candidate. Acceptance requires original
P8 (recovery active <= 2*rest+50) AND P1-P4/P7 across three seeds; P5/P6 reported.
Do not relax the acceptance test after seeing results. Report per-seed values.

Battery: fresh network per none/sugar/water/bitter/sugar+bitter/odour trial, 500 ms
rest, 1000 ms stimulus, 1000 ms recovery, dt .5 ms, stimulus weight 400, sugar/
water/bitter 100 Hz and odour 80 Hz. Original P8 counts *any* spike in the entire
recovery window; additionally record the last 500 ms to distinguish decaying
transients from a persistent lock. Late recovery is diagnostic, not a replacement
for P8. Controls: default-off exact historical spike equality, chunked-run equality,
resource depletion/recovery, signed transmission and zero-input silence.


## Screen outcome and replication decision

All six candidates are silent in the last 500 ms of recovery for seed 0, but
none passes the unchanged full-window P8 criterion. U=.1, tau=300 ms is the only
candidate preserving all of P1-P4/P7 on seed 0 and improving P8's worst count
(24,305 -> 10,520). It is therefore carried to seeds 1 and 2 alongside unchanged
controls, as specified before the screen. U=.1, tau=1000 and all stronger
candidates fail water and/or sugar criteria. No alternative acceptance threshold
is substituted. Whole-window transient firing and sustained late firing are
reported separately. Sugar->PAM is severely weakened by every candidate; this
is an additional limitation, even though P5 was not the stated acceptance gate.


## Diagnostic prediction before extended recovery

The three-seed candidate retains P1-P4/P7 in the original aggregate judge, but
seed 2 sugar+bitter has 338 neurons active in late recovery. Do not call every
trial recovered. Re-run that exact condition with unchanged parameters and
50 ms recovery bins for 5 seconds: predict a decaying transient, not a renewed
sustained state; report last-spike time and exact agreement in the original
1-second recovery window. This diagnostic does not replace P8 or change gain.

## Results

66 fresh-network trials: the seven-condition parameter screen (including no-depression control) on seed 0, plus control and selected U=.1/tau=300 replicated on seeds 1 and 2. Each parameter condition contains six stimulus conditions. Selection used seed 0 only. [All per-seed measurements](depression-results/trials.csv), [summary and unchanged P1-P8 verdicts](depression-results/summary.json), [protocol, input hashes and factors](depression-results/protocol.json).

**Acceptance not met.** U=.1/tau=300 preserves aggregate P1-P4/P7, but original P8 still fails (worst recovery mean 8,803 active vs zero rest, allowed 50). P5 sugar-to-PAM also fails. Keep depression disabled by default: the result supports further controlled modelling, not a healthy calibrated fly.

### Seed-0 parameter screen

| U | Recovery tau ms | P1/P2/P3/P4/P7 | Worst full recovery active | Maximum late recovery active | Sugar MN9 Hz | Water MN9 Hz | Sugar PAM Hz |
|---|---|---|---|---|---|---|---|
| 0 | 300 | pass/fail/pass/fail/pass | 24305 | 22057 | 10.50 | 2.00 | 1.0107 |
| 0.1 | 300 | pass/pass/pass/pass/pass | 10520 | 0 | 7.50 | 3.00 | 0.0249 |
| 0.1 | 1000 | pass/fail/pass/pass/pass | 654 | 0 | 5.50 | 1.50 | 0.0214 |
| 0.3 | 300 | pass/fail/pass/pass/pass | 171 | 0 | 2.50 | 1.50 | 0.0000 |
| 0.3 | 1000 | fail/fail/pass/pass/pass | 163 | 0 | 2.00 | 1.50 | 0.0000 |
| 0.6 | 300 | fail/fail/pass/pass/pass | 1346 | 0 | 1.50 | 1.00 | 0.0000 |
| 0.6 | 1000 | fail/fail/pass/pass/pass | 164 | 0 | 1.50 | 1.00 | 0.0000 |

Every screen candidate fails full-window P8 and P5. All have zero late recovery spikes on seed 0. Stronger depression compromises sugar/water responses instead of satisfying acceptance. The seed-0 baseline itself fails P2/P4; these pass after averaging the prescribed three seeds. Individual seed variation must not be hidden.

### Three-seed control versus U=.1/tau=300

Original verdicts use the mean over seeds 0, 1, 2, as in response_battery. These are aggregate passes, not a guarantee that every individual seed passes. Values below are per-seed lists in that order.

| Condition | MN9 Hz control | MN9 Hz depression | Full recovery active control | Full recovery active depression | Late recovery active control | Late recovery active depression |
|---|---|---|---|---|---|---|
| none | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| sugar | 10.5, 10, 12.5 | 7.5, 8.5, 10 | 19028, 19612, 23680 | 1190, 506, 174 | 16890, 17411, 22241 | 0, 0, 0 |
| water | 2, 10, 3.5 | 3, 4.5, 2 | 19604, 18987, 18564 | 2669, 25, 7472 | 18104, 17454, 16255 | 0, 0, 0 |
| bitter | 0, 0, 0.5 | 0, 1.5, 1 | 23446, 18995, 18339 | 1137, 207, 67 | 22057, 16811, 16068 | 0, 0, 0 |
| sugar+bitter | 14.5, 2.5, 6 | 4, 3.5, 4 | 18231, 18912, 19449 | 6983, 467, 7989 | 16671, 16396, 17623 | 0, 0, 338 |
| odour | 0, 0, 3 | 4, 3, 1.5 | 24305, 24711, 24402 | 10520, 9083, 6805 | 21967, 23317, 22470 | 0, 0, 0 |

| Criterion | Control | Depression | Depression detail |
|---|---|---|---|
| P1 sugar -> MN9 | PASS | PASS | 0.0 -> 8.7 Hz |
| P2 water -> MN9 | PASS | PASS | 0.0 -> 3.2 Hz |
| P3 bitter < half of sugar at MN9 | PASS | PASS | bitter 0.8 vs sugar 8.7 Hz |
| P4 sugar+bitter < sugar at MN9 | PASS | PASS | 3.8 vs 8.7 Hz |
| P5 sugar -> PAM | PASS | FAIL | 0.00 -> 0.03 Hz |
| P6 bitter -> PPL1 more than sugar | FAIL | PASS | bitter 8.79 vs sugar 6.95 Hz |
| P7 odour: own glomeruli > others | PASS | PASS | on 120.6 vs off 14.9 Hz |
| P8 recovery after stimulus | FAIL | FAIL | worst: odour 8803 active vs rest 0 |

Descriptive means +/- sample SD across three simulation seeds (not biological replication):

| Metric | Control | Depression |
|---|---|---|
| sugar / mn9 | 11.0000 +/- 1.3229 [10.0000, 12.5000] | 8.6667 +/- 1.2583 [7.5000, 10.0000] |
| water / mn9 | 5.1667 +/- 4.2525 [2.0000, 10.0000] | 3.1667 +/- 1.2583 [2.0000, 4.5000] |
| bitter / mn9 | 0.1667 +/- 0.2887 [0.0000, 0.5000] | 0.8333 +/- 0.7638 [0.0000, 1.5000] |
| sugar+bitter / mn9 | 7.6667 +/- 6.1712 [2.5000, 14.5000] | 3.8333 +/- 0.2887 [3.5000, 4.0000] |
| sugar / pam | 1.3760 +/- 0.5372 [1.0107, 1.9929] | 0.0285 +/- 0.0062 [0.0249, 0.0356] |
| bitter / ppl1 | 80.2564 +/- 7.6701 [75.1538, 89.0769] | 8.7949 +/- 3.0424 [7.0000, 12.3077] |
| sugar / ppl1 | 80.6154 +/- 8.4626 [71.3077, 87.8462] | 6.9487 +/- 0.3795 [6.6923, 7.3846] |
| odour / pn_on | 263.9744 +/- 0.9672 [262.8846, 264.7308] | 120.5769 +/- 0.5385 [119.9615, 120.9615] |
| odour / pn_off | 118.3056 +/- 0.1862 [118.1367, 118.5052] | 14.9014 +/- 0.4985 [14.4012, 15.3982] |
| odour / recovery_active | 24472.6667 +/- 212.0244 [24305.0000, 24711.0000] | 8802.6667 +/- 1873.2982 [6805.0000, 10520.0000] |

### Persistent state versus offset transient

Unchanged control exactly reproduces the handoff averages: sugar MN9 11.0 Hz, water 5.17 Hz, sugar+bitter 7.67 Hz and worst recovery 24,472.67 neurons. Its late recovery remains active in all stimulated trials. With the selected depression candidate, 14/15 stimulated trials have no spikes during the final 500 ms of recovery; seed-2 sugar+bitter has 338 late-active neurons. A no-change rerun of that trial with 50 ms bins and five seconds of recovery exactly reproduces both original recovery-window counts; its last spike is **537.5 ms after stimulus offset**, with no further spikes through 5 seconds. [Extended trace](depression-results/extended-recovery.json).

This supports termination of reverberation in the tested stimulus trials while distinguishing it from the original P8 metric, which counts any cell spiking anywhere in the first second after offset. We do not relabel P8 as passing. No awake-background, body, gait, KC-sparseness or learning benefit has been demonstrated by this experiment. Sugar-to-PAM falls from 1.38 Hz to 0.0285 Hz: uniform depression is not a substitute for modelling slow monoamines, and enabling it globally now would undermine the reward pathway.

### Implementation and validation

Added opt-in LIFParams.depression_u and depression_tau_ms. Recovery is exact between delivery events and state is retained across run boundaries. First transmission keeps resting strength; the existing baseline release probability is not multiplied by U a second time. All signed outgoing weights share their presynaptic resource; inhibitory and monoaminergic rows are depressed too. That uniformity is an explicit hypothesis. No transmitter signs, region factors, synapse counts, motor mappings or response_battery predicates changed.

47 pytest tests and ruff check pass. Ten new tests cover analytic depletion/recovery, bounds, reduced repeated transmission with identical external drive, chunk/delay continuity, no-input silence, inhibitory sign and parameter validation. U=0 additionally reproduces the historical PR #13 kernel exactly on a recurrent signed test network, including background input, for seeds 0/1/2 (5,240/5,375/5,320 spikes). [Historical control](depression-results/legacy-control.json). Main and Claude checkouts were untouched; caches and temporary scripts stay in the isolated C:/dev/noesis-codex-depression worktree.

### Reproduction

```powershell
$env:PYTHONPATH='src'
$py='C:/dev/venvs/noesis/Scripts/python.exe'
$data='<main checkout>/data'
& $py -m noesis.experiments.depression_battery --data $data --output outputs/depression --workers 3
& $py -m noesis.experiments.depression_battery --data $data --output outputs/depression --u .1 --tau-ms 300 --seeds 1 2 --workers 3
& $py -m noesis.experiments.depression_battery --data $data --output outputs/depression --u .1 --tau-ms 300 --seeds 2 --trace-recovery sugar+bitter
& $py -m ruff check --no-cache .
& $py -m pytest -q
```
The experiment imports the original groups and judge from response_battery; it does not edit that file. A fresh network is used per trial, no external reward is introduced, and both whole-window and late recovery measurements are retained. This branch is stacked directly on PR #13; PR #12 olfactory sign correction is kept separate to avoid mixing interventions.

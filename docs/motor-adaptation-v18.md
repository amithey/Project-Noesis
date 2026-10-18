# Motor-neuron adaptation: prospective neural screen

V17 failed its complete frozen seed-10 evaluation: standing 66.869% versus
original 92.222%, followed by an unrecovered fall at 3.660 seconds. Motor firing
continued while inverted. Tibia flexor/extensor pool means remain strongly
imbalanced (RM 9.566/279.495 Hz). This motivates a hypothesis, not a causal
finding: stronger spike-triggered adaptation in mapped leg motor neurons may
improve their response recovery without damaging sensory processing.

The isolated prototype changes each neuron's adaptation increment after its
own spike. Mapped leg motor neurons receive multipliers 1, 2 or 4; other neurons
retain da=1. The 200 ms decay and U=0 remain. Original graph edges and signed
weights are unchanged; there is no gait clock, leg sequence or joint target.
The production source remains exact V17 (839a0a8bd85a8ebca55048319dc95642413491a9a14c5d168fe24cc4d6e8d107).
The prototype is not installed or used for body training.

Sixteen numerical tests verify exact original/default spike events and defined
states on synthetic networks, targeted adaptation increments, input copying,
independent-neuron suppression and continuity across chunks. Thirteen further
tests reject incomplete/mismatched trials, sensory harm, weight drift, uniform
silencing, worse global recovery and changed resume source/configuration.
All 29 tests and Ruff passed before launching the full-network experiment.
These tests do not establish full-connectome default equivalence or gait.

The fixed screen has 72 trials: original and final V17 weights, multipliers
1/2/4, seeds 63/64, and none/sugar/water/bitter/sugar+bitter/odour conditions.
Each fresh neural trial uses 500 ms rest, 1,000 ms stimulus and 1,000 ms recovery.
There is no learning, motor exploration or body. Completed trials must retain
identical weights for each weight model; partial trials restart on resume.
Source, prototype, driver, plan, checkpoint and loaded brain/annotation hashes
are guarded, with a one-hour absolute process deadline.

The original P1-P8 judge is unchanged and every verdict is reported. Both
weight models must preserve every baseline-passing assay, avoid worsening
mean recovery active-neuron counts in each condition, lower mean motor recovery
firing by 20%, lower its recovery/stimulus ratio by 10%, and retain at least
50% of each nonzero baseline stimulus response. Global P8 remains explicitly
reported if still failing; localized motor recovery does not solve global
recovery by definition. Choose 2 if it passes on both models, else 4 if it does;
otherwise select none. No selection occurs before all 72 complete trials.

Independent reconstruction from the read-only sparse cache verifies the full
158,262-neuron graph, 73,594 existing plastic edges and 498 learner motor indices.
Original full-weight SHA256:
936b5bd40c5e448c05771b1be8f640b8ac5f31fa69d4e96ba7a11799cf59b4f8
Expected trained full-weight SHA256:
78b4d60d746f3b6675da63c878e2a4006c0573aa14473bd1481dc4d85b2fd284
This check does not call MotorPlasticity.load. Runtime weight hashes are checked
against these independent expectations; original signs/topology/bounds match
the retained final V17 checkpoint (8755d9888de39337cf4286526a92ca6568d38bca8deb9e965f16cb0f5a1713fc).

The first fourteen complete trials confirm the original-weight baseline and
the start of multiplier 2. Stimulus responses and persistent post-stimulus
activity are observed; there is no final screen verdict yet. The runtime report
is authoritative for later progress. Any selected candidate still requires
production integration, checkpoint compatibility, full regression checks and
physical standing/walking/obstacle/held-out validation. No best model exists;
normal startup remains unchanged and learned walking remains unproven.

## Report-replacement interruption and unchanged-source resume

The first worker exited with WinError 5 during atomic report replacement
before starting trial 17. All sixteen complete trials match both committed
and temporary ledgers and remain preserved. The file was not read-only.
A real Windows test confirmed that a normal reader can block replacement;
the attempted delete-sharing reader also failed its replacement test and
was discarded, not deployed. Its failing regression probe/log are archived
under observer-sharing-experiment. The actual handle owner is unproven.

The active observer now refuses to open the ledger without --worker-stopped.
While a worker is live, use only its process stdout; copy/verify the report
after confirmed termination. The no-flag refusal was exercised successfully.
The original 29 neural/protocol tests still pass. Neither driver nor neural
prototype nor selection plan changed. Resume under the exact same settings
retained sixteen trials, and stdout subsequently confirmed trials 17..19.
No result is selected until all 72 trials complete. Source SHA guards remain
strict; the operational failure is not a negative neural result.

## Original-model half complete; trained-weight import observed

Process stdout confirms trials 1..36 for the original model and trials 37/38
for the trained model. Every observed record retains a 2,500 ms horizon,
learning/exploration off and unchanged weights. The trained full-network
SHA256 matches the independent cache/checkpoint reconstruction
(78b4d60d746f3b6675da63c878e2a4006c0573aa14473bd1481dc4d85b2fd284).
This verifies the intended weight transfer, not sensory preservation, recovery
improvement or gait. The active ledger was not opened; full-report verification
remains deferred until worker termination. No candidate has been selected.

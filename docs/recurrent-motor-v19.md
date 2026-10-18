# Neuron-resolved motor-input diagnostic

Twelve full-network recordings completed: original/final V17 weights, production
scalar/prototype default kernels, and sugar seed63/odour seed64/awake seed65.
All six kernel pairs match byte-for-byte in their full spike-stream digests,
per-neuron window counts and defined final numerical states. Sugar/odour count
summaries reproduce the corresponding completed V18 scale-1 trials exactly.
Original/trained full-weight hashes remain verified, with no learning or body.
Ten recording tests and Ruff passed before launch. Production source remains
839a0a8bd85a8ebca55048319dc95642413491a9a14c5d168fe24cc4d6e8d107.

The late direct positive-input proxies repeatedly involve unclassified,
single_leg_neuromere, descending_neuron and sequential_leg_neuromeres sources.
In the trained awake case these classes account for approximately 21.3%,
26.1%, 21.9% and 17.3% of the positive weighted-spike input proxy. These are
weighted presynaptic spike sums over existing edges, not measured membrane
currents or proof that a population caused the fall. Anatomy and activity
alone do not justify altering a chosen population's physiology.

The awake component probe has continuing 3 Hz x60 background, no body and no
sensory feedback; its late windows are not recovery tests. In its last 250 ms,
mapped motor activity is 14.087 Hz (original) and 22.787 Hz (trained), versus
22.489/22.856 Hz across the prior full body neural windows. Per-neuron rate
correlations are 0.714/0.688. Different windows and feedback make these
comparisons descriptive; similar rates do not quantify causal contribution.

The body recorder includes all 391 Class=leg_motor_neuron cells; the mapped
actuator/diagnostic set contains 389. The comparison therefore explicitly
filters to those same 389 cells and retains the excluded counts (186/230
body events at indices 34272/51504). An initial assumption that all recorded
leg events were mapped was rejected. No source data or acceptance criterion
was changed to align the populations.

All recordings, hashes, negative controls and reference comparisons are
preserved. No new dynamics, learned weights or gait controller were selected.
The next experiment directly probes body coupling under lower background
input gain, using existing model arguments and frozen original/trained weights.
This does not install the rejected V18 adaptation candidate. It is not training
or final validation, and cannot promote a model or replace the startup bank.

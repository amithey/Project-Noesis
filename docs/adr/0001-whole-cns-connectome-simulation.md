# ADR-0001: Simulate the whole-CNS FlyWire connectome with an event-driven LIF model

- Status: accepted
- Date: 2026-10-13

## Context

Goal: an embodied fly whose behaviour is driven by the real FlyWire connectome.
The downloaded tables (`neurons.csv`, `connections_princeton.csv`) cover **158,262
neurons including the ventral nerve cord**: 1,316 descending neurons, 12,851 VNC
intrinsic neurons and 805 motor neurons (391 to the legs). Leg motor neurons receive
~60k edges from 7,415 neurons, so the path brain -> descending -> VNC -> leg motor
neuron is present in the data. (An earlier assumption that a separate motor controller
would be required was wrong.)

The connectome gives wiring only: no synaptic dynamics and no learning rule.

## Decision

1. Model every neuron as a leaky integrate-and-fire unit with the parameters of
   Shiu et al., *Nature* 2024 (tau_m 20 ms, tau_syn 5 ms, threshold -45 mV,
   0.275 mV per synapse, 1.8 ms delay, 2.2 ms refractory, dt 0.1 ms).
2. Synaptic weight = synapse count (pairs aggregated over neuropils, pairs below
   5 synapses dropped) x sign of the presynaptic neuron's predicted neurotransmitter.
   Assumptions, all configurable: ACh excitatory; GABA/Glu/histamine inhibitory;
   monoamines and unknown NT treated as excitatory.
3. Event-driven numba kernel on CPU (no NVIDIA GPU available); raw data stays
   outside git (`data/`, CC-BY-NC licence of FlyWire).
4. Validate against the published sugar-GRN -> MN9 (proboscis) result before
   attaching a body.

## Consequences

- ~2 s of wall-clock per 500 ms of simulated whole-CNS activity on a laptop CPU.
- Drive strength is a free parameter (Poisson weight); calibrated so sugar neurons
  fire at the 200-300 Hz regime where MN9 activates. At ~100 Hz the signal dies out
  after two hops.
- Learning ("being born") will have to be added on top, e.g. plasticity in selected
  populations or a trained controller; it does not come from the data.

# Read-only proprioceptive interface audit

The active V12 source remains unchanged. This diagnoses possible input errors;
it does not demonstrate a causal explanation for falling or a learned behavior.

The current hook branch in sensory.py is a sigmoid of negative tibia position,
independent of tibia velocity. Three positions and three velocities reproduce
identical rates across movement directions at each position; rest gives 76 Hz.
Mamiya et al. 2018, Figure 5, instead reports phasic direction-selective movement
responses and no tonic hold activity in the sampled hook line. Later 2023 work
distinguishes flexion- and extension-sensitive hook subtypes.

Sources: https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/ ;
https://doi.org/10.1016/j.neuron.2023.07.009 .

The actual mapped population has 142 hook-labelled and 162 claw-labelled afferents.
Hook primary types include SNpp39 (81) and SNpp41 (46); claw types include SNpp50
(99) and SNpp51 (44), with some primary/subclass conflicts. Inspected Primary and
Alternative labels identify types but do not directly label preferred movement
direction. The MANC annotation paper, Figure 59, predicts opposing tibia-motor
effects of those pairs, but this is not direct physiological tuning. A resistance-
reflex direction assignment would be an inference and needs an opposite-assignment
control; arbitrary row-parity or random preference assignments are not justified.

Source: https://elifesciences.org/reviewed-preprints/97766 .

Recorded positive tarsal loads have medians around 41..43 in simulator force units.
With the current load scale 2, force-bearing rates average 141..148 Hz versus
air baseline 2 Hz. Thus load feedback is not missing; it mostly behaves as a
contact detector with weak dynamic range for stance force. Whether adhesion or
other forces explain this distribution has not been measured, so no subtraction
of adhesive forces or retuning of the load scale is justified by this audit alone.

Reproducible read-only driver: outputs/proprioception-audit-v12/diagnose.py. Full
counts, stimuli, force quantiles and unchanged active source hash are in the JSON.
Complete the active frozen candidate evaluation before changing any sensory code.
A future candidate should distinguish position (claw) from directional movement
(hook), preserve annotated neuron identities, expose tuning uncertainty, and keep
all physical acceptance criteria and fresh matched controls.

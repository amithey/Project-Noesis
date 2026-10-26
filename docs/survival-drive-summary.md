# Survival-drive track: final summary and coordination plan — 2026-10-17

Goal: train the connectome-driven fly brain to work with its senses so a virtual fly shows fly-like,
need-driven behaviour. Scope: everything that does not need legs (leg control belongs to the locomotion track,
`codex/brain-body-feedback`). Branches (both on origin): `feature/fly-survival-drive`, `feature/network-calibration`
(built on the first). Per-result detail: `findings-sense-audit.md`, `findings-foraging-conditioning.md`,
`findings-virtual-agent.md`, `findings-network-calibration.md`, `findings-pn-decoder-agent.md`.

## What was established (all brain-only or leg-free; each with shuffle/random/reversal controls)
1. **The shared network is bistable.** With the corrected BANC brain, background ≤ 1 Hz (weight 60) is silent
   and ≥ 2 Hz is saturated (PN ≈ 115 Hz, 17-35% of neurons active); a sparse window exists only at 1.25-1.75 Hz. In
   that state any strong receptor drive gives the same population response as a random receptor set, so
   population means carry no odour identity, valence or side.
2. **Odour identity is present in a modest, structured form upstream.** With graded APL: 7 of 10 tested
   glomeruli drive their own projection neurons above random PN sets (p < 0.05), and the Kenyon-cell code of the
   same odour is more reproducible than between odours (identity index +0.14, the smallest attainable p) — but ≈ 74%
   of two odours' Kenyon-cell codes still overlap.
3. **It does not reach the descending output as odour identity** (index ≈ 0.001); a small ipsilateral side
   difference of the numbered DNa family (+7.5 to +8.7 Hz, Bonferroni-significant, survives equal receptor counts
   and disappears with shuffled sides) exists but is one-sided and ≈ 100 Hz-saturated.
4. **A larger inhibition factor produces a sparser, more specific network.** Multiplying all inhibitory weights by
   `g_I` = 4-12 lowers non-target PN rates from ≈ 120 to 8-40 Hz, Kenyon-cell activity from 26% to 8-13%, raises the
   Kenyon identity index to +0.25 and gives descending neurons a side-specific pattern (p = 0.01-0.02) — but
   silences the DNa group and removes the DNa lateralisation.
5. **Side information is linearly decodable and spread through the network:** 78-85% on odour sets never seen,
   equally from descending, projection and random central neurons.
6. **In real time, graded bilateral input is decodable from projection neurons, not descending neurons:** 72% on
   unseen concentration levels (250 ms windows; null 50%, p = 0.016), 61-68% at plume-like contrast, 77-81% at
   high contrast; 1 s averaging is worse; descending neurons at chance.
7. **A leg-free agent steered by a trained PN read-out reaches the food:** in the wide-antenna world (3 mm) 9 of 12
   poses versus 0-1 of 12 in every control (shuffled-label decoder, reversed sign, shuffled receptor sides, no
   gradient, diffusion only; Holm p ≤ 0.003).

## What was NOT shown
- **No learning of any kind inside the network.** Odour+sugar conditioning with fed/hungry states and a direct
  dopamine positive control produced no sugar- or hunger-dependent change; sugar reaches no dopaminergic type
  selectively (0 of 33), and the graph shows no sugar-specific route to reward cells.
- **No evidence that hunger changes behaviour** (wide world 9 vs 5 of 12, p = 0.107 uncorrected; standard 3 vs 4).
- **No odour recognition, no odour-dependent choice, no memory, no evolution experiment.**
- **The brain does not steer by itself.** The agent uses a read-out trained on labelled side; projection neurons
  are one synapse from the receptors; it works only after changing the world (3 mm antennae), and not in the
  standard 1 mm world (3 of 12, indistinguishable from controls).
- **`g_I` is a model assumption**, a 4-12-fold departure from the measured weights, justified only by the
  resulting operating regime; the true cause of saturation (missing synapses, no neuromodulation, neuron model) is
  untested. The first agent (DNa steering on the unscaled network) did not reach the food (0 of 4).
- Single network calibration, small pose counts (12), no body, no real fly data comparison.
- A significance-count error in an earlier table (6 vs the correct 7 glomeruli) was found and corrected; the p
  floor of several permutation nulls is the attainable minimum, noted where it occurs.

## Coordination with the locomotion track (proposal)
Facts: the locomotion branch (V14, 2026-10-16) stands ≈ 89.5% of 5 s, no promoted candidate, tripod score
unavailable; its brain runs with background 3 Hz × 60, which in this repo's operating-point scan on the corrected
brain at `g_I` = 1 is the **saturated** regime (17% active, PN ≈ 116 Hz). It was validated at `g_I` = 1.

1. **Do not change the shared brain.** `g_I` stays an opt-in parameter (default 1.0, `scale_inhibition` in
   `ei_balance_scan.py`); nothing merged into the locomotion branch changes its default dynamics.
2. **Step 2 done (brain only, `findings-network-calibration.md` section 7):** at `g_I` = 4-12 leg motor neurons are not silenced
   but become input-driven (0.35-1.6 Hz at rest versus 21 Hz tonic at `g_I` = 1), the proprioceptive reflex survives at the same sign
   and similar size, and the motor pools respond more strongly to injected current. So a body test needs a re-scaled rate-to-activation map.
   **Step 3 done (`findings-body-ab-g-inh.md`): `g_I` = 4 is not a drop-in for the body** — with the unscaled actuator the body stands but slides
   backward ≈ 50 mm; with activation re-matched it stands unreliably (0.95 / 0.41) and still drifts backward; the baseline itself varies
   0.17-0.95 between two seeds, so standing differences are not interpretable; heading is the only comparison beyond that spread.
   Original proposal for step 3, retained for the record (A/B on the locomotion pipeline, frozen weights, same seeds): run its standing protocol (V14 acceptance
   gates: standing ≥ 90%, heading progress ≥ 1 mm, two cycles in four legs, slip ≥ 20% lower than original weights)
   at `g_I` ∈ {1, 4, 12}. Before any body run, measure leg-motor and proprioceptive-pool rates at those `g_I`
   (DN activity falls to 6-17%; motor pools may fall silent), reusing `ei_balance_scan.py`/`operating_point.py`.
   If standing worsens or motor pools go silent at `g_I` > 1, the two tracks need different calibrations (e.g.
   scale only the olfactory/central inhibitory populations, not motor circuits) — an open design question.
3. **No agent-to-leg connection until walking passes its own gates.** The agent's decision comes from a PN
   read-out, which is not a descending command. The interface to the body would be an assumed turn bias (an
   asymmetric bias into left/right leg motor pools, or a DNa-group input) and must be labelled as such, like the
   existing artificial body reward.
4. **Process:** keep the two lines on separate branches; merge `feature/network-calibration` only after step 2;
   run both test suites after any merge (257 tests on the locomotion branch, this repo's added tests on top);
   write new results as dated `findings-*.md`; never overwrite the other track's checkpoints or outputs; use
   one shared data directory read-only.
5. **Decision for the user:** whether to spend the next effort on (a) the A/B above, (b) an evolution experiment
   on the leg-free agent (genome: sensory gains, hunger coupling, read-out gain; fitness: reach/energy), or (c)
   making hunger matter (e.g. making food scarce and energy loss real) — (a) is the prerequisite for any joint
   use; (b) and (c) are self-contained.

## Update 2026-10-26: where the sensory track ends (the reward-only read-out is the last experiment of this line)
Added since the summary above: inhibition scaling `g_I` (findings-network-calibration.md), side decoding and the label-trained read-out
(9 of 12 poses reach the food with 3 mm antennae, 3 of 12 with 1 mm), the body A/B showing `g_I` = 4 is not a drop-in for the legs,
and a read-out trained from reward only (reward-readout.md).

What the evidence supports:
1. The scanned network is **bistable and saturates**; population means carry no odour identity. Odour identity exists upstream (projection
   neurons, Kenyon cells), modestly; it is absent from the descending output; the DNa lateralisation seen at `g_I` = 1 disappears at higher `g_I`.
2. A sparser, more selective regime exists only after scaling **all** inhibition by 4-12, which is a model assumption (and breaks the body).
3. **Sugar reaches no dopamine cell type selectively** (0 of 33), so associative learning from sugar failed and hunger changed nothing reliably, as
   a receptor gain or as a reward multiplier.
4. Side information is decodable from projection neurons (not descending neurons). A linear read-out trained WITH side labels steers a leg-free
   point agent (9 of 12 vs 0-1 in controls) only with wide antennae. A read-out trained from reward alone learned on recorded data but did not
   reliably steer the real network (2 and 3 of 12 vs 1 for its shuffled-return control; 48 real episodes per arm).
5. **No learning inside the network has been demonstrated anywhere;** every working steering result is an external, supervised interface on
   projection-neuron activity.

Judgement: the sensory track is limited by the brain model (missing or undetected structure and its dynamics), not by the training procedure.
Recommended next step (done in part, see `connectome-audit-banc-vs-fafb.md`): a graph-level audit of the same pathways in the second connectome that is on disk (FAFB, `data/fafb`) versus the one
used (BANC): sugar receptors -> feeding/dopamine cells, odour receptors -> projection neurons -> Kenyon cells -> output neurons, and
neurotransmitter/sign assignments. No simulation is needed. If FAFB shows the pathways BANC lacks, undetected synapses are the likely cause of
points 1 and 3, and a brain built from FAFB-level counts is the experiment to run.
Note on data: the FAFB neuron/classification tables that were in the untracked `FABA/` folder were removed during an earlier clean-up (they
were judged duplicates; some were not); they can be re-downloaded from FlyWire or may be restorable from the OneDrive recycle bin. The audit
can start with what remains (`data/fafb/connections_princeton.csv.gz`, `consolidated_cell_types.csv.gz`).


### Result of that audit (2026-10-26)
The mushroom-body and dopamine pathways are 10-300x sparser in BANC than in FAFB (KC->MBON ~40x per Kenyon cell, PAM->KC 2 vs 1,084 pairs, PPL1->KC 5 vs 2,104). This was already
known from the unmerged branch `feature/mb-from-fafb` (2026-10-13), which also contains a FAFB mushroom-body transplant; the sensory experiments above were run without it.
All conclusions about learning and dopamine in this document should be read as limited by the BANC table, and repeated on a FAFB-based mushroom body before being relied on.

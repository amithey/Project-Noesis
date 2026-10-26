# Project Noesis: goal, status, and who does what (alignment note, 2026-10-26)

**TL;DR (עברית):** המטרה: מוח של זבוב אמיתי (קונקטום סרוק, 158 אלף נוירונים) מחובר לגוף מדומה, ואנחנו רוצים שהמוח ילמד להשתמש בגוף ובחושים
כמו זבוב אמיתי. שלושה סוכנים עובדים במקביל על חלקים שונים; הכלל המנחה: מודדים לפני שטוענים.

## 1. The goal
A Drosophila connectome was scanned (BANC, FlyWire format; 158,262 neurons, ~3M synaptic pairs; synapse counts rescaled with FAFB/MANC).
We turned it into a spiking network (leaky integrate-and-fire) and attached it to a physical fly body (NeuroMechFly in MuJoCo) living in a
simulated world with food, water, bitter patches, odours, vision, touch and temperature. **The goal: the fly's own brain learns to use its
body and senses the way a real fly does** — first to stand and walk, then to sense the world and act on it (find food, avoid bad things),
driven by internal need (hunger). We do not hand-program gaits or behaviours; assumed components (muscle model, sensory mappings, training
reward) are labelled as assumptions. Success is judged by measurements with controls, never by a good-looking single run.

## 2. The two halves and their present state (all numbers measured; see `docs/` for the evidence)
**A. Locomotion (brain -> body).** The frozen network (no learning) drives six legs through agonist/antagonist motor-neuron pools and a muscle
model; legs feed back proprioception and balance. State: the fly stands and makes stance/lift cycles, moves forward ~7-12 mm in 5 s, but
**whether it stays upright depends mostly on the random seed**, not on training: standing fraction of the *untrained* weights over seeds 10-19 is
0.33-0.95 (mean 0.74, sd 0.24). About 13 training cycles, 12 more births on fresh seeds, soft starts, and an evolved 84-parameter pool-gain genome
did not beat the untrained weights in a paired 10-seed comparison (differences -0.10 and +0.01, intervals include 0). Falls happen anywhere in
the 5 s and show no precursor in averaged signals (`early-fall-analysis.md`). **No walking is established.**
**B. Senses and behaviour (world -> brain -> action), no legs.** Findings: the network is bistable (silent or saturated); sugar reaches no dopamine
cell type selectively, so sugar-driven learning failed; odour identity exists upstream (projection neurons, Kenyon cells) in modest form but not in
the descending output; scaling all inhibition x12 gives a sparser, more odour-specific regime (an assumption, not a measurement); a *trained*
linear read-out of projection neurons steers a leg-free point agent to food in 9/12 starts vs 0-1/12 in all controls — only in a world with wide
antennae, and it is a trained interface rather than the brain steering itself. Hunger had no reliable effect.
**Shared root issue:** the network's operating point (bistable, saturated at the default background drive) and seed-dependent chaotic dynamics limit
both halves. Changing it (`g_I` x4) did not transfer to the body: standing became unreliable and the body slid backward.

## 3. Roles
| Agent | Focus | Branches / places |
|---|---|---|
| **GPT 6 (Codex)** | Locomotion learning pipeline: reward design, perturbation learning of motor-circuit synapses, validation protocol (V12 ... V28) | `codex/brain-body-feedback` in its own worktree |
| **Claude (Anthropic)** | Senses and survival-drive track (sense audits, hunger, odour/steering agent, network calibration); the **shared evaluation tools** (paired 10-seed evaluation, frozen probe, agent statistics); coordination and honest reporting | `feature/fly-survival-drive`, `feature/network-calibration`, `claude/locomotion-v25` |
| **Gemini 4 Argon** | **Walking.** Independent root-cause analysis of why the body falls and why training does not generalise, concrete low-cost experiments, and bounded code proposals for body/control | own branch `gemini/locomotion-review`; read-only on the others |
| **The user** | Decisions, priorities, connects the agents, runs/approves anything heavy | — |
Why Argon on walking: standing/walking is the largest unsolved blocker for the whole goal (no stable body control, no behaviour), it has absorbed the
most effort without a robust result, and a fresh, independent view of the body/control code is more useful there than a third pair of hands on senses.
The senses track is in a state where its next step depends on decisions about the network, which Claude holds the context for.

## 4. Rules for everyone
1. **Measure before claiming.** Compare with the untrained weights, paired by seed, over the development seeds 10-19 (`scripts/dev_eval.py`, bootstrap
   interval; pre-state the criterion). One seed is never evidence. State what was *not* verified.
2. **Seeds:** development 10-19; **held out 200-202 are never used for development**; training seeds 80-111 and 1000-1999 (evolution) are not reused for evaluation.
3. **Branches:** each agent writes only in its own branch/worktree; read others' freely; never overwrite another agent's outputs or checkpoints.
4. **Resources:** one Windows machine, ~2-3 GB free RAM, two heavy episodes at a time at most (one 5 s episode takes 5-14 min). Announce heavy runs to the user first.
5. **Data and secrets:** the raw connectome tables (non-commercial licence) and any credentials are never read into chat, copied, or transmitted.
6. **Defaults stay unchanged:** new behaviour is opt-in (flags); the registered protocol gates (standing, cycles, forward progress, slip, obstacle response) are not edited to pass.
7. **Write results down** as dated documents in `docs/`, negative results included; commit often; do not rewrite history.

## 5. Where to start (Argon)
Read `docs/gemini-brief-locomotion.md` (ranked questions, tools, how to run them), then `docs/early-fall-analysis.md`, `docs/evolution-pool-gains.md`,
`docs/startup-study.md`, `docs/training-progress-claude-dev-seeds.md`, `docs/training-progress-v25.md`, `docs/findings-locomotion-v24-diagnostic.md`,
`docs/findings-body-ab-g-inh.md`, the locomotion track's `docs/slip-credit-v24.md` and `docs/training-progress-v12.md`, and the code listed in the brief.
Deliver `docs/gemini-review-locomotion.md` on your branch: ranked hypotheses for the falls with predictions, the cheapest decisive tests,
comments on method, and optionally one implemented change with its paired evaluation.

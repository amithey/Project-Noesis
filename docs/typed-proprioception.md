# Type-aware proprioception: development experiment

Status: installed in the isolated worktree after V12 completed and was rejected.
The original-weight, nonlearning screen is now running. No behavioral improvement
or deployment is claimed. Default encoding remains legacy; typed variants require
an explicit CLI choice. The primary default source/model is unchanged.

Mamiya et al. 2018 reports phasic directional hook responses versus tonic claw
position responses; 2023 work identifies opposing hook preferences. The old hook
transfer ignored velocity and returned 76 Hz at a stationary rest pose. Sources:
https://pmc.ncbi.nlm.nih.gov/articles/PMC6481666/ and
https://doi.org/10.1016/j.neuron.2023.07.009 .

The MANC annotation study Figure 59 predicts opposing tibia-muscle effects of
SNpp39/41 hooks and SNpp50/51 claws:
https://elifesciences.org/reviewed-preprints/97766 .
The resistance hypothesis uses SNpp50 claw flexion-position, SNpp51 claw extension-
position, SNpp39 hook extension-motion and SNpp41 hook flexion-motion. This follows
a resistance-reflex interpretation of predicted circuit effects; it is NOT a
direct physiological annotation of these cell IDs. typed-reversed flips all four
preferences as an explicit control. Both retain the same annotated identities
and order, rather than inventing a row-parity or random split.

Known primary types determine the effective organ when a subclass conflicts.
There are 306 matched cells: SNpp50 99, SNpp51 57, SNpp39 96, SNpp41 54. The
remaining 3,615 mapped afferents keep their legacy rates exactly, including eight
unresolved hook cells. That unresolved fallback is a limitation, not a claim
that every sensory neuron is biologically correct. Loaded contact rates remain
unchanged; force saturation diagnosed earlier is not silently retuned here.

Known hooks receive rectified signed measured tibia velocity, returning the 2 Hz
base during every stationary hold. Known claws receive signed measured position.
All 2..150 Hz bounds and scale assumptions are retained. Inputs contain only
actual body state; no motor command, trajectory, oscillator or leg timing is added.
Connectome edges, signs and weights, neural dynamics, body physics, reward,
learning rate, eligibility and physical promotion thresholds are unchanged.

Both CLIs and the isolated launcher expose --proprioceptive-encoding. Model
context schema 7 records the choice and blocks normal cross-context checkpoint
loading; explicit weights-only import remains validated and resets traces.
Every training/original/candidate/ablation case receives the same chosen mapping.
External sensory ablation continues to retain joint and balance feedback.

Full staged regression and installed regression: 257 tests passed, Ruff passed.
They include hold-versus-motion tuning, opposing-type selectivity, reversed
assignment, identity/order preservation, unknown-type fallback, primary/subclass
conflicts, organ ablation, immutability, context rejection/explicit transfer,
brain-drive plumbing, the complete frozen matrix and interrupted-screen replay.
Initial test-key and staged script-path failures were fixed and their logs retained.
The PowerShell launcher parses. Source fingerprint:
738bd3a8130fc9a37eeb4b2a1351735006232e65ca1c74e60dbbac630422477b.

A four-neuron no-recurrence receiver using the real LIF backend confirms signed
motion inputs produce direction-selective hook spiking. This is a controlled
unit-level diagnostic stimulus, never a training controller or whole-brain
walking result. Coverage and receiver phase counts are in the companion JSON.

The development screen compares legacy, typed-resistance and typed-reversed on
seeds 15/16, with counterbalanced order, original weights, five seconds, learning
and exploration off, and full videos. It is bounded to one hour per invocation;
--resume replays unfinished trials from their start and preserves completed trials.
Both source and driver fingerprints are checked. It cannot promote a model and
rejects reserved validation/held-out seeds. Complete all six conditions before
choosing a training hypothesis; do not treat this screen as the required matched
trained-weight validation. Future training must still pass quiet, obstacle,
ablation and held-out gates before any normal launch change.

## First completed development baseline

Legacy seed 15 completed all five seconds with original weights, learning off and
exploration zero: 90.30% standing, 7.425 mm heading-integrated translation and
19.156 mm/s mean stance slip. Its saved weights equal initial weights exactly;
activity/eligibility traces and reward baseline remain zero. The full CSV and
video are retained. This is a baseline, not a trained walking result.

The matched typed-resistance seed-15 case is live. Remaining typed/reversed and
seed-16 cases are unfinished, so no direction hypothesis has been selected for
training. Source and driver fingerprints were reverified unchanged. A global CPU
load snapshot was 100% with about 1.3 GiB physical memory available, while learner
private memory stayed near 3.5 GB. Timing under that contention is not a controlled
performance benchmark; the fixed simulated-time requirements remain unchanged.

## First matched typed-resistance result, seed 15

The full original-weight typed-resistance trial completed: standing 78.69% versus
legacy 90.30%, stance slip 18.062 versus 19.156 mm/s and heading-integrated
translation 3.965 versus 7.425 mm. This is a decline in standing, despite a small
slip reduction. The higher mean uprightness alone does not establish standing: the
height gate and every out-of-range interval remain included. Original weights and
frozen traces were independently checked unchanged in both saved checkpoints.

The reversed-direction condition and seed 16 are still pending, so no tuning
hypothesis is selected or accepted for training. Full failures, support fractions
and posture/height failure breakdowns are preserved in
outputs/proprioception-screen-audit-v13/progress.json. The remaining comparison
must be completed before drawing a development decision; no result here can
promote a model or replace the matched trained-weight validation.

## Complete seed-15 comparison; seed 16 pending

All three original-weight conditions completed five seconds, with learning and
exploration off. These are development trials and cannot promote weights.

| Encoding | Standing | Stance slip mm/s | Heading displacement mm | Mean supporting feet |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 90.30% | 19.156 | 7.425 | 2.346 |
| Typed resistance | 78.69% | 18.062 | 3.965 | 2.598 |
| Typed reversed | 87.47% | 15.529 | 9.585 | 1.981 |

Neither typed variant reaches 90% standing in this seed. The reversed variant
reduces all-window slip by 18.94% relative to legacy, below the 20% walking target;
its lower mean support also requires caution in interpreting averaged slip. The
right hind leg has zero completed cycles in that condition. Passing translation
and cycles in other legs would not establish validated walking.

All three saved checkpoints retain exactly original weights and zero learning
traces. Posture/height failure breakdowns and upright-only slip are saved in the
companion JSON. The first typed condition improves mean uprightness but increases
high-body intervals, so the unchanged height gate is material. All seed-16 cases
remain unfinished; no preferred-direction hypothesis has been chosen for training.

## Reversed mapping does not hold across development seeds

Typed-reversed seed 16 completed five seconds with only 50.51% standing and an
unrecovered fall. Mean stance slip was 13.400 mm/s; the long fallen interval
prevents interpreting that as good walking. Legacy seed 16 is not complete, so
no cross-seed slip comparison is substituted for its matched baseline. Full video
and a frame around four simulated seconds preserve the failure. Original weights
and learning traces were rechecked unchanged. The resistance and legacy seed-16
conditions remain unfinished; no hypothesis has been selected for training.

The additional Function field labels all 142 original hook neurons as directional
proprioception and all 162 claw neurons as position proprioception. Body Part gives
only the leg, without receptor location or individual flexion/extension preference.
Current primary-priority reassignment changes the encoded organ of 36 cells,
including 25 hook/claw swaps and 11 other organ assignments. This exposes a
conflict-resolution assumption; it is not proof that primary types or subclass
labels are physiological ground truth. A subclass/Function-preserving encoding
should be compared separately after this screen, without changing running cases.
The exact audit is in proprioception-screen-audit-v13/annotation-conflicts.json.

## First hour limit and exact-protocol resume

The first invocation closed normally after 3,596.83 wall seconds. Four cases
completed. Typed-resistance seed 16 reached 4.90 simulated seconds and was marked
wall_budget, not completed; legacy seed 16 had not started. The almost-complete
trial is preserved but is not substituted for a five-second control.

After terminal exit was confirmed, --resume accepted unchanged source and driver
fingerprints/settings. Completed-case records are exactly preserved. The fifth
case restarted from time zero and original weights in attempt_0005; its partial
checkpoint is not used. All cases remain learning-off/exploration-off, and model
promotion remains disabled. The first-hour terminal report and partial artifacts
are retained for comparison with the completed replay.

## Fifth case completed by exact replay

The typed-resistance seed-16 replay completed five simulated seconds: 68.79%
standing, 15.195 mm/s all-window stance slip and 8.882 mm heading translation,
with a late unrecovered fall. Every field of all 980 partial CSV rows matches
the full replay prefix; 41,310 motor spike events match exactly.

The initial audit incorrectly treated scalar duration_ms as another spike vector.
That diagnostic assumption was corrected: partial/full neural durations are
4,850/4,950 ms, matching their awake windows, and extra spikes begin only after
the partial duration. Original weights and learning traces remain unchanged.
The 4.90-second attempt stays separately marked incomplete. Five of six original-
weight conditions are complete; the legacy seed-16 baseline is still pending.
No direction hypothesis has been selected for training or promoted.

# Training goal: standing, gripped steps, continuous walking, obstacles

This opt-in experiment replaces the proposed V30 fixed-reward extension. It starts
from the last V29 training bank (F9F), not a deployed or validated walking model.
V29 frozen seed10 stood for 95.05% of the window but slipped at15.07mm/s against a
7.49mm/s acceptance limit. One seed does not establish generalization or a trend.

Each stage trains six five-second births, then evaluates two fresh, matched
training-probe seeds with learning, motor/CNS exploration and fall cutoff off.
Training alone retains the 100ms fall cutoff. At most three rounds per stage;
failure stops for review rather than silently weakening the gate. Every stage
requires at least90% standing on both complete probes.

1. Standing: standing/support reward, mild slip and effort costs; zero progress
   reward. This deliberately isolates posture acquisition.
2. Gripped steps: grounded signed advance with a2mm/s reward target, smooth grip
   discount and slip costs. Gate: at least0.25mm advance, one real support/lift
   cycle in two legs, mean slip at most10mm/s. No credit for lifting alone.
3. Continuous walking: existing v6 reward,10mm/s reward target. Gate: at least1mm,
   two cycles in four legs, slip at most80% of matched original. Tripod reported.
4. Obstacles: v6 in the obstacle arena. Retain walking gates and require observed
   response within1s and less sustained contact than matched external-sense
   ablation. Joint/balance inputs remain active in that ablation.

These thresholds shape training only. They do not replace the final25 full frozen
controls on validation10/11 and heldout200/201/202, or the report/video/startup
verification. The curriculum never writes best-validated weights or startup files.
Final validation must start from the last bank with a compatible v6 model context.

Training seeds1000..1071; adaptive training-probe seeds10000..10023. Probes influence
stage choice and therefore are not final validation. Every stage change is based
only on those probes. Latest weights and last stage-passed anchor are separate;
anchors are retained for comparison, not silently substituted or called validated.

Original topology, signs, ±50% original-reference bounds,562 plasticity targets,
learning rate0.2,100ms eligibility,1.5mV/ms training exploration,BG40,U0 and existing
body/sensory interfaces remain. No gait clock or leg ordering is introduced.
Each birth transfers compatible weights only and resets neural/learning state.
Interrupted births are rerun from their start, not advanced from partial output.
Each invocation is bounded to3600s and resumes only with identical source/settings.

The `--fixed-objective` comparison arm uses v6 throughout, starting from the same
F9F parent and seed sequence. Before stage choices diverge, rounds are directly
matched in scene, seeds and planned births. After divergence they are not a
controlled reward-only comparison: compare only matched scene/round/horizon and
report actual completed training body seconds (falls shorten experience). No
claim that curriculum improves learning is justified before those measurements.
Longer stages or further mechanism changes require a new recorded experiment.

Run from the isolated checkout with PYTHONPATH=src:

    python -m noesis.experiments.curriculum_training --data DATA --output outputs/birth-curriculum-v31 --import-weights F9F --wall-seconds 3600

Use `--wall-seconds 3600` as separate CLI tokens, and add `--resume` on the next
invocation. No normal fly launcher is changed by this experiment.

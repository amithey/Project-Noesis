# Sensory development screen completed; training hypothesis selected

Six original-weight cases, five seconds each, learning/exploration off. Two hour-
bounded invocations completed the screen; a 4.9-second interruption replayed from
birth with all 980 observation rows and 41,310 spike events reproduced exactly.

| Mapping | Standing seed 15 | Standing seed 16 | Worst seed |
| --- | ---: | ---: | ---: |
| Original mapping | 90.3% | 32.0% | 32.0% |
| Inferred resistance mapping | 78.7% | 68.8% | 68.8% |
| Reversed mapping | 87.5% | 50.5% | 50.5% |

The inferred resistance mapping is selected for training because its worst seed
is strongest. It is not validated walking: every mapping fails 90% in at least
one seed, slip remains substantial, and the sensory preference assignments are
hypotheses. The 36 primary/subclass conflicts are explicitly retained as an
experimental policy, not physiological ground truth. All six results and failures
are reported; no model was promoted.

V14 now learns from complete V12 training weights on six fresh quiet-floor births
(seeds 17..22). Learning rate 0.2, eligibility 100 ms and exploration 1.5 mV/ms are
unchanged. The five-second validation/held-out, obstacle and sensory-ablation
requirements still apply with exploration and learning off. Source:
738bd3a8130fc9a37eeb4b2a1351735006232e65ca1c74e60dbbac630422477b.
Normal launch will change only after full behavioral validation and video review.

Evidence: outputs/proprioception-screen-v13/report.json and
outputs/proprioception-screen-audit-v13/final-results.json, decision.json and
replay-equivalence.json. Full source/receiver/annotation details are in
docs/typed-proprioception.md and docs/typed-proprioception-results.json.

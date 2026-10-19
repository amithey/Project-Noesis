# V25: six births with decaying training-only exploration — 2026-10-19

Source: branch `claude/locomotion-v25` (locomotion head `99c5ea6` plus two opt-in additions: `--exploration-schedule` in
`train_validate.py` and a diagnostic flag in `learning_lab.py` that is off by default). Registration identical to V24 except
the schedule: weights imported from the retained V24 bank (`birth-cycle-v24/.../train-5/attempt_0005/episode_000.npz`);
six quiet births, seeds 86-91, 5 s, training fall cutoff 0.1 s, learning rate 0.2, standing-gate-linear-slip-v5 reward,
background 3 Hz × 40, legacy proprioception, normalised and bias-corrected reward error, early rejection on, exploration
**1.5, 1.0, 0.6, 0.35, 0.2, 0.1 mV/ms** for births 0-5 (V24 used 1.5 throughout). Quiet validation, unchanged gates, seeds 10/11.

Motivation: the V24 candidate stood 0.93 with the training exploration current and 0.08-0.19 without it
(`findings-locomotion-v24-diagnostic.md`).

## Training births (learning and exploration on; not frozen results)
| Birth | seed | exploration | simulated s (fall cutoff) | standing | stance slip (mm/s) | heading-forward (mm) | legs with ≥ 2 cycles |
|---|---|---|---|---|---|---|---|
| 0 | 86 | 1.5 | 1.38 (fall) | 0.865 | 15.6 | +2.7 | 5 |
| 1 | 87 | 1.0 | 1.43 (fall) | 0.769 | 13.8 | +3.6 | 6 |
| 2 | 88 | 0.6 | 5.00 | 0.893 | 15.1 | +8.8 | 6 |
| 3 | 89 | 0.35 | 2.25 (fall) | 0.866 | 18.2 | +0.7 | 5 |
| 4 | 90 | 0.2 | 5.00 | 0.930 | 17.2 | +7.2 | 6 |
| 5 | 91 | 0.1 | 2.17 (fall) | 0.884 | 15.1 | +4.3 | 5 |

## Frozen quiet validation (learning off, exploration off, 5 s)
| Job | standing | stance slip (mm/s) | heading-forward (mm) | legs ≥ 2 cycles | uprightness | reward |
|---|---|---|---|---|---|---|
| original, seed 10 | 0.442 | 9.4 | +8.0 | 6 | +0.09 | −0.358 |
| original, seed 11 | 0.929 | 16.7 | +12.7 | 6 | +0.86 | +0.181 |
| **V25 candidate, seed 10** | **0.934** | **16.0** | **+9.7** | **6** | +0.88 | +0.208 |

Gate check for seed 10 (protocol unchanged): standing ≥ 0.9 **passes**, cycles **pass**, forward progress **passes**,
reward **passes**, stance slip **fails** (required ≤ 0.8 × 9.37 = 7.49 mm/s; measured 16.0). Early rejection therefore stopped
the cycle: status `rejected_validation`, `best_validated_checkpoint` stays null, seed-11 candidate and all held-out and
obstacle cases are **not run**. No claim of walking.

## Reading
- **The exploration dependence is gone:** the frozen V25 candidate stands 0.934 on seed 10 (V24 candidate 0.081), with
  exploration 0 as the protocol requires. One seed, one run.
- **The remaining failure is the slip gate only.** On seed 10 the *original* weights lie on their back for most of the
  episode (standing 0.44), so their stance slip is low (9.4 mm/s) simply because few feet are on the ground; a standing
  candidate (slip 16.0) cannot meet "≥ 20% below original" there. On seed 11 the original stands (slip 16.7) and the
  candidate's slip would have to be ≤ 13.4; its seed-10 value of 16.0 would be only about a 4% reduction against that original.
  The gate definition was not changed; this is an observation for whoever owns the protocol.
- Training births still fall often (3 of 6 ended by the 0.1 s fall cutoff), including at low exploration.

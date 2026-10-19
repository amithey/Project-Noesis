# V24 candidate: frozen evaluation with and without training-time exploration — 2026-10-19

Branch `claude/locomotion-v25` (from the locomotion branch head `99c5ea6`, 2026-10-19 17:57). The locomotion track's V24
cycle (six quiet births from seed 80 with the v5 slip credit) was rejected on quiet validation: the frozen candidate
stood 0.081 of 5 s on seed 10, against 0.442 for the original weights. Its training births had stood 0.85-0.98.

`scripts/frozen_probe.py` rebuilds the protocol's frozen job exactly (quiet arena, learning off, no fall cutoff,
standing-gate-linear-slip-v5, bg weight 40, legacy proprioception, normalised and bias-corrected error) and evaluates
the V24 last-training checkpoint (`birth-cycle-v24/.../train-5/attempt_0005/episode_000.npz`). Its exploration-off result
for seed 10 equals the protocol's recorded value (0.0808), which checks the probe. The exploration-on rows are a
**diagnostic**, not a protocol condition: the 1.5 mV/ms training current is injected into an otherwise frozen run
(weights never change), enabled by an opt-in flag that is off by default.

| Seed | exploration | standing | stance slip (mm/s) | heading-forward (mm) | supporting feet | uprightness | reward | legs with ≥ 2 cycles |
|---|---|---|---|---|---|---|---|---|
| 10 | off (protocol) | **0.081** | 5.8 | +9.2 | 1.19 | −0.49 | −0.805 | 6 of 6 (cycles 0-19) |
| 11 | off (protocol) | **0.188** | 6.7 | +9.3 | 1.36 | −0.29 | −0.672 | 6 of 6 |
| 10 | on, 1.5 mV/ms | **0.932** | 16.8 | +9.0 | 2.44 | +0.86 | +0.197 | 6 of 6 |
| 11 | on, 1.5 mV/ms | **0.936** | 15.7 | +11.9 | 2.51 | +0.87 | +0.213 | 6 of 6 |

- The same frozen weights stand 0.93 on both seeds when the training-time exploration current is present and collapse to
  0.08-0.19 when it is absent: the candidate depends on exploration to stay upright. This is the pre-stated trigger for
  the next step (annealing exploration during training); the original weights, frozen and without exploration, stand
  0.44 (seed 10) and 0.93 (seed 11) in the V24 report.
- With exploration on, stance slip is 15.7-16.8 mm/s, so the slip gate still fails; with exploration off slip is low only
  because the fly lies on its back (uprightness < 0).
- Two seeds, one checkpoint, one run each; a mechanism (e.g. the current's rectified mean drive) is not identified.

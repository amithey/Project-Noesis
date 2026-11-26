# R0e — baseline for variant B (real fly leg ranges, mean muscles), untrained parent

- **Date/time:** 2026-11-26, about 11:19–11:27 local; code `7cc8b67`
- **Command:**

```
python es_evaluate.py --legs real --kinematics <project>\data\kinematics\single_steps_untethered.pkl --muscle mean --output <out>\R0e-baseline-reallegs --label "R0e baseline, real leg ranges, untrained" --workers 3
```

- **Why B:** Amit asked for the training itself to run on a real fly's leg ranges. Native-leg training plateaued (R1b, R3, R4 flat),
  and B gives larger strides and steps on every leg. The standing drop on B is what training now has to fix. This is the reference R5 is compared against.
- **Wall:** 478 s (6 episodes + 10 s video).

### Held-out (5 s; "candidate" = "parent" = untrained brain on B)

| Seed | standing | upright | slip | v7 | v8 | forward mm | height mm | debounced steps LF..RH | extended support LF..RH | stride mm LF..RH |
|---|---|---|---|---|---|---|---|---|---|---|
| 20001 | 0.498 | 0.796 | 3.01 | −0.036 | −0.242 | −0.69 | 1.41 | 4, 2, 5, 2, 4, 6 | .62, **.08**, .46, **.20**, .56, .45 | 1.46, 1.16, 2.03, .66, 1.92, .94 |
| 20002 | 0.813 | 0.814 | 3.14 | 0.467 | 0.167 | +1.70 | 1.21 | 3, 5, 4, 5, 6, 2 | .70, **.14**, .45, .27, .35, .73 | 1.50, .93, 1.62, .60, 1.28, .92 |
| 20003 | 0.588 | 0.799 | 3.18 | 0.110 | −0.146 | −0.04 | 1.38 | 4, 1, 6, 2, 7, 5 | .72, **.06**, .56, .20, .48, .61 | 1.80, 1.20, 2.12, .65, 1.92, .98 |

Means: standing 0.633, uprightness 0.803, v7 0.180 (native-leg parent: 0.905 / 0.874 / 0.617).

### Gates
**Stand FAILED:** standing ≥ 0.97 FAIL, uprightness ≥ 0.85 FAIL, every-leg extended support ≥ 0.3 FAIL, beats parent FAIL; slip PASS.
**Walk FAILED:** stride ≥ 0.3 mm on every leg PASS, stepping 4+ legs (raw and debounced) PASS; forward, standing ≥ 0.9, all 6 legs ≥ 2 steps, every-leg support ≥ 0.2 and beats parent FAIL.

### Video
[`videos/R0e-baseline-real-legs.mp4`](../videos/R0e-baseline-real-legs.mp4) (seed 20001, 10 s): forward −1.70 mm, standing 0.372.
The fly stands much taller (height 1.5–1.8 mm vs about 1.1 mm on native legs), rears up and pitches, turns, and loses
uprightness (0.69–0.92). The legs swing through larger arcs. The left middle leg (LM) barely carries weight (extended support 0.06–0.14).

### A/B/C comparison table
See [R0d.md](R0d.md).

File: [`R0e/evaluation.json`](R0e/evaluation.json)

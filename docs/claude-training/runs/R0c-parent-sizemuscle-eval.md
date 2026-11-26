# R0c — parent with size-weighted muscle pools (`--muscle size`) — decision: **not adopted**

- **Date/time:** 2026-11-26, about 10:15–10:29 local (1 worker, alongside R4)
- **Code:** `8af20af` (size-weighted pools, stride metric) + `7958c14`
- **Command:**

```
python es_evaluate.py --muscle size --output <out>\R0c-parent-sizemuscle-eval --label "R0c parent, size-weighted muscles" --no-video --workers 1
```

- **What `--muscle size` does:** each pool's drive scales with its motor-neuron count instead of the pool's mean rate, so for example
  2 tibia antagonists at ~120 Hz no longer outweigh 17 agonists. Connectome and gains are unchanged (parent, all gains 1).
- **Wall time:** 14 min (6 episodes on 1 worker). No video. Side and rear stills from a 3 s run on seed 20001 are in `audit-legs/`.

### Held-out (5 s, seeds 20001–20003). "candidate" = "parent" here: both are the size-muscle parent

| Seed | standing | upright | slip | v7 | forward mm | debounced steps (LF..RH) | extended support (LF..RH) | stride mm (LF..RH) |
|---|---|---|---|---|---|---|---|---|
| 20001 | 0.770 | 0.842 | 2.64 | 0.407 | −0.42 | 3, 7, 2, 2, 2, 2 | .50, .35, .70, **.11**, .84, .39 | .73, 1.03, 1.66, .51, 1.81, .64 |
| 20002 | 0.789 | 0.859 | 2.65 | 0.437 | +1.02 | 4, 7, 3, 1, 2, 2 | .69, **.27**, .69, **.05**, .65, .57 | .73, 1.06, 1.60, .56, 1.50, .66 |
| 20003 | 0.864 | 0.858 | 2.38 | 0.557 | +1.30 | 3, 3, 6, 3, 4, 2 | .69, .34, .46, **.18**, .49, .84 | .77, .97, 1.48, .44, .96, .63 |

**Mean-muscle parent (R0) for comparison:** standing 0.898 / 0.907 / 0.911, uprightness 0.868 / 0.879 / 0.875, v7 0.608 / 0.617 / 0.626.

### Leg audit, seed 20001, 3 s (cmd_std / angle_std / claw-load share / debounced steps / tibia command − rest)

| Leg | parent, mean muscles | parent, size muscles | R1b, size muscles |
|---|---|---|---|
| LF | .118 / .093 / .81 / 4 / −.21 | .131 / .103 / .50 / 1 / −.09 | .132 / .103 / .55 / 2 / −.09 |
| LM | .186 / .146 / .11 / 2 / −.35 | .204 / .160 / .34 / 3 / −.20 | .198 / .157 / .47 / 3 / −.22 |
| LH | .231 / .185 / .37 / 3 / −.41 | .232 / .190 / .67 / 2 / −.30 | .214 / .174 / .68 / 4 / −.33 |
| RF | .148 / .117 / .27 / 1 / −.47 | .200 / .160 / .14 / 1 / −.27 | .187 / .149 / .05 / 0 / −.25 |
| RM | .145 / .117 / .25 / 4 / −.39 | .177 / .147 / .81 / 1 / −.20 | .180 / .152 / .82 / 1 / −.18 |
| RH | .164 / .125 / .75 / 3 / −.41 | .139 / .105 / .46 / 1 / −.25 | .140 / .104 / .18 / 3 / −.25 |

### Decision (rule set by the coordinator before the audits)
- Tibia bias shrinks: **yes** (−0.21…−0.47 → −0.09…−0.30).
- RF/RH angle_std or steps improve: **mixed** (RF angle_std .117 → .160; RH .125 → .105 and 3 → 1 steps).
- Fly still stands: **no**. It does not tip over, but in the stills it pitches nose-up about 40° and sits on its abdomen from about 1.5 s.
  Held out, standing falls to 0.77–0.86 (from 0.90–0.91), uprightness drops below 0.85 on seed 20001, and v7 falls from 0.62 to 0.47 on average.
  RF extended support gets worse (0.05–0.18).

→ **Not adopted.** Training continues with mean muscles (R5). The size model probably needs re-tuned muscle gains or
rest posture before it can stand; that would be its own experiment.

Files: [`R0c-parent-sizemuscle-eval/evaluation.json`](R0c-parent-sizemuscle-eval/evaluation.json), audits and stills in [`audit-legs/`](audit-legs/)

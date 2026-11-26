# R5 — ES stand on real fly leg ranges (variant B) — **PAUSED at Amit's request**

- **Status:** paused (processes stopped) on 2026-11-26 at about 13:13 local (11:13 UTC), during generation 10. Not resumed until Amit says so.
- **Last completed generation:** **9** (10 of 12 done). Generation 10 was in progress and is lost; it reruns on resume.
- **Best centre:** **0.6404** at gen 4 (start = R1b gains on real legs: 0.1976). The R0e untrained parent scored 0.443 on these training seeds (R5c gen 0).
- **Started:** 2026-11-26 11:16 local; code `7cc8b67`; 3 workers (R5c ran alongside it during gens 2–6).
- No report.json because the process was killed. Everything in the output folder is intact:
  `<project>\gpu-pilot\outputs\claude-es\R5-es-stand-reallegs\` → config.json, generations.jsonl, best-gains.npy, weights-best.npz (not committed), last-mean-gains.npy.
  Small copies: [`R5-paused/`](R5-paused/).

### Generations so far (4 seeds × 13 candidates, 1.5 s)

| Gen | pop mean | pop max | pop min | centre | best centre | improved | errors | gain mean [min,max] | sigma | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4223 | 0.5599 | 0.1167 | 0.1976 | 0.1976 | yes | 0 | 1.002 [0.894,1.125] | 0.0500 | 603 |
| 1 | 0.4983 | 0.6757 | 0.3205 | 0.5659 | 0.5659 | yes | 0 | 1.002 [0.892,1.134] | 0.0500 | 539 |
| 2 | 0.5016 | 0.6503 | 0.2766 | 0.4395 | 0.5659 |  | 0 | 1.003 [0.884,1.143] | 0.0500 | 724 |
| 3 | 0.4696 | 0.6239 | 0.1806 | 0.6112 | 0.6112 | yes | 0 | 1.003 [0.878,1.148] | 0.0500 | 770 |
| 4 | 0.4697 | 0.5686 | 0.3615 | 0.6404 | 0.6404 | yes | 0 | 1.003 [0.870,1.148] | 0.0500 | 802 |
| 5 | 0.5175 | 0.6797 | 0.3049 | 0.5661 | 0.6404 |  | 0 | 1.003 [0.862,1.153] | 0.0500 | 789 |
| 6 | 0.5285 | 0.7467 | 0.3207 | 0.5095 | 0.6404 |  | 0 | 1.003 [0.858,1.159] | 0.0500 | 956 |
| 7 | 0.4875 | 0.6146 | 0.3760 | 0.4029 | 0.6404 |  | 0 | 1.003 [0.855,1.162] | 0.0500 | 585 |
| 8 | 0.5046 | 0.7186 | 0.2245 | 0.5264 | 0.6404 |  | 0 | 1.003 [0.850,1.165] | 0.0500 | 539 |
| 9 | 0.5410 | 0.6551 | 0.4374 | 0.5668 | 0.6404 |  | 0 | 1.003 [0.846,1.167] | 0.0500 | 508 |

(Training fitness on B includes the extended-support bonus, so it is not comparable to R1b–R4 numbers.)

### Resume command (from `C:\Users\heyma\noesis-claude\gpu-pilot`, code ≥ `f5cd829`)
Approximate resume: the mean comes from last-mean-gains.npy and Adam/RNG restart, because this run predates opt-state.npz.

```
python es_train.py --method es --stage stand --generations 12 --seconds 1.5 --workers 3 --wall-hours 3 --output "C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\gpu-pilot\outputs\claude-es\R5-es-stand-reallegs" --seeds-per-gen 4 --sigma 0.05 --lr 0.01 --legs real --kinematics "C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\data\kinematics\single_steps_untethered.pkl" --muscle mean --init-gains "C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\gpu-pilot\outputs\claude-es\R1b-es-stand-fixedseeds\best-gains.npy" --resume
```

Then evaluate with the B flags and a video:

```
python es_evaluate.py --gains <R5 dir>\best-gains.npy --output <R5 dir>\eval --label "R5 ES stand, real legs" --workers 3 --legs real --kinematics <KIN> --muscle mean
```

(Python: `C:/Users/heyma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe` with
PYTHONPATH = `<project>\gpu-pilot\runtime;<project>\gpu-pilot\arrow-runtime`; the launcher `C:\Users\heyma\noesis-run.ps1` sets this up.)

### R5c control plan
Start from scratch after R5: same flags as R5 but **no `--init-gains`** (all ones), 3 workers, output `R5c-es-stand-reallegs-ones`.
A first parallel attempt (output `R5c`, 2 workers) completed gens 0–1 (centre 0.443 → 0.505). It was stopped because it
slowed R5 by more than 50% (956 s vs about 570 s per generation). Its folder is kept for reference.

# smoke — ES pipeline smoke test

- **Date/time:** 2026-11-25, 18:59 local; wall time 59 s total (41 s for the one generation)
- **Command** (from `gpu-pilot`):

```
python es_train.py --method es --stage stand --generations 1 --pairs 2 --seconds 0.5 --workers 2 --output <project>\gpu-pilot\outputs\claude-es\smoke
```

- **Config summary:** OpenAI-ES, 2 antithetic pairs (4 candidates + centre), 0.5 s episodes (0.45 s scored after the 50 ms settle),
  training seed 30000, sigma 0.08, lr 0.03, fine grouping → **136 groups over 85,390 plastic edges**
  (158,262 neurons, 3,006,456 edges). Parent checkpoint sha256 `fb80cfcc…26a8`, core source sha256 `f52dff5a…396b`.
- **Result:** completed, 0 errors; all expected files written (config.json, generations.jsonl, best-gains.npy,
  weights-best.npz, last-mean-gains.npy, report.json).

### Training fitness

| Gen | pop mean | pop max | pop min | centre | best centre | improved | errors | gain mean [min,max] | sigma | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.6464 | 0.7712 | 0.4560 | 0.6518 | 0.6518 | yes | 0 | 1.005 [0.970,1.030] | 0.0800 | 41 |

No held-out evaluation for the smoke test (it is a pipeline check, not a candidate). **Gates: n/a. Verdict: pipeline OK.**

### Environment notes / fixes
- Code fix: `dump()` in `src/noesis_gpu/es_brain.py` could not serialize numpy bools from the gate checks.
  It now converts numpy scalars/arrays, and the console summary in `es_evaluate.py` does the same (commit `b883767`).
  No other code changes.
- Memory (sampled every 10 s during R0): each pool worker ≈ 510–560 MB, main process ≈ 800 MB.
  Free RAM at idle was already only ~2.5 GB of 15.7 GB (other apps use ~83%), so the "<80% RAM" target cannot
  be met with any worker count. With 3 workers free RAM dropped to ~1.2–1.5 GB with no failures.
  **Chosen `--workers 3`** (= physical cores − 1).
- At 0.5 s the spread across candidates is large (0.456–0.771), so single-seed episode fitness is noisy.

Files: [`runs/smoke/`](smoke/)

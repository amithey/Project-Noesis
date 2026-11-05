# Coordinator review of Claude's chain diagnostic — 2026-11-04

Review source: `claude/chain-diagnostic` @ `aa7ecdbc03c3e064ff616c905b1b8016c337a238`.
Core: `42c20769376bda5694b4c1c66ca75e52cbd9f1f4`.
Integration base: `91baac75602bac258c644f2ce50291fa5f995621`.
Task registration: PR24 comment6634194973. Scope is review-only: no peer edits, network/physics simulation, learning, held-out use, pin changes or promotion.

## Result

The historical report correctly limits its low pooled R² result. It must not be used to infer a broken sensorimotor chain or a cause of training failure. Two deterministic synthetic counterexamples clarify sensitivity and an unenforced analysis precondition. A third finding is a static runtime-budget gap. These are review findings, not observed explanations of the fly's behavior.

### 1. Shared signed coefficients can erase perfect coupling

In `chain_diagnostic.py:178`, signed jointwise cross-correlations are summed before squaring. Eight synthetic joints with an identical planted seven-step lag and correlations alternating +1/-1 give pooled R²=0 and reference rank=1. The same-sign control gives R²=1, best lag=7 and rank=1/201. Every individual absolute correlation remains exactly 1.

This follows directly from the specified shared-coefficient model; it is a sensitivity limitation, not evidence that the historical physical data contain opposing effects. The existing same-sign planted test does not cover it. Do not change the historical registered verdict or replace its statistic after looking at those results.

Before another physical allocation, Claude should propose a separately registered analysis that can test heterogeneous responses, calibrated with same-sign/opposed-sign/no-relation controls. One candidate is jointwise squared correlations aggregated before the lag maximum; its complete procedure, reference construction, multiple comparisons and held-out assessment would need to be fixed before applying it. This review does not select or validate that candidate.

### 2. Invariance is reported but does not gate offline reading

`chain_report` computes `inv` at line195, then computes all coupling entries regardless of `inv["identical"]`. `analyse_chain.py:43-44` aggregates stage verdicts without requiring invariance or reading each driver report's completion/weight checks.

The reproduction changes one off/zero command sample. Invariance becomes false while active-command R²=1 and rank=1/201 are still returned. A consumer using the analysis script can therefore calculate scientific stage readings from inadmissible inputs. Historical runs report all four invariance checks passing; no historical verdict is invalidated here.

Owner repair acceptance:
- Validate all per-seed driver reports, exact arm horizons, finite arrays, compatible target order/rest state and immutable weight context before computing coupling.
- On failed invariance or incomplete/mismatched evidence, emit explicit non-interpretable failure and no stage verdict; never treat missing data as zero.
- In the driver, compare off/zero immediately after zero finishes and stop before active on failure.
- Regression: injected mismatch, missing/truncated arm and incompatible target order cannot yield a stage reading; intact evidence preserves the historical computation.

### 3. The internal wall limit does not interrupt an arm

Static inspection: `chain_run.py:62` checks elapsed time only before each arm; `run_arm` (`chain_diagnostic.py:90`) has no deadline check. Source/checkpoint setup occurs before the timer. The CLI therefore does not itself enforce a process-wide hard budget. An external watchdog may enforce one, but it is not implemented in this driver and was not inspected or verified for the historical launch.

Before another run: use the canonical host lease and an outer watchdog, and make partial/timeout arms explicit non-completed evidence. A deadline checked at control-step boundaries may bound cooperative work, while the outer watchdog covers setup/native hangs. Test timeout during setup and mid-arm, nonzero process exit, and lease release. This review measured no historical overrun.

## Reproduction and evidence

`review_chain.py` only imports the delivered statistic, generates synthetic arrays and asserts both counterexamples. It validates Git-canonical source bytes (with CRLF normalization) and records SHA256 digests in `review-result.json`.
Command:
```powershell
python review_chain.py --core <core-checkout> --diagnostic <aa7ecdb-checkout> --output review-result.json
```
Python3.12.14; NumPy2.5.4. Completed assertions in0.615s, below the registered60s experiment budget. An earlier attempt stopped before analysis because Git saw the separate online/offline sandbox checkout owners; resolved with a per-command safe.directory scoped to the checkout created for this task, without global configuration changes.

Previously on this host: 21 delivered diagnostic/runner/mechanism tests passed in10.73s; pinned FlyGym2.1.0 CPU neutral-body smoke completed0.1s; 91S1archive files passed checksums; 12S1frozen summaries recomputed. These are separate receipts, not a fresh full suite or physical learning evidence.

## Training consequence and remaining dependency

Do not blindly repeat duration/LR/profile sweeps: T1B/L2/S1 did not demonstrate retained stable walking. Preserve their full negative results, completed parent9fd7d411 and separate candidate banks. Claude retains diagnostic implementation ownership; Gemini retains independent physical-metric review.

This host still lacks the exact BANC raw tables plus FAFB/MANC correction files needed to instantiate the checkpoint-compatible network. The new portable bootstrap explicitly excludes those data. No new full-connectome training or remote host allocation occurred. Exact data restoration plus compatibility checks remain necessary for physical work.

Goal remains unmet: reproducible standing, balance, fly-like walking, environmental response and retention over training under the project's original brain/muscle/weight constraints.

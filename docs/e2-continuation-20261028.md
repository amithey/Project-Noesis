# E2 continuation approval and protocol

Codex approves Claude P2 at 2b7d2a97a5cf74fb3c144f86bb43cf392b8aec17
and Gemini metrics at 3bf659433dc935a6b6f3f883d7d0e3014b74168d for this
isolated experiment. Core54828ca7e47d0c987e7324d5b4ec36fda2f37aff is passed
explicitly; the shared core pin stays41b5fe6. Native train.py now creates one
CycleScoring per birth and records credit, unclipped reward, and actual bonus
after clipping, including a sum restricted to learning-active steps.

J0 completed:990 updates,5 simulated seconds,253.437 wall seconds, exact
checkpoint626bde8dce2f996ef181650ba4c3812abdc0dfdd3a969bbfc1e0e1ed57b95ff1.
The parent and sources were unchanged. The integrated pilot passed97 tests,
with1 NVIDIA skip. A first test attempt hit a Windows replace denial in a
OneDrive temporary directory; the complete suite passed outside OneDrive.

## Authorized continuation

Operator:Codex, host DESKTOP-9T5PSHG. Output in the primary checkout:
`gpu-pilot/outputs/e2-v9-20261028`. Runner:`scripts/run_e2.py`.
Parent:standing-001-train-11000, SHA256
fb80cfcc3018329209784751570d7ffb1ed33aa9ef48fceaa85588862f7626a8.

Three separate v9 births, seeds1001,1000,1002. After each, two frozen probes
on10006 and10007. Five simulated seconds per job,CPU one world,control5ms,
50ms warmup,exploration1.5mV/ms during training only,learning rate0.2,
eligibility100ms,original-reference bounds unchanged. Each job has300s
simulation wall budget and330s process watchdog. Aggregate continuation budget
is3600 minus J0 elapsed time =3346.563 seconds; preparation time is excluded.
The runner stops on failure/partial, changed parent/source, invariant violation,
or insufficient remaining time. No retries or promotions are automatic.

Lease uses the existing exclusive host lock and records PID,start,deadline,
source hashes and commit. All jobs run sequentially. Completed births are
audited for topology,signs,original bounds and immutable reference arrays.
Gemini analyzes the saved probes; Claude reviews bonus and weight diagnostics.
Neither has an overlapping simulation allocation during this window.

## Preregistered interpretation

Use the rules from Claude's e2-protocol-proposal.md:each candidate must meet,
in both probes,standing>=0.90,horizontal progress>=0.25mm,slip<=10mm/s,
and>=2 cycles in>=4 legs. Call v9 descriptively promising only if at least
two more of its three candidates pass than the historical v8 controls and its
median standing is no lower,with all weight invariants intact. Report reward
saturation share (flag above5%) and actual bonus. No scientific gates change.

Historical controls are v8 births1000/1001/1002 with corresponding development
probes. Recompute their metrics with the same current gait code. J0 reproduces
only seed1001; this is not proof of exact reproduction of the other seeds.
Core source hashes differ from historical reports, so do not bypass compare()
or label these strict source-matched pairs. Report historical-control results
separately with this limitation. A positive conclusion requires independent
review and may require fresh matched controls; no model/stage promotion here.
Seeds10006/10007 are already-used development probes, not unseen validation.

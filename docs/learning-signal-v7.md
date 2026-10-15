# Learning toward supported standing and basic walking

The goal remains active and unachieved. No candidate has been promoted or made the
primary launcher default. Connectome topology and signs remain fixed, and incoming
motor magnitudes remain bounded to 50% of their original values. No gait clock,
leg-order program or external walking controller was added.

## What the complete v6 quiet validation showed

Both original and trained weights were frozen for five seconds, with exploration
and learning off, on the same validation seeds. Joint motion and repeated foot
cycles occurred, but the requested walking criteria did not all pass.

| Seed | Original standing fraction | Trained standing fraction | Original slip mm/s | Trained slip mm/s |
|---|---:|---:|---:|---:|
| 10 | 0.83333 | 0.83030 | 5.42592 | 6.38946 |
| 11 | 0.91010 | 0.90808 | 5.81899 | 6.12793 |

Trained seed 10 advanced 4.4775 mm vs 2.7147 mm original, but its posture and slip
failed the gate. Slip worsened for both validation seeds. The old scalar reward
nevertheless improved, exposing a mismatch between that objective and supported
walking. The v6 hour ended with an incomplete obstacle control; it remains marked
`wall_budget`, not completed. Exact v6 source is preserved at
`outputs/protocol-v6-source/src` so those negative controls remain reproducible.

A separate one-thread trained-seed-10 diagnostic matched **every CSV column** of
the official eight-thread five-second evaluation. It did not establish a reliable
speedup, so the active protocol retains eight threads and the original physics
stepping path. Faster-observation prototypes were not adopted.

## Reward objective v2

Progress retains the heading-relative, contact/height/posture gate and grip-factor
attenuation. For uprightness u, valid height H, supporting-foot count n and stance
slip s in mm/s:

- Posture quality P = clip((u - 0.5) / 0.4, 0, 1).
- Support quality S = min(n / 3, 1).
- Standing quality = H * P * S.
- Reward = clip(0.25 * standing quality + 0.5 * grounded progress - 0.6 * fallen
  - 0.05 * effort - 0.3 * (s / 5) - 0.2 * obstacle cost, -1, 1).

Thus an upright airborne body cannot earn the standing term, and three supported
feet earn more than two. The slip cost stays linear above 5 mm/s instead of
saturating. Within the total reward bounds, alternating large slip bursts cannot
earn more than steady slipping with the same mean. The existing 90% stance,
1 mm progress, four legs with two cycles and 20% slip-improvement gates are intact.
These coefficients remain engineering assumptions, not a biological reward law.

## Action-credit normalization

An independent delayed-reward task used five seeds (13..17), 10 simulated seconds,
50 ms delay and reward amplitude 0.01. Learning rate was 0.2 and eligibility 100 ms
for both rules. Current credit changed relevant weights by 0.000522 on average;
the normalized prototype changed them by 0.052948. Unrelated change was 0.003776,
shuffled-reward change 0.004208 and zero-reward change exactly zero. This verifies
small-signal credit in a synthetic task; it does not prove fly walking. The
separate shuffled-reward probe is recorded in the JSON evidence. Its script must
use the pinned old source when reproduced:

```powershell
$env:PYTHONPATH='outputs/protocol-v6-source/src'
python outputs/probe_reward_credit_scale_v6.py
```

`--normalize-reward-error` keeps the raw two-second reward baseline, estimates the
two-second exponentially weighted mean squared raw prediction error, divides by
its RMS with floor 0.01, and clips the resulting learning signal to [-1,1]. The
rate 0.2 applies to this unitless signal; effective raw-error gain is explicitly
reported as `reward_error_scale`. Eligibility remains 100 ms and motor exploration
1.5 mV/ms. Presynaptic action credit and weight-sign conversion are unchanged.
Evaluation skips all learning updates and leaves normalization statistics reset.

Normalized checkpoints use NPZ version 3, including normalization state. Ordinary
loads check learning configuration, topology, motor identities, initial weights,
signs, magnitude bounds and finite/bounded trace state. Explicit `--import-weights`
transfers only validated weight tensors from versions 1/2/3, resetting traces and
reward statistics; it cannot bypass topology, signs or current magnitude bounds.
Model sidecars use schema 6 and name the reward/normalization configuration.

`--early-reject` ends a candidate cycle after a completed matched quiet-validation
case fails a mandatory gate. It never labels the unrun obstacle or held-out tests
completed and never promotes that candidate. A subsequent `--resume` continues
training from its last completed training weights. Budget-interrupted cases still
restart from episode start before proceeding; successful promotion still requires
all original controls and all validation/held-out behavioral gates.

## The interrupted v7 attempt and scoped runtime guard

The first v7 attempt reached only 2.31 simulated seconds. Its actual wall time was
11098.86 seconds, beyond the intended hour. Windows Kernel-Power events confirm
Modern Standby from 16:05:08 to 18:55:55 (reason: Austerity Battery Drain Budget
Exceeded). No partially trained weights were selected and no episode was counted
completed. The original relative timer did not enforce an absolute deadline across
this suspension.

`WallBudget` now checks the absolute monotonic deadline every 250 ms. During a
bounded job, its main thread makes a scoped SYSTEM_REQUIRED/CONTINUOUS request,
restoring its prior state on normal/error exit. This is the standard temporary
application request documented by [Microsoft](https://learn.microsoft.com/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate).
It requests CPU availability, not an always-on display or a permanent power-policy
change. It does not override manual or OS-enforced suspension; suspended code cannot
execute, and an overdue job exits at the first scheduled watchdog check on resume.

The old v7 source was archived before this guard-only change. A migration verified
that **only wall_budget.py changed**, with every neural/body/reward/learning source
byte-identical, no completed v7 jobs and no selected v7 training checkpoint. It
records both fingerprints and tags the incomplete attempt. The resumed attempt
uses the exact original seed and v6 start checkpoint, not the partial v7 weights.

All 181 tests passed, including small delayed-reward specificity, zero/unrelated
controls, normalization checkpoint continuation, invalid-state atomicity, safe
weight imports, frozen evaluation, reward ranking, early-reject/resume behavior,
subprocess deadlines, a simulated suspension clock jump, and scoped-request cleanup.
Ruff and the PowerShell launcher parser passed.

## Current run

`outputs/birth-cycle-v7/report.json` is authoritative. The resumed ten-second
quiet-floor training episode is currently live with a 3600-wall-second budget.
The behavior and promotion criteria are unchanged. Resume only once its process
is terminal:

```powershell
$env:PYTHONPATH='src'
python -u -m noesis.experiments.train_validate --data <existing-data> --output outputs/birth-cycle-v7 --episodes 1 --seconds 10 --wall-seconds 3600 --import-weights outputs/birth-cycle-v6/cycle_000/train-0/attempt_0000/episode_000.npz --normalize-reward-error --early-reject --video --resume
```

The dedicated PowerShell launcher exposes `-NormalizeRewardError` and `-EarlyReject`
along with `-Resume`, `-ImportWeights`, `-Seconds`, `-Episodes` and `-WallSeconds`.
Machine-readable evidence: `docs/learning-signal-v7-results.json`.

# Training on the upstream body

The active development plan is [training-plan.md](training-plan.md). The updated
app-goal text is [goal-text.md](goal-text.md). This document records the current
single-birth implementation and completed pilot, not a completed curriculum port.

`noesis_gpu.train` adds a **single-world experimental training birth** to the
FlyGym 2 transport pilot. It uses the existing Noesis connectome and perturbation
learning rule; it does not use FlyGym's separate walking controllers or imitation
PPO policy. The ordinary `noesis-gpu` command remains frozen, with zero exploration.

Run in the isolated Python 3.12 environment with the current isolated Noesis
source (the older main-checkout core may lack the required circuit-learning rule):

```sh
python -m noesis_gpu.train --backend cpu --noesis-source /absolute/current/noesis/src --data /absolute/noesis/data --checkpoint /absolute/completed/parent.npz --seed 11000 --seconds 5 --wall-seconds 600 --output outputs/train-11000
```

The episode contains 50 ms of initial neural/body settling with no exploration
or learning, then bounded nonperiodic motor/circuit exploration and scalar
standing feedback. Existing-edge signs and topology remain fixed. Learning rate
is 0.2, eligibility 100 ms, current <=1.5 mV/ms and original-reference bounds ±50%.
Original edge references are captured before importing the parent weights, and
remain original when a later completed checkpoint starts a new birth.

The reward uses separate **claw-to-floor** force sensors. Whole-leg sensory loads
remain available for proprioception; a femur or thorax touching ground does not
provide standing reward. Slip is estimated from the positions of the tarsus5
bodies during ground contact. This is a modelling approximation, not proof of gait.

Completed births save `weights-last.npz` and an explicitly experimental model
context. A wall-budget interruption saves `weights-partial.npz`, never a latest
or validated pointer. Start the interrupted birth again from its original parent;
there is no mid-episode state restoration. Outputs are new directories and all
raw observations, commands, pad states and rewards are retained.

Evaluate the completed candidate using the frozen command and the same seed and
conditions as its starting-bank control:

```sh
noesis-gpu --backend cpu --worlds 1 --seconds 5 --seed 10006 --noesis-source /absolute/current/noesis/src --data /absolute/noesis/data --checkpoint outputs/train-11000/weights-last.npz --output outputs/candidate-10006
```

This entry point does **not** advance the existing curriculum, select a validated
best model, or update startup. External vision/chemical senses and the obstacle
stage are not yet ported. All standing/gait/obstacle controls and full validation
remain necessary. A result under exploration is not a frozen walking result.
CUDA remains unverified on the Intel-only local machine; CPU one-world training
can run here. Training many worlds needs independent mutable weight banks and a
separately validated aggregation/selection procedure; frozen weight sharing is
not a multi-world training implementation.

## First completed local pilot

One five-second CPU training birth (seed 11000) completed with the full
158,262-neuron connectome. There were 990 learning updates after the 50 ms
settling interval. A direct parent/child checkpoint audit found 44,292 changed
edges among 85,390 selected edges. Topology, selected targets, original weight
references, original signs and ±50% bounds all remained valid. The largest
additional change was 1.195% of an original weight.

The same five-second frozen seed 10006 was tested before and after training,
with zero exploration and no learning. Actual claw-to-floor support was used:

| Metric | Starting training bank | After this birth |
| --- | ---: | ---: |
| Standing fraction | 93.13% | 97.17% |
| Mean stance slip, mm/s | 2.397 | 2.349 |
| Mean uprightness | 0.880 | 0.864 |
| Mean supporting feet | 2.483 | 2.401 |

Standing increased 4.04 percentage points in this **single pair**. Slip decreased
1.99%, below the eventual 20% gait criterion. Mean uprightness and supporting-foot
count declined. These results do not establish an overall learning trend,
generalization, learned gait, or a passed curriculum stage. The control is the
starting *training bank*, not the original untrained graph. No model was promoted.

The earlier 95.66% body-support measurement included whole-leg contacts. Its raw
canonical observations were verified bitwise identical to the rerun after adding
claw sensors; the stricter claw-only standing measurement is 93.13%.

The completed experimental weights are in
`outputs/standing-001-train-11000/weights-last.npz` (SHA-256
`fb80cfcc3018329209784751570d7ffb1ed33aa9ef48fceaa85588862f7626a8`).
See `outputs/standing-learning-001-comparison.json` for the paired results and
`continuation-status.json` for the remaining work. Tests: **23 passed, one
NVIDIA-only test skipped; Ruff passed**. The source pin stayed unchanged during
the three full runs, all processes terminated, and the legacy training worktree
remained clean.

## Second frozen training probe

The same parent and candidate completed a second full five-second frozen pair,
seed 10007. Standing increased from **91.11% to 95.05%** and mean stance slip
decreased from 2.871 to 2.243 mm/s (**21.89%**). Uprightness and supporting-foot
count declined slightly. Both candidate probes, 10006 and 10007, exceed the
90% standing threshold. This passes the intermediate standing training gate,
not final validation, gait acceptance or deployment promotion. The parent
already contains training experience; it is not the untrained original graph.

`outputs/standing-two-seed-gate.json` contains both pairs, observation/report
hashes, warmup and measurement counts, and source/checkpoint audit results.
Both native processes exited successfully and the process inventory was empty.
The next work is the interruption-safe curriculum manager and physical leg-cycle
metrics, before gripped-step training. No new training birth occurred in this pair.

`python -m noesis_gpu.evaluate --parent /completed/parent --candidate /completed/candidate --output /new/comparison.json`
audits one pair. Its `standing_gate` API requires two distinct non-held-out seeds,
the same two banks, and matched body/source/settings across all four evaluations.
It rejects partial episodes and any active learning/exploration. Tests:
**31 passed, one NVIDIA-only skipped; Ruff passed**.

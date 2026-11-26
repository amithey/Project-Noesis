# Learning laboratory validation — 2026-10-13

59 tests passed in the installed body environment. This includes plasticity
bounds/signs, checkpoint validation and transfer, frozen evaluation, artificial
reward guards, and episode integration/error cleanup. Ruff passed for the new
modules and tests. The PowerShell launcher parsed without errors.

The real cached connectome contains 158,262 table entries and 1,528,585 retained
synaptic pairs. The physical experiment maps 389 motor entries onto body joints;
37,312 existing incoming motor edges are eligible for plasticity.

Two real training episodes completed, each 0.7 simulated seconds with birth at
0.2 seconds. Episode 1 loaded episode 0's weights into fresh brain/body states.
A separate frozen original-network run recorded an MP4 with the same seed and
timing as episode 0. All three runs completed without a reported physics error.

| Run | Seed | Mean reward | Mean uprightness | Forward displacement (mm) |
| --- | --- | --- | --- | --- |
| Training 0 | 0 | -0.088105 | 0.664973 | -0.523741 |
| Training 1, transferred weights | 1 | -0.010952 | 0.723412 | -0.201315 |
| Frozen original, recorded | 0 | -0.088105 | 0.664973 | -0.523741 |

Episode 0 changed 6,669 eligible weights. Mean absolute relative change was
3.0962e-7, maximum 6.5006e-5; all connection signs were preserved. These are tiny
changes. The matched frozen and learning-on runs had identical summary metrics.
There is **no demonstrated behavioral improvement** in this short smoke test.
The different seed in episode 1 prevents attributing its different score to
training. Tilt/backward displacement are not a successful walking gait.

Artifacts are in ignored local directories:

- `outputs/learning-smoke-20261013`: config, two CSVs, summaries, and checkpoints.
- `outputs/fly-preview-20261013`: frozen baseline video, CSV, summary, checkpoint.

Headless episodes took about 159 and 182 wall seconds, and the recorded episode
took about 329 seconds. Runs overlapped, so these are not isolated performance
benchmarks. Interactive rendering should not be assumed to operate in real time.

Flight remains unimplemented. Longer training, sensory/motor calibration and
held-out matched-seed evaluation are necessary before any learning claim.

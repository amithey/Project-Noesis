# Brain/body training connection — 2026-10-14

## What was repaired

The normal `Start-Fly.ps1` path previously used raw connectivity, an independent
joint-target adapter, and disabled learning unless `-Learn` was explicitly added.
It also required a `.venv` in each checkout, complicating isolated worktree use.
The visual display advances in simulated time; hundreds of wall seconds per few
simulated seconds can look like a frozen fly.

The launcher now enables training by default, selects the existing FAFB/MANC
corrected connectome and routes motor spikes through `MuscleModel`: separate
agonist/antagonist pools, 30 ms activation dynamics, bounded equilibrium targets,
and the body's real-kinematics standing pose, damping and joint limits. The body
starts at 1.2 mm rather than 0.2 mm. Raw and legacy modes remain available in the
Python CLI for controls. A unique activity-map filename isolates concurrent runs.

Leg joint angles, velocities and loads return to annotated sensory afferents;
body tilt and rotation return through balance afferents. The existing habitat
supplies vision, smell, contact taste/touch and environmental channels. No LIF
kernel, neural adaptation rule, connectome source table or sensory mapping was
changed. The validated regional-scaling loader was integrated from the existing
`feature/gait-rhythm` work; it is not a new calibration on these trials.

Training uses explicitly artificial, bounded stochastic exploration currents
into annotated motor-neuron pools (1.5 mV/ms, 100 ms OU time constant). It neither
supplies a leg trajectory nor imposes a tripod oscillator. Frozen evaluation
defaults to zero current. Existing motor-synapse plasticity and its artificial
upright/forward reward are retained. This has not established biologically
realistic reinforcement learning.

Telemetry distinguishes muscle commands from actual measured joint motion.
PAM, PPL1, MBON and KC firing channels are also recorded, without labeling them as
subjective feelings or thought. In these models dopamine populations can respond
nonspecifically, so their rates do not establish a particular internal emotion.

## Controls and short physical validation

Fresh brain/body states, seed 0, 0.6 s duration, birth at 0.05 s, dt=0.5 ms neural /
0.1 ms physics / 5 ms control. Corrected weights, background 3 Hz x 60 except the
passive control. All listed episodes completed without a reported physics error.

| Condition | Seed | Mean joint speed (rad/s) | Mean uprightness | Forward displacement (mm) | Mean reward |
|---|---:|---:|---:|---:|---:|
| Passive: no afferents, background or exploration | 0 | 0.390407 | 0.999835 | -0.884310 | -0.020389 |
| Frozen brain, leg/balance feedback, no exploration | 0 | 2.138160 | 0.955079 | 1.157842 | 0.212746 |
| Training + exploration, empty arena | 0 | 2.261536 | 0.972061 | 0.978977 | 0.176513 |
| Training + exploration + habitat, recorded | 0 | 2.294580 | 0.931074 | 1.779266 | 0.312800 |
| Transferred training weights + habitat, recorded | 1 | 2.110776 | 0.905028 | 1.993529 | 0.331167 |

All six legs move in the connected conditions. In the empty-arena training run,
per-leg mean joint speeds are LF 1.892, LM 2.423, LH 2.788, RF 2.700, RM 1.595,
RH 2.171 rad/s. The passive control retains gravity/contact settling, so movement
without neural input is not assumed to be exactly zero. The recorded habitat
runs have mean absolute relative weight changes 1.1791e-5 and 2.2569e-5 at the
end; these are tiny changes. Different seeds and exploration conditions preclude
attributing score differences in this table to learning.

The old raw/legacy path also moved in a 0.4 s empty-arena baseline (uprightness
0.981644, forward displacement 1.280357 mm). Therefore this work does not claim
that the previous implementation had no motor connection at all. It repairs the
normal path's inconsistent model selection, physical muscle adapter, training
switch, and absence of direct movement diagnostics.

## Longer training and matched held-out evaluation

Three training episodes each ran for 2 s (seeds 0, 1, 2), resetting body and
membrane state and transferring learned weights between episodes. The final
checkpoint was evaluated on unseen seed 100 for 1 s, against original weights
with identical seed, timing and sensory conditions. Learning and exploration
were both disabled during the held-out comparison. All runs completed.

| Run | Seed | Joint speed rad/s | Uprightness | Forward mm | Mean reward | Median antagonist r | Tripod score |
|---|---:|---:|---:|---:|---:|---:|---|
| training-connected 0 | 0 | 2.256964 | 0.899326 | 1.255947 | 0.129525 | -0.150653 | null |
| training-connected 1 | 1 | 2.143555 | 0.907189 | 1.225891 | 0.142896 | -0.170865 | null |
| training-connected 2 | 2 | 2.345132 | 0.903804 | 2.850904 | 0.185937 | -0.166890 | null |
| heldout-original 0 | 100 | 2.322447 | 0.885948 | 1.974032 | 0.210299 | -0.202142 | null |
| heldout-trained 0 | 100 | 2.289499 | 0.861900 | 1.577905 | 0.171357 | -0.193633 | null |

The final mean absolute relative weight change was 3.8373e-5 (0.00384%).
All six legs moved in every training episode; a valid shared dominant tripod
rhythm was not detected. The trained checkpoint did not improve this held-out
trial: reward decreased from 0.210299 to 0.171357, and uprightness from
0.885948 to 0.861900. One held-out seed cannot establish a general performance
effect; these measurements provide no evidence of learned walking. Increased
reward across different training seeds is not treated as proof of learning.

Aggregate settings/results are in [brain-body-results.json](brain-body-results.json).
The final checkpoint and complete longer-run artifacts are also copied to the
normal checkout at `outputs/brain-body-training/`; the recorded short preview
is `outputs/brain-body-training-preview.mp4`. To resume explicitly:

```powershell
.\Start-Fly.ps1 -Checkpoint outputs\brain-body-training\episode_002.npz
```

A real live-map smoke test also completed in the normal checkout using its
own `.venv`, saving an activity snapshot with actual motor/movement telemetry.
The final full suite passed 118 tests and Ruff. The atomic-publish integration
test requires ordinary filesystem rename permissions; an initial sandboxed
run dropped that frame as designed, and the normal-permission run passed.

## Validation and artifacts

118 tests passed in the installed body environment; `ruff check src tests`
passed. The PowerShell launcher parsed successfully. The normal checkout's
`.venv` also passed the 10 embodiment/episode integration tests after applying
the patch. Integration tests explicitly hold fake physical joints still while
muscle commands change, checking that telemetry never confuses commands with
actual motion. Tests also check exploration bounds/repeatability, absent neural
readout pools, read-only shared-data cache behavior, gait null controls and
incomplete recording bins.

Local artifacts are beneath the attached `brain-body-feedback` worktree's
`outputs/`: `baseline`, `passive-control`, `frozen-control`, `muscle-corrected`,
`habitat-training` (two recorded episodes), `training-connected` (longer training),
`heldout-original`, and `heldout-trained` (matched evaluation). Outputs are
ignored by Git. Every final-path episode saves leg-motor spikes and gait metrics
in addition to the existing CSVs and checkpoints.

The normal checkout received the exact tested runtime/test/documentation patch
without changing its branch or staging its files. Existing FABA/MANC directories
and Gemini collaboration files were preserved. Commits are made in the attached
isolated worktree.

## Reproduction

Normal startup enables corrected-brain muscle training:

```powershell
.\Start-Fly.ps1
.\Start-Fly.ps1 -Learn:$false   # frozen evaluation
```

Longer matched validation from the worktree:

```powershell
$env:PYTHONPATH='src'
$brainData='C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\data'
& C:/dev/venvs/noesis/Scripts/python.exe -m noesis.experiments.learning_lab --data $brainData --output outputs/training-connected --arena habitat --episodes 3 --seconds 2 --learn
& C:/dev/venvs/noesis/Scripts/python.exe -m noesis.experiments.learning_lab --data $brainData --output outputs/heldout-original --arena habitat --episodes 1 --seconds 1 --seed 100 --motor-exploration 0
& C:/dev/venvs/noesis/Scripts/python.exe -m noesis.experiments.learning_lab --data $brainData --output outputs/heldout-trained --arena habitat --episodes 1 --seconds 1 --seed 100 --motor-exploration 0 --checkpoint outputs/training-connected/episode_002.npz
```

Output directories must be new. Existing raw-brain checkpoints are deliberately
incompatible with a corrected topology/initial-weight set. Longer time series
and multiple matched held-out seeds are required before any walking-improvement
claim. Neither forward displacement nor weight changes alone establishes gait;
leg-phase and antagonist timing are recorded for that reason. This simulation
still runs substantially slower than real time. The existing wing adapter is
kinematic and has no aerodynamic flight model.

# Connectome learning laboratory

See [the initial validation results](learning-validation.md) for measured outcomes
and limitations of the first real-data runs.

The downloaded wiring drives the existing physical NeuroMechFly body. Sensory
feedback enters the LIF network; motor-neuron activity changes leg joint targets.
There is no preprogrammed walking gait in this experiment. The muscle mapping,
sensory encoding, neural dynamics, reward, and plasticity rule are assumptions.
The downloaded table contains 158,262 annotated entries, including non-neuronal
objects. Dataset identity/provenance and exclusions still need verification.

## Run on Windows

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy scipy pandas pyarrow numba pytest ruff flygym networkx
.\Start-Fly.ps1 -Learn -Video -Seconds 3 -Episodes 3
```

The launcher opens the live MuJoCo viewer and a separate 3-D network window showing the 1,800 strongest
real connections, with schematic positions based on the neurons' annotated brain and nerve-cord regions.
These are not measured anatomical coordinates. The default portrait view follows the
visual reference: a wide compact brain above the separate nerve cord, dark background,
tiny grey-blue anatomical points and small warm-white activity highlights. The bounded
neuropil volumes are schematic, not reconstructed neuron morphology. Up to about 40,000
context points describe the shape; entries annotated NO_CONS are excluded from this
context layer. Real displayed edges remain very faint until a neuron is selected.
Edges remain faint even when their neurons fire
(solid for excitatory, dotted for inhibitory);
neurons firing above 12 Hz receive a layered halo whose size increases with firing rate.
At most the 2,000 most active neurons glow per frame to keep the viewer responsive.
Connections remain visible when neurons are silent. Click a neuron to highlight its
displayed incoming/outgoing connections and direction arrows; Escape restores the overview.
Drag to rotate and scroll to zoom. The displayed graph is a subset; the simulation still
uses the full network. The standalone brain-map CLI accepts `--edges 3000` for a denser
view or `--network-layout` for a layout based on connectivity instead of anatomy.
Simulation may be much slower than
real time, particularly when recording. Close the viewer to stop after saving
the current episode. Every run gets a new directory beneath `outputs/`.
Without `-Learn`, weights are frozen. Camera controls are MuJoCo's standard
mouse controls. By default the fly lives in a sensory habitat with physical walls,
obstacles, coloured landmarks, food, water, and autonomous visual/odor changes.
See [the habitat](habitat.md) for sensory routing, controls and model assumptions.
Adhesion is held enabled, as in the original newborn model. Flight is not implemented.

The shared world now explicitly uses an infinite collision plane at z=0 and
gravity (0, 0, -9810) mm/s², equivalent to Earth's 9.81 m/s². All fly geometries
(including head, thorax, abdomen and wings) collide with the floor. Previously
FlyGym's default only enabled leg-floor collisions, allowing the body to sink
when tilted. Both `newborn` and `learning_lab` use this configuration. Floor
reflections are disabled to make the surface boundary visually clear. Contacts
are resolved by MuJoCo, with small numerical penetration tolerances; body
coordinates are not teleported or clamped to fake contact.
Contact recovery uses a 0.2 ms time constant and damping ratio 1, with a 0.05 mm
collision margin above the visible floor to absorb impact discretization. The
contact test checks every physics step and rejects penetration beyond 0.002 mm.

With body dependencies installed, `python -m pytest tests/test_world.py -q`
checks gravitational free fall and floor contact in upright, side and upside-down
orientations using the actual MuJoCo model, without the neural network.

For recording without opening a window:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m noesis.experiments.learning_lab --learn --video --episodes 3 --output outputs/train
```

Raw CSVs are read from `data/raw/`, falling back to `downloaded file/` when
necessary; the existing processed cache is reused. Do not mix a cache with
different source tables or neurotransmitter-sign assumptions.

## What learns

Only existing connections into the mapped leg motor neurons change. An assumed
reward-modulated Hebbian rule uses coactivity traces and reward prediction error.
Connection signs are preserved and magnitudes stay within 50% of their original
values. Positive prediction error increases the magnitudes of both excitatory
and inhibitory eligible edges; this simplistic rule is not a validated model of
fly dopamine or synaptic plasticity. The original connectome files are unchanged.

Reward favors upright forward motion along world +x, penalizes falls and a joint
velocity cost, and suppresses progress credit outside plausible posture/height
or at launch-like speed. It is an artificial training objective, not an instinct
or biological reward inferred from the downloaded data.

Each episode resets brain dynamics and body state with seed `seed + episode`.
Training transfers learned weights between episodes; activity/eligibility traces
and reward baseline reset. A checkpoint stores selected synaptic weights and
learning traces, not the complete brain/body state. Its topology, starting
weights, and learning parameters must match when loading.
If trained with a nondefault rate, supply the same `--learning-rate` during
evaluation (or `-LearningRate` through the PowerShell launcher).

## Measure rather than assume success

Each episode saves an MP4 if requested, a CSV sampled every 5 ms, a JSON summary,
and an NPZ checkpoint. `config.json` records parameters, and `summary.json`
collects episode outcomes. A physics failure or viewer close is recorded as
such, not counted as a completed episode. Settling before `--birth` is excluded
from reward and movement summaries. Learning begins at birth.
CSV `weight_change` is the mean absolute relative deviation from initial weights,
including during frozen evaluation. It is not the amount changed in that frame.

Evaluate the trained weights **without** `--learn` on held-out seeds, and run
the original frozen network on the same seeds and duration:

```powershell
.\.venv\Scripts\python.exe -m noesis.experiments.learning_lab --checkpoint outputs/train/episode_002.npz --seed 100 --episodes 5 --output outputs/eval-trained
.\.venv\Scripts\python.exe -m noesis.experiments.learning_lab --seed 100 --episodes 5 --output outputs/eval-original
```

Compare `mean_reward`, `mean_uprightness`, forward displacement and completion
status across matched seeds. Weight changes or leg movement alone are not proof
of improvement. The current experiment has not established learned walking.
It has no aerodynamic wing model and cannot establish learned flight. Calibration
of joint signs, sensory gains and neural dynamics remains a prerequisite for
interpreting failed or successful training.

## Scientific references

- [Shiu et al., Nature 2024](https://www.nature.com/articles/s41586-024-07763-9):
  connectome-constrained neural modelling and sensorimotor predictions.
- [NeuroMechFly v2, Nature Methods 2024](https://www.nature.com/articles/s41592-024-02497-y):
  the embodied fly simulation underlying FlyGym.

Neither source validates the new learning rule implemented here.

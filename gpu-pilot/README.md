# FlyGym GPU integration pilot

This separate package uses the **official FlyGym `GPUSimulation`**, pinned to
FlyGym 2.1.0 commit `ca65a510c2afe6ac61c51df4f274c8d190c2f95f`.
The legacy training environment, startup script, curriculum and checkpoints are
not modified. Python 3.12+ and an NVIDIA CUDA GPU are required for GPU execution.

For the new single-world learning entry point, see [training.md](training.md).
The ordinary `noesis-gpu` transport/evaluation command remains frozen.
The updated development plan and app-goal text are [training-plan.md](training-plan.md)
and [goal-text.md](goal-text.md).

Implemented:

- Batched physics for 1, 8, 32, 100 or another chosen number of independent worlds.
- Native-to-Noesis anatomical mapping of all 42 leg DOFs; missing or duplicated
  channels fail before running. Native v2 axes are retained.
- A frozen full-connectome CPU brain per world, using existing Noesis LIF,
  motor-pool matrices, proprioceptive mapping and balance mapping.
- Existing muscle activation/equilibrium equations, with independent state.
- Ground-only contact loads from six upstream contact sensors; body-body contacts
  are not counted as ground support. Joint/velocity and balance feedback close
  the sensorimotor loop. Pad release depends on muscle commands, with no gait
  sequence, oscillator, replayed movement, or external walking controller.
- Optional weight import through the existing checkpoint validator, preserving
  topology, signs and the original-reference 50% bounds. No learning updates.
- Explicit memory budget before allocating the entire neural population.
- Recorded observations, package/source/checkpoint provenance, full-chunk counts,
  throughput and incomplete/failure status. Existing output directories are rejected.

## Install on the NVIDIA machine

In this directory, using a **new** environment (do not upgrade the training venv):

```sh
python3.12 -m venv .venv
# Linux:
. .venv/bin/activate
# Windows PowerShell alternative: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The installation uses upstream code directly; no third-party source is copied
into this package. FlyGym declares Apache-2.0 licensing. Keep the existing
connectome data local and follow its separate data license.

## First test: upstream body only

```sh
noesis-gpu --controller neutral --backend cpu --worlds 1 --seconds 0.1 --output outputs/cpu-body-1
noesis-gpu --controller neutral --backend gpu --worlds 1 --seconds 0.1 --output outputs/gpu-body-1
python -m noesis_gpu.compare --cpu outputs/cpu-body-1 --gpu outputs/gpu-body-1 --output outputs/body-comparison.json
noesis-gpu --controller neutral --backend gpu --worlds 8 --seconds 1 --output outputs/gpu-body-8
noesis-gpu --controller neutral --backend gpu --worlds 32 --seconds 1 --output outputs/gpu-body-32
noesis-gpu --controller neutral --backend gpu --worlds 100 --seconds 1 --output outputs/gpu-body-100
```

These are **neutral-pose physics tests**, not walking or brain training. There is
no hidden CPU fallback. If CUDA is missing, the GPU command fails and writes a
failure report. Compilation/setup time is recorded separately from loop time;
the first few steps can still include JIT compilation, so short runs are not
steady-state performance estimates. GPU observations currently return to CPU
every 5 ms of simulated time; neural execution is sequential on the CPU.
The comparison reports numerical differences and rejects incomplete or unmatched
controls; it does not declare learned walking or automatically accept a migration.

## Connect the existing brain

Pass the intended absolute Noesis **src directory**, read-only data directory,
and optionally a completed checkpoint. This also works with the isolated
`brain-body-feedback` source rather than the user's main checkout.

```sh
noesis-gpu --backend cpu --worlds 1 --seconds 0.1 --noesis-source /absolute/noesis/src --data /absolute/noesis/data --output outputs/cpu-brain-1
noesis-gpu --backend gpu --worlds 1 --seconds 0.1 --noesis-source /absolute/noesis/src --data /absolute/noesis/data --checkpoint /absolute/completed/episode_000.npz --output outputs/gpu-brain-1
```

After matching the one-world CPU/GPU behavior, increase `--worlds` gradually.
`--host-memory-budget-gib` defaults to 8. The estimate covers neural arrays plus
30% headroom; it is **not** a total RAM/VRAM guarantee and excludes Python,
DataFrames, physics allocations and compilation memory. Inspect real RAM/VRAM
usage before choosing a larger server. Full brains remain much more expensive
than the official body-only example. Each world's neuronal/muscle state is
independent; there is no weight averaging across worlds.
For frozen probes the topology and weights are shared read-only, reducing host
memory use; membrane potentials, delayed spikes and muscle state stay separate.

## Validation and limits

Initial transport verification on 2026-10-21: **21 tests passed, 1 NVIDIA-only test skipped**;
Ruff passed. The real v2 CPU body and a synthetic neural feedback loop were tested.
A separate 50 ms run connected the actual **158,262-neuron, 3,006,456-edge**
connectome and completed v4 checkpoint `078cfe3c...` to the real v2 CPU body.
It completed ten feedback chunks, recorded 31,016 spikes and left weights unchanged.
The final thorax height was 1.036 mm, uprightness 0.994 and three legs had ground
contact. This very short settling interval establishes neither standing nor gait.
See [local-verification.json](local-verification.json) for exact measured results.
The host has Intel Iris Xe graphics; CUDA execution, GPU throughput and the
100-world hardware run remain **unverified**. The installed local test runtime
contains the CPU dependency set; the package install above includes the GPU extra
for the NVIDIA machine. The isolated training worktree remained clean.
The subsequent training integration passed **23 tests, with one NVIDIA-only skip**.
Its completed training birth and frozen before/after comparison are documented in
[training.md](training.md); they are preliminary and do not prove walking.

```sh
NOESIS_SOURCE=/absolute/noesis/src python -m pytest -q
python -m ruff check src tests
```

`NOESIS_SOURCE` selects the exact core for testing; otherwise the adjacent main
checkout is used. Tests cover anatomical permutations, equality of the muscle
equations to the existing model, batch-size-independent brain streams, separate
states, checkpoint rejection, and incomplete-run accounting. The actual FlyGym
CPU body test requires the isolated v2 environment. CUDA acceptance is still a
separate hardware check; a CPU pass is not a GPU pass.

**This is an experimental integration, not a replacement for `train_validate`.**
The following migration work remains before any training promotion:

- Compare old/new body geometry, axis directions, neutral pose, joint limits,
  contact/adhesion dynamics and full five-second behavior. This pilot uses the
  v2 native neutral pose, ±0.7 rad joint ranges, damping 1, servo gain 45 and force
  limits ±65. These are labelled pilot assumptions, not verified v1 equivalence.
- Bring external vision/chemical senses, head/wing output and the existing
  curriculum's reward/eligibility/checkpoint lifecycle into the batched path.
- Replace sequential CPU brains with a validated batched neural implementation
  if profiling shows they dominate runtime. A GPU physics backend alone does not
  accelerate the full brain or imply a 100× speed-up.
- Validate CUDA ground-contact sensors and all private adapter mappings against
  the pinned release on real NVIDIA hardware.
- Run the existing standing, gait, slip and obstacle controls. No pilot result
  advances a stage, replaces validated weights, or changes ordinary startup.

Numba's existing neural RNG is thread-global. The pilot reseeds it for each chunk
from an independent per-world generator. This prevents cross-world interference
but changes the random schedule relative to the legacy single-fly pipeline;
identical seed numbers alone do not establish a matched legacy control.

References:

- [Official GPU tutorial](https://neuromechfly.org/tutorials/3_gpu_accelerated_simulation/)
- [Pinned upstream GPU implementation](https://github.com/NeLy-EPFL/flygym/blob/ca65a510c2afe6ac61c51df4f274c8d190c2f95f/src/flygym/warp/simulation.py)

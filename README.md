# Project-Noesis
True awakening isn't seeing the light. It's realizing the dark you exist in is just an unrendered void.

## Status
**M1: whole-CNS connectome simulation.** 158k FlyWire neurons as an event-driven LIF network
(see [ADR-0001](docs/adr/0001-whole-cns-connectome-simulation.md)). Driving the sugar-taste
neurons activates the proboscis motor neuron MN9, as in Shiu et al., *Nature* 2024.

## Quick start

For the physical fly with experimental learning, a live viewer, video recording,
and episode metrics, see [the learning laboratory](docs/learning-lab.md).
On Windows with body dependencies installed: `./Start-Fly.ps1 -Learn -Video`.
The default [sensory habitat](docs/habitat.md) provides position-dependent odors,
rendered binocular vision, contact taste, thermal/humidity regions and moving stimuli.
This is an experiment, not demonstrated learned walking or flight.

```bash
python -m venv <venv-dir> && <venv-dir>/Scripts/activate     # keep the venv outside OneDrive
pip install numpy scipy pandas pyarrow numba pytest ruff flygym networkx
# put neurons.csv and connections_princeton.csv (FlyWire) in data/raw/
export PYTHONPATH=src            # PowerShell: $env:PYTHONPATH = "src"
python -m noesis.experiments.feeding
python -m noesis.experiments.newborn --viewer --seconds 5   # live MuJoCo window
pytest
```
`pip install -e .` works on Linux/CI, but fails in a project path containing non-ASCII
characters on Windows (setuptools writes the `.pth` file as cp1252), hence `PYTHONPATH=src`.
Data is not in the repo (size, CC-BY-NC licence). Cite Dorkenwald et al. 2024 and Shiu et al. 2024.

## License / Copyright
Copyright (c) 2026 Amit Heymans. All rights reserved.
This project and its source code are proprietary and confidential. No part of this work may be reproduced, modified, or distributed without prior written permission.

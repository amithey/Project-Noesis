# Preservation snapshot validation — 2026-10-27

This backup preserves existing experimental work. It does not certify training
or merge alternative implementations.

Current hub core: `.venv/Scripts/python.exe -m pytest -q` produced **118 passed,
4 failed**. Three failures are in `tests/test_brain_map.py`: the local Matplotlib
installation loads incompatible `matplotlib` and `mpl_toolkits` components
(`matplotlib._api.check_getitem` is unavailable). The other failure is
`test_muscle_path_routes_neural_rates_and_records_actual_motion` in
`tests/test_learning_lab.py`, where the expected `live.npz` was absent. These
failures were observed during preservation and have not been repaired here.

Historical pilot results and their limitations are recorded in
`gpu-pilot/training.md`; do not substitute a historical pass for a current check.
The current pilot test attempt could not collect tests: the local Python 3.12
environment has a pandas extension compiled for NumPy 1.x but the isolated pilot
runtime supplies NumPy 2.5.4. This dependency mismatch needs a fresh compatible
environment before claiming a current pilot test pass.
The new archive script passed Ruff and Python bytecode compilation.
The hub and required training core remain separate branches. See
`coordination.md` for exact core/checkpoint hashes and data requirements.

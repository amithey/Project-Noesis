"""Post-hoc floor/sensitivity diagnosis; preserves failed CL1 and never retunes it."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'work/Noesis-Core/src'))
from noesis.lif import LIFNetwork, LIFParams, _seed
from noesis.plasticity import MotorPlasticity

source = root / 'work/closed_loop_readout_review_20261106.py'
tree = ast.parse(source.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('network', 'advance', 'probe')]
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'))
out = root / 'outputs/closed-loop-readout-cl1-20261106'
result = json.loads((out / 'result.json').read_text())
rows = {}
for name, weights in [('original', [25., -10.]), ('best_signed_bound', [37.5, -5.]), ('opposite_signed_bound', [12.5, -15.])]:
    rows[name] = [probe(np.array(weights), seed) for seed in range(60000, 60004)]
assert rows['original'] == result['baseline_probe_returns']
with np.load(out / 'numerical-evidence.npz', allow_pickle=False) as z:
    summary = {mode: {
        'max_abs_relative_weight_change': float(max(np.max(np.abs(z[f'{seed}_{mode}_weights']/[25., -10.] - 1)) for seed in range(50000, 50016))),
        'mean_actual_training_reward': float(np.mean([z[f'{seed}_{mode}_actual_reward'][10:].mean() for seed in range(50000, 50016)]))}
        for mode in ('online', 'yoked', 'off')}
receipt = dict(role='Post-hoc technical floor/sensitivity diagnostic only; no primary threshold or training parameter changed',
               source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
               frozen_returns_at_original_reference_bounds=rows, training_summary=summary, physical_simulation=False)
(out / 'calibration-posthoc.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt, indent=2))

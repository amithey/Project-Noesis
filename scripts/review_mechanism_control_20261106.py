"""Independent read-only review of Claude's exact synthetic mechanism control."""
import ast
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

root = Path(__file__).resolve().parents[1]
core = root / 'work/Noesis-Core/src'
peer = root / 'work/Noesis-Diagnostic'
sys.path[:0] = [str(core), str(peer / 'gpu-pilot/src')]
from noesis import plasticity
from noesis_gpu import mechanism_control as mc

assert Path(plasticity.__file__).resolve() == core / 'noesis/plasticity.py'
assert Path(mc.__file__).resolve() == peer / 'gpu-pilot/src/noesis_gpu/mechanism_control.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


paths = [Path(mc.__file__), Path(plasticity.__file__), core / 'noesis/lif.py',
         peer / 'docs/workstreams/claude/mechanism-control/mechanism-control-result.json']
hashes = {str(p.relative_to(root)): sha(p) for p in paths}
started = time.monotonic()
expected = json.loads(paths[-1].read_text())
observed = mc.run_control(range(16), kappa=.2, lr=.2, steps=990)
assert observed['verdict'] == expected['verdict']
max_error = 0.
for arm, rows in expected['rows'].items():
    assert len(rows) == len(observed['rows'][arm]) == 16
    for before, after in zip(rows, observed['rows'][arm]):
        assert before.keys() == after.keys()
        for key in before:
            if isinstance(before[key], bool):
                assert before[key] is after[key]
            else:
                max_error = max(max_error, abs(before[key] - after[key]))
                assert abs(before[key] - after[key]) <= 1e-12
for arm, summary in expected['summary'].items():
    for key, value in summary.items():
        assert abs(value - observed['summary'][arm][key]) <= 1e-12

reversed_rows = [mc.birth(s, -.2, learning=True, lr=.2, steps=990) for s in range(16)]
d = [r['D'] for r in reversed_rows]
reverse = dict(mean_D=float(np.mean(d)), z=mc.z_score(d),
               all_bounds_and_signs=all(.5 <= r['ratio_min'] <= r['ratio_max'] <= 1.5 and r['signs_kept'] for r in reversed_rows))
reverse['registered_pass'] = bool(reverse['mean_D'] < 0 and reverse['z'] <= -3 and reverse['all_bounds_and_signs'])

source = inspect.getsource(mc.birth)
tree = ast.parse(source)
network_runs = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == 'net' and n.func.attr == 'run']
assert not network_runs
assert all(sha(p) == hashes[str(p.relative_to(root))] for p in paths)
result = dict(status='completed_independent_synthetic_mechanism_review', seconds=time.monotonic()-started,
              peer_commit='aa7ecdbc03c3e064ff616c905b1b8016c337a238',
              core_commit='42c20769376bda5694b4c1c66ca75e52cbd9f1f4', source_sha256=hashes,
              packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'numba')},
              import_core_verified=True, source_unchanged=True, physical_simulation=False, trained_checkpoint_written=False,
              original_rows_max_absolute_error=max_error, original_verdict_reproduced=True,
              original=observed, reversed_planted_effect=reverse, reversed_rows=reversed_rows,
              static_scope=dict(net_run_calls=network_runs,
                                reward_definition='clip(.3 + externally drawn noise + kappa * planted OU history mean, -1,1)',
                                spike_definition='independent Bernoulli samples, not LIF output',
                                learned_weights_feed_back_into_reward_or_spikes=False),
              interpretation='Update-direction association control reproduced; not a closed-loop controller improvement test or proof of physical learning.',
              seed_scope='Original synthetic draw IDs0..15 only; no fly data, development or final physical probes.',
              remaining='Physical causal exploration-to-reward/credit timing and retained behavioral improvement; do not repeat this association control as if missing.')
(root / 'outputs/mechanism-control-review-20261106.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({key: result[key] for key in ('status', 'seconds', 'original_rows_max_absolute_error', 'original_verdict_reproduced', 'reversed_planted_effect')}))

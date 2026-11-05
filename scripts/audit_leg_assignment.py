"""Compare annotation conventions read-only; neither convention is biological proof."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

COMMIT = 'bbbfd7765172c6cafc30023d1786d8cb3fcdf7ac'
LEGS = ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    source = subprocess.check_output(['git', '-C', str(root), 'show',
        COMMIT + ':gpu-pilot/src/noesis_gpu/credit_timing.py'], text=True)
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'assign_legs']
    if len(nodes) != 1:
        raise ValueError('Expected one immutable assignment function')
    region = {'T1_PRONM': 0, 'T2_MESONM': 1, 'T3_METANM': 2}
    ns = {'np': np, 'NEUROMERE': region}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<immutable-assign-legs>', 'exec'), ns)
    path = root / 'data/processed/neurons.parquet'
    neurons = pd.read_parquet(path)
    if not np.array_equal(neurons.index, np.arange(len(neurons))):
        raise ValueError('Metadata row/index identity required')
    ids = np.arange(len(neurons))
    old = ns['assign_legs'](neurons, ids)
    nerve = old.copy()
    for i, row in neurons.iterrows():
        side = re.match(r'^(left|right)_', str(row['Nerve']))
        position = region.get(str(row['Top in/out region']))
        if side and position is not None:
            nerve[i] = position + (0 if side.group(1) == 'left' else 3)
    sys.path.insert(0, str(root / 'outputs/codex-e3-core-42c2076/src'))
    from noesis.motor import leg_motor_table
    table = leg_motor_table(neurons)
    motor = dict(zip(table['idx'].astype(int), table['leg']))

    def summarize(selected):
        selected = np.asarray(selected, dtype=int)
        a, b = old[selected], nerve[selected]
        changed = selected[a != b]
        comparable = [i for i in selected if i in motor]
        return dict(count=len(selected), soma_unknown=int(np.sum(a < 0)),
            nerve_first_unknown=int(np.sum(b < 0)), assignment_changed=len(changed),
            previously_unknown_now_assigned=int(np.sum((a < 0) & (b >= 0))),
            assigned_side_changed=int(np.sum((a >= 0) & (b >= 0) & (a != b))),
            nerve_first_counts={leg: int(np.sum(b == k)) for k, leg in enumerate(LEGS)},
            changed_class_counts=neurons.iloc[changed]['Class'].value_counts().astype(int).to_dict(),
            motor_table_comparable=len(comparable),
            motor_table_comparable_soma_unknown=int(sum(old[i] < 0 for i in comparable)),
            motor_table_known_side_disagreements=int(sum(old[i] >= 0 and LEGS[old[i]] != motor[i] for i in comparable)),
            unknown_motor_region_counts=neurons.iloc[[i for i in comparable if old[i] < 0]]['Top in/out region'].fillna('<missing>').value_counts().astype(int).to_dict(),
            soma_disagrees_with_actual_motor_table=int(sum(old[i] < 0 or LEGS[old[i]] != motor[i] for i in comparable)),
            nerve_first_disagrees_with_actual_motor_table=int(sum(nerve[i] < 0 or LEGS[nerve[i]] != motor[i] for i in comparable)))

    births = [root / 'gpu-pilot/outputs/logging-parity-d3-20261030/logged',
              root / 'gpu-pilot/outputs/credit-replication-c2-20261030/train-1001',
              root / 'gpu-pilot/outputs/credit-replication-c2-20261030/train-1002']
    with np.load(births[0] / 'exploration-log.npz', allow_pickle=False) as log:
        drive, credit = log['drive_idx'].copy(), log['motor_indices'].copy()
    for birth in births[1:]:
        with np.load(birth / 'exploration-log.npz', allow_pickle=False) as log:
            if not np.array_equal(drive, log['drive_idx']) or not np.array_equal(credit, log['motor_indices']):
                raise ValueError('Birth index identity mismatch')
    result = dict(status='completed_annotation_convention_audit', candidate_commit=COMMIT,
        candidate_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        groups={'logged_drive': summarize(drive), 'logged_credit': summarize(credit),
                'all_leg_motor': summarize(np.flatnonzero(neurons['Class'].eq('leg_motor_neuron')))},
        limits='Current metadata only; historical cache identity unverified. Agreement with existing motor table establishes implementation consistency, not biological validity or physical benefit.',
        simulation_started=False, weights_changed=False)
    serialized = json.dumps(result, indent=2)
    with args.output.open('x', encoding='utf-8') as out:
        out.write(serialized + '\n')
    print(serialized)


if __name__ == '__main__':
    main()

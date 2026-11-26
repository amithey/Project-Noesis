"""Isolated legacy/corrected pad-sign probe, driven solely by native neural desired angles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from adhesion_mapping import negative_femur_adhesion

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--mode',choices=('legacy','corrected'),required=True)
ap.add_argument('--seed',type=int,required=True)
for name in ('root','output'):
    ap.add_argument('--'+name,type=Path,required=True)
args=ap.parse_args()
if args.seed in (10,11,200,201,202):
    ap.error('Final physical validation seeds prohibited')
root=args.root.resolve()
pilot=root/'work/Noesis-Sensory'
core=root/'work/Noesis-Core/src'
parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
sys.path[:0]=[str(pilot/'gpu-pilot/src'),str(core)]
from noesis_gpu import controller,run

old=controller.adhesion_from_command
records=[]


def observed(desired,rest):
    pads=old(desired,rest) if args.mode=='legacy' else negative_femur_adhesion(desired,rest)
    records.append((np.asarray(desired).copy(),(np.asarray(desired)-rest).reshape(6,7)[:,3].copy(),pads.copy()))
    return pads


controller.adhesion_from_command=observed
try:
    options=run.parser().parse_args(['--backend','cpu','--worlds','1','--seconds','5','--wall-seconds','300',
                    '--seed',str(args.seed),'--noesis-source',str(core),'--data',str(root/'work/NoesisData'),
                    '--checkpoint',str(parent),'--output',str(args.output)])
    report=run.run(options)
finally:
    controller.adhesion_from_command=old
if report['status']!='completed_transport_pilot' or len(records)!=1000:
    raise RuntimeError('Incomplete pad-sign probe')
desired,delta,pads=(np.stack([r[i] for r in records]) for i in range(3))
assert desired.shape==(1000,42) and delta.shape==pads.shape==(1000,6)
assert np.isfinite(desired).all() and np.isfinite(delta).all() and np.isin(pads,[0.,1.]).all()
np.savez_compressed(args.output/'adhesion-observations.npz',desired=desired,Femur_delta=delta,pads=pads)
report['body']['adhesion_mapping']='legacy-positive-Femur' if args.mode=='legacy' else 'corrected-negative-Femur'
report.update(adhesion_probe_mode=args.mode,neural_desired_angles_drive_pads=True,
              observer_module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              mapping_module_sha256=hashlib.sha256(Path(__file__).with_name('adhesion_mapping.py').read_bytes()).hexdigest())
(args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:report[k] for k in ('status','adhesion_probe_mode','completed_control_chunks','weights_unchanged')}))

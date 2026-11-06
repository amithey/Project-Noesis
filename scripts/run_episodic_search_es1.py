"""ES1: one antithetic original-bounded synaptic search direction, native frozen physics."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import psutil

from es1_proposal import propose, write_candidate, assess

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
root = ap.parse_args().root.resolve()
pilot = root/'work/Noesis-Sensory'
core = root/'work/Noesis-Core/src'
data = root/'work/NoesisData'
parent = root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
historical = root/'work/evidence/s1-restored/gpu-pilot/outputs/standing-profile-s1-20261101'
out = root/'outputs/episodic-search-es1-20261106'
expected_parent = '9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
seeds, fresh = [10006,10007,10010,10011], [10020,10021]
assert not set(seeds+fresh) & {10,11,200,201,202}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(path):
    h = hashlib.sha256()
    for p in sorted(path.rglob('*.py')):
        h.update(p.relative_to(path).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def source_hashes():
    return dict(pilot=tree(pilot/'gpu-pilot/src'),core=tree(core),runner=sha(Path(__file__)),
                proposer=sha(Path(__file__).with_name('es1_proposal.py')))


def exact(a, b):
    with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
        return set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)


mine = {os.getpid()} | {p.pid for p in psutil.Process().parents()}


def resources():
    for p in psutil.process_iter(['pid','name']):
        if p.info['pid'] not in mine and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe'):
            raise RuntimeError('Competing Python process; no new worker')
    ram = psutil.virtual_memory().available/2**30
    if ram < 2:
        raise RuntimeError('Available RAM below2GiB')
    return ram


sys.path[:0] = [str(pilot/'gpu-pilot/src'),str(core)]
from noesis_gpu.evaluate import summarize, compare
from noesis_gpu.controller import import_checkpoint
from noesis_gpu.process_watchdog import run_windows_job, WatchdogError
from noesis.embodiment import load_training_brain
from noesis.lif import LIFNetwork
from noesis.plasticity import MotorPlasticity

assert sha(parent)==expected_parent
for item in json.loads((root/'outputs/noesis-required-data-sha256.json').read_text())['files']:
    assert sha(data/item['path'])==item['sha256']
provenance = json.loads((root/'outputs/thermo-pilot-tp2-20261106-source.json').read_text())
for label, folder in [('pilot',pilot/'gpu-pilot/src'),('core',core)]:
    assert len(list(folder.rglob('*.py')))==len(provenance[label])
    for item in provenance[label]:
        assert sha(folder/item['path'])==item['sha256'], 'Unreviewed runtime source'
hashes = source_hashes()
protocol = dict(experiment='ES1 single-generation episodic synaptic search',date='2026-11-06',
                parent_sha256=expected_parent, runtime_commit='e0cbd109bb6e018bd253c9f1b5e1698e0421b276',
                core_commit='42c20769376bda5694b4c1c66ca75e52cbd9f1f4',source_hashes=hashes,
                direction_seed=60061,epsilon_fraction_original_magnitude=.02,
                proposal='w_parent +/- .02*abs(w_original)*Rademacher[post_target]; reject clipping/sign/bound violations',
                dev_seeds=seeds,conditional_fresh_dev_seeds=fresh,seconds_per_job=5,
                process_budget_seconds=300,aggregate_budget_seconds=1800,
                gate='standing>=.90 AND>=parent, slip<=parent EACH seed; median heading AND net-forward gains>=.25mm AND positive candidate medians; median cycles/tripod not lower',
                selection='Among passing dev banks maximize min(median heading gain,median net gain); tieplus. Only selected gets same-gate fresh followup; no fallback.',
                parent_replay='each reused-seed parent native array must exactly match historical S1 parent before candidates',
                stop='timeout/error/incomplete/nonfinite/context/hash/replay/resource failure; no retries, scale changes or added directions',
                within_episode_learning=False,motor_exploration=0,external_senses=False,
                model_promoted=False,walking_proven=False)
resources()
out.mkdir(exist_ok=False)
lock = root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock'
token = uuid.uuid4().hex
with os.fdopen(os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY),'w') as f:
    json.dump(dict(owner='codex',pid=os.getpid(),token=token,start_unix=time.time(),deadline_unix=time.time()+1800),f)
started = time.monotonic()
result = dict(status='running',cases=[],parent_replay={},development_screen={},selected_bank=None,
              model_promoted=False,walking_proven=False)
tree_empty = True
bank_paths = dict(parent=parent)
bank_hashes = dict(parent=expected_parent)
env = dict(os.environ,PYTHONPATH=str(pilot/'gpu-pilot/src'),NOESIS_SOURCE=str(core),PYTHONUTF8='1',
           MPLCONFIGDIR=str(root/'work/matplotlib-cache'))


def run_case(bank, seed, check_historical=False):
    global tree_empty
    if source_hashes()!=hashes or any(sha(bank_paths[k])!=bank_hashes[k] for k in bank_paths):
        raise RuntimeError('Source or bank changed')
    if time.monotonic()-started > 1500:
        raise RuntimeError('Remaining budget below300s')
    ram = resources()
    target = out/f'{bank}-{seed}'
    command = [sys.executable,'-m','noesis_gpu.run','--backend','cpu','--worlds','1','--seconds','5',
               '--wall-seconds','300','--seed',str(seed),'--noesis-source',str(core),'--data',str(data),
               '--checkpoint',str(bank_paths[bank]),'--output',str(target)]
    print('START',bank,seed,flush=True)
    tree_empty = False
    try:
        with (out/f'{bank}-{seed}.log').open('w',encoding='utf-8') as log:
            receipt = run_windows_job(command,timeout=300,stdout=log,cwd=pilot,env=env)
        tree_empty = receipt['tree_empty']
    except WatchdogError as error:
        tree_empty = error.cleanup_verified
        raise
    write(out/f'{bank}-{seed}-receipt.json',receipt)
    assert tree_empty and not receipt['timed_out'] and receipt['exit_code']==0
    report, summary = summarize(target)
    assert report['checkpoint_sha256']==bank_hashes[bank] and report['weights_unchanged']
    if check_historical:
        assert exact(target/'observations.npz',historical/f'frozen-parent-{seed}/observations.npz'), 'Historical parent replay failed'
        result['parent_replay'][str(seed)]=True
    elif bank!='parent':
        compare(out/f'parent-{seed}',target)  # native matched context/horizon checks, not gate relaxation
    row = dict(bank=bank,seed=seed,summary=summary,process_receipt=receipt,available_ram_before_gib=ram,
               report_sha256=sha(target/'report.json'),observations_sha256=sha(target/'observations.npz'))
    result['cases'].append(row)
    write(out/'result.json',result)
    print('COMPLETE',bank,seed,'standing',summary['standing_fraction'],'heading',summary['heading_forward_distance_mm'],flush=True)
    return summary


try:
    write(out/'protocol.json',protocol)
    cn,_ = load_training_brain(data,out/'private-graph-cache')
    with np.load(parent,allow_pickle=False) as z:
        targets,weights,reference = z['motor_indices'].copy(),z['weights'].copy(),z['initial_weights'].copy()
    net = LIFNetwork(cn.W)
    loader = MotorPlasticity(net,targets,learning_rate=.2,tau_eligibility_ms=100,max_change=.5,
                             rule='perturbation',normalize_reward_error=True,bias_correct_reward_error=True)
    loader.load(parent,weights_only=True)
    np.testing.assert_array_equal(loader.initial_weights,reference)
    slots = np.searchsorted(targets,net.indices[loader.edge_indices])
    direction = np.random.default_rng(60061).choice(np.array([-1,1],dtype=np.int8),len(targets))
    proposals = propose(weights,reference,slots,direction)
    for bank,proposal in zip(('plus','minus'),proposals):
        path = out/f'weights-proposed-{bank}.npz'
        write_candidate(parent,path,proposal)
        checked = LIFNetwork(cn.W)
        import_checkpoint(checked,path)
        np.testing.assert_array_equal(checked.data[loader.edge_indices],proposal)
        mask = np.ones(len(checked.data),dtype=bool); mask[loader.edge_indices]=False
        np.testing.assert_array_equal(checked.data[mask],np.asarray(cn.W.data,float)[mask])
        bank_paths[bank],bank_hashes[bank] = path,sha(path)
        write(path.with_suffix('.model.json'),dict(method='ES1 untrained synaptic proposal; candidate only',
              parent_sha256=expected_parent,weights_sha256=bank_hashes[bank],epsilon=.02,direction_seed=60061,
              within_episode_learning=False,traces_reset=True,model_promoted=False,walking_proven=False))
        del checked
    write(out/'proposal.json',dict(bank_sha256=bank_hashes,targets=len(targets),plastic_edges=len(weights),
          direction_sha256=hashlib.sha256(direction.tobytes()).hexdigest(),
          max_abs_original_relative_step=float(np.max(np.abs((proposals[0]-weights)/reference))),
          antithetic_max_center_error=float(np.max(np.abs((proposals[0]+proposals[1])/2-weights))),
          signs_topology_original_bounds_validated=True,nonplastic_weights_exact=True,no_clipping=True))
    del cn,net,loader,mask,proposals,weights,reference,slots
    gc.collect()
    summaries = {bank:{} for bank in bank_paths}
    for seed in seeds:
        for bank in ('parent','plus','minus'):
            summaries[bank][seed] = run_case(bank,seed,check_historical=bank=='parent')
    for bank in ('plus','minus'):
        result['development_screen'][bank]=assess(summaries['parent'],summaries[bank],seeds)
    eligible = [bank for bank in ('plus','minus') if result['development_screen'][bank]['passed']]
    if eligible:
        selected = max(eligible,key=lambda bank:result['development_screen'][bank]['ranking'])
        result['selected_bank']=selected
        a,b = {},{}
        for seed in fresh:
            a[seed]=run_case('parent',seed)
            b[seed]=run_case(selected,seed)
        result['fresh_screen']=assess(a,b,fresh)
        result['status']='completed_provisional_child' if result['fresh_screen']['passed'] else 'completed_fresh_rejection'
    else:
        result['status']='completed_no_candidate'
    result['bank_sha256']=bank_hashes
except BaseException as error:
    result.update(status='failed',error_type=type(error).__name__,error=str(error))
    raise
finally:
    result.update(seconds=time.monotonic()-started,parent_unchanged=sha(parent)==expected_parent,
                  source_unchanged=source_hashes()==hashes,process_tree_empty_verified=tree_empty)
    if tree_empty and lock.exists() and json.loads(lock.read_text())['token']==token:
        lock.unlink()
    result['lease_released']=not lock.exists()
    write(out/'result.json',result)
    print('RESULT',result['status'],'seconds',result['seconds'],flush=True)

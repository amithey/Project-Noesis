"""Signed read-only OA2: prepare inventory, then audit only original training data.

No simulator, checkpoint save, objective deployment or held-out-data selection.
The alternative load reference is quasi-static and omits acceleration/other
applied forces/calibration. Hypothetical decisions are not efficacy evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from oa2_decision_sensitivity import case_key,counterfactual_returns,decision_diagnostic
from mp1_contract import validate_report,return_record

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/measured-policy-mp1-20261110'
INDEX=ROOT/'outputs/oa2-input-index-private-20261113.json'
SOURCE=('oa2_decision_sensitivity.py','test_oa2_decision_sensitivity.py','audit_oa2_decision_sensitivity.py')
FILES=dict(report='report.json',observations='observations.npz',anatomy='anatomy-observations.npz',model='anatomy-model.json')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def relative(p):return p.relative_to(ROOT).as_posix()


def local(n):
    p=(ROOT/n).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Outside workspace')
    return p


def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def prepare():
    if INDEX.exists():raise ValueError('Inventory already exists; do not overwrite')
    # Existing sealed-prefix provenance predates this posthoc hypothesis.
    previous=read(ROOT/'outputs/mp1-interruption-audit-private-20261113.json')
    pins=previous['original_hashes'].copy()
    for n,h in pins.items():
        if sha(local(n))!=h:raise ValueError('Original sealed evidence changed')
    for n in SOURCE:pins['work/'+n]=sha(ROOT/'work'/n)
    for n in ('oa1_load_reference.py','ps1_objective.py','ew1_contract.py','mp1_select.py','mp1_contract.py'):
        path='work/'+n
        if path in pins and pins[path]!=sha(local(path)):raise ValueError('Protected dependency changed')
        pins[path]=sha(local(path))
    names=[f'parent-{s}' for s in (1010,1011)]
    names += [f'trial-{d}-{s}-{sign}' for d in range(4) for s in (1010,1011) for sign in (-1,1)]
    cases=[]
    for name in names:
        files={k:relative(OUT/name/n) for k,n in FILES.items()}
        extra=[OUT/(name+'-mp1-context.json'),OUT/(name+'-receipt.json')]
        for p in [*[local(n) for n in files.values()],*extra]:
            n=relative(p)
            if n not in pins or sha(p)!=pins[n]:raise ValueError('Unsealed training input')
        c=read(local(files['report']))['mp1_context'];case_key(c)
        cases.append(dict(name=name,files=files,context=c))
    index=dict(cases=cases,pins=pins,offline_seconds=180,new_workers=0,
        limits='TRAIN ONLY, hypothetical same-buffer decisions; no adoption or efficacy, original failed verdict preserved')
    INDEX.write_text(json.dumps(index,indent=2),encoding='utf-8')
    protocol=dict(study='OA2 training-buffer decision sensitivity',training_parent_controls=2,training_probes=16,
        new_simulation_workers=0,offline_seconds=180,warmup_samples=10,return_samples=990,
        original_decision_reproduction_required=True,alternative='OA1 actuator-adjusted quasi-static load reference only',
        held_out_data_used_for_decisions=False,checkpoint_or_optimizer_write=False,
        stop='Any provenance, source, receipt, role, horizon, return or saved-state mismatch; no retry or tuning',limits=index['limits'])
    (ROOT/'outputs/oa2-public-protocol-20261113.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    print('OA2 signed private inventory and compact protocol prepared')


def run():
    started=time.monotonic();index_hash=sha(INDEX);index=read(INDEX)
    if (ROOT/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists():raise ValueError('Heavy lease active')
    def verify():
        if sha(INDEX)!=index_hash:raise ValueError('Inventory changed')
        for n,h in index['pins'].items():
            if sha(local(n))!=h:raise ValueError('Signed input/source changed')
        if time.monotonic()-started>180:raise TimeoutError('Offline budget exceeded')
    verify();rows=[];records=[]
    if len(index['cases'])!=18:raise ValueError('Incomplete inventory')
    for case in index['cases']:
        p=OUT/case['name'];r=read(local(case['files']['report']));validate_report(r);c=r['mp1_context'];case_key(c)
        if c!=case['context'] or c!=read(p.with_name(p.name+'-mp1-context.json')):raise ValueError('Context mismatch')
        receipt=read(p.with_name(p.name+'-receipt.json'))
        if receipt['exit_code']!=0 or receipt['timed_out'] or not receipt['tree_empty']:raise ValueError('Incomplete worker')
        o=arrays(local(case['files']['observations']));a=arrays(local(case['files']['anatomy']));m=read(local(case['files']['model']))
        old,new=counterfactual_returns(o,a,m)
        rows.append(dict(context=c,original_return=old,alternative_return=new))
        if c['phase']=='trial':
            hashes={k:index['pins'][n] for k,n in case['files'].items()}
            records.append(return_record(r,o,a,m,hashes))
        if time.monotonic()-started>180:raise TimeoutError('Offline budget exceeded')
    if records!=read(OUT/'training-returns.json'):raise ValueError('Original raw returns not reproduced')
    diagnostic=decision_diagnostic(rows);saved=read(OUT/'selection.json');state=arrays(OUT/'optimizer-state.npz')
    np.testing.assert_array_equal(diagnostic['original']['parent_returns'],state['baseline_returns'])
    np.testing.assert_array_equal(diagnostic['original']['probe_returns'],state['policy_returns'])
    for bank in ('candidate','negative'):
        actual=diagnostic['original']['decisions'][bank];expected=saved[bank]
        if set(actual)!=set(expected):raise ValueError('Decision fields changed')
        for k in actual:
            if k=='worst_training_gain':np.testing.assert_allclose(actual[k],expected[k],rtol=1e-12,atol=1e-14)
            elif actual[k]!=expected[k]:raise ValueError('Original selection not reproduced')
    if json.loads(str(state['selection_json']))!=saved:raise ValueError('Saved state mismatch')
    verify()
    def serial(v):return v.tolist() if isinstance(v,np.ndarray) else v
    private=dict(status='completed_readonly_OA2',rows=rows,diagnostic=diagnostic,index_sha256=index_hash,
        pins=index['pins'],elapsed_seconds=time.monotonic()-started)
    (ROOT/'outputs/oa2-decision-sensitivity-private-20261113.json').write_text(json.dumps(private,default=serial,indent=2),encoding='utf-8')
    changed=diagnostic['policy_identity_changed']
    public=dict(status=private['status'],training_parent_controls=2,training_probes=16,
        original_raw_returns_reproduced=True,original_decisions_and_saved_state_reproduced=True,
        hypothetical_candidate_identity_changed=changed['candidate'],hypothetical_negative_identity_changed=changed['negative'],
        new_simulation_workers=0,held_out_data_used_for_decisions=False,checkpoint_or_optimizer_written=False,
        alternative_reference_adopted=False,original_failed_verdict_preserved=True,model_promoted=False,
        standing_proven=False,walking_proven=False,
        limits='Posthoc same-buffer hypothetical decision sensitivity only. Quasi-static reference omits acceleration, other applied forces and calibration. No causal explanation, better reward or efficacy established.')
    (ROOT/'outputs/oa2-decision-sensitivity-minimal-20261113.json').write_text(json.dumps(public,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(public))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('prepare','run'));a=ap.parse_args()
    prepare() if a.mode=='prepare' else run()

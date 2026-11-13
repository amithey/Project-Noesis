"""Verify reusable completed MP1 evidence without declaring an incomplete study complete."""
import json
import numpy as np
import run_measured_mp1 as old
from mp1_select import TRAIN_SEEDS,DEV,FRESH,STEP,MIN_GAIN,training_panel,matched_gradient
from mp1_contract import return_record


def jobs():
    result=[('baseline','baseline','parent',1004,None,None)]
    result += [(f'parent-{s}','evaluation','parent',s,None,None) for s in TRAIN_SEEDS]
    result += [(f'trial-{d}-{s}-{sign}','trial','probe',s,d,sign) for d in range(4) for s in TRAIN_SEEDS for sign in (-1,1)]
    result += [(f'parent-{s}','evaluation','parent',s,None,None) for s in FRESH]
    result += [(f'evaluation-{b}-{s}','evaluation',b,s,None,None) for b in ('candidate','negative','gradient') for s in (*TRAIN_SEEDS,*DEV,*FRESH)]
    assert len(result)==45
    return result


def verify():
    p=old.read(old.out/'protocol.json');r=old.read(old.out/'result.json');expected=jobs()
    assert r['status']=='failed' and len(r['jobs'])==30
    assert r['source_prior_and_created_unchanged'] and r['data_unchanged'] and r['process_tree_empty_verified'] and r['lease_released']
    assert not (old.root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    old.previous.previous.base.verify_native()
    pins={}
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert old.sha(old.root/n)==h;pins[n]=h
    for n,h in r['created_hashes'].items():assert old.sha(old.root/n)==h;pins[n]=h
    for entry in old.read(old.root/'outputs/noesis-required-data-sha256.json')['files']:
        assert old.sha(old.root/'work/NoesisData'/entry['path'])==entry['sha256']
    records=[]
    for job,(name,phase,bank,seed,d,sign) in zip(r['jobs'][:29],expected[:29]):
        assert [job[k] for k in ('name','phase','bank','seed','direction','sign')]==[name,phase,bank,seed,d,sign]
        receipt=old.read(old.out/(name+'-receipt.json'));assert receipt==job['receipt']
        assert receipt['exit_code']==0 and not receipt['timed_out'] and receipt['tree_empty'] and job['available_ram_before_gib']>=2
        report,o,a,m,_=old.measured(old.out/name);c=report['mp1_context']
        assert c==old.read(old.out/(name+'-mp1-context.json'))
        assert [c[k] for k in ('phase','bank','seed','direction','sign')]==[phase,bank,seed,d,sign]
        assert c['worker_sha256']==p['source_hashes']['work/mp1_worker.py']
        cp=old.root/p['parent_checkpoint'] if bank=='parent' else old.inputs/f'probe-{d}-{sign}.npz' if bank=='probe' else old.out/(bank+'.npz')
        assert c['checkpoint_sha256']==old.sha(cp)
        imported=c['import_records'][0]
        assert imported['actual_import_exact'] and imported['topology_unchanged'] and imported['nonselected_weights_unchanged']
        if phase=='trial':
            old.native.compare(old.reference(p,'parent',seed),old.out/name)
            sources={k:old.sha(old.out/name/n) for k,n in [('report','report.json'),('observations','observations.npz'),('anatomy','anatomy-observations.npz'),('model','anatomy-model.json')]}
            records.append(return_record(report,o,a,m,sources))
    failed=r['jobs'][-1];assert failed['name']==expected[29][0]
    receipt=old.read(old.out/(failed['name']+'-receipt.json'));assert receipt==failed['receipt']
    assert receipt['exit_code']==124 and receipt['timed_out'] and receipt['tree_empty']
    assert records==old.read(old.out/'training-returns.json') and len(records)==16
    returns=training_panel(records);baseline=np.array([float(old.measured(old.out/f'parent-{s}')[-1][10:].mean()) for s in TRAIN_SEEDS])
    decisions=old.read(old.out/'selection.json')
    for bank,sign in (('candidate',1),('negative',-1)):
        margins=[min(sign*float(returns[row,index]-baseline[row]) for row in range(2)) for index in range(8)]
        best=max(range(8),key=lambda i:margins[i]);index=best if margins[best]>MIN_GAIN else -1
        assert decisions[bank]['policy_index']==index and decisions[bank]['parent_fallback']==(index<0)
        d,k=divmod(index,2);source=old.root/p['parent_checkpoint'] if index<0 else old.inputs/f'probe-{d}-{-1 if k==0 else 1}.npz'
        assert old.sha(source)==old.sha(old.out/(bank+'.npz'))
    state=old.arrays(old.out/'optimizer-state.npz')
    np.testing.assert_array_equal(state['policy_returns'],returns);np.testing.assert_array_equal(state['baseline_returns'],baseline)
    gradient=matched_gradient(state['directions'],returns)
    np.testing.assert_array_equal(gradient['relative_delta'],state['gradient_delta'])
    expected_gradient=old.apply_delta(old.arrays(old.root/p['parent_checkpoint']),state['gradient_delta'])
    actual_gradient=old.arrays(old.out/'gradient.npz')
    for k in expected_gradient:np.testing.assert_array_equal(actual_gradient[k],expected_gradient[k])
    index=decisions['candidate']['policy_index'];d,k=divmod(index,2);sign=-1 if k==0 else 1
    for seed in TRAIN_SEEDS:
        reference=old.out/f'parent-{seed}' if index<0 else old.out/f'trial-{d}-{seed}-{sign}'
        candidate=old.out/f'evaluation-candidate-{seed}'
        for n in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):old.native.exact(reference/n,candidate/n)
        assert old.read(reference/'report.json')['spikes_per_world']==old.read(candidate/'report.json')['spikes_per_world']
    for path in (old.out/'result.json',old.out/'protocol.json',old.out/'training-returns.json',old.out/(failed['name']+'-receipt.json'),old.out/(failed['name']+'.log')):
        pins[path.relative_to(old.root).as_posix()]=old.sha(path)
    return dict(status='verified_incomplete_MP1_prefix',completed_workers=29,failed_attempts=1,missing_workers=16,
        original_study_complete=False,scientific_verdict_available=False,candidate_training_replay_exact=True,
        original_hashes=pins,failed_receipt=receipt,model_promoted=False)


def main():
    payload=verify();old.write(old.root/'outputs/mp1-interruption-audit-private-20261113.json',payload)
    public={k:v for k,v in payload.items() if k not in ('original_hashes','failed_receipt')}
    public['technical_cause']='Verified automatic host idle sleep overlapped the timed-out worker'
    old.write(old.root/'outputs/mp1-interruption-minimal-20261113.json',public)
    print('29completed cells verified, original failure preserved,16cells still missing; no efficacy verdict')


if __name__=='__main__':main()

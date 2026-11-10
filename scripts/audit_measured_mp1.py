"""Terminal MP1 raw provenance and independently derived policy-selection audit."""
import json
import numpy as np
import run_measured_mp1 as run
from mp1_select import TRAIN_SEEDS,DEV,FRESH,STEP,MIN_GAIN,fixed_directions,training_panel
from mp1_contract import return_record
from ew1_update import apply_delta

root,out,read,write,sha,arrays=run.root,run.out,run.read,run.write,run.sha,run.arrays


def equivalent(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equivalent(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equivalent(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-14)
    else:assert a==b


def main():
    result=read(out/'result.json');p=read(out/'protocol.json')
    assert result['status']=='completed_MP1' and len(result['jobs'])==45
    assert result['source_prior_and_created_unchanged'] and result['data_unchanged'] and result['process_tree_empty_verified'] and result['lease_released']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    run.previous.previous.base.verify_native()
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    for n,h in result['created_hashes'].items():assert sha(root/n)==h
    for entry in read(root/'outputs/noesis-required-data-sha256.json')['files']:
        assert sha(root/'work/NoesisData'/entry['path'])==entry['sha256']
    parent_path=root/p['parent_checkpoint'];parent=arrays(parent_path);directions=fixed_directions(len(parent['weights']))
    np.testing.assert_array_equal(arrays(run.inputs/'directions.npz')['directions'],directions)
    for d in range(4):
        for sign in (-1,1):
            actual=arrays(run.inputs/f'probe-{d}-{sign}.npz');expected=apply_delta(parent,sign*STEP*directions[d])
            for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
    jobs=[('baseline','baseline','parent',1004,None,None)]
    jobs += [(f'parent-{s}','evaluation','parent',s,None,None) for s in TRAIN_SEEDS]
    jobs += [(f'trial-{d}-{s}-{sign}','trial','probe',s,d,sign) for d in range(4) for s in TRAIN_SEEDS for sign in (-1,1)]
    jobs += [(f'parent-{s}','evaluation','parent',s,None,None) for s in FRESH]
    jobs += [(f'evaluation-{b}-{s}','evaluation',b,s,None,None) for b in ('candidate','negative','gradient') for s in (*TRAIN_SEEDS,*DEV,*FRESH)]
    records=[]
    for job,(name,phase,bank,seed,d,sign) in zip(result['jobs'],jobs):
        assert [job[k] for k in ('name','phase','bank','seed','direction','sign')]==[name,phase,bank,seed,d,sign]
        receipt=read(out/(name+'-receipt.json'));assert receipt==job['receipt']
        assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and job['available_ram_before_gib']>=2
        report,o,a,m,_=run.measured(out/name);c=report['mp1_context']
        assert c==read(out/(name+'-mp1-context.json'))
        assert [c[k] for k in ('phase','bank','seed','direction','sign')]==[phase,bank,seed,d,sign]
        assert c['worker_sha256']==p['source_hashes']['work/mp1_worker.py']
        cp=parent_path if bank=='parent' else run.inputs/f'probe-{d}-{sign}.npz' if bank=='probe' else out/(bank+'.npz')
        assert c['checkpoint_sha256']==sha(cp)
        imported=c['import_records'][0];weights=arrays(cp)
        assert imported['topology_unchanged'] and imported['nonselected_weights_unchanged']
        assert imported['existing_selected_edges']==len(parent['weights'])
        assert imported['changed_selected_edges']==int(np.count_nonzero(weights['weights']!=weights['initial_weights']))
        if phase=='trial':
            run.native.compare(run.reference(p,'parent',seed),out/name)
            sources={k:sha(out/name/n) for k,n in [('report','report.json'),('observations','observations.npz'),('anatomy','anatomy-observations.npz'),('model','anatomy-model.json')]}
            records.append(return_record(report,o,a,m,sources))
        run.native.summarize(out/name)
    assert records==read(out/'training-returns.json') and len(records)==16
    for n in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):
        run.native.exact(out/'baseline'/n,root/p['baseline_folder']/n)
    assert read(out/'baseline/report.json')['spikes_per_world']==read(root/p['baseline_folder']/'report.json')['spikes_per_world']
    run.native.compare(root/p['baseline_folder'],out/'baseline')
    returns=training_panel(records)
    baseline=np.array([float(run.measured(out/f'parent-{s}')[-1][10:].mean()) for s in TRAIN_SEEDS])
    decisions={}
    for bank,sign in (('candidate',1),('negative',-1)):
        margins=[min(sign*(float(returns[row,index])-float(baseline[row])) for row in range(2)) for index in range(8)]
        best=max(range(8),key=lambda index:margins[index]);accepted=margins[best]>MIN_GAIN
        decisions[bank]=dict(accepted_for_training_update=accepted,policy_index=best if accepted else -1,
            worst_training_gain=float(margins[best]),reward_sign=sign,parent_fallback=not accepted,
            training_condition_not_independent_efficacy=True,model_promoted=False)
    equivalent(decisions,read(out/'selection.json'))
    assert result['learning']['candidate_update_applied']==decisions['candidate']['accepted_for_training_update']
    assert result['learning']['reload_reproduces_weights'] and result['learning']['completed_rounds']==1
    state=arrays(out/'optimizer-state.npz')
    np.testing.assert_array_equal(state['directions'],directions)
    np.testing.assert_array_equal(state['policy_returns'],returns);np.testing.assert_array_equal(state['baseline_returns'],baseline)
    assert int(state['completed_rounds'])==1 and str(state['parent_sha256'])==p['parent_sha256']
    assert float(state['step'])==STEP and float(state['minimum_gain'])==MIN_GAIN
    np.testing.assert_array_equal(state['training_seeds'],TRAIN_SEEDS)
    assert json.loads(str(state['return_records_json']))==records
    equivalent(decisions,json.loads(str(state['selection_json'])))
    for bank in ('candidate','negative'):
        index=decisions[bank]['policy_index'];d,k=divmod(index,2)
        source=parent_path if index<0 else run.inputs/f'probe-{d}-{-1 if k==0 else 1}.npz'
        assert sha(out/(bank+'.npz'))==sha(source)
        sidecar=read(out/(bank+'.model.json'))
        assert sidecar['measured_policy_source_sha256']==sha(source) and sidecar['checkpoint_sha256']==sha(out/(bank+'.npz'))
        assert sidecar['accepted_for_training_update']==decisions[bank]['accepted_for_training_update'] and not sidecar['model_promoted']
    coefficients=np.array([sum(float(returns[row,2*d+1]-returns[row,2*d]) for row in range(2))/(2*2*STEP) for d in range(4)])
    gradient=sum((coefficients[d]*directions[d]/4 for d in range(4)),start=np.zeros(len(parent['weights'])))
    norm=float(abs(gradient).max());delta=np.zeros_like(gradient) if norm<=1e-12 else STEP*gradient/norm
    np.testing.assert_allclose(state['gradient'],gradient,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(state['gradient_coefficients'],coefficients,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(state['gradient_delta'],delta,rtol=1e-12,atol=1e-14)
    expected=apply_delta(parent,state['gradient_delta']);actual=arrays(out/'gradient.npz')
    for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
    for bank in ('candidate','negative','gradient'):
        sidecar=read(out/(bank+'.model.json'))
        assert sidecar['common_parent_sha256']==p['parent_sha256'] and sidecar['checkpoint_sha256']==sha(out/(bank+'.npz'))
        assert sidecar['completed_rounds']==1 and sidecar['training_trial_count']==16 and sidecar['native_online_learning'] is False
        assert not sidecar['model_promoted'] and not sidecar['walking_proven']
    science=run.finish(p);equivalent(json.loads(json.dumps(science)),result['science'])
    summary=dict(status='completed_independent_MP1_audit',full_workers=45,verified_training_trials=16,
        measured_policy_selection_independently_recomputed=True,selected_policy_byte_identity_verified=True,
        matched_gradient_independently_recomputed=True,exact_baseline=True,optimizer_reload_verified=True,
        training_acceptance=decisions,comparison=science['comparison'],model_promoted=False,standing_proven=False,walking_proven=False,
        limits='One selection round; in-sample acceptance is by construction. Fixed engineering objective and legacy metrics are not final biological/robustness qualification; no full spike reconstruction.')
    write(root/'outputs/measured-policy-mp1-20261110-summary.json',summary)
    print('MP1 independent terminal audit passed; detailed private evidence not published')


if __name__=='__main__':main()

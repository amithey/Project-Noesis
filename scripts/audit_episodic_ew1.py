"""Terminal EW1 audit; full local evidence maps are never a public result table."""
import json
import numpy as np
import run_episodic_ew1 as run
from ew1_update import TRAIN_SEEDS,DIRECTIONS,PROBE_STEP,UPDATE_STEP,fixed_directions,apply_delta,training_panel
from ew1_contract import return_record
from ew1_worker import DEV

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
    assert result['status']=='completed_EW1' and len(result['jobs'])==31
    assert result['source_prior_and_created_unchanged'] and result['data_unchanged'] and result['process_tree_empty_verified'] and result['lease_released']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    run.previous.base.verify_native()
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    for n,h in result['created_hashes'].items():assert sha(root/n)==h
    for entry in read(root/'outputs/noesis-required-data-sha256.json')['files']:
        assert sha(root/'work/NoesisData'/entry['path'])==entry['sha256']
    parent=arrays(root/p['parent_checkpoint']);dirs=fixed_directions(len(parent['weights']))
    np.testing.assert_array_equal(arrays(run.inputs/'directions.npz')['directions'],dirs)
    for d in range(DIRECTIONS):
        for sign in (-1,1):
            cp=arrays(run.inputs/f'probe-{d}-{sign}.npz');expected=apply_delta(parent,sign*PROBE_STEP*dirs[d])
            for k in expected:np.testing.assert_array_equal(cp[k],expected[k])
    jobs=[('baseline','baseline','parent',1004,None,None)]
    jobs += [(f'parent-{s}','evaluation','parent',s,None,None) for s in TRAIN_SEEDS]
    jobs += [(f'trial-{d}-{s}-{sign}','trial','probe',s,d,sign) for d in range(DIRECTIONS) for s in TRAIN_SEEDS for sign in (-1,1)]
    jobs += [(f'evaluation-{b}-{s}','evaluation',b,s,None,None) for b in ('candidate','negative') for s in (*TRAIN_SEEDS,*DEV)]
    records=[]
    for actual,(name,phase,bank,seed,direction,sign) in zip(result['jobs'],jobs):
        assert [actual[k] for k in ('name','phase','bank','seed','direction','sign')]==[name,phase,bank,seed,direction,sign]
        receipt=read(out/(name+'-receipt.json'));assert receipt==actual['receipt']
        assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and actual['available_ram_before_gib']>=2
        report,o,a,m,_=run.measured(out/name);context=report['ew1_context']
        assert context==read(out/(name+'-ew1-context.json'))
        assert context['phase']==phase and context['bank']==bank and context['seed']==seed
        assert context['direction']==direction and context['sign']==sign
        assert context['worker_sha256']==p['source_hashes']['work/ew1_worker.py']
        checkpoint=root/p['parent_checkpoint'] if bank=='parent' else run.inputs/f'probe-{direction}-{sign}.npz' if bank=='probe' else out/(bank+'.npz')
        assert context['checkpoint_sha256']==sha(checkpoint)
        imported=context['import_records'][0];cp=arrays(checkpoint)
        assert imported['topology_unchanged'] and imported['nonselected_weights_unchanged']
        assert imported['existing_selected_edges']==len(parent['weights'])
        assert imported['changed_selected_edges']==int(np.count_nonzero(cp['weights']!=cp['initial_weights']))
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
    plus,minus=training_panel(records)
    # Independently derive the gradient and update, not by calling estimate_update.
    coefficients=(plus-minus).sum(axis=0)/(len(TRAIN_SEEDS)*2*PROBE_STEP)
    gradient=np.zeros(len(parent['weights']))
    for d in range(DIRECTIONS):gradient+=coefficients[d]*dirs[d]/DIRECTIONS
    norm=float(abs(gradient).max());delta=np.zeros_like(gradient) if norm<=1e-12 else UPDATE_STEP*gradient/norm
    state=arrays(out/'optimizer-state.npz')
    np.testing.assert_allclose(state['gradient'],gradient,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(state['relative_delta'],delta,rtol=1e-12,atol=1e-14)
    np.testing.assert_array_equal(state['directions'],dirs)
    np.testing.assert_array_equal(state['plus_returns'],plus);np.testing.assert_array_equal(state['minus_returns'],minus)
    assert json.loads(str(state['return_records_json']))==records and int(state['completed_updates'])==1
    assert str(state['parent_sha256'])==p['parent_sha256'] and float(state['probe_step'])==PROBE_STEP and float(state['update_step'])==UPDATE_STEP
    for bank,sign in (('candidate',1),('negative',-1)):
        cp=arrays(out/(bank+'.npz'));expected=apply_delta(parent,sign*state['relative_delta'])
        for k in parent:np.testing.assert_array_equal(cp[k],expected[k])
        sidecar=read(out/(bank+'.model.json'))
        assert sidecar['common_parent_sha256']==p['parent_sha256'] and sidecar['checkpoint_sha256']==sha(out/(bank+'.npz'))
        assert sidecar['completed_updates']==1 and sidecar['training_trial_count']==16 and not sidecar['model_promoted']
    science=run.finish(p);equivalent(json.loads(json.dumps(science)),read(out/'result.json')['science'])
    summary=dict(status='completed_independent_EW1_audit',full_workers=31,verified_training_trials=16,
        completed_between_episode_updates=1,exact_baseline=True,actual_imports_verified=True,
        independent_return_and_update_recomputation=True,optimizer_reload_verified=True,
        comparison=science['comparison'],model_promoted=False,standing_proven=False,walking_proven=False,
        limits='One update, four directions and reused development bank. No biological efficacy, independent training replication or walking qualification; no full spike reconstruction.')
    write(root/'outputs/episodic-neural-ew1-20261110-summary.json',summary)
    print('EW1 independent terminal audit passed; detailed local evidence not published')


if __name__=='__main__':main()

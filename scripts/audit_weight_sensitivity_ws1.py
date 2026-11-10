"""Independent terminal WS1 integrity and raw output audit; results stay local."""
import numpy as np
import run_weight_sensitivity_ws1 as run

root,out,read,write,sha,arrays=run.root,run.out,run.read,run.write,run.sha,run.arrays


def main():
    p=read(out/'protocol.json');result=read(out/'result.json')
    assert result['status']=='completed_WS1' and len(result['jobs'])==11
    assert result['source_prior_unchanged'] and result['data_unchanged']
    assert result['process_tree_empty_verified'] and result['lease_released']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    run.base.verify_native()
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    for entry in read(root/'outputs/noesis-required-data-sha256.json')['files']:
        assert sha(root/'work/NoesisData'/entry['path'])==entry['sha256']
    parent=arrays(run.PARENT);weight_checks={}
    for variant,direction in (('minus',-1),('plus',1)):
        cp=run.inputs/(variant+'.npz');child=arrays(cp)
        assert set(child)==set(parent)
        for k in parent:
            if k!='weights':np.testing.assert_array_equal(child[k],parent[k])
        expected=parent['weights']+direction*.05*np.abs(parent['initial_weights'])
        np.testing.assert_array_equal(child['weights'],expected)
        np.testing.assert_array_equal(np.sign(child['weights']),np.sign(parent['initial_weights']))
        ratio=np.divide(np.abs(child['weights']),np.abs(parent['initial_weights']),
                        out=np.ones_like(expected),where=parent['initial_weights']!=0)
        assert ratio.min()>=.5 and ratio.max()<=1.5
        weight_checks[variant]=dict(checkpoint_sha256=sha(cp),only_weights_changed=True,
            changed_edges=int(np.count_nonzero(child['weights']!=parent['weights'])),
            original_reference_bounds_pass=True)
    expected_jobs=[('baseline',1004)]+[(v,s) for v in ('minus','plus') for s in run.SEEDS]
    imports={}
    for job,(variant,seed) in zip(result['jobs'],expected_jobs):
        name=f'{variant}-{seed}';d=out/name
        assert job['name']==name and job['variant']==variant and job['seed']==seed
        receipt=read(out/(name+'-receipt.json'));assert receipt==job['receipt']
        assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out']
        assert job['available_ram_before_gib']>=2
        report=read(d/'report.json');context=read(out/(name+'-ws1-context.json'))
        assert report['ws1_context']==context and report['ws1_worker_completed'] and context['worker_completed']
        cp=run.PARENT if variant=='baseline' else run.inputs/(variant+'.npz')
        assert report['checkpoint_sha256']==context['checkpoint_sha256']==sha(cp)
        assert report['seeds']==[seed] and report['controller']=='brain' and report['worlds']==1
        assert not report['learning'] and report['weights_unchanged'] and report['motor_exploration']==0
        assert report['status']=='completed_transport_pilot' and report['completed_control_chunks']==1000
        assert report['requested_simulated_seconds_per_world']==report['simulated_seconds_per_world']==5
        assert len(context['import_records'])==1
        imported=context['import_records'][0];cpdata=arrays(cp)
        assert imported['actual_import_exact'] and imported['nonselected_weights_unchanged'] and imported['topology_unchanged']
        assert imported['existing_selected_edges']==len(parent['weights'])
        assert imported['changed_selected_edges']==int(np.count_nonzero(cpdata['weights']!=cpdata['initial_weights']))
        imports.setdefault(variant,set()).add(imported['full_weights_sha256'])
        for filename in ('observations.npz','adhesion-observations.npz'):
            values=arrays(d/filename);assert all(len(x)==1000 and np.isfinite(x).all() for x in values.values())
        run.native.anatomy(d,'frozen');run.native.summarize(d)
    assert all(len(x)==1 for x in imports.values())
    assert len(set(next(iter(x)) for x in imports.values()))==3
    reference=run.history.reference('parent',1004);baseline=out/'baseline-1004'
    for filename in ('observations.npz','adhesion-observations.npz','anatomy-observations.npz'):
        run.native.exact(reference/filename,baseline/filename)
    assert read(reference/'report.json')['spikes_per_world']==read(baseline/'report.json')['spikes_per_world']
    # Independently derive exact per-sample responsiveness from saved arrays.
    responsiveness={}
    for variant in ('minus','plus'):
        responsiveness[variant]={}
        for seed in run.SEEDS:
            d=out/f'{variant}-{seed}';ref=run.history.reference('parent',seed);cell={}
            for filename,keys in [('adhesion-observations.npz',('desired','pads')),
                                  ('observations.npz',('angles','position','up_body','omega'))]:
                a,b=arrays(ref/filename),arrays(d/filename)
                for key in keys:
                    difference=b[key]-a[key];flat=difference.reshape(1000,-1)
                    changed=np.flatnonzero(np.count_nonzero(flat,axis=1))
                    value=dict(changed_samples=len(changed),max_abs_delta=float(abs(flat).max()),
                               first_changed_sample=int(changed[0]) if len(changed) else None)
                    assert value==result['sensitivity'][variant][str(seed)][key]
                    cell[key]=value
            responsiveness[variant][seed]=cell
    parents={s:run.native.summarize(run.history.reference('parent',s))[1] for s in run.SEEDS[1:]}
    retention={v:run.history.standing_retention(parents,
        {s:run.native.summarize(out/f'{v}-{s}')[1] for s in run.SEEDS[1:]},run.SEEDS[1:]) for v in ('minus','plus')}
    assert retention==result['strict_development_retention']
    summary=dict(status='completed_independent_WS1_audit',all_eleven_full_horizons=True,
        baseline_exact=True,actual_native_import_verified=True,weight_checks=weight_checks,
        responsiveness=responsiveness,strict_development_retention=retention,
        constructed_not_trained=True,model_promoted=False,standing_proven=False,walking_proven=False,
        limits='Sensitivity only; no learned ability, biological calibration, general efficacy or winner. No full neural spike replay reconstruction.')
    write(root/'outputs/weight-sensitivity-ws1-20261110-summary.json',summary)
    print('WS1 terminal independent integrity/raw-response audit passed; detailed result remains local')


if __name__=='__main__':main()

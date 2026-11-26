"""Independent terminal P2 raw audit. No neural or physical integration."""
import json
import numpy as np
import run_sensory_p2 as study
from p2_sensory_history import FIELDS

root,out=study.root,study.out
read,write,sha,arrays=study.read,study.write,study.sha,study.arrays


def physical_description(directory):
    # Independent raw reduction, matching the prospective endpoint-inclusive windows.
    a=arrays(directory/'p1-telemetry.npz');o=arrays(directory/'observations.npz');c=arrays(directory/'anatomy-observations.npz')
    result={}
    for label,start_index in (('post_pulse',393),('final_1s',800)):
        sl=slice(start_index,None);os=slice(start_index-1,None)
        speed=np.sqrt((a['com_velocity'][sl]**2).sum(-1));spin=np.sqrt((o['omega'][os,0]**2).sum(-1))
        result[label]=dict(samples=len(a['time'][sl]),fraction_up_below_point7=float(np.mean(o['up_body'][os,0,2]<.7)),
            tarsal_only_fraction=float(np.mean(c['contact_mode_code'][sl]==3)),mean_non_tarsal_vertical_fraction=float(np.mean(c['non_tarsal_fraction'][sl])),
            mean_COM_speed_mm_s=float(np.mean(speed)),mean_abs_vertical_COM_speed_mm_s=float(np.mean(np.abs(a['com_velocity'][sl,2]))),
            mean_body_angular_speed_rad_s=float(np.mean(spin)),COM_height_range_mm=[float(np.min(a['com'][sl,2])),float(np.max(a['com'][sl,2]))])
    return result


def main():
    result,p=read(out/'result.json'),read(out/'protocol.json')
    assert result['status']=='completed_P2' and len(result['jobs'])==4
    assert result['source_prior_checkpoints_unchanged'] and result['data_unchanged'] and result['process_tree_empty_verified'] and result['lease_released']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
    study.base.base.verify_native();cases={};raw={}
    for job in result['jobs']:
        name=job['name'];model,condition=name.split('-');pulse=condition=='pulse';d=out/name
        receipt=read(out/(name+'-receipt.json'));assert receipt==job['receipt']
        assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and job['available_ram_before_gib']>=2
        report=read(d/'report.json');a=arrays(d/'p1-telemetry.npz');force=arrays(d/'p1-microforce.npz');m=read(d/'p1-model.json')
        n=arrays(d/'p2-neural-input-output.npz');o=arrays(d/'observations.npz');c=arrays(d/'anatomy-observations.npz');pads=arrays(d/'adhesion-observations.npz')
        assert report['status']=='completed_transport_pilot' and report['p2_worker_completed'] and report['p1_worker_completed'] and report['weights_unchanged']
        assert not report['learning'] and report['motor_exploration']==0 and report['checkpoint_sha256']==sha(study.MODELS[model])
        assert report['p2_context']==read(out/(name+'-p2-context.json')) and not report['p2_context']['motor_output_replayed']
        assert report['p2_context']['model']==model and report['p2_context']['condition']==condition
        assert all(len(v)==1001 and np.isfinite(v).all() for v in a.values()) and all(len(v)==1000 and np.isfinite(v).all() for v in n.values())
        assert all(len(v)==1000 and np.isfinite(v).all() for v in o.values()) and len(c['time'])==1001
        np.testing.assert_allclose(a['time'],np.arange(1001)/200,rtol=0,atol=1e-10)
        np.testing.assert_allclose(force['time'],np.arange(50000)/10000,rtol=0,atol=1e-10)
        np.testing.assert_array_equal(n['chunk'],np.arange(1000));np.testing.assert_array_equal(a['completed_control_chunks'],np.arange(1001))
        expected=np.zeros((50000,3));endpoint=np.zeros_like(a['xfrc_applied']);weight=np.sum(m['body_mass'])*abs(m['gravity'][2])
        if pulse:
            expected[19650:19850,0]=.1*weight;endpoint[394:398,m['pulse_target_body'],0]=.1*weight
        np.testing.assert_array_equal(force['force'],expected);np.testing.assert_array_equal(a['xfrc_applied'],endpoint)
        np.testing.assert_array_equal(a['force_used_preceding_chunk'],endpoint[:,m['pulse_target_body'],:3])
        smooth=a['qfrc_passive']-a['qfrc_bias']+a['qfrc_actuator']+a['qfrc_applied']+a['qfrc_xfrc']
        np.testing.assert_allclose(a['qfrc_smooth'],smooth,rtol=1e-9,atol=1e-8)
        np.testing.assert_allclose(a['mass_times_qacc'],smooth+a['qfrc_constraint'],rtol=1e-9,atol=1e-8)
        contacts=arrays(d/'p1-ground-contacts.npz');ground=np.zeros((1001,3));np.add.at(ground,contacts['sample_index'],contacts['force_on_fly'])
        np.testing.assert_allclose(ground,a['ground_force'],rtol=1e-10,atol=1e-9)
        dof=m['free_dof'];np.testing.assert_allclose(ground,a['qfrc_constraint'][:,dof:dof+3],rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose((a['body_xipos']*np.asarray(m['body_mass'])[None,:,None]).sum(1)/sum(m['body_mass']),a['com'],rtol=1e-12,atol=1e-12)
        reference=study.REFERENCES[model];history=arrays(reference/'observations.npz')
        assert report['p2_context']['history_sha256']==sha(reference/'observations.npz')
        for k in FIELDS:
            np.testing.assert_array_equal(n['live_'+k][1:],o[k][:-1])
            np.testing.assert_array_equal(n['input_'+k][1:],history[k][:-1])
            np.testing.assert_array_equal(n['input_'+k][0],n['live_'+k][0])
            if not pulse:np.testing.assert_array_equal(n['live_'+k],n['input_'+k])
        refpads=arrays(reference/'adhesion-observations.npz')
        for k in refpads:np.testing.assert_array_equal(pads[k],refpads[k])
        np.testing.assert_array_equal(n['pads'][:,0],refpads['pads'])
        assert report['spikes_per_world']==read(reference/'report.json')['spikes_per_world']
        if not pulse:
            for filename in ('observations.npz','adhesion-observations.npz','p1-telemetry.npz','p1-microforce.npz','anatomy-observations.npz'):
                current,old=arrays(d/filename),arrays(reference/filename);assert set(current)==set(old)
                for k in old:np.testing.assert_array_equal(current[k],old[k])
        raw[name]=(a,n,o,c)
        cases[name]=dict(full_horizon=True,physics_steps=50000,neural_input_rows=1000,neural_output_matches_own_baseline=True,
            spike_total_matches_own_baseline=True,checkpoint_unchanged=True,actual_active_force_steps=int(np.count_nonzero(expected[:,0])),
            exact_no_pulse_physical_replay=not pulse)
    descriptions={}
    for model in study.MODELS:
        baseline=raw[model+'-baseline'];pulse=raw[model+'-pulse']
        for k in FIELDS:np.testing.assert_array_equal(pulse[1]['input_'+k],baseline[1]['input_'+k])
        for k in ('commands','pads'):np.testing.assert_array_equal(pulse[1][k],baseline[1][k])
        for index,rows in ((0,394),(2,393),(3,394)):
            for k in baseline[index]:np.testing.assert_array_equal(pulse[index][k][:rows],baseline[index][k][:rows])
        descriptions[model]=dict(live_feedback=physical_description(study.base.out/(model+'-pulse')),own_history_input=physical_description(out/(model+'-pulse')))
        assert descriptions[model]==result['matched_physical_descriptions'][model]
    summary=dict(status='independently_audited_P2',runtime_source_commit='4f6011c1ee5828e840964877eeea5ea5fcdc1626',
        workers=4,simulation_seconds=5,seed=10011,aggregate_wall_seconds=result['seconds'],cases=cases,matched_physical_descriptions=descriptions,
        source_prior_data_unchanged=True,receipts_exit_zero_empty_trees=True,lease_released=True,
        scope='One selected context; compare each model separately, no model ranking or new balance pass.',
        physical_balance_validated=False,model_promoted=False,walking_proven=False)
    write(root/'outputs/sensory-causality-p2-public-result-20261107.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

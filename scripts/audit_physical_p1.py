"""Terminal-only independent raw P1 checks. No simulation or acceptance-gate changes."""
import json
import numpy as np
import run_physical_p1 as study

root,out=study.root,study.out
read,write,sha,arrays=study.read,study.write,study.sha,study.arrays


def exact_prefix(left,right,count):
    assert set(left)==set(right)
    for k in left:np.testing.assert_array_equal(left[k][:count],right[k][:count],err_msg=k)


def main():
    result,p=read(out/'result.json'),read(out/'protocol.json')
    assert result['status']=='completed_P1' and len(result['jobs'])==4
    assert result['source_prior_checkpoints_unchanged'] and result['data_unchanged'] and result['process_tree_empty_verified'] and result['lease_released']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
    study.base.verify_native();cases={};raw={}
    for job in result['jobs']:
        name=job['name'];d=out/name;model,condition=name.split('-');pulse=condition=='pulse'
        receipt=read(out/(name+'-receipt.json'));assert receipt==job['receipt']
        assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and job['available_ram_before_gib']>=2
        report=read(d/'report.json');meta=read(d/'p1-model.json');a=arrays(d/'p1-telemetry.npz');micro=arrays(d/'p1-microforce.npz')
        obs=arrays(d/'observations.npz');pads=arrays(d/'adhesion-observations.npz');anatomy=arrays(d/'anatomy-observations.npz');contacts=arrays(d/'p1-ground-contacts.npz')
        assert report['status']=='completed_transport_pilot' and report['p1_worker_completed'] and report['weights_unchanged'] and not report['learning'] and report['motor_exploration']==0
        assert report['checkpoint_sha256']==sha(study.MODELS[model])==p['checkpoint_sha256'][model] and report['seeds']==[10011]
        assert report['p1_context']==read(out/(name+'-p1-context.json'))
        assert all(len(v)==1001 and np.isfinite(v).all() for v in a.values())
        assert all(len(v)==1000 and np.isfinite(v).all() for v in obs.values())
        assert len(pads['desired'])==1000 and len(anatomy['time'])==1001
        np.testing.assert_allclose(a['time'],np.linspace(0,5,1001),rtol=0,atol=1e-10)
        np.testing.assert_allclose(micro['time'],np.arange(50000)/10000,rtol=0,atol=1e-10)
        np.testing.assert_array_equal(a['completed_control_chunks'],np.arange(1001))
        mass=np.array(meta['body_mass']);weight=mass.sum()*abs(meta['gravity'][2])
        assert np.isclose(mass.sum(),meta['model_total_mass'],rtol=1e-12,atol=1e-15)
        assert meta['pulse_enabled']==pulse and meta['production_static_fusion']
        expected=np.zeros((50000,3));frame=np.zeros((1001,3))
        if pulse:
            expected[19650:19850,0]=.1*weight;frame[394:398,0]=.1*weight
        np.testing.assert_array_equal(micro['force'],expected)
        np.testing.assert_array_equal(a['force_used_preceding_chunk'],frame)
        full_force=np.zeros_like(a['xfrc_applied']);full_force[:,meta['pulse_target_body'],:3]=frame
        np.testing.assert_array_equal(a['xfrc_applied'],full_force)
        dof=meta['free_dof'];np.testing.assert_allclose(a['qfrc_xfrc'][:,dof:dof+3],frame,rtol=0,atol=1e-12)
        smooth=a['qfrc_passive']-a['qfrc_bias']+a['qfrc_actuator']+a['qfrc_applied']+a['qfrc_xfrc']
        np.testing.assert_allclose(smooth,a['qfrc_smooth'],rtol=1e-9,atol=1e-8)
        np.testing.assert_allclose(a['mass_times_qacc'],smooth+a['qfrc_constraint'],rtol=1e-9,atol=1e-8)
        np.testing.assert_allclose((a['body_xipos']*mass[None,:,None]).sum(1)/mass.sum(),a['com'],rtol=1e-12,atol=1e-12)
        # Independently sum recorded contact forces by sample; no external pulse in this equality.
        ground=np.zeros((1001,3));np.add.at(ground,contacts['sample_index'],contacts['force_on_fly'])
        np.testing.assert_allclose(ground,a['ground_force'],rtol=1e-10,atol=1e-9)
        np.testing.assert_allclose(ground,a['qfrc_constraint'][:,dof:dof+3],rtol=1e-6,atol=1e-8)
        cases[name]=dict(full_horizon=True,actual_microsteps=50000,active_force_microsteps=int(np.count_nonzero(expected[:,0])),
            impulse_world=(micro['force'].sum(0)/10000).tolist(),maximum_dynamic_equation_error=float(np.abs(a['mass_times_qacc']-smooth-a['qfrc_constraint']).max()),
            maximum_ground_constraint_error=float(np.abs(ground-a['qfrc_constraint'][:,dof:dof+3]).max()),
            neural_only_motor_commands=True,learning_updates=0,checkpoint_unchanged=True)
        raw[name]=(a,obs,pads,anatomy)
        if not pulse:
            for filename in ('observations.npz','adhesion-observations.npz'):
                old,current=arrays(study.OLD[model]/filename),arrays(d/filename);assert set(old)==set(current)
                for k in old:np.testing.assert_array_equal(current[k],old[k],err_msg=k)
            old=arrays(study.OLD[model]/'anatomy-observations.npz')
            for k in old:np.testing.assert_array_equal(anatomy[k],old[k],err_msg=k)
            assert report['spikes_per_world']==read(study.OLD[model]/'report.json')['spikes_per_world']
            cases[name]['exact_prior_replay']=True
    responses={}
    for model in ('parent','candidate'):
        b,bo,bc,ba=raw[model+'-baseline'];a,ao,ac,aa=raw[model+'-pulse']
        for left,right,count in ((b,a,394),(bo,ao,393),(bc,ac,394),(ba,aa,394)):exact_prefix(left,right,count)
        position=np.linalg.norm(a['com']-b['com'],axis=1);velocity=np.linalg.norm(a['com_velocity']-b['com_velocity'],axis=1)
        command=np.linalg.norm(ac['desired']-bc['desired'],axis=1);changed=np.flatnonzero(command)
        response=dict(exact_prefix_verified=True,max_COM_deviation_mm=float(position.max()),max_COM_velocity_deviation_mm_s=float(velocity.max()),
            final_1s_RMS_COM_deviation_mm=float(np.sqrt(np.mean(position[-200:]**2))),final_1s_mean_velocity_deviation_mm_s=float(velocity[-200:].mean()),
            changed_neural_command_rows=int(len(changed)),maximum_neural_command_L2_difference_radians=float(command.max()),
            changed_adhesion_rows=int((ac['pads']!=bc['pads']).any(1).sum()),different_contact_mode_samples=int((aa['contact_mode_code']!=ba['contact_mode_code']).sum()),
            max_up_vector_difference=float(np.linalg.norm(ao['up_body']-bo['up_body'],axis=2).max()),first_changed_command_chunk=int(changed[0]) if len(changed) else None,
            mechanical_response_is_not_neural_or_balance_qualification=True)
        assert response==result['responses'][model];responses[model]=response
    summary=dict(status='independently_audited_P1',runtime_source_commit='bd14ca69b91c36e8bcdd05eead0cf8ebd181a841',
        simulation_seconds=5,workers=4,aggregate_wall_seconds=result['seconds'],seed=10011,cases=cases,responses=responses,
        source_prior_data_unchanged=True,receipts_exit_zero_empty_trees=True,lease_released=True,
        conclusion='Delivered pulse and matched measurement pipeline verified. Neural commands change after perturbation, but neither beneficial feedback nor stable balance/retention/walking is established.',
        limits=['One selected moving case, one direction and force; no representative balance validation.',
            'Changed neural commands are not a sensory-causality ablation or proof of useful correction.',
            'Recorded mass-times-acceleration is checked against component forces; dense mass matrix was checked online, not independently reconstructed offline.',
            'No new learning, model ranking, acceptance gate or promotion. AP2 original zero-of-three verdict unchanged.'],
        physical_balance_validated=False,neural_feedback_qualified=False,model_promoted=False,walking_proven=False)
    write(root/'outputs/physical-calibration-p1-public-result-20261107.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

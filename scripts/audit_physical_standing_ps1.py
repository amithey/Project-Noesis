"""Independent terminal raw PS1 audit; no learning or physics integration."""
import json
import numpy as np
import run_physical_standing_ps1 as study
from noesis_gpu.backend import FlyGymBatch
from noesis_gpu.learning import score_transition
from s2_contract import standing_retention
from ap2_contract import DEV,anatomical_rule
from audit_ps1_objective import physical_data

root,out=study.root,study.out
read,write,sha,arrays=study.read,study.write,study.sha,study.arrays


def independent_components(reactions,velocity,omega,up,feet,weight):
    total=reactions.sum(-1)
    upright=np.maximum(0,np.minimum(1,(up-.7)/.3))
    tarsal=np.divide(reactions[...,1]+reactions[...,2],total,out=np.zeros_like(total),where=total>1e-5)
    load=np.exp(-np.abs(total/weight-1));claws=np.minimum((feet>1e-5).sum(-1)/3,1)
    motion=1/(1+(velocity*velocity).sum(-1)+(omega*omega).sum(-1))
    quality=np.clip(upright*tarsal*load*claws*motion,0,1)
    return dict(upright_quality=upright,tarsal_share=tarsal,vertical_load_quality=load,claw_support_quality=claws,motion_quality=motion,quality=quality,reward=2*quality-1)


def main():
    result,p=read(out/'result.json'),read(out/'protocol.json')
    assert result['status']=='completed_PS1' and len(result['jobs'])==7
    assert result['source_prior_checkpoints_unchanged'] and result['data_unchanged'] and result['process_tree_empty_verified'] and result['lease_released']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
    study.base.verify_native()
    for job in result['jobs']:
        receipt=read(out/(job['name']+'-receipt.json'));assert receipt==job['receipt']
        assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out'] and job['available_ram_before_gib']>=2
    initial={k:v[0].copy() for k,v in FlyGymBatch(1,'cpu').observe().items()};training={}
    for bank in ('control','candidate'):
        d=out/f'train-{bank}';report=read(d/'report.json');study.validate_training(report,bank)
        assert report['ps1_context']==read(out/(d.name+'-ps1-context.json')) and report['import_checkpoint_sha256']==sha(study.PARENT)
        cp=d/'weights-last.npz';assert sha(cp)==report['saved_checkpoint_sha256'];sidecar=read(cp.with_suffix('.model.json'))
        for k,v in report.items():assert sidecar[k]==v
        checkpoint=study.base.base.compare_checkpoints(study.PARENT,cp);assert checkpoint['compatible'] and checkpoint['rule']['match'] and checkpoint['parameters']['match']
        original,current=arrays(study.PARENT),arrays(cp)
        for k in ('version','normalize_reward_error','bias_correct_reward_error'):np.testing.assert_array_equal(original[k],current[k])
        anatomy=study.base.base.anatomy(d,'train');account=arrays(d/'physical-reward-accounting.npz');o=arrays(d/'training-observations.npz');m=read(d/'anatomy-model.json')
        assert all(len(v)==5000 and np.isfinite(v).all() for v in account.values()) and all(len(v)==5000 and np.isfinite(v).all() for v in o.values())
        np.testing.assert_array_equal(account['control_chunk_index'],np.arange(5000));np.testing.assert_array_equal(account['time'],anatomy['time'][1:])
        np.testing.assert_array_equal(o['learning_active'],np.arange(5000)>=10)
        for i in range(5000):
            before=initial if i==0 else {k:v[i-1] for k,v in o.items()};after={k:v[i] for k,v in o.items()}
            assert score_transition(before,after,5.,objective='curriculum-standing-v7')['reward']==account['native_reward'][i]
        independently=independent_components(anatomy['region_vertical_reaction'][1:],anatomy['com_velocity'][1:],o['omega'],o['up_body'][:,2],o['foot_load'],m['total_mass']*abs(m['gravity'][2]))
        for k,v in independently.items():np.testing.assert_allclose(v,account['PS1_'+k],rtol=1e-13,atol=1e-14)
        control=np.clip(account['native_reward']-.5*anatomy['non_tarsal_fraction'][1:],-1,1)
        np.testing.assert_array_equal(control,account['AP1_reference_reward']);np.testing.assert_array_equal(o['reward'],account['applied_reward'])
        np.testing.assert_array_equal(o['reward'],control if bank=='control' else account['PS1_reward'])
        if bank=='control':
            assert sha(cp)==sha(study.OLD_TRAIN/'weights-last.npz')
            for name in ('training-observations.npz','adhesion-observations.npz','anatomy-observations.npz','exploration-log.npz'):study.base.base.exact(d/name,study.OLD_TRAIN/name)
        else:
            study.base.base.exact(d/'exploration-log.npz',out/'train-control/exploration-log.npz')
            for name in ('training-observations.npz','adhesion-observations.npz'):
                c,a=arrays(out/'train-control'/name),arrays(d/name);assert set(a)==set(c)
                for k in c:
                    if k!='reward':np.testing.assert_array_equal(c[k][:11],a[k][:11])
        training[bank]=dict(checkpoint_sha256=sha(cp),actual_objective=report['reward_objective'],recomputed_native_rewards=5000,
            recomputed_physical_rewards=5000,mean_applied_reward=float(o['reward'].mean()),full_25s=True,learning_updates=4990,original_weight_invariants=True)
    banks={};rows=[]
    for bank in ('parent','v7','AP1','candidate'):
        banks[bank]={}
        for seed in study.SEEDS:
            d=out/f'frozen-candidate-{seed}' if bank=='candidate' else study.reference(bank,seed)
            report,summary=study.base.base.summarize(d);recorded=result['science']['banks'][bank][str(seed)]
            assert summary==recorded['summary'] and not report['learning'] and report['motor_exploration']==0 and report['weights_unchanged']
            if bank=='candidate':
                assert report['ps1_worker_completed'] and report['checkpoint_sha256']==training['candidate']['checkpoint_sha256']
                assert report['ps1_context']==read(d.with_name(d.name+'-ps1-context.json'))
                study.base.base.anatomy(d,'frozen')
                for ref in ('parent','v7','AP1'):study.base.base.compare(study.reference(ref,seed),d)
            anatomy=study.prior.previous.contact_data(d);assert anatomy==recorded['anatomical']
            o={k:v[:,0] for k,v in arrays(d/'observations.npz').items()};r,v,weight=physical_data(d)
            components=independent_components(r,v,o['omega'],o['up_body'][:,2],o['foot_load'],weight);total=r.sum(-1)
            q=np.divide(r[:,0]+r[:,3],total,out=np.zeros_like(total),where=total>1e-5)
            physical=dict(mean_PS1_quality=float(components['quality'][10:].mean()),mean_COM_speed_mm_s=float(np.linalg.norm(v[10:],axis=1).mean()),
                mean_angular_speed_rad_s=float(np.linalg.norm(o['omega'][10:],axis=1).mean()),mean_non_tarsal_fraction=float(q[10:].mean()),samples=990,physical_balance_validated=False)
            for k in physical:np.testing.assert_allclose(physical[k],recorded['physical'][k],rtol=1e-12,atol=1e-14)
            banks[bank][seed]=dict(summary=summary,anatomical=anatomy,physical=physical)
            rows.append(dict(bank=bank,seed=seed,own_seed=seed==1004,physical=physical,legacy_standing_fraction=summary['standing_fraction']))
    retention={bank:standing_retention({s:banks[bank][s]['summary'] for s in DEV},{s:banks['candidate'][s]['summary'] for s in DEV},DEV) for bank in ('parent','v7','AP1')}
    assert retention==result['science']['retention']
    anatomy=anatomical_rule(*({s:banks[bank][s]['anatomical'] for s in DEV} for bank in ('parent','AP1','candidate')))
    assert anatomy==result['science']['anatomical'];directions=[]
    for seed in DEV:
        for ref in ('parent','v7','AP1'):
            c,r=banks['candidate'][seed]['physical'],banks[ref][seed]['physical']
            directions.append(c['mean_PS1_quality']>r['mean_PS1_quality'] and all(c[k]<r[k] for k in ('mean_COM_speed_mm_s','mean_angular_speed_rad_s','mean_non_tarsal_fraction')))
    passed=all(directions);assert passed==result['science']['all_four_development_physical_directions_improve']
    combined=bool(passed and all(v['passed'] for v in retention.values()) and anatomy['passed']);assert combined==result['science']['combined_exploratory_pilot_pass']
    summary=dict(status='independently_audited_PS1',runtime_source_commit='411161424a161e000ba83ae89dac2c503867d5fe',seconds=result['seconds'],new_workers=7,
        training=training,frozen_bank_positions=20,rows=rows,retention=retention,anatomical=anatomy,all_four_development_physical_directions_improve=bool(passed),
        combined_exploratory_pilot_pass=combined,source_prior_data_unchanged=True,all_full_horizons_receipts_exit_zero_trees_empty=True,lease_released=True,
        physical_balance_validated=False,model_promoted=False,walking_proven=False,
        limits='One training birth, reused shared development references and proposed engineering score. Neither synthetic contracts nor a positive pilot prove retained standing, balance, gait or general learning efficacy.')
    write(root/'outputs/physical-standing-ps1-20261107-summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','anatomical')},indent=2))


if __name__=='__main__':main()

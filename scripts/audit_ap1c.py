"""Independent terminal checks and compact full-bank AP1B interpretation."""
import hashlib,json,sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
from noesis_gpu.evaluate import summarize,compare
from noesis_gpu.learning import score_transition
from noesis_gpu.retention_audit import compare_checkpoints
from noesis_gpu.backend import FlyGymBatch
from ap1_feedback import non_tarsal_fraction,corrected_reward,validate_ap1_training
from s2_contract import standing_retention
from support_descriptor_c0 import REGIONS,describe_support

MODES=('unsupported','central_body_contact','proximal_leg_contact','tarsal_only','indeterminate_low_force')
DEV=(10006,10007,10010,10011);SEEDS=(1005,*DEV)
OUT=root/'outputs/anatomical-feedback-ap1c-20261106'
PREVIOUS=root/'outputs/anatomical-feedback-ap1b-20261106'


def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,prefix=None,skip=()):
    a,b=arrays(a),arrays(b);assert set(a)==set(b)
    for k in set(a)-set(skip):np.testing.assert_array_equal(a[k] if prefix is None else a[k][:prefix],b[k] if prefix is None else b[k][:prefix],err_msg=k)


def cell(bank,seed):
    if bank!='candidate' and seed==10007:
        return root/'outputs/body-telemetry-h2b-20261106'/('parent-10007' if bank=='parent' else 'child1005-10007')
    return (OUT if bank=='candidate' and seed in (10007,10010,10011) else PREVIOUS)/f'frozen-{bank}-{seed}'


def contact_data(directory):
    if (directory/'anatomy-observations.npz').exists():
        a=arrays(directory/'anatomy-observations.npz');m=read(directory/'anatomy-model.json')
        assert len(a['time'])==1001 and all(np.isfinite(v).all() for v in a.values())
        np.testing.assert_allclose(a['time'],np.arange(1001)*.005,atol=1e-10,rtol=0)
        assert m['region_columns']==list(REGIONS) and m['contact_mode_codebook']==list(MODES)
        assert np.isclose(m['total_mass'],m['model_total_mass'],rtol=1e-12,atol=1e-15)
        assert m['unmapped_anatomical_count']==1 and m['production_static_fusion']
        assert a['force_closure_max_error'].max()<=m['total_mass']*9810*1e-6+1e-10
        r=a['region_vertical_reaction'];codes=a['contact_mode_code'];velocity=a['com_velocity']
        np.testing.assert_array_equal(a['non_tarsal_fraction'],[non_tarsal_fraction(x) for x in r])
    else:
        m=read(directory/'telemetry-model.json');a=arrays(directory/'body-telemetry.npz');c=arrays(directory/'ground-contacts.npz')
        names=[None]*m['nbody']
        for i,n in zip(m['body_ids'],m['body_names']):names[i]=n
        r=[];codes=[]
        for frame in range(1001):
            forces=np.zeros((m['nbody'],3));use=c['sample_index']==frame
            np.add.at(forces,c['body_id'][use],c['force_on_fly'][use])
            np.testing.assert_allclose(forces.sum(0),a['ground_force'][frame],rtol=1e-12,atol=1e-10)
            d=describe_support(forces,names);r.append([d['vertical_reaction_by_region'][n] for n in REGIONS]);codes.append(MODES.index(d['contact_mode']))
        r=np.asarray(r);codes=np.asarray(codes);velocity=a['com_velocity']
    assert r.shape==(1001,4) and np.isfinite(r).all() and (r>=0).all()
    assert np.isin(codes,np.arange(5)).all()
    q=np.asarray([non_tarsal_fraction(x) for x in r]);r=r[11:];codes=codes[11:]
    total=float(r.sum());assert len(r)==990
    return dict(samples=990,mode_fractions={n:float(np.mean(codes==i)) for i,n in enumerate(MODES)},
        mean_non_tarsal_fraction=float(q[11:].mean()),
        summed_vertical_reaction_share={n:float(r[:,i].sum()/total) if total else None for i,n in enumerate(REGIONS)},
        median_abs_com_vertical_velocity_mm_s=float(np.median(np.abs(velocity[11:,2]))),descriptor_not_balance_proof=True)


def main():
    result=read(OUT/'result.json');p=read(OUT/'protocol.json')
    assert result['status']=='completed_AP1C' and len(result['jobs'])==3 and len(result['cells'])==15
    assert result['source_prior_children_unchanged'] and result['data_unchanged'] and result['lease_released'] and result['process_tree_empty_verified']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    for group in ('source_hashes','prior_hashes','artifact_hashes'):
        for n,h in p[group].items():assert sha(root/n)==h
    prov=read(root/'outputs/thermo-pilot-tp2-20261106-source.json')
    for label,folder in (('pilot',root/'work/Noesis-Sensory/gpu-pilot/src'),('core',root/'work/Noesis-Core/src')):
        assert len(list(folder.rglob('*.py')))==len(prov[label])
        for entry in prov[label]:assert sha(folder/entry['path'])==entry['sha256']
    for f in read(root/'outputs/noesis-required-data-sha256.json')['files']:assert sha(root/'work/NoesisData'/f['path'])==f['sha256']
    for job in result['jobs']:
        receipt=read(OUT/(job['name']+'-receipt.json'));assert receipt==job['receipt']
        assert receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out']
        assert job['available_ram_before_gib']>=2
    parent=root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz'
    control=root/'outputs/anatomical-feedback-ap1-20261106/train-control-1005'
    candidate=PREVIOUS/'train-candidate-1005'
    initial={k:v[0].copy() for k,v in FlyGymBatch(1,'cpu').observe().items()}
    train_audit={}
    for directory,strength in ((control,0.),(candidate,.5)):
        report=read(directory/'report.json');validate_ap1_training(report,strength)
        cp=directory/'weights-last.npz';assert sha(cp)==report['saved_checkpoint_sha256']
        checkpoint=compare_checkpoints(parent,cp);assert checkpoint['compatible'] and checkpoint['rule']['match'] and checkpoint['parameters']['match']
        original,current=arrays(parent),arrays(cp)
        for k in ('version','normalize_reward_error','bias_correct_reward_error'):np.testing.assert_array_equal(original[k],current[k])
        sidecar=read(cp.with_suffix('.model.json'))
        for k,v in report.items():assert sidecar[k]==v
        a=arrays(directory/'anatomy-observations.npz');r=arrays(directory/'reward-accounting.npz');o=arrays(directory/'training-observations.npz')
        assert all(len(v)==5001 and np.isfinite(v).all() for v in a.values())
        assert all(len(v)==5000 and np.isfinite(v).all() for v in r.values())
        assert all(len(v)==5000 and np.isfinite(v).all() for v in o.values())
        np.testing.assert_array_equal(r['non_tarsal_fraction'],[non_tarsal_fraction(x) for x in a['region_vertical_reaction'][1:]])
        np.testing.assert_array_equal(r['time'],a['time'][1:]);np.testing.assert_array_equal(r['control_chunk_index'],np.arange(5000))
        np.testing.assert_array_equal(r['applied_reward'],[corrected_reward(n,q,strength) for n,q in zip(r['native_reward'],r['non_tarsal_fraction'])])
        np.testing.assert_array_equal(r['applied_reward'],o['reward']);np.testing.assert_array_equal(r['effective_penalty'],r['native_reward']-r['applied_reward'])
        np.testing.assert_array_equal(o['learning_active'],np.arange(5000)>=10)
        for i in range(5000):
            before=initial if i==0 else {k:v[i-1] for k,v in o.items()}
            after={k:v[i] for k,v in o.items()}
            assert score_transition(before,after,5.,objective='curriculum-standing-v7')['reward']==r['native_reward'][i]
        train_audit['control' if strength==0 else 'candidate']=dict(checkpoint_sha256=sha(cp),recomputed_native_transitions=5000,
            actual_objective=report['reward_objective'],mean_native_reward=float(r['native_reward'].mean()),mean_applied_reward=float(r['applied_reward'].mean()),
            mean_effective_penalty=float(r['effective_penalty'].mean()),nonzero_penalty_rows=int(np.count_nonzero(r['effective_penalty'])),checkpoint_audit=checkpoint)
    for n in ('training-observations.npz','adhesion-observations.npz','exploration-log.npz','weights-last.npz'):
        exact(control/n,root/'outputs/training-seed-sr1-20261106/train-1005'/n)
    exact(control/'exploration-log.npz',candidate/'exploration-log.npz',skip=('credit',))
    for n in ('training-observations.npz','adhesion-observations.npz'):exact(control/n,candidate/n,prefix=11,skip=('reward',))
    banks={b:{} for b in ('parent','control','candidate')};rows=[]
    for bank in banks:
        for seed in SEEDS:
            directory=cell(bank,seed);report,summary=summarize(directory);anatomical=contact_data(directory)
            recorded=result['cells'][f'{bank}-{seed}'];assert recorded['summary']==summary and recorded['anatomical']==anatomical
            assert report['weights_unchanged'] and not report['learning'] and report['motor_exploration']==0
            if bank!='parent':compare(cell('parent',seed),directory)
            if bank!='candidate':
                old=(root/'outputs/training-seed-sr1-20261106/parent-own-1005' if seed==1005 else root/'outputs/adhesion-sign-a1-20261106'/f'legacy-{seed}') if bank=='parent' else root/'outputs/training-seed-sr1-20261106'/f'child-1005-{seed}'
                for n in ('observations.npz','adhesion-observations.npz'):exact(directory/n,old/n)
                assert report['spikes_per_world']==read(old/'report.json')['spikes_per_world']
            banks[bank][seed]=summary
            rows.append(dict(bank=bank,seed=seed,own_seed=seed==1005,standing_fraction=summary['standing_fraction'],stance_slip_mm_s=summary['mean_stance_slip_mm_s'],
                heading_forward_mm=summary['heading_forward_distance_mm'],net_forward_mm=summary['net_forward_displacement_mm'],total_leg_steps=summary['total_leg_steps'],
                tripod_score=summary['tripod_coordination']['tripod_coordination_score'],regime=summary['locomotion_regime'],**anatomical))
    retention={b:standing_retention({s:banks[b][s] for s in DEV},{s:banks['candidate'][s] for s in DEV},DEV) for b in ('parent','control')}
    assert result['retention']==retention
    partial=PREVIOUS/'frozen-candidate-10007';complete=OUT/'frozen-candidate-10007'
    old_body,new_body=arrays(partial/'observations.npz'),arrays(complete/'observations.npz')
    old_a,new_a=arrays(partial/'anatomy-observations.npz'),arrays(complete/'anatomy-observations.npz')
    for k,v in old_body.items():np.testing.assert_array_equal(new_body[k][:366],v)
    for k,v in old_a.items():np.testing.assert_array_equal(new_a[k][:367],v)
    np.testing.assert_array_equal(new_a['compound_proximal_vertical_reaction'][:367],np.zeros(367))
    assert new_a['compound_proximal_vertical_reaction'][367]>0
    assert result['failed_partial_prefix_exact']['body_chunks']==366
    previous=read(PREVIOUS/'result.json')
    for i,job in enumerate(previous['jobs']):
        receipt=read(PREVIOUS/(job['name']+'-receipt.json'))
        assert receipt==job['receipt'] and receipt['tree_empty'] and not receipt['timed_out']
        assert receipt['exit_code']==(1 if i==11 else 0)
    by={(r['bank'],r['seed']):r for r in rows};comparison={}
    for seed in SEEDS:
        c=by['candidate',seed];comparison[str(seed)]={}
        for b in ('parent','control'):
            r=by[b,seed]
            comparison[str(seed)][b]=dict(tarsal_only_fraction_gain=c['mode_fractions']['tarsal_only']-r['mode_fractions']['tarsal_only'],
                mean_non_tarsal_fraction_change=c['mean_non_tarsal_fraction']-r['mean_non_tarsal_fraction'])
    assert comparison==result['anatomical_comparisons']
    anatomical_pass=all(v['tarsal_only_fraction_gain']>0 and v['mean_non_tarsal_fraction_change']<0 for s in DEV for v in comparison[str(s)].values())
    assert anatomical_pass==result['anatomical_dev_each_vs_both']
    a0_seconds=read(root/'outputs/anatomical-feedback-ap1-20261106/result.json')['seconds']
    verdict=dict(status='audited_complete_AP1C',evaluated_training_source='a05b0daa3c4416f1ffc0a2c59f8770f77069bd8b',evaluated_correction_source='a1fbcd3240c839e13e665bf7fdebfde7dc223cd0',correction_aggregate_seconds=result['seconds'],previous_B_aggregate_seconds=previous['seconds'],original_A0_control_aggregate_seconds=a0_seconds,total_AP1_allocation_seconds=result['seconds']+previous['seconds']+a0_seconds,
        new_frozen_jobs=3,previous_B_complete_workers=11,previous_B_partial_failed_workers=1,reused_complete_disabled_births=1,reused_H2B_frozen_cells=2,frozen_cells=15,training=train_audit,rows=rows,
        retention=retention,anatomical_comparisons=comparison,anatomical_primary_pass=anatomical_pass,
        combined_pilot_rule_pass=bool(anatomical_pass and all(v['passed'] for v in retention.values())),
        unchanged_source_data_priors=True,all_full_horizons_exit_zero_empty_tree=True,lease_released=True,
        failed_366_body_367_anatomy_prefix_exact=True,first_compound_force=float(new_a['compound_proximal_vertical_reaction'][367]),independent_birth_confirmation_required=True,pilot_training_births=1,model_promoted=False,walking_proven=False,
        limits=['One selected development training seed, no independent-birth confirmation.',
                'Contact modes and reaction fractions are descriptors, not validated standing/balance labels.',
                'A retained legacy proxy or leg cycles would not certify walking or environment response.',
                'All raw artifacts are local signed evidence until authenticated Release upload.'])
    dest=root/'outputs/anatomical-feedback-ap1c-20261106-summary.json';dest.write_text(json.dumps(verdict,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in verdict.items() if k not in ('training','rows','anatomical_comparisons','limits')}))


if __name__=='__main__':main()

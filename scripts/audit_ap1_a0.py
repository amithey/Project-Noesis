"""Independent terminal audit of AP1 disabled replay and floating mass gate."""
import hashlib,json,sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'work/Noesis-Sensory/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
from ap1_feedback import validate_ap1_training,non_tarsal_fraction
from noesis_gpu.retention_audit import compare_checkpoints


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def main():
    bank=root/'outputs/anatomical-feedback-ap1-20261106';d=bank/'train-control-1005';old=root/'outputs/training-seed-sr1-20261106/train-1005'
    terminal=read(bank/'result.json');p=read(bank/'protocol.json');r=read(d/'report.json');receipt=read(bank/'train-control-1005-receipt.json')
    assert terminal['status']=='failed' and terminal['error_type']=='AssertionError'
    assert terminal['lease_released'] and terminal['process_tree_empty_verified']
    assert terminal['source_prior_children_unchanged'] and terminal['data_unchanged']
    assert len(terminal['jobs'])==1 and receipt['exit_code']==0 and receipt['tree_empty'] and not receipt['timed_out']
    validate_ap1_training(r,0.)
    for n,h in p['source_hashes'].items():assert sha(root/n)==h
    for n,h in p['prior_hashes'].items():assert sha(root/n)==h
    for n in ('training-observations.npz','adhesion-observations.npz','exploration-log.npz','weights-last.npz'):
        a,b=arrays(d/n),arrays(old/n);assert set(a)==set(b)
        for k in a:np.testing.assert_array_equal(a[k],b[k],err_msg=f'{n}:{k}')
    assert sha(d/'weights-last.npz')==p['control_checkpoint_sha256']==r['saved_checkpoint_sha256']
    model=read(d/'anatomy-model.json');a=arrays(d/'anatomy-observations.npz');reward=arrays(d/'reward-accounting.npz');obs=arrays(d/'training-observations.npz')
    assert all(len(v)==5001 and np.isfinite(v).all() for v in a.values())
    assert all(len(v)==5000 and np.isfinite(v).all() for v in reward.values())
    np.testing.assert_allclose(a['time'],np.arange(5001)*.005,rtol=0,atol=1e-10)
    np.testing.assert_array_equal(a['non_tarsal_fraction'],[non_tarsal_fraction(x) for x in a['region_vertical_reaction']])
    np.testing.assert_array_equal(reward['time'],a['time'][1:])
    np.testing.assert_array_equal(reward['non_tarsal_fraction'],a['non_tarsal_fraction'][1:])
    np.testing.assert_array_equal(reward['control_chunk_index'],np.arange(5000))
    np.testing.assert_array_equal(reward['native_reward'],reward['applied_reward'])
    np.testing.assert_array_equal(reward['applied_reward'],obs['reward'])
    np.testing.assert_array_equal(reward['effective_penalty'],np.zeros(5000))
    np.testing.assert_array_equal(obs['learning_active'],np.arange(5000)>=10)
    assert model['total_mass']!=model['model_total_mass']
    assert np.isclose(model['total_mass'],model['model_total_mass'],rtol=1e-12,atol=1e-15)
    assert model['unmapped_anatomical_count']==1 and model['production_static_fusion']
    assert a['force_closure_max_error'].max()<=1e-6*model['total_mass']*9810+1e-10
    assert a['upstream_claw_force_max_error'].max()<=1e-10
    sidecar=read((d/'weights-last.npz').with_suffix('.model.json'))
    for k,v in r.items():assert sidecar[k]==v
    cp=compare_checkpoints(root/'work/evidence/s1-restored/gpu-pilot/outputs/e4-exploration-20261029/train-control-1000/weights-last.npz',d/'weights-last.npz')
    assert cp['compatible'] and cp['rule']['match'] and cp['parameters']['match']
    # Independently recompute native v7 from raw adjacent observations for every
    # transition whose before state is stored (first initial state is not stored).
    from noesis_gpu.learning import score_transition
    for i in range(1,5000):
        native=score_transition({k:v[i-1] for k,v in obs.items()},{k:v[i] for k,v in obs.items()},5.,objective='curriculum-standing-v7')
        assert native['reward']==reward['native_reward'][i]
    selected=[bank/'protocol.json',bank/'result.json',bank/'train-control-1005-receipt.json',bank/'train-control-1005.log']+list(d.glob('*.npz'))+list(d.glob('*.json'))
    result=dict(status='audited_complete_disabled_replay_technical_gate_only',source_commit='7753ba9e64c8eb6c48a69ee0b93121517ab42d8a',
        stopped_gate='Exact equality of differently ordered floating-point mass sums',
        total_compiled_subtree_mass=model['total_mass'],total_model_mass=model['model_total_mass'],
        absolute_difference=abs(model['total_mass']-model['model_total_mass']),
        existing_H2_mass_tolerance=dict(rtol=1e-12,atol=1e-15),
        full_control_seconds=25,updates=4990,checkpoint_sha256=sha(d/'weights-last.npz'),
        all_SR1_checkpoint_body_pad_exploration_arrays_exact=True,all_reward_accounting_exact=True,
        independent_native_reward_transitions_recomputed=4999,force_and_readonly_checks_passed=True,
        candidate_started=False,frozen_probes_started=0,scientific_comparison_available=False,
        model_promoted=False,walking_proven=False,original_bank_preserved=True,
        file_hashes={f.relative_to(root).as_posix():sha(f) for f in selected})
    dest=root/'outputs/anatomical-feedback-ap1-a0-audit-20261106.json';dest.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='file_hashes'}))


if __name__=='__main__':main()

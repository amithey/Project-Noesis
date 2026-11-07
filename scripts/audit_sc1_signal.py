"""SC1 fixed offline PS1 signal audit. Raw diagnostics/private integrity stay local."""
import json,time,zipfile
import numpy as np
import run_physical_standing_ps1 as study
from sc1_signal import normalized_signal,lag_diagnostics,LAGS,SHIFTS,SKIP_UPDATES

root,out=study.root,study.out
read,write,sha,arrays=study.read,study.write,study.sha,study.arrays


def main():
    started=time.monotonic();result=read(out/'result.json');assert result['status']=='completed_PS1' and result['lease_released']
    assert not (root/'work/Project-Noesis/gpu-pilot/outputs/codex-training-host.lock').exists()
    study.base.verify_native();receipt=read(root/'outputs/physical-standing-ps1-20261107-artifact.json');archive=root/'outputs'/receipt['path']
    assert sha(archive)==receipt['sha256']=='317b45fb3bcca26e65f1e56025607f630b38723e9ad6f9e1d390dd6de079cbd4'
    with zipfile.ZipFile(archive) as z:manifest=json.loads(z.read('manifest.json'))
    sources={('work/'+n):sha(root/'work'/n) for n in ('sc1_signal.py','test_sc1_signal.py','audit_sc1_signal.py')};inputs={};banks={}
    for bank in ('control','candidate'):
        if time.monotonic()-started>180:raise RuntimeError('Offline180s budget')
        d=out/f'train-{bank}';selected=[d/n for n in ('weights-last.npz','training-observations.npz','physical-reward-accounting.npz','exploration-log.npz','report.json')]+[out/(d.name+'.log')]
        for path in selected:
            n=path.relative_to(root).as_posix();assert sha(path)==manifest[n]['sha256'];inputs[n]=sha(path)
        o=arrays(d/'training-observations.npz');account=arrays(d/'physical-reward-accounting.npz');credit=arrays(d/'exploration-log.npz');cp=arrays(d/'weights-last.npz')
        np.testing.assert_array_equal(o['reward'],account['applied_reward']);np.testing.assert_array_equal(credit['control_chunk_index'],np.arange(10,5000))
        assert len(credit['credit'])==len(credit['drive'])==4990
        normalized=normalized_signal(o['reward'][10:]);checked=0
        for line in (out/(d.name+'.log')).read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row.get('event')!='training_progress':continue
            chunk=round(row['seconds']/.005)-1;i=chunk-10;assert 0<=i<4990
            for a,b in (('baseline','reward_baseline'),('raw_error','prediction_error'),('variance','reward_variance'),('scale','reward_error_scale'),('signal','learning_signal')):
                assert normalized[a][i]==row['learning'][b]
            checked+=1
        assert checked==50
        for k,field in (('baseline','reward_baseline'),('variance','reward_variance'),('mass','reward_normalization_mass')):assert normalized[k][-1]==float(cp[field])
        targets=credit['motor_indices'];slots=np.searchsorted(targets,credit['drive_idx']);assert np.all(slots<len(targets));np.testing.assert_array_equal(targets[slots],credit['drive_idx'])
        x=credit['credit'][:,slots];active=np.flatnonzero(np.std(credit['credit'],axis=0)>0);np.testing.assert_array_equal(np.sort(slots),active)
        signal=normalized['signal'][SKIP_UPDATES:];diagnostic=lag_diagnostics(x,normalized['signal'])
        banks[bank]=dict(progress_points_exact=checked,final_checkpoint_moments_exact=True,learning_updates=4990,driven_credit_targets=len(slots),
            mean_absolute_normalized_signal=float(np.abs(signal).mean()),zero_signal_fraction=float(np.mean(signal==0)),clipped_signal_fraction=float(np.mean(np.abs(signal)==1)),
            raw_error_std=float(normalized['raw_error'][SKIP_UPDATES:].std()),lag_descriptions=diagnostic,
            correlation_is_not_causality=True)
    for n,h in inputs.items():assert sha(root/n)==h
    for n,h in sources.items():assert sha(root/n)==h
    full=dict(status='completed_readonly_SC1',seconds=time.monotonic()-started,physics_steps=0,neural_steps=0,learning_updates=0,
        fixed_lags_control_chunks=list(LAGS),fixed_circular_shifts=list(SHIFTS),skip_updates=SKIP_UPDATES,banks=banks,
        limits='Post-hoc observational closed-loop correlations and circular shifts are descriptive, not independent replicates, causal gradients, efficacy/no-signal proof or a justified delay selection.',
        source_hashes=sources,input_hashes=inputs,model_promoted=False,walking_proven=False)
    write(root/'outputs/learning-signal-sc1-private-20261107.json',full)
    minimal=dict(status='completed_readonly_SC1',source_and_input_integrity_passed=True,all_logged_normalizer_values_and_final_moments_exact=True,
        physics_steps=0,neural_steps=0,learning_updates=0,model_promoted=False,walking_proven=False,
        conclusion='Native normalization/timing reconstruction matches recorded execution. Detailed observational signal diagnostics remain local; they do not establish useful causal learning or justify a delay change.')
    write(root/'outputs/learning-signal-sc1-minimal-result-20261107.json',minimal);print(json.dumps(minimal,indent=2))


if __name__=='__main__':main()

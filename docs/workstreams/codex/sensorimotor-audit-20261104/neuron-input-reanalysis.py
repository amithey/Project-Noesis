"""Exploratory reanalysis of existing fixed-weight records; no physical runs or learning."""
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

started=time.monotonic()
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'work/Noesis-Diagnostic/gpu-pilot/src'),str(root/'work/Noesis-Core/src')]
from noesis_gpu.chain_diagnostic import RECORDED, invariance_report

def correlations(x,y,shifts):
    x=(x-x.mean(0))/x.std(0)
    y=(y-y.mean(0))/y.std(0)
    cross=np.fft.irfft(np.conj(np.fft.rfft(x,axis=0))*np.fft.rfft(y,axis=0),n=len(x),axis=0)/len(x)
    r=cross[0]
    null=cross[shifts].mean(1)
    observed=float(r.mean())
    return {'mean_same_neuron_r_lag0':observed,
            'positive_direction_reference_rank':float((1+np.sum(null>=observed))/(len(shifts)+1)),
            'common_shift_reference_mean_r_quantiles':np.quantile(null,[.01,.05,.5,.95,.99]).tolist(),
            'individual_r_quantiles':np.quantile(r,[0,.05,.25,.5,.75,.95,1]).tolist(),
            'secondary_mean_r_squared':float(np.mean(r*r))}

# Verify the FFT reference indexing and the inability of a constant response to produce a valid correlation.
check=np.random.default_rng(9).normal(size=(1000,4))
c=correlations(check,check,np.array([200,300,400]))
assert abs(c['mean_same_neuron_r_lag0']-1)<1e-12
shift=200
cross=np.fft.irfft(np.conj(np.fft.rfft(check-check.mean(0),axis=0))*np.fft.rfft(check-check.mean(0),axis=0),n=1000,axis=0)
direct=np.sum((check-check.mean(0))*np.roll(check-check.mean(0),-shift,axis=0),axis=0)
np.testing.assert_allclose(cross[shift],direct,atol=1e-10)

archive=root/'work/evidence/chain-diagnostic-evidence.zip'
archive_sha=hashlib.sha256(archive.read_bytes()).hexdigest()
assert archive_sha=='c4d762c4a198b4a53679dc93b9de77c44a586a5ca1b96a323e72afc50fc45084'
result={'analysis':'NIR exploratory same-neuron current-to-firing reanalysis',
        'registration':'PR26 comment6635476098','input_archive_sha256':archive_sha,
        'independent_confirmatory_test':False,'physical_runs':0,'learning':False,
        'primary':'mean same-neuron Pearson r at lag0; common valid-neuron mask across arms',
        'reference':'1000 common shifts across all neuron columns, uniformly200..800samples,seed0; descriptive only',
        'cases':[],'model_promoted':False,'walking_proven':False}
with zipfile.ZipFile(archive) as z:
    for seed in [10006,10007,10010,10011]:
        report=json.loads(z.read(f'seed-{seed}-report.json'))
        assert report['status']=='completed_chain_diagnostic' and report['seconds']==5
        assert report['completed_arms']==['off','zero','active'] and report['weights_identical_across_arms']
        assert report['checkpoint_sha256']=='9fd7d41177bf2c745809401c3b142a832661d1a18d0b117b30cb02aea63fea3c'
        assert len(set(report['initial_weights_sha256'].values()))==1
        arms={}
        inputs={}
        for name in ['off','zero','active']:
            payload=z.read(f'seed-{seed}-chain-{name}.npz')
            inputs[name]=hashlib.sha256(payload).hexdigest()
            with np.load(io.BytesIO(payload),allow_pickle=False) as a:
                arms[name]={k:a[k].copy() for k in a.files}
            for key in RECORDED:
                assert len(arms[name][key])==1000 and np.isfinite(arms[name][key]).all()
            assert (arms[name]['circuit_spikes']>=0).all()
        assert invariance_report(arms['off'],arms['zero'])['identical']
        for name in ['off','zero']:
            assert not arms[name]['drive'].any()
            np.testing.assert_array_equal(arms[name]['drive_idx'],arms['active']['drive_idx'])
            np.testing.assert_array_equal(arms[name]['rest'],arms['active']['rest'])
        x=arms['active']['drive'].astype(float)
        assert x.shape==(1000,453)
        mask=x.std(0)>0
        silent={}
        for name in arms:
            assert arms[name]['circuit_spikes'].shape==x.shape
            varying=arms[name]['circuit_spikes'].std(0)>0
            silent[name]=int((~varying).sum())
            mask &= varying
        assert mask.any()
        shifts=np.random.default_rng(0).integers(200,801,size=1000)
        scores={name:correlations(x[:,mask],arms[name]['circuit_spikes'][:,mask].astype(float),shifts) for name in arms}
        assert scores['off']==scores['zero']
        result['cases'].append({'seed':seed,'source_arrays_sha256':inputs,
                                'invariance_passed':True,'targets_total':453,'common_valid_targets':int(mask.sum()),
                                'constant_response_targets_by_arm':silent,'arms':scores})
result['seconds']=time.monotonic()-started
assert result['seconds']<60
(root/'outputs/neuron-input-reanalysis-20261104.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps([{'seed':r['seed'],'valid_targets':r['common_valid_targets'],'arms':{k:{'mean_r':v['mean_same_neuron_r_lag0'],'rank':v['positive_direction_reference_rank']} for k,v in r['arms'].items()}} for r in result['cases']],indent=2))


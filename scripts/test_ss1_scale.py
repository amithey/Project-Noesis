import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from ss1_scale import TRAIN,SCALES,analyze,panel,validate_worker


def fixture():
    p=np.full(6,-.5);r=np.array([[[-.5-h*.5,-.5+h*.5] for h in SCALES] for _ in TRAIN]);return p,r


def records():
    p,r=fixture();rows=[]
    for s,seed in enumerate(TRAIN):
        for h,k in [(0.,0)]+[(h,k) for h in SCALES for k in (-1,1)]:
            c=dict(seed=seed,phase='parent' if h==0 else 'trial',bank='parent' if h==0 else 'probe',
                direction=None if h==0 else 0,sign=None if h==0 else k,amplitude=None if h==0 else h)
            value=p[s] if h==0 else r[s,SCALES.index(h),int(k==1)]
            rows.append(dict(context=c,**{'return':float(value)},complete_horizon=True,actual_import_verified=True,
                warmup_samples=10,measured_samples=990,return_objective='physical-standing-ps1-v1',
                checkpoint_sha256='a'*64,return_source_sha256='b'*64))
    return rows


class Tests(unittest.TestCase):
    def test_linear_compatible_and_no_mutation(self):
        p,r=fixture();before=r.copy();a=analyze(p,r)
        self.assertEqual(a['compatible_contexts'],6);self.assertTrue(a['smallest_scale_cross_context_sign_consistent'])
        np.testing.assert_array_equal(r,before)

    def test_cross_context_opposition_distinct_from_local_stability(self):
        p,r=fixture();r[3:]=2*p[3:,None,None]-r[3:];a=analyze(p,r)
        self.assertTrue(a['all_contexts_compatible']);self.assertFalse(a['smallest_scale_cross_context_sign_consistent'])

    def test_zero_signal_not_stable(self):
        p,r=fixture();r[:]=p[:,None,None];a=analyze(p,r)
        self.assertEqual(a['compatible_contexts'],0);self.assertFalse(a['smallest_scale_cross_context_sign_consistent'])

    def test_nonlinear_and_sign_unstable_rejected(self):
        p,r=fixture()
        for i,h in enumerate(SCALES):r[:,i,:]+=100*h*h
        self.assertEqual(analyze(p,r)['compatible_contexts'],0)
        p,r=fixture();r[:,2]=r[:,2,::-1];self.assertEqual(analyze(p,r)['compatible_contexts'],0)

    def test_complete_panel_and_rejections(self):
        rows=records();before=copy.deepcopy(rows);p,r=panel(rows);self.assertEqual(r.shape,(6,3,2));self.assertEqual(rows,before)
        for data in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):panel(data)
        for field,value in [('seed',11008),('seed',10),('phase','evaluation'),('direction',1),('amplitude',.1)]:
            data=records();data[1]['context'][field]=value
            with self.assertRaises(ValueError):panel(data)
        for field,value in [('complete_horizon',False),('return',True),('return',float('nan')),('checkpoint_sha256','z'*64)]:
            data=records();data[0][field]=value
            with self.assertRaises(ValueError):panel(data)

    def test_worker_registration_and_output_separation(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);runtime=base/'runtime';runtime.mkdir()
            a=SimpleNamespace(runtime_root=runtime,output=base/'new',phase='trial',bank='probe',seed=1010,direction=0,sign=-1,amplitude=.005)
            validate_worker(a);a.amplitude=.01
            with self.assertRaises(ValueError):validate_worker(a)
            a.amplitude=.005;a.output=runtime/'forbidden'
            with self.assertRaises(ValueError):validate_worker(a)

    def test_invalid_analysis(self):
        with self.assertRaises(ValueError):analyze(np.zeros(2),np.zeros((2,3,2)))
        with self.assertRaises(ValueError):analyze(np.zeros(6),np.full((6,3,2),np.inf))


if __name__=='__main__':unittest.main()

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mp2_select import TRAIN,select,panel,key,MIN_GAIN
from mp2_worker import validate


def records():
    out=[]
    for s in TRAIN:
        for d,k in [(-1,0)]+[(d,k) for d in range(4) for k in (-1,1)]:
            out.append(dict(context=dict(seed=s,phase='parent' if d<0 else 'trial',bank='parent' if d<0 else 'probe',
                direction=None if d<0 else d,sign=None if d<0 else k),**{'return':0. if d<0 else .1},
                complete_horizon=True,actual_import_verified=True,warmup_samples=10,measured_samples=990,
                return_objective='physical-standing-ps1-v1',checkpoint_sha256='a'*64,return_source_sha256='b'*64))
    return out


class Tests(unittest.TestCase):
    def test_six_contexts_reject_two_context_overfit(self):
        p=np.zeros(6);v=np.full((6,8),-.1);v[:2,0]=.5;v[:,1]=.1
        self.assertEqual(select(p,v)['policy_index'],1)
        v[5,1]=-.2;self.assertTrue(select(p,v)['parent_fallback'])

    def test_margin_tie_and_reversed(self):
        p=np.zeros(6);v=np.full((6,8),MIN_GAIN)
        self.assertTrue(select(p,v)['parent_fallback'])
        v[:]=.1;self.assertEqual(select(p,v)['policy_index'],0)
        self.assertTrue(select(p,v,-1)['parent_fallback'])
        v[:,5]=-.2;self.assertEqual(select(p,v,-1)['policy_index'],5)

    def test_complete_panel_and_no_mutation(self):
        r=records();before=copy.deepcopy(r);p,v=panel(r)
        self.assertEqual(p.shape,(6,));self.assertEqual(v.shape,(6,8));self.assertEqual(r,before)

    def test_missing_duplicate_and_bad_provenance(self):
        for r in (records()[:-1],records()+[records()[0]]):
            with self.assertRaises(ValueError):panel(r)
        for field,value in [('complete_horizon',False),('actual_import_verified',False),('warmup_samples',0),
                ('measured_samples',989),('return_objective','other'),('checkpoint_sha256','z'*64),
                ('return_source_sha256',''),('return',float('nan')),('return',True)]:
            r=records();r[0][field]=value
            with self.assertRaises(ValueError):panel(r)

    def test_wrong_roles_and_held_out_rejected(self):
        for field,value in [('seed',11008),('seed',10006),('seed',10),('seed',True),('phase','evaluation'),('bank','candidate')]:
            r=records();r[0]['context'][field]=value
            with self.assertRaises(ValueError):panel(r)

    def test_selector_shape_finite_and_sign(self):
        for p,v,sign in [(np.zeros(2),np.zeros((2,8)),1),(np.zeros(6),np.full((6,8),np.nan),1),
                (np.zeros(6),np.full((6,8),1.1),1),(np.zeros(6),np.zeros((6,8)),True)]:
            with self.assertRaises(ValueError):select(p,v,sign)

    def test_worker_output_separation_and_seed_roles(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runtime=root/'legacy';runtime.mkdir()
            a=SimpleNamespace(runtime_root=runtime,output=root/'new',phase='parent',bank='parent',seed=1012,direction=None,sign=None)
            validate(a)
            a.output=runtime/'forbidden'
            with self.assertRaises(ValueError):validate(a)
            a.output=root/'new';a.seed=1010
            with self.assertRaises(ValueError):validate(a)
            a.phase='baseline';validate(a)
            a.output.mkdir()
            with self.assertRaises(ValueError):validate(a)


if __name__=='__main__':unittest.main()

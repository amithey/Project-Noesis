import copy
import unittest
import numpy as np
from oa2_decision_sensitivity import case_key,counterfactual_returns,decision_diagnostic
from support_descriptor_c1 import REGIONS


def rows():
    out=[]
    for s in (1010,1011):
        out.append(dict(context=dict(seed=s,phase='evaluation',bank='parent',direction=None,sign=None),original_return=0.,alternative_return=0.))
        for d in range(4):
            for k in (-1,1):
                value=.2 if (d,k)==(0,-1) else .1
                out.append(dict(context=dict(seed=s,phase='trial',bank='probe',direction=d,sign=k),original_return=value,alternative_return=value))
    return out


class Tests(unittest.TestCase):
    def test_same_returns_same_decision(self):
        r=decision_diagnostic(rows())
        self.assertEqual(r['policy_identity_changed'],dict(candidate=False,negative=False))
        self.assertEqual(r['original']['decisions']['candidate']['policy_index'],0)

    def test_alternative_can_change_hypothetical_decision(self):
        data=rows()
        for r in data:
            c=r['context']
            if (c['direction'],c['sign'])==(1,1):r['alternative_return']=.3
        result=decision_diagnostic(data)
        self.assertTrue(result['policy_identity_changed']['candidate'])
        self.assertEqual(result['alternative']['decisions']['candidate']['policy_index'],3)

    def test_missing_duplicate_rejected(self):
        for data in (rows()[:-1],rows()+[rows()[0]]):
            with self.assertRaises(ValueError):decision_diagnostic(data)

    def test_wrong_role_and_held_out_rejected(self):
        for field,value in [('seed',11008),('seed',10006),('seed',10),('phase','baseline'),('bank','candidate'),('seed',True)]:
            data=rows();data[0]['context'][field]=value
            with self.assertRaises(ValueError):decision_diagnostic(data)
        with self.assertRaises(ValueError):case_key(dict(seed=1010,phase='evaluation',bank='probe',direction=0,sign=-1))

    def test_invalid_return_rejected(self):
        for value in (True,float('nan'),float('inf'),1.1):
            data=rows();data[0]['alternative_return']=value
            with self.assertRaises(ValueError):decision_diagnostic(data)

    def test_zero_actuator_and_analytic_force_sensitivity(self):
        o=dict(omega=np.zeros((1000,1,3)),up_body=np.tile([0.,0.,1.],(1000,1,1)),foot_load=np.ones((1000,1,6)))
        a=dict(time=np.arange(1001)*.005,region_vertical_reaction=np.tile([0.,1.,0.,0.],(1001,1)),
            com_velocity=np.zeros((1001,3)),ground_force=np.tile([0.,0.,1.],(1001,1)),root_actuator_force=np.zeros((1001,3)))
        m=dict(region_columns=list(REGIONS),total_mass=1.,gravity=[0,0,-1.])
        before=copy.deepcopy(a)
        old,new=counterfactual_returns(o,a,m);self.assertEqual(old,new);self.assertEqual(old,1.)
        for k in a:np.testing.assert_array_equal(a[k],before[k])
        a['root_actuator_force'][:,2]=-.5
        old,new=counterfactual_returns(o,a,m);self.assertEqual(old,1.)
        self.assertAlmostEqual(new,2*np.exp(-.5)-1)


if __name__=='__main__':unittest.main()

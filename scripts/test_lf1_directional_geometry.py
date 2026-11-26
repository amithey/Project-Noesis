import unittest
from types import SimpleNamespace
import numpy as np
from lf1_directional_geometry import geometry,robust_linear,certificate,RADIUS


class Tests(unittest.TestCase):
    def test_aligned_analytic_optimum(self):
        g=np.tile([2.,1.,0.,0.],(6,1));r=robust_linear(g)
        self.assertAlmostEqual(r['lower_bound'],2*RADIUS)
        self.assertAlmostEqual(r['upper_bound'],2*RADIUS)
        self.assertEqual(r['classification'],'positive_surrogate_margin')

    def test_opposed_and_zero(self):
        for g in (np.vstack([np.ones((3,4)),-np.ones((3,4))]),np.zeros((6,4))):
            r=robust_linear(g);self.assertAlmostEqual(r['lower_bound'],0.)
            self.assertAlmostEqual(r['upper_bound'],0.);self.assertEqual(r['classification'],'no_margin_above_threshold')

    def test_mixed_needs_two_directions(self):
        g=np.array([[1.,0.,0.,0.]]*3+[[0.,1.,0.,0.]]*3)
        r=robust_linear(g);self.assertAlmostEqual(r['lower_bound'],RADIUS/2)
        self.assertAlmostEqual(abs(r['coefficients']).sum(),RADIUS)

    def test_exact_odd_even_decomposition_and_no_mutation(self):
        p=np.full(6,-.5);odd=np.tile([.01,-.02,.03,0.],(6,1));even=np.full((6,4),.04)
        q=np.zeros((6,8));q[:,::2]=p[:,None]+even-odd;q[:,1::2]=p[:,None]+even+odd
        before=q.copy();r=geometry(p,q)
        np.testing.assert_allclose(r['odd_amplitude'],odd,atol=1e-15)
        np.testing.assert_allclose(r['midpoint_departure'],even,atol=1e-15)
        self.assertEqual(r['midpoint_exceeds_odd_cells'],24);np.testing.assert_array_equal(q,before)

    def test_failed_solver_rejected(self):
        def failed(*args,**kwargs):return SimpleNamespace(success=False,status=1)
        with self.assertRaises(ValueError):robust_linear(np.ones((6,4)),failed)

    def test_success_flag_does_not_override_infeasible_output(self):
        calls=[]
        def fake(*args,**kwargs):
            calls.append(1)
            return SimpleNamespace(success=True,status=0,x=np.ones(9) if len(calls)==1 else np.ones(7))
        with self.assertRaises(ValueError):robust_linear(np.ones((6,4)),fake)

    def test_certificate_rejects_bad_gap_and_constraints(self):
        g=np.ones((6,4));w=np.ones(6)/6
        with self.assertRaises(ValueError):certificate(g,np.zeros(4),w)
        with self.assertRaises(ValueError):certificate(g,np.ones(4),w)
        with self.assertRaises(ValueError):certificate(g,np.array([RADIUS,0,0,0]),np.ones(6))

    def test_bad_panel_and_matrix_rejected(self):
        for p,q in ((np.zeros(2),np.zeros((2,8))),(np.zeros(6),np.full((6,8),np.nan)),(np.zeros(6),np.full((6,8),2.))):
            with self.assertRaises(ValueError):geometry(p,q)
        with self.assertRaises(ValueError):robust_linear(np.zeros((5,4)))


if __name__=='__main__':unittest.main()

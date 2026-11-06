import unittest
import numpy as np
from select_contracts import reciprocal_contract,inverse,inverse_sqrt_contract,sqrt_enclosure
from fractions import Fraction

class Selection(unittest.TestCase):
    def test_endpoint_selected_iterations_keep_original_target(self):
        for k in (1,2,15,16,31,32,64,128):
            r=reciprocal_contract((.5/k,2.));x=np.linspace(*r['domain'],16385)
            residual=np.max(np.abs(x*inverse(x,r)-1))
            self.assertLessEqual(residual,1e-12+8*np.finfo(float).eps)
            self.assertGreaterEqual(r['iterations'],1)
        self.assertEqual(reciprocal_contract((.5/128,2.))['iterations'],13)
    def test_invalid_domain_fails(self):
        for d in ((-1.,2.),(0.,2.),(2.,1.),(np.nan,2.)):
            with self.assertRaises(ValueError):reciprocal_contract(d)
    def test_inverse_sqrt_seed_and_exact_root_enclosure(self):
        for interval in ((.01,1000.),(.317,1.186),(1.,320.),(1.,1.000001)):
            r=inverse_sqrt_contract(interval)
            self.assertEqual(float.fromhex(r['seed_hex']),r['seed'])
            self.assertLessEqual(float(r['ideal_relative_error_upper']),1e-6)
            self.assertLessEqual(r['sampled_relative_error'],1e-6+8*np.finfo(float).eps)
            for x in interval:
                value=Fraction.from_float(x)*Fraction.from_float(r['seed'])**2
                lo,hi=sqrt_enclosure(value)
                self.assertLessEqual(lo*lo,value);self.assertGreaterEqual(hi*hi,value)
if __name__=='__main__':unittest.main()

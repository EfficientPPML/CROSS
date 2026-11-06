from decimal import Decimal
from fractions import Fraction
import unittest
from certify_rounded_reciprocal import certify

class Certificates(unittest.TestCase):
    def test_exact_rounded_affine_not_constant_seed(self):
        rule={'domain':[.5/74,2.],'seed':2/(.5/74+2),'iterations':13}
        result=certify(rule)
        self.assertTrue(result['passed']);self.assertTrue(result['exact_stationary_point_in_domain'])
        w=Fraction.from_float(-rule['seed']*rule['seed']);b=Fraction.from_float(2*rule['seed'])
        upper=Fraction(*map(int,result['exact_initial_residual_upper']))
        for i in range(1001):
            x=Fraction.from_float(rule['domain'][0])+(Fraction.from_float(rule['domain'][1])-Fraction.from_float(rule['domain'][0]))*i/1000
            self.assertLessEqual(abs(1-w*x*x-b*x),upper)
        self.assertLess(Decimal(result['final_relative_residual_upper']),Decimal('1e-12'))
    def test_too_few_updates_are_reported_not_selected(self):
        result=certify({'domain':[.5/128,2.],'seed':2/(.5/128+2),'iterations':1})
        self.assertFalse(result['passed']);self.assertEqual(result['iterations'],1)
    def test_invalid_and_nonconvergent_rules_fail(self):
        for rule in ({'domain':[0.,1.],'seed':1.,'iterations':1},
                     {'domain':[1.,2.],'seed':10.,'iterations':2},
                     {'domain':[1.,2.],'seed':1.,'iterations':True}):
            with self.subTest(rule=rule),self.assertRaises(ValueError):certify(rule)
if __name__=='__main__':unittest.main()

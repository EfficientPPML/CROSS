import unittest
import numpy as np
from calibrate import accumulate,domain
from audit import summarize

class Calibration(unittest.TestCase):
    def test_streaming_counts_and_extrema(self):
        out={}
        for x in ([-3,0],[],[2,5]):accumulate(out,{'x':summarize(np.array(x),[-1,1])})
        self.assertEqual(out['x']['examples'],3)
        self.assertEqual(out['x']['examples_with_violations'],2)
        self.assertEqual(out['x']['count'],4)
        self.assertEqual((out['x']['minimum'],out['x']['maximum']),(-3,5))
    def test_prior_domain_rules_are_unchanged(self):
        self.assertEqual(domain(-3,5),[-4,6])
        self.assertEqual(domain(2,10,True),[1.6,12])
        for a,b in ((0,1),(-1,2),(np.nan,2)):
            with self.assertRaises(ValueError):domain(a,b,True)
if __name__=='__main__':unittest.main()

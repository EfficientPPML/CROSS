"""Focused algebra/reporting regressions; no ciphertext or model-accuracy claim."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_pretrained_bert_arithmetic as candidate
import numpy as np


class ArithmeticTests(unittest.TestCase):
    def test_balanced_chebyshev_matches_independent_clenshaw(self):
        # The recurrence uses T(2k)=2*T(k)^2-1 and
        # T(2k+1)=2*T(k)*T(k+1)-T(1), including interval endpoints.
        for degree in (2, 3, 12, 33, 128):
            with self.subTest(degree=degree):
                coefficients = np.random.default_rng(degree).normal(size=degree+1)/(1+np.arange(degree+1))**2
                contract = SimpleNamespace(domain=(-19., 11.), coefficients=coefficients)
                x = np.linspace(-19., 11., 513)
                engine = candidate.Arithmetic()
                result = engine.polynomial(candidate.Value(x), contract, 'chebyshev', 'test')
                exact = np.polynomial.chebyshev.chebval(2*x/30+8/30, coefficients)
                np.testing.assert_allclose(result.data, exact, rtol=1e-12, atol=1e-12)
                self.assertEqual(engine.counts['ciphertext_tensor_multiplications'], degree-1)
                self.assertEqual(result.depth, 2+2*int(np.ceil(np.log2(degree))))
                self.assertEqual(engine.domain_failures(), {})

    def test_power_basis_matches_independent_chebyshev(self):
        contract = candidate.nonlinear_approximation('inverse_sqrt', (.2, 1.4), 16)
        x = np.linspace(.2, 1.4, 257)
        engine = candidate.Arithmetic()
        result = engine.polynomial(candidate.Value(x), contract, 'power', 'inverse')
        expected = np.polynomial.chebyshev.chebval(2*x/1.2-1.6/1.2, contract.coefficients)
        np.testing.assert_allclose(result.data, expected, atol=2e-13, rtol=2e-13)
        self.assertEqual(result.depth, 6)

    def test_input_scale_is_fused_without_extra_depth(self):
        contract = candidate.nonlinear_approximation('tanh', (-1., 1.), 12)
        inputs = np.linspace(-24., 24., 129)
        engine = candidate.Arithmetic()
        result = engine.polynomial(candidate.Value(inputs), contract, 'power', 'tanh', input_scale=1/32)
        expected = np.polynomial.chebyshev.chebval(inputs/32, contract.coefficients)
        np.testing.assert_allclose(result.data, expected, atol=1e-14, rtol=1e-14)
        self.assertEqual(result.depth, 6)
        self.assertEqual(engine.domains['tanh']['maximum'], .75)

    def test_newton_reciprocal_fixed_loop_and_endpoints(self):
        domain, count = (.5/16, 2.), 10
        x = np.linspace(*domain, 513)
        engine = candidate.Arithmetic()
        result = engine.inverse(candidate.Value(x), domain, count, 'reciprocal')
        expected = np.full_like(x, 2/sum(domain))
        for _ in range(count):
            expected *= 2-x*expected
        np.testing.assert_allclose(result.data, expected, atol=1e-14, rtol=1e-14)
        self.assertLess(np.max(np.abs(x*result.data-1)), 3e-14)
        self.assertEqual(result.depth, 1+3*(count-1))
        self.assertEqual(engine.counts['ciphertext_tensor_multiplications'], 2*(count-1))

    def test_newton_inverse_sqrt_fixed_loop_and_endpoints(self):
        rule = dict(domain=(.741, 320.), method='newton', iterations=11)
        x = np.geomspace(*rule['domain'], 513)
        engine = candidate.Arithmetic()
        result = engine.inverse_sqrt(candidate.Value(x), rule, 'inverse_sqrt')
        expected = np.full_like(x, 1/np.sqrt(rule['domain'][1]))
        for _ in range(rule['iterations']):
            expected *= 1.5-.5*x*expected**2
        np.testing.assert_allclose(result.data, expected, atol=1e-14, rtol=1e-14)
        # The manifest's fixed iteration target is 1e-6 over the full public
        # interval; smaller observed-fixture errors are not endpoint bounds.
        self.assertLess(np.max(np.abs(np.sqrt(x)*result.data-1)), 1e-6)
        self.assertEqual(result.depth, 41)
        self.assertEqual(engine.counts['ciphertext_tensor_multiplications'], 30)

    def test_balanced_chebyshev_does_not_clip_or_hide_failure(self):
        contract = SimpleNamespace(domain=(-1., 1.), coefficients=(0., 0., 1.))
        engine = candidate.Arithmetic()
        result = engine.polynomial(candidate.Value(np.array([-2., 2.])), contract, 'chebyshev', 'outside')
        np.testing.assert_array_equal(result.data, [7., 7.])
        self.assertEqual(engine.domain_failures()['outside']['below'], 1)
        self.assertEqual(engine.domain_failures()['outside']['above'], 1)

    def test_all_domain_violations_accumulate(self):
        engine = candidate.Arithmetic()
        engine.observe('first', np.array([-2., 2., np.nan, np.inf, -np.inf]), (-1., 1.))
        engine.observe('valid', np.array([0., 1.]), (0., 1.))
        engine.observe('later', np.array([4.]), (0., 1.))
        failures = engine.domain_failures()
        self.assertEqual(set(failures), {'first', 'later'})
        self.assertEqual(failures['first']['nonfinite'], 3)
        self.assertEqual(failures['first']['minimum'], -2.)
        self.assertEqual(failures['first']['maximum'], 2.)

    def test_logit_gate_preserves_class_one_ties(self):
        exact = np.array([[1., 1.]])
        self.assertTrue(candidate.gate(exact, exact)['binary_tie_rule_pass'])
        shifted = candidate.gate(np.array([[1.+1e-6, 1.]]), exact)
        self.assertTrue(shifted['logit_tolerance_pass'])
        self.assertFalse(shifted['binary_tie_rule_pass'])
        self.assertFalse(candidate.gate(np.array([[1., 1.1]]), exact)['logit_tolerance_pass'])
        self.assertFalse(candidate.gate(np.array([[np.nan, 1.]]), exact)['finite'])
        self.assertFalse(candidate.gate(np.zeros((1, 3)), exact)['shape_matches'])


if __name__ == '__main__':
    unittest.main()

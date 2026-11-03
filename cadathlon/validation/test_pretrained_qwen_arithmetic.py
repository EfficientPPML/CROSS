"""CPU algebra/causal-layout checks, not encrypted or pretrained accuracy tests."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan_pretrained_qwen_arithmetic as candidate

REFERENCE = candidate.load_module(candidate.ROOT/'cadathlon/agent_pset/p4_qwen/src/reference.py')


class QwenArithmeticTests(unittest.TestCase):
    def test_rotary_half_split_matches_reference(self):
        data = np.random.default_rng(81).normal(size=(2, 3, 4, 8))
        engine = candidate.QwenArithmetic()
        actual = engine.rope(candidate.Value(data), 1e6, 'rope')
        np.testing.assert_allclose(actual.data, REFERENCE.rope(data, 1e6), atol=1e-14, rtol=1e-14)
        self.assertEqual(actual.depth, 1)

    def test_causal_rows_match_independent_masked_attention(self):
        rng = np.random.default_rng(63)
        q, k, v = [rng.normal(0, .2, (1, 2, 4, 6)) for _ in range(3)]
        engine = candidate.QwenArithmetic()
        rule = dict(centered_domain=(-2., 2.), squarings=3, initial_reciprocal_iterations=6,
            reciprocal_iterations_by_keys={str(n): candidate.reciprocal_iterations(.5/n, 2.) for n in range(1, 5)})
        actual = engine.causal_attention(candidate.Value(q), candidate.Value(k), candidate.Value(v), rule, 'causal')
        scores = q@k.swapaxes(-1, -2)/np.sqrt(q.shape[-1])
        masked = np.where(np.tril(np.ones((4, 4), bool)), scores, -np.inf)
        probs = np.exp(masked-masked.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(actual.data, probs@v, atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(actual.data[:, :, 0], v[:, :, 0])
        self.assertEqual(engine.domain_failures(), {})
        self.assertFalse(any('query0' in label for label in engine.domains))

    def test_future_keys_do_not_affect_earlier_outputs(self):
        rng = np.random.default_rng(6)
        q, k, v = [rng.normal(0, .1, (1, 2, 4, 6)) for _ in range(3)]
        rule = dict(centered_domain=(-2., 2.), squarings=3, initial_reciprocal_iterations=6,
            reciprocal_iterations_by_keys={str(n): candidate.reciprocal_iterations(.5/n, 2.) for n in range(1, 5)})
        def run(key, value):
            return candidate.QwenArithmetic().causal_attention(candidate.Value(q), candidate.Value(key), candidate.Value(value), rule, 'causal').data
        expected = run(k, v)
        k[:, :, 2:] *= -2
        v[:, :, 2:] += 10
        actual = run(k, v)
        np.testing.assert_array_equal(actual[:, :, :2], expected[:, :, :2])
        self.assertGreater(np.max(np.abs(actual[:, :, 2:]-expected[:, :, 2:])), 1)

    def test_full_graph_gqa_head_width_different_from_hidden(self):
        config = dict(hidden_size=6, intermediate_size=9, vocab_size=11,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4,
            num_hidden_layers=2, rms_norm_eps=1e-6, rope_theta=1e6)
        weights = REFERENCE.toy_weights(config, 721)
        for sequence in (1, 3, 4):
            with self.subTest(sequence=sequence):
                embedding = weights['model.embed_tokens.weight'][np.arange(sequence)[None]]
                exact = candidate.trace_exact(config, weights, embedding, REFERENCE)
                oracle = REFERENCE.forward(config, weights, embedding)
                np.testing.assert_allclose(exact['logits'], oracle, atol=1e-14, rtol=1e-14)
                manifest, _ = candidate.choose_manifest([exact])
                approximate, engine, blocks, operators = candidate.approximate_forward(config, weights, embedding, manifest)
                np.testing.assert_allclose(approximate.data, oracle, atol=1e-4, rtol=1e-4)
                self.assertEqual(approximate.data.shape, (1, 11))
                self.assertEqual(set(operators), set(exact['operators']))
                self.assertEqual(len(blocks), 2)
                self.assertEqual(engine.domain_failures(), {})

    def test_rational_silu_matches_function_on_wide_interval(self):
        values = np.linspace(-149., 316., 16385)
        rule = dict(domain=(-149., 316.), method='rational_tanh', doublings=8,
                    base_degree=12, reciprocal_iterations=5)
        engine = candidate.QwenArithmetic()
        actual = engine.silu(candidate.Value(values), rule, 'silu')
        expected = values*np.exp(-np.logaddexp(0, -values))
        np.testing.assert_allclose(actual.data, candidate.silu_rational(values, rule), atol=3e-12, rtol=3e-12)
        self.assertLess(np.max(np.abs(actual.data-expected)), 2e-6)
        self.assertEqual(engine.domain_failures(), {})

    def test_nrmse_floor_and_nonfinite_gate(self):
        checks = candidate.metric(np.full((1, 3), .001), np.zeros((1, 3)))
        self.assertAlmostEqual(checks['nrmse'], .1)
        self.assertFalse(candidate.metric(np.full((1, 3), np.nan), np.zeros((1, 3)))['finite'])
        self.assertFalse(candidate.metric(np.zeros((1, 2)), np.zeros((1, 3)))['shape_matches'])


if __name__ == '__main__':
    unittest.main()

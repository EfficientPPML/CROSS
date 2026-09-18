"""
Finite Field Test Suite

Test cases:
- Montgomery Single Modulus Context
- Barrett Single Modulus Context
- Shoup Single Modulus Context

Terminology:
- Modulus: Single form of modulus.

Usage:
- Specify the overall modulus for the context, and corresponding parameter required for the modular reduction.
- Then feed "modulus" and "parameters" to the context constructor.
- Then context->modular_reduction(input) to get the reduced result for certain inputs.
"""

from absl.testing import absltest
import warnings
from absl.testing import parameterized
import finite_field as ff_context
import jax.numpy as jnp
import jax
import numpy as np
import util

testing_params = [{'testcase_name': '0'}]

@parameterized.named_parameters(testing_params)
class FiniteFieldTest(parameterized.TestCase):
  def setUp(self):
    # Setup random input data and their modmul reference results.
    self.modulus = util.find_moduli_ntt(1, 31, 16)[0]
    self.random_key = jax.random.key(0)
    self.a = jax.random.randint(self.random_key, (1,), 0, self.modulus-1, dtype=jnp.int32)
    self.b = jax.random.randint(self.random_key, (1,), 0, self.modulus-1, dtype=jnp.int32)
    self.ab = self.a.astype(jnp.uint64) * self.b.astype(jnp.uint64)
    self.ab_modq = (self.ab % self.modulus).astype(jnp.uint32)

  # @absltest.skip("test single implementation")
  def test_montgomery_single_moduli_context(self):
    context = ff_context.MontgomeryContext(self.modulus)
    a_mont = context.to_computation_format(self.a[0].astype(jnp.uint64))
    b_mont = context.to_computation_format(self.b[0].astype(jnp.uint64))
    ab_mont = a_mont.astype(jnp.uint64) * b_mont.astype(jnp.uint64)
    result_mont = context.modular_reduction(ab_mont)
    result = context.to_original_format(result_mont.astype(jnp.uint64))
    np.testing.assert_array_equal(result[0], self.ab_modq)

  # @absltest.skip("test single implementation")
  def test_barrett_single_moduli_context(self):
    context = ff_context.BarrettContext(self.modulus)
    ab = self.a.astype(jnp.uint64) * self.b.astype(jnp.uint64)
    result = context.modular_reduction(ab)
    np.testing.assert_array_equal(result[0], self.ab_modq)

  # @absltest.skip("test single implementation")
  def test_shoup_single_moduli_context(self):
    context = ff_context.ShoupContext(self.modulus)
    warnings.warn("Shoup's reduction requires one operand to be known ahead of time.")
    a_precomputed = context.precompute_constant_operand(self.a.astype(jnp.uint64))
    ab = self.a.astype(jnp.uint64) * self.b.astype(jnp.uint64)
    ab_shoup = a_precomputed * self.b.astype(jnp.uint64)
    result_shoup = context.modular_reduction(ab, ab_shoup)
    result = context.to_original_format(result_shoup.astype(jnp.uint64))
    np.testing.assert_array_equal(result[0], self.ab_modq)

  def test_barrett_small_modulus_congruence(self):
    """Regression: Barrett split mask must be (1<<w)-1; 32749 (w=30) fails with a hard-coded 0xFFFFFFFF for inputs past 2^(2*ceil(log2 q))."""
    for q in (40961, 32749):
      context = ff_context.BarrettContext(q)
      key = jax.random.key(q)
      z = jax.random.bits(key, (40000,), dtype=jnp.uint32).astype(jnp.uint64)
      out = np.asarray(context.modular_reduction(z)).astype(np.uint64)
      exp = np.asarray(z).astype(np.uint64) % q
      np.testing.assert_array_equal(out, exp)

  def test_barrett_w32_uses_static_low_limb(self):
    """Common CKKS moduli must not build a dynamic mask in accelerator HLO."""
    value = jnp.ones((1,), dtype=jnp.uint64)
    common_context = ff_context.BarrettContext(269402113)
    common_primitives = {
        equation.primitive.name
        for equation in jax.make_jaxpr(
            common_context.modular_reduction
        )(value).jaxpr.eqns
    }
    self.assertNotIn('shift_left', common_primitives)

    small_context = ff_context.BarrettContext(257)
    small_primitives = {
        equation.primitive.name
        for equation in jax.make_jaxpr(
            small_context.modular_reduction
        )(value).jaxpr.eqns
    }
    self.assertIn('shift_left', small_primitives)

  def test_barrett_full_u32_moduli_use_exact_multiply_high(self):
    """Full-width uint32 moduli avoid overflow in the s=64 quotient estimate."""
    moduli = [2147565569, 4_000_000_007]
    lhs = jnp.array([1991667896, 3560135641], dtype=jnp.uint64)
    rhs = jnp.array([1680396232, 2463123356], dtype=jnp.uint64)
    products = lhs * rhs
    expected = np.array([194502194, 529288040], dtype=np.uint32)
    context = ff_context.BarrettContext(moduli)

    np.testing.assert_array_equal(
        np.asarray(context.modular_reduction(products)), expected
    )
    reduce_one = jax.jit(
        lambda value, index: context.modular_reduction_single_modulus(
            value, index
        )
    )
    for modulus_index, product in enumerate(products):
      with self.subTest(modulus=moduli[modulus_index]):
        np.testing.assert_array_equal(
            np.asarray(
                context.modular_reduction_single_modulus(
                    product, modulus_index
                )
            ),
            expected[modulus_index],
        )
        self.assertEqual(
            int(reduce_one(product, jnp.int32(modulus_index))),
            int(expected[modulus_index]),
        )

    q = 4_000_000_007
    z = jnp.uint64(8_627_448_631_396_274_408)
    context = ff_context.BarrettContext(q)
    self.assertEqual(int(context.modular_reduction(z)[0]), 298_239_330)
    self.assertEqual(
        int(context.modular_reduction_single_modulus(z, 0)), 298_239_330
    )

  # @absltest.skip("test single implementation")
  def test_bat_lazy_single_moduli_context(self):
    context = ff_context.BATLazyContext(self.modulus)
    warnings.warn("BATLazy's reduction requires one operand to be known ahead of time.")
    result = context.modular_reduction(self.ab)
    # Check mathematical correctness: result % modulus == expected % modulus
    # Note: Lazy reduction guarantees result is congruent to ab mod q, but not necessarily strictly < q.
    # We verify the congruence property.
    res_mod = context.to_original_format(result.astype(jnp.uint64))
    np.testing.assert_array_equal(res_mod[0], self.ab_modq)


class MultiModuliFiniteFieldTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.moduli = util.find_moduli_ntt(4, 29, 32)
    shape = (3, 2, 16, len(self.moduli))
    self.a = util.random_batched_ciphertext(
        shape, self.moduli, dtype=jnp.uint32
    ).astype(jnp.uint64)
    self.b = util.random_batched_ciphertext(
        shape, self.moduli, dtype=jnp.uint32
    ).astype(jnp.uint64)
    self.ab = self.a * self.b
    moduli = jnp.asarray(self.moduli, dtype=jnp.uint64)
    self.ab_modq = (self.ab % moduli).astype(jnp.uint32)

  def test_montgomery_multi_moduli_context(self):
    context = ff_context.MontgomeryContext(self.moduli)
    a_mont = context.to_computation_format(self.a)
    b_mont = context.to_computation_format(self.b)
    result_mont = context.modular_reduction(
        a_mont.astype(jnp.uint64) * b_mont.astype(jnp.uint64)
    )
    result = context.to_original_format(result_mont.astype(jnp.uint64))
    np.testing.assert_array_equal(result, self.ab_modq)

  def test_barrett_multi_moduli_context(self):
    context = ff_context.BarrettContext(self.moduli)
    np.testing.assert_array_equal(
        context.modular_reduction(self.ab), self.ab_modq
    )

  def test_shoup_multi_moduli_context(self):
    context = ff_context.ShoupContext(self.moduli)
    a_precomputed = context.precompute_constant_operand(self.a)
    result_shoup = context.modular_reduction(
        self.ab, a_precomputed * self.b
    )
    result = context.to_original_format(result_shoup.astype(jnp.uint64))
    np.testing.assert_array_equal(result, self.ab_modq)


class CiphertextBoundaryValidationTest(absltest.TestCase):

  def test_rank_five_array_requires_canonical_dtype(self):
    shape = (1, 2, 2, 4, 3)
    value = jnp.zeros(shape, dtype=jnp.uint32)
    self.assertIs(
        ff_context.check_rank5_array(
            value, 'test', batch=1, num_elements=2,
            degree_layout=(2, 4), num_moduli=3
        ),
        value,
    )
    with self.assertRaisesRegex(ValueError, 'ciphertext dtype uint32'):
      ff_context.check_rank5_array(value.astype(jnp.uint64), 'test')

  def test_tiled_operator_layout_must_match_kernel(self):
    self.assertEqual(
        ff_context.canonical_degree_layout(2, 4, None, 'test'), (2, 4)
    )
    with self.assertRaisesRegex(ValueError, 'tiled kernel layout'):
      ff_context.canonical_degree_layout(2, 4, (1, 8), 'test')


if __name__ == "__main__":
  absltest.main()

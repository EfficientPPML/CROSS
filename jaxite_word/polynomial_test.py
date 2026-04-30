"""
Polynomial class test suite.

Tests core Polynomial arithmetic operations: mul, modmul, mod_reduce,
and NTT round-trip conversion.
"""

import jax
import jax.numpy as jnp
import numpy as np
import finite_field as ff_context
import polynomial as poly
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)


class PolynomialDomainConversionTest(absltest.TestCase):
  """Test Polynomial domain conversion methods (NTT, coefficients, compute format)."""

  def setUp(self):
    super().setUp()
    self.shapes = {
      "batch": 3,
      "num_elements": 2,
      "num_moduli": 4,
      "degree": 16,
      "precision": 29,
    }
    self.params_barrett = {"r": 4, "c": 4, "finite_field_context": ff_context.BarrettContext}
    self.params_montgomery = {"r": 4, "c": 4, "finite_field_context": ff_context.MontgomeryContext}
    self.params_shoup = {"r": 4, "c": 4, "finite_field_context": ff_context.ShoupContext}

  def _round_trip(self, parameters):
    p = poly.Polynomial(self.shapes, parameters)
    p.random_init()
    original = p.get_batch_polynomial()
    p.to_compute_format()
    p.to_ntt_form()
    p.to_coeffs_form()
    p.to_original_format()
    np.testing.assert_array_equal(original, p.get_batch_polynomial())

  def test_ntt_round_trip_barrett(self):
    self._round_trip(self.params_barrett)

  def test_ntt_round_trip_montgomery(self):
    self._round_trip(self.params_montgomery)

  def test_ntt_round_trip_shoup(self):
    self._round_trip(self.params_shoup)

  def test_to_ntt_form_and_back(self):
    p = poly.Polynomial(self.shapes, self.params_barrett)
    p.random_init()
    original = p.get_batch_polynomial()
    p.to_ntt_form()
    # After NTT, data should differ from original coefficients
    self.assertFalse(jnp.array_equal(original, p.get_batch_polynomial()))
    p.to_coeffs_form()
    np.testing.assert_array_equal(original, p.get_batch_polynomial())

  def test_to_compute_format_and_back(self):
    p = poly.Polynomial(self.shapes, self.params_montgomery)
    p.random_init()
    original = p.get_batch_polynomial()
    p.to_compute_format()
    p.to_original_format()
    np.testing.assert_array_equal(original, p.get_batch_polynomial())


class PolynomialArithmeticTest(parameterized.TestCase):
  """Test Polynomial arithmetic: mul, modmul, mod_reduce."""

  def setUp(self):
    super().setUp()
    self.shapes = {
      "batch": 1,
      "num_elements": 1,
      "num_moduli": 2,
      "degree": 8,
      "precision": 29,
    }
    self.ct1 = poly.Polynomial(self.shapes)
    self.ct1.random_init()
    self.ct2 = poly.Polynomial(self.shapes, {'moduli': self.ct1.get_moduli()})
    self.ct2.random_init()

    self.arr1 = self.ct1.get_batch_polynomial()
    self.arr2 = self.ct2.get_batch_polynomial()

  def test_mul_polynomial(self):
    expected = self.arr1.astype(jnp.uint64) * self.arr2.astype(jnp.uint64)
    self.ct1.mul(self.ct2)
    np.testing.assert_array_equal(self.ct1.get_batch_polynomial(), expected)

  def test_mul_array(self):
    expected = self.arr1.astype(jnp.uint64) * self.arr2.astype(jnp.uint64)
    self.ct1.mul(self.arr2)
    np.testing.assert_array_equal(self.ct1.get_batch_polynomial(), expected)

  def test_modmul_polynomial(self):
    expected_temp = self.arr1.astype(jnp.uint64) * self.arr2.astype(jnp.uint64)
    expected = self.ct1.ntt_ctx.ff_ctx.modular_reduction(expected_temp).astype(self.ct1.modulus_dtype)

    self.ct1.modmul(self.ct2)
    np.testing.assert_array_equal(self.ct1.get_batch_polynomial(), expected)

  def test_modmul_array(self):
    expected_temp = self.arr1.astype(jnp.uint64) * self.arr2.astype(jnp.uint64)
    expected = self.ct1.ntt_ctx.ff_ctx.modular_reduction(expected_temp).astype(self.ct1.modulus_dtype)

    self.ct1.modmul(self.arr2)
    np.testing.assert_array_equal(self.ct1.get_batch_polynomial(), expected)

  def test_mod_reduce(self):
    self.ct1.polynomial = self.ct1.polynomial.astype(jnp.uint64) * 100
    expected = self.ct1.ntt_ctx.ff_ctx.modular_reduction(self.ct1.polynomial).astype(self.ct1.modulus_dtype)

    self.ct1.mod_reduce()
    np.testing.assert_array_equal(self.ct1.get_batch_polynomial(), expected)


if __name__ == "__main__":
  absltest.main()

"""
Polynomial class test suite.

Tests canonical payload invariants, core arithmetic, modulus dropping, and
NTT/pytree round trips.
"""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import finite_field as ff_context
import ntt_mm as ntt
import polynomial as poly
import util
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)


class PolynomialPayloadTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.shapes = {
        "batch": 2,
        "num_elements": 2,
        "num_moduli": 3,
        "degree": 32,
        "precision": 29,
    }

  def test_constructor_derives_stable_rank_five_layout(self):
    p = poly.Polynomial(self.shapes)

    self.assertEqual(p.degree_layout, (4, 8))
    self.assertEqual(p.shape, (2, 2, 4, 8, 3))
    self.assertIs(p.validate(), p)

  def test_constructor_validates_explicit_layout(self):
    shapes = {**self.shapes, "degree_layout": (2, 16)}
    self.assertEqual(poly.Polynomial(shapes).shape, (2, 2, 2, 16, 3))

    with self.assertRaisesRegex(ValueError, "exactly two"):
      poly.Polynomial({**self.shapes, "degree_layout": (32,)})
    with self.assertRaisesRegex(ValueError, "does not multiply"):
      poly.Polynomial({**self.shapes, "degree_layout": (2, 15)})

  def test_direct_assignment_rejects_legacy_and_wrong_shapes(self):
    p = poly.Polynomial(self.shapes)

    for shape in ((2, 32, 3), (2, 2, 32, 3)):
      with self.subTest(shape=shape):
        with self.assertRaisesRegex(ValueError, "rank 5"):
          p.polynomial = jnp.zeros(shape, dtype=jnp.uint32)
    with self.assertRaisesRegex(ValueError, "payload shape"):
      p.polynomial = jnp.zeros((1, 2, 4, 8, 3), dtype=jnp.uint32)

  def test_from_array_replace_payload_and_to_array(self):
    payload = jnp.arange(2 * 2 * 4 * 8 * 3, dtype=jnp.uint32).reshape(
        2, 2, 4, 8, 3
    )
    p = poly.Polynomial.from_array(payload, self.shapes)
    np.testing.assert_array_equal(p.to_array(), payload)

    replacement = payload + 1
    self.assertIs(p.replace_payload(replacement), p)
    np.testing.assert_array_equal(p.to_array(), replacement)
    with self.assertRaisesRegex(ValueError, "rank 5"):
      p.replace_payload(replacement.reshape(2, 2, 32, 3))

  def test_setters_preserve_canonical_payload(self):
    p = poly.Polynomial(self.shapes)
    p.set_polynomial(0, jnp.ones((2, 4, 8, 3), dtype=jnp.uint32))
    p.set_element(1, jnp.ones((2, 4, 8, 3), dtype=jnp.uint32))
    p.set_limb(2, jnp.ones((2, 2, 4, 8), dtype=jnp.uint32))
    self.assertEqual(p.shape, (2, 2, 4, 8, 3))
    with self.assertRaisesRegex(ValueError, "rank 5"):
      p.set_batch_polynomial(jnp.zeros((2, 2, 32, 3), dtype=jnp.uint32))

  def test_batch_slice_preserves_canonical_metadata(self):
    p = poly.Polynomial(self.shapes)
    p.random_init()

    item = p.batch_slice(1)

    self.assertEqual(item.shape, (1, 2, 4, 8, 3))
    self.assertEqual(tuple(item.moduli), tuple(p.moduli))
    np.testing.assert_array_equal(item.to_array(), p.to_array()[1:2])
    with self.assertRaises(IndexError):
      p.batch_slice(2)

  def test_compatible_shallow_copy_reuses_jit_compilation(self):
    p = poly.Polynomial(self.shapes)
    cloned = copy.copy(p)
    traces = 0

    def identity(value):
      nonlocal traces
      traces += 1
      return value

    compiled_identity = jax.jit(identity)
    compiled_identity(p).to_array().block_until_ready()
    compiled_identity(cloned).to_array().block_until_ready()

    self.assertEqual(
        jax.tree_util.tree_structure(cloned),
        jax.tree_util.tree_structure(p),
    )
    self.assertEqual(traces, 1)
    self.assertIs(cloned.validate(), cloned)

  def test_old_pytree_definition_survives_lower_level_clone(self):
    p = poly.Polynomial(self.shapes)
    old_leaves, old_treedef = jax.tree_util.tree_flatten(p)

    lowered = p.drop_last_modulus()
    jax.tree_util.tree_flatten(lowered)
    restored = jax.tree_util.tree_unflatten(old_treedef, old_leaves)

    self.assertEqual(restored.num_moduli, 3)
    self.assertEqual(restored.shape, (2, 2, 4, 8, 3))
    restored.validate()

  def test_drop_last_modulus_slices_payload_and_context(self):
    p = poly.Polynomial(self.shapes)
    p.random_init()
    original = p.to_array()
    original_moduli = tuple(p.moduli)

    result = p.drop_last_modulus()

    np.testing.assert_array_equal(result.to_array(), original[..., :-1])
    self.assertEqual(result.shape, (2, 2, 4, 8, 2))
    self.assertEqual(result.num_moduli, 2)
    self.assertEqual(tuple(result.moduli), original_moduli[:-1])
    self.assertEqual(tuple(result.moduli_array.shape), (2,))
    self.assertEqual(len(result.ntt_ctx.ff_ctx.moduli), 2)
    self.assertEqual(p.shape, (2, 2, 4, 8, 3))
    self.assertEqual(tuple(p.moduli), original_moduli)
    p.validate()
    result.validate()

  def test_drop_last_modulus_supports_mod_reduce_only_context(self):
    ff_ctx = ff_context.BarrettContext(moduli=[17, 97, 113])
    p = poly.Polynomial(
        self.shapes,
        {'moduli': [17, 97, 113], 'ntt_ctx': ff_ctx},
    )
    lowered = p.drop_last_modulus()

    self.assertIs(lowered.ntt_ctx, lowered.ntt_ctx.ff_ctx)
    self.assertEqual(lowered.moduli, [17, 97])
    self.assertEqual(lowered.shape, (2, 2, 4, 8, 2))

  def test_pytree_and_jit_round_trip_validate_payload(self):
    p = poly.Polynomial(self.shapes)
    p.random_init()

    leaves, treedef = jax.tree_util.tree_flatten(p)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    compiled = jax.jit(lambda value: value)(p)

    np.testing.assert_array_equal(restored.to_array(), p.to_array())
    np.testing.assert_array_equal(compiled.to_array(), p.to_array())
    self.assertEqual(compiled.shape, p.shape)

    _, pytree_id = p.tree_flatten()
    with self.assertRaisesRegex(ValueError, "rank 5"):
      poly.Polynomial.tree_unflatten(
          pytree_id,
          (p.to_array().reshape(2, 2, 32, 3),),
      )


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

  def test_ntt_round_trip_bat_lazy_injected(self):
    # BATLazyContext injection replaces the deleted BAT_lazy boolean.
    self._round_trip(
        {"r": 4, "c": 4, "finite_field_context": ff_context.BATLazyContext}
    )

  def test_shoup_not_injectable_via_finite_field_context(self):
    # Shoup's two-operand modular_reduction has no hook; must raise, not silently dispatch.
    with self.assertRaises(ValueError):
      poly.Polynomial(self.shapes, self.params_shoup)

  def test_ntt_round_trip_shoup_via_ntt_ctx(self):
    # Shoup stays usable via direct 'ntt_ctx' injection.
    moduli = util.find_moduli_ntt(
        self.shapes["num_moduli"],
        self.shapes["precision"],
        2 * self.shapes["degree"],
    )
    shoup_ntt = ntt.NTTCiphertextShoupContext(
        moduli=moduli,
        parameters={
            "r": 4,
            "c": 4,
            "finite_field_context": ff_context.ShoupContext(moduli=moduli),
        },
    )
    self._round_trip({"moduli": moduli, "ntt_ctx": shoup_ntt})

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


class PolynomialPrivateArithmeticTest(parameterized.TestCase):
  """Test private mutable arithmetic helpers used by kernel diagnostics."""

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
    self.ct1._mul_wide_inplace(self.ct2)
    np.testing.assert_array_equal(self.ct1.get_batch_polynomial(), expected)

  def test_mul_array(self):
    with self.assertRaisesRegex(TypeError, "Polynomial operand"):
      self.ct1._mul_wide_inplace(self.arr2)

  def test_modmul_polynomial(self):
    expected_temp = self.arr1.astype(jnp.uint64) * self.arr2.astype(jnp.uint64)
    expected = self.ct1.ntt_ctx.ff_ctx.modular_reduction(expected_temp).astype(self.ct1.modulus_dtype)

    self.ct1._modmul_inplace(self.ct2)
    np.testing.assert_array_equal(self.ct1.get_batch_polynomial(), expected)

  def test_modmul_array(self):
    with self.assertRaisesRegex(TypeError, "Polynomial operand"):
      self.ct1._modmul_inplace(self.arr2)

  def test_mod_reduce(self):
    self.ct1.polynomial = self.ct1.polynomial.astype(jnp.uint64) * 100
    expected = self.ct1.ntt_ctx.ff_ctx.modular_reduction(self.ct1.polynomial).astype(self.ct1.modulus_dtype)

    self.ct1._mod_reduce_inplace()
    np.testing.assert_array_equal(self.ct1.get_batch_polynomial(), expected)


if __name__ == "__main__":
  absltest.main()

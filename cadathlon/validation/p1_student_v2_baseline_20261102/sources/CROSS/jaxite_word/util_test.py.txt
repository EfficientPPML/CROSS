"""Tests for guarded exact NumPy utility paths."""

import math
from unittest import mock

from absl.testing import absltest
import jax
import numpy as np

import util

try:
  import pytest
except ModuleNotFoundError:
  pytestmark = []
else:
  pytestmark = [
      pytest.mark.correctness,
      pytest.mark.security,
      pytest.mark.unit,
  ]


def _reference_negacyclic_ntt(values, modulus, psi):
  """Compute an independent quadratic negacyclic NTT oracle."""
  omega = pow(psi, 2, modulus)
  return [
      sum(
          (int(value) % modulus)
          * pow(psi, coefficient, modulus)
          * pow(omega, coefficient * frequency, modulus)
          for coefficient, value in enumerate(values)
      ) % modulus
      for frequency in range(len(values))
  ]


class GuardedNttExactnessTest(absltest.TestCase):

  def test_correct_check_defines_numpy_exactness_envelope(self):
    psi = util.root_of_unity(16, 97)
    self.assertTrue(util._correct_check([97], degree=8, psi=psi))
    self.assertTrue(util._correct_check(degree=8))
    self.assertFalse(
        util._correct_check([1 << 31], degree=8, psi=psi)
    )
    self.assertFalse(util._correct_check([97], degree=6))
    self.assertFalse(util._correct_check([97], degree=8, psi=1))

  def test_fast_ntt_reduces_arbitrary_precision_inputs_exactly(self):
    modulus = 97
    degree = 8
    psi = util.root_of_unity(2 * degree, modulus)
    values = [
        (-1 if i % 2 else 1) * ((1 << (130 + i)) + 17 * i)
        for i in range(degree)
    ]

    with mock.patch.object(
        util,
        "ntt_negacyclic_bit_reverse_np",
        wraps=util.ntt_negacyclic_bit_reverse_np,
    ) as numpy_forward:
      transformed = util.ntt_negacyclic_bit_reverse(
          values, modulus, psi
      )

    numpy_forward.assert_called_once()
    self.assertEqual(
        transformed,
        _reference_negacyclic_ntt(values, modulus, psi),
    )
    self.assertEqual(
        util.intt_negacyclic_bit_reverse(
            transformed, modulus, psi
        ),
        [value % modulus for value in values],
    )

  def test_large_modulus_uses_scalar_differential_and_round_trips(self):
    modulus = 1099511627873
    psi = 108163207722
    values = [
        140719340484,
        258793550908,
        1037017667747,
        836022225947,
        462949041686,
        1069849980556,
        858288075698,
        586028475693,
    ]
    expected = _reference_negacyclic_ntt(values, modulus, psi)

    with mock.patch.object(
        util,
        "ntt_negacyclic_bit_reverse_np",
        side_effect=AssertionError("unsafe NumPy path selected"),
    ):
      transformed = util.ntt_negacyclic_bit_reverse(
          values, modulus, psi
      )
      array_transformed = util.ntt_negacyclic_bit_reverse(
          np.asarray(values, dtype=np.uint64), modulus, psi
      )

    self.assertEqual(transformed, expected)
    self.assertEqual(array_transformed.dtype, object)
    self.assertEqual(array_transformed.tolist(), expected)
    self.assertEqual(
        util.intt_negacyclic_bit_reverse(
            transformed, modulus, psi
        ),
        values,
    )
    with self.assertRaisesRegex(ValueError, "correctness envelope"):
      util.ntt_negacyclic_bit_reverse_np(values, modulus, psi)

  def test_public_ntt_rejects_invalid_radix_two_parameters(self):
    with self.assertRaisesRegex(ValueError, "power-of-two"):
      util.ntt_negacyclic_bit_reverse([0, 1, 2], 7, 1)
    with self.assertRaisesRegex(ValueError, "primitive"):
      util.ntt_negacyclic_bit_reverse([0, 1, 2, 3], 17, 1)


class PTowerGenerationTest(absltest.TestCase):

  def test_exact_product_extends_bit_length_lower_bound(self):
    q_towers = [31, 23]
    self.assertEqual(util.q_partition_products(q_towers, dnum=1), [31 * 23])
    self.assertEqual(
        util.compute_num_p_towers(q_towers, dnum=1, aux_bits=5), 2
    )

    p_towers = util.generate_p_towers(
        q_towers, dnum=1, degree=2, aux_bits=5
    )
    self.assertLen(p_towers, 3)
    self.assertGreaterEqual(math.prod(p_towers), math.prod(q_towers))


class BatControlGenerationTest(absltest.TestCase):

  def test_host_shifted_mod_bytes_matches_jax_path(self):
    values = np.array(
        [[0, 1, 2], [0x12345678, 2147480000, 2147481000]],
        dtype=np.uint64,
    )
    moduli = np.array([2147483489, 2147483137, 2147482817], dtype=np.uint64)

    actual = util.shifted_mod_bytes_host(values, moduli)
    with jax.enable_x64():
      expected = np.asarray(util.shifted_mod_bytes(values, moduli))

    self.assertEqual(actual.dtype, np.uint8)
    self.assertEqual(actual.shape, (4, 2, 3, 4))
    np.testing.assert_array_equal(actual, expected)


class UtilityInputValidationTest(absltest.TestCase):

  def test_ciphertext_parser_rejects_ragged_evaluations(self):
    serialized = """\
Element 0:
0: EVAL: [1 2] modulus: 17
1: EVAL: [3] modulus: 19
"""
    with self.assertRaises(ValueError):
      util.parse_ciphertext_string(serialized)

  def test_root_of_unity_rejects_an_incompatible_order(self):
    with self.assertRaisesRegex(ValueError, "q-1 must be divisible"):
      util.root_of_unity(6, 17)

  def test_gamma_beta_requires_a_source_and_dropped_modulus(self):
    with self.assertRaisesRegex(ValueError, "at least 2 moduli"):
      util.gamma_beta_calculation([17])

  def test_random_ciphertext_requires_one_modulus_per_tower(self):
    with self.assertRaisesRegex(ValueError, "final shape dimension"):
      util.random_ciphertext((1, 8, 2), [17])
    with self.assertRaisesRegex(ValueError, "final shape dimension"):
      util.random_batched_ciphertext((1, 2, 2, 2, 2), [17])

  def test_host_shifted_mod_bytes_accepts_a_noncontiguous_matrix(self):
    values = np.arange(12, dtype=np.uint64).reshape(2, 2, 3).transpose(1, 0, 2)
    self.assertFalse(values.flags.c_contiguous)
    moduli = np.array([2147483489, 2147483137, 2147482817], dtype=np.uint64)

    actual = util.shifted_mod_bytes_host(values, moduli)
    with jax.enable_x64():
      expected = np.asarray(util.shifted_mod_bytes(values, moduli))

    np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
  absltest.main()

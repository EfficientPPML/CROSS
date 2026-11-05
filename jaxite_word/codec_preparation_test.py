"""Exact regression checks for host-side CKKS plaintext preparation."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
from absl.testing import absltest, parameterized

import ckks_ctx


class NativeEncodeResiduesTest(parameterized.TestCase):
  @parameterized.product(
      bound=[127, (1 << 61) - 1, (1 << 63) - 513],
      log_approx=[0, 5, 70],
      noise_scale_degree=[1, 2, 3],
  )
  def test_matches_scalar_fit_and_arbitrary_precision_scaling(
      self, bound, log_approx, noise_scale_degree,
  ):
    rng = np.random.default_rng(19)
    coefficients = [0, 1, bound // 2, bound // 2 + 1, bound - 1, bound]
    coefficients += rng.integers(0, bound, size=64).tolist()
    moduli = [17, 65537, 1073741857, 2147352577]
    scale = float(1 << 60)
    actual = ckks_ctx._native_encode_residues(
        coefficients, bound, moduli, log_approx, scale, noise_scale_degree,
    )
    multiplier = (1 << log_approx) * pow(int(round(scale)), noise_scale_degree - 1)
    expected = np.asarray([
        [(value * multiplier) % q for value in
         ckks_ctx.fit_to_native_vector(coefficients, bound, q, len(coefficients))]
        for q in moduli
    ], dtype=np.uint64).T
    np.testing.assert_array_equal(actual, expected)

  @parameterized.parameters(
      ([-1], 127, [17]),
      ([1 << 64], 127, [17]),
      ([1], 1 << 64, [17]),
      ([1], 127, [1 << 32]),
  )
  def test_preserves_scalar_fallback(self, coefficients, bound, moduli):
    self.assertIsNone(ckks_ctx._native_encode_residues(
        coefficients, bound, moduli, 0, 16.0, 1,
    ))

  def test_sparse_slot_encoding_preserves_ring_degree(self):
    coefficients, bound, moduli = [1, 2, 126, 125], 127, [17, 65537]
    actual = ckks_ctx._native_encode_residues(
        coefficients, bound, moduli, 0, 16.0, 1, ring_dim=32,
    )
    expected = np.asarray([
        ckks_ctx.fit_to_native_vector(coefficients, bound, q, 32)
        for q in moduli
    ], dtype=np.uint64).T
    np.testing.assert_array_equal(actual, expected)


class NativeInverseFFTTest(parameterized.TestCase):
  @parameterized.product(slots=[1, 8, 128, 4096, 16384], complex_input=[False, True])
  def test_is_bit_exact_with_scalar_butterflies(self, slots, complex_input):
    rng = np.random.default_rng(42)
    values = rng.normal(size=slots).astype(np.complex128)
    if complex_input:
      values += 1j * rng.normal(size=slots)
    original = [complex(value) for value in values]
    expected = original.copy()
    ckks_ctx.FFTSpecialInv(expected, slots * 4)
    actual = ckks_ctx._native_special_inverse_fft(original, slots * 4)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(original, values)


class SparsePrivateEncoderTest(parameterized.TestCase):
  @parameterized.product(slot_count=[1, 2, 4, 8, 16], complex_input=[False, True])
  def test_sparse_private_encoding_matches_full_periodic_encoding(
      self, slot_count, complex_input,
  ):
    # Independent full-slot encoding establishes the represented periodic
    # values and coefficient stride, rather than mirroring the sparse code.
    degree, moduli, scale = 32, [193, 257], float(1 << 20)
    rng = np.random.default_rng(103)
    values = rng.normal(size=slot_count).astype(np.complex128)
    if complex_input:
      values += 1j * rng.normal(size=slot_count)
    full = np.tile(values, degree // 2 // slot_count)
    actual = ckks_ctx._ckks_encode(values, 2 * degree, moduli, [], scale)
    expected = ckks_ctx._ckks_encode(full, 2 * degree, moduli, [], scale)
    np.testing.assert_array_equal(actual, expected)
    # The shared context must remain a full-slot encoder after sparse use.
    lite = ckks_ctx._get_or_make_encode_ctx(
        moduli, degree, scale, 61, (1 << 63) - 513)
    self.assertEqual(lite.num_slots, degree // 2)
    np.testing.assert_array_equal(
        ckks_ctx._fast_encode(full, lite, scale=scale, allow_complex=True),
        expected)
    if slot_count != degree // 2:
      with self.assertRaisesRegex(ValueError, 'expected exactly'):
        ckks_ctx._fast_encode(values, lite, scale=scale, allow_complex=True)

  @parameterized.parameters(0, 3, 17, 32)
  def test_invalid_private_sparse_counts_rejected(self, count):
    for encode in (ckks_ctx._ckks_encode, ckks_ctx._ckks_encode_array):
      with self.assertRaisesRegex(ValueError, 'power-of-two divisor'):
        encode([1.] * count, 64, [193, 257], [], float(1 << 20))

  def test_full_slot_wrapper_stays_bit_exact_with_scalar_encoder(self):
    values = [complex(i / 32, (i % 3) / 32) for i in range(16)]
    args = (values, 64, [193, 257], [], float(1 << 20))
    np.testing.assert_array_equal(ckks_ctx._ckks_encode(*args),
                                 ckks_ctx._ckks_encode_fall_back(*args))


class ArrayPrivateEncoderTest(parameterized.TestCase):
  @parameterized.product(
      slot_count=[1, 4, 16], num_q=[1, 2], scale=[2.**20, 2.**80],
      noise_scale_degree=[1, 2], max_bits_in_word=[31, 61],
  )
  def test_matches_independent_scalar_codec_and_preserves_list_api(
      self, slot_count, num_q, scale, noise_scale_degree, max_bits_in_word,
  ):
    degree, moduli = 32, [193, 257][:num_q]
    values = [complex((i-3)/16, (i % 3)/32) for i in range(slot_count)]
    kwargs = dict(noise_scale_degree=noise_scale_degree,
                  max_bits_in_word=max_bits_in_word)
    # The independent scalar codec sees a full periodic vector. This checks
    # sparse placement and large-scale rounding without using the new helper
    # itself as the numerical oracle.
    expected = ckks_ctx._ckks_encode_fall_back(
        values*(degree//2//slot_count), 2*degree, moduli, [], scale, **kwargs)
    actual = ckks_ctx._ckks_encode_array(
        values, 2*degree, moduli, [], scale, **kwargs)
    compatibility = ckks_ctx._ckks_encode(
        values, 2*degree, moduli, [], scale, **kwargs)
    self.assertEqual(actual.dtype, np.uint64)
    self.assertEqual(actual.shape, (degree, num_q))
    self.assertIsInstance(compatibility, list)
    self.assertIsInstance(compatibility[0], list)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(compatibility, expected)

  def test_bootstrap_diagonal_uses_array_interface(self):
    import jax
    from jaxite_word.bootstrapping import Bootstrap
    jax.config.update('jax_enable_x64', True)
    # Codec-only wiring fixture: no key, ciphertext, or security claim.
    engine = object.__new__(Bootstrap)
    engine._degree, engine._num_slots, engine._sparse = 32, 4, True
    engine._cache = SimpleNamespace(degree_layout=(4, 8))
    engine.ctx = SimpleNamespace(scaling_factor=2.**20, parameters={})
    diagonal = np.array([complex(i/16, (i % 3)/32) for i in range(8)])
    expected = ckks_ctx._ckks_encode_fall_back(
        list(diagonal)*2, 64, [193, 257], [], 2.**20)
    with mock.patch.object(ckks_ctx, '_ckks_encode',
                           side_effect=AssertionError('list roundtrip used')):
      result = engine._encode_diagonal(diagonal, [193, 257], 2)
    self.assertEqual(result.dtype, np.uint32)
    self.assertEqual(result.shape, (4, 8, 2))
    np.testing.assert_array_equal(np.asarray(result).reshape(32, 2), expected)


  @parameterized.parameters(1, 2)
  def test_bootstrap_scalar_add_uses_array_interface_at_both_noise_degrees(
      self, noise_scale_degree,
  ):
    import jax
    import jax.numpy as jnp
    from jaxite_word.bootstrapping import Bootstrap
    jax.config.update('jax_enable_x64', True)
    # Codec/addition-only standard-residue fixture, with no encryption claim.
    moduli, scale, scalar = [193, 257], 2.**20, -.3125
    engine = object.__new__(Bootstrap)
    engine._degree, engine._num_slots = 32, 4
    field = SimpleNamespace(to_computation_format=lambda value: value)
    engine._cache = SimpleNamespace(degree_layout=(4, 8),
        q_moduli_at_level=lambda level: moduli,
        ff_q_max=SimpleNamespace(slice=lambda count: field))
    engine._ensure_canonical = lambda value, level: value
    engine.ctx = SimpleNamespace(parameters={})
    original = np.arange(128, dtype=np.uint64).reshape(1, 2, 4, 8, 2)
    ciphertext = SimpleNamespace(polynomial=jnp.asarray(original, dtype=jnp.uint32),
        _ckks_scale=scale**noise_scale_degree, _ckks_nsd=noise_scale_degree)
    plaintext = np.asarray(ckks_ctx._ckks_encode_fall_back(
        [complex(scalar)]*16, 64, moduli, [], scale), dtype=np.uint64)
    if noise_scale_degree == 2:
      plaintext = (plaintext*np.asarray([int(scale) % q for q in moduli], dtype=np.uint64)) % moduli
    expected = original.copy()
    expected[:, :1] = (expected[:, :1]+plaintext.reshape(1, 1, 4, 8, 2)) % moduli
    with mock.patch.object(ckks_ctx, '_ckks_encode',
                           side_effect=AssertionError('list roundtrip used')):
      result = engine._add_scalar(ciphertext, scalar, 0, noise_scale_degree)
    np.testing.assert_array_equal(result.polynomial, expected.astype(np.uint32))
    self.assertEqual(result._ckks_scale, scale**noise_scale_degree)
    self.assertEqual(result._ckks_nsd, noise_scale_degree)


if __name__ == '__main__':
  absltest.main()

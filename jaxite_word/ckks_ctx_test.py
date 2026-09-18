import math
import os
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

import ckks_ctx
import finite_field
import he_params
import polynomial
import util
import key_gen as kg
from absl.testing import absltest
from absl.testing import parameterized

try:
  import pytest
except ModuleNotFoundError:
  pytestmark = []
else:
  pytestmark = [pytest.mark.correctness, pytest.mark.integration]


Polynomial = polynomial.Polynomial

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_traceback_filtering', 'off')

testing_params = [
  {
    'testcase_name': '0',
  }
]

@parameterized.named_parameters(testing_params)
class CKKSContextTest(parameterized.TestCase):
  def setUp(self):
    self.degree = 16
    self.num_slots = 8
    self.dnum = 3
    self.r, self.c = 4, 4

    self.scaling_factor = 563019763943521
    self.q_towers = [1073742881, 1073742721, 1073741441, 1073741857, 524353]
    self.p_towers = [1073740609, 1073739937, 1073739649]
    self.q = 696985728458547852910430465530901300664961
    self.qt = 1329229981441028949792278227703286337
    self.p = 30
    self.CKKS_M_FACTOR = 1
    self.noise_scale_degree = 1
    self.max_bits_in_word = 61
    self.sigma = 3.190000057220458984375
    self.degree_layout = (self.r, self.c)

    key_pair = kg.gen_pke_pair(self.q_towers, self.p_towers, self.degree)
    self.params = {
        "degree": self.degree,
        "num_slots": self.num_slots,
        "scaling_factor": self.scaling_factor,
        "output_scale": self.scaling_factor,
        "q_towers": self.q_towers,
        "p_towers": self.p_towers,
        "p": self.p,
        "CKKS_M_FACTOR": self.CKKS_M_FACTOR,
        "max_bits_in_word": self.max_bits_in_word,
        "noise_scale_degree": self.noise_scale_degree,
        "public_key": key_pair["public_key"],
        "secret_key": key_pair["secret_key"],
        "degree_layout": self.degree_layout,
    }
    self.real_values_input_in1 = [
        complex(0.25, 0), complex(0.5, 0), complex(0.75, 0), complex(1, 0),
        complex(2, 0), complex(3, 0), complex(4, 0), complex(5, 0),
    ]
    self.real_values_input_in2 = [
        complex(5, 0), complex(4, 0), complex(3, 0), complex(2, 0),
        complex(1, 0), complex(0.75, 0), complex(0.5, 0), complex(0.25, 0),
    ]
    self.real_values_multiply_result = [
        complex(1.25, 0), complex(2, 0), complex(2.25, 0), complex(2, 0),
        complex(2, 0), complex(2.25, 0), complex(2, 0), complex(1.25, 0),
    ]
    self.real_values_rotate_result = [
        complex(0.5, 0), complex(0.75, 0), complex(1, 0), complex(2, 0),
        complex(3, 0), complex(4, 0), complex(5, 0), complex(0.25, 0),
    ]

  def test_public_slot_normalization_rejects_nonzero_imaginary_values(self):
    self.assertEqual(
        ckks_ctx._validated_slots(
            [1.0 + 0.0j, -2.0],
            name='slots',
            expected_size=None,
            allow_complex=False,
        ),
        [1.0 + 0.0j, -2.0 + 0.0j],
    )
    with self.assertRaisesRegex(TypeError, 'decoding is real-valued'):
      ckks_ctx._validated_slots(
          [1.0 + 2.0j],
          name='slots',
          expected_size=None,
          allow_complex=False,
      )
    with self.assertRaisesRegex(ValueError, 'expected exactly 2'):
      ckks_ctx._validated_slots(
          [1.0], name='slots', expected_size=2, allow_complex=False
      )

  def test_fast_decode_approximation_guard_fails_closed(self):
    valid = np.zeros(self.num_slots, dtype=np.complex128)
    ckks_ctx._validate_decode_approximation(
        valid, ckks_ctx._decode_conjugate(valid)
    )

    invalid = valid.copy()
    invalid[0] = 1.0j
    with self.assertRaisesRegex(
        ckks_ctx.ApproximationErrorTooHigh, 'approximation error is too high'
    ):
      ckks_ctx._validate_decode_approximation(
          invalid, ckks_ctx._decode_conjugate(invalid)
      )

  def test_encrypt_decrypt_cache_rejects_stale_id_reuse(self):
    """id()-keyed caches can collide after GC; plant a poisoned entry at a live key and assert _get_*_cache rebuilds instead of reusing it."""
    import copy as _copy
    ctx = ckks_ctx.CKKSContext(self.params)

    # ----- encrypt cache -----
    num_q = len(self.q_towers)
    good = ckks_ctx._get_encrypt_cache(ctx, num_q)
    key = (id(ctx), num_q)
    stale = _copy.copy(good)                 # real _EncryptCache, then poisoned
    stale.degree = good.degree * 2           # mismatched invariant
    stale.q_int = [q + 1 for q in good.q_int]
    stale._pk_ref = object()                 # different key-source identity
    ckks_ctx._encrypt_cache[key] = stale
    rebuilt = ckks_ctx._get_encrypt_cache(ctx, num_q)
    self.assertIsNot(rebuilt, stale)
    self.assertEqual(rebuilt.degree, int(ctx.degree))
    self.assertEqual(rebuilt.q_int, [int(q) for q in ctx.q_towers[:num_q]])
    self.assertIs(rebuilt._pk_ref, ctx.public_key)

    # ----- decrypt cache -----
    num_m = len(self.q_towers)
    good_d = ckks_ctx._get_decrypt_cache(ctx, num_m)
    key_d = (id(ctx), num_m)
    stale_d = _copy.copy(good_d)
    stale_d.degree = good_d.degree * 2
    stale_d.q_int = [q + 1 for q in good_d.q_int]
    stale_d._sk_ref = object()
    ckks_ctx._decrypt_cache[key_d] = stale_d
    rebuilt_d = ckks_ctx._get_decrypt_cache(ctx, num_m)
    self.assertIsNot(rebuilt_d, stale_d)
    self.assertEqual(rebuilt_d.degree, int(ctx.degree))
    self.assertIs(rebuilt_d._sk_ref, ctx.secret_key)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encode_decode(self):
    # Paramters Setup
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Decoding
    decoded_values = ctx.decode(encoded_ct, is_ntt=True)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  def test_encode_scale_is_independent_of_encryption_noise_scale_degree(self):
    params = dict(self.params)
    params['noise_scale_degree'] = 2
    ctx = ckks_ctx.CKKSContext(params)

    encoded = ctx.encode(self.real_values_input_in1)
    decoded = ctx.decode(encoded, is_ntt=True)

    np.testing.assert_array_almost_equal(
        decoded, self.real_values_input_in1, decimal=3
    )

  def test_encode_does_not_require_public_key(self):
    params = dict(self.params)
    params.pop('public_key')
    params.pop('secret_key')
    ctx = ckks_ctx.CKKSContext(params)

    encoded = ctx.encode(self.real_values_input_in1)
    decoded = ctx.decode(encoded, is_ntt=True)

    self.assertEqual(encoded.polynomial.dtype, jnp.uint32)
    np.testing.assert_array_almost_equal(
        decoded, self.real_values_input_in1, decimal=3
    )
    with self.assertRaisesRegex(ValueError, 'Public key'):
      ctx.encrypt_slots(self.real_values_input_in1)

  def test_sparse_slots_round_trip_in_reference_and_fast_decode(self):
    params = dict(self.params)
    params['num_slots'] = 2
    ctx = ckks_ctx.CKKSContext(params)
    values = [complex(0.25, 0), complex(3.0, 0)]

    ciphertext = ctx.encrypt(ctx.encode(values))
    plaintext = ctx.decrypt(ciphertext)
    reference = ctx.decode(plaintext, is_ntt=False)
    fast = ckks_ctx._fast_decrypt_decode(
        ciphertext, ctx, ctx.output_scale
    )[0]

    np.testing.assert_array_almost_equal(reference, values, decimal=3)
    np.testing.assert_array_almost_equal(fast, values, decimal=3)

  def test_rejects_invalid_sparse_slot_count(self):
    params = dict(self.params)
    params['num_slots'] = 3
    with self.assertRaisesRegex(ValueError, 'power-of-two divisor'):
      ckks_ctx.CKKSContext(params)

  def test_lazy_parameter_cache_rejects_invalid_levels(self):
    ctx = ckks_ctx.CKKSContext(self.params)
    ctx.program_initialization(
        total_rotation_indices=[], dnum=self.dnum,
        r=self.r, c=self.c, batch=1
    )

    with self.assertRaisesRegex(ValueError, r'level must be in \[0,'):
      ctx._param_cache.get_sliced_ntt_q(-1)
    with self.assertRaisesRegex(ValueError, r'level must be in \[0,'):
      ctx._param_cache.get_sliced_ntt_q(ctx.max_level + 1)

  def test_program_initialization_rejects_layout_disagreement(self):
    ctx = ckks_ctx.CKKSContext(self.params)
    with self.assertRaisesRegex(ValueError, 'must match program NTT layout'):
      ctx.program_initialization(
          total_rotation_indices=[],
          dnum=self.dnum,
          r=self.r,
          c=self.c,
          degree_layout=(2, 8),
      )

  def test_hybrid_p_tower_validation_compares_exact_products(self):
    """Equal limb widths cannot hide an insufficient auxiliary product."""
    with self.assertRaisesRegex(
        ValueError, 'P-tower product.*Q partition'
    ):
      util.validate_barrett_bconv_moduli(
          q_towers=[97, 113], p_towers=[73, 89], dnum=1
      )

    # The reverse rounding case must not reject a sufficient single P limb:
    # sum(bit_length(Q_i)) is 11, while bit_length(P) is only 10.
    util.validate_barrett_bconv_moduli(
        q_towers=[17, 41], p_towers=[769], dnum=1
    )

  def test_program_initialization_uses_exact_p_tower_validation(self):
    params = dict(self.params)
    params['q_towers'] = [97, 113]
    params['p_towers'] = [73, 89]
    ctx = ckks_ctx.CKKSContext(params)

    with self.assertRaisesRegex(
        ValueError, 'P-tower product.*Q partition'
    ):
      ctx.program_initialization(
          total_rotation_indices=[], dnum=1, r=self.r, c=self.c
      )

  def test_rotation_key_noise_parameters_propagate_and_fail_closed(self):
    params = dict(self.params, sigma=6.5, noise_scale_degree=7)
    for cache_rotation_keys in (True, False):
      ctx = ckks_ctx.CKKSContext(params)
      with mock.patch.object(
          ckks_ctx.kg,
          'gen_rotation_key',
          wraps=ckks_ctx.kg.gen_rotation_key,
      ) as generate:
        ctx.program_initialization(
            total_rotation_indices=[1],
            dnum=self.dnum,
            r=self.r,
            c=self.c,
            cache_rotation_keys=cache_rotation_keys,
        )
        if not cache_rotation_keys:
          ctx._param_cache.get_rot_key(1, level=ctx.max_level)
      self.assertEqual(generate.call_args.kwargs['noise_std'], 6.5)
      self.assertEqual(generate.call_args.kwargs['noise_scale'], 7)

    for name, value, message in (
        ('sigma', 0, 'sigma must be a finite positive number'),
        (
            'noise_scale_degree',
            0,
            'noise_scale_degree must be a positive int',
        ),
    ):
      with self.subTest(name=name):
        ctx = ckks_ctx.CKKSContext(dict(self.params, **{name: value}))
        with self.assertRaisesRegex(ValueError, message):
          ctx.program_initialization(
              total_rotation_indices=[],
              dnum=self.dnum,
              r=self.r,
              c=self.c,
          )

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_decrypt(self):
    # Paramters Setup
    ctx = ckks_ctx.CKKSContext(self.params)
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Decryption
    decrypted_ct = ctx.decrypt(encrypted_ct)
    # Step 4: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  def test_combined_codecs_track_scale_independently_of_output_scale(self):
    params = dict(self.params)
    params['output_scale'] = self.scaling_factor * 2
    ctx = ckks_ctx.CKKSContext(params)

    ciphertext = ctx.encrypt_slots(
        self.real_values_input_in1, scale=self.scaling_factor
    )
    self.assertEqual(ciphertext._ckks_scale, float(self.scaling_factor))
    np.testing.assert_array_almost_equal(
        ctx.decrypt_slots(ciphertext), self.real_values_input_in1, decimal=3
    )

    batch = ctx.encrypt_slots_batch(
        [self.real_values_input_in1, self.real_values_input_in2],
        scale=self.scaling_factor,
    )
    self.assertEqual(batch._ckks_scale, float(self.scaling_factor))
    decoded_batch = ctx.decrypt_slots_batch(batch)
    np.testing.assert_array_almost_equal(
        decoded_batch[0], self.real_values_input_in1, decimal=3
    )
    np.testing.assert_array_almost_equal(
        decoded_batch[1], self.real_values_input_in2, decimal=3
    )

  def test_montgomery_combined_encrypt_boundaries_round_trip(self):
    """Fused single/batch encryption must enter Montgomery form once."""
    ctx = ckks_ctx.CKKSContext(dict(self.params))
    ctx.program_initialization(
        total_rotation_indices=[],
        dnum=self.dnum,
        r=self.r,
        c=self.c,
        batch=1,
        finite_field_context=finite_field.MontgomeryContext,
    )

    ciphertext = ctx.encrypt_slots(
        self.real_values_input_in1, scale=self.scaling_factor
    )
    self.assertIsInstance(
        ciphertext.ntt_ctx.ff_ctx, finite_field.MontgomeryContext
    )
    np.testing.assert_array_almost_equal(
        ctx.decrypt_slots(ciphertext, self.scaling_factor),
        self.real_values_input_in1,
        decimal=3,
    )

    batch = ctx.encrypt_slots_batch(
        [self.real_values_input_in1, self.real_values_input_in2],
        scale=self.scaling_factor,
    )
    decoded_batch = ctx.decrypt_slots_batch(batch, self.scaling_factor)
    np.testing.assert_array_almost_equal(
        decoded_batch[0], self.real_values_input_in1, decimal=3
    )
    np.testing.assert_array_almost_equal(
        decoded_batch[1], self.real_values_input_in2, decimal=3
    )

    encoded = ctx.encode(self.real_values_input_in1)
    separately_encrypted = ctx.encrypt(encoded)
    np.testing.assert_array_almost_equal(
        ctx.decode(ctx.decrypt(separately_encrypted)),
        self.real_values_input_in1,
        decimal=3,
    )

  def test_montgomery_level_operations_round_trip(self):
    """Injected multiply, rotation, and rescale remain in Montgomery form."""
    evaluation_key = kg.gen_evaluation_key(
        self.params['secret_key'],
        q=self.q_towers,
        P=self.p_towers,
        noise_std=self.sigma,
        noise_scale=1,
        dnum=self.dnum,
    )
    eval_key_a = jnp.asarray(
        evaluation_key['a'], dtype=jnp.uint32
    ).transpose(0, 2, 1)
    eval_key_b = jnp.asarray(
        evaluation_key['b'], dtype=jnp.uint32
    ).transpose(0, 2, 1)
    params = dict(self.params)
    params['evaluation_key'] = [eval_key_a, eval_key_b]
    ctx = ckks_ctx.CKKSContext(params)
    ctx.program_initialization(
        total_rotation_indices=[1],
        dnum=self.dnum,
        r=self.r,
        c=self.c,
        finite_field_context=finite_field.MontgomeryContext,
    )

    ciphertext1 = ctx.encrypt_slots(self.real_values_input_in1)
    ciphertext2 = ctx.encrypt_slots(self.real_values_input_in2)

    rotated = ctx.he_rot[ctx.max_level, 1].rotate(ciphertext1)
    np.testing.assert_array_almost_equal(
        ctx.decrypt_slots(rotated), self.real_values_rotate_result, decimal=3
    )

    rescaled = ctx.he_rescale[
        ctx.max_level, ctx.max_level - 1
    ].rescale(ciphertext1)
    np.testing.assert_array_almost_equal(
        ctx.decrypt_slots(rescaled), self.real_values_input_in1, decimal=3
    )

    multiplied = ctx.he_mul[ctx.max_level - 1].mul(
        ciphertext1, ciphertext2
    )
    np.testing.assert_array_almost_equal(
        ctx.decrypt_slots(multiplied),
        self.real_values_multiply_result,
        decimal=3,
    )

  def test_montgomery_public_plaintext_operations_round_trip(self):
    """Public add/mul convert standard plaintexts at the HE boundary."""
    ctx = ckks_ctx.CKKSContext(dict(self.params))
    ctx.program_initialization(
        total_rotation_indices=[],
        dnum=self.dnum,
        r=self.r,
        c=self.c,
        batch=1,
        finite_field_context=finite_field.MontgomeryContext,
    )
    level = ctx.max_level
    plaintext = ctx.encode_at_level(
        self.real_values_input_in2, level, scale=self.scaling_factor
    )
    plaintext_before = np.asarray(plaintext.polynomial).copy()
    prepared = ctx._param_cache._prepare_plaintext_payload(
        plaintext.polynomial, level
    )
    expected_prepared = ctx._param_cache.get_sliced_ff_q(
        level
    ).to_computation_format(
        plaintext.polynomial.astype(jnp.uint64)
    )
    np.testing.assert_array_equal(prepared, expected_prepared)
    np.testing.assert_array_equal(plaintext.polynomial, plaintext_before)

    multiplied = ctx.ptct_mul[level].mul(
        ctx.encrypt_slots(self.real_values_input_in1), plaintext
    )
    np.testing.assert_array_almost_equal(
        ctx.decrypt_slots(multiplied),
        self.real_values_multiply_result,
        decimal=3,
    )

    added = ctx.he_add[level].add_plain(
        ctx.encrypt_slots(self.real_values_input_in1), plaintext
    )
    expected_sum = np.asarray(self.real_values_input_in1) + np.asarray(
        self.real_values_input_in2
    )
    np.testing.assert_array_almost_equal(
        ctx.decrypt_slots(added), expected_sum, decimal=3
    )
    np.testing.assert_array_equal(plaintext.polynomial, plaintext_before)

  def test_separate_codec_path_preserves_encode_scale(self):
    params = dict(self.params)
    params['output_scale'] = self.scaling_factor * 2
    ctx = ckks_ctx.CKKSContext(params)

    encoded = ctx.encode(self.real_values_input_in1)
    ciphertext = ctx.encrypt(encoded)
    self.assertEqual(encoded._ckks_scale, float(self.scaling_factor))
    self.assertEqual(ciphertext._ckks_scale, float(self.scaling_factor))
    np.testing.assert_array_almost_equal(
        ctx.decrypt_slots(ciphertext), self.real_values_input_in1, decimal=3
    )
    decrypted = ctx.decrypt(ciphertext)
    np.testing.assert_array_almost_equal(
        ctx.decode(decrypted), self.real_values_input_in1, decimal=3
    )

  def test_context_boundaries_use_rank_five_and_reject_batches(self):
    ctx = ckks_ctx.CKKSContext(self.params)
    plaintext = ctx.encode(self.real_values_input_in1)
    self.assertEqual(plaintext.polynomial.dtype, jnp.uint32)
    self.assertEqual(
        plaintext.polynomial.shape,
        (1, 1, *self.degree_layout, len(self.q_towers)),
    )
    ciphertext = ctx.encrypt(plaintext)
    self.assertEqual(ciphertext.polynomial.dtype, jnp.uint32)
    self.assertEqual(
        ciphertext.polynomial.shape,
        (1, 2, *self.degree_layout, len(self.q_towers)),
    )
    decrypted = ctx.decrypt(ciphertext)
    self.assertEqual(
        decrypted.polynomial.shape,
        (1, 1, *self.degree_layout, len(self.q_towers)),
    )
    with self.assertRaisesRegex(TypeError, "must be a Polynomial"):
      ctx.encrypt(plaintext.to_array())
    with self.assertRaisesRegex(TypeError, "must be a Polynomial"):
      ctx.decrypt(ciphertext.to_array())
    with self.assertRaisesRegex(ValueError, 'must contain 2'):
      ctx.decrypt(plaintext)

    fast_ciphertext = ctx.encrypt_slots(
        self.real_values_input_in1, scale=self.scaling_factor
    )
    self.assertIsInstance(fast_ciphertext, Polynomial)
    self.assertEqual(fast_ciphertext.shape, ciphertext.shape)
    self.assertEqual(fast_ciphertext.polynomial.dtype, jnp.uint32)
    fast_decoded = ctx.decrypt_slots(
        fast_ciphertext, self.scaling_factor
    )
    np.testing.assert_array_almost_equal(
        fast_decoded, self.real_values_input_in1, decimal=3
    )

    batch_ciphertext = ctx.encrypt_slots_batch(
        [self.real_values_input_in1, self.real_values_input_in2],
        scale=self.scaling_factor,
    )
    self.assertEqual(
        batch_ciphertext.shape,
        (2, 2, *self.degree_layout, len(self.q_towers)),
    )
    self.assertEqual(batch_ciphertext.polynomial.dtype, jnp.uint32)
    self.assertIs(batch_ciphertext.ntt_ctx, fast_ciphertext.ntt_ctx)
    batch_decoded = ctx.decrypt_slots_batch(
        batch_ciphertext, self.scaling_factor
    )
    np.testing.assert_array_almost_equal(
        batch_decoded[0], self.real_values_input_in1, decimal=3
    )
    np.testing.assert_array_almost_equal(
        batch_decoded[1], self.real_values_input_in2, decimal=3
    )
    with self.assertRaisesRegex(ValueError, 'at least one'):
      ctx.encrypt_slots_batch([], self.scaling_factor)

    wrong_dtype = ciphertext._clone_with_payload(
        ciphertext.polynomial.astype(jnp.uint64)
    )
    with self.assertRaisesRegex(ValueError, 'payload dtype'):
      ctx.decrypt(wrong_dtype)

    wide_shapes = {
        "batch": 1,
        "num_elements": 1,
        "degree": self.degree,
        "num_moduli": len(self.q_towers),
        "precision": 64,
        "degree_layout": self.degree_layout,
    }
    wide_plaintext = Polynomial.from_array(
        plaintext.polynomial.astype(jnp.uint64),
        wide_shapes,
        {"moduli": self.q_towers, "ntt_ctx": plaintext.ntt_ctx},
    )
    with self.assertRaisesRegex(ValueError, 'canonical precision=32/uint32'):
      ctx.encrypt(wide_plaintext)

    wide_shapes["num_elements"] = 2
    wide_ciphertext = Polynomial.from_array(
        ciphertext.polynomial.astype(jnp.uint64),
        wide_shapes,
        {"moduli": self.q_towers, "ntt_ctx": ciphertext.ntt_ctx},
    )
    with self.assertRaisesRegex(ValueError, 'canonical precision=32/uint32'):
      ctx.decrypt(wide_ciphertext)

    batched_shapes = {
        "batch": 2,
        "num_elements": 2,
        "degree": self.degree,
        "num_moduli": len(self.q_towers),
        "precision": 32,
        "degree_layout": self.degree_layout,
    }
    batched = Polynomial.from_array(
        jnp.concatenate([ciphertext.polynomial, ciphertext.polynomial], axis=0),
        batched_shapes,
        {"moduli": self.q_towers},
    )
    with self.assertRaisesRegex(NotImplementedError, "exactly one"):
      ctx.decrypt(batched)
    with self.assertRaisesRegex(NotImplementedError, "exactly one"):
      ctx.decrypt(batched)
    batched_plaintext_shapes = dict(batched_shapes)
    batched_plaintext_shapes["num_elements"] = 1
    batched_plaintext = Polynomial.from_array(
        jnp.concatenate([plaintext.polynomial, plaintext.polynomial], axis=0),
        batched_plaintext_shapes,
        {"moduli": self.q_towers},
    )
    with self.assertRaisesRegex(NotImplementedError, "exactly one"):
      ctx.encrypt(batched_plaintext)

  def test_fast_encrypt_and_decrypt_are_bit_exact_without_external_cache(self):
    ctx = ckks_ctx.CKKSContext(self.params)
    plaintext = ctx.encode(self.real_values_input_in1)
    num_q = len(self.q_towers)
    psi_pairs = [
        util.root_of_unity(2 * self.degree, modulus)
        for modulus in self.q_towers
    ]
    v_coeffs = [(index * 31 + 7) & 1 for index in range(self.degree)]
    e0_coeffs = [(index * 13 - 5) % 11 - 5
                 for index in range(self.degree)]
    e1_coeffs = [(index * 17 + 3) % 11 - 5
                 for index in range(self.degree)]

    def to_eval_rns(coefficients):
      return [
          list(util.bit_reverse_array(util.ntt_negacyclic_bit_reverse(
              [coefficient % modulus for coefficient in coefficients],
              modulus,
              psi,
          )))
          for modulus, psi in zip(self.q_towers, psi_pairs)
      ]

    v_rns = to_eval_rns(v_coeffs)
    errors = [to_eval_rns(e0_coeffs), to_eval_rns(e1_coeffs)]
    expected = np.asarray(
        ckks_ctx._ckks_encrypt_list_reference(
            plaintext=np.asarray(
                plaintext.polynomial[0, 0].reshape(self.degree, num_q)
            ).tolist(),
            public_key=[
                [list(ctx.public_key[element][tower])
                 for tower in range(num_q)]
                for element in range(2)
            ],
            q_towers=self.q_towers,
            v=v_rns,
            e=errors,
        ),
        dtype=np.uint64,
    )
    actual = ckks_ctx._fast_encrypt_from_plaintext(
        plaintext, ctx, v=v_rns, e=errors
    )
    actual_array = np.asarray(
        actual.polynomial[0].reshape(2, self.degree, num_q),
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(actual_array, expected)

    expected_plaintext = np.asarray(
        ckks_ctx._ckks_decrypt_list_reference(
            ciphertext=expected.tolist(),
            private_key=[list(row) for row in ctx.secret_key[:num_q]],
            q_towers=self.q_towers,
        ),
        dtype=np.uint64,
    )
    actual_plaintext = ckks_ctx._fast_decrypt_to_rns_coeffs(actual, ctx)
    np.testing.assert_array_equal(actual_plaintext, expected_plaintext)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_rotate_decrypt(self):
    rotate_idx = 1
    ctx = ckks_ctx.CKKSContext(self.params)
    ctx.program_initialization(
        total_rotation_indices=[rotate_idx],
        dnum=self.dnum,
        r=self.r,
        c=self.c,
    )
    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    # Step 3: Rotate
    result_ct = ctx.he_rot[ctx.max_level, rotate_idx].rotate(encrypted_ct)
    self.assertEqual(result_ct._ckks_scale, float(self.scaling_factor))
    # Step 4: Decryption
    decrypted_ct = ctx.decrypt(result_ct)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_rotate_result, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_rescale_decrypt(self):
    params = self.params.copy()
    params.update({
        "output_scale": (self.scaling_factor/self.q_towers[-1]),
    })
    ctx = ckks_ctx.CKKSContext(params)
    ctx.program_initialization(
        total_rotation_indices=[],
        dnum=self.dnum,
        r=self.r,
        c=self.c,
    )

    # Step 1: Encoding
    encoded_ct = ctx.encode(self.real_values_input_in1)
    # Step 2: Encryption
    encrypted_ct = ctx.encrypt(encoded_ct)
    ct_out = ctx.he_rescale[
        ctx.max_level, ctx.max_level - 1
    ].rescale(encrypted_ct)
    self.assertAlmostEqual(
        ct_out._ckks_scale,
        self.scaling_factor / self.q_towers[-1],
    )
    decrypted_ct = ctx.decrypt(ct_out)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_ct)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_input_in1, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_encrypt_multiply_decrypt(self):
    """
    Test the encryption, multiplication, and decryption of the CKKS context.
    See hemul_test.py for the debugging version
    """
    # Paramters Setup
    self.ek = kg.gen_evaluation_key(self.params["secret_key"], q=self.q_towers, P=self.p_towers, noise_std=self.sigma, noise_scale=1, dnum=3)
    eval_key_a, eval_key_b = jnp.array(self.ek["a"], dtype=jnp.uint32).transpose(0,2,1), jnp.array(self.ek["b"], dtype=jnp.uint32).transpose(0,2,1)
    params = self.params.copy()
    params.update({
        "evaluation_key": [eval_key_a, eval_key_b],
        "output_scale": (self.scaling_factor/self.q_towers[-1])**2,
    })
    ctx = ckks_ctx.CKKSContext(params)
    ctx.program_initialization(
        total_rotation_indices=[],
        dnum=self.dnum,
        r=self.r,
        c=self.c,
    )
    # Step 1: Encoding
    encoded_ct1 = ctx.encode(self.real_values_input_in1)
    encoded_ct2 = ctx.encode(self.real_values_input_in2)
    # Step 2: Encryption
    encrypted_ct1 = ctx.encrypt(encoded_ct1)
    encrypted_ct2 = ctx.encrypt(encoded_ct2)
    output_level = ctx.max_level - 1
    multiply = ctx.he_mul[output_level]

    tensor_product = multiply.hemul_no_relin(
        encrypted_ct1, encrypted_ct2
    )
    self.assertEqual(tensor_product.num_elements, 3)
    self.assertEqual(tensor_product.num_moduli, len(self.q_towers))
    self.assertEqual(
        tensor_product._ckks_scale, float(self.scaling_factor)**2
    )

    relinearized = multiply.relinearize(tensor_product)
    self.assertEqual(relinearized.num_elements, 2)
    self.assertEqual(relinearized.num_moduli, len(self.q_towers))
    self.assertEqual(relinearized._ckks_scale, tensor_product._ckks_scale)

    explicitly_rescaled = ctx.he_rescale[
        ctx.max_level, output_level
    ].rescale(relinearized)
    self.assertEqual(explicitly_rescaled.num_elements, 2)
    self.assertEqual(
        explicitly_rescaled.num_moduli, len(self.q_towers) - 1
    )
    self.assertAlmostEqual(
        explicitly_rescaled._ckks_scale,
        float(self.scaling_factor)**2 / self.q_towers[-1],
    )
    explicit_decoded = ctx.decode(ctx.decrypt(explicitly_rescaled))
    np.testing.assert_array_almost_equal(
        explicit_decoded, self.real_values_multiply_result, decimal=3
    )

    encrypted_result = multiply.mul(
        encrypted_ct1, encrypted_ct2
    )
    decrypted_result = ctx.decrypt(encrypted_result)
    # Step 5: Decoding
    decoded_values = ctx.decode(decrypted_result, is_ntt=False)
    np.testing.assert_array_almost_equal(decoded_values, self.real_values_multiply_result, decimal=3)

  # @absltest.skip("test a single experiment")
  def test_ckks_context_composite_rescale(self):
    """
    Test that composite rescaling correctly calculates scale factors.

    This test verifies the mathematical correctness of composite scaling:
    - composite_degree=k groups k moduli into a single logical scale
    - composite_scale_factor = product of last k moduli
    - After rescale: effective_scale = original_scale / composite_scale_factor

    This follows the algorithm from ePrint 2023/1462:
    "High-precision RNS-CKKS on fixed but smaller word-size architectures"
    """
    # Test with different composite_degree values
    for composite_degree in [1, 2]:
        # Calculate expected composite scale
        expected_composite_scale = 1
        for i in range(composite_degree):
            expected_composite_scale *= self.q_towers[-(i + 1)]

        params = self.params.copy()
        params.update({
            "composite_degree": composite_degree,
            "output_scale": self.scaling_factor,
        })

        ctx = ckks_ctx.CKKSContext(params)

        # Verify composite scale factor
        self.assertEqual(
            ctx.composite_scale_factor,
            expected_composite_scale,
            f"Failed for composite_degree={composite_degree}"
        )

        # Calculate effective scale bits
        composite_bits = expected_composite_scale.bit_length()
        expected_bits_per_modulus = composite_bits / composite_degree

        # For 30-bit moduli, composite degree k should give ~k*30 bits
        self.assertGreater(
            expected_bits_per_modulus,
            15,  # Each modulus contributes at least 15 bits
            f"composite_degree={composite_degree} should use reasonable moduli"
        )


# ===========================================================================
# Bit-exact correctness gates for the vectorized fast encrypt/encode and
# decrypt/decode paths (formerly in `encrypt_fast_test.py` and
# `decrypt_fast_test.py`).
#
# Encrypt side — reference is `ckks_ctx._ckks_encrypt_list_reference` +
# `ckks_ctx._ckks_encode_fall_back` (pure-Python). We require:
#   * forward NTT (negacyclic, bit-reversed output) bit-equal to the reference
#     for random and edge inputs;
#   * `_fast_encrypt_from_plaintext` bit-equal to the list reference when
#     given the same `(v, e)`;
#   * full private fast-codec round-trip through context-equivalent kernels
#     recovers the input slot vector within CKKS noise tolerance.
#
# Decrypt side — reference is the private nested-list implementation
# `_ckks_decrypt_list_reference`. We require all RNS coefficients to be
# **bit-equal**; decoded slot values must be within CKKS noise tolerance.
#
# Both classes share one fully-initialized `CKKSContext`, built here.
#
# It used to be unpickled from a LoLA demo cache. That cache can no longer
# be produced -- `save_cache` raises -- so the file was permanently absent
# and both classes permanently skipped. The codec paths under test are
# properties of the ring, not of any model, so the ring is built directly:
# the same 7Q+3P degree-2048 shape these assertions were written against.
# ===========================================================================
_CODEC_DEGREE = 2048
_CODEC_NUM_SLOTS = _CODEC_DEGREE // 2
_CODEC_NUM_Q = 7
_CODEC_NUM_P = 3
_CODEC_DNUM = 3
_CODEC_R, _CODEC_C = 32, 64
# 30-bit q towers keep a two-limb scale under the 61-bit word; 31-bit p
# towers mirror the auxiliary pool the demos used.
_CODEC_Q_BITS, _CODEC_P_BITS = 30, 31
_CODEC_CONTEXT = None


def _codec_ring():
    """Deterministic NTT primes for the codec ring: (q_towers, p_towers)."""
    q_towers = list(
        he_params.gen_ntt_primes(_CODEC_DEGREE, _CODEC_Q_BITS, _CODEC_NUM_Q)
    )
    p_towers = list(
        he_params.gen_ntt_primes(
            _CODEC_DEGREE, _CODEC_P_BITS, _CODEC_NUM_P, avoid=q_towers
        )
    )
    return q_towers, p_towers


def _codec_scale(q_towers) -> float:
    """Two-limb scaling factor, as the demo rings use."""
    return float(q_towers[0]) * float(q_towers[1])


def _load_codec_context():
    """One initialized CKKSContext, built once and shared across both classes.

    Keys are generated here, so this is the only real cost (~0.1 s at degree
    2048) and it is paid once per process.
    """
    global _CODEC_CONTEXT
    if _CODEC_CONTEXT is None:
        q_towers, p_towers = _codec_ring()
        scale = _codec_scale(q_towers)
        key_pair = kg.gen_pke_pair(q_towers, p_towers, _CODEC_DEGREE)
        ctx = ckks_ctx.CKKSContext({
            "degree": _CODEC_DEGREE,
            "num_slots": _CODEC_NUM_SLOTS,
            "scaling_factor": scale,
            "output_scale": scale,
            "q_towers": q_towers,
            "p_towers": p_towers,
            "p": _CODEC_Q_BITS,
            "CKKS_M_FACTOR": 1,
            "max_bits_in_word": 61,
            "noise_scale_degree": 1,
            "public_key": key_pair["public_key"],
            "secret_key": key_pair["secret_key"],
        })
        # No rotations: these tests exercise the codec, not the evaluator.
        ctx.program_initialization(
            total_rotation_indices=[], dnum=_CODEC_DNUM,
            r=_CODEC_R, c=_CODEC_C,
        )
        _CODEC_CONTEXT = ctx
    return _CODEC_CONTEXT


def _build_ct(ct_np: np.ndarray, q_towers, deg) -> Polynomial:
    """Wrap a flat local test array in the canonical ciphertext layout."""
    nq = ct_np.shape[-1]
    r = math.isqrt(deg)
    while deg % r:
        r -= 1
    degree_layout = (r, deg // r)
    dc = Polynomial(
        {"batch": 1, "num_elements": 2, "degree": deg,
         "precision": 32, "num_moduli": nq,
         "degree_layout": degree_layout},
        {"moduli": q_towers[:nq]},
    )
    dc.polynomial = jnp.asarray(ct_np, jnp.uint32).reshape(
        1, 2, *degree_layout, nq
    )
    return dc


def _reference_decrypt(ct_np: np.ndarray, ctx) -> np.ndarray:
    """Run the private nested-list decrypt reference for bit-exact tests."""
    nq = ct_np.shape[-1]
    return np.asarray(
        ckks_ctx._ckks_decrypt_list_reference(
            ciphertext=np.asarray(ct_np, dtype=np.uint64).tolist(),
            private_key=[list(row) for row in ctx.secret_key[:nq]],
            q_towers=list(ctx.q_towers[:nq]),
        ),
        dtype=np.uint64,
    )


def _build_plaintext(rns: np.ndarray, q_towers, deg) -> Polynomial:
    r = math.isqrt(deg)
    while deg % r:
        r -= 1
    degree_layout = (r, deg // r)
    nq = rns.shape[-1]
    return Polynomial.from_array(
        jnp.asarray(rns, jnp.uint32).reshape(1, 1, *degree_layout, nq),
        {
            "batch": 1,
            "num_elements": 1,
            "degree": deg,
            "precision": 32,
            "num_moduli": nq,
            "degree_layout": degree_layout,
        },
        {"moduli": q_towers[:nq]},
    )


def _ref_ntt(coeffs_per_tower, q_towers, psi_pairs, N):
    ref = np.empty((N, len(q_towers)), dtype=np.uint64)
    for m, (qm, psi) in enumerate(zip(q_towers, psi_pairs)):
        nt = util.ntt_negacyclic_bit_reverse(coeffs_per_tower[m], int(qm), psi)
        rev = util.bit_reverse_array(nt)
        ref[:, m] = np.array(rev, dtype=np.uint64)
    return ref


class FastEncryptCorrectness(absltest.TestCase):
    """Bit-exact tests for the vectorized fast encrypt / encode path."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Build (and key) the shared ring once, loudly, before any test runs.
        _load_codec_context()

    def test_forward_ntt_random(self):
        ctx = _load_codec_context()
        N = ctx.degree
        M = len(ctx.q_towers)
        psi_pairs = [util.root_of_unity(2 * N, q) for q in ctx.q_towers]
        rng = np.random.default_rng(0xBEEF)
        coeffs = []
        for j in range(M):
            qj = int(ctx.q_towers[j])
            coeffs.append(
                [int(c) for c in rng.integers(0, qj, size=N, dtype=np.uint64)]
            )
        ref = _ref_ntt(coeffs, ctx.q_towers, psi_pairs, N)
        rns = np.zeros((N, M), dtype=np.uint64)
        for j in range(M):
            rns[:, j] = np.array(coeffs[j], dtype=np.uint64)
        cache = ckks_ctx._get_encrypt_cache(ctx, M)
        fast = ckks_ctx._vectorized_ntt(rns, cache)
        self.assertTrue(np.array_equal(ref, fast))

    def test_forward_ntt_edge(self):
        ctx = _load_codec_context()
        N = ctx.degree
        M = len(ctx.q_towers)
        psi_pairs = [util.root_of_unity(2 * N, q) for q in ctx.q_towers]
        # Edge: zeros, ones, q-1, alternating.
        coeffs = []
        for j in range(M):
            qj = int(ctx.q_towers[j])
            row = ([0, 1, qj - 1, qj // 2] * (N // 4 + 1))[:N]
            coeffs.append(row)
        ref = _ref_ntt(coeffs, ctx.q_towers, psi_pairs, N)
        rns = np.zeros((N, M), dtype=np.uint64)
        for j in range(M):
            rns[:, j] = np.array(coeffs[j], dtype=np.uint64)
        cache = ckks_ctx._get_encrypt_cache(ctx, M)
        fast = ckks_ctx._vectorized_ntt(rns, cache)
        self.assertTrue(np.array_equal(ref, fast))

    def test_encrypt_bit_equal_with_seeded_v_e(self):
        ctx = _load_codec_context()
        NUM_SLOTS = _CODEC_NUM_SLOTS
        N = ctx.degree
        M = len(ctx.q_towers)
        psi_pairs = [util.root_of_unity(2 * N, q) for q in ctx.q_towers]
        # Deterministic v ∈ {0, 1, ..} (small, mirrors sampling shape; the
        # bit-exact equivalence holds for arbitrary v including ternary).
        v_coeffs = [(i * 31 + 7) & 1 for i in range(N)]
        e0_coeffs = [(i * 13 - 5) % 11 - 5 for i in range(N)]
        e1_coeffs = [(i * 17 + 3) % 11 - 5 for i in range(N)]
        v_rns, e0_rns, e1_rns = [], [], []
        for j in range(M):
            qj = int(ctx.q_towers[j])
            psi = psi_pairs[j]
            v_rns.append(list(util.bit_reverse_array(
                util.ntt_negacyclic_bit_reverse(
                    [c % qj for c in v_coeffs], qj, psi))))
            e0_rns.append(list(util.bit_reverse_array(
                util.ntt_negacyclic_bit_reverse(
                    [c % qj for c in e0_coeffs], qj, psi))))
            e1_rns.append(list(util.bit_reverse_array(
                util.ntt_negacyclic_bit_reverse(
                    [c % qj for c in e1_coeffs], qj, psi))))
        slots = [complex(i * 0.1, 0.0) for i in range(NUM_SLOTS)]
        pt = ctx.encode(slots)
        # Use the private list implementation for a bit-exact baseline.
        leg = ckks_ctx._ckks_encrypt_list_reference(
            plaintext=pt.polynomial[0, 0].reshape(N, M).tolist(),
            public_key=[
                [list(ctx.public_key[k][j]) for j in range(M)]
                for k in range(2)
            ],
            q_towers=list(ctx.q_towers),
            v=v_rns, e=[e0_rns, e1_rns],
        )
        leg_arr = np.array(leg, dtype=np.uint64)               # (2, N, M)
        fast = ckks_ctx._fast_encrypt_from_plaintext(
            pt, ctx, v=v_rns, e=[e0_rns, e1_rns],
        )
        fast_arr = np.asarray(
            fast.polynomial[0].reshape(2, N, M), dtype=np.uint64
        )
        self.assertTrue(np.array_equal(leg_arr, fast_arr))

    def test_round_trip_random_slots(self):
        ctx = _load_codec_context()
        NUM_SLOTS, SF = _CODEC_NUM_SLOTS, ctx.output_scale
        rng = np.random.default_rng(0xDEAD)
        for trial in range(3):
            slots = np.zeros(NUM_SLOTS, dtype=complex)
            slots[: 200] = rng.standard_normal(200) * 5.0
            ciphertext = ckks_ctx._fast_encode_encrypt(
                [complex(v) for v in slots], ctx, scale=SF
            )
            recovered = ckks_ctx._fast_decrypt_decode(ciphertext, ctx, SF)[0]
            err = float(np.max(np.abs(recovered[:200] - slots[:200].real)))
            self.assertLess(err, 1e-3,
                            f"trial={trial} round-trip max-err={err:.2e}")


class FastDecryptCorrectness(absltest.TestCase):
    """Bit-exact tests for the vectorized fast decrypt / decode path."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Build (and key) the shared ring once, loudly, before any test runs.
        _load_codec_context()

    def test_random_ciphertexts_all_levels(self):
        ctx = _load_codec_context()
        deg = ctx.degree
        q_full = ctx.q_towers
        for nq in range(1, len(q_full) + 1):
            with self.subTest(nq=nq):
                rng = np.random.default_rng(1234 + nq)
                ct = np.zeros((2, deg, nq), dtype=np.uint64)
                for j in range(nq):
                    ct[0, :, j] = rng.integers(0, int(q_full[j]),
                                               size=deg, dtype=np.uint64)
                    ct[1, :, j] = rng.integers(0, int(q_full[j]),
                                               size=deg, dtype=np.uint64)
                leg = _reference_decrypt(ct, ctx)
                fast = ckks_ctx._fast_decrypt_to_rns_coeffs(
                    _build_ct(ct, q_full, deg), ctx
                )
                self.assertTrue(
                    np.array_equal(leg, fast),
                    f"RNS mismatch at nq={nq}: max|diff|="
                    f"{int(np.max(np.abs(leg.astype(np.int64) - fast.astype(np.int64))))}"
                )

    def test_edge_coefficients(self):
        ctx = _load_codec_context()
        deg = ctx.degree
        q_full = ctx.q_towers
        for nq in (1, min(4, len(ctx.secret_key)),
                   len(ctx.secret_key)):
            with self.subTest(nq=nq):
                ct = np.zeros((2, deg, nq), dtype=np.uint64)
                for j in range(nq):
                    qv = int(q_full[j])
                    ct[0, :, j] = np.array(
                        [0, qv - 1, qv >> 1, (qv >> 1) - 1, (qv >> 1) + 1] *
                        (deg // 5 + 1), dtype=np.uint64)[:deg]
                    ct[1, :, j] = np.array(
                        [qv - 1, 0, 1, qv - 2, qv >> 2] *
                        (deg // 5 + 1), dtype=np.uint64)[:deg]
                leg = _reference_decrypt(ct, ctx)
                fast = ckks_ctx._fast_decrypt_to_rns_coeffs(
                    _build_ct(ct, q_full, deg), ctx
                )
                self.assertTrue(np.array_equal(leg, fast))

    def test_evaluated_ciphertext(self):
        """The same checks on a real evaluated ciphertext, not random limbs.

        This used to run a LoLA Mapping and decrypt its output. A Mapping is
        minutes of build and gigabytes of constants, and what the assertions
        below actually need is a ciphertext that came out of an HE operation
        at a reduced level rather than a uniform random draw. A mod-switch
        gives exactly that for the price of one op.
        """
        ctx = _load_codec_context()
        scale = ctx.output_scale
        checked = 64

        slots = np.zeros(_CODEC_NUM_SLOTS, dtype=complex)
        slots[:checked] = np.linspace(-3.0, 3.0, checked)
        ct = ckks_ctx._fast_encode_encrypt(
            [complex(v) for v in slots], ctx, scale=scale
        )
        self.assertEqual(ct.num_moduli, _CODEC_NUM_Q)

        top = _CODEC_NUM_Q - 1
        evaluated = ctx.he_level_reduce[top, top - 2].level_reduce(ct)
        self.assertEqual(evaluated.num_moduli, _CODEC_NUM_Q - 2)

        nq = evaluated.num_moduli
        ct_np = np.asarray(
            evaluated.polynomial.reshape(2, _CODEC_DEGREE, nq), dtype=np.uint64
        )
        leg_rns = _reference_decrypt(ct_np, ctx)
        fast_rns = ckks_ctx._fast_decrypt_to_rns_coeffs(evaluated, ctx)
        self.assertTrue(np.array_equal(leg_rns, fast_rns))

        leg_plaintext = _build_plaintext(
            leg_rns, ctx.q_towers, _CODEC_DEGREE
        )
        leg_slots = np.asarray(
            ctx.decode(
                leg_plaintext, is_ntt=False, validate_approximation=False
            ), dtype=complex
        ).real
        fast_slots = ckks_ctx._fast_decrypt_decode(evaluated, ctx, scale)[0]
        self.assertLess(
            float(np.max(np.abs(leg_slots[:checked] - fast_slots[:checked]))),
            1e-6, "decoded slot values exceed CKKS tolerance",
        )
        # A mod-switch preserves the value, so the decode is checkable
        # against the plaintext that went in -- the cache-era test could not
        # do this, because it decrypted a whole network's output.
        self.assertLess(
            float(np.max(np.abs(fast_slots[:checked] - slots[:checked].real))),
            1e-6, "mod-switched round trip lost the encoded values",
        )

    def test_batched_repeated_calls(self):
        """Cache should stay correct across many decrypt calls at varying nq."""
        ctx = _load_codec_context()
        deg = ctx.degree
        q_full = ctx.q_towers
        rng = np.random.default_rng(0xAA)
        for trial in range(8):
            nq = 1 + (trial % 4) * 2          # 1, 3, 5, 7
            ct = np.zeros((2, deg, nq), dtype=np.uint64)
            for j in range(nq):
                ct[0, :, j] = rng.integers(0, int(q_full[j]), size=deg,
                                            dtype=np.uint64)
                ct[1, :, j] = rng.integers(0, int(q_full[j]), size=deg,
                                            dtype=np.uint64)
            leg = _reference_decrypt(ct, ctx)
            fast = ckks_ctx._fast_decrypt_to_rns_coeffs(
                _build_ct(ct, q_full, deg), ctx
            )
            self.assertTrue(np.array_equal(leg, fast),
                            f"trial={trial} nq={nq} batched-call mismatch")

    def test_zero_ciphertext(self):
        ctx = _load_codec_context()
        deg = ctx.degree
        for nq in (1, min(4, len(ctx.secret_key)),
                   len(ctx.secret_key)):
            ct = np.zeros((2, deg, nq), dtype=np.uint64)
            leg = _reference_decrypt(ct, ctx)
            fast = ckks_ctx._fast_decrypt_to_rns_coeffs(
                _build_ct(ct, ctx.q_towers, deg), ctx
            )
            self.assertTrue(np.array_equal(leg, fast))
            self.assertTrue(np.all(fast == 0))


if 'pytest' in globals():
  def _mark_parameterized_methods(cls, prefix, *markers):
    matched = False
    for method_name in dir(cls):
      if method_name.startswith(prefix):
        getattr(cls, method_name).pytestmark = list(markers)
        matched = True
    if not matched:
      raise RuntimeError(f'no parameterized test method starts with {prefix!r}')

  _mark_parameterized_methods(
      CKKSContextTest,
      'test_public_slot_normalization_rejects_nonzero_imaginary_values',
      pytest.mark.security,
      pytest.mark.contract,
  )
  _mark_parameterized_methods(
      CKKSContextTest,
      'test_fast_decode_approximation_guard_fails_closed',
      pytest.mark.security,
  )
  _mark_parameterized_methods(
      CKKSContextTest,
      'test_encrypt_decrypt_cache_rejects_stale_id_reuse',
      pytest.mark.security,
      pytest.mark.contract,
  )
  _mark_parameterized_methods(
      CKKSContextTest,
      'test_encode_does_not_require_public_key',
      pytest.mark.contract,
  )
  _mark_parameterized_methods(
      CKKSContextTest,
      'test_program_initialization_rejects_layout_disagreement',
      pytest.mark.contract,
  )
  _mark_parameterized_methods(
      CKKSContextTest,
      'test_hybrid_p_tower_validation_compares_exact_products',
      pytest.mark.security,
  )
  _mark_parameterized_methods(
      CKKSContextTest,
      'test_program_initialization_uses_exact_p_tower_validation',
      pytest.mark.security,
  )
  _mark_parameterized_methods(
      CKKSContextTest,
      'test_rotation_key_noise_parameters_propagate_and_fail_closed',
      pytest.mark.security,
  )
  _mark_parameterized_methods(
      CKKSContextTest,
      'test_context_boundaries_use_rank_five_and_reject_batches',
      pytest.mark.contract,
  )


def _cache(q_towers, composite_degree):
  cache = he_params.HEParameterCache.__new__(he_params.HEParameterCache)
  cache.q_towers = list(q_towers)
  cache.composite_degree = composite_degree
  cache.num_q = len(q_towers)
  cache.max_level = (cache.num_q - 1) // composite_degree
  return cache


# Captured as std::hexfloat from OpenFHE 1.5.1 commit 1306d14, not
# recomputed with the CROSS implementation. Each row is
# (num_q, direct scale, recursive scale, real-big scale).
_OPENFHE_Q = (
    2147352577,
    1068236801,
    1079443457,
    1080360961,
    1068433409,
    1068564481,
)
_OPENFHE_SCALES = {
    1: (
        (1, '0x1.fd88000800000p+29', '0x1.de799d1434486p+29', None),
        (
            2,
            '0x1.fd60000800000p+29',
            '0x1.edaeeeb83ff00p+29',
            '0x1.dc059d7d83aa3p+59',
        ),
        (
            3,
            '0x1.015c000400000p+30',
            '0x1.f81761429f539p+29',
            '0x1.f04e088c7922ap+59',
        ),
        (
            4,
            '0x1.0194000400000p+30',
            '0x1.fd980088a2ce1p+29',
            '0x1.fb32e62ffcd48p+59',
        ),
        (
            5,
            '0x1.fd78000800000p+29',
            '0x1.fd88000800000p+29',
            '0x1.fb130c2fec400p+59',
        ),
        (
            6,
            '0x1.fd88000800000p+29',
            '0x1.fd88000800000p+29',
            '0x1.fb130c2fec400p+59',
        ),
    ),
    2: (
        (
            2,
            '0x1.fb031fefec000p+59',
            '0x1.f05c77c247a93p+59',
            '0x1.e13338bb90acap+119',
        ),
        (
            4,
            '0x1.02f225380bc00p+60',
            '0x1.fb031fefec000p+59',
            '0x1.f612b0450a31cp+119',
        ),
        (
            6,
            '0x1.fb031fefec000p+59',
            '0x1.fb031fefec000p+59',
            '0x1.f612b0450a31cp+119',
        ),
    ),
}
_OPENFHE_MOD_REDUCE = (
    '0x1.fff8000400000p+30',
    '0x1.fd60000800000p+29',
    '0x1.015c000400000p+30',
    '0x1.0194000400000p+30',
    '0x1.fd78000800000p+29',
    '0x1.fd88000800000p+29',
)


class HEParameterScaleIndexTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cache = _cache([101, 103, 107, 109, 113], 2)

  def test_level_mapping_and_scales(self):
    self.assertEqual(
        [self.cache.num_q_at_level(level) for level in range(3)], [1, 3, 5]
    )
    self.assertEqual(self.cache.q_moduli_at_level(1), [101, 103, 107])
    self.assertEqual(
        [self.cache.level_for_num_q(num_q) for num_q in (1, 3, 5)],
        [0, 1, 2],
    )
    self.assertEqual(self.cache.scaling_factor_at_level(1), float(103 * 107))
    self.assertAlmostEqual(
        self.cache.scaling_factor_real_big(1),
        self.cache.scaling_factor_recursive(1) ** 2,
    )

  def test_level_and_limb_indices_fail_closed(self):
    level_helpers = (
        self.cache.num_q_at_level,
        self.cache.q_moduli_at_level,
        self.cache.scaling_factor_at_level,
        self.cache.scaling_factor_recursive,
        self.cache.scaling_factor_real_big,
    )
    for helper in level_helpers:
      with self.subTest(helper=helper.__name__, kind='type'):
        with self.assertRaisesRegex(TypeError, 'level must be an int'):
          helper(True)
      with self.subTest(helper=helper.__name__, kind='range'):
        with self.assertRaisesRegex(ValueError, 'level .* outside'):
          helper(-1)

    for num_q in (2, 4):
      with self.assertRaisesRegex(ValueError, 'is not a valid CD2 level'):
        self.cache.level_for_num_q(num_q)
    with self.assertRaisesRegex(ValueError, 'big scaling-factor index'):
      self.cache.scaling_factor_real_big(0)
    with self.assertRaisesRegex(ValueError, 'prime_index .* outside'):
      self.cache.mod_reduce_factor(-1)


class OpenFHEScaleGoldenVectorTest(absltest.TestCase):

  def test_cd1_and_cd2_scales_match_openfhe(self):
    for composite_degree, rows in _OPENFHE_SCALES.items():
      cache = _cache(_OPENFHE_Q, composite_degree)
      for level, (num_q, direct, recursive, real_big) in enumerate(rows):
        with self.subTest(composite_degree=composite_degree, level=level):
          self.assertEqual(cache.num_q_at_level(level), num_q)
          self.assertEqual(
              cache.scaling_factor_at_level(level), float.fromhex(direct)
          )
          expected_recursive = float.fromhex(recursive)
          self.assertEqual(
              cache.scaling_factor_recursive(level), expected_recursive
          )
          self.assertEqual(
              cache.recursive_scaling_factor_for(
                  _OPENFHE_Q, composite_degree, level
              ),
              expected_recursive,
          )
          if real_big is None:
            with self.assertRaisesRegex(ValueError, 'big scaling-factor index'):
              cache.scaling_factor_real_big(level)
          else:
            self.assertEqual(
                cache.scaling_factor_real_big(level),
                float.fromhex(real_big),
            )

  def test_mod_reduce_factors_match_openfhe(self):
    cache = _cache(_OPENFHE_Q, 2)
    self.assertEqual(
        [cache.mod_reduce_factor(i) for i in range(len(_OPENFHE_Q))],
        [float.fromhex(value) for value in _OPENFHE_MOD_REDUCE],
    )




if 'pytest' in globals():
  HEParameterScaleIndexTest.pytestmark = [pytest.mark.security]
  OpenFHEScaleGoldenVectorTest.pytestmark = [pytest.mark.security]


if __name__ == "__main__":
  absltest.main()

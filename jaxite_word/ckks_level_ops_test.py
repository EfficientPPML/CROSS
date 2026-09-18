"""Focused tests for level-indexed CKKSContext operations."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

import bootstrapping
import ckks_ctx
import key_gen as kg
from polynomial import Polynomial


jax.config.update('jax_enable_x64', True)


class CKKSLevelOpsTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.degree = 16
    cls.num_slots = 8
    cls.r, cls.c = 4, 4
    cls.dnum = 3
    cls.scaling_factor = 563019763943521
    cls.q_towers = [
        1073742881,
        1073742721,
        1073741441,
        1073741857,
        524353,
    ]
    cls.p_towers = [1073740609, 1073739937, 1073739649]
    key_pair = kg.gen_pke_pair(cls.q_towers, cls.p_towers, cls.degree)
    evaluation_key = kg.gen_evaluation_key(
        key_pair['secret_key'],
        q=cls.q_towers,
        P=cls.p_towers,
        noise_std=3.190000057220458984375,
        noise_scale=1,
        dnum=cls.dnum,
    )
    eval_a = jnp.asarray(evaluation_key['a'], jnp.uint32).transpose(0, 2, 1)
    eval_b = jnp.asarray(evaluation_key['b'], jnp.uint32).transpose(0, 2, 1)
    cls.params = {
        'degree': cls.degree,
        'num_slots': cls.num_slots,
        'scaling_factor': cls.scaling_factor,
        'output_scale': cls.scaling_factor,
        'q_towers': cls.q_towers,
        'p_towers': cls.p_towers,
        'p': 30,
        'CKKS_M_FACTOR': 1,
        'max_bits_in_word': 61,
        'noise_scale_degree': 1,
        'public_key': key_pair['public_key'],
        'secret_key': key_pair['secret_key'],
        'evaluation_key': [eval_a, eval_b],
        'degree_layout': (cls.r, cls.c),
    }
    cls.ctx = ckks_ctx.CKKSContext(cls.params)
    cls.ctx.program_initialization(
        total_rotation_indices=[],
        dnum=cls.dnum,
        r=cls.r,
        c=cls.c,
        batch=2,
    )
    composite_params = dict(cls.params)
    composite_params['composite_degree'] = 2
    cls.composite_ctx = ckks_ctx.CKKSContext(composite_params)
    cls.composite_ctx.program_initialization(
        total_rotation_indices=[],
        dnum=cls.dnum,
        r=cls.r,
        c=cls.c,
        batch=2,
    )

  def _polynomial(
      self,
      level,
      *,
      batch=2,
      num_elements=2,
      offset=0,
      dtype=jnp.uint32,
      ctx=None,
  ):
    ctx = self.ctx if ctx is None else ctx
    cache = ctx._param_cache
    moduli = cache.q_moduli_at_level(level)
    shape = (batch, num_elements, self.r, self.c, len(moduli))
    values = np.arange(np.prod(shape), dtype=np.uint64).reshape(shape) + offset
    payload = (values % np.asarray(moduli, dtype=np.uint64)).astype(np.uint32)
    result = Polynomial.from_array(
        jnp.asarray(payload, dtype=dtype),
        {
            'batch': batch,
            'num_elements': num_elements,
            'degree': self.degree,
            'num_moduli': len(moduli),
            'precision': 32,
            'degree_layout': (self.r, self.c),
        },
        {
            'moduli': moduli,
            'ntt_ctx': cache.get_sliced_ntt_q(level),
        },
    )
    return result

  def test_encode_at_level_uses_exact_level_layout_and_scale(self):
    slots = [complex(value / 8.0, 0) for value in range(self.num_slots)]
    for level in (0, 2, self.ctx.max_level):
      # Keep the test scale below even the level-zero modulus product so direct
      # encode/decode remains meaningful at every level.
      scale = float(1 << 20)
      plaintext = self.ctx.encode_at_level(slots, level, scale=scale)
      expected_moduli = self.ctx._param_cache.q_moduli_at_level(level)
      self.assertEqual(plaintext.batch, 1)
      self.assertEqual(plaintext.num_elements, 1)
      self.assertEqual(plaintext.degree_layout, (self.r, self.c))
      self.assertSequenceEqual(tuple(plaintext.moduli), tuple(expected_moduli))
      self.assertEqual(
          plaintext.shape,
          (1, 1, self.r, self.c, len(expected_moduli)),
      )
      self.assertIs(
          plaintext.ntt_ctx,
          self.ctx._param_cache.get_sliced_ntt_q(level),
      )
      self.assertEqual(plaintext._ckks_scale, float(scale))
      np.testing.assert_allclose(
          np.asarray(self.ctx.decode(plaintext, is_ntt=True)),
          np.asarray(slots).real,
          rtol=0,
          atol=1e-3,
      )

    default_scale = self.ctx.encode_at_level(slots, 1)
    self.assertEqual(default_scale._ckks_scale, float(self.scaling_factor))

  def test_encode_at_level_rejects_uninitialized_or_invalid_level(self):
    uninitialized = ckks_ctx.CKKSContext(self.params)
    with self.assertRaisesRegex(RuntimeError, 'program_initialization'):
      uninitialized.encode_at_level([0j] * self.num_slots, 0)
    with self.assertRaisesRegex(TypeError, 'level must be an int'):
      self.ctx.encode_at_level([0j] * self.num_slots, '0')
    with self.assertRaisesRegex(ValueError, 'out of range'):
      self.ctx.encode_at_level([0j] * self.num_slots, -1)
    with self.assertRaisesRegex(ValueError, 'out of range'):
      self.ctx.encode_at_level(
          [0j] * self.num_slots, self.ctx.max_level + 1
      )

  def test_add_plain_broadcasts_and_matches_private_jitted_array(self):
    level = 2
    ciphertext = self._polynomial(level, offset=7)
    plaintext = self.ctx.encode_at_level(
        [complex(value + 1, 0) for value in range(self.num_slots)], level
    )
    operation = self.ctx.he_add[level]

    interpreted = operation.add_plain(ciphertext, plaintext)
    raw = jax.jit(operation._add_plain_array)(
        ciphertext.polynomial, plaintext.polynomial
    )
    np.testing.assert_array_equal(interpreted.polynomial, raw)

    moduli = jnp.asarray(plaintext.moduli, dtype=jnp.uint64)
    expected_c0 = (
        ciphertext.polynomial[:, :1].astype(jnp.uint64)
        + plaintext.polynomial.astype(jnp.uint64)
    ) % moduli
    np.testing.assert_array_equal(
        interpreted.polynomial[:, :1], expected_c0.astype(jnp.uint32)
    )
    np.testing.assert_array_equal(
        interpreted.polynomial[:, 1:2], ciphertext.polynomial[:, 1:2]
    )

  def test_add_plain_validates_plaintext_level_batch_elements_and_raw_shape(self):
    level = 2
    operation = self.ctx.he_add[level]
    ciphertext = self._polynomial(level)
    valid = self.ctx.encode_at_level([1 + 0j] * self.num_slots, level)

    with self.assertRaisesRegex(ValueError, 'batch=1'):
      operation.add_plain(
          ciphertext,
          self._polynomial(level, batch=2, num_elements=1),
      )
    with self.assertRaisesRegex(ValueError, 'num_elements=1'):
      operation.add_plain(
          ciphertext, self._polynomial(level, batch=1, num_elements=2)
      )
    with self.assertRaisesRegex(ValueError, 'towers|moduli'):
      operation.add_plain(
          ciphertext,
          self.ctx.encode_at_level([1 + 0j] * self.num_slots, level - 1),
      )
    with self.assertRaisesRegex(ValueError, 'dtype uint32'):
      operation._add_plain_array(
          ciphertext.polynomial, valid.polynomial.astype(jnp.uint64)
      )
    with self.assertRaisesRegex(ValueError, 'rank-5'):
      operation._add_plain_array(
          ciphertext.polynomial, valid.polynomial[0]
      )

    wide = Polynomial.from_array(
        ciphertext.polynomial.astype(jnp.uint64),
        {
            'batch': ciphertext.batch,
            'num_elements': ciphertext.num_elements,
            'degree': ciphertext.degree,
            'num_moduli': ciphertext.num_moduli,
            'precision': 64,
            'degree_layout': ciphertext.degree_layout,
        },
        {'moduli': ciphertext.moduli, 'ntt_ctx': ciphertext.ntt_ctx},
    )
    with self.assertRaisesRegex(ValueError, 'canonical precision=32/uint32'):
      operation.add_plain(wide, valid)

    ciphertext._ckks_scale = 2.0
    valid._ckks_scale = 4.0
    with self.assertRaisesRegex(ValueError, 'does not match ciphertext scale'):
      operation.add_plain(ciphertext, valid)

  def test_add_and_sub_reject_incompatible_scale_and_noise_metadata(self):
    level = self.ctx.max_level
    for operation in (
        self.ctx.he_add[level].add,
        self.ctx.he_sub[level].sub,
    ):
      with self.subTest(operation=operation.__qualname__, metadata='scale'):
        left = self._polynomial(level, offset=1)
        right = self._polynomial(level, offset=2)
        left._ckks_scale = 8.0
        right._ckks_scale = 16.0
        with self.assertRaisesRegex(ValueError, 'incompatible scales'):
          operation(left, right)

      with self.subTest(operation=operation.__qualname__, metadata='nsd'):
        left = self._polynomial(level, offset=1)
        right = self._polynomial(level, offset=2)
        left._ckks_nsd = 1
        right._ckks_nsd = 2
        with self.assertRaisesRegex(ValueError, 'incompatible nsd'):
          operation(left, right)

  def test_bootstrap_rejects_noncanonical_ciphertext_wrappers(self):
    level = self.ctx.max_level
    valid = self._polynomial(level)
    engine = bootstrapping.Bootstrap(self.ctx)

    with self.assertRaisesRegex(ValueError, 'num_elements=2'):
      engine._ensure_canonical(
          self._polynomial(level, num_elements=1), level
      )

    wide = Polynomial.from_array(
        valid.polynomial.astype(jnp.uint64),
        {
            'batch': valid.batch,
            'num_elements': valid.num_elements,
            'degree': valid.degree,
            'num_moduli': valid.num_moduli,
            'precision': 64,
            'degree_layout': valid.degree_layout,
        },
        {'moduli': valid.moduli, 'ntt_ctx': valid.ntt_ctx},
    )
    with self.assertRaisesRegex(ValueError, 'precision=32/uint32'):
      engine._ensure_canonical(wide, level)

  def test_explicit_ptct_constants_are_stateless_and_match_raw_path(self):
    level = self.ctx.max_level
    ciphertext = self._polynomial(level, offset=13)
    constant_one = self.ctx.encode_at_level(
        [1 + 0j] * self.num_slots, level
    )
    constant_two = self.ctx.encode_at_level(
        [2 + 0j] * self.num_slots, level
    )
    operation = self.ctx.ptct_mul[level]

    first_one = operation.mul(ciphertext, constant_one)
    result_two = operation.mul(ciphertext, constant_two)
    second_one = operation.mul(ciphertext, constant_one)

    np.testing.assert_array_equal(first_one.polynomial, second_one.polynomial)
    self.assertFalse(
        np.array_equal(first_one.polynomial, result_two.polynomial)
    )
    np.testing.assert_array_equal(
        first_one.polynomial,
        operation._mul_array(
            ciphertext.polynomial, constant_one.polynomial
        ),
    )
    ciphertext._ckks_scale = 3.0
    ciphertext._ckks_nsd = 1
    scaled = operation.mul(ciphertext, constant_one)
    self.assertEqual(
        scaled._ckks_scale, 3.0 * constant_one._ckks_scale
    )
    self.assertEqual(scaled._ckks_nsd, 2)
    self.assertFalse(hasattr(operation, 'set_plaintext'))
    self.assertFalse(hasattr(operation, 'precompute_bat'))
    with self.assertRaises(TypeError):
      operation.mul(ciphertext)

  def test_explicit_ptct_constant_validation_and_bat_error(self):
    level = self.ctx.max_level
    ciphertext = self._polynomial(level)
    operation = self.ctx.ptct_mul[level]
    constant = self.ctx.encode_at_level([1 + 0j] * self.num_slots, level)

    with self.assertRaisesRegex(TypeError, 'requires Polynomial'):
      operation.mul(ciphertext, constant.polynomial)
    with self.assertRaisesRegex(ValueError, 'batch=1'):
      operation.mul(
          ciphertext,
          self._polynomial(level, batch=2, num_elements=1),
      )
    with self.assertRaisesRegex(ValueError, 'towers|moduli'):
      operation.mul(
          ciphertext,
          self.ctx.encode_at_level(
              [1 + 0j] * self.num_slots, level - 1
          ),
      )
    with self.assertRaisesRegex(TypeError, 'unexpected keyword'):
      operation.mul(ciphertext, constant, use_bat=True)

  def test_binary_mul_private_jitted_array_matches_public_path(self):
    output_level = self.ctx.max_level - 1
    input_level = output_level + 1
    left = self._polynomial(input_level, offset=17)
    right = self._polynomial(input_level, offset=101)
    operation = self.ctx.he_mul[output_level]

    interpreted = operation.mul(left, right)
    raw = jax.jit(operation._mul_array)(left.polynomial, right.polynomial)
    np.testing.assert_array_equal(interpreted.polynomial, raw)
    self.assertEqual(
        raw.shape[-1], self.ctx._param_cache.num_q_at_level(output_level)
    )

    with self.assertRaisesRegex(ValueError, 'expected batch=2'):
      operation._mul_array(left.polynomial, right.polynomial[:1])
    with self.assertRaisesRegex(ValueError, 'dtype uint32'):
      operation._mul_array(
          left.polynomial, right.polynomial.astype(jnp.uint64)
      )

  def test_split_mul_control_is_materialized_with_cd1_accessor(self):
    operation = self.ctx.he_mul[0]

    self.assertIn('_no_relin_operator', operation.__dict__)
    self.assertIsNot(operation._no_relin_operator, operation._hemul)

  def test_mul_paths_track_their_actual_scales(self):
    for ctx in (self.ctx, self.composite_ctx):
      with self.subTest(composite_degree=ctx.composite_degree):
        output_level = ctx.max_level - 1
        input_level = output_level + 1
        left = self._polynomial(input_level, offset=37, ctx=ctx)
        right = self._polynomial(input_level, offset=149, ctx=ctx)
        divisor = np.prod(
            left.moduli[-ctx.composite_degree:], dtype=object
        )
        left._ckks_scale = float(divisor * 2)
        right._ckks_scale = float(divisor * 3)
        operation = ctx.he_mul[output_level]

        unified = operation.mul(left, right)
        expected_unified = (
            (left._ckks_scale / divisor) * (right._ckks_scale / divisor)
            if ctx.composite_degree == 1
            else left._ckks_scale * right._ckks_scale / divisor
        )
        self.assertEqual(unified._ckks_scale, expected_unified)

        three_elements = operation.hemul_no_relin(left, right)
        self.assertEqual(
            three_elements._ckks_scale,
            left._ckks_scale * right._ckks_scale,
        )
        relinearized = operation.relinearize(three_elements)
        self.assertEqual(
            relinearized._ckks_scale, three_elements._ckks_scale
        )
        split = ctx.he_rescale[input_level, output_level].rescale(
            relinearized
        )
        self.assertEqual(
            split._ckks_scale,
            left._ckks_scale * right._ckks_scale / divisor,
        )

  def test_binary_mul_array_matches_public_composite_rescale_path(self):
    ctx = self.composite_ctx
    output_level = ctx.max_level - 1
    input_level = output_level + 1
    left = self._polynomial(input_level, offset=29, ctx=ctx)
    right = self._polynomial(input_level, offset=211, ctx=ctx)
    operation = ctx.he_mul[output_level]

    interpreted = operation.mul(left, right)
    raw = jax.jit(operation._mul_array)(left.polynomial, right.polynomial)
    np.testing.assert_array_equal(interpreted.polynomial, raw)
    self.assertEqual(
        raw.shape[-1], ctx._param_cache.num_q_at_level(output_level)
    )



# =============================================================================
# Level-reduction primitive behavior
# =============================================================================

def _context():
  q_towers = [1073742881, 1073742721, 1073741441, 1073741857, 524353]
  p_towers = [1073740609, 1073739937, 1073739649]
  keys = kg.gen_pke_pair(q_towers, p_towers, 16)
  context = ckks_ctx.CKKSContext({
      'degree': 16, 'num_slots': 8,
      'scaling_factor': 563019763943521, 'output_scale': 563019763943521,
      'q_towers': q_towers, 'p_towers': p_towers, 'p': 30,
      'CKKS_M_FACTOR': 1, 'max_bits_in_word': 61, 'noise_scale_degree': 1,
      'public_key': keys['public_key'], 'secret_key': keys['secret_key'],
      'degree_layout': (4, 4), 'composite_degree': 1,
  })
  context.program_initialization(
      [], dnum=3, r=4, c=4, degree_layout=(4, 4), batch=1, perf_test=True
  )
  return context, len(q_towers) - 1


_VALUES = [0.5, -1.25, 3.0, 0.0, 2.5, -0.75, 1.0, -2.0]


class LevelReduceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.ctx, self.max_level = _context()
    self.ciphertext = self.ctx.encrypt(self.ctx.encode(_VALUES))

  def test_it_drops_the_requested_modulus_limbs(self):
    reduced = self.ctx.he_level_reduce[
        self.max_level, self.max_level - 2
    ].level_reduce(self.ciphertext)
    self.assertEqual(
        reduced.num_moduli, self.ciphertext.num_moduli - 2
    )

  def test_it_preserves_the_tracked_scale(self):
    before = getattr(self.ciphertext, '_ckks_scale')
    reduced = self.ctx.he_level_reduce[
        self.max_level, self.max_level - 2
    ].level_reduce(self.ciphertext)
    self.assertEqual(getattr(reduced, '_ckks_scale'), before)

  def test_rescale_divides_the_scale_where_level_reduce_does_not(self):
    """The distinction the two primitives exist for."""
    before = getattr(self.ciphertext, '_ckks_scale')
    rescaled = self.ctx.he_rescale[
        self.max_level, self.max_level - 2
    ].rescale(self.ciphertext)
    self.assertNotEqual(getattr(rescaled, '_ckks_scale'), before)
    reduced = self.ctx.he_level_reduce[
        self.max_level, self.max_level - 2
    ].level_reduce(self.ciphertext)
    self.assertEqual(getattr(reduced, '_ckks_scale'), before)

  def test_it_preserves_the_noise_scale_degree(self):
    self.ciphertext._ckks_nsd = 2
    reduced = self.ctx.he_level_reduce[
        self.max_level, self.max_level - 1
    ].level_reduce(self.ciphertext)
    self.assertEqual(getattr(reduced, '_ckks_nsd'), 2)

  def test_the_decrypted_value_is_unchanged(self):
    reduced = self.ctx.he_level_reduce[
        self.max_level, self.max_level - 2
    ].level_reduce(self.ciphertext)
    recovered = np.asarray(
        self.ctx.decode(self.ctx.decrypt(reduced))
    )[:len(_VALUES)].real
    np.testing.assert_allclose(recovered, _VALUES, atol=1e-6)

  def test_the_raw_array_hook_slices_the_modulus_axis(self):
    operator = self.ctx.he_level_reduce[self.max_level, self.max_level - 1]
    reduced = operator._level_reduce_array(self.ciphertext.polynomial)
    self.assertEqual(
        reduced.shape[-1], self.ciphertext.polynomial.shape[-1] - 1
    )
    self.assertEqual(
        reduced.shape[:-1], self.ciphertext.polynomial.shape[:-1]
    )

  def test_raising_the_level_is_refused(self):
    with self.assertRaisesRegex(ValueError, 'must be greater than'):
      self.ctx.he_level_reduce[1, 3]

  def test_an_equal_level_is_refused(self):
    with self.assertRaisesRegex(ValueError, 'must be greater than'):
      self.ctx.he_level_reduce[2, 2]

  def test_underflowing_the_chain_is_refused(self):
    with self.assertRaisesRegex(ValueError, 'out of range'):
      self.ctx.he_level_reduce[self.max_level, -1]


if __name__ == '__main__':
  absltest.main()

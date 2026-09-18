"""Functional tests for the BSGS preprocess-then-matvec lifecycle.

Like ``hemul_test``, these construct the private operation class, preprocess its
controls, and call the operation-named execution entry. Tests use a small CKKS
context and compare decoded output with cleartext matrix-vector multiplication.
"""

from typing import List
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import bsgs
import ckks_ctx
import key_gen as kg
from polynomial import Polynomial

_BSGSMatVecAtLevel = bsgs._BSGSMatVecAtLevel

try:
  import pytest
except ModuleNotFoundError:
  pytestmark = []
else:
  pytestmark = [pytest.mark.correctness, pytest.mark.unit]

jax.config.update('jax_enable_x64', True)


def _cleartext_matvec(a: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
  """Reference: zero-pad a and v to n x n and n, return a_padded @ v_padded."""
  m, k = a.shape
  a_padded = np.zeros((n, n), dtype=np.float64)
  a_padded[:m, :k] = a
  v_padded = np.zeros(n, dtype=np.float64)
  v_padded[: len(v)] = v
  return a_padded @ v_padded


class BSGSMatVecTest(parameterized.TestCase):
  """BSGS correctness at small CKKS parameters."""

  def setUp(self):
    super().setUp()
    self.degree = 16
    self.num_slots = 8
    self.dnum = 3
    self.r, self.c = 4, 4

    self.scaling_factor = 563019763943521
    self.q_towers = [
        1073742881,
        1073742721,
        1073741441,
        1073741857,
        524353,
    ]
    self.p_towers = [1073740609, 1073739937, 1073739649]

    self.params_base = {
        'degree': self.degree,
        'num_slots': self.num_slots,
        'scaling_factor': self.scaling_factor,
        'output_scale': self.scaling_factor,
        'q_towers': self.q_towers,
        'p_towers': self.p_towers,
        'p': 30,
        'CKKS_M_FACTOR': 1,
        'max_bits_in_word': 61,
        'noise_scale_degree': 1,
    }

  def _build_context(self, rotation_indices: List[int]):
    key_pair = kg.gen_pke_pair(self.q_towers, self.p_towers, self.degree)
    params = dict(self.params_base)
    params['public_key'] = key_pair['public_key']
    params['secret_key'] = key_pair['secret_key']
    ctx = ckks_ctx.CKKSContext(params)
    ctx.program_initialization(
        total_rotation_indices=rotation_indices,
        dnum=self.dnum,
        r=self.r,
        c=self.c,
    )
    return ctx

  def _encrypt_vec(self, ctx, vec: np.ndarray, num_q: int) -> Polynomial:
    """Encode + encrypt a vector; reshape to (1, 2, r, c, num_q)."""
    pt = ctx.encode([complex(float(x)) for x in vec])
    ct = ctx.encrypt(pt)
    p = Polynomial(
        {
            'batch': 1,
            'num_elements': 2,
            'degree': self.degree,
            'num_moduli': num_q,
            'precision': 32,
            'degree_layout': (self.r, self.c),
        },
        {'moduli': self.q_towers[:num_q]},
    )
    p.polynomial = ct.polynomial.reshape(1, 2, self.r, self.c, num_q)
    return p

  def _decrypt_result(self, ctx, ct: Polynomial, scale: int) -> np.ndarray:
    """Decrypt + decode, returning real parts of the slot vector."""
    old_scale = ctx.output_scale
    ctx.output_scale = scale
    try:
      decoded = ctx.decode(
          ctx.decrypt(ct), is_ntt=False, validate_approximation=False
      )
    finally:
      ctx.output_scale = old_scale
    return np.array([complex(v).real for v in decoded])

  def _run_bsgs_with_op(self, a_matrix: np.ndarray, v_vec: np.ndarray, n: int):
    """Run a BSGS matvec, returning (decoded[:n], context-bound op)."""
    n1, n2 = bsgs.compute_bsgs_params(n)
    indices = bsgs.required_rotation_indices(n, n1, n2)
    ctx = self._build_context(indices)

    num_q_in = len(self.q_towers)
    ct_v = self._encrypt_vec(ctx, v_vec, num_q_in)

    # Construct, preprocess the matrix once, then execute the named operation.
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    op.preprocess(a_matrix)
    result_ct = op.matvec(ct_v)
    # After 1-level rescale the scale is scaling_factor (since ptct product is
    # at scaling_factor^2, then divided by last-Q during rescale).
    decoded = self._decrypt_result(ctx, result_ct, scale=self.scaling_factor)
    return decoded[:n], op

  def _run_bsgs(self, a_matrix: np.ndarray, v_vec: np.ndarray, n: int):
    decoded, _op = self._run_bsgs_with_op(a_matrix, v_vec, n)
    return decoded

  def test_identity_matvec(self):
    n = 8
    a = np.eye(n, dtype=np.float64)
    v = np.array([0.5, 1.0, -0.5, 2.0, 0.25, -1.5, 3.0, 0.1])
    out = self._run_bsgs(a, v, n)
    np.testing.assert_allclose(out, v, atol=5e-2)

  def test_preprocess_rejects_complex_diagonals(self):
    n = 8
    n1, n2 = bsgs.compute_bsgs_params(n)
    ctx = self._build_context(bsgs.required_rotation_indices(n, n1, n2))
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    with self.assertRaisesRegex(TypeError, 'values must be real'):
      op.preprocess({0: np.ones(n, dtype=np.complex128)})

  def test_private_matvec_operands_support_outer_fused_jit(self):
    n = 8
    n1, n2 = bsgs.compute_bsgs_params(n)
    indices = bsgs.required_rotation_indices(n, n1, n2)
    ctx = self._build_context(indices)
    values = np.linspace(-0.4, 0.3, n)
    ciphertext = self._encrypt_vec(ctx, values, len(self.q_towers))
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    op.preprocess(np.eye(n))
    self.assertEmpty(ctx.he_rot._instances)

    expected = op.matvec(ciphertext).polynomial
    operands = op._matvec_operands()

    @jax.jit
    def fused(ct_data, dynamic_operands):
      return op._matvec_array(ct_data, *dynamic_operands)

    actual = fused(ciphertext.polynomial, operands)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

  def test_streaming_matches_standard_and_synchronizes_each_giant_step(self):
    n = 8
    n1, n2 = bsgs.compute_bsgs_params(n)
    ctx = self._build_context(bsgs.required_rotation_indices(n, n1, n2))
    rng = np.random.default_rng(23)
    values = rng.standard_normal(n) * 0.2
    ciphertext = self._encrypt_vec(ctx, values, len(self.q_towers))
    diag_map = {
        index: rng.standard_normal(n) * 0.1 for index in range(n)
    }
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)

    op.preprocess(diag_map)
    expected = np.asarray(op.matvec(ciphertext).polynomial)

    op.preprocess(diag_map, n_jobs=2, memory_bounded=True)
    expected_syncs = n2
    original_block_until_ready = jax.block_until_ready
    with mock.patch.object(
        bsgs.jax,
        'block_until_ready',
        wraps=original_block_until_ready,
    ) as synchronize:
      actual = op.matvec(ciphertext)

    np.testing.assert_array_equal(np.asarray(actual.polynomial), expected)
    self.assertEqual(synchronize.call_count, expected_syncs)

  def test_lazy_streaming_requests_one_giant_step_and_matches_eager(self):
    n = 8
    n1, n2 = bsgs.compute_bsgs_params(n)
    ctx = self._build_context(bsgs.required_rotation_indices(n, n1, n2))
    rng = np.random.default_rng(29)
    values = rng.standard_normal(n) * 0.2
    ciphertext = self._encrypt_vec(ctx, values, len(self.q_towers))
    diag_map = {
        index: rng.standard_normal(n) * 0.1 for index in range(n)
    }
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    op.preprocess(diag_map)
    expected = np.asarray(op.matvec(ciphertext).polynomial)

    class RecordingSource:

      dimension = n

      def __init__(self):
        self.requests = []

      def as_dict(self):
        raise AssertionError('lazy streaming requested the full diagonal map')

      def materialize_diagonals(self, indices):
        indices = tuple(indices)
        if len(indices) > n1:
          raise AssertionError('more than one giant step was materialized')
        self.requests.append(indices)
        return {index: diag_map[index] for index in indices}

    source = RecordingSource()
    op.preprocess(
        source,
        n_jobs=1,
        memory_bounded=True,
        active_diagonal_indices=tuple(range(n)),
    )
    actual = op.matvec(ciphertext)

    np.testing.assert_array_equal(np.asarray(actual.polynomial), expected)
    self.assertEqual(
        source.requests,
        [(0, 1), (2, 3), (4, 5), (6, 7)],
    )

  def test_lazy_streaming_rejects_complex_provider_values(self):
    n = 8
    n1, n2 = bsgs.compute_bsgs_params(n)
    ctx = self._build_context(bsgs.required_rotation_indices(n, n1, n2))
    ciphertext = self._encrypt_vec(
        ctx, np.linspace(-0.2, 0.2, n), len(self.q_towers)
    )
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)

    class ComplexSource:

      dimension = n

      @staticmethod
      def materialize_diagonals(indices):
        return {
            index: np.ones(n, dtype=np.complex128)
            for index in indices
        }

    op.preprocess(
        ComplexSource(),
        n_jobs=1,
        memory_bounded=True,
        active_diagonal_indices=(0,),
    )
    with self.assertRaisesRegex(TypeError, 'values must be real'):
      op.matvec(ciphertext)

  def test_streaming_rejects_non_polynomial_before_encoding(self):
    n = 8
    n1, n2 = bsgs.compute_bsgs_params(n)
    ctx = self._build_context(bsgs.required_rotation_indices(n, n1, n2))
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    op.preprocess({0: np.ones(n)}, memory_bounded=True)
    raw_ciphertext = jnp.zeros(
        (1, 2, self.r, self.c, len(self.q_towers)), dtype=jnp.uint32
    )

    with self.assertRaisesRegex(TypeError, 'requires Polynomial'):
      op.matvec(raw_ciphertext)

  def test_streaming_rejects_any_encode_underflow(self):
    n = 8
    n1, n2 = bsgs.compute_bsgs_params(n)
    ctx = self._build_context(bsgs.required_rotation_indices(n, n1, n2))
    values = np.linspace(-0.35, 0.35, n)
    ciphertext = self._encrypt_vec(ctx, values, len(self.q_towers))
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    op.preprocess({0: np.ones(n)}, memory_bounded=True)
    with mock.patch.object(
        op._encoder,
        'encode',
        side_effect=ckks_ctx.ScalingFactorTooSmall('too small'),
    ):
      with self.assertRaisesRegex(ValueError, 'underflowed CKKS encode'):
        op.matvec(ciphertext)

  def test_random_square(self):
    n = 8
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, n)) * 0.3
    v = rng.standard_normal(n) * 0.5
    expected = _cleartext_matvec(a, v, n)
    out = self._run_bsgs(a, v, n)
    np.testing.assert_allclose(out, expected, atol=5e-2)

  def test_rectangular_padded(self):
    n = 8
    rng = np.random.default_rng(1)
    # 3 outputs, 5 inputs, embedded in 8 slots.
    a = rng.standard_normal((3, 5)) * 0.3
    v = np.zeros(n)
    v[:5] = rng.standard_normal(5) * 0.5
    expected = _cleartext_matvec(a, v[:5], n)
    out = self._run_bsgs(a, v, n)
    # First 3 slots carry the matvec; the remaining slots should be ~0.
    np.testing.assert_allclose(out[:3], expected[:3], atol=5e-2)
    np.testing.assert_allclose(out[3:], 0.0, atol=5e-2)

  def test_partial_subresolution_matrix_is_rejected(self):
    """A non-zero diagonal may not disappear because another one encodes."""
    n = 8
    rng = np.random.default_rng(7)
    a = rng.standard_normal((n, n)) * 0.3
    dead_k = 3  # zero out diagonal 3 to ~1e-40 (below the encode floor)
    for i in range(n):
      a[i, (i + dead_k) % n] = 1e-40
    n1, n2 = bsgs.compute_bsgs_params(n)
    indices = bsgs.required_rotation_indices(n, n1, n2)
    ctx = self._build_context(indices)
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    with self.assertRaisesRegex(
        ValueError, 'below the plaintext encode floor'
    ):
      op.preprocess(a)

  def test_wide_dynamic_range_keeps_o1_diagonal(self):
    """Regression: with a huge matrix peak (~2^31) the old relative tol
    (`peak * 1e-9` ~ 2.1) dropped perfectly-encodable O(1) diagonals. The
    absolute encode-floor criterion must KEEP them and compute correctly."""
    n = 8
    a = np.zeros((n, n))
    for i in range(n):
      a[i, (i + 1) % n] = 0.5          # O(1) diagonal (k=1), max|entry|=0.5
    a[0, 0] = 2.0 ** 31                # huge peak on the main diagonal (k=0)
    v = np.ones(n)
    v[0] = 0.0                         # column 0 is zero -> huge entry adds 0
    expected = _cleartext_matvec(a, v, n)

    # Sanity: 0.5 < peak * 1e-9, i.e. the OLD relative rule would have dropped
    # the O(1) diagonal — this is exactly the regression.
    self.assertLess(0.5, float(np.max(np.abs(a))) * 1e-9)

    out, _ = self._run_bsgs_with_op(a, v, n)
    np.testing.assert_allclose(out, expected, atol=5e-2)

  def test_all_subresolution_raises(self):
    """A matrix whose every non-zero diagonal is sub-resolution must raise
    (a scale misconfiguration must not silently return an all-zero result)."""
    n = 8
    a = np.full((n, n), 1e-40)  # every entry far below the encode floor
    n1, n2 = bsgs.compute_bsgs_params(n)
    indices = bsgs.required_rotation_indices(n, n1, n2)
    ctx = self._build_context(indices)
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    with self.assertRaises(ValueError):
      op.preprocess(a)

  def test_preprocess_sparse_diagonals_matches_dense(self):
    """Dense and sparse preprocessing produce the same matvec."""
    n = 8
    rng = np.random.default_rng(11)
    a = rng.standard_normal((5, 6)) * 0.4   # rectangular, padded to n x n
    v = rng.standard_normal(n) * 0.5

    n1, n2 = bsgs.compute_bsgs_params(n)
    indices = bsgs.required_rotation_indices(n, n1, n2)
    ctx = self._build_context(indices)
    ct_v = self._encrypt_vec(ctx, v, len(self.q_towers))

    # Dense path.
    op_dense = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    op_dense.preprocess(a)

    # Build the sparse input with the same convention as dense preprocessing.
    padded = np.zeros((n, n), dtype=a.dtype)
    padded[:a.shape[0], :a.shape[1]] = a
    diagonals = [
        np.asarray([padded[i, (i + k) % n] for i in range(n)])
        for k in range(n)
    ]
    diag_map = {k: diagonals[k] for k in range(n) if np.any(diagonals[k])}
    op_sparse = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    op_sparse.preprocess(diag_map)

    # The two preprocessing inputs produce the same ciphertext and result.
    res_dense = op_dense.matvec(ct_v)
    res_sparse = op_sparse.matvec(ct_v)
    np.testing.assert_array_equal(
        np.asarray(res_dense.polynomial), np.asarray(res_sparse.polynomial))
    dec_dense = self._decrypt_result(ctx, res_dense, scale=self.scaling_factor)
    dec_sparse = self._decrypt_result(
        ctx, res_sparse, scale=self.scaling_factor)
    np.testing.assert_array_equal(dec_dense[:n], dec_sparse[:n])
    # And both track the cleartext matvec within the CKKS noise floor.
    expected = _cleartext_matvec(a, v, n)
    np.testing.assert_allclose(dec_dense[:n], expected, atol=5e-2)

  def test_setup_sparse_diagonals_validates_inputs(self):
    """Bad keys / shapes are rejected before reaching the encoder."""
    n = 8
    n1, n2 = bsgs.compute_bsgs_params(n)
    indices = bsgs.required_rotation_indices(n, n1, n2)
    ctx = self._build_context(indices)
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, n1, n2)
    with self.assertRaises(ValueError):        # key out of [0, n)
      op.preprocess({n: np.ones(n)})
    with self.assertRaises(ValueError):        # wrong length
      op.preprocess({0: np.ones(n + 1)})
    with self.assertRaises(ValueError):        # not a 1-D ndarray
      op.preprocess({0: [1.0] * n})
    with self.assertRaisesRegex(TypeError, 'values must be real'):
      op.preprocess({
          0: np.ones(n, dtype=np.complex128)
      })

  def test_mapping_schedule_controls_factors_and_exact_keys(self):
    n = 8
    pt_scale = self.q_towers[-1]
    diag_map = {1: np.ones(n), 6: np.ones(n)}
    # Mapping selected (4, 2) and initialized only the derived exact keys.
    ctx = self._build_context([1, 2, 4])
    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, 4, 2)
    op.preprocess(
        diag_map,
        pt_scale=pt_scale,
        active_diagonal_indices=tuple(diag_map),
    )
    v = np.arange(n, dtype=np.float64) / 10
    result = op.matvec(self._encrypt_vec(ctx, v, len(self.q_towers)))
    decoded = self._decrypt_result(ctx, result, scale=self.scaling_factor)
    expected = np.roll(v, -1) + np.roll(v, -6)
    np.testing.assert_allclose(decoded[:n], expected, atol=5e-2)

  def test_mapping_schedule_rejects_active_diagonal_drift(self):
    n = 8
    pt_scale = self.q_towers[-1]
    planned_map = {0: np.ones(n)}
    ctx = self._build_context([1])

    op = _BSGSMatVecAtLevel(ctx, ctx.max_level, n, 1, 8)
    with self.assertRaisesRegex(ValueError, 'active diagonals do not match'):
      op.preprocess(
          {1: np.ones(n)},
          pt_scale=pt_scale,
          active_diagonal_indices=tuple(planned_map),
      )

  def test_required_rotation_indices(self):
    # Utility check: baby 1..n1-1, giant n1, 2n1, ..., (n2-1)n1
    indices = bsgs.required_rotation_indices(16, 4, 4)
    self.assertEqual(indices, sorted({1, 2, 3, 4, 8, 12}))

  def test_low_level_operator_is_not_public(self):
    self.assertFalse(hasattr(bsgs, 'BSGSMatVec'))
    self.assertNotIn('_BSGSMatVecAtLevel', bsgs.__all__)

  def test_evaluator_exposes_only_preprocess_and_matvec(self):
    public_methods = {
        name
        for name, value in vars(_BSGSMatVecAtLevel).items()
        if callable(value) and not name.startswith('_')
    }
    self.assertEqual(public_methods, {'preprocess', 'matvec'})

  def test_bsgs_params_factoring(self):
    # Non-power-of-two should still factor to n1*n2=n
    n1, n2 = bsgs.compute_bsgs_params(15)
    self.assertEqual(n1 * n2, 15)
    n1, n2 = bsgs.compute_bsgs_params(1024)
    self.assertEqual(n1 * n2, 1024)


if __name__ == '__main__':
  absltest.main()

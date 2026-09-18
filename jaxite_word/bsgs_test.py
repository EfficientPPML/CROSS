"""Functional correctness tests for bsgs.BSGSMatVec.

Runs on a small-degree CKKS context (degree=16, 8 slots) so tests complete in
seconds. Compares BSGS output (decoded) against the cleartext matvec reference.
"""

from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import bsgs
import ckks_ctx
import key_gen as kg
from polynomial import Polynomial

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
        total_hemul_levels=ctx.max_level,
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
    import ckks_ctx as cc
    dc = Polynomial(
        {
            'batch': 1,
            'num_elements': 2,
            'degree': self.degree,
            'precision': 32,
            'num_moduli': ct.num_moduli,
            'degree_layout': (self.degree,),
        },
        {'moduli': self.q_towers[: ct.num_moduli]},
    )
    dc.polynomial = ct.polynomial.reshape(1, 2, self.degree, ct.num_moduli)
    old_scale = ctx.output_scale
    old_bypass = getattr(cc, 'BYPASS_DECODE_STDDEV_CHECK', False)
    ctx.output_scale = scale
    cc.BYPASS_DECODE_STDDEV_CHECK = True
    try:
      decoded = ctx.decode(ctx.decrypt(dc), is_ntt=False)
    finally:
      ctx.output_scale = old_scale
      cc.BYPASS_DECODE_STDDEV_CHECK = old_bypass
    return np.array([complex(v).real for v in decoded])

  def _run_bsgs(self, a_matrix: np.ndarray, v_vec: np.ndarray, n: int):
    n1, n2 = bsgs.compute_bsgs_params(n)
    indices = bsgs.BSGSMatVec.required_rotation_indices(n, n1, n2)
    ctx = self._build_context(indices)

    num_q_in = len(self.q_towers)
    ct_v = self._encrypt_vec(ctx, v_vec, num_q_in)

    op = bsgs.BSGSMatVec(ctx, level=ctx.max_level, n=n, n1=n1, n2=n2)
    op.encode_matrix(a_matrix)
    result_ct = op.mul(ct_v)
    # After 1-level rescale the scale is scaling_factor (since ptct product is
    # at scaling_factor^2, then divided by last-Q during rescale).
    decoded = self._decrypt_result(ctx, result_ct, scale=self.scaling_factor)
    return decoded[:n]

  def test_identity_matvec(self):
    n = 8
    a = np.eye(n, dtype=np.float64)
    v = np.array([0.5, 1.0, -0.5, 2.0, 0.25, -1.5, 3.0, 0.1])
    out = self._run_bsgs(a, v, n)
    np.testing.assert_allclose(out, v, atol=5e-2)

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

  def test_required_rotation_indices(self):
    # Utility check: baby 1..n1-1, giant n1, 2n1, ..., (n2-1)n1
    indices = bsgs.BSGSMatVec.required_rotation_indices(16, 4, 4)
    self.assertEqual(indices, sorted({1, 2, 3, 4, 8, 12}))

  def test_bsgs_params_factoring(self):
    # Non-power-of-two should still factor to n1*n2=n
    n1, n2 = bsgs.compute_bsgs_params(15)
    self.assertEqual(n1 * n2, 15)
    n1, n2 = bsgs.compute_bsgs_params(1024)
    self.assertEqual(n1 * n2, 1024)


if __name__ == '__main__':
  absltest.main()

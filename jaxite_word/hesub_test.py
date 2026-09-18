"""Private subtract-kernel tests over Barrett and Montgomery backends."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import finite_field as ff
import hesub
from polynomial import Polynomial

jax.config.update("jax_enable_x64", True)

REDUCTION_CONTEXTS = [
    ("barrett", ff.BarrettContext),
    ("montgomery", ff.MontgomeryContext),
]


class HESubKernelTest(parameterized.TestCase):
  """RNS ground-truth, cross-backend, and immutability kernel checks."""

  def setUp(self):
    super().setUp()
    self.degree = 16
    self.batch = 1
    self.num_elements = 2
    # NTT-friendly primes < 2^31 (Montgomery-valid), reused from ptct_mul_test.
    self.q_towers = [1073742881, 1073742721, 1073741441, 1073741857, 524353]

  def _draw(self, rng):
    cols = [
        rng.randint(0, q, size=(self.batch, self.num_elements, self.degree))
        for q in self.q_towers
    ]
    return np.stack(cols, axis=-1).astype(np.uint64).reshape(
        self.batch, self.num_elements, self.degree, 1, len(self.q_towers)
    )

  def _build_ct(self, data_std, ff_ctx_cls):
    ctx = ff_ctx_cls(self.q_towers)
    data = ctx.to_computation_format(jnp.asarray(data_std, jnp.uint64))
    ct = Polynomial(
        {"batch": self.batch, "num_elements": self.num_elements,
         "degree": self.degree, "num_moduli": len(self.q_towers),
         "precision": 32, "degree_layout": (self.degree, 1)},
        {"moduli": self.q_towers, "finite_field_context": ff_ctx_cls})
    ct.polynomial = jnp.asarray(data, jnp.uint32)
    return ct

  def _run(self, a, b, ff_ctx_cls):
    op = hesub._HESubKernel(self.q_towers, finite_field_context=ff_ctx_cls)
    out = op.sub(self._build_ct(a, ff_ctx_cls), self._build_ct(b, ff_ctx_cls))
    ctx = ff_ctx_cls(self.q_towers)
    return np.asarray(
        ctx.to_original_format(jnp.asarray(out.polynomial, jnp.uint64)),
        dtype=np.uint64)

  @parameterized.named_parameters(*REDUCTION_CONTEXTS)
  def test_rns_ground_truth(self, ff_ctx_cls):
    rng = np.random.RandomState(0)
    a, b = self._draw(rng), self._draw(rng)
    q = np.array(self.q_towers, dtype=np.int64)
    # Signed difference then %q gives the canonical positive residue.
    expected = ((a.astype(np.int64) - b.astype(np.int64)) % q).astype(np.uint64)
    np.testing.assert_array_equal(self._run(a, b, ff_ctx_cls), expected)

  def test_montgomery_matches_barrett(self):
    rng = np.random.RandomState(1)
    a, b = self._draw(rng), self._draw(rng)
    np.testing.assert_array_equal(
        self._run(a, b, ff.BarrettContext),
        self._run(a, b, ff.MontgomeryContext))

  @parameterized.named_parameters(*REDUCTION_CONTEXTS)
  def test_input_immutability(self, ff_ctx_cls):
    rng = np.random.RandomState(2)
    a, b = self._draw(rng), self._draw(rng)
    ct1, ct2 = self._build_ct(a, ff_ctx_cls), self._build_ct(b, ff_ctx_cls)
    p1, p2 = np.asarray(ct1.polynomial), np.asarray(ct2.polynomial)
    hesub._HESubKernel(
        self.q_towers, finite_field_context=ff_ctx_cls
    ).sub(ct1, ct2)
    np.testing.assert_array_equal(np.asarray(ct1.polynomial), p1)
    np.testing.assert_array_equal(np.asarray(ct2.polynomial), p2)

  def test_operand_level_mismatch_raises(self):
    rng = np.random.RandomState(3)
    ct1 = self._build_ct(self._draw(rng), ff.BarrettContext)
    short_q = self.q_towers[:-1]  # ct2 one tower short (different level)
    ct2 = Polynomial(
        {"batch": self.batch, "num_elements": self.num_elements,
         "degree": self.degree, "num_moduli": len(short_q),
         "precision": 32, "degree_layout": (self.degree, 1)},
        {"moduli": short_q, "finite_field_context": ff.BarrettContext})
    op = hesub._HESubKernel(
        self.q_towers, finite_field_context=ff.BarrettContext
    )
    with self.assertRaises(ValueError):
      op.sub(ct1, ct2)

  def test_reuses_finite_field_context_instance(self):
    ff_ctx = ff.BarrettContext(self.q_towers)
    op = hesub._HESubKernel(
        self.q_towers, finite_field_context=ff_ctx
    )
    self.assertIs(op.ff_ctx, ff_ctx)
    self.assertIs(op.ff_context_cls, ff.BarrettContext)
    with self.assertRaisesRegex(ValueError, 'moduli do not match'):
      hesub._HESubKernel(
          self.q_towers, finite_field_context=ff.BarrettContext(
              self.q_towers[:-1]
          )
      )


if __name__ == "__main__":
  absltest.main()

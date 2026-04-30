"""Performance benchmark for bsgs.BSGSMatVec.

Measures wall-time of a warmed-up `BSGSMatVec.mul` at a few representative
matrix sizes, reporting rotation count (baby + giant) and per-mul latency.
"""

import time
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import bsgs
import ckks_ctx
import key_gen as kg
from polynomial import Polynomial

jax.config.update('jax_enable_x64', True)


class BSGSMatVecPerfTest(absltest.TestCase):
  """Warmed-up latency per `mul` for various square sizes."""

  def setUp(self):
    super().setUp()
    # Small CKKS params: degree=16, 8 slots. For n>8 we pad v into higher
    # degrees — so for n>=32 bump degree/num_slots accordingly.
    self.dnum = 3
    self.scaling_factor = 563019763943521
    self.q_towers = [
        1073742881,
        1073742721,
        1073741441,
        1073741857,
        524353,
    ]
    self.p_towers = [1073740609, 1073739937, 1073739649]
    # At these small Q moduli only degree=16 is supported. We vary n within
    # the context's slot capacity.
    self.sizes: List[Tuple[int, int, int, int, int]] = [
        # (degree, num_slots, r, c, n)
        (16, 8, 4, 4, 4),
        (16, 8, 4, 4, 8),
    ]

  def _build_context(self, degree, num_slots, r, c, rotation_indices):
    params = {
        'degree': degree,
        'num_slots': num_slots,
        'scaling_factor': self.scaling_factor,
        'output_scale': self.scaling_factor,
        'q_towers': self.q_towers,
        'p_towers': self.p_towers,
        'p': 30,
        'CKKS_M_FACTOR': 1,
        'max_bits_in_word': 61,
        'noise_scale_degree': 1,
    }
    kp = kg.gen_pke_pair(self.q_towers, self.p_towers, degree)
    params['public_key'] = kp['public_key']
    params['secret_key'] = kp['secret_key']
    ctx = ckks_ctx.CKKSContext(params)
    ctx.program_initialization(
        total_hemul_levels=ctx.max_level,
        total_rotation_indices=rotation_indices,
        dnum=self.dnum,
        r=r,
        c=c,
    )
    return ctx

  def _bench_size(self, degree, num_slots, r, c, n):
    n1, n2 = bsgs.compute_bsgs_params(n)
    rot_indices = bsgs.BSGSMatVec.required_rotation_indices(n, n1, n2)
    ctx = self._build_context(degree, num_slots, r, c, rot_indices)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)) * 0.3
    v = rng.standard_normal(n) * 0.5

    # Pad v into num_slots before encoding.
    v_padded = np.zeros(num_slots)
    v_padded[:n] = v
    pt = ctx.encode([complex(float(x)) for x in v_padded])
    enc_ct = ctx.encrypt(pt)
    num_q = len(self.q_towers)
    ct = Polynomial(
        {
            'batch': 1,
            'num_elements': 2,
            'degree': degree,
            'num_moduli': num_q,
            'precision': 32,
            'degree_layout': (r, c),
        },
        {'moduli': self.q_towers},
    )
    ct.polynomial = enc_ct.polynomial.reshape(1, 2, r, c, num_q)

    op = bsgs.BSGSMatVec(ctx, level=ctx.max_level, n=n, n1=n1, n2=n2)
    op.encode_matrix(A)

    # Warm up (trigger JIT compile)
    out = op.mul(ct)
    jax.block_until_ready(out.polynomial)

    # Time
    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
      out = op.mul(ct)
      jax.block_until_ready(out.polynomial)
    dt_ms = (time.perf_counter() - t0) / iters * 1000.0

    rotations = (n1 - 1) + (n2 - 1)
    print(
        f'  n={n:4d}  n1={n1:3d}  n2={n2:3d}  '
        f'rotations={rotations:3d}  per-mul={dt_ms:7.2f} ms'
    )
    return dt_ms

  def test_bsgs_matvec_performance(self):
    print('\n=== BSGSMatVec per-mul latency ===')
    for degree, num_slots, r, c, n in self.sizes:
      print(f'[size] degree={degree} slots={num_slots} r={r} c={c} n={n}')
      self._bench_size(degree, num_slots, r, c, n)


if __name__ == '__main__':
  absltest.main()

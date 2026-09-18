"""Performance benchmark for the context-level BSGS matvec facade.

Two benchmarks:

  * ``test_bsgs_matvec_performance`` — toy degree=16 sanity sizes. At this size
    the kernel is dispatch/overhead-bound, so per-mul deltas are dominated by
    host overhead and ~10% run-to-run noise.

  * ``test_bsgs_matvec_performance_realistic`` — production ring dimension
    (default degree=4096). Builds a real CKKSContext, encodes a dense matrix
    via the actual ``ctx.bsgs_matvec`` path, and reports per-mul latency three ways:
      - wall-clock sync (block each iter)  -> host dispatch + device
      - wall-clock async (block once)      -> dispatch-pipelined
      - Xprof device self-time             -> true on-device kernel time
    plus an Xprof HLO-category breakdown (skipped if xprof is unavailable).

    Configurable via env vars:
      BSGS_PERF_DEGREE (default 4096), BSGS_PERF_NQ (q-limbs, default 4),
      BSGS_PERF_R (row layout factor, default 64),
      BSGS_PERF_SIZES (comma-separated matrix dims, default "64,256").
"""

import glob
import json
import os
import shutil
import tempfile
import time

try:
  import pytest
except ModuleNotFoundError:
  pytestmark = []
else:
  pytestmark = [
      pytest.mark.performance,
      pytest.mark.benchmark,
      pytest.mark.integration,
      pytest.mark.slow,
  ]
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

import bsgs
import ckks_ctx
import key_gen as kg
import util
from polynomial import Polynomial

jax.config.update('jax_enable_x64', True)

# Xprof is an optional dependency; the realistic benchmark falls back to
# wall-clock-only timing when it (or the trace tooling) is unavailable.
try:
  from xprof.convert import raw_to_tool_data as _xprof
  _HAS_XPROF = True
except Exception:  # pragma: no cover - environment dependent
  _xprof = None
  _HAS_XPROF = False


def _xprof_device_us(logdir: str, iters: int):
  """Sum on-device HLO self-time from an Xprof trace; return (us/mul, by-cat)."""
  pbs = glob.glob(os.path.join(logdir, '**', '*.xplane.pb'), recursive=True)
  if not pbs:
    return None, {}
  raw, _ = _xprof.xspace_to_tool_data(pbs, 'hlo_stats', {})
  d = json.loads(raw)
  labels = [col['label'] for col in d['cols']]
  self_i = labels.index('Total self time (us)')
  cat_i = labels.index('HLO op category')
  total = 0.0
  by_cat = {}
  for row in d['rows']:
    v = row['c'][self_i].get('v')
    try:
      v = float(v)
    except (TypeError, ValueError):
      v = 0.0
    total += v
    cat = row['c'][cat_i].get('v')
    by_cat[cat] = by_cat.get(cat, 0.0) + v
  return total / iters, {k: v / iters for k, v in by_cat.items()}


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
        total_rotation_indices=rotation_indices,
        dnum=self.dnum,
        r=r,
        c=c,
    )
    return ctx

  def _bench_size(self, degree, num_slots, r, c, n):
    n1, n2 = bsgs.compute_bsgs_params(n)
    rot_indices = bsgs.required_rotation_indices(n, n1, n2)
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

    op = ctx.bsgs_matvec[ctx.max_level, n, n1, n2]
    op.preprocess(A)

    # Warm up (trigger JIT compile)
    out = op.matvec(ct)
    jax.block_until_ready(out.polynomial)

    # Time
    iters = 5
    t0 = time.perf_counter()
    for _ in range(iters):
      out = op.matvec(ct)
      jax.block_until_ready(out.polynomial)
    dt_ms = (time.perf_counter() - t0) / iters * 1000.0

    rotations = (n1 - 1) + (n2 - 1)
    print(
        f'  n={n:4d}  n1={n1:3d}  n2={n2:3d}  '
        f'rotations={rotations:3d}  per-mul={dt_ms:7.2f} ms'
    )
    return dt_ms

  def test_bsgs_matvec_performance(self):
    print('\n=== BSGSMatVec per-mul latency (toy degree=16) ===')
    for degree, num_slots, r, c, n in self.sizes:
      print(f'[size] degree={degree} slots={num_slots} r={r} c={c} n={n}')
      self._bench_size(degree, num_slots, r, c, n)

  # ------------------------------------------------------------------
  # Realistic-degree benchmark (production ring dimension + Xprof)
  # ------------------------------------------------------------------
  def _build_realistic(self, degree, r, c, nq, n):
    """Build a real CKKSContext + context-bound BSGS op at production degree.

    Uses composite_degree=1 (bsgs assumes a single-modulus rescale) and
    NTT-friendly primes (= 1 mod 2*degree) from util.find_moduli_ntt, since
    composite_prime_gen is not always available.
    """
    cycl = 2 * degree
    q_towers = util.find_moduli_ntt(nq, 30, cycl)
    p_towers = util.generate_p_towers(q_towers, dnum=self.dnum, degree=degree)
    kp = kg.gen_pke_pair(q_towers, p_towers, degree)
    sf0 = float(q_towers[-1])
    params = {
        'degree': degree, 'num_slots': degree // 2, 'scaling_factor': sf0,
        'output_scale': sf0, 'q_towers': q_towers, 'p_towers': p_towers,
        'p': 30, 'CKKS_M_FACTOR': 1, 'max_bits_in_word': 61,
        'noise_scale_degree': 1,
        'public_key': kp['public_key'], 'secret_key': kp['secret_key'],
    }
    ctx = ckks_ctx.CKKSContext(params)
    n1, n2 = bsgs.compute_bsgs_params(n)
    rot = bsgs.required_rotation_indices(n, n1, n2)
    t = time.perf_counter()
    ctx.program_initialization(
        total_rotation_indices=rot,
        dnum=self.dnum, r=r, c=c)
    prog_s = time.perf_counter() - t
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n)) * 0.1
    op = ctx.bsgs_matvec[ctx.max_level, n, n1, n2]
    t = time.perf_counter()
    op.preprocess(A)
    enc_s = time.perf_counter() - t
    v = np.zeros(ctx.num_slots)
    v[:n] = rng.standard_normal(n) * 0.5
    pt = ctx.encode([complex(float(x)) for x in v])
    enc = ctx.encrypt(pt)
    ct_data = enc.polynomial.reshape(1, 2, r, c, nq)
    return op, ct_data, op._matvec_operands(), (n1, n2, prog_s, enc_s)

  def _bench_realistic_size(self, degree, r, c, nq, n,
                            wall_iters=20, trace_iters=50):
    op, ct_data, operands, (n1, n2, prog_s, enc_s) = self._build_realistic(
        degree, r, c, nq, n)
    # Benchmark the raw-array entry used by the fused scheduler.
    fn = jax.jit(op._matvec_array)
    for _ in range(3):
      jax.block_until_ready(fn(ct_data, *operands))

    t = time.perf_counter()
    for _ in range(wall_iters):
      jax.block_until_ready(fn(ct_data, *operands))
    sync_ms = (time.perf_counter() - t) / wall_iters * 1000

    t = time.perf_counter()
    rs = [fn(ct_data, *operands) for _ in range(wall_iters)]
    jax.block_until_ready(rs)
    async_ms = (time.perf_counter() - t) / wall_iters * 1000

    dev_ms, by_cat = None, {}
    if _HAS_XPROF:
      logdir = tempfile.mkdtemp(prefix='bsgs_xprof_')
      try:
        with jax.profiler.trace(logdir):
          for _ in range(trace_iters):
            rr = fn(ct_data, *operands)
          jax.block_until_ready(rr)
        dev_us, by_cat = _xprof_device_us(logdir, trace_iters)
        dev_ms = None if dev_us is None else dev_us / 1000.0
      finally:
        shutil.rmtree(logdir, ignore_errors=True)

    rotations = (n1 - 1) + (n2 - 1)
    dev_str = 'n/a' if dev_ms is None else f'{dev_ms:7.3f}'
    print(
        f'  n={n:4d}  n1={n1:3d}  n2={n2:3d}  rotations={rotations:3d}  '
        f'sync={sync_ms:7.3f} ms  async={async_ms:7.3f} ms  '
        f'xprof_dev={dev_str} ms   '
        f'(preprocess: prog_init={prog_s:.1f}s encode={enc_s:.1f}s)'
    )
    if by_cat:
      top = sorted(by_cat.items(), key=lambda kv: -kv[1])[:6]
      print('     device breakdown (us/mul): ' +
            ', '.join(f'{k}={v:.0f}' for k, v in top))
    return sync_ms, async_ms, dev_ms

  def test_bsgs_matvec_performance_realistic(self):
    degree = int(os.environ.get('BSGS_PERF_DEGREE', '4096'))
    nq = int(os.environ.get('BSGS_PERF_NQ', '4'))
    r = int(os.environ.get('BSGS_PERF_R', '64'))
    c = degree // r
    self.assertEqual(r * c, degree, 'BSGS_PERF_R must divide degree')
    sizes = [int(x) for x in os.environ.get('BSGS_PERF_SIZES', '64,256').split(',')]
    print(f'\n=== BSGSMatVec per-mul latency (realistic degree={degree}, '
          f'r={r} c={c}, {nq} q-limbs) ===')
    if not _HAS_XPROF:
      print('  [note] xprof unavailable -> wall-clock only (no on-device time)')
    for n in sizes:
      self._bench_realistic_size(degree, r, c, nq, n)


if __name__ == '__main__':
  absltest.main()

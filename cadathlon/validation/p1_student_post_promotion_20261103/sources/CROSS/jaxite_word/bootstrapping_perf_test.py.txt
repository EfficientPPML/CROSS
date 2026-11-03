#!/usr/bin/env python3
"""Performance test for bootstrapping.py across eight tensor cores.

The CKKS bootstrap wall-clock is dominated by the CoeffToSlot / SlotToCoeff
homomorphic linear transforms, whose inner loop is the plaintext x ciphertext
multiply (``ptct_mul``).  This test profiles that kernel with the ciphertext
batch sharded across the device mesh (8 tensor cores -> (2, 4) mesh via
util.create_sharding), at the tower counts a bootstrap actually traverses, and
reports per-iteration latency and aggregate throughput (ciphertexts/second).

It also runs the OpenFHE reference full-bootstrap binary for an end-to-end
wall-clock point of comparison.

Why kernel-level: the full bootstrap orchestration contains Python-side,
non-JIT-able steps (encode/FFT, per-level key selection), so it cannot be
compiled and sharded as a single XLA program.  ``ptct_mul`` is the compute-bound
kernel that dominates accelerator time, so its sharded throughput is the
meaningful "performance on N tensor cores" signal.

Run on 8 (simulated) CPU cores:
    XLA_FLAGS="--xla_force_host_platform_device_count=8" \
        python bootstrapping_perf_test.py
On a real 8-core TPU: python bootstrapping_perf_test.py
Sweep degrees: BOOTSTRAP_PERF_DEGREES="256,4096" python bootstrapping_perf_test.py
"""
from __future__ import annotations

import math
import os
import subprocess
import time

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

jax.config.update("jax_enable_x64", True)

import ckks_ctx
import key_gen as kg
import util
from composite_prime_gen import composite_prime_gen
from polynomial import Polynomial
from profiler import KernelWrapper, Profiler, collect_logs, kernel_perf_setup

# Bootstrap intermediates legitimately exceed the decode approximation limit.
# Scoped so the guard is restored for anything running later in this process.
_decode_guard_bypass = ckks_ctx.bypass_decode_stddev_check()


def setUpModule():
  _decode_guard_bypass.__enter__()


def tearDownModule():
  _decode_guard_bypass.__exit__(None, None, None)

_DEGREES = [int(d) for d in
            os.environ.get("BOOTSTRAP_PERF_DEGREES", "256").split(",") if d]
# Representative tower counts a bootstrap traverses (C2S high -> ApproxMod low).
_TOWER_COUNTS = [int(t) for t in
                 os.environ.get("BOOTSTRAP_PERF_TOWERS", "40,24,12").split(",")]

_OPENFHE_DIR = os.environ.get("OPENFHE_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "openfhe-development"))
# Degree- and parameter-matched OpenFHE reference.  The parameterized binary
# accepts log2(ringDim), transform budgets, retained levels, scalingMod,
# firstMod, and registerWordSize.  This benchmark uses the same cd=2,
# scalingMod=56, firstMod=61, levelBudget={1,1}, and maximum Q-tower count as
# the CROSS context below.
_OPENFHE_BIN = os.path.join(
    _OPENFHE_DIR, "build", "bin", "examples", "pke",
    "bootstrap-stage-compare-param")


def _ptct_mul_kernel(input_array, parameters):
  ct = parameters["ct_template"]
  ct.polynomial = input_array
  op = parameters["ptct_op"]
  return op._mul_encoded(ct, parameters["pt_ntt"]).polynomial


def _run_openfhe_reference(degree):
  """Wall-clock (ms) of OpenFHE EvalBootstrap at the SAME ring dimension, or nan
  if the matched binary is missing / errors (e.g. ringDim too small)."""
  if not os.path.exists(_OPENFHE_BIN):
    return float("nan")
  log_degree = int(math.log2(degree))
  if 1 << log_degree != degree:
    return float("nan")
  max_towers = max(_TOWER_COUNTS)
  # Uniform-ternary [1,1] bootstrap depth is 16.  With cd=2, a chain with NQ
  # raw towers has max logical level NQ/2-1, so this retains the remainder.
  levels_after = max_towers // 2 - 1 - 16
  if max_towers % 2 or levels_after < 0:
    return float("nan")
  args = [
      str(log_degree), "1", "1", str(levels_after), "56", "61", "31"
  ]
  try:
    t0 = time.perf_counter()
    proc = subprocess.run([_OPENFHE_BIN, *args], capture_output=True,
                          text=True, timeout=300)
    t1 = time.perf_counter()
  except Exception:
    return float("nan")
  return 1000.0 * (t1 - t0) if proc.returncode == 0 else float("nan")


class BootstrappingPerfTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.output_trace_root, self.profiler_config = kernel_perf_setup(
        __file__, iterations=5, enable_sharding=True
    )

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    collect_logs(os.path.dirname(os.path.abspath(__file__)),
                 output_csv_name="bootstrapping_profiling")

  def _build_ctx(self, degree, nq, batch=1):
    cd = 2
    r = 16 if degree <= 256 else 64
    c = degree // r
    cycl = 2 * degree
    q_towers = composite_prime_gen(cd, nq, 61, 56, cycl, 31)
    p_towers = util.generate_p_towers(q_towers, dnum=3, degree=degree)
    key_pair = kg.gen_pke_pair(q_towers, [], degree)
    sf0 = float(q_towers[-2]) * float(q_towers[-1])
    params = dict(
        degree=degree, num_slots=degree // 2, scaling_factor=sf0,
        output_scale=sf0, q_towers=q_towers, p_towers=p_towers,
        p=round(math.log2(sf0)), CKKS_M_FACTOR=1, max_bits_in_word=61,
        noise_scale_degree=1, composite_degree=cd,
        public_key=key_pair["public_key"], secret_key=key_pair["secret_key"])
    ctx = ckks_ctx.CKKSContext(params)
    # The context must be initialized for the same batch the kernels are fed.
    # ctx.ptct_mul[level] bakes in the batch and rejects anything else, so
    # hardcoding batch=1 here made the sharded test (batch = device count)
    # fail with "expected batch=1, got shape (8, 2, ...)".
    ctx.program_initialization(total_rotation_indices=[], dnum=3, r=r, c=c,
                               batch=batch)
    return ctx, r, c, q_towers

  @parameterized.named_parameters(*[(f"degree_{d}", d) for d in _DEGREES])
  def test_ptct_mul_sharded(self, degree):
    try:
      mesh, partition_spec = util.create_sharding()
    except RuntimeError as exc:
      self.skipTest(str(exc))
    cores = jax.device_count()
    batch = max(1, cores)
    axis_names = mesh.axis_names
    batch_part = axis_names if len(axis_names) > 1 else axis_names[0]
    none4 = (None,) * 4
    in_shd = jax.sharding.NamedSharding(mesh, partition_spec(batch_part, *none4))
    out_shd = jax.sharding.NamedSharding(mesh, partition_spec(batch_part, *none4))

    print(f"\n=== ptct_mul (linear-transform kernel), degree={degree}, "
          f"{cores} tensor cores, batch={batch} ===")

    nq_max = max(_TOWER_COUNTS)
    ctx, r, c, q_towers = self._build_ctx(degree, nq_max, batch=batch)
    bs = ctx.he_bootstrap.configure(level_budget=[1, 1])
    num_slots = ctx.num_slots

    profiler = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"bootstrapping_ptct_N{degree}",
        configuration=self.profiler_config)

    results = []
    for towers in _TOWER_COUNTS:
      if towers > nq_max:
        continue
      level = ctx._param_cache.max_level - (nq_max - towers) // 2
      num_q = ctx._param_cache.num_q_at_level(level)

      ct_template = Polynomial(
          {"batch": batch, "num_elements": 2, "degree": degree,
           "precision": 32, "num_moduli": num_q, "degree_layout": (r, c)},
          parameters={"moduli": q_towers[:num_q]})
      enc = jnp.asarray(ctx.encrypt(ctx.encode(
          [complex(0.3, 0)] * num_slots)).polynomial).reshape(-1)
      base = enc[:2 * r * c * num_q].reshape(1, 2, r, c, num_q)
      batched = jnp.broadcast_to(
          base, (batch, 2, r, c, num_q)).astype(jnp.uint32)

      pt_ntt = bs._engine._encode_diagonal(
          jnp.ones(num_slots, dtype=jnp.complex128),
          ctx._param_cache.q_moduli_at_level(level), num_q)

      w = KernelWrapper(
          kernel_name=f"ptct_N{degree}_q{num_q}",
          function_to_wrap=_ptct_mul_kernel,
          input_structs=[((batch, 2, r, c, num_q), jnp.uint32)],
          parameters={"ct_template": ct_template,
                      "ptct_op": ctx.ptct_mul[level], "pt_ntt": pt_ntt},
          mesh=mesh, input_shardings=(in_shd,), output_sharding=out_shd,
          enable_sharding=True)

      profiler.add_profile(
          name=f"ptct_N{degree}_q{num_q}", kernel_wrapper=w,
          kernel_setting_cols={"degree": degree, "num_moduli": num_q,
                               "batch": batch, "cores": cores})

      compiled = w.get_compiled_function()
      batched = jax.device_put(batched, in_shd)  # place on the (x,y) mesh
      for _ in range(2):
        compiled(batched).block_until_ready()
      n = 30
      t0 = time.perf_counter()
      for _ in range(n):
        compiled(batched).block_until_ready()
      ms = 1000.0 * (time.perf_counter() - t0) / n
      thr = 1000.0 * batch / ms
      results.append((num_q, ms, thr))
      print(f"  towers={num_q:3d}: {ms:8.3f} ms/iter   "
            f"{thr:9.1f} ct/s  ({batch} ct across {cores} cores)")

    profiler.run()

    openfhe_ms = _run_openfhe_reference(degree)
    print(f"\n  OpenFHE full EvalBootstrap at ringDim={degree} (1 core, reference): "
          f"{openfhe_ms:.1f} ms" if openfhe_ms == openfhe_ms
          else f"\n  OpenFHE reference N/A at ringDim={degree} "
               f"(binary missing or ringDim too small).")
    self.assertTrue(results, "no ptct_mul kernels profiled")


if __name__ == "__main__":
  absltest.main()


import os
import sys
import math
import csv
import json
import gzip
import warnings
import statistics
import concurrent.futures
from typing import Callable, List, Any, Dict, Optional, Tuple, Union
import functools
import re
import copy
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import jax.sharding as shd
import util
from absl.testing import absltest
from absl.testing import parameterized

# JAX configuration
jax.config.update("jax_enable_x64", True)
ENABLE_INITIAL_COPY_PROFILE = False

# ==========================================
# CHANGE ME! Evaluation Setup
# ==========================================
BATCH_SIZE_LIST_LOW_DEGREE = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
BATCH_SIZE_LIST_HIHG_DEGREE = [64, 128, 256, 512, 1024]
# Largest sharded NTT input shard, in uint32 elements per chip, that the sweep
# will profile. Measured on a TPU v6e-8 (32 GiB HBM per chip): every batch
# whose local shard is below 2**29 elements ran, and the two at or above it
# (degree 16384 with batch 16384 and 32768) failed with RESOURCE_EXHAUSTED
# because XLA's temporaries for the sharded NTT (32 GiB reported for the
# 2**29 shard) no longer fit beside the inputs. Batches at or above the cap
# are skipped with a printed reason rather than recorded as a silent profiler
# error; raise the cap on chips with more HBM.
_MAX_LOCAL_NTT_ELEMENTS = 1 << 29
TEST_PARAMS_NTT=[('2_12', 4096, 4, BATCH_SIZE_LIST_LOW_DEGREE), ('2_13', 8192, 8, BATCH_SIZE_LIST_LOW_DEGREE), ('2_14', 16384, 16, BATCH_SIZE_LIST_LOW_DEGREE), ('2_16_L48', 65536, 48, BATCH_SIZE_LIST_HIHG_DEGREE)]


# ==========================================
# UTIL (from util.py)
# ==========================================

def to_tuple(a):
  """Create to convert numpy array into tuple."""
  try:
    return tuple(to_tuple(i) for i in a)
  except TypeError:
    return a

def extended_gcd(a, b):
  """Return a tuple of (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
  if b == 0:
    return (a, 1, 0)
  else:
    g, x, y = extended_gcd(b, a % b)
    return (g, y, x - (a // b) * y)

def modinv(x: int, q: int) -> int:
  """Returns the inverse of x mod q."""
  return int(pow(x, -1, q))

def prime_factors(n):
  """Return the set of prime factors of n."""
  factors = set()
  # Divide out factors of 2
  while n % 2 == 0:
    factors.add(2)
    n //= 2
  # Check odd factors from 3 to sqrt(n)
  p = 3
  while p**2 <= n:
    while n % p == 0:
      factors.add(p)
      n //= p
    p += 2
  if n > 1:
    factors.add(n)
  return factors

def find_generator(q):
  """Find a primitive root modulo q."""
  phi = q - 1
  factors = prime_factors(phi)

  # Test candidates from 2 to q-1.
  for g in range(2, q):
    is_generator = all(pow(g, phi // p, q) != 1 for p in factors)
    if is_generator:
      return g
  raise ValueError("No generator found, check that q is prime.")

def gcd(a, b):
    return math.gcd(a, b)

def root_of_unity(m: int, q: int) -> int:
    """Canonical primitive m-th root of unity modulo q that **works with NTT**."""
    if m <= 0 or (q - 1) % m != 0:
      raise ValueError("q-1 must be divisible by positive m")
    # Step 1: multiplicative generator of Z_q^*
    g = find_generator(q)
    # Step 2: raise to (q-1)/m to get an m-th root candidate
    r = pow(g, (q - 1) // m, q)
    # Step 3: among r^k with gcd(k,m)=1, pick the minimal value whose order is exactly m
    candidates = []
    half = m // 2
    for k in range(1, m):
        if gcd(k, m) != 1:
            continue
        psi = pow(r, k, q)
        if pow(psi, half, q) == q - 1 and pow(psi, m, q) == 1:
            candidates.append(psi)
    if not candidates:
      raise ValueError("No primitive m-th root found")
    return min(candidates)

NTT_PARAMETERS_BY_DEGREE = {
  16: {
    "moduli": [1073759809, 1073759041, 1073759777, 1073758337, 1073759329, 1073758849, 1073759233, 1073738273, 1073754113, 1073738753, 1073753729, 1073738977, 1073753281, 1073739041, 1073753089, 1073747137, 1073752417, 1073739169, 1073745697, 1073739361, 1073752129, 1073746337, 1073748737, 1073746529, 1073748289, 1073747393, 1073749889, 1073748449, 1073751713, 1073749153, 1073750593, 1073749409, 1073751521, 1073750017, 1073751169, 1073750497, 1073751073, 1073750113, 1073750849, 1073739617, 1073746273, 1073745473, 1073745889, 1073742881, 1073745377, 1073739649, 1073745121, 1073741953, 1073744993, 1073739937, 1073744417, 1073742913, 1073744257, 1073742113, 1073743457, 1073742209, 1073743393, 1073740609, 1073742721, 1073741441, 1073741857, 524353],
    "root_of_unity": [149761193, 17168328, 145519847, 68042513, 3491826, 21109149, 48183983, 49547540, 15369996, 12935385, 1093151, 90892563, 108899655, 56634236, 235160291, 12265314, 191995239, 21404433, 40083131, 3916344, 113671079, 34500367, 61894143, 20463380, 13205216, 60050555, 145308815, 87067229, 10533116, 133048918, 13697511, 47895671, 14807533, 10994638, 25005605, 44429319, 77617905, 22756112, 21182116, 46947055, 41148497, 163086225, 60397627, 176334344, 30766686, 77429283, 67466901, 67653750, 4536048, 135444559, 63788661, 110966687, 9716122, 12174708, 49591386, 81862273, 51874541, 12155428, 60746932, 68809976, 28870916, 19017],
  },
  4096: {
    "moduli": [268730369, 268689409, 268361729, 268582913, 268369921, 268460033, 557057, 1152921504606830593, 1152921504606748673],
    "root_of_unity": [8801, 19068, 58939, 11033, 62736, 77090, 474, 116777451583545, 271802498405390],
  },
  8192: {
    "moduli": [269402113, 268091393, 268730369, 268271617, 269221889, 268664833, 268861441, 268369921, 268582913, 557057, 1152921504606830593, 1152921504606748673],
    "root_of_unity": [18987, 2826, 1678, 18925, 2446, 31335, 40892, 65274, 15787, 268, 25959043411404, 100406242475323],
  },
  16384: {
    "moduli": [274726913, 272760833, 274628609, 267059201, 270499841, 267550721, 270237697, 267943937, 268861441, 268042241, 268730369, 268238849, 269844481, 268271617, 269221889, 268369921, 268664833, 557057, 1152921504606748673, 1152921504606683137, 1152921504606584833],
    "root_of_unity": [9358, 15613, 1976, 5381, 15236, 9622, 5177, 2469, 792, 63914, 9742, 12308, 3704, 7216, 7564, 10360, 2023, 19, 62213374832584, 212089012217363, 92166579128688],
  },
  65536: {
    "moduli": [384040961, 376569857, 371458049, 375521281, 371589121, 383778817, 377880577, 379453441, 323092481, 351797249, 349962241, 351404033, 260702209, 308150273, 304742401, 307888129, 302776321, 306708481, 304218113, 347996161, 319291393, 347078657, 323223553, 337248257, 323878913, 336855041, 329515009, 332660737, 329777153, 335413249, 325844993, 330301441, 327548929, 332267521, 328728577, 344850433, 336068609, 340000769, 261488641, 302252033, 297664513, 299499521, 261881857, 295305217, 263323649, 277086209, 263454721, 292159489, 279838721, 291373057, 284950529, 290455553, 281935873, 285474817, 283508737, 288882689, 264634369, 276430849, 270532609, 274726913, 272760833, 276037633, 265420801, 270794753, 268042241, 269221889, 786433],
    "root_of_unity": [1197, 4622, 9335, 5748, 719, 1497, 2281, 3163, 3548, 80, 6577, 4942, 435, 3498, 316, 4503, 1433, 5766, 440, 2739, 1792, 13, 545, 7539, 7418, 7033, 32540, 1301, 4354, 16962, 10301, 289, 4195, 3322, 1005, 1747, 13384, 7659, 2200, 1035, 2142, 6961, 2774, 910, 43, 1949, 4343, 6648, 787, 2879, 4743, 563, 3385, 5648, 5875, 9494, 2122, 852, 6279, 1335, 712, 2017, 929, 142, 5274, 3264, 8],
  },
}

moduli_28_list = {
  degree: params["moduli"]
  for degree, params in NTT_PARAMETERS_BY_DEGREE.items()
}

# ==========================================
# FINITE FIELD (from finite_field.py)
# ==========================================

class FiniteFieldContextBase():
    def __init__(self, moduli: int):
        self.moduli = moduli

    def to_computation_format(self, a: int):
        return a

    def to_original_format(self, a: int):
        return a

    def get_jax_parameters(self):
        return {}

    def modular_reduction(self, a: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    def drop_last_modulus(self):
        raise NotImplementedError("Subclasses must implement this method")

class MontgomeryContext(FiniteFieldContextBase):
    def __init__(self, moduli: Union[List[int], int]):
        super().__init__(moduli)
        self.moduli = moduli
        if type(self.moduli) is int:
          self.moduli = [self.moduli]
        self.w = 32
        self.w_inv = [modinv(1 << self.w, m) for m in self.moduli]
        self.w_inv_reduction = jnp.array(self.w_inv, jnp.uint64)

        self.moduli_reduction = jnp.array(self.moduli, jnp.uint64)

        self.moduli_inv_32 = [modinv(m, 2**32) for m in self.moduli]
        self.moduli_low16 = [m & 0xFFFF for m in self.moduli]
        self.moduli_high16 = [m >> 16 for m in self.moduli]

        self.q = jnp.array(self.moduli, dtype=jnp.uint32)
        self.q_low = jnp.array(self.moduli_low16, dtype=jnp.uint32)
        self.q_high = jnp.array(self.moduli_high16, dtype=jnp.uint32)
        self.q_inv_32 = jnp.array(self.moduli_inv_32, dtype=jnp.uint32)

    def to_computation_format(self, a: int):
        return (a << self.w) % self.moduli_reduction

    def to_original_format(self, a: jnp.ndarray):
        return (a * self.w_inv_reduction) % self.moduli_reduction

    def get_jax_parameters(self):
        return {
            "moduli": to_tuple(self.moduli),
            "moduli_inv_32": to_tuple(self.moduli_inv_32),
            "moduli_low": to_tuple(self.moduli_low16),
            "moduli_high": to_tuple(self.moduli_high16)
        }

    def modular_reduction(self, z: jnp.ndarray) -> jnp.ndarray:
        #Local constants
        MASK32 = 0xFFFFFFFF
        MASK16 = 0xFFFF
        SHIFT16 = 16
        SHIFT32 = 32
        # Ensure dimensions for broadcasting
        q = self.q
        q_low = self.q_low
        q_high = self.q_high
        q_inv_32 = self.q_inv_32

        # Computation
        z_low = z.astype(jnp.uint32)
        z_high = (z >> SHIFT32).astype(jnp.uint32)
        t = (z_low * q_inv_32) & MASK32
        t_low = t & MASK16
        t_high = (t >> SHIFT16) & MASK16

        prod_high = t_high * q_high  # This contributes directly to upper 32 bits
        prod_mid_high = t_high * q_low  # Upper 16 bits go to upper 32 bits
        prod_mid_low = t_low * q_high   # Upper 16 bits go to upper 32 bits
        prod_low = t_low * q_low        # Upper 16 bits contribute to middle part
        mid_low = (prod_mid_high & MASK16) + (prod_mid_low & MASK16) + (prod_low >> SHIFT16)
        mid_high = (prod_mid_high >> SHIFT16) + (prod_mid_low >> SHIFT16) + (mid_low >> SHIFT16)

        # Final upper 32 bits
        t_final = prod_high + mid_high
        b = z_high + q - t_final
        return b.astype(jnp.uint32)

    def modular_reduction_single_modulus(self, z: jnp.ndarray, limb_index: int) -> jnp.ndarray:
        # Simplified for single modulus if needed, but reusing logic
        # Actually logic is broadcastable. If z is (..., M), and props are (M), it works.
        # But if z is (...,) and we need specifc limb params:
        MASK32 = 0xFFFFFFFF
        MASK16 = 0xFFFF
        SHIFT16 = 16
        SHIFT32 = 32

        q = self.q[limb_index]
        q_low = self.q_low[limb_index]
        q_high = self.q_high[limb_index]
        q_inv_32 = self.q_inv_32[limb_index]

        z_low = z.astype(jnp.uint32)
        z_high = (z >> SHIFT32).astype(jnp.uint32)
        t = (z_low * q_inv_32) & MASK32
        t_low = t & MASK16
        t_high = (t >> SHIFT16) & MASK16

        prod_high = t_high * q_high
        prod_mid_high = t_high * q_low
        prod_mid_low = t_low * q_high
        prod_low = t_low * q_low
        mid_low = (prod_mid_high & MASK16) + (prod_mid_low & MASK16) + (prod_low >> SHIFT16)
        mid_high = (prod_mid_high >> SHIFT16) + (prod_mid_low >> SHIFT16) + (mid_low >> SHIFT16)

        t_final = prod_high + mid_high
        b = z_high + q - t_final
        return b.astype(jnp.uint32)

    def drop_last_modulus(self):
        self.moduli_reduction = self.moduli_reduction[:-1]
        self.q = self.q[:-1]
        self.q_low = self.q_low[:-1]
        self.q_high = self.q_high[:-1]
        self.q_inv_32 = self.q_inv_32[:-1]

from profiler import KernelWrapper, Profiler, collect_module_logs
from profiler import kernel_perf_setup

# ==========================================
# NTT MM (from ntt_mm.py)
# ==========================================

def gen_twiddle_matrix(rows, cols, q, omega):
  r_idx = np.arange(rows, dtype=np.int64)[:, None]
  c_idx = np.arange(cols, dtype=np.int64)[None, :]
  exponents = r_idx * c_idx
  twiddle_matrix = np.zeros((rows, cols), dtype=int)
  def compute_row(r):
    for c in range(cols):
      twiddle_matrix[r, c] = pow(int(omega), int(exponents[r, c]), int(q))
  with concurrent.futures.ThreadPoolExecutor() as executor:
    list(executor.map(compute_row, range(rows)))
  return twiddle_matrix

def gen_twiddle_matrix_inv(rows, cols, q, omega):
  twiddle_matrix_inv = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    for c in range(cols):
      twiddle_matrix_inv[r, c] = pow(int(omega), int(-r * c), int(q))
  return twiddle_matrix_inv

class NTTCiphertextContextBase():
    def __init__(self, moduli: int, parameters: dict):
        self.ff_ctx = parameters.get("finite_field_context", None)
        self.num_bytes = 4
        self.moduli = moduli
        self.parameters = parameters
        self.r = parameters.get("r", 0)
        self.c = parameters.get("c", 0)
        if self.r <= 0 or self.c <= 0:
            raise ValueError(
                f'r and c must be positive, got r={self.r}, c={self.c}'
            )
        self.transform_length = self.r * self.c
        self.psi_list = [root_of_unity(2 * self.transform_length, q) for q in self.moduli]
        self.omega_list = [
            (psi ** 2) % q
            for psi, q in zip(self.psi_list, self.moduli, strict=True)
        ]

        self.ntt_tf_step1, self.ntt_tf_step2, self.ntt_tf_step3 = self.ntt_coefficients_precompute()
        self.ntt_bat_tf_step1 = self.basis_aligned_transformation(self.to_computation_format(self.ntt_tf_step1))
        self.ntt_tf_step2 = self.to_computation_format(self.ntt_tf_step2).astype(jnp.uint64)
        self.ntt_bat_tf_step3 = self.basis_aligned_transformation(self.to_computation_format(self.ntt_tf_step3))

    def ntt_coefficients_precompute(self):
        tf_step1_list, tf_step2_list, tf_step3_list = [], [], []
        for i, modulus in enumerate(self.moduli):
            omega_col = pow(self.omega_list[i], self.c, modulus)
            omega_row = pow(self.omega_list[i], self.r, modulus)
            tf_step1_one_modulus = gen_twiddle_matrix(self.r, self.r, modulus, omega_col)
            tf_step2_one_modulus = gen_twiddle_matrix(self.r, self.c, modulus, self.omega_list[i])
            tf_step3_one_modulus = gen_twiddle_matrix(self.c, self.c, modulus, omega_row)
            tf_step1_list.append(tf_step1_one_modulus)
            tf_step2_list.append(tf_step2_one_modulus)
            tf_step3_list.append(tf_step3_one_modulus)
        tf_step1 = jnp.array(tf_step1_list, dtype=jnp.uint32).transpose(1,2,0)
        tf_step2 = jnp.array(tf_step2_list, dtype=jnp.uint32).transpose(1,2,0)
        tf_step3 = jnp.array(tf_step3_list, dtype=jnp.uint32).transpose(1,2,0)
        return tf_step1, tf_step2, tf_step3

    def to_computation_format(self, a: np.ndarray):
        return self.ff_ctx.to_computation_format(a.astype(jnp.uint64)).astype(jnp.uint32)

    def basis_aligned_transformation(self, matrix: np.ndarray):
        return util.shifted_mod_bytes(
            matrix, self.moduli
        ).transpose(1, 0, 2, 4, 3)

    def ntt(self, v: jax.Array):
        result_step1 = util.matmul(v, self.ntt_bat_tf_step1, "brcmq,zqrpm->bzcmp")
        result_step1_reduced = self.ff_ctx.modular_reduction(result_step1)
        result_step2 = jnp.multiply(result_step1_reduced.astype(jnp.uint64), self.ntt_tf_step2)
        result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
        result_step3 = util.matmul(result_step2_reduced, self.ntt_bat_tf_step3, "brcmq,cqnpm->bnrmp")
        result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
        return result_step3_reduced

class NTTCiphertextMontgomeryContext(NTTCiphertextContextBase):
    def __init__(self, moduli: int, parameters: dict):
        super().__init__(moduli, parameters)
        if type(self.moduli) is int:
            self.moduli = [self.moduli]
        if self.ff_ctx is None:
          self.ff_ctx = MontgomeryContext(moduli)

# ==========================================
# TEST (from ntt_mm_perf_test.py)
# ==========================================

DEGREE_TO_RC_NTT = {
    65536: (128, 512), # Make some dimension 128 is always helpful!
    32768: (128, 256),
    16384: (128, 128),
    8192: (128, 64),
    4096: (128, 32),
    2048: (128, 16),
}

def _ntt_kernel(input_array, parameters):
  """Kernel wrapper entry point used by KernelWrapper."""
  return parameters["ctx"].ntt(input_array)

class NTTMMShardedPerformanceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.output_trace_root, self.profiler_config = kernel_perf_setup(
        __file__, enable_sharding=True
    )

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    collect_module_logs(__file__, "tabVII_ntt_profiling")


  def _create_sharded_kernel_wrapper(self, kernel_name, ctx, batch, rows, cols, num_moduli, mesh, batch_sharding):
    input_shape = (batch, rows, cols, num_moduli)
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_ntt_kernel,
        input_structs=[(input_shape, jnp.uint32)],
        parameters={"ctx": ctx},
        mesh=mesh,
        input_shardings=(batch_sharding,),
        output_sharding=batch_sharding,
        enable_sharding=True,
    )

  def _profile_context_sharded(self, profile_prefix, ctx_cls, ff_ctx_cls, degree, num_limbs, batch_size_list):
    rows, cols = DEGREE_TO_RC_NTT[degree]
    moduli = moduli_28_list[degree][:num_limbs]
    try:
      mesh, partition_spec = util.create_sharding()
      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
      batch_sharding = jax.sharding.NamedSharding(
          mesh,
          partition_spec(batch_partition, None, None, None),
      )
    except RuntimeError as exc:
      self.skipTest(str(exc))
      return

    profiler_config = self.profiler_config.copy()
    profiler_config["enable_sharding"] = True
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"sharding_{profile_prefix}_degree_{degree}",
        configuration=profiler_config,
    )

    for batch in batch_size_list:
      kernel_name = f"sharding_{profile_prefix}_batch_{batch}"
      local_elements = batch * rows * cols * len(moduli) // max(mesh.size, 1)
      if local_elements >= _MAX_LOCAL_NTT_ELEMENTS:
        print(
            f"[tabVII] skipping {kernel_name}: {local_elements:,} uint32 "
            f"elements per chip is at or above the {_MAX_LOCAL_NTT_ELEMENTS:,} "
            "cap (the sharded NTT's XLA temporaries do not fit 32 GiB of HBM)"
        )
        continue

      ctx_parameters = {
          "r": rows,
          "c": cols,
          "finite_field_context": ff_ctx_cls(moduli=moduli),
      }
      ctx = ctx_cls(moduli=moduli, parameters=ctx_parameters)

      kernel_wrapper = self._create_sharded_kernel_wrapper(
          kernel_name=kernel_name,
          ctx=ctx,
          batch=batch,
          rows=rows,
          cols=cols,
          num_moduli=len(moduli),
          mesh=mesh,
          batch_sharding=batch_sharding,
      )

      profiler_instance.add_profile(
          name=kernel_name,
          kernel_wrapper=kernel_wrapper,
          kernel_setting_cols={
              "degree": degree,
              "num_limbs": num_limbs,
              "batch": batch,
              "rows": rows,
              "cols": cols,
              "sharding": "batch",
          },
      )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()

  @parameterized.named_parameters(*TEST_PARAMS_NTT)
  def test_sharded_NTT_Montgomery_performance(self, degree, num_limbs, batch_size_list):
    self._profile_context_sharded(
        profile_prefix="ntt_montgomery",
        ctx_cls=NTTCiphertextMontgomeryContext,
        ff_ctx_cls=MontgomeryContext,
        degree=degree,
        num_limbs=num_limbs,
        batch_size_list=batch_size_list,
    )


if __name__ == "__main__":
  absltest.main()

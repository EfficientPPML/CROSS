"""A module for operations on test CKKS evaluation kernels including.

- NTT
"""
import contextlib
import csv
import functools
import io
import math
import re
import sys

import jax
import jax.numpy as jnp
import util
import bconv as bconv
import ntt_mm as ntt
import ntt_sm as ntt_single
import finite_field as ff_context
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
jax.config.update("jax_enable_x64", True)

TEST_PARAMS_VEC_MUL=[('VecMul_limb18', 65536, 18), ('VecMul_limb19', 65536, 19), ('VecMul_limb20', 65536, 20), ('VecMul_limb21', 65536, 21), ('VecMul_limb22', 65536, 22), ('VecMul_limb23', 65536, 23), ('VecMul_limb24', 65536, 24), ('VecMul_limb25', 65536, 25), ('VecMul_limb26', 65536, 26), ('VecMul_limb27', 65536, 27), ('VecMul_limb28', 65536, 28), ('VecMul_limb29', 65536, 29), ('VecMul_limb30', 65536, 30), ('VecMul_limb31', 65536, 31), ('VecMul_limb32', 65536, 32), ('VecMul_limb33', 65536, 33), ('VecMul_limb34', 65536, 34), ('VecMul_limb35', 65536, 35), ('VecMul_limb36', 65536, 36), ('VecMul_limb37', 65536, 37), ('VecMul_limb38', 65536, 38), ('VecMul_limb39', 65536, 39), ('VecMul_limb40', 65536, 40), ('VecMul_limb43', 65536, 43), ('VecMul_limb44', 65536, 44), ('VecMul_limb45', 65536, 45), ('VecMul_limb47', 65536, 47), ('VecMul_limb48', 65536, 48), ('VecMul_limb49', 65536, 49), ('VecMul_limb51', 65536, 51), ('VecMul_limb52', 65536, 52), ('VecMul_limb53', 65536, 53)]

TEST_PARAMS_VEC_ADD=[('VecAdd_limb18', 65536, 18), ('VecAdd_limb19', 65536, 19), ('VecAdd_limb20', 65536, 20), ('VecAdd_limb21', 65536, 21), ('VecAdd_limb22', 65536, 22), ('VecAdd_limb23', 65536, 23), ('VecAdd_limb24', 65536, 24), ('VecAdd_limb25', 65536, 25), ('VecAdd_limb26', 65536, 26), ('VecAdd_limb27', 65536, 27), ('VecAdd_limb28', 65536, 28), ('VecAdd_limb29', 65536, 29), ('VecAdd_limb30', 65536, 30), ('VecAdd_limb31', 65536, 31), ('VecAdd_limb32', 65536, 32), ('VecAdd_limb33', 65536, 33), ('VecAdd_limb34', 65536, 34), ('VecAdd_limb35', 65536, 35), ('VecAdd_limb36', 65536, 36), ('VecAdd_limb37', 65536, 37), ('VecAdd_limb38', 65536, 38), ('VecAdd_limb39', 65536, 39), ('VecAdd_limb40', 65536, 40), ('VecAdd_limb43', 65536, 43), ('VecAdd_limb44', 65536, 44), ('VecAdd_limb45', 65536, 45), ('VecAdd_limb47', 65536, 47), ('VecAdd_limb48', 65536, 48), ('VecAdd_limb49', 65536, 49), ('VecAdd_limb51', 65536, 51), ('VecAdd_limb52', 65536, 52), ('VecAdd_limb53', 65536, 53)]

TEST_PARAMS_BConv_OP=[('BConv_limb10to21', 65536, 10, 21), ('BConv_limb10to22', 65536, 10, 22), ('BConv_limb10to23', 65536, 10, 23), ('BConv_limb10to29', 65536, 10, 29), ('BConv_limb10to30', 65536, 10, 30), ('BConv_limb11to24', 65536, 11, 24), ('BConv_limb11to25', 65536, 11, 25), ('BConv_limb11to26', 65536, 11, 26), ('BConv_limb11to32', 65536, 11, 32), ('BConv_limb11to33', 65536, 11, 33), ('BConv_limb11to34', 65536, 11, 34), ('BConv_limb12to27', 65536, 12, 27), ('BConv_limb12to28', 65536, 12, 28), ('BConv_limb12to33', 65536, 12, 33), ('BConv_limb12to35', 65536, 12, 35), ('BConv_limb12to36', 65536, 12, 36), ('BConv_limb12to37', 65536, 12, 37), ('BConv_limb13to30', 65536, 13, 30), ('BConv_limb13to31', 65536, 13, 31), ('BConv_limb13to32', 65536, 13, 32), ('BConv_limb13to38', 65536, 13, 38), ('BConv_limb13to39', 65536, 13, 39), ('BConv_limb13to40', 65536, 13, 40), ('BConv_limb14to33', 65536, 14, 33), ('BConv_limb14to34', 65536, 14, 34), ('BConv_limb14to35', 65536, 14, 35), ('BConv_limb15to36', 65536, 15, 36), ('BConv_limb15to37', 65536, 15, 37), ('BConv_limb15to38', 65536, 15, 38), ('BConv_limb7to20', 65536, 7, 20), ('BConv_limb7to21', 65536, 7, 21), ('BConv_limb7to22', 65536, 7, 22), ('BConv_limb8to23', 65536, 8, 23), ('BConv_limb8to24', 65536, 8, 24), ('BConv_limb8to25', 65536, 8, 25), ('BConv_limb9to18', 65536, 9, 18), ('BConv_limb9to19', 65536, 9, 19), ('BConv_limb9to20', 65536, 9, 20), ('BConv_limb9to26', 65536, 9, 26), ('BConv_limb9to27', 65536, 9, 27), ('BConv_limb9to28', 65536, 9, 28)]

TEST_PARAMS_NTT_MULTI_MODULI=[('NTT_limb10', 65536, 10), ('NTT_limb11', 65536, 11), ('NTT_limb12', 65536, 12), ('NTT_limb13', 65536, 13), ('NTT_limb14', 65536, 14), ('NTT_limb15', 65536, 15), ('NTT_limb18', 65536, 18), ('NTT_limb19', 65536, 19), ('NTT_limb20', 65536, 20), ('NTT_limb21', 65536, 21), ('NTT_limb22', 65536, 22), ('NTT_limb23', 65536, 23), ('NTT_limb24', 65536, 24), ('NTT_limb25', 65536, 25), ('NTT_limb26', 65536, 26), ('NTT_limb27', 65536, 27), ('NTT_limb28', 65536, 28), ('NTT_limb29', 65536, 29), ('NTT_limb30', 65536, 30), ('NTT_limb31', 65536, 31), ('NTT_limb32', 65536, 32), ('NTT_limb33', 65536, 33), ('NTT_limb34', 65536, 34), ('NTT_limb35', 65536, 35), ('NTT_limb36', 65536, 36), ('NTT_limb37', 65536, 37), ('NTT_limb38', 65536, 38), ('NTT_limb39', 65536, 39), ('NTT_limb40', 65536, 40), ('NTT_limb7', 65536, 7), ('NTT_limb8', 65536, 8), ('NTT_limb9', 65536, 9)]

TEST_PARAMS_NTT_SINGLE_MODULI=[('NTT_limb1', 65536, 1)]

TEST_PARAMS_AUTOMORPHISM=TEST_PARAMS_AUTOMORPH=[('Automorph_limb19', 65536, 19), ('Automorph_limb20', 65536, 20), ('Automorph_limb21', 65536, 21), ('Automorph_limb22', 65536, 22), ('Automorph_limb23', 65536, 23), ('Automorph_limb24', 65536, 24), ('Automorph_limb27', 65536, 27), ('Automorph_limb28', 65536, 28), ('Automorph_limb29', 65536, 29), ('Automorph_limb31', 65536, 31), ('Automorph_limb32', 65536, 32), ('Automorph_limb33', 65536, 33), ('Automorph_limb34', 65536, 34), ('Automorph_limb35', 65536, 35), ('Automorph_limb36', 65536, 36), ('Automorph_limb37', 65536, 37), ('Automorph_limb38', 65536, 38), ('Automorph_limb39', 65536, 39), ('Automorph_limb47', 65536, 47), ('Automorph_limb48', 65536, 48), ('Automorph_limb49', 65536, 49), ('Automorph_limb51', 65536, 51), ('Automorph_limb52', 65536, 52), ('Automorph_limb53', 65536, 53)]

DEGREE65536_PARAMS = util.NTT_PARAMETERS_BY_DEGREE[65536]
DEGREE65536_MODULI = DEGREE65536_PARAMS["moduli"]
DEGREE65536_ROOTS = DEGREE65536_PARAMS["root_of_unity"]

mesh, partition_spec = util.create_sharding()
axis_names = mesh.axis_names
batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]

# Input shape: (Batch, Elements, Degree, Moduli)
# We want to shard on Batch, which is axis 0.
batch_sharding = jax.sharding.NamedSharding(
    mesh,
    partition_spec(batch_partition,),
)
num_devices = jax.device_count()
batch = num_devices

def precompute_auto_map(n: int, k: int) -> List[int]:
    """Python translation of nbtheory2.cpp PrecomputeAutoMap (lines 264-276).

    Args:
        n: ring dimension (assumed power of two).
        k: automorphism index.

    Returns:
        A list `precomp` of length `n` where precomp[jrev] = idxrev as computed
        by the C++ routine.
    """
    m = n << 1  # cyclOrder
    logm = int(round(math.log2(m)))
    logn = int(round(math.log2(n)))

    precomp: List[int] = [0] * n
    for j in range(n):
        j_tmp = (j << 1) + 1
        t = j_tmp * k
        # ((t % m) >> 1) but written to mirror the C++ bit ops exactly
        idx = (t - ((t >> logm) << logm)) >> 1

        j_rev = util.bit_reverse(j, logn)
        idx_rev = util.bit_reverse(idx, logn)
        precomp[j_rev] = idx_rev

    return precomp


def find_automorphism_index_2n_complex(i: int, m: int) -> int:
    """Python translation of nbtheory2.cpp FindAutomorphismIndex2nComplex (243-263).

    Mirrors the C++ logic including early exits, power-of-two validation, and
    modulus via bitmask for m being a power of two.
    """
    if i == 0:
        return 1
    if i == (m - 1):
        return int(i)

    if not util.is_power_of_two(m):
        raise ValueError("m should be a power of two.")

    # Conjugation automorphism generator
    g0 = pow(5, -1, m) if i < 0 else 5
    g = g0
    i_unsigned = abs(i)
    mask = m - 1
    for _ in range(1, i_unsigned):
        # Equivalent to (g * g0) % m since m is a power of two
        g = (g * g0) & mask
    return int(g)

COEF_AUTOMORPHISM_MAP = jnp.array(precompute_auto_map(65536, find_automorphism_index_2n_complex(1, 65536)), dtype=jnp.uint16)

class CKKSEvalBootstrappingTest(parameterized.TestCase):
  """A base class for running bootstrap tests.

  Example Test Case:
    If use GF(17) and N = 8 (so q=17 and N=8).
    In GF(17), the multiplicative group has order 16.
    Suppose the forward transform used a primitive 8th root of unity.
    For example, we can use omega = 2, since 2^8 mod 17 == 1 and its order is 8.
  """

  def __init__(self, *args, **kwargs):
    super(CKKSEvalBootstrappingTest, self).__init__(*args, **kwargs)
    self.random_key = jax.random.key(0)
    self.overall_moduli = DEGREE65536_MODULI
    self.bconv = bconv.BConvBarrett(DEGREE65536_MODULI)


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_NTT_MULTI_MODULI)
  def test_ntt_mm(
      self,
      degree,
      num_limbs,
  ):
    if degree == 65536:
      r = 256
      c = 256
    else:
      raise NotImplementedError("Only support degree 65536 for now")
    q_list = self.overall_moduli[0:num_limbs]
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.MontgomeryContext(moduli=q_list),
    }
    ntt_ctx = ntt.NTTCiphertextMontgomeryContext(moduli=q_list, parameters=parameters)
    test_in_twisted = jax.device_put(util.random_batched_ciphertext((batch, r, c, num_limbs), self.overall_moduli[:num_limbs], dtype=jnp.uint32), batch_sharding)
    jit_ntt = jax.jit(jax.named_call(
      ntt_ctx.ntt,
      name="jit_ntt_mm",
    ),
      in_shardings=batch_sharding,
      out_shardings=batch_sharding,
    )
    _ = jit_ntt(test_in_twisted)

    # performance measurement
    tasks = [
        (jit_ntt, (test_in_twisted,)),
    ]
    profile_name = "jit_ntt_mm"
    kernel_name = "jit_ntt_mm"
    latency = util.profile_jax_functions_xprof(tasks, profile_name, kernel_name) / batch
    print(f"ntt_common({degree},{num_limbs}) - latency:{latency}")


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_NTT_SINGLE_MODULI)
  def test_ntt_sm(
      self,
      degree,
      num_limbs,
  ):
    q = self.overall_moduli[0]
    if degree == 65536:
      r = 256
      c = 256
    else:
      raise NotImplementedError("Only support degree 65536 for now")
    if math.log2(q) > 32:
      print("Skip this test as we don't support modulus > 32, because numpy as max precision as 64")
      return
    parameters = {
        "r": r,
        "c": c,
        "finite_field_context": ff_context.MontgomeryContext(moduli=q),
    }
    ntt_single_ctx = ntt_single.NTTMontgomeryContext(moduli=q, parameters=parameters)
    test_in_twisted = jax.device_put(util.random_parameters((batch, r, c), [q], dtype=jnp.uint32), batch_sharding)
    jit_ntt = jax.jit(jax.named_call(
      ntt_single_ctx.ntt,
      name="jit_ntt_sm",
    ),
      in_shardings=batch_sharding,
      out_shardings=batch_sharding,
    )
    _ = jit_ntt(test_in_twisted)

    # performance measurement
    tasks = [
        (jit_ntt, (test_in_twisted,)),
    ]
    profile_name = "jit_ntt_sm"
    kernel_name = "jit_ntt_sm"
    latency = util.profile_jax_functions_xprof(tasks, profile_name, kernel_name) / batch
    print(f"ntt_common({degree},{num_limbs}) - latency:{latency}")


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_VEC_ADD)
  def test_vec_mod_add(
      self,
      degree,
      num_limbs,
  ):
    if degree == 65536:
      r = 256
      c = 256
    else:
      raise NotImplementedError("Only support degree 65536 for now")
    q_list = jnp.array(self.overall_moduli[0:num_limbs], dtype=jnp.uint32)
    test_in_array = jax.device_put(util.random_batched_ciphertext((batch, r, c, num_limbs), q_list, dtype=jnp.uint32), batch_sharding)
    test_in_array_2 = jax.device_put(util.random_batched_ciphertext((batch, r, c, num_limbs), q_list, dtype=jnp.uint32), batch_sharding)

    def vec_mod_add(a, b):
      c = a + b
      # Reshape q for broadcasting: (num_limbs,) -> (num_limbs, 1)
      return c
    kernel_name = "jit_vec_mod_add"
    func_dut = jax.jit(
      jax.named_call(
        vec_mod_add,
        name=kernel_name,
      ),
      in_shardings=(batch_sharding, batch_sharding),
      out_shardings=batch_sharding,
    ).lower(
      jax.ShapeDtypeStruct(test_in_array.shape, dtype=jnp.uint32, sharding=batch_sharding),
      jax.ShapeDtypeStruct(test_in_array_2.shape, dtype=jnp.uint32, sharding=batch_sharding),
    ).compile()
    jax.block_until_ready(func_dut(test_in_array, test_in_array_2))

    # performance measurement
    tasks = [
        (func_dut, (test_in_array, test_in_array_2)),
    ]
    profile_name = "test_vec_mod_add"
    latency = util.profile_jax_functions_xprof(tasks, profile_name, kernel_name) / batch
    print(f"add({num_limbs},{degree}) - latency:{latency}")

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_VEC_MUL)
  def test_vec_mod_mul(
      self,
      degree,
      num_limbs,
  ):
    if degree == 65536:
      r = 256
      c = 256
    else:
      raise NotImplementedError("Only support degree 65536 for now")
    profile_name = "jit_vec_mod_mul"
    kernel_name = "jit_vec_mod_mul"
    q_list = self.overall_moduli[0:num_limbs]
    ff_ctx = ff_context.BarrettContext(moduli=q_list)
    test_in_array = jax.device_put(util.random_batched_ciphertext((batch, r, c, num_limbs), q_list, dtype=jnp.uint32), batch_sharding)
    test_in_array_2 = jax.device_put(util.random_batched_ciphertext((batch, r, c, num_limbs), q_list, dtype=jnp.uint32), batch_sharding)
    def vec_mod_mul(a, b):
      c = a.astype(jnp.uint64) + b.astype(jnp.uint64)
      return ff_ctx.modular_reduction(c)
    func_dut = jax.jit(
        jax.named_call(
          vec_mod_mul,
          name=kernel_name,
        ),
        in_shardings=(batch_sharding, batch_sharding),
        out_shardings=batch_sharding,
      ).lower(
      jax.ShapeDtypeStruct(test_in_array.shape, dtype=jnp.uint32, sharding=batch_sharding),
      jax.ShapeDtypeStruct(test_in_array_2.shape, dtype=jnp.uint32, sharding=batch_sharding),
    ).compile()
    jax.block_until_ready(func_dut(test_in_array, test_in_array_2))

    # performance measurement
    tasks = [
        (func_dut, (test_in_array, test_in_array_2)),
    ]
    latency = util.profile_jax_functions_xprof(tasks, profile_name, kernel_name) / batch
    print(f"mult({num_limbs},{degree}) - latency:{latency}")

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_AUTOMORPHISM)
  def test_automorphism(
    self,
    degree,
    num_limbs,
  ):
    q_list = self.overall_moduli[0:num_limbs]
    test_in_array = jax.device_put(util.random_ciphertext((batch, degree, num_limbs), q_list, dtype=jnp.uint32), batch_sharding)
    def automorphism(a):
      c = jnp.take(a, COEF_AUTOMORPHISM_MAP, axis=-2)
      return c

    func_dut = jax.jit(
      jax.named_call(
        automorphism,
        name="jit_automorphism",
      ),
      in_shardings=batch_sharding,
      out_shardings=batch_sharding,
    ).lower(
      jax.ShapeDtypeStruct(test_in_array.shape, dtype=jnp.uint32, sharding=batch_sharding),
    ).compile()

    jax.block_until_ready(func_dut(test_in_array))
    # performance measurement
    tasks = [
        (func_dut, (test_in_array,)),
    ]
    profile_name = "jit(automorphism)"
    kernel_name = "automorphism"
    latency = util.profile_jax_functions_xprof(tasks, profile_name, kernel_name) / batch
    print(f"automorph({num_limbs},{degree}) - latency:{latency}")


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_BConv_OP)
  def test_basis_change(
    self,
    degree,
    limb_in,
    limb_out,
  ):
    if degree == 65536:
      r = 256
      c = 256
    else:
      raise NotImplementedError("Only support degree 65536 for now")
    in_indices = list(range(limb_in))
    out_indices = list(range(limb_in, limb_in+limb_out))
    self.bconv.control_gen([(in_indices, out_indices)], perf_test=True)
    test_in_array = jax.device_put(util.random_batched_ciphertext((batch, r, c, limb_in), self.overall_moduli[:limb_in], dtype=jnp.uint32), batch_sharding)
    profile_name = "basis_change"
    kernel_name = "jit_inner"
    jax_bconv = jax.jit(
      jax.named_call(
        functools.partial(self.bconv.basis_change_bat),
        name=kernel_name,
      ),
      in_shardings=batch_sharding,
      out_shardings=batch_sharding,
    ).lower(
      jax.ShapeDtypeStruct(test_in_array.shape, dtype=jnp.uint32, sharding=batch_sharding)
    ).compile()

    jax.block_until_ready(jax_bconv(test_in_array))
    # performance measurement
    tasks = [
        (jax_bconv, (test_in_array,)),
    ]
    latency = util.profile_jax_functions_xprof(tasks, profile_name, kernel_name) / batch
    print(f"basis_convert({limb_in},{limb_out}) - latency:{latency}")


def _normalize_dims(dim: str) -> str:
  """Return a comparable dimension key without spaces or parentheses."""
  return dim.strip().lstrip("(").rstrip(")").replace(" ", "")


def _dims_key(*numbers) -> str:
  return ",".join(str(n) for n in numbers)


def _parse_latency_log(log_text: str) -> dict[tuple[str, str], float]:
  """Parse latency lines emitted by this module's tests."""
  latencies: dict[tuple[str, str], float] = {}

  for pattern, builder in (
      (r"add\((\d+),\s*(\d+)\)\s*-\s*latency:([0-9.]+)",
       lambda m: ("add", _dims_key(m.group(1), m.group(2)), float(m.group(3)))),
      (r"mult\((\d+),\s*(\d+)\)\s*-\s*latency:([0-9.]+)",
       lambda m: ("mult", _dims_key(m.group(1), m.group(2)), float(m.group(3)))),
      (r"automorph\((\d+),\s*(\d+)\)\s*-\s*latency:([0-9.]+)",
       lambda m: ("automorph", _dims_key(m.group(1), m.group(2)), float(m.group(3)))),
      # NTT single-modulus batch path prints the degree first and includes batch info.
      (r"ntt_common\((\d+),\s*(\d+)\)\s*b\d+\s*-\s*latency:([0-9.]+)",
       lambda m: ("ntt_common", _dims_key(m.group(2), m.group(1)), float(m.group(3)))),
      (r"ntt_common\((\d+),\s*(\d+)\)\s*-\s*latency:([0-9.]+)",
       lambda m: ("ntt_common", _dims_key(m.group(2), m.group(1)), float(m.group(3)))),
      (r"basis_convert\((\d+),\s*(\d+)\)\s*-\s*latency:([0-9.]+)",
       lambda m: ("basis_convert", _dims_key(m.group(1), m.group(2), 65536), float(m.group(3)))),
  ):
    for match in re.finditer(pattern, log_text):
      kernel, dims_key, latency = builder(match)
      latencies[(kernel, _normalize_dims(f"({dims_key})"))] = latency

  return latencies


BTS_KERNEL_INVOCATION_CSV = """kernel name,dim,count
add,"(18, 65536)",2
add,"(19, 65536)",16
add,"(20, 65536)",16
add,"(21, 65536)",16
add,"(22, 65536)",8
add,"(23, 65536)",8
add,"(24, 65536)",16
add,"(25, 65536)",16
add,"(26, 65536)",16
add,"(27, 65536)",124
add,"(28, 65536)",160
add,"(29, 65536)",146
add,"(30, 65536)",300
add,"(31, 65536)",102
add,"(32, 65536)",70
add,"(33, 65536)",53
add,"(34, 65536)",16
add,"(35, 65536)",24
add,"(36, 65536)",24
add,"(37, 65536)",16
add,"(38, 65536)",8
add,"(39, 65536)",30
add,"(40, 65536)",40
add,"(43, 65536)",32
add,"(44, 65536)",16
add,"(45, 65536)",12
add,"(47, 65536)",84
add,"(48, 65536)",84
add,"(49, 65536)",84
add,"(51, 65536)",36
add,"(52, 65536)",36
add,"(53, 65536)",36
automorph,"(19, 65536)",14
automorph,"(20, 65536)",14
automorph,"(21, 65536)",14
automorph,"(22, 65536)",6
automorph,"(23, 65536)",6
automorph,"(24, 65536)",6
automorph,"(27, 65536)",42
automorph,"(28, 65536)",42
automorph,"(29, 65536)",42
automorph,"(31, 65536)",18
automorph,"(32, 65536)",18
automorph,"(33, 65536)",20
automorph,"(34, 65536)",14
automorph,"(35, 65536)",14
automorph,"(36, 65536)",14
automorph,"(37, 65536)",6
automorph,"(38, 65536)",6
automorph,"(39, 65536)",6
automorph,"(47, 65536)",42
automorph,"(48, 65536)",42
automorph,"(49, 65536)",42
automorph,"(51, 65536)",18
automorph,"(52, 65536)",18
automorph,"(53, 65536)",18
basis_convert,"(10, 21, 65536)",2
basis_convert,"(10, 22, 65536)",2
basis_convert,"(10, 23, 65536)",2
basis_convert,"(10, 29, 65536)",18
basis_convert,"(10, 30, 65536)",30
basis_convert,"(11, 24, 65536)",4
basis_convert,"(11, 25, 65536)",4
basis_convert,"(11, 26, 65536)",4
basis_convert,"(11, 32, 65536)",24
basis_convert,"(11, 33, 65536)",12
basis_convert,"(11, 34, 65536)",9
basis_convert,"(12, 27, 65536)",12
basis_convert,"(12, 28, 65536)",20
basis_convert,"(12, 33, 65536)",2
basis_convert,"(12, 35, 65536)",3
basis_convert,"(12, 36, 65536)",3
basis_convert,"(12, 37, 65536)",3
basis_convert,"(13, 30, 65536)",16
basis_convert,"(13, 31, 65536)",8
basis_convert,"(13, 32, 65536)",4
basis_convert,"(13, 38, 65536)",3
basis_convert,"(13, 39, 65536)",3
basis_convert,"(13, 40, 65536)",3
basis_convert,"(14, 33, 65536)",2
basis_convert,"(14, 34, 65536)",2
basis_convert,"(14, 35, 65536)",2
basis_convert,"(15, 36, 65536)",2
basis_convert,"(15, 37, 65536)",2
basis_convert,"(15, 38, 65536)",2
basis_convert,"(7, 20, 65536)",3
basis_convert,"(7, 21, 65536)",3
basis_convert,"(7, 22, 65536)",3
basis_convert,"(8, 23, 65536)",3
basis_convert,"(8, 24, 65536)",3
basis_convert,"(8, 25, 65536)",3
basis_convert,"(9, 18, 65536)",2
basis_convert,"(9, 19, 65536)",2
basis_convert,"(9, 20, 65536)",2
basis_convert,"(9, 26, 65536)",6
basis_convert,"(9, 27, 65536)",6
basis_convert,"(9, 28, 65536)",6
mult,"(18, 65536)",2
mult,"(19, 65536)",3
mult,"(20, 65536)",3
mult,"(21, 65536)",3
mult,"(22, 65536)",3
mult,"(23, 65536)",3
mult,"(24, 65536)",5
mult,"(25, 65536)",14
mult,"(26, 65536)",14
mult,"(27, 65536)",142
mult,"(28, 65536)",176
mult,"(29, 65536)",188
mult,"(30, 65536)",288
mult,"(31, 65536)",122
mult,"(32, 65536)",86
mult,"(33, 65536)",70
mult,"(34, 65536)",3
mult,"(35, 65536)",15
mult,"(36, 65536)",15
mult,"(37, 65536)",15
mult,"(38, 65536)",3
mult,"(39, 65536)",37
mult,"(40, 65536)",60
mult,"(43, 65536)",48
mult,"(44, 65536)",24
mult,"(45, 65536)",18
mult,"(47, 65536)",114
mult,"(48, 65536)",114
mult,"(49, 65536)",114
mult,"(51, 65536)",50
mult,"(52, 65536)",50
mult,"(53, 65536)",50
ntt_common,"(1, 65536)",44
ntt_common,"(10, 65536)",54
ntt_common,"(11, 65536)",57
ntt_common,"(12, 65536)",43
ntt_common,"(13, 65536)",37
ntt_common,"(14, 65536)",6
ntt_common,"(15, 65536)",6
ntt_common,"(18, 65536)",2
ntt_common,"(19, 65536)",2
ntt_common,"(20, 65536)",5
ntt_common,"(21, 65536)",5
ntt_common,"(22, 65536)",5
ntt_common,"(23, 65536)",5
ntt_common,"(24, 65536)",7
ntt_common,"(25, 65536)",7
ntt_common,"(26, 65536)",12
ntt_common,"(27, 65536)",22
ntt_common,"(28, 65536)",34
ntt_common,"(29, 65536)",34
ntt_common,"(30, 65536)",54
ntt_common,"(31, 65536)",12
ntt_common,"(32, 65536)",30
ntt_common,"(33, 65536)",16
ntt_common,"(34, 65536)",11
ntt_common,"(35, 65536)",5
ntt_common,"(36, 65536)",5
ntt_common,"(37, 65536)",5
ntt_common,"(38, 65536)",5
ntt_common,"(39, 65536)",3
ntt_common,"(40, 65536)",3
ntt_common,"(7, 65536)",9
ntt_common,"(8, 65536)",9
ntt_common,"(9, 65536)",24"""


def accumulate_latency_from_log(
    log_text: str,
    # csv_path: str = "tabIX_bts_kernel_invocation.csv", # Deprecated
) -> float:
  """Accumulate total latency using log output and invocation counts.

  Args:
    log_text: Raw text containing latency printouts from this module.
    # csv_path: Path to the CSV with invocation counts.

  Returns:
    Total accumulated latency after multiplying each kernel latency by its
    invocation count, except for ntt_common (1,65536) which is already applied.
  """
  latencies = _parse_latency_log(log_text)
  total_latency = 0.0

  # with open(csv_path, newline="") as csvfile:
  with io.StringIO(BTS_KERNEL_INVOCATION_CSV) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      kernel = row["kernel name"]
      dims_key = _normalize_dims(row["dim"])
      count = int(row["count"])
      latency = latencies.get((kernel, dims_key))
      if latency is None:
        continue
      if kernel == "ntt_common" and dims_key == "1,65536":
        total_latency += latency
      else:
        total_latency += latency * count

  return total_latency


def print_latency_ratios(log_text: str) -> None:
  """Print ratio of overall latency from each kernel type."""
  latencies = _parse_latency_log(log_text)

  # Accumulate latency per kernel type
  kernel_totals: dict[str, float] = {}
  total_overall = 0.0

  with io.StringIO(BTS_KERNEL_INVOCATION_CSV) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      kernel = row["kernel name"]
      dims_key = _normalize_dims(row["dim"])
      count = int(row["count"])

      latency = latencies.get((kernel, dims_key))
      if latency is None:
        continue

      if kernel == "ntt_common" and dims_key == "1,65536":
        invocation_latency = latency
      else:
        invocation_latency = latency * count

      kernel_totals[kernel] = kernel_totals.get(kernel, 0.0) + invocation_latency
      total_overall += invocation_latency

  print("\nLatency Ratios by Kernel Type:")
  print("-" * 30)
  for kernel in ["mult", "ntt_common", "basis_convert", "automorph", "add"]:
    if kernel in kernel_totals:
        ratio = (kernel_totals[kernel] / total_overall) * 100
        print(f"{kernel:<15}: {ratio:6.2f}%")
  print("-" * 30)


if __name__ == "__main__":
  class _Tee(io.StringIO):
    """StringIO that also mirrors writes to another stream."""

    def __init__(self, mirror_stream):
      super().__init__()
      self._mirror_stream = mirror_stream

    def write(self, s):
      if self._mirror_stream is not None:
        self._mirror_stream.write(s)
      return super().write(s)

    def flush(self):
      if self._mirror_stream is not None:
        self._mirror_stream.flush()
      return super().flush()

  tee_buffer = _Tee(sys.stdout)
  exit_code = 0
  try:
    with contextlib.redirect_stdout(tee_buffer):
      absltest.main()
  except SystemExit as exc:
    exit_code = exc.code if isinstance(exc.code, int) else 0

  log_text = tee_buffer.getvalue()
  total_latency = accumulate_latency_from_log(log_text)
  print(f"Total accumulated latency: {total_latency}")
  print_latency_ratios(log_text)
  if exit_code:
    sys.exit(exit_code)

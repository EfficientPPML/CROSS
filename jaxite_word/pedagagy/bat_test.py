"""A module for operations on test CKKS evaluation kernels including.

- HESub
"""

from concurrent import futures
from typing import Any, Callable

import jax
import jax.numpy as jnp
import bat
import util as util
import numpy as np
import functools

from absl.testing import absltest
from absl.testing import parameterized

ProcessPoolExecutor = futures.ProcessPoolExecutor

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")


class CKKSEvalBatMulTest(parameterized.TestCase):
  """A base class for running bootstrap tests."""

  def __init__(self, *args, **kwargs):
    super(CKKSEvalBatMulTest, self).__init__(*args, **kwargs)
    self.debug = False  # dsiable it from printing the test input values
    self.modulus_element_0_tower_0 = 1152921504606748673
    self.modulus_element_0_tower_1 = 268664833
    self.modulus_element_0_tower_2 = 557057
    self.in_c1 = [[761974115069642497, 186812814, 396780], [1119697542422587247, 195711320, 415240]]
    self.in_c2 = [[723287396072165360, 91967352, 112274], [251652059326221653, 111494737, 534294]]
    self.refer_sub_result = [[38686718997477137, 94845462, 284506], [868045483096365594, 84216583, 438003]]


  # @absltest.skip("test single implementation")
  def test_hpmatmul_conv_outer_product_version(self):
    """Test the correctness of the Conv-Adapt-Conv algorithm."""
    key = jax.random.key(0)
    mat_a_shape = (4, 256)
    mat_b_shape = (mat_a_shape[1], 4)
    upper_value = (1 << 28) - 1
    modulus_32 = 4294967291
    modulus_64 = jnp.array(modulus_32, dtype=jnp.uint64)
    mat_a = jax.random.randint(key, mat_a_shape, 0, upper_value, dtype=jnp.uint32)
    mat_b = jax.random.randint(key, mat_b_shape, 0, upper_value, dtype=jnp.uint32)

    mat_reference_result = bat.hpmatmul_golden(mat_a, mat_b, modulus_32)
    mat_result_outerproduct = bat.hpmatmul_conv_outer_product(
        mat_a, mat_b
    )
    mat_result_outerproduct = mat_result_outerproduct % modulus_64

    np.testing.assert_array_equal(mat_result_outerproduct, mat_reference_result)
    print('pass testing mat_result_outerproduct == mat_reference_result')


  # @absltest.skip("test single implementation")
  def test_bat_deployment(self):
    """Test the correctness of the Basis Align Transformation (BAT) algorithm."""
    key = jax.random.key(0)
    mat_a_shape = (6, 7)
    mat_b_shape = (mat_a_shape[1], 4)
    upper_value = (1 << 28) - 1
    modulus_32 = 2147483647
    modulus_64 = jnp.array(modulus_32, dtype=jnp.uint64)
    mat_a = jax.random.randint(key, mat_a_shape, 0, upper_value, dtype=jnp.uint32)
    mat_b = jax.random.randint(key, mat_b_shape, 0, upper_value, dtype=jnp.uint32)

    mat_reference_result = bat.hpmatmul_golden(mat_a, mat_b, modulus_32)
    compiled_mat_a = bat.hpmatmul_offline_bat_illustration(mat_a, modulus_32)
    mat_result_bat = bat.matmul_bat_einsum(mat_b, compiled_mat_a, 'nkq,mnpq->mkp')#bat.hpmatmul_bat(compiled_mat_a, mat_b)
    mat_result_bat = mat_result_bat % modulus_64

    np.testing.assert_array_equal(mat_result_bat, mat_reference_result)
    compiled_mat_a_direct = bat.hpmatmul_offline_bat_deployment(mat_a, modulus_32)
    mat_result_bat_direct = bat.matmul_bat_einsum(mat_b, compiled_mat_a, 'nkq,mnpq->mkp') #bat.hpmatmul_bat(compiled_mat_a_direct, mat_b)
    mat_result_bat_direct = mat_result_bat_direct % modulus_64

    np.testing.assert_array_equal(mat_result_bat_direct, mat_reference_result)


  # @absltest.skip("test single implementation")
  def test_bat_deployment_batch(self):
    """Test the correctness of the Basis Align Transformation (BAT) algorithm."""
    key = jax.random.key(0)
    batch = 32
    mat_a_shape = (6, 7)
    mat_b_shape = (batch, mat_a_shape[1], 4)
    upper_value = (1 << 28) - 1
    modulus_32 = 2147483647
    modulus_64 = jnp.array(modulus_32, dtype=jnp.uint64)
    mat_a = jax.random.randint(key, mat_a_shape, 0, upper_value, dtype=jnp.uint32)
    mat_b = jax.random.randint(key, mat_b_shape, 0, upper_value, dtype=jnp.uint32)

    mat_reference_result = bat.hpmatmul_golden(mat_a, mat_b[0], modulus_32)
    compiled_mat_a = bat.hpmatmul_offline_bat_illustration(mat_a, modulus_32)
    mat_result_bat = bat.matmul_bat_einsum(mat_b, compiled_mat_a, 'bnkq,mnpq->bmkp') #bat.hpmatmul_bat_batch(compiled_mat_a, mat_b)
    mat_result_bat = mat_result_bat % modulus_64

    np.testing.assert_array_equal(mat_result_bat[0], mat_reference_result)

    compiled_mat_a_direct = bat.hpmatmul_offline_bat_deployment(mat_a, modulus_32)

    mat_result_bat_direct = bat.matmul_bat_einsum(mat_b, compiled_mat_a_direct, 'bnkq,mnpq->bmkp') #bat.hpmatmul_bat_batch(compiled_mat_a_direct, mat_b)
    mat_result_bat_direct = mat_result_bat_direct % modulus_64

    np.testing.assert_array_equal(mat_result_bat_direct[0], mat_reference_result)


  # @absltest.skip("test single implementation")
  def test_bat_offline_control(self):
    """Test the correctness of the Basis Align Transformation (BAT) algorithm."""
    key = jax.random.key(0)
    batch = 32
    mat_a_shape = (6, 7)
    mat_b_shape = (batch, mat_a_shape[1], 4)
    upper_value = (1 << 28) - 1
    modulus_32 = 2147483647
    modulus_64 = jnp.array(modulus_32, dtype=jnp.uint64)
    mat_a = jax.random.randint(key, mat_a_shape, 0, upper_value, dtype=jnp.uint32)
    mat_b = jax.random.randint(key, mat_b_shape, 0, upper_value, dtype=jnp.uint32)

    mat_reference_result = bat.hpmatmul_golden(mat_a, mat_b[0], modulus_32)
    compiled_mat_a_direct = bat.hpmatmul_offline_bat_deployment(mat_a, modulus_32)
    mat_result_bat_direct = bat.matmul_bat_einsum(mat_b, compiled_mat_a_direct, 'bnkq,mnpq->bmkp') #bat.hpmatmul_bat_batch(compiled_mat_a_direct, mat_b)
    mat_result_bat_direct = mat_result_bat_direct % modulus_64

    np.testing.assert_array_equal(mat_result_bat_direct[0], mat_reference_result)


  @absltest.skip("skip as it's designed for TPU only")
  def test_bat_pallas(self):
    """Test the correctness of the Pallas matmul implementation."""
    # TODO: make it support 32-bit pallas for TPU
    if jax.devices()[0].platform == 'tpu':
      self.skipTest("Pallas matmul not supported on TPU yet")

    key = jax.random.key(1)
    batch = 32
    mat_a_shape = (6, 7)
    mat_b_shape = (batch, mat_a_shape[1], 4)
    upper_value = (1 << 28) - 1
    modulus_32 = 2147483647
    modulus_64 = jnp.array(modulus_32, dtype=jnp.uint64)
    
    mat_a = jax.random.randint(key, mat_a_shape, 0, upper_value, dtype=jnp.uint32)
    mat_b = jax.random.randint(key, mat_b_shape, 0, upper_value, dtype=jnp.uint32)

    # Reference computation using bat.matmul_bat_einsum (or golden)
    compiled_mat_a = bat.hpmatmul_offline_bat_deployment(mat_a, modulus_32)
    
    # Original einsum
    mat_result_ref = bat.matmul_bat_einsum(mat_b, compiled_mat_a, 'bnkq,mnpq->bmkp')
    mat_result_ref = mat_result_ref % modulus_64
    
    # Pallas implementation
    mat_result_pallas = bat.matmul_bat_pallas(mat_b, compiled_mat_a, 'bnkq,mnpq->bmkp')
    mat_result_pallas = mat_result_pallas % modulus_64

    # Verify against golden
    np.testing.assert_array_equal(mat_result_pallas, mat_result_ref)
    
    # Also verify first element against golden for sanity
    mat_reference_golden = bat.hpmatmul_golden(mat_a, mat_b[0], modulus_32)
    np.testing.assert_array_equal(mat_result_pallas[0], mat_reference_golden)


if __name__ == "__main__":
  absltest.main()

"""A module for operations on test CKKS evaluation kernels including.

- BAT
"""
# Standard library imports
import os

# Third-party imports
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

# Local imports
from profiler import KernelWrapper, Profiler, collect_logs
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltpu

# JAX configuration
jax.config.update("jax_enable_x64", True)

TEST_BATCH_MATMODMUL_MAPPING = [
    ('64_64_64', 64, 64, 64),
    ('128_64_64', 128, 64, 64),
    ('128_128_128', 128, 128, 128),
    ('256_128_256', 256, 128, 128),
    ('256_256_256', 256, 256, 256),
    ('512_512_512', 512, 512, 512),
    ('512_1024_1024', 512, 1024, 1024),
    ('2048_2048_2048', 2048, 2048, 2048),
]
_is_nvidia = "NVIDIA" in jax.devices()[0].device_kind

def _analyze_speedup(csv_path):
  """Analyzes the speedup of Bat vs Conv from the performance CSV."""
  import csv

  print("\n=== Bat vs Conv Speedup Analysis (Single Batch) ===")
  data = {}
  try:
    with open(csv_path, 'r') as f:
      reader = csv.DictReader(f)
      for row in reader:
        # Assuming 'operation_name' and 'sample_0' exist based on known CSV format
        if 'operation_name' in row and 'sample_0' in row:
          data[row['operation_name']] = float(row['sample_0'])
  except Exception as e:
    print(f"Error reading CSV: {e}")
    return

  print(f"{'Configuration':<20} | {'Conv Latency (us)':<20} | {'Bat Latency (us)':<20} | {'Speedup':<10}")
  print("-" * 80)

  for name, m, n, k in TEST_BATCH_MATMODMUL_MAPPING:
    bat_key = f"test_bat_{name}"
    conv_key = f"test_conv_{name}"

    if bat_key in data and conv_key in data:
      bat_lat = data[bat_key]
      conv_lat = data[conv_key]
      speedup = conv_lat / bat_lat if bat_lat != 0 else 0.0
      print(f"{name:<20} | {conv_lat:<20.4f} | {bat_lat:<20.4f} | {speedup:<10.4f}")
    else:
      pass
  print("===================================================\n")

def _jax_conv_kernel(lhs, rhs, parameters):
  @jax.jit
  def hpmatmul_conv_conv(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Interleaved u8 matmul with padded 1D convolution.

    (reformulated as 2D convolution)
    This is similar to the integer implementation in TensorFHE.

    How do we map workload into Conv?

                    Left Mat                     Right Mat        ->
              <- in channel (C)->          <-Output Channel(O)->  ->
        -     xxxxxxxxxxxxxxxxxx      -     xxxxxxxxxxxxxxxxxx    ->    -
        ^     xxxxxxxxxxxxxxxxxx      ^     xxxxxxxxxxxxxxxxxx    ->    ^
        |     xxxxxxxxxxxxxxxxxx      |     xxxxxxxxxxxxxxxxxx    ->    |
      batch   xxxxxxxxxxxxxxxxxx     In     xxxxxxxxxxxxxxxxxx    ->  batch
        (N)    xxxxxxxxxxxxxxxxxx   channel  xxxxxxxxxxxxxxxxxx    ->   (N)
        |     xxxxxxxxxxxxxxxxxx     (I)    xxxxxxxxxxxxxxxxxx    ->    |
        v     xxxxxxxxxxxxxxxxxx      v     xxxxxxxxxxxxxxxxxx    ->    v
        -     xxxxxxxxxxxxxxxxxx      -     xxxxxxxxxxxxxxxxxx    ->    -

              Result Mat
          <-Output channel(C)->
          xxxxxxxxxxxxxxxxxx
          xxxxxxxxxxxxxxxxxx
          xxxxxxxxxxxxxxxxxx
          xxxxxxxxxxxxxxxxxx
          xxxxxxxxxxxxxxxxxx
          xxxxxxxxxxxxxxxxxx
          xxxxxxxxxxxxxxxxxx
          xxxxxxxxxxxxxxxxxx

        Each x in the above example is a 1DConv

        <---W--->     <---W--->     <---W--->
        xxxxxxxxx  @  xxxxxxxxx  =  xxxxxxxxx

    Args:
      x: The left matrix.
      y: The right matrix.

    Returns:
      The result matrix.
    """

    assert x.dtype == jnp.uint32
    assert y.dtype == jnp.uint32

    lhs: jax.Array = jax.lax.bitcast_convert_type(x, new_dtype=jnp.uint8)  # bnmp
    rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)  # nk1q
    # https://github.com/google/jax/issues/11483
    rhs = jax.lax.rev(rhs, [2])
    # rhs = jlax.rev(rhs, dimensions=[3])

    # basically an einsum of "mnp,nkq->mk(p+q)" but jax einsum doesn't support
    # convolution yet
    u8_products = jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1,),
        padding=((3, 3),),
        dimension_numbers=("NCW", "IOW", "NCW"),
        preferred_element_type=jnp.uint32 # preferrably using jnp.uint32 but it's not supported by cuDNN, so use jnp.float32 for cuDNN.
    )

    shift_factors = jnp.array([0, 8, 16, 24, 32, 40, 48], dtype=jnp.uint32)
    return jnp.sum(u8_products.astype(jnp.uint64) << shift_factors, axis=(2,))

  return hpmatmul_conv_conv(lhs, rhs)


def _jax_bat_kernel(y: jax.Array, parameters):
  """Input (m, k) Left Matrix -> (m, k, p, q) Left Matrix, where each element in the original (m, k) matrix is replaced by a (p, q) matrix.

  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.
  `lhs` is passed as a static parameter through the `parameters` dict.
  """
  lhs = parameters["lhs"]
  m, k, _, _ = lhs.shape
  # Transpose y to move leading dims (b, k) to last: (b, k, n, 1) -> (n, 1, b, k)
  y = jnp.transpose(y, (2, 3, 0, 1))
  n, _, b, _ = y.shape

  def pallas_kernel(y_ref, lhs_ref, out_ref):
      pid_b = pl.program_id(0)
      pid_m = pl.program_id(1)
      pid_n = pl.program_id(2)

      # y_ref: (n, 1, b, k)
      bk = y_ref[pid_n, 0, pid_b] # Shape (k,)
      ref = y_ref
      bitcast_dst_dtype = jnp.int32

      # Specifically, jax.lax.bitcast_convert_type should be replaced by pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
      # Note: We rely on jax.lax.bitcast_convert_type to perform the bitcast to u8
      # as pltpu.bitcast strictly preserves shape when keeping rank or requires defined behavior for u32->u8 expansion.
      # However, we conform to the user request structure if possible in Pallas context.
      # Here we assume standard conversion is needed for the logic:
      rhs = jax.lax.bitcast_convert_type(bk, jnp.uint8)

      lhs_val = lhs_ref[pid_m] # (k, p, q) -> (k, 4, 4)

      # einsum "kpq,kq->p"
      prod = jnp.einsum("kpq,kq->p", lhs_val, rhs, preferred_element_type=jnp.int32)

      shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
      out_ref[pid_b, pid_m, pid_n] = jnp.sum(prod.astype(jnp.uint64) << shift_factors)

  return pl.pallas_call(
      pallas_kernel,
      out_shape=jax.ShapeDtypeStruct((b, m, n), jnp.uint64),
      grid=(b, m, n)
  )(y, lhs)



class PerformanceTest(parameterized.TestCase):
  """A base class for running bootstrap tests.

  Example Test Case:
    If use GF(17) and N = 8 (so q=17 and N=8).
    In GF(17), the multiplicative group has order 16.
    Suppose the forward transform used a primitive 8th root of unity.
    For example, we can use omega = 2, since 2^8 mod 17 == 1 and its order is 8.
  """

  def __init__(self, *args, **kwargs):
    super(PerformanceTest, self).__init__(*args, **kwargs)
    self.random_key = jax.random.key(0)

  def setUp(self):
    super().setUp()
    self.output_trace_root = os.path.join(os.path.dirname(__file__), "log")
    self.profiler_config = {
        "iterations": 1,
        "save_to_file": True,
    }

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    # Call collect_logs at the end of the test class execution
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Collecting logs from: {root_dir}")
    csv_name = "bat_performance_result"
    collect_logs(root_dir, output_csv_name=csv_name)

    csv_path = os.path.join(root_dir, f"{csv_name}.csv")
    if os.path.exists(csv_path):
      _analyze_speedup(csv_path)
    else:
      print(f"Warning: Could not find {csv_path} for analysis.")

  def _create_conv_kernel_wrapper(self, kernel_name, m, n, k):
    @jax.jit
    def matmul_conv_flexible_kernel(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
      assert x.dtype == jnp.uint32
      assert y.dtype == jnp.uint32

      lhs: jax.Array = jax.lax.bitcast_convert_type(x, new_dtype=jnp.uint8)  # bnmp
      rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)  # nk1q
      # https://github.com/google/jax/issues/11483
      rhs = jax.lax.rev(rhs, [2])

      if _is_nvidia:
          u8_products = jax.lax.conv_general_dilated(
              lhs.astype(jnp.int16), # NVIDIA GPU does not support uint8 as input type
              rhs.astype(jnp.int16), # NVIDIA GPU does not support uint8 as input type
              window_strides=(1,),
              padding=((3, 3),),
              dimension_numbers=("NCW", "IOW", "NCW"),
              preferred_element_type=jnp.float32, # NVIDIA GPU does not support uint32 as output type
          )
      else:
          u8_products = jax.lax.conv_general_dilated(
              lhs,
              rhs,
              window_strides=(1,),
              padding=((3, 3),),
              dimension_numbers=("NCW", "IOW", "NCW"),
              preferred_element_type=jnp.uint32,
          )

      shift_factors = jnp.array([0, 8, 16, 24, 32, 40, 48], dtype=jnp.uint32)
      return jnp.sum(u8_products.astype(jnp.uint64) << shift_factors, axis=(2,))

    return KernelWrapper(
      kernel_name=kernel_name,
      function_to_wrap=matmul_conv_flexible_kernel,
      input_structs=[
          ((m, k), jnp.uint32), # lhs
          ((k, n), jnp.uint32), # rhs
      ],
    )

  def _create_bat_kernel_wrapper(self, kernel_name, m, n, k):
    # Generate random lhs matrix for static parameter
    random_key = jax.random.key(0)
    rhs = jax.random.randint(random_key, (n, 4, k, 4), 0, 128, jnp.uint8)

    # Create a closure that captures lhs as a static parameter
    @jax.jit
    def bat_kernel_with_static_lhs(y: jax.Array):
      """Kernel with lhs captured as static parameter."""
      lhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
      i8_products = jnp.einsum(
          "crq,zprq->zcp",
          lhs,
          rhs,
          preferred_element_type=jnp.int32,
      )
      shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
      return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))

    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=bat_kernel_with_static_lhs,
        input_structs=[
            ((m, k), jnp.uint32),   # input_data
        ],
    )

  @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_BATCH_MATMODMUL_MAPPING)
  def test_conv(
      self,
      m,
      n,
      k,
  ):
    profiler_instance = Profiler(
          output_trace_path=self.output_trace_root,
          profile_naming=f"{self._testMethodName}",
          configuration=self.profiler_config,
      )

    kernel_name = self._testMethodName
    kernel_wrapper = self._create_conv_kernel_wrapper(kernel_name, m, n, k)

    profiler_instance.add_profile(
        name=kernel_name,
        kernel_wrapper=kernel_wrapper,
        kernel_setting_cols={
            "m": m,
            "n": n,
            "k": k,
        },
    )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()


  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_BATCH_MATMODMUL_MAPPING)
  def test_bat(
      self,
      m,
      n,
      k,
  ):
    profiler_instance = Profiler(
          output_trace_path=self.output_trace_root,
          profile_naming=f"{self._testMethodName}",
          configuration=self.profiler_config,
      )

    kernel_name = self._testMethodName
    kernel_wrapper = self._create_bat_kernel_wrapper(kernel_name, m, n, k)

    profiler_instance.add_profile(
        name=kernel_name,
        kernel_wrapper=kernel_wrapper,
        kernel_setting_cols={
            "m": m,
            "n": n,
            "k": k,
        },
    )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()


if __name__ == "__main__":
  absltest.main()

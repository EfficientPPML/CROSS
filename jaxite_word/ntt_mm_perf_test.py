import os

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

import finite_field as ff_context
import ntt_mm as ntt
import util
from profiler import KernelWrapper, Profiler, collect_logs

# JAX configuration
jax.config.update("jax_enable_x64", True)

BATCH_SIZE_LIST_LOW_DEGREE = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] # small degree batch size
BATCH_SIZE_LIST_HIHG_DEGREE = [1, 2, 4, 8, 16, 32, 64, 128] # large degree batch size
TEST_PARAMS_NTT=[('2_12', 4096, 4, BATCH_SIZE_LIST_LOW_DEGREE), ('2_13', 8192, 8, BATCH_SIZE_LIST_LOW_DEGREE), ('2_14', 16384, 16, BATCH_SIZE_LIST_LOW_DEGREE), ('2_16_L48', 65536, 48, BATCH_SIZE_LIST_HIHG_DEGREE)]

# Degree to (r, c) mapping for NTT layout configurations
DEGREE_TO_RC_MAPPING = {
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

class NTTMMPerformanceTest(parameterized.TestCase):
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
    # Determine the root directory relative to this script
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Collecting logs from: {root_dir}")
    collect_logs(root_dir)

  def _create_kernel_wrapper(self, kernel_name, ctx, batch, rows, cols, num_moduli):
    input_shape = (batch, rows, cols, num_moduli)
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_ntt_kernel,
        input_structs=[(input_shape, jnp.uint32)],
        parameters={"ctx": ctx},
    )

  def _profile_context(self, profile_prefix, ctx_cls, ff_ctx_cls, degree, num_limbs, batch_size_list):
    rows, cols = DEGREE_TO_RC_MAPPING[degree]
    moduli = util.moduli_28_list[degree][:num_limbs]
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"{profile_prefix}_degree_{degree}",
        configuration=self.profiler_config,
    )

    for batch in batch_size_list:
      parameters = {
          "r": rows,
          "c": cols,
          "finite_field_context": ff_ctx_cls(moduli=moduli),
      }
      ctx = ctx_cls(moduli=moduli, parameters=parameters, perf_test=True)

      kernel_wrapper = self._create_kernel_wrapper(
          kernel_name=f"{profile_prefix}_batch_{batch}",
          ctx=ctx,
          batch=batch,
          rows=rows,
          cols=cols,
          num_moduli=len(moduli),
      )

      profiler_instance.add_profile(
          name=f"{profile_prefix}_batch_{batch}",
          kernel_wrapper=kernel_wrapper,
          kernel_setting_cols={
              "degree": degree,
              "num_limbs": num_limbs,
              "batch": batch,
              "rows": rows,
              "cols": cols,
          },
      )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_NTT)
  def test_NTT_Barrett_performance(self, degree, num_limbs, batch_size_list):
    self._profile_context(
        profile_prefix="ntt_barrett",
        ctx_cls=ntt.NTTCiphertextBarrettContext,
        ff_ctx_cls=ff_context.BarrettContext,
        degree=degree,
        num_limbs=num_limbs,
        batch_size_list=batch_size_list,
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_NTT)
  def test_NTT_Montgomery_performance(self, degree, num_limbs, batch_size_list):
    self._profile_context(
        profile_prefix="ntt_montgomery",
        ctx_cls=ntt.NTTCiphertextMontgomeryContext,
        ff_ctx_cls=ff_context.MontgomeryContext,
        degree=degree,
        num_limbs=num_limbs,
        batch_size_list=batch_size_list,
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_NTT)
  def test_NTT_Shoup_performance(self, degree, num_limbs, batch_size_list):
    self._profile_context(
        profile_prefix="ntt_shoup",
        ctx_cls=ntt.NTTCiphertextShoupContext,
        ff_ctx_cls=ff_context.ShoupContext,
        degree=degree,
        num_limbs=num_limbs,
        batch_size_list=batch_size_list,
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_NTT)
  def test_NTT_BATLazy_performance(self, degree, num_limbs, batch_size_list):
    self._profile_context(
        profile_prefix="ntt_BATLazy",
        ctx_cls=ntt.NTTCiphertextBATLazyContext,
        ff_ctx_cls=ff_context.BATLazyContext,
        degree=degree,
        num_limbs=num_limbs,
        batch_size_list=batch_size_list,
    )


class NTTMMShardedPerformanceTest(parameterized.TestCase):
  """Profiles NTT contexts with sharding across the batch dimension."""

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
    collect_logs(root_dir)

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
    rows, cols = DEGREE_TO_RC_MAPPING[degree]
    moduli = util.moduli_28_list[degree][:num_limbs]
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

    profiler_config = self.profiler_config.copy()
    profiler_config["enable_sharding"] = True
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"sharding_{profile_prefix}_degree_{degree}",
        configuration=profiler_config,
    )

    for batch in batch_size_list:
      ctx_parameters = {
          "r": rows,
          "c": cols,
          "finite_field_context": ff_ctx_cls(moduli=moduli),
      }
      ctx = ctx_cls(moduli=moduli, parameters=ctx_parameters, perf_test=True)

      kernel_name = f"sharding_{profile_prefix}_batch_{batch}"
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

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_NTT)
  def test_sharded_NTT_Barrett_performance(self, degree, num_limbs, batch_size_list):
    self._profile_context_sharded(
        profile_prefix="ntt_barrett",
        ctx_cls=ntt.NTTCiphertextBarrettContext,
        ff_ctx_cls=ff_context.BarrettContext,
        degree=degree,
        num_limbs=num_limbs,
        batch_size_list=batch_size_list,
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_NTT)
  def test_sharded_NTT_Montgomery_performance(self, degree, num_limbs, batch_size_list):
    self._profile_context_sharded(
        profile_prefix="ntt_montgomery",
        ctx_cls=ntt.NTTCiphertextMontgomeryContext,
        ff_ctx_cls=ff_context.MontgomeryContext,
        degree=degree,
        num_limbs=num_limbs,
        batch_size_list=batch_size_list,
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_NTT)
  def test_sharded_NTT_Shoup_performance(self, degree, num_limbs, batch_size_list):
    self._profile_context_sharded(
        profile_prefix="ntt_shoup",
        ctx_cls=ntt.NTTCiphertextShoupContext,
        ff_ctx_cls=ff_context.ShoupContext,
        degree=degree,
        num_limbs=num_limbs,
        batch_size_list=batch_size_list,
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS_NTT)
  def test_sharded_NTT_BATLazy_performance(self, degree, num_limbs, batch_size_list):
    self._profile_context_sharded(
        profile_prefix="ntt_BATLazy",
        ctx_cls=ntt.NTTCiphertextBATLazyContext,
        ff_ctx_cls=ff_context.BATLazyContext,
        degree=degree,
        num_limbs=num_limbs,
        batch_size_list=batch_size_list,
    )


if __name__ == "__main__":
  absltest.main()


from absl.testing import absltest
from absl.testing import parameterized
import finite_field as ff_context
import jax.numpy as jnp
import jax
import numpy as np
import util
import os
from profiler import KernelWrapper, Profiler, collect_logs


def _montgomery_kernel(rhs, parameters):
    context = parameters['context']
    lhs = parameters['lhs']
    ab_mont = lhs.astype(jnp.uint64) * rhs.astype(jnp.uint64)
    return context.modular_reduction(ab_mont)

def _barrett_kernel(rhs, parameters):
    context = parameters['context']
    lhs = parameters['lhs']
    ab = lhs.astype(jnp.uint64) * rhs.astype(jnp.uint64)
    return context.modular_reduction(ab)

def _shoup_kernel(rhs, parameters):
    context = parameters['context']
    lhs = parameters['lhs']
    lhs_pre = parameters['lhs_pre']
    ab = lhs.astype(jnp.uint64) * rhs.astype(jnp.uint64)
    ab_shoup = lhs_pre * rhs.astype(jnp.uint64)
    return context.modular_reduction(ab, ab_shoup)

def _bat_lazy_kernel(rhs, parameters):
    context = parameters['context']
    lhs = parameters['lhs']
    ab = lhs.astype(jnp.uint64) * rhs.astype(jnp.uint64)
    return context.modular_reduction(ab)

# Test parameters sweeping
# (batch, element, r, c, moduli)
# moduli=51, element=2, r=c=256
BATCH_SIZES = [1, 2]#, 4, 8, 16, 32, 64]

class FiniteFieldPerfTest(parameterized.TestCase):

  def setUp(self):
      self.output_trace_root = os.path.join(os.path.dirname(__file__), "log")
      self.profiler_config = {
          "iterations": 1,
          "save_to_file": True,
      }

  @classmethod
  def tearDownClass(cls):
      super().tearDownClass()
      root_dir = os.path.dirname(os.path.abspath(__file__))
      print(f"Collecting logs from: {root_dir}")
      collect_logs(root_dir, output_csv_name="finite_field_profiling")

  # @absltest.skip("skip this test for now")
  def test_perf_montgomery(self):
    print("Testing Montgomery Performance")
    element = 2
    r = 256
    c = 256
    precision = 31
    moduli = util.find_moduli_ntt(51, precision, 16)
    context = ff_context.MontgomeryContext(moduli)

    profiler_instance = Profiler(
          output_trace_path=self.output_trace_root,
          profile_naming=f"{self._testMethodName}",
          configuration=self.profiler_config,
    )

    for batch in BATCH_SIZES:
        print(f"Running Montgomery for batch size: {batch}")
        shape = (batch, element, r, c, len(moduli))
        kernel_name = f"Montgomery_Perf_Batch_{batch}"
        
        # Prepare static parameters
        key = jax.random.PRNGKey(0)
        lhs = jax.random.randint(key, shape, 0, 0xFFFFFFFF, dtype=jnp.uint32)
        
        kernel_wrapper = KernelWrapper(
            kernel_name=kernel_name,
            function_to_wrap=_montgomery_kernel,
            input_structs=[
                (shape, jnp.uint32), # rhs
            ],
            parameters={"context": context, "lhs": lhs}
        )
        
        profiler_instance.add_profile(
            name=kernel_name,
            kernel_wrapper=kernel_wrapper,
            kernel_setting_cols={
                "batch": batch,
                "algorithm": "montgomery",
                "moduli_count": len(moduli)
            },
        )
    
    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()

  # @absltest.skip("skip this test for now")
  def test_perf_barrett(self):
    print("Testing Barrett Performance")
    element = 2
    r = 256
    c = 256
    precision = 31
    moduli = util.find_moduli_ntt(51, precision, 16)
    context = ff_context.BarrettContext(moduli)

    profiler_instance = Profiler(
          output_trace_path=self.output_trace_root,
          profile_naming=f"{self._testMethodName}",
          configuration=self.profiler_config,
    )

    for batch in BATCH_SIZES:
        print(f"Running Barrett for batch size: {batch}")
        shape = (batch, element, r, c, len(moduli))
        kernel_name = f"Barrett_Perf_Batch_{batch}"

        # Prepare static parameters
        key = jax.random.PRNGKey(0)
        lhs = jax.random.randint(key, shape, 0, 0xFFFFFFFF, dtype=jnp.uint32)

        kernel_wrapper = KernelWrapper(
            kernel_name=kernel_name,
            function_to_wrap=_barrett_kernel,
            input_structs=[
                (shape, jnp.uint32), # rhs
            ],
            parameters={"context": context, "lhs": lhs}
        )

        profiler_instance.add_profile(
            name=kernel_name,
            kernel_wrapper=kernel_wrapper,
            kernel_setting_cols={
                "batch": batch,
                "algorithm": "barrett",
                "moduli_count": len(moduli)
            },
        )
    
    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()

  # @absltest.skip("skip this test for now")
  def test_perf_shoup(self):
    print("Testing Shoup Performance")
    element = 2
    r = 256
    c = 256
    precision = 31
    moduli = util.find_moduli_ntt(51, precision, 16)
    
    # Context initialization
    context = ff_context.ShoupContext(moduli)

    profiler_instance = Profiler(
          output_trace_path=self.output_trace_root,
          profile_naming=f"{self._testMethodName}",
          configuration=self.profiler_config,
    )

    for batch in BATCH_SIZES:
        print(f"Running Shoup for batch size: {batch}")
        shape = (batch, element, r, c, len(moduli))
        kernel_name = f"Shoup_Perf_Batch_{batch}"
        
        # Note: Shoup kernel needs 3 inputs: lhs, rhs, lhs_pre
        
        # Prepare static parameters
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        lhs = jax.random.bits(k1, shape, dtype=jnp.uint64)
        lhs_pre = jax.random.bits(k2, shape, dtype=jnp.uint64)

        kernel_wrapper = KernelWrapper(
            kernel_name=kernel_name,
            function_to_wrap=_shoup_kernel,
            input_structs=[
                (shape, jnp.uint64), # rhs
            ],
            parameters={"context": context, "lhs": lhs, "lhs_pre": lhs_pre}
        )

        profiler_instance.add_profile(
            name=kernel_name,
            kernel_wrapper=kernel_wrapper,
            kernel_setting_cols={
                "batch": batch,
                "algorithm": "shoup",
                "moduli_count": len(moduli)
            },
        )
    
    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()

  # @absltest.skip("skip this test for now")
  def test_perf_bat_lazy(self):
    print("Testing BAT Lazy Performance")
    element = 2
    r = 256
    c = 256
    precision = 31
    moduli = util.find_moduli_ntt(51, precision, 16)
    context = ff_context.BATLazyContext(moduli)

    profiler_instance = Profiler(
          output_trace_path=self.output_trace_root,
          profile_naming=f"{self._testMethodName}",
          configuration=self.profiler_config,
    )

    for batch in BATCH_SIZES:
        print(f"Running BATLazy for batch size: {batch}")
        shape = (batch, element, r, c, len(moduli))
        kernel_name = f"BATLazy_Perf_Batch_{batch}"

        # Prepare static parameters
        key = jax.random.PRNGKey(0)
        lhs = jax.random.randint(key, shape, 0, 0xFFFFFFFF, dtype=jnp.uint32)

        kernel_wrapper = KernelWrapper(
            kernel_name=kernel_name,
            function_to_wrap=_bat_lazy_kernel,
            input_structs=[
                (shape, jnp.uint32), # rhs
            ],
            parameters={"context": context, "lhs": lhs}
        )

        profiler_instance.add_profile(
            name=kernel_name,
            kernel_wrapper=kernel_wrapper,
            kernel_setting_cols={
                "batch": batch,
                "algorithm": "bat_lazy",
                "moduli_count": len(moduli)
            },
        )
    
    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()


if __name__ == "__main__":
  absltest.main()

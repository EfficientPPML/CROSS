import os
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized
import bconv
import util
from profiler import KernelWrapper, Profiler, collect_logs

# Use 64-bit precision as in bconv.py
jax.config.update("jax_enable_x64", True)

# name, limb_in, limb_out, degree
PERF_TEST_PARAMS = [
    ("0", 12, 28, 65536), # 18 limbs in, 45 limbs out, 65536 degree
    ("1", 12, 36, 65536),
    ("2", 16, 40, 65536),
    ("3", 20, 48, 65536),
    ("4", 24, 56, 65536),
]

def _jax_bconv_bat_kernel(data_in, parameters):
  return parameters["bconv"].basis_change_bat(data_in)

def _jax_bconv_kernel(data_in, parameters):
  return parameters["bconv"].basis_change(data_in)

class BConvPerformanceTest(parameterized.TestCase):
  
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
    collect_logs(root_dir, output_csv_name="bconv_profiling")

  def _create_kernel_wrapper(self, kernel_name, function_to_wrap, batch, degree, limb_in, bconv_obj):
    # Input shape: (batch, degree, limb_in) based on original test usage
    # data_in = jax.random.randint(key, (batch, degree, limb_in), ...)
    input_shape = (batch, degree, limb_in)
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=function_to_wrap,
        input_structs=[
            (input_shape, jnp.uint32),
        ],
        parameters={"bconv": bconv_obj},
    )

  # @absltest.skip("test a single implementation")
  @parameterized.named_parameters(*PERF_TEST_PARAMS)
  def test_BConvBat(self, limb_in, limb_out, degree):
    
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"{self._testMethodName}_N{degree}",
        configuration=self.profiler_config,
    )

    # Setup bconv object
    limb_in_modulus = util.find_moduli_ntt(limb_in, 28, degree)
    limb_out_modulus = util.find_moduli_ntt(limb_out, 28, degree)
    overall_moduli = limb_in_modulus + limb_out_modulus
    in_indices = list(range(len(limb_in_modulus)))
    out_indices = list(range(len(limb_in_modulus), len(overall_moduli)))

    _bconv = bconv.BConvBarrett(overall_moduli)
    _bconv.control_gen([(in_indices, out_indices)], perf_test=True)

    for batch in [1, 8, 16, 32, 64, 128, 256]:
      print(f"Running for batch size: {batch}")
      
      kernel_name = f"{self._testMethodName}_b{batch}"
      
      kernel_wrapper = self._create_kernel_wrapper(
          kernel_name=kernel_name,
          function_to_wrap=_jax_bconv_bat_kernel,
          batch=batch,
          degree=degree,
          limb_in=limb_in,
          bconv_obj=_bconv
      )
      
      profiler_instance.add_profile(
          name=kernel_name,
          kernel_wrapper=kernel_wrapper,
          kernel_setting_cols={
              "degree": degree,
              "limb_in": limb_in,
              "limb_out": limb_out,
              "batch": batch,
              "kernel_type": "bat"
          },
      )
    
    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()

  @parameterized.named_parameters(*PERF_TEST_PARAMS)
  def test_basis_change(self, limb_in, limb_out, degree):
    
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"{self._testMethodName}_N{degree}",
        configuration=self.profiler_config,
    )

    # Setup bconv object
    limb_in_modulus = util.find_moduli_ntt(limb_in, 28, degree)
    limb_out_modulus = util.find_moduli_ntt(limb_out, 28, degree)
    overall_moduli = limb_in_modulus + limb_out_modulus
    in_indices = list(range(len(limb_in_modulus)))
    out_indices = list(range(len(limb_in_modulus), len(overall_moduli)))

    _bconv = bconv.BConvBarrett(overall_moduli)
    _bconv.control_gen([(in_indices, out_indices)], perf_test=True)

    for batch in [1, 8, 16, 32, 64, 128, 256]:
      print(f"Running for batch size: {batch}")
      
      kernel_name = f"{self._testMethodName}_b{batch}"
      
      kernel_wrapper = self._create_kernel_wrapper(
          kernel_name=kernel_name,
          function_to_wrap=_jax_bconv_kernel,
          batch=batch,
          degree=degree,
          limb_in=limb_in,
          bconv_obj=_bconv
      )
      
      profiler_instance.add_profile(
          name=kernel_name,
          kernel_wrapper=kernel_wrapper,
          kernel_setting_cols={
              "degree": degree,
              "limb_in": limb_in,
              "limb_out": limb_out,
              "batch": batch,
              "kernel_type": "default"
          },
      )
    
    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()


if __name__ == "__main__":
    absltest.main()

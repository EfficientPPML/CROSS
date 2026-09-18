import os
import csv
import collections
import re
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
batch_list = [1, 8, 16, 32, 64, 128, 256]

def analyze_bconv_speedup(csv_file):
    # Store data: data[param_index][batch][kernel_type] = latency
    data = collections.defaultdict(lambda: collections.defaultdict(dict))

    # Parameters parameters mapping
    params_map = {p[0]: (p[1], p[2], p[3]) for p in PERF_TEST_PARAMS}

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                op_name = row['operation_name']
                # Parse op_name to get index and batch
                # Format: test_basis_change_{index}_b{batch} or test_BConvBat_{index}_b{batch}

                if "test_basis_change" in op_name:
                    match = re.search(r'test_basis_change_(\d+)_b(\d+)', op_name)
                    if match:
                        idx = match.group(1)
                        batch = int(match.group(2))
                        latency = float(row['sample_0'])
                        data[idx][batch]['basis_change'] = latency

                elif "test_BConvBat" in op_name:
                    match = re.search(r'test_BConvBat_(\d+)_b(\d+)', op_name)
                    if match:
                        idx = match.group(1)
                        batch = int(match.group(2))
                        latency = float(row['sample_0'])
                        data[idx][batch]['BConvBat'] = latency

    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
        return

    print(f"{'Index':<5} {'Limb In':<8} {'Limb Out':<8} {'Batch':<6} {'Basis Change (ms)':<18} {'BConvBat (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    single_batch_highlights = []

    sorted_indices = sorted(data.keys(), key=lambda x: int(x))

    for idx in sorted_indices:
        if idx not in params_map:
            continue

        limb_in, limb_out, _ = params_map[idx]
        batches = sorted(data[idx].keys())

        for batch in batches:
            timings = data[idx][batch]
            if 'basis_change' in timings and 'BConvBat' in timings:
                bc_time = timings['basis_change']
                bb_time = timings['BConvBat']
                speedup = bc_time / bb_time

                print(f"{idx:<5} {limb_in:<8} {limb_out:<8} {batch:<6} {bc_time:<18.4f} {bb_time:<15.4f} {speedup:<10.2f}x")

                if batch == 1:
                    single_batch_highlights.append({
                        'index': idx,
                        'limb_in': limb_in,
                        'limb_out': limb_out,
                        'bc_time': bc_time,
                        'bb_time': bb_time,
                        'speedup': speedup
                    })

    print("\\n" + "="*80)
    print("SINGLE BATCH LATENCY COMPARISON HIGHLIGHTS")
    print("="*80)
    print(f"{'Index':<5} {'Limb In':<8} {'Limb Out':<8} {'Basis Change (ms)':<18} {'BConvBat (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    for item in single_batch_highlights:
        print(f"{item['index']:<5} {item['limb_in']:<8} {item['limb_out']:<8} {item['bc_time']:<18.4f} {item['bb_time']:<15.4f} {item['speedup']:<10.2f}x")

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

    print("\nAnalyzing BConv Speedup...")
    csv_path = os.path.join(root_dir, "bconv_profiling.csv")
    analyze_bconv_speedup(csv_path)

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

  @parameterized.named_parameters(*PERF_TEST_PARAMS)
  # @absltest.skip("test a single implementation")
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
    # Using perf_test=True for potentially faster setup/mock constants if supported,
    # though here we are providing real moduli so it might handle it or we stick to False if we want real math verification.
    # However, for pure performance profiling of the kernel execution, perf_test=True is often preferred to skip expensive precomputes.
    # The prompt implies alignment with add_perf_test which has perf_test params.
    _bconv.control_gen([(in_indices, out_indices)], perf_test=True)

    for batch in batch_list:
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

    for batch in batch_list:
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

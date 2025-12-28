"""A module for operations on test CKKS evaluation kernels including.

- Subtraction logic performance test
"""
# Standard library imports
import os

# Third-party imports
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

# Local imports
import util
from profiler import KernelWrapper, Profiler, collect_logs

# JAX configuration
jax.config.update("jax_enable_x64", True)

# D=65536, HB=1 (Batch=2), M=51
TEST_PARAMS = [
    (
        'SubLogic', 65536, 1, 51
    )
]

def _jax_sub_where(current_approx_down_in_ad, tower_new_basis_jax_ad, parameters):
    """Kernel wrapper for version 1: using jnp.where"""
    overall_moduli_jax_ad = parameters["overall_moduli_jax_ad"]
    sub_result_ad = jnp.where(
        current_approx_down_in_ad < tower_new_basis_jax_ad,
        current_approx_down_in_ad + overall_moduli_jax_ad - tower_new_basis_jax_ad,
        current_approx_down_in_ad - tower_new_basis_jax_ad,
    )
    return sub_result_ad

def _jax_sub_static(current_approx_down_in_ad, tower_new_basis_jax_ad, parameters):
    """Kernel wrapper for version 2: using static computation"""
    overall_moduli_jax_ad = parameters["overall_moduli_jax_ad"]
    diff_ad = current_approx_down_in_ad + overall_moduli_jax_ad - tower_new_basis_jax_ad
    pred_ad = current_approx_down_in_ad >= tower_new_basis_jax_ad
    # Note: original code had diff_ad - (moduli * pred)
    # The 'pred_ad' is boolean. 'moduli * pred' effectively selects using arithmetic.
    sub_result_ad = diff_ad - (overall_moduli_jax_ad * pred_ad)
    return sub_result_ad


class PerformanceTest(parameterized.TestCase):
    """A base class for running subtraction logic tests."""

    def __init__(self, *args, **kwargs):
        super(PerformanceTest, self).__init__(*args, **kwargs)
        self.random_key = jax.random.key(0)

    def setUp(self):
        super().setUp()
        self.output_trace_root = os.path.join(os.path.dirname(__file__), "log")
        self.profiler_config = {
            "iterations": 2, # Run enough iterations to get stable numbers, add_perf_test uses 1 but maybe we want more for microbench
            "save_to_file": True,
        }

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        # Call collect_logs at the end of the test class execution
        root_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Collecting logs from: {root_dir}")
        collect_logs(root_dir, output_csv_name="sub_profiling")

    def _create_kernel_wrapper(self, kernel_name, func, shape, moduli):
        # inputs are: current_approx, tower_new_basis
        # parameters: overall_moduli

        # We pass overall_moduli as a parameter to the wrapper so it's treated as static/constant if needed,
        # or we can pass it as dynamic input. In the real code it's self.overall_moduli... so it's likely a constant/static from the class.
        # However, Profiler usually handles arrays in 'parameters' as JAX params.
        # Let's pass it as a parameter to match 'add_perf_test' style where modulus is a param.

        return KernelWrapper(
            kernel_name=kernel_name,
            function_to_wrap=func,
            input_structs=[
                (shape, jnp.uint32), # current_approx_down_in_ad
                (shape, jnp.uint32), # tower_new_basis_jax_ad
            ],
            parameters={"overall_moduli_jax_ad": jnp.array(moduli, dtype=jnp.uint32).reshape(1, 1, 1, 1, -1)},
            # Reshape moduli to match broadcasting (1, 1, 1, 1, M) to (HB*2, 1, 1, D, M) ??
            # The user code had: self.overall_moduli_jax_ad - tower_new_basis_jax_ad
            # shape of current is (2*HB, 1, 1, D, M).
            # We will rely on broadcasting.
        )

    @parameterized.named_parameters(*TEST_PARAMS)
    def test_sub_logic(self, degree, hb, m_targ):
        """Test subtraction logic performance."""
        print("generate data")

        # Shape: (2*HB, 1, 1, D, M_targ)
        batch_dim = 2 * hb
        shape = (batch_dim, 1, 1, degree, m_targ)

        # Generate some dummy moduli
        # We don't strictly need valid primes for performance testing, just numbers.
        # But let's use some likely sizes.
        moduli = [268042241] * m_targ # Just reuse one

        profiler_instance = Profiler(
            output_trace_path=self.output_trace_root,
            profile_naming=f"{self._testMethodName}",
            configuration=self.profiler_config,
        )

        print(f"Running for shape: {shape}")

        # Version 1: Where
        kernel_name_v1 = "sub_where"
        wrapper_v1 = self._create_kernel_wrapper(
            kernel_name_v1,
            _jax_sub_where,
            shape,
            moduli
        )
        profiler_instance.add_profile(
            name=kernel_name_v1,
            kernel_wrapper=wrapper_v1,
            kernel_setting_cols={
                "variant": "where",
                "D": degree,
                "M": m_targ
            },
        )

        # Version 2: Static
        kernel_name_v2 = "sub_static"
        wrapper_v2 = self._create_kernel_wrapper(
            kernel_name_v2,
            _jax_sub_static,
            shape,
            moduli
        )
        profiler_instance.add_profile(
            name=kernel_name_v2,
            kernel_wrapper=wrapper_v2,
            kernel_setting_cols={
                "variant": "static",
                "D": degree,
                "M": m_targ
            },
        )

        profiler_instance.profile_all_profilers()
        profiler_instance.post_process_all_profilers()


if __name__ == "__main__":
  absltest.main()

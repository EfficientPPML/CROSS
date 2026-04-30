"""Performance tests for ptct_mul.py using profiler.py."""

import os

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

import ptct_mul
import util
from profiler import KernelWrapper, Profiler, collect_logs

jax.config.update("jax_enable_x64", True)

BarrettContext = ptct_mul.BarrettContext
HEPtCtMul = ptct_mul.HEPtCtMul

BATCH_SIZE_LIST_LOW_DEGREE = [4, 16, 64]
BATCH_SIZE_LIST_HIGH_DEGREE = [1, 2, 4]
TEST_PARAMS_PTCT_MUL = [
    ("2_12_L4", 4096, 4, BATCH_SIZE_LIST_LOW_DEGREE),
    ("2_14_L8", 16384, 8, BATCH_SIZE_LIST_HIGH_DEGREE),
]

DEGREE_TO_RC_MAPPING = {
    65536: (128, 512),
    32768: (128, 256),
    16384: (128, 128),
    8192: (128, 64),
    4096: (128, 32),
    2048: (128, 16),
}

def _ptct_mul_vpu_kernel(ct_data, parameters):
    product = ct_data.astype(jnp.uint64) * parameters["pt_ntt"].astype(jnp.uint64)
    reduced = parameters["barrett_ctx"].modular_reduction(product)
    return reduced.astype(jnp.uint32)


def _ptct_mul_bat_kernel(ct_data, parameters):
    ct_bytes = jax.lax.bitcast_convert_type(ct_data, jnp.uint8)
    partial = jnp.einsum(
        "bercmq, rcmqp -> bercmp",
        ct_bytes,
        parameters["pt_bat_per_modulus"],
        preferred_element_type=jnp.uint32,
    )
    result_u64 = jnp.sum(
        partial.astype(jnp.uint64) << parameters["shift_factors"],
        axis=-1,
    )
    reduced = parameters["barrett_ctx"].modular_reduction(result_u64)
    return reduced.astype(jnp.uint32)


class PtCtMulPerformanceTest(parameterized.TestCase):
    """Profiles VPU and BAT ptct_mul implementations."""

    def setUp(self):
        super().setUp()
        self.assertEqual(
            jax.devices()[0].platform,
            "tpu",
            msg=f"ptct_mul perf tests require TPU; found {jax.devices()}",
        )
        self.output_trace_root = os.path.join(os.path.dirname(__file__), "log")
        self.profiler_config = {
            "iterations": 1,
            "save_to_file": True,
        }

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if jax.devices()[0].platform != "tpu":
            print("Skipping ptct_mul log collection: TPU backend is required.")
            return
        root_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"Collecting logs from: {root_dir}")
        collect_logs(root_dir, output_csv_name="ptct_mul_profiling")

    def _create_kernel_wrapper(
        self,
        kernel_name,
        kernel_fn,
        kernel_parameters,
        batch,
        num_elements,
        rows,
        cols,
        num_moduli,
    ):
        input_shape = (batch, num_elements, rows, cols, num_moduli)
        return KernelWrapper(
            kernel_name=kernel_name,
            function_to_wrap=kernel_fn,
            input_structs=[(input_shape, jnp.uint32)],
            parameters=kernel_parameters,
        )

    def _create_vpu_parameters(self, moduli, pt_ntt):
        return {
            "barrett_ctx": BarrettContext(moduli=moduli),
            "pt_ntt": pt_ntt.astype(jnp.uint32),
        }

    def _create_bat_parameters(self, moduli, rows, cols, pt_ntt):
        helper = HEPtCtMul(
            batch=1,
            r=rows,
            c=cols,
            moduli=moduli,
            degree_layout=(rows, cols),
        )
        helper.precompute_plaintext_bat(pt_ntt)
        return {
            "barrett_ctx": BarrettContext(moduli=moduli),
            "pt_bat_per_modulus": helper.pt_bat.reshape(rows, cols, len(moduli), 4, 4),
            "shift_factors": jnp.array([0, 8, 16, 24], dtype=jnp.uint32),
        }

    def _profile_context(self, profile_prefix, degree, num_limbs, batch_size_list, use_bat):
        rows, cols = DEGREE_TO_RC_MAPPING[degree]
        num_elements = 2
        moduli = util.moduli_28_list[degree][:num_limbs]

        profiler_instance = Profiler(
            output_trace_path=self.output_trace_root,
            profile_naming=f"{profile_prefix}_degree_{degree}",
            configuration=self.profiler_config,
        )

        pt_ntt = util.random_parameters((rows, cols, num_limbs), moduli, dtype=jnp.uint32)
        kernel_fn = _ptct_mul_bat_kernel if use_bat else _ptct_mul_vpu_kernel
        kernel_parameters = (
            self._create_bat_parameters(moduli, rows, cols, pt_ntt)
            if use_bat
            else self._create_vpu_parameters(moduli, pt_ntt)
        )

        for batch in batch_size_list:
            kernel_name = f"{profile_prefix}_batch_{batch}"
            kernel_wrapper = self._create_kernel_wrapper(
                kernel_name=kernel_name,
                kernel_fn=kernel_fn,
                kernel_parameters=kernel_parameters,
                batch=batch,
                num_elements=num_elements,
                rows=rows,
                cols=cols,
                num_moduli=num_limbs,
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
                    "num_elements": num_elements,
                    "implementation": "bat" if use_bat else "vpu",
                },
            )

        profiler_instance.profile_all_profilers()
        profiler_instance.post_process_all_profilers()

    @parameterized.named_parameters(*TEST_PARAMS_PTCT_MUL)
    def test_ptct_mul_vpu_performance(self, degree, num_limbs, batch_size_list):
        self._profile_context(
            profile_prefix="ptct_mul_vpu",
            degree=degree,
            num_limbs=num_limbs,
            batch_size_list=batch_size_list,
            use_bat=False,
        )

    @parameterized.named_parameters(*TEST_PARAMS_PTCT_MUL)
    def test_ptct_mul_bat_performance(self, degree, num_limbs, batch_size_list):
        self._profile_context(
            profile_prefix="ptct_mul_bat",
            degree=degree,
            num_limbs=num_limbs,
            batch_size_list=batch_size_list,
            use_bat=True,
        )


if __name__ == "__main__":
    absltest.main()

"""Performance tests for ptct_mul.py using profiler.py."""

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

import finite_field
import ptct_mul
import util
from polynomial import Polynomial
from profiler import KernelWrapper, Profiler, collect_module_logs
from profiler import kernel_perf_setup, require_tpu

jax.config.update("jax_enable_x64", True)

BarrettContext = ptct_mul.BarrettContext
MontgomeryContext = finite_field.MontgomeryContext
_HEPtCtMulKernel = ptct_mul._HEPtCtMulKernel

# Profile both modular-reduction backends so Barrett vs Montgomery cost is
# directly comparable in the collected CSV (distinguished by the "reduction"
# setting column). Montgomery requires all moduli < 2^31.
REDUCTION_CONTEXTS = [
    ("barrett", BarrettContext),
    ("montgomery", MontgomeryContext),
]

TEST_PARAMS_PTCT_MUL = [
    # CL_256_256: degree 65536 tiled as r=c=256, 51 Q limbs -- matches the
    # config profiled for hemul/herot/rescale/matvec so the pt*ct cost is
    # directly comparable. batch=1 for the apples-to-apples single-op number.
    ("CL_256_256", 65536, 51, [1]),
]

DEGREE_TO_RC_MAPPING = {
    65536: (256, 256),
    32768: (128, 256),
    16384: (128, 128),
    8192: (128, 64),
    4096: (128, 32),
    2048: (128, 16),
}

def _ptct_mul_kernel(ct_data, parameters):
    """Profile the private pt-ct kernel (VPU or BAT) under its configured
    reduction backend, so the Montgomery strictify and plaintext encoding are
    exactly what ship in ptct_mul.py."""
    op = parameters["ptct_op"]
    ct = parameters["ct_input"]
    ct.polynomial = ct_data
    return op.mul(ct, use_bat=parameters["use_bat"]).polynomial


class PtCtMulPerformanceTest(parameterized.TestCase):
    """Profiles VPU and BAT ptct_mul implementations."""

    def setUp(self):
        super().setUp()
        require_tpu(self, "ptct_mul")
        self.output_trace_root, self.profiler_config = kernel_perf_setup(
            __file__
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        collect_module_logs(__file__, "ptct_mul_profiling", tpu_only=True)

    def _build_ptct_op(self, ff_ctx_cls, moduli, rows, cols, num_limbs, use_bat):
        """Construct the private pt-ct kernel with the reduction backend and set
        its plaintext (standard-form; set_plaintext/precompute_plaintext_bat
        Montgomery-encode it internally when needed)."""
        op = _HEPtCtMulKernel(
            batch=1,
            r=rows,
            c=cols,
            moduli=moduli,
            degree_layout=(rows, cols),
            finite_field_context=ff_ctx_cls,
        )
        pt_ntt = util.random_parameters(
            (rows, cols, num_limbs), moduli, dtype=jnp.uint32
        )
        if use_bat:
            op.precompute_plaintext_bat(pt_ntt)
        else:
            op.set_plaintext(pt_ntt)
        return op

    def _profile_context(self, profile_prefix, degree, num_limbs, batch_size_list, use_bat):
        rows, cols = DEGREE_TO_RC_MAPPING[degree]
        num_elements = 2
        moduli = util.moduli_28_list[degree][:num_limbs]

        profiler_instance = Profiler(
            output_trace_path=self.output_trace_root,
            profile_naming=f"{profile_prefix}_degree_{degree}",
            configuration=self.profiler_config,
        )

        for red_name, ff_ctx_cls in REDUCTION_CONTEXTS:
            op = self._build_ptct_op(
                ff_ctx_cls, moduli, rows, cols, num_limbs, use_bat
            )
            for batch in batch_size_list:
                # Input must be encoded with the op's reduction context; the entry guard rejects a mismatch.
                ct_input = Polynomial(
                    {
                        "batch": batch,
                        "num_elements": num_elements,
                        "degree": rows * cols,
                        "precision": 32,
                        "num_moduli": num_limbs,
                        "degree_layout": (rows, cols),
                    },
                    parameters={"moduli": moduli, "finite_field_context": ff_ctx_cls},
                )
                kernel_name = f"{profile_prefix}_{red_name}_batch_{batch}"
                kernel_wrapper = KernelWrapper(
                    kernel_name=kernel_name,
                    function_to_wrap=_ptct_mul_kernel,
                    input_structs=[
                        ((batch, num_elements, rows, cols, num_limbs), jnp.uint32)
                    ],
                    parameters={
                        "ptct_op": op,
                        "ct_input": ct_input,
                        "use_bat": use_bat,
                    },
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
                        "reduction": red_name,
                    },
                )

        profiler_instance.run()

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

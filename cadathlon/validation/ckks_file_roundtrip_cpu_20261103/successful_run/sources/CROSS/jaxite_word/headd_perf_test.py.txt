"""Performance tests for headd.py using profiler.py."""

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

import finite_field
import headd
from polynomial import Polynomial
from profiler import KernelWrapper, Profiler, collect_module_logs
from profiler import kernel_perf_setup, require_tpu

jax.config.update("jax_enable_x64", True)

# Profile both modular-reduction backends so Barrett vs Montgomery cost is
# directly comparable in the collected CSV (distinguished by the "reduction"
# setting column). Montgomery requires all moduli < 2^31.
REDUCTION_CONTEXTS = [
    ("barrett", finite_field.BarrettContext),
    ("montgomery", finite_field.MontgomeryContext),
]

TEST_PARAMS = [
    (
      'CL_256_256', 65536, 256, 256, 51, 3,
      [349962241, 306708481, 304218113, 347996161, 319291393, 347078657, 323223553, 337248257, 323878913, 336855041, 329515009, 332660737, 329777153, 335413249, 325844993, 330301441, 327548929, 332267521, 328728577, 344850433, 336068609, 340000769, 261488641, 302252033, 297664513, 299499521, 261881857, 295305217, 263323649, 277086209, 263454721, 292159489, 279838721, 291373057, 284950529, 290455553, 281935873, 285474817, 283508737, 288882689, 264634369, 276430849, 270532609, 274726913, 272760833, 276037633, 265420801, 270794753, 268042241, 269221889, 786433],
      [260702209, 260571137, 258605057, 257949697, 256770049, 256376833, 254279681, 253493249, 253100033, 249561089, 246415361, 245760001, 245235713, 244973569, 244842497, 241827841, 240648193, 239861761],
      True
    ),
    # (
    #   'test', 16, 4, 4, 51, 3,
    #   [167772161, 125042689, 113246209, 120586241, 115081217, 123863041, 115212289, 120324097, 106037249, 117964801, 114032641, 115998721, 107216897, 113115137, 111280129, 112066561, 111149057, 167510017, 126222337, 167116801, 145489921, 165019649, 150863873, 164233217, 126615553, 163577857, 151257089, 160038913, 151388161, 159645697, 155189249, 158334977, 152174593, 156499969, 154533889, 158072833, 127664129, 149815297, 142344193, 147849217, 135135233, 144310273, 127795201, 141557761, 136314881, 140771329, 130809857, 138412033, 132120577, 134348801, 786433],
    #   [268042241, 265420801, 264634369, 263454721, 263323649, 261881857, 261488641, 260702209, 260571137, 258605057, 257949697, 256770049, 256376833, 254279681, 253493249, 253100033, 249561089],
    #    True
    # ),
]


def _headd_kernel(ct_data, parameters):
  """Profile the private add kernel under its reduction backend."""
  op = parameters["headd_op"]
  ct1 = parameters["ct1"]
  ct2 = parameters["ct2"]
  ct1.polynomial = ct_data
  ct2.polynomial = ct_data
  return op.add(ct1, ct2).polynomial


class HEAddKernelPerformanceTest(parameterized.TestCase):
  """Profiles the private add kernel."""

  def setUp(self):
    super().setUp()
    require_tpu(self, "headd")
    self.output_trace_root, self.profiler_config = kernel_perf_setup(__file__)

  @classmethod
  def tearDownClass(cls):
    super().tearDownClass()
    collect_module_logs(__file__, "headd_profiling", tpu_only=True)

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_headd(self, degree, r, c, limbs, dnum, moduli, extend_moduli, perf_test):
    num_elements = 2
    degree_layout = (r, c)

    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"headd_N{degree}",
        configuration=self.profiler_config,
    )

    batch_list = [1]
    for red_name, ff_ctx_cls in REDUCTION_CONTEXTS:
      op = headd._HEAddKernel(moduli, finite_field_context=ff_ctx_cls)
      for batch in batch_list:
        # Input must carry the op's reduction context; the entry guard rejects a mismatch.
        ct_params = {"moduli": moduli, "finite_field_context": ff_ctx_cls}
        ct_shapes = {
            "batch": batch, "num_elements": num_elements, "degree": r * c,
            "precision": 32, "num_moduli": limbs, "degree_layout": degree_layout,
        }
        ct1 = Polynomial(ct_shapes, parameters=ct_params)
        ct2 = Polynomial(ct_shapes, parameters=ct_params)
        kernel_name = f"headd_{red_name}_B{batch}"
        kernel_wrapper = KernelWrapper(
            kernel_name=kernel_name,
            function_to_wrap=_headd_kernel,
            input_structs=[((batch, num_elements, r, c, limbs), jnp.uint32)],
            parameters={"headd_op": op, "ct1": ct1, "ct2": ct2},
        )
        profiler_instance.add_profile(
            name=kernel_name,
            kernel_wrapper=kernel_wrapper,
            kernel_setting_cols={
                "degree": degree,
                "num_limbs": limbs,
                "r": r,
                "c": c,
                "batch": batch,
                "num_elements": num_elements,
                "reduction": red_name,
            },
        )

    profiler_instance.run()


if __name__ == "__main__":
  absltest.main()

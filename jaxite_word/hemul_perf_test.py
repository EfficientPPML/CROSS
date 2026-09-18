"""A module for operations on test CKKS evaluation kernels including.

- HEMul
"""
# Standard library imports
import math
import random
import functools
import os

# Third-party imports
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized
from hemul import HEMul
from polynomial import Polynomial
from profiler import KernelWrapper, Profiler, collect_logs
import util

# JAX configuration
# os.environ['JAX_PLATFORM_NAME'] = 'cpu'
jax.config.update("jax_enable_x64", True)

TEST_PARAMS=[
              (
                'HEAP', 8192, 128, 64, 8, 3,
                [269402113, 268271617, 269221889, 268664833, 268861441, 268369921, 268582913, 557057],
                [268238849, 268189697, 268091393, 268042241],
                True,
              ),
              (
                'FIDESlib_128_512', 65536, 128, 512, 60, 3,
                [167903233,99483649,102629377,100139009,101711873,95027201,98697217,95813633,167772161,118751233,125042689,113246209,120586241,115081217,123863041,115212289,120324097,106037249,117964801,114032641,115998721,107216897,113115137,111280129,112066561,111149057,167510017,126222337,167116801,145489921,165019649,150863873,164233217,126615553,163577857,151257089,160038913,151388161,159645697,155189249,158334977,152174593,156499969,154533889,158072833,127664129,149815297,142344193,147849217,135135233,144310273,127795201,141557761,136314881,140771329,130809857,138412033,132120577,134348801,786433],
                [268042241,265420801,264634369,263454721,263323649,261881857,261488641,260702209,260571137,258605057,257949697,256770049,256376833,254279681,253493249,253100033,249561089,246415361,245760001,245235713],
                True,
              ),
              ( # r, c = 512, 128 is less efficient that r, c = 128, 512
                'Cheddar3_128_512', 65536, 128, 512, 48, 3,
                [347996161, 319291393, 347078657, 323223553, 337248257, 323878913, 336855041, 329515009, 332660737, 329777153, 335413249, 325844993, 330301441, 327548929, 332267521, 328728577, 344850433, 336068609, 340000769, 261488641, 302252033, 297664513, 299499521, 261881857, 295305217, 263323649, 277086209, 263454721, 292159489, 279838721, 291373057, 284950529, 290455553, 281935873, 285474817, 283508737, 288882689, 264634369, 276430849, 270532609, 274726913, 272760833, 276037633, 265420801, 270794753, 268042241, 269221889, 786433],
                [260702209,260571137,258605057,257949697,256770049,256376833,254279681,253493249,253100033,249561089,246415361,245760001,245235713,244973569,244842497,241827841,240648193],
                True,
              ),
              (
                'Cheddar3_256_256', 65536, 256, 256, 48, 3,
                [347996161, 319291393, 347078657, 323223553, 337248257, 323878913, 336855041, 329515009, 332660737, 329777153, 335413249, 325844993, 330301441, 327548929, 332267521, 328728577, 344850433, 336068609, 340000769, 261488641, 302252033, 297664513, 299499521, 261881857, 295305217, 263323649, 277086209, 263454721, 292159489, 279838721, 291373057, 284950529, 290455553, 281935873, 285474817, 283508737, 288882689, 264634369, 276430849, 270532609, 274726913, 272760833, 276037633, 265420801, 270794753, 268042241, 269221889, 786433],
                [260702209,260571137,258605057,257949697,256770049,256376833,254279681,253493249,253100033,249561089,246415361,245760001,245235713,244973569,244842497,241827841,240648193],
                True,
              ),
              (
                'BASALISC_128_512', 65536, 128, 512, 64, 3,
                [384040961, 371589121, 383778817, 377880577, 379453441, 323092481, 351797249, 349962241, 351404033, 260702209, 308150273, 304742401, 307888129, 302776321, 306708481, 304218113, 347996161, 319291393, 347078657, 323223553, 337248257, 323878913, 336855041, 329515009, 332660737, 329777153, 335413249, 325844993, 330301441, 327548929, 332267521, 328728577, 344850433, 336068609, 340000769, 261488641, 302252033, 297664513, 299499521, 261881857, 295305217, 263323649, 277086209, 263454721, 292159489, 279838721, 291373057, 284950529, 290455553, 281935873, 285474817, 283508737, 288882689, 264634369, 276430849, 270532609, 274726913, 272760833, 276037633, 265420801, 270794753, 268042241, 269221889, 786433],
                [260571137, 258605057, 257949697, 256770049, 256376833, 254279681, 253493249, 253100033, 249561089, 246415361, 245760001, 245235713, 244973569, 244842497, 241827841, 240648193, 239861761, 239337473, 236716033, 236584961, 235798529, 234356737, 232652801],
                True,
              ),
              ( # 128, 512 is less efficient for FAB configuration.
                'FAB_512_128', 65536, 512, 128, 64, 4,
                [167903233,87293953,89522177,85590017,125698049,99483649,102629377,100139009,101711873,95027201,98697217,95813633,167772161,118751233,125042689,113246209,120586241,115081217,123863041,115212289,120324097,106037249,117964801,114032641,115998721,107216897,113115137,111280129,112066561,111149057,167510017,126222337,167116801,145489921,165019649,150863873,164233217,126615553,163577857,151257089,160038913,151388161,159645697,155189249,158334977,152174593,156499969,154533889,158072833,127664129,149815297,142344193,147849217,135135233,144310273,127795201,141557761,136314881,140771329,130809857,138412033,132120577,134348801,786433],
                [268042241,265420801,264634369,263454721,263323649,261881857,261488641,260702209,260571137,258605057,257949697,256770049,256376833,254279681,253493249,253100033],
                True
              ),
              (
                'FAB_256_256', 65536, 256, 256, 64, 4,
                [167903233,87293953,89522177,85590017,125698049,99483649,102629377,100139009,101711873,95027201,98697217,95813633,167772161,118751233,125042689,113246209,120586241,115081217,123863041,115212289,120324097,106037249,117964801,114032641,115998721,107216897,113115137,111280129,112066561,111149057,167510017,126222337,167116801,145489921,165019649,150863873,164233217,126615553,163577857,151257089,160038913,151388161,159645697,155189249,158334977,152174593,156499969,154533889,158072833,127664129,149815297,142344193,147849217,135135233,144310273,127795201,141557761,136314881,140771329,130809857,138412033,132120577,134348801,786433],
                [268042241,265420801,264634369,263454721,263323649,261881857,261488641,260702209,260571137,258605057,257949697,256770049,256376833,254279681,253493249,253100033],
                True
              ),
              (
                'Wrapdrive_128_512', 65536, 128, 512, 36, 3,
                [347078657,325844993,330301441,327548929,332267521,328728577,344850433,336068609,340000769,261488641,302252033,297664513,299499521,261881857,295305217,263323649,277086209,263454721,292159489,279838721,291373057,284950529,290455553,281935873,285474817,283508737,288882689,264634369,276430849,270532609,274726913,272760833,276037633,265420801,270794753,786433],
                [260702209, 260571137, 258605057, 257949697, 256770049, 256376833, 254279681, 253493249, 253100033, 249561089, 246415361, 245760001, 245235713, 244973569],
                True
              ),
              (
                'Wrapdrive_256_256', 65536, 256, 256, 36, 3,
                [347078657,325844993,330301441,327548929,332267521,328728577,344850433,336068609,340000769,261488641,302252033,297664513,299499521,261881857,295305217,263323649,277086209,263454721,292159489,279838721,291373057,284950529,290455553,281935873,285474817,283508737,288882689,264634369,276430849,270532609,274726913,272760833,276037633,265420801,270794753,786433],
                [260702209, 260571137, 258605057, 257949697, 256770049, 256376833, 254279681, 253493249, 253100033, 249561089, 246415361, 245760001, 245235713, 244973569],
                True
              ),
              (
                'CL_128_512', 65536, 128, 512, 51, 3,
                [349962241, 306708481, 304218113, 347996161, 319291393, 347078657, 323223553, 337248257, 323878913, 336855041, 329515009, 332660737, 329777153, 335413249, 325844993, 330301441, 327548929, 332267521, 328728577, 344850433, 336068609, 340000769, 261488641, 302252033, 297664513, 299499521, 261881857, 295305217, 263323649, 277086209, 263454721, 292159489, 279838721, 291373057, 284950529, 290455553, 281935873, 285474817, 283508737, 288882689, 264634369, 276430849, 270532609, 274726913, 272760833, 276037633, 265420801, 270794753, 268042241, 269221889, 786433],
                [260702209, 260571137, 258605057, 257949697, 256770049, 256376833, 254279681, 253493249, 253100033, 249561089, 246415361, 245760001, 245235713, 244973569, 244842497, 241827841, 240648193, 239861761],
                True
              ),
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
              # (
              #   'test2', 32, 8, 4, 51, 3,
              #   [268460033, 268460161, 268459777, 268457153, 268459649, 268453441, 268457857, 268457537, 268456193, 268454209, 268455553, 268454657, 268455809, 268455169, 268451329, 268440449, 268450817, 268425089, 268450241, 268426049, 268449409, 268428161, 268449281, 268440577, 268445057, 268440833, 268444993, 268429633, 268443841, 268437889, 268443713, 268438657, 268448641, 268440961, 268447873, 268441153, 268447297, 268441601, 268445377, 268444609, 268444481, 268441729, 268438337, 268431169, 268439681, 268437569, 268438913, 268438081, 268436801, 268432897, 268435649],
              #   [268042241, 265420801, 264634369, 263454721, 263323649, 261881857, 261488641, 260702209, 260571137],
              #    True
              # ),
            ]


scaling_factor, encoding_precision, encryption_precision = 2**16, 10, 8

def _hemul_mul_kernel(input_array, parameters):
  """Kernel wrapper entry point used by KernelWrapper."""
  ct_input = parameters["ct_input"]
  ct_input.polynomial = input_array
  result = parameters["he_mul"].mul(ct_input)
  return result.polynomial

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
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Collecting logs from: {root_dir}")
    collect_logs(root_dir, output_csv_name="he_mult_profiling")

  def _create_kernel_wrapper(self, kernel_name, he_mul, batch, elements, degree_layout, limbs, moduli):
    input_shape = (batch, elements, *degree_layout, limbs)
    degree = 1
    for d in degree_layout:
        degree *= d
    ct_in_shapes = {
        'batch': batch, 'num_elements': elements, 'degree': degree,
        'precision': 32, 'num_moduli': limbs, 'degree_layout': degree_layout,
    }
    ct_input = Polynomial(ct_in_shapes, parameters={'moduli': moduli, "BAT_lazy": False})
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_hemul_mul_kernel,
        input_structs=[(input_shape, jnp.uint32)],
        parameters={"he_mul": he_mul, "ct_input": ct_input},
    )

  @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS)
  def test_hemul(self, degree, r, c, limbs, dnum, moduli, extend_moduli, perf_test):
    if extend_moduli is None:
      self.skipTest("extend_moduli is None, skip test")
    num_eval_mult = 1

    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"hemul_N{degree}",
        configuration=self.profiler_config,
    )

    elements = 4
    original_moduli = moduli
    eval_key = {"a": util.random_parameters((dnum, degree, len(original_moduli)+len(extend_moduli)), moduli, dtype=jnp.uint32), "b": util.random_parameters((dnum, degree, len(original_moduli)+len(extend_moduli)), moduli, dtype=jnp.uint32)}
    evalkey_a_vector = jnp.array(eval_key["a"], dtype=jnp.uint32)
    evalkey_b_vector = jnp.array(eval_key["b"], dtype=jnp.uint32)

    batch_list = [1]
    degree_layout = (r, c)
    for batch in batch_list:
        print(f"Running for batch size: {batch}")
        # Re-generate input data for each batch
        in_ciphertexts = util.random_parameters((batch, elements, *degree_layout, limbs), moduli, dtype=jnp.uint32)

        # Instantiate HEMul for current batch
        he_mul = HEMul(batch, r, c, dnum, num_eval_mult, original_moduli, extend_moduli)
        he_mul.control_gen(degree_layout=degree_layout, perf_test=perf_test)
        he_mul.setup_relinearization(evalkey_a_vector, evalkey_b_vector)

        test_case_name = self._testMethodName
        kernel_wrapper = self._create_kernel_wrapper(
            kernel_name=f"{test_case_name}_B{batch}",
            he_mul=he_mul,
            batch=batch,
            elements=elements,
            degree_layout=degree_layout,
            limbs=limbs,
            moduli=moduli,
        )

        profiler_instance.add_profile(
            name=f"{test_case_name}_B{batch}",
            kernel_wrapper=kernel_wrapper,
            kernel_setting_cols={
                "degree": degree,
                "num_limbs": limbs,
                "r": r,
                "c": c,
                "batch": batch,
                "num_elements": elements,
            },
        )


    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()


class BatchDimensionShardingTest(parameterized.TestCase):
  """Profiles HEMul contexts with sharding across the batch dimension."""

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
    collect_logs(root_dir, output_csv_name="he_mult_profiling")

  def _create_sharded_kernel_wrapper(self, kernel_name, he_mul, batch, elements, degree_layout, limbs, moduli, mesh, input_sharding, output_sharding):
    input_shape = (batch, elements, *degree_layout, limbs)
    degree = 1
    for d in degree_layout:
        degree *= d
    ct_in_shapes = {
        'batch': batch, 'num_elements': elements, 'degree': degree,
        'precision': 32, 'num_moduli': limbs, 'degree_layout': degree_layout,
    }
    ct_input = Polynomial(ct_in_shapes, parameters={'moduli': moduli, "BAT_lazy": False})
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_hemul_mul_kernel,
        input_structs=[(input_shape, jnp.uint32)],
        parameters={"he_mul": he_mul, "ct_input": ct_input},
        mesh=mesh,
        input_shardings=(input_sharding,),
        output_sharding=output_sharding,
        enable_sharding=True,
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS)
  def test_hemul_sharded(self, degree, r, c, limbs, dnum, moduli, extend_moduli, perf_test):
    if extend_moduli is None:
      self.skipTest("extend_moduli is None, skip test")

    try:
      mesh, partition_spec = util.create_sharding()
      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
    except RuntimeError as exc:
      self.skipTest(str(exc))

    profiler_config = self.profiler_config.copy()
    profiler_config["enable_sharding"] = True
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"hemul_N{degree}",
        configuration=profiler_config,
    )

    num_devices = jax.device_count()
    if num_devices == 0:
        batch_list = [1] # Fallback if no devices found, though likely raises RuntimeError earlier
    else:
        batch_list = [num_devices]

    elements = 4
    num_eval_mult = 1
    original_moduli = moduli
    eval_key = {"a": util.random_parameters((dnum, degree, len(original_moduli)+len(extend_moduli)), moduli, dtype=jnp.uint32), "b": util.random_parameters((dnum, degree, len(original_moduli)+len(extend_moduli)), moduli, dtype=jnp.uint32)}
    evalkey_a_vector = jnp.array(eval_key["a"], dtype=jnp.uint32)
    evalkey_b_vector = jnp.array(eval_key["b"], dtype=jnp.uint32)
    degree_layout = (r, c)

    # Input sharding: (batch, elements, *degree_layout, limbs) — 5D
    input_none_dims = (None,) * (1 + len(degree_layout) + 1)  # elements + degree_layout dims + limbs
    input_sharding = jax.sharding.NamedSharding(
        mesh, partition_spec(batch_partition, *input_none_dims))
    # Output sharding: (batch, 2, *degree_layout, idx_cur_last_tower) — also 5D
    output_none_dims = (None,) * (1 + len(degree_layout) + 1)
    output_sharding = jax.sharding.NamedSharding(
        mesh, partition_spec(batch_partition, *output_none_dims))

    for batch in batch_list:
      print(f"Running sharded for batch size: {batch}")

      # Instantiate HEMul for current batch
      he_mul = HEMul(batch, r, c, dnum, num_eval_mult, original_moduli, extend_moduli)
      he_mul.control_gen(degree_layout=degree_layout, perf_test=perf_test)
      he_mul.setup_relinearization(evalkey_a_vector, evalkey_b_vector)

      # Extract the test case name from the method name
      test_case_name = self._testMethodName
      kernel_name = f"{test_case_name}_B{batch}"

      kernel_wrapper = self._create_sharded_kernel_wrapper(
          kernel_name=kernel_name,
          he_mul=he_mul,
          batch=batch,
          elements=elements,
          degree_layout=degree_layout,
          limbs=limbs,
          moduli=moduli,
          mesh=mesh,
          input_sharding=input_sharding,
          output_sharding=output_sharding,
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
              "num_elements": elements,
              "sharding": "batch",
          },
      )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()


if __name__ == "__main__":
  absltest.main()

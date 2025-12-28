"""A module for operations on test CKKS evaluation kernels including.

- Rescale
"""
# Standard library imports
import os

# Third-party imports
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

# Local imports
from ciphertext import Ciphertext
import util
from profiler import KernelWrapper, Profiler, collect_logs

# JAX configuration
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
            ]

scaling_factor, encoding_precision, encryption_precision = 2**16, 10, 8

def _rescale_kernel(input_array, parameters):
  """Kernel wrapper entry point used by KernelWrapper."""
  ct = parameters["ct"]
  # The input array is expected to be the batched ciphertext data
  ct.set_batch_ciphertext(input_array)
  return ct.rescale()

class RescalePerformanceTest(parameterized.TestCase):
  """A base class for running rescale performance tests."""

  def __init__(self, *args, **kwargs):
    super(RescalePerformanceTest, self).__init__(*args, **kwargs)
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
    collect_logs(root_dir, output_csv_name="rescale_profiling")

  def _create_kernel_wrapper(self, kernel_name, ct, batch, elements, degree_layout, num_moduli):
    input_shape = (batch, elements, *degree_layout, num_moduli)
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_rescale_kernel,
        input_structs=[(input_shape, jnp.uint32)],
        parameters={"ct": ct},
    )

  @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS)
  def test_rescale(self, degree, r, c, limbs, dnum, moduli, extend_moduli, perf_test):
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"rescale_N{degree}",
        configuration=self.profiler_config,
    )
    elements = 2
    batch_list = [1]
    degree_layout = (r, c)
    for batch in batch_list:
        print(f"Running for batch size: {batch}")

        shapes = {'batch': batch, 'num_elements': elements, 'degree': degree, 'num_moduli': len(moduli), 'precision': 32}
        params = {'moduli': moduli, 'r': r, 'c': c}
        ct = Ciphertext(shapes, params)
        ct.random_init()
        ct.modulus_switch_control_gen(degree_layout=degree_layout, perf_test=perf_test)

        test_case_name = self._testMethodName

        kernel_wrapper = self._create_kernel_wrapper(
            kernel_name=f"{test_case_name}_B{batch}",
            ct=ct,
            batch=batch,
            elements=elements,
            degree_layout=degree_layout,
            num_moduli=len(moduli),
        )

        profiler_instance.add_profile(
            name=f"{test_case_name}_B{batch}",
            kernel_wrapper=kernel_wrapper,
            kernel_setting_cols={
                "degree": degree,
                "num_limbs": len(moduli),
                "r": r,
                "c": c,
                "batch": batch,
                "num_elements": elements,
            },
        )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()


class RescaleShardedPerformanceTest(parameterized.TestCase):
  """Profiles Rescale with sharding across the batch dimension."""

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
    collect_logs(root_dir, output_csv_name="rescale_profiling")

  def _create_sharded_kernel_wrapper(self, kernel_name, ct, batch, elements, degree_layout, num_moduli, mesh, batch_sharding_input, batch_sharding_output):
    input_shape = (batch, elements, *degree_layout, num_moduli)
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_rescale_kernel,
        input_structs=[(input_shape, jnp.uint32)],
        parameters={"ct": ct},
        mesh=mesh,
        input_shardings=(batch_sharding_input,),
        output_sharding=batch_sharding_output,
        enable_sharding=True,
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS)
  def test_rescale_sharded(self, degree, r, c, limbs, dnum, moduli, extend_moduli, perf_test):
    try:
      mesh, partition_spec = util.create_sharding()
      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]

      # Input shape: (Batch, Elements, Degree, Moduli)
      batch_sharding_input = jax.sharding.NamedSharding(
          mesh,
          partition_spec(batch_partition,),
      )
    except RuntimeError as exc:
      self.skipTest(str(exc))

    profiler_config = self.profiler_config.copy()
    profiler_config["enable_sharding"] = True
    profiler_instance = Profiler(
        output_trace_path=self.output_trace_root,
        profile_naming=f"{self._testMethodName}_N{degree}",
        configuration=profiler_config,
    )

    elements = 2
    degree_layout = (r, c)
    num_devices = jax.device_count()
    batch_list = [num_devices]
    for batch in batch_list:
        print(f"Running for batch size: {batch}")

        shapes = {'batch': batch, 'num_elements': elements, 'degree': degree, 'num_moduli': len(moduli), 'precision': 32, 'degree_layout': degree_layout}
        params = {'moduli': moduli, 'r': r, 'c': c}
        ct = Ciphertext(shapes, params)
        ct.random_init()
        ct.modulus_switch_control_gen(degree_layout=degree_layout, perf_test=perf_test)

        test_case_name = self._testMethodName

        kernel_wrapper = self._create_sharded_kernel_wrapper(
            kernel_name=f"{test_case_name}_B{batch}",
            ct=ct,
            batch=batch,
            elements=elements,
            degree_layout=degree_layout,
            num_moduli=len(moduli),
            mesh=mesh,
            batch_sharding_input=batch_sharding_input,
            batch_sharding_output=batch_sharding_input,
        )

        profiler_instance.add_profile(
            name=f"{test_case_name}_B{batch}",
            kernel_wrapper=kernel_wrapper,
            kernel_setting_cols={
                "degree": degree,
                "num_limbs": len(moduli),
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

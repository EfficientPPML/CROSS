"""A module for operations on test CKKS evaluation kernels including.

- HEAdd
"""
# Standard library imports
import os

# Third-party imports
import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized

# Local imports
import add
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

def _jax_add_kernel(c1, c2, parameters):
  """Kernel wrapper entry point used by KernelWrapper."""
  return add.jax_add(c1, c2, parameters["moduli"])

class PerformanceTest(parameterized.TestCase):
  """A base class for running add tests."""

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
    # Call collect_logs at the end of the test class execution
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Collecting logs from: {root_dir}")
    collect_logs(root_dir, output_csv_name="add_profiling")

  def _create_kernel_wrapper(self, kernel_name, batch, degree, moduli):
    input_shape = (batch, len(moduli), degree)
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_jax_add_kernel,
        input_structs=[
            (input_shape, jnp.uint32),
            (input_shape, jnp.uint32)
        ],
        parameters={"moduli": jnp.array(moduli, dtype=jnp.uint32)},
    )

  # @absltest.skip("test single implementation")
  @parameterized.named_parameters(*TEST_PARAMS)
  def test_jax_add(self, degree, r, c, limbs, dnum, moduli, extend_moduli, perf_test):
      """Test adding two ciphertexts."""
      print("generate data")

      profiler_instance = Profiler(
          output_trace_path=self.output_trace_root,
          profile_naming=f"{self._testMethodName}_N{degree}",
          configuration=self.profiler_config,
      )

      # Using original batch sizes
      for batch in [1, 8, 16, 32]:
        print(f"Running for batch size: {batch}")

        kernel_name = f"{self._testMethodName}_b{batch}"

        kernel_wrapper = self._create_kernel_wrapper(
            kernel_name=kernel_name,
            batch=batch,
            degree=degree,
            moduli=moduli
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
            },
        )

      profiler_instance.profile_all_profilers()
      profiler_instance.post_process_all_profilers()


class BatchDimensionShardingTest(parameterized.TestCase):
  """Profiles HEAdd contexts with sharding across the batch dimension."""

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
    collect_logs(root_dir, output_csv_name="add_profiling")

  def _create_sharded_kernel_wrapper(self, kernel_name, batch, degree, moduli, mesh, batch_sharding):
    input_shape = (batch, len(moduli), degree)
    return KernelWrapper(
        kernel_name=kernel_name,
        function_to_wrap=_jax_add_kernel,
        input_structs=[
            (input_shape, jnp.uint32),
            (input_shape, jnp.uint32)
        ],
        parameters={"moduli": jnp.array(moduli, dtype=jnp.uint32)},
        mesh=mesh,
        input_shardings=(batch_sharding, batch_sharding),
        output_sharding=batch_sharding,
        enable_sharding=True,
    )

  @parameterized.named_parameters(*TEST_PARAMS)
  def test_jax_add_sharded(self, degree, r, c, limbs, dnum, moduli, extend_moduli, perf_test):
    try:
      mesh, partition_spec = util.create_sharding()
      axis_names = mesh.axis_names
      batch_partition = axis_names if len(axis_names) > 1 else axis_names[0]
      # Input shape is (batch, moduli, degree), so partition on axis 0
      batch_sharding = jax.sharding.NamedSharding(
          mesh,
          partition_spec(batch_partition, None, None),
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

    num_devices = jax.device_count()
    batch_list = [num_devices]

    for batch in batch_list:
      print(f"Running sharded for batch size: {batch}")

      kernel_name = f"{self._testMethodName}_b{batch}"

      kernel_wrapper = self._create_sharded_kernel_wrapper(
          kernel_name=kernel_name,
          batch=batch,
          degree=degree,
          moduli=moduli,
          mesh=mesh,
          batch_sharding=batch_sharding
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
              "sharding": "batch",
          },
      )

    profiler_instance.profile_all_profilers()
    profiler_instance.post_process_all_profilers()


if __name__ == "__main__":
  absltest.main()

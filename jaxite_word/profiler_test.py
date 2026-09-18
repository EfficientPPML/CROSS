"""Correctness tests for the reusable kernel profiling helpers."""

import tempfile

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np

import profiler


class ProfilerUtilityTest(absltest.TestCase):

  def test_list_add_rejects_mismatched_lengths(self):
    self.assertEqual(profiler.list_add([1, 2], [3, 4]), [4, 6])
    with self.assertRaisesRegex(ValueError, "same length"):
      profiler.list_add([1], [2, 3])

  def test_kernel_wrapper_defaults_are_isolated_and_kernel_executes(self):
    first = profiler.KernelWrapper(
        "add_one_first", lambda x: x + 1,
        [((2,), jnp.float32)],
    )
    second = profiler.KernelWrapper(
        "add_one_second", lambda x: x + 1,
        [((2,), jnp.float32)],
    )
    first.parameters["sentinel"] = 1
    self.assertEmpty(second.parameters)
    np.testing.assert_array_equal(
        first.get_compiled_function()(jnp.array([1, 2], dtype=jnp.float32)),
        [2, 3],
    )

  def test_profile_setting_defaults_are_isolated(self):
    with tempfile.TemporaryDirectory() as tmp:
      instance = profiler.Profiler(tmp, "isolation")
      instance.add_profile("first", object())
      instance.add_profile("second", object())
    instance.profiles[0]["settings"]["sentinel"] = 1
    self.assertEmpty(instance.profiles[1]["settings"])


if __name__ == "__main__":
  absltest.main()

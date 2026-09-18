from absl.testing import absltest
import numpy as np

import cleartext_ops


class CleartextOpsTest(absltest.TestCase):

  def test_conv_bias_quad_and_matmul(self):
    image = np.arange(16, dtype=np.float64)
    weights = np.ones(4, dtype=np.float64)
    conv = cleartext_ops.conv2d_ref(
        image, weights, 1, 1, 4, 4, 2, 2, 2, 0
    )
    np.testing.assert_array_equal(conv, [10.0, 18.0, 42.0, 50.0])
    biased = cleartext_ops.add_conv_bias(conv, [2.0], 1, 2, 2)
    squared = cleartext_ops.quad_ref(biased)
    np.testing.assert_array_equal(
        cleartext_ops.matmul_ref(squared, np.eye(4), 4, 4), squared
    )

  def test_pooling(self):
    values = np.arange(16, dtype=np.float64)
    expected = np.array([2.5, 4.5, 10.5, 12.5])
    np.testing.assert_array_equal(
        cleartext_ops.avg_pool_2x2(values, 1, 4, 4), expected
    )
    np.testing.assert_array_equal(
        cleartext_ops.adaptive_avg_pool_2x2(values, 1, 4, 4), expected
    )


if __name__ == '__main__':
  absltest.main()

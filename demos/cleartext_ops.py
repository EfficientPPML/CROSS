"""Packing-independent NumPy reference operations shared by HE demos."""

import numpy as np


def conv2d_ref(x, weights, ci, co, hi, wi, kh, kw, stride, pad):
  ho = (hi + 2 * pad - kh) // stride + 1
  wo = (wi + 2 * pad - kw) // stride + 1
  out = np.zeros(co * ho * wo, dtype=np.float64)
  for out_channel in range(co):
    for out_row in range(ho):
      for out_col in range(wo):
        value = 0.0
        for in_channel in range(ci):
          for kernel_row in range(kh):
            for kernel_col in range(kw):
              in_row = out_row * stride + kernel_row - pad
              in_col = out_col * stride + kernel_col - pad
              if 0 <= in_row < hi and 0 <= in_col < wi:
                value += (
                    x[in_channel * hi * wi + in_row * wi + in_col]
                    * weights[
                        out_channel * ci * kh * kw
                        + in_channel * kh * kw
                        + kernel_row * kw
                        + kernel_col
                    ]
                )
        out[out_channel * ho * wo + out_row * wo + out_col] = value
  return out


def conv_bias(bias, channels, height, width):
  """Broadcast channel bias to a flat CHW tensor."""
  return np.repeat(
      np.asarray(bias[:channels], dtype=np.float64), height * width
  )


def add_conv_bias(values, bias, channels, height, width):
  return values + conv_bias(bias, channels, height, width)


def avg_pool_2x2(values, channels, height, width, stride=2):
  out_height, out_width = height // stride, width // stride
  out = np.zeros(channels * out_height * out_width, dtype=np.float64)
  for channel in range(channels):
    for out_row in range(out_height):
      for out_col in range(out_width):
        value = 0.0
        for pool_row in range(2):
          for pool_col in range(2):
            value += values[
                channel * height * width
                + (out_row * stride + pool_row) * width
                + out_col * stride
                + pool_col
            ]
        out[
            channel * out_height * out_width + out_row * out_width + out_col
        ] = value / 4.0
  return out


def adaptive_avg_pool_2x2(
    values, channels, height, width, out_h=2, out_w=2
):
  if height % out_h != 0 or width % out_w != 0:
    raise NotImplementedError('non-divisible adaptive pool')
  kernel_height, kernel_width = height // out_h, width // out_w
  out = np.zeros(channels * out_h * out_w, dtype=np.float64)
  scale = 1.0 / (kernel_height * kernel_width)
  for channel in range(channels):
    for out_row in range(out_h):
      for out_col in range(out_w):
        value = 0.0
        for kernel_row in range(kernel_height):
          for kernel_col in range(kernel_width):
            value += values[
                channel * height * width
                + (out_row * kernel_height + kernel_row) * width
                + out_col * kernel_width
                + kernel_col
            ]
        out[
            channel * out_h * out_w + out_row * out_w + out_col
        ] = value * scale
  return out


def matmul_ref(values, weights, n_out, k_in):
  return weights.reshape(n_out, k_in) @ values[:k_in]


def quad_ref(values):
  return values * values

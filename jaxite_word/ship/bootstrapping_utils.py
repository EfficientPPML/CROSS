# Adapted from Google Jaxite PR147 for CROSS, 2026-10-27. See README.md and LICENSE.
"""Utilities for SHIP CKKS bootstrapping."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from . import math as ckks_math


def rot_group(m: int, nh: int, g: int = 5) -> list[int]:
  """Generates the rotation group."""
  r = [1]
  for _ in range(1, nh):
    r.append((r[-1] * g) % m)
  return r


def get_bit_reverse_perm_indices(size: int) -> jax.Array:
  """Returns the bit-reversal permutation indices of a given size."""
  indices = jnp.arange(size, dtype=jnp.int32)
  bits = int(math.log2(size))
  return ckks_math.bit_reverse(indices, bits)


@dataclasses.dataclass
class SpecialInverseFFT:
  """Kernel for specialized inverse FFT for CKKS slots encoding."""

  degree: int = 0
  br_indices: jax.Array = dataclasses.field(  # pyrefly: ignore[bad-assignment]
    default_factory=lambda: np.empty((0,), dtype=np.int32)
  )
  level_roots_real: tuple[jax.Array, ...] = ()
  level_roots_imag: tuple[jax.Array, ...] = ()

  def precompute_constants(self, degree: int):
    """Precomputes roots of unity and bit-reversal indices."""
    self.degree = degree
    m = self.degree * 2
    nh = self.degree // 2
    roots = np.exp(2j * np.pi * np.arange(m) / m)
    roots_real = np.real(roots)
    roots_imag = np.imag(roots)
    rg = np.array(rot_group(m, nh, g=5))

    level_roots_real = []
    level_roots_imag = []
    length = nh
    while length >= 2:
      half = length >> 1
      lenq = length << 2
      step = m // lenq
      mod = rg[:half] % lenq
      idx = ((lenq - mod) % lenq) * step
      level_roots_real.append(jnp.array(roots_real[idx], dtype=jnp.float64))
      level_roots_imag.append(jnp.array(roots_imag[idx], dtype=jnp.float64))
      length >>= 1

    indices = jnp.arange(nh, dtype=jnp.int32)
    bits = int(math.log2(nh))
    self.br_indices = ckks_math.bit_reverse(indices, bits)
    self.level_roots_real = tuple(level_roots_real)
    self.level_roots_imag = tuple(level_roots_imag)

  def __call__(self, vals: jax.Array) -> jax.Array:
    nh = self.degree // 2

    curr_real = jax.lax.real(vals)
    curr_imag = jax.lax.imag(vals)

    num_levels = len(self.level_roots_real)
    length = nh
    for i in range(num_levels):
      half = length >> 1
      r_real = self.level_roots_real[i]
      r_imag = self.level_roots_imag[i]

      shape = curr_real.shape
      curr_real_reshaped = curr_real.reshape(-1, length)
      curr_imag_reshaped = curr_imag.reshape(-1, length)
      u_real = curr_real_reshaped[:, :half]
      u_imag = curr_imag_reshaped[:, :half]
      t_real = curr_real_reshaped[:, half:]
      t_imag = curr_imag_reshaped[:, half:]

      new_u_real = u_real + t_real
      new_u_imag = u_imag + t_imag

      diff_real = u_real - t_real
      diff_imag = u_imag - t_imag

      new_v_real = diff_real * r_real[None, :] - diff_imag * r_imag[None, :]
      new_v_imag = diff_real * r_imag[None, :] + diff_imag * r_real[None, :]

      curr_real = jnp.concatenate([new_u_real, new_v_real], axis=1).reshape(shape)
      curr_imag = jnp.concatenate([new_u_imag, new_v_imag], axis=1).reshape(shape)
      length >>= 1

    curr_real = curr_real * jnp.array(1.0 / nh, dtype=jnp.float64)
    curr_imag = curr_imag * jnp.array(1.0 / nh, dtype=jnp.float64)
    curr_real = curr_real[..., self.br_indices]
    curr_imag = curr_imag[..., self.br_indices]
    return jax.lax.complex(curr_real, curr_imag)


def compute_v0_slots(
  b_coeffs_flat: jax.Array,
  q0: int | jax.Array,
  num_slots: int,
  gamma: float = 1.0,
) -> jax.Array:
  """Computes v0 slot values (Algorithm 1, Step 1)."""
  q0_jax = jnp.asarray(q0)
  phase = (2.0 * np.pi) / q0_jax.astype(jnp.float64)
  exponent_real = b_coeffs_flat[:num_slots].astype(jnp.float64) * phase
  exponent_complex = jax.lax.complex(jnp.zeros_like(exponent_real), exponent_real)
  w_b = jnp.exp(exponent_complex)
  coeff = gamma * jnp.array(-0.25j / np.pi, dtype=jnp.complex128)
  return coeff * w_b


def compute_a_slots(
  a_lower: jax.Array,
  a_upper: jax.Array,
  q0: int | jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Computes w1_a and w2_a slot values (Algorithm 1, Steps 2-3)."""
  q0_jax = jnp.asarray(q0)
  phase = (2.0 * np.pi) / q0_jax.astype(jnp.float64)
  exponent_real_1 = a_lower.astype(jnp.float64) * phase
  exponent_complex_1 = jax.lax.complex(jnp.zeros_like(exponent_real_1), exponent_real_1)
  w1_a_slots = jnp.exp(exponent_complex_1)
  exponent_real_2 = a_upper.astype(jnp.float64) * phase
  exponent_complex_2 = jax.lax.complex(jnp.zeros_like(exponent_real_2), exponent_real_2)
  w2_a_slots = jnp.exp(exponent_complex_2)
  return w1_a_slots, w2_a_slots

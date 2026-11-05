# Adapted from Google Jaxite PR147 for CROSS, 2026-10-27. See README.md and LICENSE.
"""Utility functions for homomorphic blind rotation."""

import math

import jax
import jax.numpy as jnp

from . import math as ckks_math


def compute_automorphism_indices(degree: int, g: int) -> jax.Array:
  """Computes target indices for automorphism X -> X^g in NTT domain."""
  bits = int(math.log2(degree))
  indices = jnp.arange(degree, dtype=jnp.uint32)

  # Bit-reverse indices to map the bit-reversed layout of jaxite's NTT
  br_indices = ckks_math.bit_reverse(indices, bits)
  g_u32 = jnp.array(g, dtype=jnp.uint32)
  target_roots = (((2 * br_indices + 1) * g_u32 - 1) // 2) % degree
  target_indices = ckks_math.bit_reverse(target_roots, bits)
  return target_indices


def apply_automorphism_ntt_with_indices(
  data: jax.Array, indices: jax.Array
) -> jax.Array:
  """Applies automorphism using precomputed indices."""
  return jnp.take(data, indices, axis=-2)


def apply_automorphism_ntt(data: jax.Array, g: int) -> jax.Array:
  """Applies the automorphism X -> X^g to a polynomial in the NTT domain.

  Handles the bit-reversed layout of jaxite's NTT representation.

  Args:
    data: The polynomial in NTT domain. Shape (..., degree, num_moduli).
    g: The automorphism generator (must be odd).

  Returns:
    The permuted polynomial in NTT domain with the same shape.
  """
  degree = data.shape[-2]
  target_indices = compute_automorphism_indices(degree, g)
  return apply_automorphism_ntt_with_indices(data, target_indices)

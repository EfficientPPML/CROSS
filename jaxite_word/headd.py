"""Homomorphic ciphertext addition with per-RNS-tower modular reduction."""

import jax
import jax.numpy as jnp

import finite_field
import polynomial

Polynomial = polynomial.Polynomial
# Aliased only for the _HEAddKernel.__init__ default argument.
BarrettContext = finite_field.BarrettContext
jax.config.update("jax_enable_x64", True)


class _HEAddKernel:
  """Private element-wise ciphertext-add kernel."""

  def __init__(self, moduli, finite_field_context=BarrettContext):
    self.moduli = list(moduli)
    self.num_moduli = len(moduli)
    if isinstance(finite_field_context, type):
      self.ff_context_cls = finite_field_context
      self.ff_ctx = finite_field_context(moduli=self.moduli)
    elif isinstance(
        finite_field_context, finite_field.FiniteFieldContextBase
    ):
      self.ff_context_cls = type(finite_field_context)
      self.ff_ctx = finite_field_context
      if tuple(self.ff_ctx.moduli) != tuple(self.moduli):
        raise ValueError(
            '_HEAddKernel finite-field context moduli do not match the '
            f'configured moduli ({tuple(self.ff_ctx.moduli)} vs '
            f'{tuple(self.moduli)}).'
        )
    else:
      raise TypeError(
          'finite_field_context must be a finite-field context class or '
          'instance.'
      )

  def add(self, ct1: Polynomial, ct2: Polynomial) -> Polynomial:
    """Return a new Polynomial holding ct1 + ct2 reduced per tower.

    Inputs must carry strict per-tower residues in [0, q) (op-boundary contract).
    """
    finite_field.check_binary_ct_operands(
        self.ff_context_cls, self.num_moduli, ct1, ct2, "_HEAddKernel.add",
        moduli=self.moduli
    )
    reduced = self._add_unchecked(ct1.polynomial, ct2.polynomial)
    return ct1._clone_with_payload(reduced.astype(ct1.modulus_dtype))

  def _add_array(self, ct1_data, ct2_data):
    """Private raw-array entry point for fused/JIT regions."""
    finite_field.check_rank5_array(
        ct1_data, '_HEAddKernel._add_array', num_moduli=self.num_moduli
    )
    finite_field.check_rank5_array(
        ct2_data, '_HEAddKernel._add_array', num_moduli=self.num_moduli
    )
    if ct1_data.shape != ct2_data.shape:
      raise ValueError(
          f'_HEAddKernel._add_array: operand shapes differ '
          f'({ct1_data.shape} vs {ct2_data.shape}).'
      )
    return self._add_unchecked(ct1_data, ct2_data)

  def _add_unchecked(self, ct1_data, ct2_data):
    s = ct1_data.astype(jnp.uint64) + ct2_data.astype(jnp.uint64)
    # Add is linear (R passes through); a modular_reduction REDC would strip R, so canonicalize with strictify_after_accumulation.
    return self.ff_ctx.strictify_after_accumulation(s, 2)


__all__ = []

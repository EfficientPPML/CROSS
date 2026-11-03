"""
Name: JAX Finite Field Context Integration

Name Template: <Framework><Representation><Reduction><Strategy>Context<Base>
    - <Framework>:  (JAX accelerator backend).
    - <Representation>: [Optional]
        - Empty: Standard scalar.
        - RNS: Residue Number System.
        - DRNS: Digitized RNS.
        - RD: Radix Decomposition (Big Integer simulation).
    - <Reduction>: [Optional]
        - Montgomery: Montgomery reduction.
        - Barrett: Barrett reduction.
        - Shoup: Shoup reduction.
    - <Strategy>: [Optional]
        - MultipleModuli: vectorized over moduli.
        - Lazy: Lazy reduction.
        - Opt/Opt2: Optimization levels or specific variants.
    - Context: Class suffix.
    - <Base>: [Optional] Abstract base class.

Explanation: This module adapts the generic finite field contexts for use with JAX. It inherits from the base contexts in `finite_field_context.py` and adds functionality to precompute and format parameters (such as modular inverses, RNS matrices, and bit-shifted constants) into JAX-compatible arrays. It serves as the configuration bridge between the mathematical specifications and the JAX kernels.
"""

import util
import math
from typing import Callable, List, Union
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)


def check_reduction_match(op_ff_cls, data_ff, op_name):
  """Trace-time check that data's ff_ctx matches the op's configured reduction class; mixed Montgomery/Barrett data is silent garbage (residues off by R)."""
  if not isinstance(data_ff, op_ff_cls):
    raise ValueError(
        f"{op_name} configured for {op_ff_cls.__name__} but received "
        f"{type(data_ff).__name__} data (mixing reductions is silent garbage); "
        "re-encode with the op's reduction context"
    )


def check_rank5_array(
    value, op_name, *, batch=None, num_elements=None, degree_layout=None,
    num_moduli=None, dtype=jnp.uint32
):
  """Validates a canonical raw ciphertext array at an explicit array boundary."""
  if not hasattr(value, 'ndim') or not hasattr(value, 'shape'):
    raise TypeError(f'{op_name} requires an array, got {type(value).__name__}.')
  if dtype is not None:
    actual_dtype = getattr(value, 'dtype', None)
    expected_dtype = jnp.dtype(dtype)
    if actual_dtype is None or jnp.dtype(actual_dtype) != expected_dtype:
      raise ValueError(
          f'{op_name} requires ciphertext dtype {expected_dtype}, got '
          f'{actual_dtype}.'
      )
  if value.ndim != 5:
    raise ValueError(
        f'{op_name} requires rank-5 shape '
        '(batch, num_elements, r, c, num_moduli); '
        f'got rank {value.ndim} shape {value.shape}.'
    )
  expected = (batch, num_elements, *(degree_layout or (None, None)), num_moduli)
  labels = ('batch', 'num_elements', 'r', 'c', 'num_moduli')
  for axis, (actual, wanted) in enumerate(
      zip(value.shape, expected, strict=True)
  ):
    if wanted is not None and actual != wanted:
      raise ValueError(
          f'{op_name} expected {labels[axis]}={wanted}, got shape {value.shape}.'
      )
  return value


def canonical_degree_layout(r, c, degree_layout, op_name):
  """Returns the only layout accepted by an operator implemented for r x c."""
  expected = (r, c)
  actual = expected if degree_layout is None else tuple(degree_layout)
  if actual != expected:
    raise ValueError(
        f'{op_name} degree_layout must match its tiled kernel layout '
        f'{expected}, got {actual}.'
    )
  return actual


def check_ct_operand(
    op_ff_cls, num_moduli, ct, op_name, *, batch=None, num_elements=None,
    degree_layout=None, moduli=None
):
  """Validates a canonical Polynomial and its operator-specific metadata."""
  # Lazy import avoids the finite_field <-> polynomial module import cycle.
  import polynomial
  if not isinstance(ct, polynomial.Polynomial):
    raise TypeError(
        f'{op_name} requires Polynomial, got {type(ct).__name__}; '
        'raw arrays are accepted only by private fused-region kernels.'
    )
  ct.validate()
  expected_dtype = jnp.dtype(ct.modulus_dtype)
  if ct.precision != 32 or expected_dtype != jnp.dtype(jnp.uint32):
    raise ValueError(
        f'{op_name}: Polynomial must use the canonical '
        'precision=32/uint32 ciphertext representation.'
    )
  if ct.polynomial.dtype != expected_dtype:
    raise ValueError(
        f'{op_name}: Polynomial payload dtype {ct.polynomial.dtype} does not '
        f'match its {expected_dtype} ciphertext representation.'
    )
  check_reduction_match(op_ff_cls, ct.ntt_ctx.ff_ctx, op_name)
  check_rank5_array(
      ct.polynomial,
      op_name,
      batch=batch,
      num_elements=num_elements,
      degree_layout=degree_layout,
      num_moduli=num_moduli,
      dtype=expected_dtype,
  )
  if ct.num_moduli != num_moduli:
    raise ValueError(
        f'{op_name}: ciphertext has {ct.num_moduli} towers, op configured '
        f'for {num_moduli}.'
    )
  if degree_layout is not None and tuple(ct.degree_layout) != tuple(degree_layout):
    raise ValueError(
        f'{op_name}: ciphertext layout {ct.degree_layout} does not match '
        f'operator layout {tuple(degree_layout)}.'
    )
  if moduli is not None and tuple(ct.moduli) != tuple(moduli):
    raise ValueError(
        f'{op_name}: ciphertext moduli {tuple(ct.moduli)} do not match '
        f'operator moduli {tuple(moduli)}.'
    )
  return ct


def check_binary_ct_operands(
    op_ff_cls, num_moduli, ct1, ct2, op_name, *, batch=None,
    num_elements=None, degree_layout=None, moduli=None
):
  """Validates two Polynomial operands at a public binary-op boundary."""
  check_ct_operand(
      op_ff_cls, num_moduli, ct1, op_name, batch=batch,
      num_elements=num_elements, degree_layout=degree_layout, moduli=moduli
  )
  check_ct_operand(
      op_ff_cls, num_moduli, ct2, op_name, batch=batch,
      num_elements=num_elements, degree_layout=degree_layout, moduli=moduli
  )
  if ct1.polynomial.shape != ct2.polynomial.shape:
    raise ValueError(
        f"{op_name}: operand shapes differ "
        f"({ct1.polynomial.shape} vs {ct2.polynomial.shape})"
    )
  if tuple(ct1.moduli) != tuple(ct2.moduli):
    raise ValueError(
        f'{op_name}: operand moduli differ '
        f'({tuple(ct1.moduli)} vs {tuple(ct2.moduli)}).'
    )


########################
# Base Context Class
########################
class FiniteFieldContextBase:

  # Hooks name the paired NTT-ciphertext/BConv classes as STRINGS to avoid an import cycle (ntt_mm/bconv import this module).
  # None = not injectable via the generic pipeline (ShoupContext: two-operand modular_reduction); its NTT ctx can still be passed as ntt_ctx.
  ntt_ciphertext_context_cls = None
  bconv_cls = None

  # True when the computation format equals standard residues (Barrett & co.);
  # Montgomery overrides to False (data carries an R factor). Representation
  # boundaries (encrypt/decrypt, ModRaise) use this to skip no-op conversions.
  computation_format_is_standard = True

  def __init__(self, moduli: int):
    self.moduli = moduli

  @property
  def ff_ctx(self):
    """Self-reference allowing this context to serve as ntt_ctx for mod_reduce-only Polynomials.

    Private Polynomial kernel diagnostics call modular_reduction through the
    wrapper's NTT context.
    By returning self, any finite field context can be injected directly as
    ntt_ctx for ciphertexts that only need modular reduction (no NTT/INTT).
    """
    return self

  def to_computation_format(self, a):
    return a

  def to_original_format(self, a):
    return a

  # Plug-and-play reduction primitives. Three rules:
  #  1. Eval ops never branch on isinstance; the reduction algorithm is selected purely by the injected ff_ctx.
  #  2. Defaults are correct for strict standard-format reductions (Barrett/Shoup): modular_reduction returns canonical [0, q) residues, format hooks are identity.
  #  3. Lazy/non-standard contexts (Montgomery) override only what differs.

  def strictify(self, x):
    """Canonicalize a single modular_reduction output into [0, q); identity for strict backends.

    Ops call it unconditionally after an NTT/modmul so the op-boundary contract
    (strict residues) holds for every backend.
    """
    return x

  def strictify_after_accumulation(self, x, num_terms):
    """Canonicalize a SUM of num_terms modular_reduction outputs into [0, q).

    Lazy Montgomery data must be normalized with conditional subtracts instead
    (a second reduction would strip an R factor); MontgomeryContext overrides.
    """
    return self.modular_reduction(x)

  def reduce_scaled_sum(self, main, scale, addend):
    """Return canonicalize(main*scale + addend) in computation format.

    addend sits one modular_reduction away from main*scale. Strict backends fuse
    into a single reduction; MontgomeryContext overrides (reduce the product
    first, conditional-subtract the lazy sum).
    """
    prod = main.astype(jnp.uint64) * scale.astype(jnp.uint64)
    return self.modular_reduction(prod + addend.astype(jnp.uint64))

  def encode_prescale_constant(self, c):
    """Encode a constant multiplying STANDARD-format data so one modular_reduction of the product lands in computation format.

    Identity for strict backends; Montgomery scales by R^2 so
    MontRed(standard * c*R^2) = standard*c*R.
    """
    once = self.to_computation_format(jnp.asarray(c).astype(jnp.uint64))
    return self.to_computation_format(jnp.asarray(once).astype(jnp.uint64))

  def get_jax_parameters(self):
    return {}

  def validate_moduli(self, moduli):
    """Raise if moduli fall outside this backend's supported envelope.

    Default: no constraint (Barrett/Shoup tolerate the full 32-bit range);
    Montgomery overrides (lazy REDC wrap-safe only for q < 2^31).
    """
    return

  def modular_reduction(self, a: jnp.ndarray) -> jnp.ndarray:
    raise NotImplementedError("Subclasses must implement this method")

  def modular_reduction_single_modulus(self, a: jnp.ndarray, limb_index: int) -> jnp.ndarray:
    raise NotImplementedError("Subclasses must implement this method")

  def drop_last_modulus(self):
    raise NotImplementedError("Subclasses must implement this method")

  def slice(self, num_moduli: int) -> "FiniteFieldContextBase":
    raise NotImplementedError("Subclasses must implement this method")

  def concat(self, other: "FiniteFieldContextBase") -> "FiniteFieldContextBase":
    raise NotImplementedError("Subclasses must implement this method")


########################
# Montgomery Modulus Reduction Context
########################
class MontgomeryContext(FiniteFieldContextBase):

  ntt_ciphertext_context_cls = "NTTCiphertextMontgomeryContext"
  bconv_cls = "BConvMontgomery"
  computation_format_is_standard = False

  def __init__(self, moduli: Union[List[int], int]):
    super().__init__(moduli)
    self.moduli = moduli
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    self.w = 32
    self.w_inv = [util.modinv(1 << self.w, m) for m in self.moduli]
    self.w_inv_reduction = jnp.array(self.w_inv, jnp.uint64)

    self.moduli_reduction = jnp.array(self.moduli, jnp.uint64)

    self.moduli_inv_32 = [util.modinv(m, 2**32) for m in self.moduli]
    self.moduli_low16 = [m & 0xFFFF for m in self.moduli]
    self.moduli_high16 = [m >> 16 for m in self.moduli]

    self.q = jnp.array(self.moduli, dtype=jnp.uint32)
    self.q_low = jnp.array(self.moduli_low16, dtype=jnp.uint32)
    self.q_high = jnp.array(self.moduli_high16, dtype=jnp.uint32)
    self.q_inv_32 = jnp.array(self.moduli_inv_32, dtype=jnp.uint32)

    # Validate at construction: a directly built context (no NTT ctx) otherwise silently accepts q >= 2^31.
    self.validate_moduli(self.moduli)

  def to_computation_format(self, a: int):
    # return [(a * (1 << self.w)) % m for m in self.moduli] # The algorithm being performed
    # uint64 before the << 32 shift: a uint32 input would wrap to zero.
    a = jnp.asarray(a, jnp.uint64)
    return ((a << self.w) % self.moduli_reduction).astype(jnp.uint32)

  def to_original_format(self, a: jnp.ndarray):
    return (a * self.w_inv_reduction) % self.moduli_reduction

  def get_jax_parameters(self):
    return {
        "moduli": util.to_tuple(self.moduli),
        "moduli_inv_32": util.to_tuple(self.moduli_inv_32),
        "moduli_low": util.to_tuple(self.moduli_low16),
        "moduli_high": util.to_tuple(self.moduli_high16),
    }

  def validate_moduli(self, moduli):
    # Lazy REDC is wrap-safe only for q < 2^31: its step-2 product 2q^2 must
    # stay under 2^32*(2^32-q); an out-of-envelope q ~ 2^31 silently zeros the
    # result. Raise (not assert) so the guard survives `python -O`.
    bad = [
        q for q in ([moduli] if isinstance(moduli, int) else moduli)
        if q >= 2**31
    ]
    if bad:
      raise ValueError(
          f"Montgomery reduction requires moduli < 2**31; got {bad}"
      )

  def modular_reduction(self, z: jnp.ndarray) -> jnp.ndarray:
    """Montgomery reduction from u64 to u32 optimized version using only 32-bit operations

    Args:
        z: - is u64 array of shape (B, M) - input

    parameters:
        moduli:
            - Tuple parameters constants
            - is u32 array of shape (M)
            - modular or moduli
        moduli_low:
            - Tuple parameters constants
            - is u32 array of shape (M)
            - low 16 bits of modular or moduli
        moduli_high:
            - Tuple parameters constants
            - is u32 array of shape (M)
            - high 16 bits of modular or moduli
        moduli_inv_32:
            - Tuple parameters constants
            - is u32 array of shape (M)
            - modular inverse of q mod 2^32
    Returns:
        - is u32 array of shape (B, M)
        - output
        - reduced value
    """

    # Local constants
    MASK32 = 0xFFFFFFFF
    MASK16 = 0xFFFF
    SHIFT16 = 16
    SHIFT32 = 32
    # Ensure dimensions for broadcasting
    q = self.q
    q_low = self.q_low
    q_high = self.q_high
    q_inv_32 = self.q_inv_32

    # Computation
    z_low = z.astype(jnp.uint32)
    z_high = (z >> SHIFT32).astype(jnp.uint32)
    t = (z_low * q_inv_32) & MASK32
    t_low = t & MASK16
    t_high = (t >> SHIFT16) & MASK16

    prod_high = t_high * q_high  # This contributes directly to upper 32 bits
    prod_mid_high = t_high * q_low  # Upper 16 bits go to upper 32 bits
    prod_mid_low = t_low * q_high  # Upper 16 bits go to upper 32 bits
    prod_low = t_low * q_low  # Upper 16 bits contribute to middle part
    mid_low = (
        (prod_mid_high & MASK16)
        + (prod_mid_low & MASK16)
        + (prod_low >> SHIFT16)
    )
    mid_high = (
        (prod_mid_high >> SHIFT16)
        + (prod_mid_low >> SHIFT16)
        + (mid_low >> SHIFT16)
    )

    # Final upper 32 bits
    t_final = prod_high + mid_high
    b = z_high + q - t_final
    # Ensure strict reduction
    # b = jnp.where(b >= q, b - q, b).astype(jnp.uint32)
    return b.astype(jnp.uint32)

  # Montgomery overrides: modular_reduction is LAZY (returns [0, 2q)) and data lives in computation
  # format x*R mod q, so these canonicalize/combine with conditional subtracts instead of extra
  # reductions — an extra reduction would strip an R factor.

  def _strictify_from_multiple(self, x, bound_multiple):
    """Reduce x from [0, bound_multiple*q) to [0, q) via a compare-select chain (a Montgomery reduction here would strip an R factor)."""
    q = jnp.asarray(self.moduli_reduction, dtype=x.dtype)
    m = 1
    while m * 2 < bound_multiple:
      m *= 2
    while m >= 1:
      mq = m * q
      x = jnp.where(x >= mq, x - mq, x)
      m //= 2
    return x

  def strictify(self, x):
    q = jnp.asarray(self.moduli_reduction, dtype=x.dtype)
    return jnp.where(x >= q, x - q, x)

  def strictify_after_accumulation(self, x, num_terms):
    # Each summed product is a lazy Montgomery reduction in [0, 2q); a sum of
    # `num_terms` of them is in [0, 2*num_terms*q).
    return self._strictify_from_multiple(x, 2 * num_terms)

  def reduce_scaled_sum(self, main, scale, addend):
    # main*scale reduces once (MontRed) to the R^1 product, lazy in [0, 2q);
    # adding the R^1 `addend` (also lazy) yields [0, 4q).  A second MontRed
    # would strip an R factor, so normalize with conditional subtracts.
    prod = main.astype(jnp.uint64) * scale.astype(jnp.uint64)
    reduced = self.modular_reduction(prod).astype(jnp.uint64)
    combined = reduced + addend.astype(jnp.uint64)
    return self._strictify_from_multiple(combined, 4)

  def drop_last_modulus(self):
    # self.moduli_reduction, self.moduli_inv_32, self.moduli_low16, self.moduli_high16 are not updated here.
    # Because they are not used in the reduction.
    # self.moduli = self.moduli[:-1]
    self.moduli_reduction = self.moduli_reduction[:-1]
    self.q = self.q[:-1]
    self.q_low = self.q_low[:-1]
    self.q_high = self.q_high[:-1]
    self.q_inv_32 = self.q_inv_32[:-1]

  def slice(self, num_moduli: int) -> "MontgomeryContext":
    """Return a view over the first `num_moduli` entries.

    Montgomery reduction is element-wise per modulus, so slicing the
    parameter arrays produces a valid context for a moduli prefix (the
    mirror of BarrettContext.slice). JAX array slicing shares memory —
    no data is copied. The moduli were validated (< 2^31) at the parent's
    construction, so no re-validation is needed.
    """
    if num_moduli > len(self.moduli):
      raise ValueError(
          f"num_moduli ({num_moduli}) exceeds moduli count ({len(self.moduli)})"
      )
    if num_moduli < 1:
      raise ValueError(f"num_moduli must be >= 1, got {num_moduli}")
    ctx = object.__new__(MontgomeryContext)
    ctx.moduli = self.moduli[:num_moduli]
    ctx.w = self.w
    ctx.w_inv = self.w_inv[:num_moduli]
    ctx.w_inv_reduction = self.w_inv_reduction[:num_moduli]
    ctx.moduli_reduction = self.moduli_reduction[:num_moduli]
    ctx.moduli_inv_32 = self.moduli_inv_32[:num_moduli]
    ctx.moduli_low16 = self.moduli_low16[:num_moduli]
    ctx.moduli_high16 = self.moduli_high16[:num_moduli]
    ctx.q = self.q[:num_moduli]
    ctx.q_low = self.q_low[:num_moduli]
    ctx.q_high = self.q_high[:num_moduli]
    ctx.q_inv_32 = self.q_inv_32[:num_moduli]
    return ctx

  def concat(self, other: "MontgomeryContext") -> "MontgomeryContext":
    """Concatenate this context with another to form a combined context.

    Used to build Q+P contexts from separate Q and P contexts. Montgomery
    reduction is element-wise, so concatenation is valid; both operands
    were validated (< 2^31) at their own construction.
    """
    if not isinstance(other, MontgomeryContext):
      raise TypeError(
          "MontgomeryContext.concat requires another MontgomeryContext, got "
          f"{type(other).__name__} (mixing reductions is silent garbage)"
      )
    ctx = object.__new__(MontgomeryContext)
    ctx.moduli = list(self.moduli) + list(other.moduli)
    ctx.w = self.w
    ctx.w_inv = list(self.w_inv) + list(other.w_inv)
    ctx.w_inv_reduction = jnp.concatenate(
        [self.w_inv_reduction, other.w_inv_reduction]
    )
    ctx.moduli_reduction = jnp.concatenate(
        [self.moduli_reduction, other.moduli_reduction]
    )
    ctx.moduli_inv_32 = list(self.moduli_inv_32) + list(other.moduli_inv_32)
    ctx.moduli_low16 = list(self.moduli_low16) + list(other.moduli_low16)
    ctx.moduli_high16 = list(self.moduli_high16) + list(other.moduli_high16)
    ctx.q = jnp.concatenate([self.q, other.q])
    ctx.q_low = jnp.concatenate([self.q_low, other.q_low])
    ctx.q_high = jnp.concatenate([self.q_high, other.q_high])
    ctx.q_inv_32 = jnp.concatenate([self.q_inv_32, other.q_inv_32])
    return ctx


########################
# Barrett Modulus Reduction Context
########################
def _mul_high_u64_from_u32_limbs(lhs, rhs):
  """Return floor(lhs * rhs / 2**64) without overflowing uint64.

  Both operands are split into 32-bit limbs. The carry ordering keeps every
  partial product and addition below 2**64, including the s=64 Barrett case
  where directly evaluating the high-limb product can require 65 bits.
  """
  mask32 = jnp.uint64(0xFFFFFFFF)
  lhs = jnp.asarray(lhs, dtype=jnp.uint64)
  rhs = jnp.asarray(rhs, dtype=jnp.uint64)
  lhs_low = lhs & mask32
  lhs_high = lhs >> jnp.uint64(32)
  rhs_low = rhs & mask32
  rhs_high = rhs >> jnp.uint64(32)

  low_product = lhs_low * rhs_low
  cross = lhs_high * rhs_low + (low_product >> jnp.uint64(32))
  cross_low = cross & mask32
  cross_high = cross >> jnp.uint64(32)
  other_cross = lhs_low * rhs_high + cross_low
  return (
      lhs_high * rhs_high
      + cross_high
      + (other_cross >> jnp.uint64(32))
  )


class BarrettContext(FiniteFieldContextBase):

  ntt_ciphertext_context_cls = "NTTCiphertextBarrettContext"
  bconv_cls = "BConvBarrett"

  def __init__(self, moduli: Union[List[int], int]):
    super().__init__(moduli)
    self.moduli = moduli
    if type(self.moduli) is int:
      self.moduli = [self.moduli]

    # Use int(m) to prevent JAX float32 truncation: when self.moduli comes from
    # a jnp.array, each m is a JAX scalar and arithmetic falls back to float32
    # (23-bit mantissa), corrupting Barrett constants for moduli > 2^24.
    self.barrett_s = [2 * math.ceil(math.log2(int(m))) for m in self.moduli]
    self.barrett_w = [min(s, 32) for s in self.barrett_s]
    self.barrett_s_w = [
        s - w for s, w in zip(self.barrett_s, self.barrett_w, strict=True)
    ]
    self.barrett_m = [
        math.floor(2**s / int(m))
        for s, m in zip(self.barrett_s, self.moduli, strict=True)
    ]
    # used for run-time reduction
    self.m = jnp.array(self.barrett_m, dtype=jnp.uint64)
    self.moduli_reduction = jnp.array(self.moduli, dtype=jnp.uint64)
    self.w = jnp.array(self.barrett_w, dtype=jnp.uint16)
    self.s_w = jnp.array(self.barrett_s_w, dtype=jnp.uint16)

  def to_computation_format(self, a):
    return a

  def to_original_format(self, a):
    return a

  def get_jax_parameters(self):
    return {
        "barrett_m": util.to_tuple(self.barrett_m),
        "moduli": util.to_tuple(self.moduli),
        "barrett_w": util.to_tuple(self.barrett_w),
        "barrett_s_w": util.to_tuple(self.barrett_s_w),
    }

  def modular_reduction(self, z: jnp.ndarray) -> jnp.ndarray:
    """Vectorized implementation of the Barrett reduction.

    Supports every uint32 modulus. Inputs must satisfy
    ``z < 2**(2*ceil(log2(q)))``. For full-width moduli ``q > 2**31`` that
    exponent is 64, so the quotient estimate uses an overflow-free 32-bit-limb
    multiply-high.

    This implementation sets the internal shift width `w` to `min(s, 32)` so it
    works with small modulus `moduli < 2^16`.

    Args:
        z: The input value.
        moduli: The RNS moduli.
        s_w: The bit width of moduli.
        w: The internal shift width.
        m: The precomputed value for Barrett reduction.

    Returns:
        The result of the Barrett reduction.
    """
    m = self.m
    moduli = self.moduli_reduction
    w = self.w
    s_w = self.s_w

    # HERot/CKKS moduli use w=32. Keep that common case static so TPU HLO does
    # not rebuild (1 << w) - 1 at every reduction site. The dynamic mask is
    # still required for mixed/small-modulus contexts where w < 32.
    if all(s >= 32 for s in self.barrett_s):
      z1 = z & jnp.uint64(0xFFFFFFFF)
    else:
      mask = (jnp.uint64(1) << w.astype(jnp.uint64)) - jnp.uint64(1)
      z1 = z & mask
    z2 = z >> w
    fast_quotient = (((z1 * m) >> w) + (z2 * m)) >> s_w
    if any(s == 64 for s in self.barrett_s):
      full_width_quotient = _mul_high_u64_from_u32_limbs(z, m)
      if all(s == 64 for s in self.barrett_s):
        t = full_width_quotient
      else:
        t = jnp.where(s_w == 32, full_width_quotient, fast_quotient)
    else:
      # Preserve the original low-cost kernel for common q < 2**31 contexts.
      t = fast_quotient
    z = z - t * moduli
    pred = z >= moduli
    return jnp.where(pred, z - moduli, z).astype(jnp.uint32)
    # return (z - moduli * pred).astype(jnp.uint32)

  def modular_reduction_single_modulus(
      self, z: jnp.ndarray, modulus_index: int
  ) -> jnp.ndarray:
    """Vectorized implementation of the Barrett reduction.

    Supports every uint32 modulus. Inputs must satisfy
    ``z < 2**(2*ceil(log2(q)))``. For full-width moduli ``q > 2**31``, the
    s=64 case uses the same overflow-free multiply-high quotient as the
    multi-modulus path.

    This implementation sets the internal shift width `w` to `min(s, 32)` so it
    works with small modulus `moduli < 2^16`.

    Args:
        z: The input value.
        moduli: The RNS moduli.
        s_w: The bit width of moduli.
        w: The internal shift width.
        m: The precomputed value for Barrett reduction.

    Returns:
        The result of the Barrett reduction.
    """
    m = self.m[modulus_index]
    moduli = self.moduli_reduction[modulus_index]
    w = self.w[modulus_index]
    s_w = self.s_w[modulus_index]

    # Avoid materializing a dynamic mask for the common static w=32 case.
    if all(s >= 32 for s in self.barrett_s):
      z1 = z.astype(jnp.uint32)
    else:
      mask = (jnp.uint64(1) << w.astype(jnp.uint64)) - jnp.uint64(1)
      z1 = (z & mask).astype(jnp.uint32)
    z2 = (z >> w).astype(jnp.uint32)
    fast_quotient = (((z1 * m) >> w) + (z2 * m)) >> s_w
    if any(s == 64 for s in self.barrett_s):
      full_width_quotient = _mul_high_u64_from_u32_limbs(z, m)
      t = jnp.where(s_w == 32, full_width_quotient, fast_quotient)
    else:
      t = fast_quotient
    z = z - t * moduli
    pred = z >= moduli
    return jnp.where(pred, z - moduli, z).astype(jnp.uint32)
    # return (z - moduli * pred).astype(jnp.uint32)

  def drop_last_modulus(self):
    # barrett_s drives the static s=64 kernel selection and must stay aligned
    # with the runtime arrays. The other Python lists are precomputation-only.
    # self.moduli = self.moduli[:-1]
    self.barrett_s = self.barrett_s[:-1]
    self.m = self.m[:-1]
    self.moduli_reduction = self.moduli_reduction[:-1]
    self.w = self.w[:-1]
    self.s_w = self.s_w[:-1]

  def slice(self, num_moduli: int) -> "BarrettContext":
    """Return a view over the first `num_moduli` entries.

    Barrett reduction is element-wise per modulus, so slicing the
    parameter arrays produces a valid context for a moduli prefix.
    JAX array slicing shares memory — no data is copied.

    Args:
        num_moduli: Number of moduli to keep (must be in [1, len(self.moduli)]).

    Returns:
        A new BarrettContext with sliced parameter arrays.
    """
    if num_moduli > len(self.moduli):
      raise ValueError(
          f"num_moduli ({num_moduli}) exceeds moduli count ({len(self.moduli)})"
      )
    if num_moduli < 1:
      raise ValueError(f"num_moduli must be >= 1, got {num_moduli}")
    ctx = object.__new__(BarrettContext)
    ctx.moduli = self.moduli[:num_moduli]
    ctx.barrett_s = self.barrett_s[:num_moduli]
    ctx.m = self.m[:num_moduli]
    ctx.moduli_reduction = self.moduli_reduction[:num_moduli]
    ctx.w = self.w[:num_moduli]
    ctx.s_w = self.s_w[:num_moduli]
    return ctx

  def concat(self, other: "BarrettContext") -> "BarrettContext":
    """Concatenate this context with another to form a combined context.

    Useful for constructing Q+P contexts from separate Q and P contexts.
    Barrett reduction is element-wise, so concatenation is valid.

    Args:
        other: Another BarrettContext to append.

    Returns:
        A new BarrettContext with concatenated parameter arrays.
    """
    ctx = object.__new__(BarrettContext)
    ctx.moduli = list(self.moduli) + list(other.moduli)
    ctx.barrett_s = list(self.barrett_s) + list(other.barrett_s)
    ctx.m = jnp.concatenate([self.m, other.m])
    ctx.moduli_reduction = jnp.concatenate(
        [self.moduli_reduction, other.moduli_reduction]
    )
    ctx.w = jnp.concatenate([self.w, other.w])
    ctx.s_w = jnp.concatenate([self.s_w, other.s_w])
    return ctx


########################
# Shoup Modulus Reduction Context
########################
class ShoupContext(FiniteFieldContextBase):

  def __init__(self, moduli: Union[List[int], int]):
    super().__init__(moduli)
    self.moduli = moduli
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    self.moduli_reduction = jnp.array(self.moduli, jnp.uint64)
    self.q = jnp.array(self.moduli, dtype=jnp.uint64)
    self.w = 32

  def to_computation_format(self, a: jnp.ndarray):
    # return [(a % m) for m in self.moduli] # The algorithm being performed
    return (a % self.moduli_reduction).astype(jnp.uint32)

  def to_original_format(self, a: jnp.ndarray):
    return (a % self.moduli_reduction).astype(jnp.uint32)

  def precompute_constant_operand(self, a: int):
    # return [(a * (1 << self.w)) // m for m in self.moduli] # The algorithm being performed
    return (a << self.w) // self.moduli_reduction

  def get_jax_parameters(self):
    return {
        "moduli": util.to_tuple(self.moduli),
    }

  def modular_reduction(self, z: jnp.ndarray, z_s: jnp.ndarray) -> jnp.ndarray:
    """Shoup's reduction from u64 to u32

    Args:
        z: - is u64 array of shape (B, M) - input - z = a * b
        z_s: - is u64 array of shape (B, M) - input - z_s = a * b_s - b_s is b
          in Shoup's precomputation format

    parameters:
        moduli:
            - Tuple parameters constants
            - is u32 array of shape (M)
            - modular or moduli
    Returns:
        - is u32 array of shape (B, M)
        - output
        - reduced value
    """
    t = z_s >> 32
    u = z - t * self.q
    # Ensure strict reduction: u can be in [0, 2q), needs conditional subtract
    # for moduli >= 2^31 where 2q overflows uint32.
    u = jnp.where(u >= self.q, u - self.q, u)
    return u.astype(jnp.uint32)

  def drop_last_modulus(self):
    # self.moduli is not updated here.
    # Because it is used in the precomputation.
    # self.moduli = self.moduli[:-1]
    self.moduli_reduction = self.moduli_reduction[:-1]
    self.q = self.q[:-1]


########################
# BAT Lazy Reduction Context
########################
class BATLazyContext(FiniteFieldContextBase):

  ntt_ciphertext_context_cls = "NTTCiphertextBATLazyContext"
  bconv_cls = "BConvBATLazy"

  def __init__(self, moduli: Union[List[int], int]):
    super().__init__(moduli)
    self.moduli = moduli
    if type(self.moduli) is int:
      self.moduli = [self.moduli]

    # L=4 bytes (for 32-bit modulus)
    self.L = 4

    # Precompute R matrix for each modulus
    # R_i,j corresponds to the j-th byte of (256^(i+L) mod q)
    # Dimensions: (M, 4, 4) because we have 4 high-bytes (B) and 4 result-bytes (L)
    moduli_arr = jnp.array(self.moduli, dtype=jnp.uint64)

    # 1. Vectorize 'i' loop (bytes 4, 5, 6, 7): Compute r_val = 256^(i+4) % m
    shifts_i = jnp.arange(4, 8, dtype=jnp.uint64) * 8
    # Broadcast shape: (1, 4) vs (M, 1) -> (M, 4)
    r_vals = (jnp.array(1, dtype=jnp.uint64) << shifts_i[None, :]) % moduli_arr[
        :, None
    ]

    # 2. Vectorize 'j' loop: Split r_vals into 4 bytes (little endian)
    shifts_j = jnp.arange(4, dtype=jnp.uint64) * 8
    # Result: (M, 4, 4)
    self.R = ((r_vals[:, :, None] >> shifts_j[None, None, :]) & 0xFF).astype(
        jnp.uint8
    )
    self.moduli_reduction = jnp.array(self.moduli, jnp.uint64)

  def to_computation_format(self, a: int):
    return a

  def to_original_format(self, a: jnp.ndarray):
    return (a % self.moduli_reduction).astype(jnp.uint32)

  def get_jax_parameters(self):
    return {"moduli": util.to_tuple(self.moduli), "R": self.R}

  def modular_reduction(self, z: jnp.ndarray) -> jnp.ndarray:
    """BAT Lazy Reduction from u64 to u32

    Implements: result = B @ R + A
    where z is split into Lower Part A (bytes 0-3) and Higher Part B (bytes
    4-7).

    Args:
        z: u64 array of shape (..., M) if RNS, or arbitrary shape if single
          modulus.

    Returns:
        u32 array (Partially reduced)
    """
    # 1. Extract bytes from z using bitcast
    # This treats the 64-bit integers as vectors of 8 bytes (Little Endian)
    z_bytes = jax.lax.bitcast_convert_type(
        z.astype(jnp.uint64), new_dtype=jnp.uint8
    )

    # 2. Split into Lower Part A (bytes 0-3) and Higher Part B (bytes 4-7)
    # A_bytes, B_bytes each have shape (..., 4) where ... matches z's shape.
    A_bytes, B_bytes = jnp.split(z_bytes, 2, axis=-1)

    # 3. Perform Matrix Multiplication: LazyReductionResult = B @ R + A
    # Logic:
    # - If we have a single modulus (M=1), we assume ALL input elements should be
    #   reduced by this same modulus, regardless of input shape dimensions.
    # - If we have multiple moduli (M>1), we assume the LAST dimension of input
    #   corresponds to the moduli dimension M.

    # Unified implementation for both Single Modulus and RNS
    # Use einsum for automatic broadcasting and hardware-efficient 8-bit matmul
    # - Single Modulus: B (..., 4) @ R_squeezed (4, 4) -> (..., 4)
    # - RNS: B (..., M, 4) @ R (M, 4, 4) -> (..., M, 4)
    # Note: jnp.squeeze ensures R is (4, 4) when M=1, matching the "Single Modulus" lack of M-dim.
    # We perform the input in 8-bit and accumulate in 32-bit for TPU efficiency.
    matmul_res = jnp.einsum(
        "...i,...ij->...j", B_bytes, self.R, preferred_element_type=jnp.uint32
    )

    # 4. Add Lower Part A
    result_bytes = matmul_res + A_bytes

    # 5. Reconstruct integer
    return util.reconstruct(result_bytes)

  def drop_last_modulus(self):
    self.moduli = self.moduli[:-1]
    self.R = self.R[:-1]

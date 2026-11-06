"""Polynomial-Plaintext Multiplication for CKKS on TPU.

Implements ct-pt multiply matching OpenFHE's EvalMult(ciphertext, plaintext):
  For each element c[i] in the ciphertext: result[i] = c[i] * pt (mod q per
  tower)
  No relinearization is needed (output stays 2-element).

TPU optimization considerations:
  - MXU: 8-bit integer matrix multiply engine — used via BAT (Basis Aligned
    Transformation) where the plaintext is precomputed into a byte-decomposed
    format offline.
  - VPU: 32-bit vectorized arithmetic at (8, 128) SIMD granularity — used for
    Barrett reduction and the standard modmul fallback path.
  - Data layout: (batch, num_elements, r, c, num_moduli) where (r, c) tiles map
    to TPU's (8, 128) processing granularity at large degrees.

When BAT helps
--------------
BAT only pays off when the underlying op is structurally a matmul (a
contraction sums many u32 products). It then trades 4x work-inflation per
scalar mul for MXU peak throughput at 8-bit precision, amortizing the
byte-decomposition over the contraction (NTT twiddles, BConv, key-switch).

For pointwise ops like ct x pt there is no contraction: the VPU runs it as one
HBM-bound fused kernel, while BAT only adds byte-decomposition and layout
shuffles. Default is VPU; reach for BAT inside ntt_mm.py / bconv.py, not here.

Measured on TPU v6e (batch 1, 2 elements, warm median of 5):

    degree   limbs   mul_vpu    mul_bat     ratio
      4096       4    160 us     819 us       5.1x
     16384      16    159 us    10.6 ms      66.4x
     65536      51    238 us     134 ms     566.0x

The gap widens with size because the VPU path is launch-bound (roughly flat
from 2^12 to 2^16) while BAT's cost grows with the number of tiny matmuls.
Two effects compose, and neither is a tuning problem:

  1. The einsum contracts only the 4-byte axis. The modulus index ``m`` is a
     batched index, not a contraction -- different moduli are independent, so
     summing over them would be wrong. That leaves a depth-4 contraction on a
     128-wide MXU (~3% utilization) spread over batch*elems*r*c*M independent
     4x4x4 matmuls: 6.7M of them at N=65536/M=51. The einsum alone is already
     14.6 ms, 73x the entire VPU multiply.
  2. Reconstructing the u64 product from the byte partials costs a further ~9x
     (14.6 ms -> 134 ms). TPU has no native 64-bit integer unit, so the u64
     accumulate is emulated in the matmul's output loop. This is not a
     separable fusion artifact: inserting ``lax.optimization_barrier`` between
     the einsum and the reconstruct changes nothing (measured 1.0x), because
     the consumer's dtype is what forces the schedule.
"""

import jax
import jax.numpy as jnp
import finite_field
import polynomial
import util

Polynomial = polynomial.Polynomial
# Aliased only for the _HEPtCtMulKernel.__init__ default argument.
BarrettContext = finite_field.BarrettContext
jax.config.update("jax_enable_x64", True)


class _HEPtCtMulKernel:
  """Private polynomial-plaintext multiplication kernel.

  Default path is VPU (element-wise modmul). BAT is opt-in via
  ``mul(ct, use_bat=True)`` after ``precompute_plaintext_bat`` and is only
  beneficial when the workload can amortize the MXU tile and byte-decomposition
  overhead; for standalone ct*pt at typical ring dimensions the VPU path is
  significantly faster.

  Architecture:
    Standard path (VPU, default):
      result = barrett_reduce(ct * pt)   — element-wise u64 multiply + Barrett
      on VPU

    BAT path (MXU + VPU, opt-in):
      1. Offline: precompute pt_bat[r, c, M*4, 4] via
      util.shifted_mod_bytes
      2. Runtime: ct_bytes = bitcast(ct, u8) → shape (..., r, c, M, 4)
                  reshape ct_bytes to (..., r, c*M, 4) to increase MXU
                  contraction width
                  partial = einsum("...nq, nqb -> ...nb", ct_bytes, pt_bat_flat)
                  [8-bit MXU]
                  result = sum(partial << [0,8,16,24])   [VPU shift+add]
                  result = barrett_reduce(result)        [VPU]

    The BAT path moves the u32×u32 multiply onto MXU by decomposing it into
    four u8×u8→u32 partial products (the schoolbook multiplication of 4-byte
    integers), which maps directly to TPU's 8-bit matrix multiply unit.
  """

  def __init__(self, batch, r, c, moduli, degree_layout=None,
               finite_field_context=BarrettContext):
    self.batch = batch
    self.r = r
    self.c = c
    self.moduli = moduli
    self.num_moduli = len(moduli)
    self.degree_layout = finite_field.canonical_degree_layout(
        r, c, degree_layout, '_HEPtCtMulKernel'
    )
    self.ring_dim = r * c
    self.ff_context_cls = finite_field_context
    self.ff_ctx = finite_field_context(moduli=moduli)
    self.pt_bat = None  # Set by precompute_plaintext_bat
    self.pt_ntt = None  # Set by set_plaintext

  def set_plaintext(self, pt_ntt: jnp.ndarray):
    """Set the plaintext polynomial (must be in NTT/EVAL form).

    The plaintext arrives in standard representation. In Montgomery mode it
    is converted to Montgomery form (pt * R mod q) so that
    MontRed(ct_mont * pt_mont) keeps the ciphertext in Montgomery form.

    Invalidates any previously precomputed BAT representation.

    Args:
        pt_ntt: Plaintext in NTT domain (standard representation), shape
          (*degree_layout, num_moduli) or (1, 1, *degree_layout, num_moduli)
          for broadcasting.
    """
    pt = jnp.asarray(pt_ntt, dtype=jnp.uint64)
    pt = self.ff_ctx.to_computation_format(pt)
    self.pt_ntt = pt.astype(jnp.uint32)
    self.pt_bat = None  # Invalidate stale BAT

  def precompute_plaintext_bat(self, pt_ntt: jnp.ndarray):
    """Offline: precompute BAT representation of plaintext for MXU path.

    Transforms each plaintext coefficient into a 4×4 byte matrix such that
    the u32×u32 element-wise multiply can be performed as u8 matrix multiply.

    For plaintext value p at position (i, j, m) with modulus q_m:
      bat[i, j, m, a, b] = b-th byte of ((p << 8*a) mod q_m)
      for a in [0..3] (input byte index), b in [0..3] (output byte index)

    The ct × pt product is then:
      result = sum_a( ct_byte[a] * bat[a, :] )  (8-bit matmul)

    Args:
        pt_ntt: Plaintext in NTT domain (standard representation), shape
          (*degree_layout, num_moduli).
    """
    pt = jnp.asarray(pt_ntt, dtype=jnp.uint64)
    # Computation-format encode (Montgomery: pt*R), as in set_plaintext.
    pt = self.ff_ctx.to_computation_format(pt).astype(jnp.uint64)
    moduli_arr = jnp.array(self.moduli, dtype=jnp.uint64)

    pt_bytes = util.shifted_mod_bytes(pt, moduli_arr)

    # Transpose to (*degree_layout, num_moduli, 4_input, 4_output)
    # = (*degree_layout, M, 4, 4)
    ndim = len(self.degree_layout)
    # Current: (4_input, *degree_layout, M, 4_output)
    # Target:  (*degree_layout, M, 4_input, 4_output)
    perm = list(range(1, 1 + ndim)) + [1 + ndim, 0, 1 + ndim + 1]
    pt_bat = pt_bytes.transpose(perm)

    # Reshape for efficient MXU: fold (M, 4_input) into one dimension
    # (*degree_layout, M*4, 4)
    spatial_shape = pt_bat.shape[:ndim]
    self.pt_bat = pt_bat.reshape(*spatial_shape, self.num_moduli * 4, 4)
    self.pt_ntt = pt.astype(jnp.uint32)

  def _mul_array(self, ct_data, pt_ntt):
    """Privately multiply a canonical raw payload using the VPU path."""
    finite_field.check_rank5_array(
        ct_data,
        '_HEPtCtMulKernel._mul_array',
        batch=self.batch,
        degree_layout=self.degree_layout,
        num_moduli=self.num_moduli,
    )
    if not hasattr(pt_ntt, 'shape') or pt_ntt.shape[-3:] != (
        *self.degree_layout, self.num_moduli
    ):
      raise ValueError(
          '_HEPtCtMulKernel._mul_array plaintext must end in '
          f'{(*self.degree_layout, self.num_moduli)}, got '
          f'{getattr(pt_ntt, "shape", None)}.'
      )
    return self._mul_array_unchecked(ct_data, pt_ntt)

  def _mul_array_unchecked(self, ct_data, pt_ntt):
    product = ct_data.astype(jnp.uint64) * pt_ntt.astype(jnp.uint64)
    reduced = self.ff_ctx.modular_reduction(product)
    return self.ff_ctx.strictify(reduced).astype(jnp.uint32)

  def mul_vpu(self, ct: Polynomial) -> Polynomial:
    """Polynomial-plaintext multiply using VPU (element-wise modmul).

    This is the simple path: u32 → u64 multiply → Barrett reduction → u32.
    Maps entirely to TPU's vector processing unit.

    Args:
        ct: Polynomial with shape (batch, num_elements, *degree_layout,
          num_moduli).

    Returns:
        Polynomial with same shape (ct * pt mod q per tower).
    """
    if self.pt_ntt is None:
      raise RuntimeError("Plaintext not set. Call set_plaintext() first.")

    finite_field.check_ct_operand(
        self.ff_context_cls,
        self.num_moduli,
        ct,
        '_HEPtCtMulKernel.mul',
        batch=self.batch,
        degree_layout=self.degree_layout,
        moduli=self.moduli,
    )
    return ct._clone_with_payload(
        self._mul_array_unchecked(ct.polynomial, self.pt_ntt)
    )

  def mul_bat(self, ct: Polynomial) -> Polynomial:
    """Polynomial-plaintext multiply using BAT (MXU-accelerated).

    Decomposes the ciphertext into 4 bytes per coefficient, then uses
    8-bit matrix multiply against the precomputed plaintext BAT matrix.
    This offloads the u32×u32 multiply to TPU's MXU (8-bit matmul engine),
    leaving only the Barrett reduction on the VPU.

    Memory access pattern:
      - ct bytes: (batch, num_elements, *degree_layout, M, 4) — contiguous read
      - pt_bat:   (*degree_layout, M, 4, 4) — precomputed constant, broadcast
      - The contraction is only the 4-byte axis. ``M`` is a *batched* index,
        not a contraction: each modulus is an independent residue and summing
        across them would be wrong. So this is a batch of 4x4x4 matmuls, not
        one wide matmul, and the MXU runs at ~3% utilization. See the module
        docstring for the measured cost -- this path is much slower than
        ``mul_vpu`` and exists for comparison, not for production use.
      - pt_bat is 16 bytes per coefficient against the plaintext's 4, so it
        also reads 4x the constant traffic of the VPU path.

    Args:
        ct: Polynomial with shape (batch, num_elements, *degree_layout,
          num_moduli).

    Returns:
        Polynomial with same shape (ct * pt mod q per tower).
    """
    if self.pt_bat is None:
      raise RuntimeError(
          "BAT plaintext not precomputed. Call precompute_plaintext_bat()"
          " first."
      )

    finite_field.check_ct_operand(
        self.ff_context_cls,
        self.num_moduli,
        ct,
        '_HEPtCtMulKernel.mul',
        batch=self.batch,
        degree_layout=self.degree_layout,
        moduli=self.moduli,
    )

    ct_data = ct.polynomial  # (batch, elems, r, c, M) u32

    # Byte-decompose ciphertext: u32 → 4×u8
    # (batch, elems, r, c, M, 4)
    ct_bytes = jax.lax.bitcast_convert_type(
        ct_data.astype(jnp.uint32), jnp.uint8
    )

    # Use the per-modulus BAT (unflatten pt_bat)
    pt_bat_per_m = self.pt_bat.reshape(
        *self.degree_layout, self.num_moduli, 4, 4
    )

    # einsum: contract over 4 input bytes, per spatial position and modulus
    # "bercmq, rcmqp -> bercmp"
    # b=batch, e=elements, r=r, c=c, m=moduli, q=4(in_bytes), p=4(out_bytes)
    partial = jnp.einsum(
        "bercmq, rcmqp -> bercmp",
        ct_bytes,
        pt_bat_per_m,
        preferred_element_type=jnp.uint32,
    )

    # Reconstruct u64 from 4 output bytes
    result_u64 = util.reconstruct(partial)

    # Modular reduction (VPU)
    reduced = self.ff_ctx.modular_reduction(result_u64)
    # Keep the op-boundary contract strict (see mul_vpu). Identity for Barrett.
    reduced = self.ff_ctx.strictify(reduced)
    return ct._clone_with_payload(reduced.astype(jnp.uint32))

  def mul(self, ct: Polynomial, use_bat: bool = False) -> Polynomial:
    """Polynomial-plaintext multiply.

    Defaults to the VPU (element-wise modmul) path. BAT is only used when
    the caller explicitly opts in AND a BAT plaintext has been precomputed.

    Args:
        ct: Input ciphertext.
        use_bat: Opt into the MXU/BAT path. Requires
          ``precompute_plaintext_bat`` to have been called. Default False.

    Returns:
        A new ciphertext containing ``ct * pt mod q``.
    """
    if use_bat:
      if self.pt_bat is None:
        raise RuntimeError(
            "use_bat=True but BAT plaintext not precomputed. "
            "Call precompute_plaintext_bat() first, or use the default "
            "VPU path (use_bat=False)."
        )
      return self.mul_bat(ct)
    return self.mul_vpu(ct)


__all__ = []

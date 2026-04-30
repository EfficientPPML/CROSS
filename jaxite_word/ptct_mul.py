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
shuffles. Measured 30-300x slower than VPU at N = 2^12..2^14. Default is VPU;
reach for BAT inside ntt_mm.py / bconv.py, not here.
"""

import jax
import jax.numpy as jnp
import finite_field
import polynomial

Polynomial = polynomial.Polynomial
BarrettContext = finite_field.BarrettContext
jax.config.update("jax_enable_x64", True)


class HEPtCtMul:
  """Polynomial-Plaintext multiplication with optional BAT optimization.

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
      _basis_aligned_transformation
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

  def __init__(self, batch, r, c, moduli, degree_layout=None):
    self.batch = batch
    self.r = r
    self.c = c
    self.moduli = moduli
    self.num_moduli = len(moduli)
    self.degree_layout = degree_layout or (r, c)
    self.ring_dim = r * c
    self.barrett_ctx = BarrettContext(moduli=moduli)
    self.pt_bat = None  # Set by precompute_plaintext_bat
    self.pt_ntt = None  # Set by set_plaintext

  def set_plaintext(self, pt_ntt: jnp.ndarray):
    """Set the plaintext polynomial (must be in NTT/EVAL form).

    Invalidates any previously precomputed BAT representation.

    Args:
        pt_ntt: Plaintext in NTT domain, shape (*degree_layout, num_moduli) or
          (1, 1, *degree_layout, num_moduli) for broadcasting.
    """
    self.pt_ntt = jnp.asarray(pt_ntt, dtype=jnp.uint32)
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
        pt_ntt: Plaintext in NTT domain, shape (*degree_layout, num_moduli).
    """
    pt = jnp.asarray(pt_ntt, dtype=jnp.uint64)
    moduli_arr = jnp.array(self.moduli, dtype=jnp.uint64)

    # Compute (pt << 8*byte_idx) mod q for each byte position
    # Shape: (4, *degree_layout, num_moduli)
    pt_shifted = jnp.stack(
        [(pt << (8 * byte_idx)) % moduli_arr for byte_idx in range(4)], axis=0
    ).astype(jnp.uint32)

    # Decompose each shifted value into 4 bytes
    # bitcast u32 → u8: adds trailing dim of 4
    # Shape: (4, *degree_layout, num_moduli, 4)
    pt_bytes = jax.lax.bitcast_convert_type(pt_shifted, jnp.uint8)

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
    self.pt_ntt = jnp.asarray(pt_ntt, dtype=jnp.uint32)

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

    ct_data = ct.polynomial.astype(jnp.uint64)
    pt_data = self.pt_ntt.astype(jnp.uint64)
    product = ct_data * pt_data
    reduced = self.barrett_ctx.modular_reduction(product)
    ct.polynomial = reduced.astype(jnp.uint32)
    return ct

  def mul_bat(self, ct: Polynomial) -> Polynomial:
    """Polynomial-plaintext multiply using BAT (MXU-accelerated).

    Decomposes the ciphertext into 4 bytes per coefficient, then uses
    8-bit matrix multiply against the precomputed plaintext BAT matrix.
    This offloads the u32×u32 multiply to TPU's MXU (8-bit matmul engine),
    leaving only the Barrett reduction on the VPU.

    Memory access pattern:
      - ct bytes: (batch, num_elements, *degree_layout, M*4) — contiguous read
      - pt_bat:   (*degree_layout, M*4, 4) — precomputed constant, broadcast
      - The (M*4) contraction dimension is large enough for efficient MXU tiling
        (e.g., M=50 → contraction dim = 200)

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

    ct_data = ct.polynomial  # (batch, elems, r, c, M) u32

    # Byte-decompose ciphertext: u32 → 4×u8
    # (batch, elems, r, c, M, 4)
    ct_bytes = jax.lax.bitcast_convert_type(ct_data, jnp.uint8)

    # Reshape: fold (M, 4) into contraction dimension
    # (batch, elems, r, c, M*4)
    orig_shape = ct_bytes.shape
    spatial = orig_shape[:-2]  # (batch, elems, r, c)
    ct_flat = ct_bytes.reshape(*spatial, self.num_moduli * 4)

    # 8-bit einsum on MXU: contract over M*4
    # ct_flat:  (batch, elems, r, c, M*4)   [byte-decomposed ciphertext]
    # pt_bat:   (r, c, M*4, 4)              [BAT-precomputed plaintext]
    # result:   (batch, elems, r, c, 4)      [4 output bytes per position]
    #
    # But we need per-modulus output, not collapsed.
    # The BAT matrix is block-diagonal over M:
    #   for modulus m: ct_bytes[..., m*4:(m+1)*4] × pt_bat[..., m*4:(m+1)*4, :4]
    #
    # With the flat einsum, the cross-modulus products contribute noise.
    # We need a GROUPED approach: process each modulus independently.
    #
    # Reshape to separate moduli:
    # ct_bytes: (batch, elems, r, c, M, 4)  — already have this
    # pt_bat:   (r, c, M, 4, 4)             — before flattening

    # Use the per-modulus BAT (unflatten pt_bat)
    pt_bat_per_m = self.pt_bat.reshape(
        *self.degree_layout, self.num_moduli, 4, 4
    )

    # einsum: contract over 4 input bytes, per spatial position and modulus
    # "bercmq, rcmqp -> bercmp"
    # b=batch, e=elements, r=r, c=c, m=moduli, q=4(in_bytes), p=4(out_bytes)
    shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)

    partial = jnp.einsum(
        "bercmq, rcmqp -> bercmp",
        ct_bytes,
        pt_bat_per_m,
        preferred_element_type=jnp.uint32,
    )

    # Reconstruct u64 from 4 output bytes
    result_u64 = jnp.sum(partial.astype(jnp.uint64) << shift_factors, axis=-1)

    # Barrett reduction (VPU)
    reduced = self.barrett_ctx.modular_reduction(result_u64)
    ct.polynomial = reduced.astype(jnp.uint32)
    return ct

  def mul(self, ct: Polynomial, use_bat: bool = False) -> Polynomial:
    """Polynomial-plaintext multiply.

    Defaults to the VPU (element-wise modmul) path. BAT is only used when
    the caller explicitly opts in AND a BAT plaintext has been precomputed.

    Args:
        ct: Input ciphertext.
        use_bat: Opt into the MXU/BAT path. Requires
          ``precompute_plaintext_bat`` to have been called. Default False.

    Returns:
        Modified ciphertext (ct * pt mod q).
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

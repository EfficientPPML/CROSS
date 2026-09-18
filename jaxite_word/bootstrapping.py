"""CKKS Bootstrapping for jaxite_word.

Pipeline stages (following OpenFHE FHECKKSRNS::EvalBootstrap):

    1. ModRaise -- lift level-0 ciphertext to full modulus chain.
    2. CoeffToSlot -- homomorphic encoding FFT via diagonal linear transforms.
    3. Approximate mod reduction -- Chebyshev for sin(2*pi*x)/(2*pi) + double-angle.
    4. SlotToCoeff -- inverse encoding FFT via diagonal linear transforms.

Every homomorphic operation is dispatched through the level-indexed
accessors on CKKSContext (he_mul, he_rot, he_rescale, ptct_mul).
"""

from __future__ import annotations

import copy
import gc
import math
import numbers
import os
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp

import polynomial
import util

jax.config.update("jax_enable_x64", True)


# ============================================================================
# ModRaise
# ============================================================================

def _mod_raise_array(ct_polynomial: jnp.ndarray,
                     src_moduli: Sequence[int],
                     dst_moduli: Sequence[int]) -> jnp.ndarray:
  """OpenFHE-compatible composite modulus-raising kernel.

  The input polynomial is in NTT domain. The extension must be done
  in COEFFICIENT domain (because NTT roots differ between primes),
  then each new limb is NTT'd with its own roots.

  OpenFHE's ``FHECKKSRNS::ExtendCiphertext`` centers each CRT digit while
  switching that digit's native polynomial into the enlarged basis, but it
  deliberately does *not* reduce or globally center the resulting
  interpolation sum modulo the depleted basis product.  It extends

      sum_i center_qi((a_i * QHat_i^-1) mod q_i) * QHat_i

  directly into the new limbs.  Consequently the lift retains a multiple of
  the depleted modulus Q.  That integer structure is required by the
  subsequent approximate modular reduction; replacing it with the centered
  representative changes the bootstrapping input distribution.

  Steps: INTT → unreduced CRT interpolation → extend → NTT.
  """
  import util as _util

  src_moduli = list(src_moduli)
  dst_moduli = list(dst_moduli)
  if dst_moduli[:len(src_moduli)] != src_moduli:
    raise ValueError(
        "mod_raise requires dst_moduli to begin with src_moduli; got "
        f"src={src_moduli} dst={dst_moduli}")
  num_src = len(src_moduli)
  num_dst = len(dst_moduli)
  if num_dst == num_src:
    return ct_polynomial

  arr = jnp.asarray(ct_polynomial)
  if arr.ndim != 5:
    raise ValueError(
        '_mod_raise_array requires canonical rank-5 shape '
        '(batch, num_elements, r, c, num_moduli); '
        f'got rank {arr.ndim} shape {arr.shape}.'
    )
  _, _, r_dim, c_dim, actual_num_src = arr.shape
  if actual_num_src != num_src:
    raise ValueError(
        f'_mod_raise_array received {actual_num_src} towers but src_moduli '
        f'contains {num_src}.'
    )
  poly_degree = r_dim * c_dim

  # Flatten to (num_polys, degree, num_src)
  flat = arr.reshape(-1, poly_degree, num_src)
  num_polys = flat.shape[0]

  # CRT constants
  Q = 1
  for q in src_moduli:
    Q *= int(q)
  Qi = [Q // int(q) for q in src_moduli]
  inv_crt = [
      pow(qi, -1, int(q)) for qi, q in zip(Qi, src_moduli, strict=True)
  ]
  # Process each polynomial independently
  num_new = num_dst - num_src
  added_list = []

  for p_idx in range(num_polys):
    poly_ntt = flat[p_idx]  # (poly_degree, num_src)

    # Step 1: INTT each source limb to coefficient domain
    coefs_per_mod = []
    for i, q in enumerate(src_moduli):
      ntt_vals = [int(poly_ntt[j, i]) for j in range(poly_degree)]
      psi = _util.root_of_unity(2 * poly_degree, q)
      coefs = _util.intt_negacyclic_bit_reverse(
          _util.bit_reverse_array(ntt_vals), q, psi)
      coefs_per_mod.append(coefs)

    # Step 2: form OpenFHE's signed-digit, unreduced CRT interpolation sum.
    # DCRTPolyImpl::operator=(PolyType) centers each native CRT digit during
    # SwitchModulus.  Do not subsequently apply ``% Q`` or globally center:
    # the resulting v*Q difference from the canonical lift is intentional.
    coefs_lifted = []
    for j in range(poly_degree):
      combined = 0
      for i in range(num_src):
        crt_digit = (coefs_per_mod[i][j] * inv_crt[i]) % src_moduli[i]
        if crt_digit > src_moduli[i] // 2:
          crt_digit -= src_moduli[i]
        combined += crt_digit * Qi[i]
      coefs_lifted.append(combined)

    # Step 3: For each new prime: reduce coefficients + NTT
    for m_idx, q in enumerate(dst_moduli[num_src:]):
      coefs_q = [c % int(q) for c in coefs_lifted]
      psi = _util.root_of_unity(2 * poly_degree, q)
      ntt_vals = _util.bit_reverse_array(
          _util.ntt_negacyclic_bit_reverse(coefs_q, q, psi))
      added_list.append(jnp.array(ntt_vals, dtype=ct_polynomial.dtype))

  # Reassemble: added_list has (num_polys * num_new) arrays of (poly_degree,)
  added_arr = jnp.stack(added_list).reshape(num_polys, num_new, poly_degree)
  added_arr = added_arr.transpose(0, 2, 1)  # (num_polys, poly_degree, num_new)
  added = added_arr.reshape(arr.shape[:-1] + (num_new,))
  return jnp.concatenate([ct_polynomial, added], axis=-1)


# ============================================================================
# Paterson-Stockmeyer helpers (module-level, matching OpenFHE ckksrns-utils.cpp)
# ============================================================================

_PS_DELTA = 2 ** (-44)  # Matches OpenFHE's 0x1p-44 tolerance


def _ps_is_not_equal_zero(v: float, delta: float = _PS_DELTA) -> bool:
  return abs(v) > delta


def _ps_is_not_equal_one(v: float, delta: float = _PS_DELTA) -> bool:
  return abs(v - 1.0) > delta


def _ps_degree(coeffs: List[float]) -> int:
  """Exact last non-zero index, matching OpenFHE ``Degree(..., 0.0)``."""
  i = len(coeffs) - 1
  while i > 0:
    if _ps_is_not_equal_zero(coeffs[i], delta=0.0):
      break
    i -= 1
  return i


def compute_degrees_ps(n: int) -> Tuple[int, int]:
  """Compute k, m for Paterson-Stockmeyer such that n < k*(2^m - 1).

  Port of OpenFHE ComputeDegreesPS (ckksrns-utils.cpp).
  """
  if n == 0:
    raise ValueError("ComputeDegreesPS: degree is zero")

  _UPPER = 2204
  _RANGEMAP = [
      (2, 1), (11, 2), (13, 3), (17, 2), (55, 3), (59, 4),
      (76, 3), (239, 4), (247, 5), (284, 4), (991, 5), (1007, 6),
      (1083, 5), (2015, 6), (2031, 7), (2204, 6),
  ]

  if n <= _UPPER:
    for upper, m_val in _RANGEMAP:
      if n <= upper:
        m = m_val
        break
    k = int(math.floor(n / ((1 << m) - 1)) + 1)
    return k, m

  # Heuristic for larger degrees.
  best_k, best_m, best_mult = None, None, float('inf')
  for k in range(1, n + 1):
    for m in range(1, int(math.ceil(math.log2(n / k) + 1) + 1) + 1):
      if n < k * ((1 << m) - 1):
        if abs(math.floor(math.log2(k))
               - math.floor(math.log2(math.sqrt(n / 2)))) <= 1.0:
          mult = k + 2 * m + (1 << (m - 1)) - 4
          if mult < best_mult:
            best_mult = mult
            best_k, best_m = k, m
  if best_k is None:
    raise ValueError(f"ComputeDegreesPS: no valid k,m for degree {n}")
  return best_k, best_m


def long_division_chebyshev(
    f: List[float], g: List[float]
) -> Tuple[List[float], List[float]]:
  """Chebyshev-basis polynomial long division: f = q * g + r.

  Port of OpenFHE LongDivisionChebyshev (ckksrns-utils.cpp).
  Uses T_n * T_m = 0.5 * (T_{n+m} + T_{|n-m|}).
  Convention: c_0 NOT halved (returned q,r also NOT halved).
  """
  f = [float(x) for x in f]
  g = [float(x) for x in g]
  # Trim trailing negligible coefficients (matching OpenFHE's Degree check)
  n = _ps_degree(f)
  f = f[:n + 1]
  k = _ps_degree(g)
  g = g[:k + 1]
  if n < k:
    return [0.0], list(f)

  q = [0.0] * (n - k + 1)
  r = list(f)

  while n > k:
    d = [0.0] * (n + 1)
    q[n - k] = 2.0 * r[-1]
    if _ps_is_not_equal_one(g[k]):
      q[n - k] /= g[-1]

    if k == n - k:
      d[0] = 2.0 * g[n - k]
      for i in range(1, 2 * k + 1):
        d[i] = g[abs(n - k - i)]
    elif k > (n - k):
      d[0] = 2.0 * g[n - k]
      for i in range(1, k - (n - k) + 1):
        d[i] = g[abs(n - k - i)] + g[n - k + i]
      for i in range(k - (n - k) + 1, n + 1):
        d[i] = g[abs(i - n + k)]
    else:
      d[n - k] = g[0]
      for i in range(n - 2 * k, n + 1):
        if i != n - k:
          d[i] = g[abs(i - n + k)]

    if _ps_is_not_equal_one(r[-1]):
      d = [di * r[-1] for di in d]
    if _ps_is_not_equal_one(g[-1]):
      d = [di / g[-1] for di in d]

    r = [r[i] - d[i] for i in range(len(r))]
    if len(r) > 1:
      n = _ps_degree(r)
      r = r[:n + 1]

  if n == k:
    d = list(g)
    q[0] = r[-1]
    if _ps_is_not_equal_one(g[-1]):
      q[0] /= g[-1]
    if _ps_is_not_equal_one(r[-1]):
      d = [di * r[-1] for di in d]
    if _ps_is_not_equal_one(g[-1]):
      d = [di / g[-1] for di in d]
    r = [r[i] - d[i] for i in range(len(r))]
    if len(r) > 1:
      n = _ps_degree(r)
      r = r[:n + 1]

  q[0] *= 2.0
  return q, r


# ============================================================================
# Chebyshev polynomial evaluation
# ============================================================================

def chebyshev_coefficients_for_sine(
    a: float, b: float, degree: int, K: int = 1
) -> jnp.ndarray:
  """Compute Chebyshev series coefficients for f(x) = sin(2*pi*K*x)/(2*pi*K)
  on the interval [a, b].

  For bootstrap with uniform ternary secret key distribution, K=512 creates
  512 periods in [-1,1], matching the expected overflow count after prescale.
  """
  n = degree + 1
  k = jnp.arange(n)
  x = jnp.cos(jnp.pi * (k + 0.5) / n)
  mapped = 0.5 * (b - a) * x + 0.5 * (a + b)
  fx = (1.0 / (2.0 * jnp.pi * K)) * jnp.sin(2.0 * jnp.pi * K * mapped)
  j_arr = jnp.arange(n)
  cos_matrix = jnp.cos(jnp.pi * jnp.outer(j_arr, k + 0.5) / n)
  coeffs = (2.0 / n) * (cos_matrix @ fx)
  coeffs = coeffs.at[0].set(coeffs[0] * 0.5)
  return coeffs


def eval_chebyshev_numeric(coeffs: jnp.ndarray,
                           x: jnp.ndarray,
                           a: float,
                           b: float) -> jnp.ndarray:
  """Reference Chebyshev series evaluator (jax.numpy)."""
  coeffs = jnp.asarray(coeffs)
  x = jnp.asarray(x)
  xt = (2.0 * x - (a + b)) / (b - a)
  Tprev = jnp.ones_like(xt)
  Tcur = xt.copy()
  out = coeffs[0] * Tprev + (coeffs[1] * Tcur if len(coeffs) > 1 else 0.0)
  for j in range(2, len(coeffs)):
    Tnext = 2.0 * xt * Tcur - Tprev
    out = out + coeffs[j] * Tnext
    Tprev, Tcur = Tcur, Tnext
  return out


# ============================================================================
# Numeric linear transform reference
# ============================================================================

def eval_linear_transform_numeric(diagonals: dict, x: jnp.ndarray) -> jnp.ndarray:
  """Reference implementation of a diagonal linear transform (jax.numpy)."""
  x = jnp.asarray(x)
  n = len(x)
  out = jnp.zeros_like(x, dtype=jnp.complex128)
  for k, d in diagonals.items():
    rotated = jnp.roll(x, -k)
    dd = jnp.asarray(d)
    if dd.shape[0] != n:
      dd = jnp.tile(dd, n // dd.shape[0])
    out = out + rotated * dd
  return out


# ============================================================================
# CKKS DFT matrix construction
# ============================================================================

def _ckks_rotation_group(m: int, nh: int, g: int = 5) -> List[int]:
  """Compute the CKKS rotation group: powers of g mod m."""
  r = [1]
  for _ in range(1, nh):
    r.append((r[-1] * g) % m)
  return r


def _build_ckks_dft_matrix(n_slots: int, m: int,
                           conjugate_transpose: bool = False):
  """Build the CKKS encoding DFT matrix U0, or its conjugate transpose.

  ``U0[i, j] = exp(2*pi*i * rot[i] * j / m)``.

  With ``conjugate_transpose=True`` this returns ``U0.conj().T`` without ever
  forming ``U0`` and transposing it.  XLA:TPU cannot lower a complex128
  transpose -- ``lax.transpose`` on c128 fails with ``RET_CHECK failure ...
  f64[N,N]{0,1:T(8,128)S(1)} vs c128[N,N]{1,0:T(8,128)}`` in
  xprecision_emitters -- which otherwise breaks every bootstrap on TPU during
  control generation.  Because

      (U0.conj().T)[a, b] = conj(U0[b, a]) = exp(-2*pi*i * a * rot[b] / m)

  the conjugate transpose has the same closed form with the outer product
  taken in the opposite order and the exponent negated, so the matrix is built
  directly in its final orientation.

  This is control generation, not runtime ciphertext evaluation, so construct
  the constants with host NumPy. TPU7x cannot lower either the implicit
  ``f64 -> c128`` conversion in ``jnp.exp(2j * angle)`` or a root
  ``lax.complex(f64, f64)`` result in the JAX 0.9 runtime. Keeping the matrix
  and its extracted diagonals on the host also avoids compiling an operation
  whose only consumer is the host-side CKKS plaintext encoder.
  """
  import numpy as _np

  rot_group = _ckks_rotation_group(m, n_slots)
  rot_arr = _np.asarray(rot_group, dtype=_np.float64)
  j_arr = _np.arange(n_slots, dtype=_np.float64)
  if conjugate_transpose:
    angle = -2.0 * _np.pi * _np.outer(j_arr, rot_arr) / m
  else:
    angle = 2.0 * _np.pi * _np.outer(rot_arr, j_arr) / m
  return _np.cos(angle) + 1j * _np.sin(angle)


def _extract_diagonals(matrix) -> Dict[int, object]:
  """Extract cyclic diagonals as host arrays for plaintext encoding."""
  import numpy as _np

  matrix = _np.asarray(matrix)
  n = matrix.shape[0]
  i_arr = _np.arange(n)
  diags = {}
  for k in range(n):
    col_indices = (i_arr + k) % n
    d = matrix[i_arr, col_indices]
    if _np.max(_np.abs(d)) > 1e-15:
      diags[k] = d
  return diags


def _select_layers(log_slots: int, budget: int) -> Tuple[int, int, int]:
  """Port of OpenFHE's SelectLayers (ckksrns-utils.cpp:55-78).

  Returns (layers_collapse, rows, rem_collapse).
  """
  if budget <= 0:
    budget = 1
  layers = math.ceil(log_slots / budget)
  rows = log_slots // layers
  rem = log_slots % layers
  dim = rows + (1 if rem != 0 else 0)
  if dim < budget:
    layers -= 1
    if layers <= 0:
      layers = 1
    rows = log_slots // layers
    rem = log_slots - rows * layers
    dim = rows + (1 if rem != 0 else 0)
    if dim > budget:
      while dim != budget and rows > 0:
        rows -= 1
        rem = log_slots - rows * layers
        dim = rows + (1 if rem != 0 else 0)
  return layers, rows, rem


def _reduce_rotation(index: int, slots: int) -> int:
  """Port of OpenFHE's ReduceRotation."""
  if slots == 0:
    return index
  return ((index % slots) + slots) % slots


def _bitrev_permutation(n: int, log_n: int) -> List[int]:
  """Return the bit-reversal permutation for n = 2^log_n elements."""
  perm = []
  for i in range(n):
    r = 0
    val = i
    for _ in range(log_n):
      r = (r << 1) | (val & 1)
      val >>= 1
    perm.append(r)
  return perm


def _coeff_encoding_one_level(m: int, rot_group: List[int],
                               n_slots: int,
                               flag_i: bool = False):
  """Port of OpenFHE's CoeffEncodingOneLevel (ckksrns-utils.cpp:437-477).

  Returns coeff array of shape (3*log2slots, n_slots) containing the
  three-diagonal butterfly coefficients for each FFT stage.

  coeff[s]            = "shifted right by 2^s" diagonal
  coeff[s+log2slots]  = "no shift" diagonal
  coeff[s+2*log2slots] = "shifted left by 2^s" diagonal

  flag_i: sparse-packing i-variant -- multiplies the innermost (mm == 2)
  butterfly by exp(-i*pi/2) = -i (OpenFHE neg_exp_M_PI), used to build the
  imaginary block of the concatenated sparse diagonals.
  """
  import numpy as _np
  log2slots = int(math.log2(n_slots))
  coeff = _np.zeros((3 * log2slots, n_slots), dtype=_np.complex128)
  pows = _np.array([_np.exp(2j * _np.pi * j / m) for j in range(m + 1)])

  for mm in (n_slots >> s for s in range(log2slots)):
    s = int(math.log2(mm)) - 1
    c0 = coeff[s]                    # shifted right
    c1 = coeff[s + log2slots]        # no shift
    c2 = coeff[s + 2 * log2slots]    # shifted left
    lenq = mm << 2
    lenh = mm >> 1
    b = complex(0.0, -1.0) if (flag_i and mm == 2) else 1.0
    for k in range(0, n_slots, mm):
      klenh = k + lenh
      c2[k:klenh] = b
      c1[k:klenh] = b
      for j in range(lenh):
        mod_val = rot_group[j] % lenq
        w = b * pows[(lenq - mod_val) * (m // lenq)]
        c1[klenh + j] = -w
        c0[klenh + j] = w
  # Host array on purpose: the only consumer is the matching
  # `_coeff_*_collapse`, which builds in NumPy and does the single
  # host->device conversion itself on the diagonals it returns.
  # Handing back a device array made the collapse pull it straight
  # back with np.array(), and that complex128 device->host read aborts
  # the TPU slice (FAILED_PRECONDITION / SLICE_FAILURE_SW_INJECT_ERROR).
  return coeff


def _coeff_decoding_one_level(m: int, rot_group: List[int],
                               n_slots: int,
                               flag_i: bool = False):
  """Port of OpenFHE's CoeffDecodingOneLevel (ckksrns-utils.cpp:479-519).

  Returns coeff array of shape (3*log2slots, n_slots) for the decoding
  (DIT / forward FFT) direction.

  flag_i: sparse-packing i-variant -- multiplies the innermost (mm == 2)
  butterfly by exp(+i*pi/2) = +i (OpenFHE pos_exp_M_PI).
  """
  import numpy as _np
  log2slots = int(math.log2(n_slots))
  coeff = _np.zeros((3 * log2slots, n_slots), dtype=_np.complex128)
  pows = _np.array([_np.exp(2j * _np.pi * j / m) for j in range(m + 1)])

  for mm in (2 << s for s in range(log2slots)):
    s = int(math.log2(mm)) - 1
    c0 = coeff[s]
    c1 = coeff[s + log2slots]
    c2 = coeff[s + 2 * log2slots]
    lenq = mm << 2
    lenh = mm >> 1
    b = complex(0.0, 1.0) if (flag_i and mm == 2) else 1.0
    for k in range(0, n_slots, mm):
      klenh = k + lenh
      c0[klenh:klenh + lenh] = b     # shifted right (lower half)
      c1[k:klenh] = b                # no shift (upper half)
      for j in range(lenh):
        jk = k + j
        mod_val = rot_group[j] % lenq
        w = b * pows[mod_val * (m // lenq)]
        c2[jk] = w                   # shifted left
        c1[jk + lenh] = -w           # no shift (lower half)
  # Host array on purpose: the only consumer is the matching
  # `_coeff_*_collapse`, which builds in NumPy and does the single
  # host->device conversion itself on the diagonals it returns.
  # Handing back a device array made the collapse pull it straight
  # back with np.array(), and that complex128 device->host read aborts
  # the TPU slice (FAILED_PRECONDITION / SLICE_FAILURE_SW_INJECT_ERROR).
  return coeff


def _coeff_encoding_collapse(coeff1: jnp.ndarray, n_slots: int,
                              level_budget: int,
                              layers_collapse: int,
                              rem_collapse: int) -> List[List[jnp.ndarray]]:
  """Port of OpenFHE's CoeffEncodingCollapse (ckksrns-utils.cpp:521-614).

  Collapses encoding (DIF) butterfly stages into per-level diagonal sets.
  Remainder goes to coeff[0] (first level in the output array).
  """
  import numpy as _np
  log2slots = int(math.log2(n_slots))
  dim_collapse = level_budget
  flag_rem = 1 if rem_collapse != 0 else 0

  num_rotations = (1 << (layers_collapse + 1)) - 1
  num_rotations_rem = (1 << (rem_collapse + 1)) - 1

  c1 = [_np.array(coeff1[i]) for i in range(3 * log2slots)]

  coeff = [[_np.zeros(n_slots, dtype=_np.complex128)
            for _ in range(num_rotations)]
           for _ in range(dim_collapse)]
  if flag_rem:
    coeff[0] = [_np.zeros(n_slots, dtype=_np.complex128)
                for _ in range(num_rotations_rem)]

  if layers_collapse:
    zeros = lambda nr: [_np.zeros(n_slots, dtype=_np.complex128)
                        for _ in range(nr)]
    for s in range(dim_collapse - 1, flag_rem - 1, -1):
      top = log2slots - (dim_collapse - 1 - s) * layers_collapse - 1

      coeff[s][0] = c1[top].copy()
      coeff[s][1] = c1[top + log2slots].copy()
      coeff[s][2] = c1[top + 2 * log2slots].copy()

      for l in range(1, layers_collapse):
        temp = [c.copy() for c in coeff[s]]
        coeff[s] = zeros(num_rotations)
        for u in range((1 << (l + 1)) - 1):
          for k in range(n_slots):
            coeff[s][2 * u][k] += (
                c1[top - l][k]
                * temp[u][_reduce_rotation(k - (1 << (top - l)), n_slots)])
            coeff[s][2 * u + 1][k] += (
                c1[top - l + log2slots][k] * temp[u][k])
            coeff[s][2 * u + 2][k] += (
                c1[top - l + 2 * log2slots][k]
                * temp[u][_reduce_rotation(k + (1 << (top - l)), n_slots)])

  if flag_rem and rem_collapse:
    zeros_rem = lambda: [_np.zeros(n_slots, dtype=_np.complex128)
                         for _ in range(num_rotations_rem)]
    s = 0
    top = log2slots - (dim_collapse - 1 - s) * layers_collapse - 1

    coeff[s][0] = c1[top].copy()
    coeff[s][1] = c1[top + log2slots].copy()
    coeff[s][2] = c1[top + 2 * log2slots].copy()

    for l in range(1, rem_collapse):
      temp = [c.copy() for c in coeff[s]]
      coeff[s] = zeros_rem()
      for u in range((1 << (l + 1)) - 1):
        for k in range(n_slots):
          coeff[s][2 * u][k] += (
              c1[top - l][k]
              * temp[u][_reduce_rotation(k - (1 << (top - l)), n_slots)])
          coeff[s][2 * u + 1][k] += (
              c1[top - l + log2slots][k] * temp[u][k])
          coeff[s][2 * u + 2][k] += (
              c1[top - l + 2 * log2slots][k]
              * temp[u][_reduce_rotation(k + (1 << (top - l)), n_slots)])

  # These complex constants are consumed by the host plaintext encoder. Keep
  # them in NumPy; transferring c128 control data to TPU is both unnecessary
  # and unsupported by some TPU7x layouts in JAX 0.9.
  return coeff


def _coeff_decoding_collapse(coeff1: jnp.ndarray, n_slots: int,
                              level_budget: int,
                              layers_collapse: int,
                              rem_collapse: int) -> List[List[jnp.ndarray]]:
  """Port of OpenFHE's CoeffDecodingCollapse (ckksrns-utils.cpp:616-701).

  Collapses decoding (DIT) butterfly stages into per-level diagonal sets.
  Remainder goes to coeff[dimCollapse-1] (last level in the output array).
  Uses a different collapsing formula from the encoding direction.
  """
  import numpy as _np
  log2slots = int(math.log2(n_slots))
  dim_collapse = level_budget
  flag_rem = 1 if rem_collapse != 0 else 0
  rows_collapse = dim_collapse - flag_rem

  num_rotations = (1 << (layers_collapse + 1)) - 1
  num_rotations_rem = (1 << (rem_collapse + 1)) - 1

  c1 = [_np.array(coeff1[i]) for i in range(3 * log2slots)]

  coeff = [[_np.zeros(n_slots, dtype=_np.complex128)
            for _ in range(num_rotations)]
           for _ in range(dim_collapse)]
  if flag_rem:
    coeff[dim_collapse - 1] = [_np.zeros(n_slots, dtype=_np.complex128)
                                for _ in range(num_rotations_rem)]

  if layers_collapse:
    zeros = lambda nr: [_np.zeros(n_slots, dtype=_np.complex128)
                        for _ in range(nr)]
    for s in range(rows_collapse):
      base = s * layers_collapse
      coeff[s][0] = c1[base].copy()
      coeff[s][1] = c1[base + log2slots].copy()
      coeff[s][2] = c1[base + 2 * log2slots].copy()

      for l in range(1, layers_collapse):
        temp = [c.copy() for c in coeff[s]]
        coeff[s] = zeros(num_rotations)
        for t in range(3):
          shift = 0 if t == 0 else ((1 << l) if t == 1 else (1 << (l + 1)))
          for u in range((1 << (l + 1)) - 1):
            for k in range(n_slots):
              coeff[s][u + shift][k] += (
                  c1[base + l + t * log2slots][k] * temp[u][k])

  if flag_rem and rem_collapse:
    zeros_rem = lambda: [_np.zeros(n_slots, dtype=_np.complex128)
                         for _ in range(num_rotations_rem)]
    s = rows_collapse
    base = s * layers_collapse

    coeff[s][0] = c1[base].copy()
    coeff[s][1] = c1[base + log2slots].copy()
    coeff[s][2] = c1[base + 2 * log2slots].copy()

    for l in range(1, rem_collapse):
      temp = [c.copy() for c in coeff[s]]
      coeff[s] = zeros_rem()
      for t in range(3):
        shift = 0 if t == 0 else ((1 << l) if t == 1 else (1 << (l + 1)))
        for u in range((1 << (l + 1)) - 1):
          for k in range(n_slots):
            coeff[s][u + shift][k] += (
                c1[base + l + t * log2slots][k] * temp[u][k])

  # These complex constants are consumed by the host plaintext encoder. Keep
  # them in NumPy; transferring c128 control data to TPU is both unnecessary
  # and unsupported by some TPU7x layouts in JAX 0.9.
  return coeff


def _collapsed_to_diag_dicts(coeff: List[List[jnp.ndarray]],
                              n_slots: int,
                              level_budget: int,
                              layers_collapse: int,
                              rem_collapse: int,
                              for_encoding: bool = True,
                              key_mod: int = None,
                              ) -> List[Dict[int, jnp.ndarray]]:
  """Convert collapsed coefficients to rotation-indexed diagonal dicts.

  Each collapsed level s has a set of diagonal vectors indexed by ij.
  The rotation index follows OpenFHE's BSGS split (EvalCoeffsToSlots /
  EvalSlotsToCoeffs rot_in/rot_out):

    C2S: rot = RR(scale*g*(ij//g), key_mod) + RR(scale*((ij%g)-offset), n_slots)
    S2C: rot = RR(scale*(ij-offset), key_mod)

  where key_mod = M/4-equivalent pattern period (== n_slots at full packing,
  where the split collapses to the historical RR(scale*(ij-offset), n_slots);
  == 2*n_slots for sparse packing, where the concatenated [d | i*d] diagonal
  patterns are 2*n_slots-periodic and giant rotations must keep their >=
  n_slots components).

  For encoding (C2S): levels are processed from s=lvlb-1 down to 0.
  For decoding (S2C): levels are processed from s=0 up to lvlb-1.

  Args:
    for_encoding: If True, produces C2S levels (outermost first).
                  If False, produces S2C levels (innermost first).
    key_mod: rotation-key modulus (pattern period); default n_slots.
  """
  import numpy as _np
  flag_rem = 1 if rem_collapse != 0 else 0
  num_rotations = (1 << (layers_collapse + 1)) - 1
  num_rotations_rem = (1 << (rem_collapse + 1)) - 1
  if key_mod is None:
    key_mod = n_slots

  # OpenFHE GetCollapsedFFTParams baby/giant split (ckksrns-utils.cpp:723-727)
  g = 1 << (layers_collapse // 2 + 1 + (1 if num_rotations > 7 else 0))
  g_rem = (1 << (rem_collapse // 2 + 1 + (1 if num_rotations_rem > 7 else 0))) \
      if rem_collapse else 0

  def _key_enc(scale, ij, offset):
    # EvalCoeffsToSlots: rot_out mod M4 (giant), rot_in mod slots (baby).
    i, j = ij // g, ij % g
    return (_reduce_rotation(scale * g * i, key_mod)
            + _reduce_rotation(scale * (j - offset), n_slots)) % key_mod

  def _key_enc_rem(ij, offset):
    i, j = ij // g_rem, ij % g_rem
    return (_reduce_rotation(g_rem * i, key_mod)
            + _reduce_rotation(j - offset, n_slots)) % key_mod

  diag_list = []

  if for_encoding:
    # Encoding (C2S): process levels from s=lvlb-1 down to 0.
    # Regular levels first (s=lvlb-1 down to flagRem), then remainder.
    stop = 0 if flag_rem else -1
    for s in range(level_budget - 1, stop, -1):
      scale = 1 << ((s - flag_rem) * layers_collapse + rem_collapse)
      offset = (num_rotations + 1) // 2 - 1
      nr = num_rotations
      diags = {}
      for ij in range(nr):
        d = _np.array(coeff[s][ij])
        if _np.max(_np.abs(d)) < 1e-15:
          continue
        rot = _key_enc(scale, ij, offset)
        if rot in diags:
          diags[rot] = diags[rot] + d
        else:
          diags[rot] = d.copy()
      diag_list.append(diags)

    if flag_rem:
      offset = (num_rotations_rem + 1) // 2 - 1
      diags = {}
      for ij in range(num_rotations_rem):
        d = _np.array(coeff[0][ij])
        if _np.max(_np.abs(d)) < 1e-15:
          continue
        rot = _key_enc_rem(ij, offset)
        if rot in diags:
          diags[rot] = diags[rot] + d
        else:
          diags[rot] = d.copy()
      diag_list.append(diags)

  else:
    # Decoding (S2C): process levels from s=0 up to smax, then remainder.
    # (Matches OpenFHE EvalSlotsToCoeffs line 2089: for s=0..smax.)
    # EvalSlotsToCoeffs reduces BOTH rot_in and rot_out mod M4 (key_mod).
    smax = level_budget - flag_rem
    for s in range(smax):
      scale = 1 << (s * layers_collapse)
      offset = (num_rotations + 1) // 2 - 1
      diags = {}
      for ij in range(num_rotations):
        d = _np.array(coeff[s][ij])
        if _np.max(_np.abs(d)) < 1e-15:
          continue
        rot = _reduce_rotation((ij - offset) * scale, key_mod)
        if rot in diags:
          diags[rot] = diags[rot] + d
        else:
          diags[rot] = d.copy()
      diag_list.append(diags)

    if flag_rem:
      # Remainder level is at coeff[dimCollapse-1] for decoding.
      s_rem = level_budget - 1
      scale_rem = 1 << (smax * layers_collapse)
      offset = (num_rotations_rem + 1) // 2 - 1
      diags = {}
      for ij in range(num_rotations_rem):
        d = _np.array(coeff[s_rem][ij])
        if _np.max(_np.abs(d)) < 1e-15:
          continue
        rot = _reduce_rotation((ij - offset) * scale_rem, key_mod)
        if rot in diags:
          diags[rot] = diags[rot] + d
        else:
          diags[rot] = d.copy()
      diag_list.append(diags)

  return diag_list


def _decompose_dft_into_levels(n_slots: int, m: int,
                                num_levels: int,
                                inverse: bool = False,
                                flag_i: bool = False,
                                key_mod: int = None,
                                ) -> List[Dict[int, jnp.ndarray]]:
  """Decompose the CKKS DFT matrix into diagonal linear transforms.

  Port of OpenFHE's CoeffEncoding/DecodingCollapse (ckksrns-utils.cpp).

  For C2S (inverse=False): produces the encoding-direction (DIF FFT)
  butterfly decomposition. Applied to a ciphertext, computes
  ``P @ U^H @ x`` where P is the implicit bit-reversal permutation
  and U^H is the conjugate transpose of the DFT matrix.
  The bit-reversal is harmless because the subsequent per-slot
  approximate mod reduction is order-independent, and S2C undoes it.

  For S2C (inverse=True): produces the decoding-direction (DIT FFT)
  butterfly decomposition. Undoes the bit-reversal from C2S and
  applies the forward DFT.

  Args:
    n_slots: Number of CKKS slots (must be a power of 2).
    m: Cyclotomic order (= 4 * n_slots for fully packed).
    num_levels: Level budget (number of ciphertext levels to consume).
    inverse: False for C2S (encoding/DIF), True for S2C (decoding/DIT).

  Returns:
    List of diagonal dicts, one per level.
    Each dict maps rotation_index -> diagonal_vector.
  """
  log_n = int(math.log2(n_slots)) if n_slots > 0 else 0

  if log_n == 0 or n_slots <= 1:
    import numpy as _np
    return [{0: _np.ones(max(1, n_slots), dtype=_np.complex128)}]

  if num_levels <= 0 or num_levels > log_n:
    num_levels = max(1, log_n)

  # For single level, OpenFHE uses the direct linear-transform path
  # (U0hatT/U0), not the collapsed FFT path, so there is no implicit
  # bit-reversal permutation.
  if num_levels == 1:
    # Build U0 or U0^H directly: forming U0 and transposing it does not lower
    # on TPU for complex128 (see `_build_ckks_dft_matrix`).
    mat = _build_ckks_dft_matrix(n_slots, m, conjugate_transpose=not inverse)
    return [_extract_diagonals(mat)]

  rot_group = _ckks_rotation_group(m, n_slots)
  layers_collapse, _, rem_collapse = _select_layers(log_n, num_levels)

  if inverse:
    # S2C (decoding / DIT direction)
    coeff1 = _coeff_decoding_one_level(m, rot_group, n_slots, flag_i=flag_i)
    coeff = _coeff_decoding_collapse(coeff1, n_slots, num_levels,
                                      layers_collapse, rem_collapse)
  else:
    # C2S (encoding / DIF direction)
    coeff1 = _coeff_encoding_one_level(m, rot_group, n_slots, flag_i=flag_i)
    coeff = _coeff_encoding_collapse(coeff1, n_slots, num_levels,
                                      layers_collapse, rem_collapse)

  return _collapsed_to_diag_dicts(
      coeff, n_slots, num_levels,
      layers_collapse, rem_collapse,
      for_encoding=(not inverse),
      key_mod=key_mod)


def _decompose_dft_into_hoisted_groups(
    n_slots: int,
    m: int,
    num_levels: int,
    inverse: bool = False,
    flag_i: bool = False,
    key_mod: int = None,
):
  """Return unmerged OpenFHE BSGS groups for collapsed FFT evaluation.

  ``_collapsed_to_diag_dicts`` intentionally merges positions whose total
  rotations coincide modulo the slot count.  That is algebraically valid for
  ordinary Q-domain evaluation, but OpenFHE downs c0 once per *outer group*;
  ApproxModDown is not additive, so merging those groups changes bootstrap
  noise.  This helper preserves every original ``ij`` position and its exact
  baby/giant assignment.

  Result shape: ``levels -> groups -> {giant, entries}``, where entries are
  ``(baby_rotation, diagonal_vector)`` pairs in runtime application order.
  """
  import numpy as _np

  log_n = int(math.log2(n_slots)) if n_slots > 0 else 0
  if log_n == 0 or n_slots <= 1:
    return None
  if num_levels <= 0 or num_levels > log_n:
    num_levels = max(1, log_n)
  if num_levels == 1:
    return None
  if key_mod is None:
    key_mod = n_slots

  rot_group = _ckks_rotation_group(m, n_slots)
  layers_collapse, _, rem_collapse = _select_layers(log_n, num_levels)
  if inverse:
    coeff1 = _coeff_decoding_one_level(
        m, rot_group, n_slots, flag_i=flag_i
    )
    coeff = _coeff_decoding_collapse(
        coeff1, n_slots, num_levels, layers_collapse, rem_collapse
    )
  else:
    coeff1 = _coeff_encoding_one_level(
        m, rot_group, n_slots, flag_i=flag_i
    )
    coeff = _coeff_encoding_collapse(
        coeff1, n_slots, num_levels, layers_collapse, rem_collapse
    )

  num_rot = (1 << (layers_collapse + 1)) - 1
  num_rot_rem = (1 << (rem_collapse + 1)) - 1
  g = 1 << (
      layers_collapse // 2 + 1 + (1 if num_rot > 7 else 0)
  )
  g_rem = (
      1 << (rem_collapse // 2 + 1 + (1 if num_rot_rem > 7 else 0))
      if rem_collapse else 0
  )
  flag_rem = 1 if rem_collapse else 0

  def _make_groups(s, scale, count, baby_count):
    offset = (count + 1) // 2 - 1
    outer_count = (count + 1) // baby_count
    groups = []
    for i in range(outer_count):
      giant = _reduce_rotation(scale * baby_count * i, key_mod)
      entries = []
      for j in range(baby_count):
        ij = i * baby_count + j
        if ij >= count:
          continue
        diagonal = _np.asarray(coeff[s][ij])
        if _np.max(_np.abs(diagonal)) < 1e-15:
          continue
        baby = _reduce_rotation(scale * (j - offset), key_mod)
        entries.append((int(baby), diagonal))
      if entries:
        groups.append({'giant': int(giant), 'entries': entries})
    return groups

  levels = []
  if not inverse:
    stop = 0 if flag_rem else -1
    for s in range(num_levels - 1, stop, -1):
      scale = 1 << (
          (s - flag_rem) * layers_collapse + rem_collapse
      )
      levels.append(_make_groups(s, scale, num_rot, g))
    if flag_rem:
      levels.append(_make_groups(0, 1, num_rot_rem, g_rem))
  else:
    smax = num_levels - flag_rem
    for s in range(smax):
      scale = 1 << (s * layers_collapse)
      levels.append(_make_groups(s, scale, num_rot, g))
    if flag_rem:
      scale = 1 << (smax * layers_collapse)
      levels.append(_make_groups(
          num_levels - 1, scale, num_rot_rem, g_rem
      ))

  return levels


def _collapsed_hoisted_plan(diags, n_slots):
  """Factor one collapsed FFT diagonal map into OpenFHE-style BSGS groups.

  Collapsed FFT supports are centered arithmetic progressions
  ``[-(g-1), ..., g-1] * stride``.  The non-positive half is the hoisted
  baby set; the positive half is represented as the same baby set followed
  by the giant rotation ``g * stride``.  At cyclic wrap levels that giant is
  zero, so the already-merged diagonal map is one baby-only group.

  Returns ``[(giant_rotation, [(diag_key, baby_rotation), ...]), ...]``.
  The zero-giant group is always first.
  """
  keys = sorted({int(k) % int(n_slots) for k in diags})
  if not keys:
    return []

  def _signed(k):
    return k - n_slots if k > n_slots // 2 else k

  signed = {k: _signed(k) for k in keys}
  nonzero = [abs(v) for v in signed.values() if v]
  stride = 0
  for value in nonzero:
    stride = math.gcd(stride, value)
  stride = max(1, stride)
  units = {k: signed[k] // stride for k in keys}
  max_abs = max(abs(v) for v in units.values())
  g = 1
  while g <= max_abs:
    g <<= 1
  giant = (g * stride) % n_slots

  if giant == 0:
    return [(0, [(k, k) for k in keys])]

  zero_group = []
  giant_group = []
  for k in keys:
    unit = units[k]
    if unit <= 0:
      zero_group.append((k, k))
    else:
      baby = ((unit - g) * stride) % n_slots
      giant_group.append((k, baby))
  plan = [(0, zero_group)]
  if giant_group:
    plan.append((giant, giant_group))
  return plan


def _bootstrap_rotation_indices(
    degree: int,
    n_slots: int,
    level_budget: Sequence[int],
) -> Tuple[int, ...]:
  """Plan every non-conjugation rotation without building DFT diagonals."""
  c2s_levels, s2c_levels = level_budget
  log_slots = int(math.log2(n_slots)) if n_slots > 0 else 0
  sparse = n_slots < degree // 2
  key_mod = 2 * n_slots if sparse else n_slots
  rotations = set()

  def add_transform(levels: int, *, inverse: bool) -> None:
    if n_slots <= 1:
      return
    if levels == 1:
      diagonal_keys = set(range(n_slots))
      rotations.update(diagonal_keys - {0})
      if not sparse:
        for giant, entries in _collapsed_hoisted_plan(
            diagonal_keys, n_slots
        ):
          rotations.update({giant, *(baby for _, baby in entries)} - {0})
      return

    layers, _, remainder = _select_layers(log_slots, levels)
    remainder_flag = int(bool(remainder))
    regular_count = (1 << (layers + 1)) - 1
    remainder_count = (1 << (remainder + 1)) - 1
    regular_babies = 1 << (
        layers // 2 + 1 + int(regular_count > 7)
    )
    remainder_babies = (
        1 << (remainder // 2 + 1 + int(remainder_count > 7))
        if remainder else 0
    )
    specifications = []
    if inverse:
      for level in range(levels - remainder_flag):
        specifications.append((
            1 << (level * layers), regular_count, regular_babies
        ))
      if remainder_flag:
        specifications.append((
            1 << ((levels - remainder_flag) * layers),
            remainder_count,
            remainder_babies,
        ))
    else:
      stop = 0 if remainder_flag else -1
      for level in range(levels - 1, stop, -1):
        specifications.append((
            1 << ((level - remainder_flag) * layers + remainder),
            regular_count,
            regular_babies,
        ))
      if remainder_flag:
        specifications.append((1, remainder_count, remainder_babies))

    for scale, count, baby_count in specifications:
      offset = (count + 1) // 2 - 1
      for position in range(count):
        if inverse:
          diagonal = _reduce_rotation(
              scale * (position - offset), key_mod
          )
        else:
          giant_index, baby_index = divmod(position, baby_count)
          diagonal = (
              _reduce_rotation(
                  scale * baby_count * giant_index, key_mod
              )
              + _reduce_rotation(
                  scale * (baby_index - offset), n_slots
              )
          ) % key_mod
        if diagonal:
          rotations.add(diagonal)
        if not sparse:
          giant_index, baby_index = divmod(position, baby_count)
          giant = _reduce_rotation(
              scale * baby_count * giant_index, key_mod
          )
          baby = _reduce_rotation(
              scale * (baby_index - offset), key_mod
          )
          rotations.update({giant, baby} - {0})

  add_transform(c2s_levels, inverse=False)
  add_transform(s2c_levels, inverse=True)
  if sparse:
    partial = 1
    gap = (degree // 2) // n_slots
    while partial < gap:
      rotations.add(partial * n_slots)
      partial <<= 1
    rotations.add(n_slots)
  return tuple(sorted(rotations))


# ============================================================================
# Bootstrap class
# ============================================================================

class Bootstrap:
  """Full CKKS bootstrap orchestrator.

  Usage:
      ctx = CKKSContext(params)
      ctx.program_initialization(...)
      bs = Bootstrap(ctx)
      bs.control_gen(level_budget=[4,4])
      bs.setup_key()
      ct_out = bs.bootstrap(ct_in)
  """

  def __init__(self, ctx):
    self.ctx = ctx
    self._cache = getattr(ctx, "_param_cache", None)
    self._degree = ctx.degree
    self._num_slots = ctx.num_slots
    self._m = 2 * self._degree
    # Sparse packing (num_slots < N/2): gap = how many times the slot pattern
    # repeats across the ring (OpenFHE ckksrns-fhe.cpp sparse branch).
    # num_slots must be a power-of-2 divisor of N/2.
    self._gap = max(1, (self._degree // 2) // self._num_slots)
    self._sparse = self._gap > 1
    if self._sparse:
      if self._num_slots & (self._num_slots - 1):
        raise ValueError(f"sparse num_slots must be a power of 2, got "
                         f"{self._num_slots}")
      if self._num_slots * self._gap != self._degree // 2:
        raise ValueError(f"num_slots={self._num_slots} must divide "
                         f"N/2={self._degree // 2}")
    self._composite_degree = ctx.composite_degree
    # Low-memory mode (env BOOTSTRAP_LOW_MEM=1): drop JAX compilation caches
    # at bootstrap stage boundaries.  At large N (>=16384) the accumulated
    # per-(level, op) compiled executables -- whose baked-in constants include
    # sizable tables -- exhaust the memory budget mid-bootstrap (observed:
    # cgroup OOM at ~100 GiB virtual at N=16384), while every ciphertext is
    # only a few MB.  Clearing recompiles later stages (pure wall-clock cost,
    # bit-identical results).  BOOTSTRAP_MEM_TRACE=1 prints per-stage memory.
    self._low_mem = os.environ.get("BOOTSTRAP_LOW_MEM") == "1"
    self._mem_trace = os.environ.get("BOOTSTRAP_MEM_TRACE") == "1"
    # DC-bias correction (default OFF). Experiments found an additive,
    # input-independent output component for a fixed key. Subtracting a
    # once-precomputed bootstrap(0) template can cancel that component and
    # composes with Meta-BTS. Enable with BOOTSTRAP_DC_SUBTRACT=1 and call
    # precompute_dc_template().
    self._dc_subtract = os.environ.get("BOOTSTRAP_DC_SUBTRACT") == "1"
    self._dc_template = None          # output ciphertext of bootstrap(zero)
    self._computing_dc_template = False  # recursion guard

    # Populated by control_gen()
    self._level_budget = None
    self._cheby_coeffs = None
    self._cheby_degree = None
    self._K = None
    self._R = None
    self._c2s_encoded = None
    self._c2s_raw_diags = None
    self._c2s_hoisted_groups = None
    self._s2c_encoded = None
    self._s2c_raw_diags = None
    self._s2c_hoisted_groups = None
    self._s2c_encoded_start_level = None
    self._all_rot_indices = None
    self._c2s_start_level = None
    self._hoisted_lt_plaintexts = {}

  def _stage_checkpoint(self, tag: str, ct=None) -> None:
    """Optional per-stage memory hygiene/tracing (see __init__).

    ``ct`` (a ciphertext, or list of them) is block_until_ready'd BEFORE the
    cache eviction.  Under JAX async dispatch the heavy ops (he_mul/relin
    key-switch, rescale BConv) return immediately and their large intermediate
    device buffers stay live until forced; without a barrier they accumulate
    across a loop.  The C2S/S2C linear transforms already block within their
    loop; the ApproxMod PS/DA loops did NOT, so at N=65536 the Chebyshev +
    double-angle stage piled up ~54 GB of uncomputed intermediates and OOM'd
    (past the 50+50 GiB cgroup cap).  Forcing completion here bounds the live
    set to one op's working memory.
    """
    if self._low_mem:
      if ct is not None:
        try:
          polys = []
          for c in (ct if isinstance(ct, (list, tuple)) else (ct,)):
            p = getattr(c, "polynomial", None)
            if p is not None:
              polys.append(p)
          if polys:
            jax.block_until_ready(polys)
        except Exception:
          pass
      # Evict the per-(level, rot_index) operator caches.  ctx.he_rot[...] (and
      # he_mul/ptct_mul/he_rescale) memoize a HE*AtLevel instance per key, each
      # holding degree-sized NTT/twiddle tables + sliced keys.  A single C2S/S2C
      # level instantiates 30-60 fresh rotation operators; left cached they
      # accumulate ~21 GB per level at N=16384 (the dominant peak, NOT freed by
      # jax.clear_caches()).  Rebuilt on demand next stage (recompute cost only).
      for acc_name in ("he_rot", "he_mul", "ptct_mul", "he_rescale"):
        acc = getattr(self.ctx, acc_name, None)
        inst = getattr(acc, "_instances", None)
        if isinstance(inst, dict):
          inst.clear()
      # Max-level rotation keys are retained in raw numpy form so lower-level
      # HYBRID partition boundaries remain identical to OpenFHE.  The formatted
      # JAX copies are reconstructible views and otherwise duplicate ~30 GiB at
      # N=65536 once every rotation has been touched.
      formatted_rot_keys = getattr(
          self._cache, "_formatted_rotation_keys", None
      )
      if isinstance(formatted_rot_keys, dict):
        formatted_rot_keys.clear()
      # Hoisted QP diagonals are consumed once per transform level.  Meta-BTS
      # can lazily re-encode them on a later bootstrap; retaining every level
      # through the ApproxMod stage only raises the large-N memory peak.
      self._hoisted_lt_plaintexts.clear()
      jax.clear_caches()
      gc.collect()
    if self._mem_trace:
      try:
        cur = int(open("/sys/fs/cgroup/memory.current").read()) / 2**30
        swp = int(open("/sys/fs/cgroup/memory.swap.current").read()) / 2**30
        print(f"  [bs-mem {tag:14s}] ram={cur:5.1f}G swap={swp:5.1f}G "
              f"total={cur+swp:5.1f}G", flush=True)
      except OSError:
        pass

  def _require_param_cache(self):
    """Return the context parameter cache, refreshing after late init."""
    cache = getattr(self.ctx, "_param_cache", None)
    if cache is None:
      raise RuntimeError(
          "CKKSContext.program_initialization(...) must be called before "
          "Bootstrap control generation or encrypted bootstrap operations.")
    self._cache = cache
    return cache

  # ------------------------------------------------------------------
  # Per-ciphertext dynamic scale tracking (matching OpenFHE FLEXIBLEAUTO)
  #
  # Tracks the actual encoding scale of the ciphertext message.
  # Used for:
  # 1. Correct decoding at the output of operations (proper_decrypt)
  # 2. Ensuring encode/decode scale consistency across the pipeline
  #
  # For simple operation chains (ptct_mul + rescale), the scale stays
  # near the per-level SF. For deep Chebyshev evaluation, the tracked
  # scale can overflow float64 (the mathematical scale grows super-
  # exponentially through recursive squarings). In such cases, use
  # the per-level SF as a decode approximation.
  # ------------------------------------------------------------------

  def _get_scale(self, ct) -> float:
    """Get the tracked CKKS scaling factor for a ciphertext.

    Returns the actual scale value. If no tracked scale is set,
    returns the per-level SF based on the inferred level.
    Falls back to per-level SF if the tracked value overflows float64.
    """
    sf = getattr(ct, '_ckks_scale', None)
    if sf is not None and math.isfinite(sf) and sf > 0:
      return sf
    # Infer from level
    self._require_param_cache()
    level = self._infer_level(ct)
    return self._openfhe_scale_at_level(level)

  def _openfhe_scale_at_level(self, level: int) -> float:
    """Return OpenFHE's recursively evolved scale for a CROSS level.

    CROSS levels count upward with the number of retained composite groups,
    whereas OpenFHE indexes ``m_scalingFactorsReal`` by the number of dropped
    individual towers. ``HEParameterCache.scaling_factor_recursive`` performs
    that mapping.  Direct top-prime products are physical rescale divisors,
    not the scale used to encode auxiliary plaintexts.
    """
    self._require_param_cache()
    level = max(0, min(int(level), self._cache.max_level))
    return self._cache.scaling_factor_recursive(level)

  def _set_scale(self, ct, sf: float):
    """Set the tracked CKKS scaling factor on a ciphertext."""
    ct._ckks_scale = sf

  def _copy_scale(self, dst, src):
    """Copy scale tracking from src to dst ciphertext."""
    sf = getattr(src, '_ckks_scale', None)
    if sf is not None:
      dst._ckks_scale = sf

  # ------------------------------------------------------------------
  # Per-ciphertext NoiseScaleDeg (nsd) tracking
  #
  # Matches OpenFHE's NoiseScaleDeg attribute:
  #   nsd=1: ciphertext at scale S (post-rescale, normal state)
  #   nsd=2: ciphertext at scale S^2 (post-multiply, pre-rescale)
  #
  # OpenFHE flow:
  #   EvalSquare/EvalMult -> nsd=2
  #   AdjustLevelsAndDepthToOneInPlace -> if nsd==2: ModReduceInternal
  #   ModReduceInternal -> nsd=1
  #
  # In CROSS, _he_op_mul handles the pre-rescale (nsd=2 -> nsd=1)
  # internally when inputs are at nsd=2, matching OpenFHE's
  # AdjustLevelsAndDepthToOneInPlace.
  # ------------------------------------------------------------------

  def _get_nsd(self, ct) -> int:
    """Get the NoiseScaleDeg for a ciphertext. Default is 1."""
    return getattr(ct, '_ckks_nsd', 1)

  def _set_nsd(self, ct, nsd: int):
    """Set the NoiseScaleDeg on a ciphertext."""
    ct._ckks_nsd = nsd

  def _copy_nsd(self, dst, src):
    """Copy NSD tracking from src to dst ciphertext."""
    nsd = getattr(src, '_ckks_nsd', None)
    if nsd is not None:
      dst._ckks_nsd = nsd

  def _ensure_nsd1(self, ct, level: int):
    """Rescale ciphertext to nsd=1 by repeated ModReduceInternal.

    Matches OpenFHE's AdjustLevelsAndDepthToOneInPlace.
    Handles nsd=2, nsd=3, etc. by looping (each rescale decrements by 1).

    Returns (ct, new_level).
    """
    nsd = self._get_nsd(ct)
    while nsd > 1 and level > 0:
      ct = self._he_op_rescale(level, level - 1, ct)
      level -= 1
      nsd = self._get_nsd(ct)
    return ct, level

  # ------------------------------------------------------------------
  # Bootstrap depth / NQ computation
  # ------------------------------------------------------------------

  @staticmethod
  def compute_bootstrap_nq(cd, level_budget=None, headroom=0,
                           secret_key_dist="uniform_ternary"):
    """Compute the minimum Q-tower count (NQ) for a bootstrap configuration.

    Matches OpenFHE's GetBootstrapDepth logic.  The caller uses the
    returned NQ to generate the prime chain before context creation.

    Args:
      cd: composite degree (e.g. 2).
      level_budget: [c2s_levels, s2c_levels], default [4, 4].
      headroom: post-bootstrap logical levels to keep (OpenFHE's
          levelsAfterBoot).  Default 0 = fully exhausted output.
      secret_key_dist: "uniform_ternary" or "sparse_ternary".

    Returns:
      dict with 'nq', 'depth', 'boot_depth', 'max_level',
      'c2s_start', 'approx_mod_start', 'expected_output_level',
      'expected_output_nq'.

    ``headroom`` is OpenFHE's ``levelsAfterBootstrapping``: the number of
    logical composite levels retained after the bootstrap.  For example,
    cd=2, level_budget=[1,1], headroom=5 gives NQ=44 and a level-5/q12
    output; level_budget=[4,4], headroom=6 gives NQ=58 and level-6/q14.
    """
    if level_budget is None:
      level_budget = [4, 4]
    c2s_levels, s2c_levels = level_budget

    # OpenFHE GetModDepthInternal uses the Paterson-Stockmeyer depth table,
    # not ceil(log2(degree+1)).  Degree 88 has depth 8 and degree 59 has
    # depth 7; adding the respective double-angle counts gives 14 and 10.
    if secret_key_dist == "uniform_ternary":
      approx_mod_depth = 8 + 6
    else:
      approx_mod_depth = 7 + 3

    # OpenFHE GetBootstrapDepth is exactly:
    #   GetModDepthInternal(secretKeyDist) + lb_C2S + lb_S2C.
    # Each transform budget includes its prescale/pre-S2C boundary level, so
    # expanding this into ad-hoc rescale terms is prone to off-by-one errors.
    boot_consumed = approx_mod_depth + c2s_levels + s2c_levels

    depth = boot_consumed + headroom
    nq = cd * depth + cd  # nq = 2*depth + 2 for cd=2
    max_level = (nq - 1) // cd
    c2s_start = max_level - 1
    am_start = c2s_start - max(0, c2s_levels - 1)
    output_level = max_level - boot_consumed
    output_nq = cd * output_level + cd if output_level >= 0 else cd

    return {
        'nq': nq,
        'depth': depth,
        'boot_depth': boot_consumed,
        'max_level': max_level,
        'c2s_start': c2s_start,
        'approx_mod_start': am_start,
        'expected_output_level': output_level,
        'expected_output_nq': output_nq,
        'headroom': headroom,
    }

  @staticmethod
  def required_rotation_indices(
      degree: int,
      num_slots: int,
      level_budget: Sequence[int],
  ) -> Tuple[int, ...]:
    """Return the exact rotation-key contract for one configuration."""
    return tuple(sorted({
        *_bootstrap_rotation_indices(degree, num_slots, level_budget),
        2 * degree - 1,
    }))

  # ------------------------------------------------------------------
  # control_gen
  # ------------------------------------------------------------------

  def control_gen(self, level_budget=None, secret_key_dist="uniform_ternary"):
    """Offline precomputation of all bootstrap parameters.

    Computes OpenFHE-matching scaling constants (deg, correction,
    correctionFactor, pre, post, scalar, corFactor, scaleEnc, scaleDec)
    and pre-encodes diagonal matrices with appropriate prescaling.
    """
    import numpy as _np

    self._require_param_cache()
    if level_budget is None:
      level_budget = [4, 4]
    if len(level_budget) != 2 or any(
        not isinstance(level, int) or level < 1 for level in level_budget
    ):
      raise ValueError("level_budget must contain two positive integers")
    max_transform_levels = max(1, int(math.log2(self._num_slots)))
    if any(level > max_transform_levels for level in level_budget):
      raise ValueError(
          f"level_budget entries cannot exceed log2(num_slots)="
          f"{max_transform_levels}"
      )
    # Encoded QP diagonals depend on the transform controls and scale schedule.
    # Re-running control generation on the same object must not reuse arrays
    # retained under identities from the preceding configuration.
    self._hoisted_lt_plaintexts.clear()
    self._level_budget = list(level_budget)
    c2s_levels, s2c_levels = level_budget

    # Chebyshev parameters
    if secret_key_dist == "uniform_ternary":
      self._cheby_degree, self._K, self._R = 88, 512, 6
    else:
      self._cheby_degree, self._K, self._R = 59, 256, 3

    # Compute Chebyshev coefficients for the bootstrap seed function.
    # The seed is: f(x) = (2pi)^{-1/2^R} * cos(2pi/2^R * (x - 0.25))
    # projected on [-K, K], matching OpenFHE's EvalChebyshevCoefficients.
    self._cheby_coeffs = jnp.asarray(
        util.compute_bootstrap_chebyshev_coefficients(
            self._K, self._R, self._cheby_degree),
        dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Step 1a: Compute OpenFHE-matching bootstrap scaling constants.
    #
    # With per-level scaling factors (FLEXIBLEAUTO), sf(0) = qDouble
    # (product of first cd primes). This means:
    #   pre = sf(0) / qDouble = 1.0 (exact, no precision loss!)
    #   deg = round(log2(qDouble / powP)) where powP = 2^round(log2(sf(0)))
    #   For our primes: qDouble ≈ 2^61, powP = 2^61, deg = 0
    #
    # Following OpenFHE FHECKKSRNS::EvalBootstrapSetup:
    #   qDouble = product of first 2 raw q_towers
    #   sf(0) = GetScalingFactorReal(0) = qDouble for FLEXIBLEAUTO cd=2
    #   powP = 2^round(log2(sf(0)))
    #   deg = round(log2(qDouble / powP))
    #   correctionFactor from heuristic, clamped to [7, 14]
    #   correction = correctionFactor - deg
    # ------------------------------------------------------------------
    q_towers = self._cache.q_towers
    cd = self._composite_degree

    # qDouble = product of first cd raw moduli (matches OpenFHE).
    self._qDouble = 1
    for i in range(cd):
      self._qDouble *= q_towers[i]

    # sf(0) = per-level scaling factor at level 0 = qDouble for FLEXIBLEAUTO.
    # GetScalingFactorReal(0): recursive index zero, represented by the full
    # chain's CROSS level (no towers dropped).
    self._sf0 = self._openfhe_scale_at_level(self._cache.max_level)

    powP = 2.0 ** round(math.log2(self._sf0))
    qDouble_f = float(self._qDouble)

    self._deg = round(math.log2(qDouble_f / powP)) if qDouble_f > powP else 0

    # correctionFactor: matches OpenFHE ckksrns-fhe.cpp ~line 800.
    # OpenFHE uses M/2 and numSlots where M = cyclotomic order = 2*N.
    # In CROSS, ctx.degree is the polynomial/ring dimension N.
    N_ring = self._degree
    n_slots = self._num_slots  # actual slot count (may be < N/2 for sparse packing)
    tmp = round(-0.2419 * (2.0 * math.log2(N_ring) + math.log2(n_slots))
                + 19.081)
    self._correctionFactor = max(7, min(14, tmp))
    self._correction = self._correctionFactor - self._deg

    # Derived scaling constants
    # OpenFHE has TWO different 'pre' values:
    # 1. Setup pre (EvalBootstrapSetup): used for scaleEnc = pre/k_setup.
    #    For UNIFORM_TERNARY, k_setup=1.
    #    For cd>1: pre_setup = 1.0.  For cd=1: pre_setup = qDouble / factor.
    # 2. Bootstrap pre (EvalBootstrap line 545): used for prescale.
    #    For FIXEDMANUAL/COMPOSITESCALINGMANUAL: pre_boot = sf0 / qDouble.
    if cd > 1:
      self._pre = 1.0                    # setup pre for scaleEnc
    else:
      self._pre = self._sf0 / qDouble_f  # setup pre for scaleEnc
    self._pre_bootstrap = self._sf0 / qDouble_f  # bootstrap pre for prescale
    self._post = 2.0 ** self._deg if self._deg >= 0 else 1.0
    self._scalar = self._post                # = 2^deg (same as post)
    # OpenFHE source-equivalent transform constants for UNIFORM_TERNARY:
    # setup k=1 gives scaleEnc=pre, runtime prescale is pre/(K*N), and the
    # post-S2C correction is exactly 2^correction.  The previous degree/slot
    # compensation and 4*num_slots gain formed a compensating end-to-end
    # factor, but changed the C2S/ApproxMod landing by 4x.
    self._corFactor = float(1 << self._correction)
    self._scaleEnc = self._pre if secret_key_dist == "uniform_ternary" \
        else 1.0 / self._K
    self._scaleDec = qDouble_f / self._sf0
    self._m_enc = 4 * n_slots
    self._c2s_normalization_gain = 1.0

    # Level allocation.
    # The prescale step (Step 1c) consumes one level before C2S, so
    # C2S diagonals must be encoded starting at max_level - 1.
    max_level = self.ctx.max_level
    # OpenFHE fully-packed bootstrap applies ModReduceInternal after
    # prescale and before C2S, so C2S starts one level below max_level.
    self._c2s_start_level = max_level - 1
    # C2S rescales only between collapsed linear-transform levels.  The
    # final C2S transform output remains at nsd=2, matching OpenFHE's
    # EvalLinearTransform output before ApproxMod.

    # DFT diagonal decomposition.
    # C2S (CoeffToSlot) uses the encoding-direction (DIF) decomposition.
    # This computes P @ U^H @ x (≈ P @ U^{-1} @ x up to N scaling),
    # where P is the bit-reversal permutation inherent in DIF FFT.
    # The bit-reversal is harmless: approx mod reduction is per-slot
    # (order-independent), and S2C's DIT structure undoes it.
    #
    # S2C (SlotToCoeff) uses the decoding-direction (DIT) decomposition.
    # This expects bit-reversed input (from C2S) and produces natural-order
    # slot output, applying U (forward DFT).
    #
    # In _decompose_dft_into_levels: inverse=False → C2S, inverse=True → S2C.
    if self._sparse:
      # Sparse packing: OpenFHE builds TWO collapses -- the plain one and the
      # flag_i (innermost butterfly x (-+i)) variant -- and concatenates them
      # horizontally into 2*n_slots-length diagonal patterns embedded in the
      # N/2 slot space (EvalCoeffsToSlots/SlotsToCoeffsPrecompute sparse
      # branches, ckksrns-fhe.cpp 1617-1618 / 1772-1773).  The [d | i*d]
      # two-block structure is what makes the conjugate-ADD single-ApproxMod
      # lossless and the post-S2C rotate-by-slots reconstructive.  Rotation
      # keys keep their >= n_slots components (key_mod = 2*n_slots).
      if min(c2s_levels, s2c_levels) < 2:
        raise ValueError("sparse packing requires level_budget >= 2 "
                         "(the direct LT path is not ported for sparse)")
      key_mod = 2 * n_slots

      def _concat_dicts(list_a, list_b):
        out = []
        for da, db in zip(list_a, list_b, strict=True):
          keys = set(da.keys()) | set(db.keys())
          merged = {}
          for k in keys:
            va = da.get(k)
            vb = db.get(k)
            if va is None:
              va = _np.zeros(n_slots, dtype=_np.complex128)
            if vb is None:
              vb = _np.zeros(n_slots, dtype=_np.complex128)
            merged[k] = _np.concatenate([_np.asarray(va), _np.asarray(vb)])
          out.append(merged)
        return out

      c2s_diag_list = _concat_dicts(
          _decompose_dft_into_levels(n_slots, self._m_enc, c2s_levels,
                                     inverse=False, key_mod=key_mod),
          _decompose_dft_into_levels(n_slots, self._m_enc, c2s_levels,
                                     inverse=False, flag_i=True,
                                     key_mod=key_mod))
      s2c_diag_list = _concat_dicts(
          _decompose_dft_into_levels(n_slots, self._m_enc, s2c_levels,
                                     inverse=True, key_mod=key_mod),
          _decompose_dft_into_levels(n_slots, self._m_enc, s2c_levels,
                                     inverse=True, flag_i=True,
                                     key_mod=key_mod))
      pad = _np.ones(2 * n_slots, dtype=_np.complex128)
      c2s_hoisted_groups = None
      s2c_hoisted_groups = None
    else:
      c2s_diag_list = _decompose_dft_into_levels(
          n_slots, self._m_enc, c2s_levels, inverse=False)
      s2c_diag_list = _decompose_dft_into_levels(
          n_slots, self._m_enc, s2c_levels, inverse=True)
      c2s_hoisted_groups = _decompose_dft_into_hoisted_groups(
          n_slots, self._m_enc, c2s_levels, inverse=False
      )
      s2c_hoisted_groups = _decompose_dft_into_hoisted_groups(
          n_slots, self._m_enc, s2c_levels, inverse=True
      )
      pad = _np.ones(n_slots, dtype=_np.complex128)

    while len(c2s_diag_list) < c2s_levels:
      c2s_diag_list.append({0: pad})
    c2s_diag_list = c2s_diag_list[:c2s_levels]
    while len(s2c_diag_list) < s2c_levels:
      s2c_diag_list.append({0: pad})
    s2c_diag_list = s2c_diag_list[:s2c_levels]

    def _align_hoisted_levels(levels, requested_levels):
      if levels is None:
        return None
      while len(levels) < requested_levels:
        levels.append([{
            'giant': 0,
            'entries': [(0, pad)],
        }])
      return levels[:requested_levels]

    c2s_hoisted_groups = _align_hoisted_levels(
        c2s_hoisted_groups, c2s_levels
    )
    s2c_hoisted_groups = _align_hoisted_levels(
        s2c_hoisted_groups, s2c_levels
    )

    # Step 1b: Apply scaleEnc/scaleDec to LAST level only (matching OpenFHE).
    # OpenFHE applies the scale factor only to the last set of diagonals
    # in both C2S and S2C (see EvalCoeffsToSlotsPrecompute line 1581,
    # EvalSlotsToCoeffsPrecompute line 1734).
    #
    # Prescale pre/(K*N) is applied as a SEPARATE EvalMult + rescale step
    # in bootstrap(), matching OpenFHE's flow. This ensures slot values after
    # C2S are in the correct range for the Chebyshev + double-angle iterations.
    # (If absorbed into C2S, the values become too small for the DA "near-zero trap".)
    # OpenFHE EvalBootstrap line 545+661: pre_boot * (1.0 / (K * N))
    # where pre_boot = GetScalingFactorReal(0) / qDouble.
    self._prescale_factor = self._pre_bootstrap / (self._K * self._degree)

    last_c2s = len(c2s_diag_list) - 1
    last_s2c = len(s2c_diag_list) - 1
    for level_idx, diags in enumerate(c2s_diag_list):
      for k in list(diags.keys()):
        if level_idx == last_c2s:
          diags[k] = diags[k] * self._scaleEnc * self._c2s_normalization_gain
    for level_idx, diags in enumerate(s2c_diag_list):
      for k in list(diags.keys()):
        if level_idx == last_s2c:
          diags[k] = diags[k] * self._scaleDec  # scaleDec on LAST level only

    def _scale_hoisted_last(levels, factor):
      if not levels:
        return levels
      last = len(levels) - 1
      for level_idx, groups in enumerate(levels):
        if level_idx != last:
          continue
        for group in groups:
          group['entries'] = [
              (baby, diagonal * factor)
              for baby, diagonal in group['entries']
          ]
      return levels

    c2s_hoisted_groups = _scale_hoisted_last(
        c2s_hoisted_groups,
        self._scaleEnc * self._c2s_normalization_gain,
    )
    s2c_hoisted_groups = _scale_hoisted_last(
        s2c_hoisted_groups, self._scaleDec
    )

    self._all_rot_indices = list(_bootstrap_rotation_indices(
        self._degree, n_slots, level_budget
    ))

    # Pre-encode all diagonals into NTT form.
    # C2S diagonals: encoded at known fixed levels (c2s_start_level down).
    skip_legacy_c2s_plaintexts = (
        self._low_mem
        and os.environ.get("BOOTSTRAP_DISABLE_HOISTED_LT") != "1"
        and not self._sparse
        and self._num_slots > 1
    )
    if skip_legacy_c2s_plaintexts:
      # The Q-domain fallback encodings are unused by the full-packed hoisted
      # path and cost over a GiB at N=65536.  Keep the raw diagonals; they are
      # encoded directly in QP by the selected evaluator.
      self._c2s_encoded = []
    else:
      self._c2s_encoded = self._pre_encode_diags(
          c2s_diag_list, self._c2s_start_level)
    self._c2s_raw_diags = c2s_diag_list
    self._c2s_hoisted_groups = c2s_hoisted_groups
    # S2C diagonals: the actual start level depends on how many levels
    # ApproxMod consumes, which varies with the Chebyshev PS tree depth.
    # Store raw complex diagonals and encode lazily in slot_to_coef().
    self._s2c_raw_diags = s2c_diag_list
    self._s2c_hoisted_groups = s2c_hoisted_groups
    self._s2c_encoded = None  # Will be populated on first slot_to_coef call

  def _pre_encode_diags(self, diag_list, start_level):
    """Pre-encode a list of per-level diagonal dicts into NTT form.

    Uses per-level scaling factors (FLEXIBLEAUTO) so that encoded
    diagonals match the rescale divisor at each level.
    """
    self._require_param_cache()
    encoded_list = []
    for level_idx, diags in enumerate(diag_list):
      lev = start_level - level_idx
      if lev < 0:
        encoded_list.append({})
        continue
      moduli = self._cache.q_moduli_at_level(lev)
      num_q = len(moduli)
      sf_level = self._openfhe_scale_at_level(lev)
      encoded_list.append({
          int(k): self._encode_diagonal(d, moduli, num_q, scale=sf_level)
          for k, d in diags.items()
      })
    return encoded_list

  # ------------------------------------------------------------------
  # setup_key
  # ------------------------------------------------------------------

  def setup_key(self):
    """Generate and register all rotation keys needed for bootstrap.

    Also registers the conjugation key at Galois index M-1 (used by
    the Step 3b conjugate trick).
    """
    if self._all_rot_indices is None:
      raise RuntimeError("control_gen() must be called before setup_key()")
    self._require_param_cache()

    import key_gen as kg
    cache = self._cache

    all_indices = list(self._all_rot_indices)
    conj_idx = self._m - 1
    if conj_idx not in all_indices:
      all_indices.append(conj_idx)

    for rot_idx in all_indices:
      if rot_idx not in cache.coef_maps:
        cache.coef_maps[rot_idx] = jnp.asarray(
            util.precompute_auto_map(
                self._degree,
                kg.find_automorphism_index_2n_complex(rot_idx, self._m)),
            dtype=jnp.int32)
      if rot_idx not in cache.rot_indices:
        cache.rot_indices.append(rot_idx)

  # ------------------------------------------------------------------
  # Public pipeline stages
  # ------------------------------------------------------------------

  def bootstrap(self, ct):
    """Full bootstrap: AdjustCiphertext -> ModRaise -> Prescale -> C2S ->
    ApproxMod -> S2C -> Correction.

    Pipeline (matching OpenFHE FHECKKSRNS::EvalBootstrap lines 558-824):

    Step 0:  ModReduceInternal to normalize nsd to 1 (no-op if already 1).
    Step 1a: AdjustCiphertext -- scale message by adjustmentFactor and
             drop cd primes.  Requires the depleted ct to have >= 2*cd
             primes.  If the depleted ct has exactly cd primes, this step
             is SKIPPED (the correction is folded into corFactor).
    Step 1b: ExtendCiphertext (mod_raise) -- OpenFHE's unreduced CRT
             interpolation lift to the full chain.
    Step 1c: Prescale by pre/(K*N) -- separate EvalMult + rescale.
    Step 2:  CoeffToSlot.
    Step 3:  Conjugate trick -> ApproxMod (Chebyshev + DA) -> recombine.
    Step 4:  SlotToCoeff.
    Step 5:  corFactor = 2^correction multiply to restore original scale.
    """
    self._require_param_cache()
    self._ensure_canonical(ct)
    cd = self._composite_degree
    actual_nq = ct.polynomial.shape[-1]

    # Step 1a: AdjustCiphertext (OpenFHE ckksrns-fhe.cpp:2228-2262)
    #
    # For FLEXIBLEAUTO / COMPOSITESCALINGAUTO:
    #   adjustmentFactor = (targetSF/sourceSF) * (modToDrop/sourceSF) * 2^(-correction)
    #
    # This is an EvalMult(adjustmentFactor) followed by ModReduceInternal(cd),
    # consuming cd primes.  The ct must have >= 2*cd primes.
    adjust_applied = False
    if actual_nq > cd:
      ct = self._adjust_ciphertext(ct)
      adjust_applied = True
      # After AdjustCiphertext: ct has actual_nq - cd primes.
      # If it started with 2*cd, it now has cd primes.

    raised = self.mod_raise(ct)

    # Step 1c: Prescale by pre/(K*N).  OpenFHE keeps the sparse partial-sum
    # rotations at nsd=2 and performs ModReduceInternal only afterwards; this
    # attenuates their key-switch noise together with the prescaled message.
    # Fully packed mode has no partial sum and rescales immediately.
    # OpenFHE line 661: cc->EvalMultInPlace(ct, pre/(K*N)).
    raised_level = self._infer_level(raised)
    raised = self._ptct_mul_scalar(raised, self._prescale_factor, raised_level)
    self._stage_checkpoint("prescale")

    # Sparse packing: fold the gap redundant copies of the slot pattern
    # before the prescale rescale and C2S (OpenFHE ckksrns-fhe.cpp 772-784):
    #   for (j = 1; j < N/(2*slots); j <<= 1)
    #     raised += EvalRotate(raised, j*slots)
    if self._sparse:
      ps_level = self._infer_level(raised)
      self._ensure_canonical(raised, ps_level)
      j = 1
      while j < self._gap:
        rot = self._rotate_by(raised, j * self._num_slots, ps_level)
        raised = self._ct_add(raised, rot, ps_level)
        j <<= 1
      self._stage_checkpoint("partial-sum")

    raised_level = self._infer_level(raised)
    if self._get_nsd(raised) == 2 and raised_level > 0:
      raised = self._he_op_rescale(raised_level, raised_level - 1, raised,
                                   decrement_nsd=True)

    slot_ct = self.coef_to_slot(raised)
    self._stage_checkpoint("C2S")

    # Step 3b: Conjugate trick for fully-packed mode.
    # C2S output has complex slot values z = a + bi. Chebyshev is real,
    # so split into real/imag:
    #   conj_ct = Conjugate(slot_ct)                       # a - bi
    #   real_ct = slot_ct + conj_ct                        # 2a
    #   imag_ct = (slot_ct - conj_ct) * X^(3*num_slots)    # 2b
    # Apply approx_mod to each, then recombine:
    #   imag_reduced * X^(num_slots) + real_reduced
    conj_level = self._infer_level(slot_ct)
    self._ensure_canonical(slot_ct, conj_level)
    conj_ct = self._conjugate(slot_ct, conj_level)
    if self._sparse:
      # Sparse packing: the slot values are real-embedded, so conjugate-ADD
      # gives 2*Re(z) and a SINGLE ApproxMod suffices; there is no imag path
      # and no X^(3*slots) monomial (OpenFHE ckksrns-fhe.cpp 761/786).
      real_ct = self._ct_add(slot_ct, conj_ct, conj_level)
      reduced = self.eval_approx_mod_reduction(real_ct)
      self._stage_checkpoint("approxmod-re")
    else:
      real_ct = self._ct_add(slot_ct, conj_ct, conj_level)
      imag_ct = self._ct_sub(slot_ct, conj_ct, conj_level)
      imag_ct = self._mult_by_monomial(
          imag_ct, 3 * self._num_slots, conj_level)

      real_reduced = self.eval_approx_mod_reduction(real_ct)
      self._stage_checkpoint("approxmod-re")
      imag_reduced = self.eval_approx_mod_reduction(imag_ct)
      self._stage_checkpoint("approxmod-im")

      imag_level = self._infer_level(imag_reduced)
      imag_reduced = self._mult_by_monomial(
          imag_reduced, self._num_slots, imag_level)

      real_level = self._infer_level(real_reduced)
      target_level = min(real_level, imag_level)
      if real_level > target_level:
        real_reduced, real_level = self._match_levels(
            real_reduced, real_level, target_level)
      if imag_level > target_level:
        imag_reduced, imag_level = self._match_levels(
            imag_reduced, imag_level, target_level)

      self._ensure_canonical(real_reduced, target_level)
      self._ensure_canonical(imag_reduced, target_level)
      reduced = self._ct_add(real_reduced, imag_reduced, target_level)

    # Pre-S2C ModReduceInternal: rescale nsd=2 DA output to nsd=1.
    reduced_level = self._infer_level(reduced)
    if self._get_nsd(reduced) == 2 and reduced_level > 0:
      reduced = self._he_op_rescale(reduced_level, reduced_level - 1, reduced)
    self._stage_checkpoint("recombine")

    result = self.slot_to_coef(reduced)
    self._stage_checkpoint("S2C")

    # Sparse packing: reconstruct the gap-strided coefficient layout with one
    # rotate-by-slots + add (OpenFHE ckksrns-fhe.cpp 818, before corFactor).
    if self._sparse:
      s2c_level = self._infer_level(result)
      self._ensure_canonical(result, s2c_level)
      rot = self._rotate_by(result, self._num_slots, s2c_level)
      result = self._ct_add(result, rot, s2c_level)

    # Step 2a: Post-S2C correction factor.
    # corFactor = 2^correction undoes the 2^(-correction) from
    # AdjustCiphertext, exactly as in OpenFHE.
    # Only apply if AdjustCiphertext was actually applied; when skipped
    # (depleted ct with nq <= cd), there is no 2^(-correction) to undo.
    #
    # corFactor is an exact power-of-two integer coefficient multiply and
    # consumes no level.
    effective_corFactor = self._corFactor if adjust_applied else 1.0
    corr_residual = 1.0  # = int_factor / corFactor, folds into the output scale
    if effective_corFactor != 1.0:
      result_level = self._infer_level(result)
      self._ensure_canonical(result, result_level)
      moduli = self._cache.q_moduli_at_level(result_level)
      int_factor = int(round(effective_corFactor))
      corr_residual = int_factor / effective_corFactor
      moduli_arr = jnp.array(moduli, dtype=jnp.uint64)
      result.polynomial = (
          (result.polynomial.astype(jnp.uint64) * int_factor) % moduli_arr
      ).astype(jnp.uint32)
    # Preserve the scale tracked through the complete C2S -> ApproxMod -> S2C
    # chain.  Real composite-prime products need not equal the cache's nominal
    # per-level scale, so resetting it here injects a constant decode gain.
    # Only the rounded integer correction needs to be folded into that scale.
    if corr_residual != 1.0:
      self._set_scale(result, self._get_scale(result) * corr_residual)

    # Optionally subtract a separately precomputed bootstrap(0) template.
    if (self._dc_subtract and self._dc_template is not None
        and not self._computing_dc_template):
      result = self._dc_correct(result)

    return self._ensure_canonical(result)

  def _dc_correct(self, result):
    """Subtract the cached DC template (bootstrap of zero) from a bootstrap output.

    Uses the SCALE-SAFE _ct_op_aligned (not the raw _ct_sub): the template was
    precomputed in a separate bootstrap, so its tracked scale/level can differ
    from the live result by a small per-level-SF drift that the raw subtraction
    silently honors as garbage (observed: correct at N=4096/8192 where they
    happen to match, but garbage at N=16384). _ct_op_aligned aligns level + scale
    before combining, matching OpenFHE AdjustForAddOrSubInPlace.
    """
    tmpl = self._dc_template
    r_lvl = self._infer_level(result)
    t_lvl = self._infer_level(tmpl)
    if self._mem_trace:
      def _s(ct):
        try:
          return f"2^{math.log2(max(self._get_scale(ct), 1.0)):.3f}"
        except Exception:
          return "?"
      print(f"[dc_correct] result lvl={r_lvl} nsd={self._get_nsd(result)} "
            f"scale={_s(result)} | tmpl lvl={t_lvl} nsd={self._get_nsd(tmpl)} "
            f"scale={_s(tmpl)}", flush=True)
    out, _ = self._ct_op_aligned(result, r_lvl, tmpl, t_lvl, op='sub')
    return out

  def precompute_dc_template(self):
    """Compute and cache bootstrap(zero) as the DC-bias template. Call once after
    setup_key(). Costs one bootstrap. Required for BOOTSTRAP_DC_SUBTRACT to act."""
    self._require_param_cache()
    slots = [0j] * self._num_slots
    ct = self.ctx.encrypt(self.ctx.encode(slots))
    self._ensure_canonical(ct)
    self._set_scale(ct, self._sf0)
    self._set_nsd(ct, 1)
    ct_dep, _ = self._level_reduce(ct, self._infer_level(ct), 1)
    self._computing_dc_template = True
    try:
      self._dc_template = self.bootstrap(ct_dep)
    finally:
      self._computing_dc_template = False
    return self._dc_template

  def _mult_by_integer(self, ct, int_factor: int, level: int):
    """Multiply ciphertext polynomial coefficients by an integer (mod q).

    OpenFHE MultByIntegerInPlace: scales the MESSAGE (and noise) by int_factor
    without changing the tracked scale or consuming a level.  Both message and
    noise scale by int_factor, which is exactly what Meta-BTS needs so the
    bootstrap error becomes extractable after scale-up.  Returns a new ct.
    """
    self._ensure_canonical(ct, level)
    moduli = self._cache.q_moduli_at_level(level)
    num_q = len(moduli)
    moduli_arr = jnp.array(moduli, dtype=jnp.uint64).reshape(
        1, 1, 1, 1, num_q
    )
    factors = jnp.array(
        [int(int_factor) % int(q) for q in moduli], dtype=jnp.uint64
    ).reshape(1, 1, 1, 1, num_q)
    out = copy.copy(ct)
    out.polynomial = (
        (ct.polynomial.astype(jnp.uint64) * factors)
        % moduli_arr
    ).astype(jnp.uint32)
    self._copy_scale(out, ct)
    self._copy_nsd(out, ct)
    return out

  @staticmethod
  def _meta_precision(precision) -> int:
    """Validate a Meta-BTS precision and return its integer bit count."""
    if isinstance(precision, bool) or not isinstance(precision, numbers.Integral):
      raise TypeError(
          f"Meta-BTS precision must be an integer, got {precision!r}."
      )
    precision = int(precision)
    if not 1 <= precision <= 62:
      raise ValueError(
          f"Meta-BTS precision must be in [1, 62], got {precision}."
      )
    return precision

  def meta_bootstrap(self, ct, precision: int):
    """Meta-BTS: two-iteration bootstrap for ~doubled precision.

    Port of OpenFHE FHECKKSRNS::EvalBootstrap numIterations=2 path
    (ckksrns-fhe.cpp:501-548).  Algorithm:

      b1   = bootstrap(input)            # message (m + e1)
      b1P  = b1 * 2^precision            # P*(m+e1) = Pm + P*e1
      err  = (b1P mod q_input) - (input * 2^precision)   # = P*e1  (depleted)
      b2   = bootstrap(err)              # P*e1 + e2
      out  = b1P - b2                    # Pm - e2
      out  = out / 2^precision           # m - e2/P   (via scale *= P, free)

    Since e2 ~ 2^-precision * |P*e1| = 2^-precision * P * e1, the final error
    e2/P ~ 2^-precision * e1 -- the precision roughly doubles.

    REQUIRES post-bootstrap headroom: the first bootstrap output must have more
    Q-towers than the depleted input (else the error cannot be formed and we
    fall back to the single-iteration result).

    Args:
      ct: depleted ciphertext to refresh.
      precision: number of bits to scale the error by (~ the single-pass
        bit-precision; P = 2^precision).
    Returns:
      Refreshed ciphertext with ~2x the single-pass precision.
    """
    self._require_param_cache()
    self._ensure_canonical(ct)
    precision = self._meta_precision(precision)
    P = 1 << precision
    init_nq = ct.polynomial.shape[-1]
    init_level = self._infer_level(ct)

    # Preserve a clean copy of the depleted input for the error computation.
    ct_in = copy.copy(ct)
    ct_in.polynomial = ct.polynomial.copy()
    self._copy_scale(ct_in, ct)
    self._copy_nsd(ct_in, ct)

    # --- First bootstrap ---
    b1 = self.bootstrap(ct)
    L_out = self._infer_level(b1)
    # The bootstrap output is nsd=2; the error formation subtracts it against
    # the nsd=1 depleted input, and _adjust_ct_to_match_scale aligns only
    # same-nsd operands. Normalize to canonical nsd=1 first so the message
    # cancels exactly (a cross-nsd subtract leaks a message-proportional term).
    b1, L_out = self._ensure_nsd1(b1, L_out)
    boot_nq = b1.polynomial.shape[-1]

    # Headroom check (OpenFHE: if bootstrappingSizeQ <= initSizeQ, bail).
    # Meta-BTS requires the first-pass output to have more Q-towers than the
    # depleted input so the error can be formed and re-bootstrapped; without it
    # the two-iteration refinement is impossible.
    if boot_nq <= init_nq:
      raise RuntimeError(
          f"meta_bootstrap requires post-bootstrap headroom: first-pass output "
          f"has boot_nq={boot_nq} towers <= input init_nq={init_nq}. Increase NQ "
          f"(e.g. NQ=58 at N=4096) so the bootstrap output level exceeds the "
          f"input level.")

    # --- Scale up the first-pass result by P:  Pm + P*e1 ---
    b1P = self._mult_by_integer(b1, P, L_out)

    # --- Form the bootstrap error at the input level ---
    # A SECOND scaled copy of b1 is brought down to the input level + scale to
    # subtract against the scaled input.  We pass b1P_err at its HIGH level and
    # ctP at the input level so _ct_sub_aligned takes the float64 scale+level
    # alignment path (a same-level subtraction would accept the ~0.5% per-level
    # SF mismatch, which would swamp the small P*e1 error).
    b1P_err = copy.copy(b1P)
    b1P_err.polynomial = b1P.polynomial.copy()
    self._copy_scale(b1P_err, b1P)
    self._copy_nsd(b1P_err, b1P)

    # Scale the original input by P: Pm  (at the input level/scale).
    ctP = self._mult_by_integer(ct_in, P, init_level)

    # error = (Pm + P*e1) - Pm = P*e1.  b1P_err (high level) is adjusted to
    # ctP's level and scale via float64 correction, then subtracted.
    error, err_level = self._ct_sub_aligned(b1P_err, L_out, ctP, init_level)
    # Present the error like a fresh depleted ciphertext for re-bootstrap.
    el = self._infer_level(error)
    if el > init_level:
      error, el = self._level_reduce(error, el, init_level)
    self._set_scale(error, self._sf0)
    self._set_nsd(error, 1)

    # Free first-pass intermediates before the second bootstrap.  At large N the
    # two bootstraps otherwise accumulate ~30 GB of JAX buffers each and OOM; the
    # only objects needed past this point are b1P (final subtract) and `error`
    # (the second bootstrap input).  Dropping b1/b1P_err/ct_in/ctP and forcing a
    # GC reclaims the first pass's working set.
    import gc as _gc
    del b1, b1P_err, ct_in, ctP
    _gc.collect()

    # --- Second bootstrap: refine the error ---
    b2 = self.bootstrap(error)
    L2 = self._infer_level(b2)
    # Normalize to nsd=1 so the final b1P - b2 combine (b1P is nsd=1) is
    # same-nsd and aligns exactly.
    b2, L2 = self._ensure_nsd1(b2, L2)

    # --- Final: b1P - b2 = Pm - e2,  then /P via scale (free) ---
    final, f_level = self._ct_sub_aligned(b1P, L_out, b2, L2)
    self._set_scale(final, self._get_scale(final) * P)
    return self._ensure_canonical(final)

  def _adjust_ciphertext(self, ct):
    """Adjust ciphertext before mod-raise (OpenFHE AdjustCiphertext).

    Matches OpenFHE ckksrns-fhe.cpp lines 2228-2262 for
    FLEXIBLEAUTO / COMPOSITESCALINGAUTO:

      adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF) * 2^(-correction)

    where:
      targetSF  = GetScalingFactorReal(0) = sf(0)
      sourceSF  = ciphertext's current scaling factor
      modToDrop = product of the last cd primes of the ciphertext
      correction= correctionFactor - deg

    Performs EvalMult(adjustmentFactor) then ModReduceInternal(cd),
    consuming cd primes and setting the scale to targetSF.
    """
    cd = self._composite_degree
    actual_nq = ct.polynomial.shape[-1]
    if actual_nq <= cd:
      return ct  # Cannot drop primes from a ct with <= cd primes

    # Determine sourceSF (the ciphertext's current encoding scale).
    # If tracked, use the tracked value.  Otherwise derive from level.
    sourceSF = self._get_scale(ct)

    # targetSF = sf(0) = per-level scaling factor at level 0.
    targetSF = self._sf0

    # modToDrop = product of the LAST cd primes of the ciphertext.
    # The ciphertext's primes are q_towers[0 .. actual_nq-1].
    # The "last cd" are q_towers[actual_nq - cd .. actual_nq - 1].
    q_towers = self._cache.q_towers
    modToDrop = 1.0
    for j in range(cd):
      modToDrop *= float(q_towers[actual_nq - 1 - j])

    # adjustmentFactor = (targetSF/sourceSF) * (modToDrop/sourceSF) *
    # 2^(-correction)
    correction_pow = math.pow(2, -self._correction)
    adjustmentFactor = (targetSF / sourceSF) * (modToDrop / sourceSF) * correction_pow

    # Determine the level for ptct_mul.
    # The ct with actual_nq primes is at level:
    ct_level = self._infer_level(ct)

    # EvalMult by adjustmentFactor (ptct_mul).
    self._ensure_canonical(ct)
    ct = self._ptct_mul_scalar(ct, adjustmentFactor, ct_level)

    # ModReduceInternal(cd): drop the last cd primes.
    # In CROSS this is he_op_rescale which drops cd primes.
    ct_level_after = ct_level - 1
    if ct_level_after >= 0:
      ct = self._he_op_rescale(ct_level, ct_level_after, ct)

    # Set the scaling factor to targetSF (matching OpenFHE line 2261).
    self._set_scale(ct, targetSF)

    return ct

  def mod_raise(self, ct):
    """Raise a depleted ciphertext with OpenFHE's composite CRT lift.

    _mod_raise_array is a standard-residue algorithm (its signed-digit CRT
    interpolation reads the integer value of each residue), so under a
    Montgomery backend the payload is decoded to standard form for the lift
    and re-encoded afterwards — one conversion pair per bootstrap. Identity
    for standard-format backends.
    """
    cache = self._require_param_cache()
    self._ensure_canonical(ct)
    payload = ct.to_array()
    standard_backend = cache.ff_q_max.computation_format_is_standard
    if not standard_backend:
      ff_depleted = cache.ff_q_max.slice(len(ct.moduli))
      payload = jnp.asarray(
          ff_depleted.to_original_format(
              jnp.asarray(payload, jnp.uint64)),
          jnp.uint32)
    raised_poly = _mod_raise_array(payload, ct.moduli, cache.q_towers)
    if not standard_backend:
      raised_poly = cache.ff_q_max.to_computation_format(
          jnp.asarray(raised_poly, jnp.uint64))
    out = self._new_ciphertext(raised_poly, cache.max_level)
    # After mod_raise, the ct is at max level with the same message
    # as before (exact CRT extension). The scale is the original scale
    # of the depleted ct = scaling_factor_at_level(0) for a level-0 ct.
    # This is qDouble = product of first cd primes.
    if hasattr(self, '_sf0') and self._sf0 is not None:
      self._set_scale(out, self._sf0)
    # ModRaise changes the coefficient representatives but not their residues
    # in the depleted basis, so the CKKS noise-scale degree is unchanged.
    # The input comes from AdjustCiphertext (nsd=1 after rescale) or
    # the depleted ct (default nsd=1). Set explicitly for clarity.
    self._set_nsd(out, self._get_nsd(ct))
    return out

  def coef_to_slot(self, ct):
    """Homomorphic encoding FFT via diagonal linear transforms."""
    self._require_param_cache()
    if self._c2s_encoded is None:
      raise RuntimeError("control_gen() must be called before coef_to_slot()")

    self._ensure_canonical(ct)
    current_ct = ct
    start_level = self._c2s_start_level
    current_level = self._infer_level(current_ct)
    if current_level > start_level:
      current_ct = self._he_op_rescale(current_level, start_level,
                                       current_ct, decrement_nsd=True)

    # OpenFHE's one-level linear transform hoists the c1 decomposition and
    # retains every baby-step product in QP until giant-step aggregation.
    # The direct path is now the default; the legacy per-diagonal Q path is
    # retained behind an explicit diagnostic opt-out.
    use_hoisted_lt = (
        os.environ.get("BOOTSTRAP_DISABLE_HOISTED_LT") != "1"
        and not self._sparse
        and self._num_slots > 1
        and self._c2s_raw_diags is not None
        and len(self._c2s_raw_diags) == 1
    )
    if use_hoisted_lt:
      return self._eval_linear_transform_hoisted(
          current_ct, self._c2s_raw_diags[0], start_level
      )

    use_hoisted_collapsed = (
        os.environ.get("BOOTSTRAP_DISABLE_HOISTED_LT") != "1"
        and not self._sparse
        and self._num_slots > 1
        and self._c2s_raw_diags is not None
        and len(self._c2s_raw_diags) > 1
    )
    if use_hoisted_collapsed:
      for i, diags in enumerate(self._c2s_raw_diags):
        level = start_level - i
        exact_groups = (
            self._c2s_hoisted_groups[i]
            if self._c2s_hoisted_groups is not None
            else None
        )
        current_ct = self._eval_collapsed_linear_transform_hoisted(
            current_ct, diags, level, exact_groups=exact_groups
        )
        if i < len(self._c2s_raw_diags) - 1 and level > 0:
          current_ct = self._he_op_rescale(
              level, level - 1, current_ct, decrement_nsd=True
          )
        self._stage_checkpoint(f"c2s-L{i}", current_ct)
      return self._ensure_canonical(current_ct)

    # Low-memory control generation may omit these Q-domain plaintexts when
    # the hoisted evaluator is selected.  If the diagnostic opt-out is toggled
    # before evaluation, rebuild the deterministic legacy encodings instead of
    # silently applying an empty transform.
    if not self._c2s_encoded:
      self._c2s_encoded = self._pre_encode_diags(
          self._c2s_raw_diags, start_level
      )
    for i, enc in enumerate(self._c2s_encoded):
      level = start_level - i
      if level < 0:
        break
      self._ensure_canonical(current_ct, level)
      current_ct = self._eval_linear_transform(current_ct, enc, level)
      self._ensure_canonical(current_ct, level)
      if i < len(self._c2s_encoded) - 1 and level > 0:
        # decrement_nsd=True: nsd=3→2 (preserves nsd=2 flow from prescale)
        current_ct = self._he_op_rescale(level, level - 1, current_ct,
                                         decrement_nsd=True)
      self._stage_checkpoint(f"c2s-L{i}")

    return self._ensure_canonical(current_ct)

  def eval_approx_mod_reduction(self, ct):
    """Approximate modular reduction: Chebyshev + double-angle.

    Matches OpenFHE FHECKKSRNS::EvalChebyshevSeriesPS +
    ApplyDoubleAngleIterations exactly.

    Steps 1c + 2a changes:
    - REMOVED the 1/(2K) input scaling (now done in bootstrap() prescale)
    - REMOVED the K * 2^R output scaling (replaced by corFactor in bootstrap())
    - For COMPOSITE: do NOT multiply by 2^deg after DA
    """
    self._require_param_cache()
    if self._cheby_coeffs is None:
      raise RuntimeError(
          "control_gen() must be called before eval_approx_mod_reduction()")

    level = self._infer_level(ct)
    R = self._R
    self._ensure_canonical(ct, level)

    # OpenFHE explicitly normalizes the nsd=2 C2S/conjugation output before
    # EvalChebyshevSeries.  Doing this once here is not interchangeable with
    # letting individual PS branches normalize their own copies: the latter
    # consumed an extra level along the longest branch and made the entire
    # ApproxMod endpoint one composite group too low.
    ct, level = self._ensure_nsd1(ct, level)

    # Chebyshev evaluation (Paterson-Stockmeyer)
    ct, level = self._eval_chebyshev_ps(ct, level)
    self._stage_checkpoint("cheby-ps", ct)

    # OpenFHE's explicit pre-double-angle ModReduce: PS returns nsd=2.  The
    # first DA square then runs at this normalized level without consuming a
    # further group; later iterations normalize the preceding nsd=2 result.
    ct, level = self._ensure_nsd1(ct, level)

    # With skip_rescale=False: Chebyshev output is at S² (stable).  The
    # explicit normalization above consumes the first DA level; the first
    # square then stays at that level, while later S² inputs normalize inside
    # _he_op_mul.  Across R iterations the stage consumes R levels and keeps
    # each DA result at S².

    # Double-angle iterations (OpenFHE ApplyDoubleAngleIterations)
    for iter_i in range(1 - R, 1):
      if level < 1:
        break
      scalar = -math.pow(2.0 * math.pi, -math.pow(2.0, iter_i))
      self._ensure_canonical(ct, level)
      mul_level = level if self._get_nsd(ct) == 1 else level - 1
      if mul_level < 0:
        break
      ct_sq = self._he_op_mul(mul_level, ct, ct)
      new_level = self._infer_level(ct_sq)
      # 2*ct via ct+ct (free, no level) instead of ptct_mul(2.0).
      # OpenFHE: EvalAddInPlace(ct, EvalAdd(ct, scalar)) = 2*ct + scalar
      ct_2x = self._ct_add(ct_sq, ct_sq, new_level)  # 2 * ct_sq (free, no level)
      ct_2x = self._add_scalar(ct_2x, scalar, new_level, noise_scale_deg=2)
      ct, level = ct_2x, new_level
      # Barrier on the DA result: each iteration squares a degree-N ciphertext
      # via relin key-switch; without forcing completion the intermediates from
      # all R iterations stay live (the N=65536 OOM site).
      self._stage_checkpoint(f"da{iter_i + self._R}", ct)

    # Step 2a: Scale message back up after Chebyshev + double-angle.
    # OpenFHE line 710-712: MultByInteger(2^deg) for FLEXIBLEAUTO.
    # For COMPOSITESCALINGAUTO/MANUAL (cd>1), this is SKIPPED because
    # pre = sf/qDouble ≈ 1/2^deg already absorbs the factor.
    # (OpenFHE line 710: "if st != COMPOSITESCALINGAUTO && st != COMPOSITESCALINGMANUAL")
    if self._composite_degree <= 1 and self._deg > 0:
      # Free integer multiply (no level consumed, like corFactor).
      int_scalar = int(self._scalar)
      moduli = self._cache.q_moduli_at_level(level)
      moduli_arr = jnp.array(moduli, dtype=jnp.uint64)
      self._ensure_canonical(ct, level)
      ct.polynomial = (
          (ct.polynomial.astype(jnp.uint64) * int_scalar) % moduli_arr
      ).astype(jnp.uint32)

    return self._ensure_canonical(ct, level)

  def slot_to_coef(self, ct):
    """Inverse encoding FFT via diagonal linear transforms."""
    self._require_param_cache()
    if self._s2c_raw_diags is None:
      raise RuntimeError("control_gen() must be called before slot_to_coef()")

    self._ensure_canonical(ct)
    current_ct = ct
    start_level = self._infer_level(current_ct)

    use_hoisted_lt = (
        os.environ.get("BOOTSTRAP_DISABLE_HOISTED_LT") != "1"
        and not self._sparse
        and self._num_slots > 1
        and len(self._s2c_raw_diags) == 1
    )

    if use_hoisted_lt:
      return self._eval_linear_transform_hoisted(
          current_ct, self._s2c_raw_diags[0], start_level
      )

    use_hoisted_collapsed = (
        os.environ.get("BOOTSTRAP_DISABLE_HOISTED_LT") != "1"
        and not self._sparse
        and self._num_slots > 1
        and len(self._s2c_raw_diags) > 1
    )
    if use_hoisted_collapsed:
      for i, diags in enumerate(self._s2c_raw_diags):
        level = start_level - i
        exact_groups = (
            self._s2c_hoisted_groups[i]
            if self._s2c_hoisted_groups is not None
            else None
        )
        current_ct = self._eval_collapsed_linear_transform_hoisted(
            current_ct, diags, level, exact_groups=exact_groups
        )
        if i < len(self._s2c_raw_diags) - 1 and level > 0:
          current_ct = self._he_op_rescale(
              level, level - 1, current_ct, decrement_nsd=True
          )
        self._stage_checkpoint(f"s2c-L{i}", current_ct)
      return self._ensure_canonical(current_ct)

    # Lazily encode S2C diagonals at the actual start level.
    if self._s2c_encoded is None or \
       not self._s2c_encoded or \
       self._s2c_encoded_start_level != start_level:
      self._s2c_encoded = self._pre_encode_diags(
          self._s2c_raw_diags, start_level)
      self._s2c_encoded_start_level = start_level

    for i, enc in enumerate(self._s2c_encoded):
      level = start_level - i
      if level < 0:
        break
      self._ensure_canonical(current_ct, level)
      current_ct = self._eval_linear_transform(current_ct, enc, level)
      self._ensure_canonical(current_ct, level)
      if i < len(self._s2c_encoded) - 1 and level > 0:
        current_ct = self._he_op_rescale(level, level - 1, current_ct,
                                         decrement_nsd=True)
      self._stage_checkpoint(f"s2c-L{i}")

    return self._ensure_canonical(current_ct)

  def _eval_linear_transform_hoisted(self, ct, raw_diags, level):
    """Evaluate a direct linear transform with OpenFHE's hoisted QP BSGS.

    This mirrors ``FHECKKSRNS::EvalLinearTransform``:

    * decompose c1 once and reuse those digits for every baby rotation;
    * multiply and add baby terms in the extended QP basis;
    * ModDown each non-zero giant aggregate once;
    * rotate giant aggregates back into QP with ``addFirst=False``;
    * perform one final ModDown, then restore separately accumulated c0.

    The operation does not rescale or consume a level.  It changes nsd from
    d to d+1 because the encoded diagonals carry one scaling factor.
    """
    import bsgs
    import numpy as np

    self._ensure_canonical(ct, level)
    n = self._num_slots
    layers = max(1, int(math.log2(n)))
    num_rotations = (1 << (layers + 1)) - 1
    b_step = 1 << (
        layers // 2 + 1 + (1 if num_rotations > 7 else 0)
    )
    b_step = min(n, b_step)
    while b_step > 1 and n % b_step:
      b_step //= 2
    g_step = (n + b_step - 1) // b_step

    # Any rotation operator at this level owns the same HYBRID/BConv
    # controls.  Its explicit key/map arguments are varied below.
    sample_index = 1 if n > 1 else 0
    sample_op = self.ctx.he_rot[level, sample_index]
    qp_moduli = list(sample_op.extended_moduli)
    q_moduli = list(self._cache.q_moduli_at_level(level))
    m_qp = len(qp_moduli)
    sf_level = self._openfhe_scale_at_level(level)

    cache_key = (id(raw_diags), int(level), int(b_step))
    pts = self._hoisted_lt_plaintexts.get(cache_key)
    if pts is None:
      zero_diag = np.zeros(n, dtype=np.complex128)
      groups = []
      for j in range(g_step):
        group = []
        for i in range(b_step):
          k = j * b_step + i
          diag = np.asarray(raw_diags.get(k, zero_diag))
          shifted = bsgs.pre_rotate_diagonal(diag, j, b_step)
          group.append(self._encode_diagonal(
              shifted, qp_moduli, m_qp, scale=sf_level
          ))
        groups.append(jnp.stack(group, axis=0))
      pts = jnp.stack(groups, axis=0)
      self._hoisted_lt_plaintexts[cache_key] = pts

    # JIT the shape-stable split primitives.  Rotation keys and maps remain
    # runtime operands, so one executable serves all baby/giant indices.
    decompose = jax.jit(sample_op._decompose_array)
    extend = jax.jit(
        lambda data: sample_op._key_switch_extend_array(
            data, include_first=True
        )
    )
    hoisted_with_first = jax.jit(
        lambda data, ds, ea, eb, cm: sample_op._hoisted_rotate_array(
            data, ds, ea, eb, cm, include_first=True
        )
    )
    hoisted_without_first = jax.jit(
        lambda data, ds, ea, eb, cm: sample_op._hoisted_rotate_array(
            data, ds, ea, eb, cm, include_first=False
        )
    )
    mod_down = jax.jit(sample_op._mod_down_array)
    mul_plain = jax.jit(sample_op._mul_plain_array)
    automorphism = jax.jit(
        lambda data, cm: sample_op._automorphism_array(data, cm)
    )

    qp_moduli_arr = jnp.asarray(qp_moduli, dtype=jnp.uint64).reshape(
        1, 1, 1, 1, m_qp
    )
    q_moduli_arr = jnp.asarray(q_moduli, dtype=jnp.uint64).reshape(
        1, 1, 1, 1, len(q_moduli)
    )

    def _mod_add(a, b, moduli):
      return ((
          a.astype(jnp.uint64) + b.astype(jnp.uint64)
      ) % moduli).astype(jnp.uint32)

    ct_data = ct.polynomial
    digits = decompose(ct_data)
    baby = [extend(ct_data)]
    for i in range(1, b_step):
      ea, eb, cm = self.ctx.he_rot[level, i]._rotation_state()
      baby.append(hoisted_with_first(ct_data, digits, ea, eb, cm))

    first = None
    result_ext = None
    for j in range(g_step):
      inner = None
      for i in range(b_step):
        product = mul_plain(baby[i], pts[j, i])
        inner = product if inner is None else _mod_add(
            inner, product, qp_moduli_arr
        )

      if j == 0:
        first = mod_down(inner[:, 0:1])
        result_ext = inner.at[:, 0:1].set(
            jnp.zeros_like(inner[:, 0:1])
        )
        continue

      inner_q = mod_down(inner)
      giant_index = j * b_step
      giant_op = self.ctx.he_rot[level, giant_index]
      ea, eb, cm = giant_op._rotation_state()
      rotated_first = automorphism(inner_q[:, 0:1], cm)
      first = _mod_add(first, rotated_first, q_moduli_arr)

      inner_digits = decompose(inner_q)
      giant_ext = hoisted_without_first(
          inner_q, inner_digits, ea, eb, cm
      )
      result_ext = _mod_add(result_ext, giant_ext, qp_moduli_arr)

    result_data = mod_down(result_ext)
    result_data = result_data.at[:, 0:1].set(
        _mod_add(result_data[:, 0:1], first, q_moduli_arr)
    )
    result_data = jax.block_until_ready(result_data)

    out = self._new_ciphertext(result_data, level)
    self._set_scale(out, self._get_scale(ct) * sf_level)
    self._set_nsd(out, self._get_nsd(ct) + 1)
    return self._ensure_canonical(out, level)

  def _eval_collapsed_linear_transform_hoisted(
      self, ct, raw_diags, level, exact_groups=None
  ):
    """Evaluate one collapsed FFT level with hoisted QP rotations.

    OpenFHE's collapsed C2S/S2C routines use centered BSGS outer groups per
    level.  Original groups remain separate even when a giant rotation wraps
    to identity, because each group has its own c0 ModDown.  This is the same
    extended-basis accumulation as the direct transform above, but with the
    sparse centered rotation support produced by
    :func:`_decompose_dft_into_levels`.
    """
    import numpy as np

    self._ensure_canonical(ct, level)
    plan = _collapsed_hoisted_plan(raw_diags, self._num_slots)
    if not plan:
      raise ValueError('collapsed linear transform has no diagonals')

    # The ordinary diagonal map merges equal rotations modulo the slot count.
    # Keep that as a diagnostic fallback, but production control generation
    # supplies the original OpenFHE outer groups so cyclic-wrap groups remain
    # separate across their non-additive ApproxModDown boundaries.
    if exact_groups is not None:
      runtime_groups = [
          (
              int(group['giant']),
              [
                  (int(baby), np.asarray(diagonal))
                  for baby, diagonal in group['entries']
              ],
          )
          for group in exact_groups
      ]
    else:
      runtime_groups = []
      for giant, entries in plan:
        runtime_groups.append((
            int(giant),
            [
                (int(baby), np.asarray(raw_diags[int(diag_key)]))
                for diag_key, baby in entries
            ],
        ))

    all_babies = sorted({
        int(baby) for _, entries in runtime_groups for baby, _ in entries
        if int(baby) != 0
    })
    all_giants = [
        int(giant) for giant, _ in runtime_groups if int(giant) != 0
    ]
    sample_index = (all_babies + all_giants)[0]
    sample_op = self.ctx.he_rot[level, sample_index]
    if self._low_mem:
      # Key conversion is offline accessor state. Finish its transfer before
      # dispatching more operators so their copies cannot overlap.
      sample_ea, sample_eb, _ = sample_op._rotation_state()
      jax.block_until_ready((sample_ea, sample_eb))
    qp_moduli = list(sample_op.extended_moduli)
    q_moduli = list(self._cache.q_moduli_at_level(level))
    m_qp = len(qp_moduli)
    sf_level = self._openfhe_scale_at_level(level)

    cache_key = (
        'collapsed', id(raw_diags), id(exact_groups), int(level)
    )
    encoded_groups = self._hoisted_lt_plaintexts.get(cache_key)
    if encoded_groups is None:
      encoded_groups = []
      for giant, entries in runtime_groups:
        if exact_groups is not None:
          # Preserve entry and outer-group boundaries exactly.  In particular,
          # two cyclic-wrap groups may both have giant=0; combining them before
          # ModDown changes the key-switch rounding noise.
          baby_indices = [baby for baby, _ in entries]
          clear_diagonals = []
          for _, diagonal in entries:
            clear = diagonal
            if giant:
              clear = np.roll(clear, int(giant))
            clear_diagonals.append(clear)
        else:
          # The fallback starts from an already merged diagonal map.  Merge
          # any additional baby collisions introduced by its inferred plan.
          clear_by_baby = {}
          for baby, diagonal in entries:
            clear = diagonal
            if giant:
              clear = np.roll(clear, int(giant))
            if baby in clear_by_baby:
              clear_by_baby[baby] = clear_by_baby[baby] + clear
            else:
              clear_by_baby[baby] = clear.copy()
          baby_indices = sorted(clear_by_baby)
          clear_diagonals = [clear_by_baby[baby] for baby in baby_indices]
        plaintexts = jnp.stack([
            self._encode_diagonal(
                clear, qp_moduli, m_qp, scale=sf_level
            )
            for clear in clear_diagonals
        ], axis=0)
        encoded_groups.append((int(giant), baby_indices, plaintexts))
      self._hoisted_lt_plaintexts[cache_key] = encoded_groups
    if self._low_mem:
      # Finish QP encoding/stack transfers before key-switch dispatch.  The
      # plaintext cache itself is modest, but its source buffers otherwise
      # overlap the much larger rotation-key workspaces asynchronously.
      jax.block_until_ready([
          plaintexts for _, _, plaintexts in encoded_groups
      ])
      gc.collect()

    decompose = jax.jit(sample_op._decompose_array)
    extend = jax.jit(
        lambda data: sample_op._key_switch_extend_array(
            data, include_first=True
        )
    )
    hoisted_with_first = jax.jit(
        lambda data, ds, ea, eb, cm: sample_op._hoisted_rotate_array(
            data, ds, ea, eb, cm, include_first=True
        )
    )
    hoisted_without_first = jax.jit(
        lambda data, ds, ea, eb, cm: sample_op._hoisted_rotate_array(
            data, ds, ea, eb, cm, include_first=False
        )
    )
    mod_down = jax.jit(sample_op._mod_down_array)
    mul_plain = jax.jit(sample_op._mul_plain_array)
    automorphism = jax.jit(
        lambda data, cm: sample_op._automorphism_array(data, cm)
    )

    qp_moduli_arr = jnp.asarray(qp_moduli, dtype=jnp.uint64).reshape(
        1, 1, 1, 1, m_qp
    )
    q_moduli_arr = jnp.asarray(q_moduli, dtype=jnp.uint64).reshape(
        1, 1, 1, 1, len(q_moduli)
    )

    def _mod_add(a, b, moduli):
      return ((
          a.astype(jnp.uint64) + b.astype(jnp.uint64)
      ) % moduli).astype(jnp.uint32)

    ct_data = ct.polynomial
    digits = decompose(ct_data)
    baby_zero = extend(ct_data)
    if self._low_mem:
      # CPU dispatch is asynchronous.  Without barriers, every baby rotation
      # keeps its formatted max-level key, sliced operator state, and key-switch
      # workspace alive at once; this drove N=32768 to the cgroup RAM ceiling.
      digits = jax.block_until_ready(digits)
      baby_zero = jax.block_until_ready(baby_zero)
    baby_ext = {0: baby_zero}
    for baby in all_babies:
      rot_op = self.ctx.he_rot[level, baby]
      ea, eb, cm = rot_op._rotation_state()
      rotated = hoisted_with_first(
          ct_data, digits, ea, eb, cm
      )
      if self._low_mem:
        rotated = jax.block_until_ready(rotated)
        jax.block_until_ready((ea, eb))
      baby_ext[baby] = rotated
      if self._low_mem and baby != sample_index:
        # The raw max-level key is retained, so reconstructing these JAX views
        # preserves its HYBRID partition boundaries and exact key material.
        self.ctx.he_rot._instances.pop((level, baby), None)
        self._cache._formatted_rotation_keys.pop(baby, None)
        del rot_op, ea, eb, cm, rotated
        gc.collect()

    first = None
    outer = None
    for group_index, (giant, baby_indices, plaintexts) in enumerate(
        encoded_groups
    ):
      inner = None
      for pt_index, baby in enumerate(baby_indices):
        product = mul_plain(baby_ext[int(baby)], plaintexts[pt_index])
        inner = (
            product
            if inner is None
            else _mod_add(inner, product, qp_moduli_arr)
        )
        if self._low_mem:
          inner = jax.block_until_ready(inner)
          del product

      if group_index == 0:
        if giant != 0:
          raise RuntimeError('collapsed hoisted plan must start at giant 0')
        first = mod_down(inner[:, 0:1])
        outer = inner.at[:, 0:1].set(jnp.zeros_like(inner[:, 0:1]))
        if self._low_mem:
          first = jax.block_until_ready(first)
          outer = jax.block_until_ready(outer)
        continue

      if giant == 0:
        first = _mod_add(
            first, mod_down(inner[:, 0:1]), q_moduli_arr
        )
        inner = inner.at[:, 0:1].set(jnp.zeros_like(inner[:, 0:1]))
        outer = _mod_add(outer, inner, qp_moduli_arr)
        if self._low_mem:
          first = jax.block_until_ready(first)
          outer = jax.block_until_ready(outer)
        continue

      inner_q = mod_down(inner)
      if self._low_mem:
        inner_q = jax.block_until_ready(inner_q)
      giant_op = self.ctx.he_rot[level, giant]
      ea, eb, cm = giant_op._rotation_state()
      new_first = _mod_add(
          first, automorphism(inner_q[:, 0:1], cm), q_moduli_arr
      )
      giant_digits = decompose(inner_q)
      giant_ext = hoisted_without_first(
          inner_q, giant_digits, ea, eb, cm
      )
      new_outer = _mod_add(outer, giant_ext, qp_moduli_arr)
      if self._low_mem:
        new_first = jax.block_until_ready(new_first)
        new_outer = jax.block_until_ready(new_outer)
        jax.block_until_ready((ea, eb))
      first, outer = new_first, new_outer
      if self._low_mem and giant != sample_index:
        self.ctx.he_rot._instances.pop((level, giant), None)
        self._cache._formatted_rotation_keys.pop(giant, None)
        del giant_op, ea, eb, cm, giant_digits, giant_ext
        gc.collect()

    if self._low_mem:
      del baby_ext, baby_zero, digits, inner
      gc.collect()
    result_data = mod_down(outer)
    result_data = result_data.at[:, 0:1].set(
        _mod_add(result_data[:, 0:1], first, q_moduli_arr)
    )
    result_data = jax.block_until_ready(result_data)

    out = self._new_ciphertext(result_data, level)
    self._set_scale(out, self._get_scale(ct) * sf_level)
    self._set_nsd(out, self._get_nsd(ct) + 1)
    return self._ensure_canonical(out, level)

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _new_ciphertext(self, payload, level):
    """Wrap an internal array result in canonical level metadata."""
    cache = self._require_param_cache()
    payload = jnp.asarray(payload)
    num_q = len(cache.q_moduli_at_level(level))
    expected_tail = (*cache.degree_layout, num_q)
    if payload.ndim != 5 or payload.shape[0] != cache.batch or \
       tuple(payload.shape[-3:]) != expected_tail:
      raise ValueError(
          f'ciphertext payload for level {level} must have shape '
          f'({cache.batch}, num_elements, {expected_tail[0]}, '
          f'{expected_tail[1]}, {expected_tail[2]}); got {payload.shape}.'
      )
    shapes = {
        'batch': cache.batch,
        'num_elements': payload.shape[1],
        'degree': cache.degree,
        'num_moduli': num_q,
        'precision': 32,
        'degree_layout': cache.degree_layout,
    }
    parameters = {
        'moduli': cache.q_moduli_at_level(level),
        'ntt_ctx': cache.get_sliced_ntt_q(level),
    }
    return polynomial.Polynomial.from_array(
        payload.astype(jnp.uint32), shapes, parameters
    )

  def _ensure_canonical(self, ct, level=None):
    """Validate the sole public ciphertext representation."""
    if not isinstance(ct, polynomial.Polynomial):
      raise TypeError(
          f'Bootstrap requires Polynomial, got {type(ct).__name__}.'
      )
    ct.validate()
    # Mirror the ckks_ctx boundary: the sole public ciphertext representation
    # is precision=32/uint32. Comparing the payload against the value's own
    # declared dtype cannot catch a self-consistent 64-bit wrapper.
    if (
        ct.precision != 32
        or jnp.dtype(ct.modulus_dtype) != jnp.dtype(jnp.uint32)
    ):
      raise ValueError(
          'Bootstrap requires the canonical precision=32/uint32 Polynomial '
          f'representation; got precision={ct.precision} and modulus dtype '
          f'{jnp.dtype(ct.modulus_dtype)}.'
      )
    if ct.polynomial.dtype != jnp.dtype(ct.modulus_dtype):
      raise ValueError(
          'Bootstrap requires the canonical ciphertext dtype '
          f'{jnp.dtype(ct.modulus_dtype)}, got {ct.polynomial.dtype}.'
      )
    if ct.num_elements != 2:
      raise ValueError(
          'Bootstrap requires a canonical num_elements=2 ciphertext, got '
          f'num_elements={ct.num_elements}.'
      )
    expected = (
        self._cache.batch,
        ct.num_elements,
        *self._cache.degree_layout,
        ct.num_moduli,
    )
    if tuple(ct.shape) != expected:
      raise ValueError(
          f'Bootstrap expected ciphertext shape {expected}, got {ct.shape}.'
      )
    if level is None:
      matching_levels = [
          candidate
          for candidate in range(self._cache.max_level + 1)
          if self._cache.num_q_at_level(candidate) == ct.num_moduli
      ]
      if not matching_levels:
        raise ValueError(
            f'Bootstrap received {ct.num_moduli} moduli, which is not a '
            'level in this parameter cache.'
        )
      level = matching_levels[0]
    expected_moduli = tuple(self._cache.q_moduli_at_level(level))
    if tuple(ct.moduli) != expected_moduli:
      raise ValueError(
          f'Bootstrap expected level {level} moduli {expected_moduli}, got '
          f'{tuple(ct.moduli)}.'
      )
    return ct

  def _he_op_mul(self, level, ct1, ct2):
    """ct-ct multiply with NSD-aware depth management.

    Matches OpenFHE EvalMult + AdjustLevelsAndDepthToOneInPlace.

    Accepts operands at ANY nsd.  nsd=2 operands are pre-rescaled to
    nsd=1 internally.  The pre-rescale IS the level consumption for
    those operands — no separate ensure_nsd1 is needed by the caller.

    Level alignment: operands may be at different levels.  The one with
    more moduli is rescaled or level-reduced to match the other.

    Args:
      level: Desired output level.  Used as a hint; the actual output
             level is determined by the operands after alignment.
      ct1, ct2: Input ciphertexts at any nsd/level.

    Returns:
      Result ciphertext at nsd=2.
    """
    has_tracked_1 = hasattr(ct1, '_ckks_scale') and ct1._ckks_scale is not None
    has_tracked_2 = hasattr(ct2, '_ckks_scale') and ct2._ckks_scale is not None
    sf1_orig = ct1._ckks_scale if has_tracked_1 else None
    sf2_orig = ct2._ckks_scale if has_tracked_2 else None

    cd = self._composite_degree
    nsd1 = self._get_nsd(ct1)
    nsd2 = self._get_nsd(ct2)

    if cd >= 2:
      # --- Step 1: pre-rescale any nsd=2 operand to nsd=1 ----------------
      # Capture original levels for scale tracking.
      lev_orig1 = self._infer_level(ct1)
      lev_orig2 = self._infer_level(ct2)
      if nsd1 == 2:
        ct1 = self._he_op_rescale(lev_orig1, lev_orig1 - 1, ct1)
      if nsd2 == 2:
        ct2 = self._he_op_rescale(lev_orig2, lev_orig2 - 1, ct2)

      # --- Step 2: level-align to the caller's target level ---------------
      # After pre-rescale, both are nsd=1.  Level-reduce any that are
      # above the caller's target level.
      nq_target = len(self._cache.q_moduli_at_level(level))
      if ct1.polynomial.shape[-1] > nq_target:
        ct1, _ = self._level_reduce(ct1, self._infer_level(ct1), level)
      if ct2.polynomial.shape[-1] > nq_target:
        ct2, _ = self._level_reduce(ct2, self._infer_level(ct2), level)

      # --- Step 3: find the matching he_mul for the actual nq ------------
      nq_actual = ct1.polynomial.shape[-1]
      mul_level = max(0, level - 1)
      nq_expected = (self.ctx.he_mul[mul_level].input_num_moduli
                     if mul_level <= self._cache.max_level else 0)
      if nq_actual != nq_expected:
        for L in range(self._cache.max_level, -1, -1):
          if self.ctx.he_mul[L].input_num_moduli == nq_actual:
            mul_level = L
            break

      # OpenFHE-parity: tensor + relinearize WITHOUT the folded rescale, so the
      # product stays at the operands' level with nsd=2 (main's he_mul[L].mul
      # folds in a composite rescale; this engine rescales explicitly and
      # tracks nsd itself).
      op = self.ctx.he_mul[mul_level]
      result = op.relinearize(op.hemul_no_relin(ct1, ct2))
    else:
      op = self.ctx.he_mul[level]
      result = op.relinearize(op.hemul_no_relin(ct1, ct2))

    # --- Post-multiply bookkeeping --------------------------------------
    self._ensure_canonical(result)
    if result.num_elements != 2:
      raise ValueError(
          f'HE multiply must return two elements, got {result.num_elements}.'
      )
    self._set_nsd(result, 2)

    # Trim if he_mul returned more moduli than target
    out_level = self._infer_level(result)
    nq_out = result.polynomial.shape[-1]
    nq_want = len(self._cache.q_moduli_at_level(level))
    if nq_out > nq_want:
      result, _ = self._level_reduce(result, out_level, level)

    # --- Scale tracking --------------------------------------------------
    # Use the ORIGINAL tracked scales captured before any pre-rescale.
    # For nsd=2 operands, divide by the q_pair that was actually dropped
    # during the pre-rescale (top cd primes at the operand's original level).
    if has_tracked_1 and has_tracked_2:
      sf1 = sf1_orig if sf1_orig is not None else ct1._ckks_scale
      sf2 = sf2_orig if sf2_orig is not None else ct2._ckks_scale
      q_towers = self._cache.q_towers

      if cd >= 2:
        was_nsd2 = (nsd1 == 2 or nsd2 == 2)
        if was_nsd2:
          # Each nsd=2 operand was pre-rescaled from its original level
          # (lev_orig1/lev_orig2) to lev_orig-1.  The q_pair for each is
          # the product of the top cd primes at the original level.
          def _qpair(lev_orig):
            nq_src = len(self._cache.q_moduli_at_level(lev_orig))
            qp = 1.0
            for j in range(cd):
              qp *= float(q_towers[nq_src - 1 - j])
            return qp

          sf1_eff = sf1 / _qpair(lev_orig1) if nsd1 == 2 else sf1
          sf2_eff = sf2 / _qpair(lev_orig2) if nsd2 == 2 else sf2
          result_sf = sf1_eff * sf2_eff
        else:
          result_sf = sf1 * sf2
      else:
        input_level = min(level + 1, self._cache.max_level)
        num_q_input = len(self._cache.q_moduli_at_level(input_level))
        q_pair = 1.0
        for i in range(cd):
          q_pair *= float(q_towers[num_q_input - 1 - i])
        result_sf = (sf1 / q_pair) * (sf2 / q_pair)

      if math.isfinite(result_sf) and result_sf > 0:
        self._set_scale(result, result_sf)

    # Low-mem: bound the cumulative operator/compiled-kernel footprint of long
    # multiply chains (Chebyshev PS spans ~13 levels with no stage boundary;
    # per-level HEMul instances + XLA executables accumulated ~40 GB at
    # N=32768 and OOM'd the 100G RAM+swap budget).  Every K muls, drop all
    # cached operator instances + compiled fns (bit-identical; recompile
    # cost only).
    if self._low_mem:
      self._lm_mul_count = getattr(self, "_lm_mul_count", 0) + 1
      if self._lm_mul_count % 6 == 0:
        result.polynomial = jax.block_until_ready(result.polynomial)
        for acc_name in ("he_rot", "he_mul", "ptct_mul", "he_rescale"):
          acc = getattr(self.ctx, acc_name, None)
          inst = getattr(acc, "_instances", None)
          if isinstance(inst, dict):
            inst.clear()
        jax.clear_caches()
        gc.collect()
    return result

  def _he_op_rescale(self, src_level, dst_level, ct, decrement_nsd=False):
    """Rescale a canonical ciphertext between two declared levels.

    After rescale: result.sf = ct.sf / product_of_dropped_primes
    (matching OpenFHE ModReduceInternalInPlace line 189:
     ct.sf /= GetModReduceFactor(sizeQl - 1 - i) for each prime dropped)

    Args:
      decrement_nsd: If True, decrement nsd by 1 (min 1) instead of
        forcing nsd=1.  Used in C2S/S2C rescales to preserve the nsd=2
        flow that matches OpenFHE FIXEDMANUAL.
    """
    has_tracked_scale = hasattr(ct, '_ckks_scale') and ct._ckks_scale is not None
    old_sf = ct._ckks_scale if has_tracked_scale else None

    self._ensure_canonical(ct, src_level)
    result = self.ctx.he_rescale[src_level, dst_level].rescale(ct)
    self._ensure_canonical(result, dst_level)

    if has_tracked_scale and old_sf is not None:
      # Compute the product of dropped primes (the rescale divisor).
      cd = self._composite_degree
      num_drops = (src_level - dst_level) * cd
      num_q_src = len(self._cache.q_moduli_at_level(src_level))
      q_towers = self._cache.q_towers
      divisor = 1.0
      for i in range(num_drops):
        divisor *= float(q_towers[num_q_src - 1 - i])
      new_sf = old_sf / divisor
      if math.isfinite(new_sf) and new_sf > 0:
        self._set_scale(result, new_sf)
    if decrement_nsd:
      old_nsd = self._get_nsd(ct)
      self._set_nsd(result, max(1, old_nsd - 1))
    else:
      self._set_nsd(result, 1)
    return result

  def _match_levels(self, ct, ct_level, target_level):
    """Rescale ct down to target_level if ct_level > target_level."""
    if ct_level > target_level:
      return self._he_op_rescale(ct_level, target_level, ct), target_level
    return ct, ct_level

  def _level_reduce(self, ct, ct_level, target_level):
    """Drop moduli to reach target_level WITHOUT rescaling (scale preserved).

    Unlike _match_levels (which performs ModReduce = drop moduli + divide
    coefficients by the dropped modulus), this simply truncates the CRT
    representation. The scale is unchanged.

    Equivalent in value to OpenFHE's LevelReduceInPlace, but returns a new
    wrapper so aliases of the input keep valid level metadata.
    """
    self._require_param_cache()
    if ct_level <= target_level:
      return ct, ct_level
    target_num_q = len(self._cache.q_moduli_at_level(target_level))
    self._ensure_canonical(ct, ct_level)
    reduced_payload = ct.polynomial[..., :target_num_q]
    target_moduli = self._cache.q_moduli_at_level(target_level)
    reduced = ct._clone_with_payload(
        reduced_payload,
        moduli=target_moduli,
        ntt_ctx=self._cache.get_sliced_ntt_q(target_level),
    )
    return reduced, target_level

  def _adjust_ct_to_match_scale(self, ct_to_adjust, ct_to_adjust_level,
                                 target_ct):
    """Adjust ct_to_adjust's scale to match target_ct's tracked scale.

    Port of OpenFHE's AdjustLevelsAndDepthInPlace for the specific case
    where ct_to_adjust (nsd=2, at higher CROSS level = more primes) needs
    to be brought to match target_ct (nsd=2, at lower CROSS level = fewer
    primes).

    OpenFHE mechanism (ckksrns-leveledshe.cpp:670-683):
      correction = scf_target / scf_source * q_pair / SF_recursive
      EvalMultCoreInPlace(ct, correction)   — encodes at SF_recursive
      ModReduceInternalInPlace(ct, cd)      — drops cd primes
      LevelReduceInternalInPlace to match

    Net effect: polynomial *= (scf_target / scf_source), changing scale
    from scf_source to scf_target. The float64 precision (~52 bits) is
    far better than the ~20-bit mismatch from raw _mod_sub.

    IMPORTANT: EvalMultCoreInPlace encodes at the RECURSIVE SF (not the
    tracked scale). This is different from _ptct_mul_scalar which encodes
    at the tracked scale. We implement this with a direct encoding.

    Cost: 1 multiplicative level from ct_to_adjust's chain.

    Returns: (adjusted_ct, adjusted_level) where tracked scale ≈ target's.
    """
    cd = self._composite_degree
    scf_target = target_ct._ckks_scale if hasattr(target_ct, '_ckks_scale') and target_ct._ckks_scale else None
    scf_source = ct_to_adjust._ckks_scale if hasattr(ct_to_adjust, '_ckks_scale') and ct_to_adjust._ckks_scale else None

    if scf_target is None or scf_source is None or scf_source == 0:
      target_level = self._infer_level(target_ct)
      ct_to_adjust, _ = self._level_reduce(
          ct_to_adjust, ct_to_adjust_level, target_level)
      return ct_to_adjust, target_level

    # Recursive SF at ct_to_adjust's level (matching OpenFHE's GetScalingFactorReal).
    # OpenFHE computes SF recursively: sf[k] = sf[k-cd]^2 / product(cd primes).
    # Using the same recursive SF ensures consistent encoding with OpenFHE's
    # scale tracking throughout the bootstrap pipeline.
    sf_rec = self._openfhe_scale_at_level(ct_to_adjust_level)

    # Product of top cd primes at ct_to_adjust's level (rescale divisor)
    nq_adj = len(self._cache.q_moduli_at_level(ct_to_adjust_level))
    q_towers = self._cache.q_towers
    q_pair = 1.0
    for j in range(cd):
      q_pair *= float(q_towers[nq_adj - 1 - j])

    # Correction: target_scale / source_scale * q_pair / SF_recursive
    correction = (scf_target / scf_source) * (q_pair / sf_rec)

    # Step 1: EvalMultCoreInPlace — direct integer CRT multiplication.
    # Matches OpenFHE's GetElementForEvalMult: encode correction * SF as
    # a single integer, reduce mod each prime, multiply all ciphertext
    # coefficients by that constant. NO FFT/CKKS encoding — avoids the
    # float64 precision loss from the FFT butterfly chain.
    actual_nq = ct_to_adjust.polynomial.shape[-1]
    moduli = self._cache.q_towers[:actual_nq]
    num_q = actual_nq
    self._ensure_canonical(ct_to_adjust)

    # Encode correction*sf_rec as integer CRT residues (same OpenFHE
    # GetElementForEvalMult overflow split used on the Chebyshev path).
    factors = self._int_crt_factors(correction, sf_rec, moduli)

    # Multiply every ciphertext coefficient by factors[i] (mod q_i)
    factors_arr = jnp.array(factors, dtype=jnp.uint64).reshape(1, 1, 1, 1, num_q)
    adjusted = copy.copy(ct_to_adjust)
    adjusted.polynomial = (
        (ct_to_adjust.polynomial.astype(jnp.uint64) * factors_arr) % \
        jnp.array(moduli, dtype=jnp.uint64).reshape(1, 1, 1, 1, num_q)
    ).astype(jnp.uint32)
    self._ensure_canonical(adjusted)
    # Tracked scale update: source * SF (matching OpenFHE)
    self._set_scale(adjusted, scf_source * sf_rec)
    self._set_nsd(adjusted, self._get_nsd(ct_to_adjust) + 1)

    # Step 2: ModReduce (drops cd primes)
    actual_level = self._infer_level(adjusted)
    adjusted = self._he_op_rescale(actual_level, actual_level - 1, adjusted)
    adj_level = actual_level - 1

    # Step 3: LevelReduce to match target level
    target_level = self._infer_level(target_ct)
    if adj_level > target_level:
      adjusted, adj_level = self._level_reduce(adjusted, adj_level, target_level)

    # After EvalMultCore + ModReduce, the polynomial carries the target's
    # nsd=2 scale (S1_nsd2) regardless of the actual NSD metadata.
    # Set tracked scale to match target's. Do NOT override NSD — leave it
    # at the actual post-ModReduce value (nsd=1). The raw _ct_sub/_ct_add
    # in _ct_op_aligned will use max(nsd_a, nsd_b) for the result, which
    # is correct because the polynomial scales match (the adjustment made
    # them equal). Lying about NSD would cause downstream _he_op_mul to
    # incorrectly pre-rescale, destroying the bootstrap.
    self._set_scale(adjusted, scf_target)
    # NSD left as-is from _he_op_rescale (= 1 after ModReduce)
    return adjusted, adj_level

  def _eval_linear_transform(self, ct, encoded_diags: dict, level: int):
    """Evaluate a dense diagonal linear transform using pre-encoded diags."""
    cache = self._cache
    num_q = len(cache.q_moduli_at_level(level))
    self._ensure_canonical(ct, level)

    accum = None
    for k, pt_ntt in encoded_diags.items():
      k = int(k)
      # A tower-count mismatch means the diagonal was pre-encoded for a
      # different level than the one being evaluated; silently skipping it
      # would drop a term and corrupt the linear transform, so fail loudly.
      if pt_ntt.shape[-1] != num_q:
        raise RuntimeError(
            f"linear-transform diagonal k={k} pre-encoded with "
            f"{pt_ntt.shape[-1]} towers but level {level} expects {num_q}; "
            f"diagonal encoding is out of sync with the evaluation level.")

      if k == 0:
        rotated = copy.copy(ct)
        rotated.polynomial = ct.polynomial
      else:
        rot_input = copy.copy(ct)
        rot_input.polynomial = ct.polynomial
        rotated = self.ctx.he_rot[level, k].rotate(rot_input)
      self._ensure_canonical(rotated, level)

      op = self.ctx.ptct_mul[level]
      out = op._mul_encoded(rotated, pt_ntt)
      self._ensure_canonical(out, level)

      if accum is None:
        accum = out
      else:
        # Must use uint64 for addition to avoid uint32 overflow when
        # accumulating many ptct_mul results (each up to q < 2^31,
        # sum of 16+ terms exceeds 2^32).
        moduli = cache.q_moduli_at_level(level)
        moduli_arr = jnp.array(moduli, dtype=jnp.uint64)
        accum.polynomial = (
            (accum.polynomial.astype(jnp.uint64)
             + out.polynomial.astype(jnp.uint64)) % moduli_arr
        ).astype(jnp.uint32)

      # Low-mem hygiene: serialize the per-diagonal rotate+ptct_mul+accumulate
      # and free each diagonal's key-switch buffers as we go, so async dispatch
      # doesn't keep them all live at once.  Secondary to the dominant fix --
      # evicting the cached per-(level, rot_index) operators in
      # _stage_checkpoint (those degree-sized NTT tables, not these transient
      # buffers, are what accumulate ~21 GB/level and OOM at N=16384).
      # Bit-identical; costs async pipelining, only on the memory-pressure path.
      if self._low_mem:
        accum.polynomial = jax.block_until_ready(accum.polynomial)
        del out, rotated
        # Evict the just-used rotation operator immediately: each rot index
        # is used exactly ONCE per linear-transform level, but its
        # HERotAtLevel instance holds degree-sized NTT tables + regenerated
        # per-level keys.  Waiting for the stage-boundary eviction lets 30-60
        # of them accumulate WITHIN one C2S level (~15-20 GB at N=32768,
        # which OOM'd under the 100G RAM+swap budget).
        if k != 0:
          inst = getattr(self.ctx.he_rot, "_instances", None)
          if isinstance(inst, dict):
            inst.pop((level, k), None)
        gc.collect()

    if accum is not None:
      # Propagate scale: ptct_mul of ct(S) × diag(S_encode) → result(S × S_encode).
      has_tracked = hasattr(ct, '_ckks_scale') and ct._ckks_scale is not None
      if has_tracked and math.isfinite(ct._ckks_scale):
        encode_sf = self._openfhe_scale_at_level(level)
        new_sf = ct._ckks_scale * encode_sf
        if math.isfinite(new_sf):
          self._set_scale(accum, new_sf)
        else:
          self._copy_scale(accum, ct)
      # ptct_mul increments nsd by 1: nsd=1→2, nsd=2→3.
      self._set_nsd(accum, self._get_nsd(ct) + 1)
      return accum
    return ct

  def _eval_chebyshev_direct(self, ct, level: int):
    """Evaluate Chebyshev series via direct three-term recurrence.

    Computes: result = sum_{j=0}^{degree} c_j * T_j(x)
    where T_0=1, T_1=x, T_{n+1} = 2*x*T_n - T_{n-1}.

    The input ct is expected at nsd=2 (from eval_approx_mod_reduction's
    ptct_mul(1.0) call). All operations handle NSD consistently:
    - Before ptct_mul: ensure nsd=1 (matching OpenFHE's AdjustDepth)
    - Before _he_op_mul: nsd=2 inputs are handled automatically
    - add_scalar uses the ct's actual NSD

    Returns (result_ct, result_level).
    """
    coeffs = self._cheby_coeffs
    degree = len(coeffs) - 1

    # Ensure x is at nsd=1 for ptct_mul
    self._ensure_canonical(ct, level)
    x_ct = copy.copy(ct)
    x_ct.polynomial = ct.polynomial.copy()
    self._copy_nsd(x_ct, ct)
    x_ct, x_level = self._ensure_nsd1(x_ct, level)

    # Start with c_1 * x (if c_1 != 0)
    c1 = float(coeffs[1]) if degree >= 1 else 0.0
    x_for_ptct = copy.copy(x_ct)
    x_for_ptct.polynomial = x_ct.polynomial.copy()
    result = self._ptct_mul_scalar(x_for_ptct, c1, x_level)
    result_level = x_level
    # Add 0.5 * c_0
    c0_half = 0.5 * float(coeffs[0])
    if abs(c0_half) > 1e-30:
      self._add_scalar(result, c0_half, result_level,
                       noise_scale_deg=self._get_nsd(result))

    if degree <= 1:
      return result, result_level

    # Three-term recurrence using nsd=2 convention.
    # The _he_op_mul calls produce nsd=2 outputs, which is correct for
    # the subtraction (both T_next and T_prev are nsd=2 after their
    # respective multiplies). For ptct_mul of c_n * T_n, we ensure nsd=1.
    T_prev_ct = None  # T_0 = 1 (not stored as ciphertext)
    T_prev_level = x_level
    T_cur_ct = copy.copy(x_ct)
    T_cur_ct.polynomial = x_ct.polynomial.copy()
    T_cur_level = x_level

    for n in range(2, degree + 1):
      # T_next = 2*x*T_cur - T_prev
      # _he_op_mul handles NSD internally: nsd=2 inputs get pre-rescaled.
      x_copy = copy.copy(x_ct)
      x_copy.polynomial = x_ct.polynomial.copy()
      self._copy_nsd(x_copy, x_ct)
      T_cur_copy = copy.copy(T_cur_ct)
      T_cur_copy.polynomial = T_cur_ct.polynomial.copy()
      self._copy_nsd(T_cur_copy, T_cur_ct)

      # Match levels
      tgt = min(x_level, T_cur_level)
      if x_level > tgt:
        x_copy, _ = self._level_reduce(x_copy, x_level, tgt)
      if T_cur_level > tgt:
        T_cur_copy, _ = self._level_reduce(T_cur_copy, T_cur_level, tgt)

      # _he_op_mul: for nsd=1 inputs at level L, output is at L with nsd=2.
      # For nsd=2 inputs at level L, pre-rescale to L-1, output at L-1 with nsd=2.
      mul_out = tgt  # output level for nsd=1 inputs
      if self._get_nsd(x_copy) == 2 or self._get_nsd(T_cur_copy) == 2:
        mul_out = tgt - 1  # nsd=2 inputs consume a level via pre-rescale
      if mul_out < 0:
        break

      T_next = self._he_op_mul(mul_out, x_copy, T_cur_copy)
      T_next_level = mul_out
      # 2 * (x * T_cur) via ct + ct (free, preserves nsd=2)
      T_next = self._ct_add(T_next, T_next, T_next_level)

      # - T_prev
      if T_prev_ct is not None:
        # T_prev may be nsd=2 from previous _he_op_mul.
        # Both T_next and T_prev should be at nsd=2 for consistent subtraction.
        T_prev_copy = copy.copy(T_prev_ct)
        T_prev_copy.polynomial = T_prev_ct.polynomial.copy()
        self._copy_nsd(T_prev_copy, T_prev_ct)
        # Ensure same NSD for subtraction
        if self._get_nsd(T_prev_copy) != self._get_nsd(T_next):
          T_prev_copy, T_prev_copy_level = self._ensure_nsd1(
              T_prev_copy, T_prev_level)
          T_next, T_next_level = self._ensure_nsd1(T_next, T_next_level)
        else:
          T_prev_copy_level = T_prev_level
        T_next, T_next_level = self._ct_sub_aligned(
            T_next, T_next_level, T_prev_copy, T_prev_copy_level)
      else:
        # T_prev was T_0 = 1, subtract 1 as scalar
        self._add_scalar(T_next, -1.0, T_next_level,
                         noise_scale_deg=self._get_nsd(T_next))

      # Accumulate c_n * T_n: ensure nsd=1 before ptct_mul
      c_n = float(coeffs[n])
      if abs(c_n) > 1e-30:
        T_for_ptct = copy.copy(T_next)
        T_for_ptct.polynomial = T_next.polynomial.copy()
        self._copy_nsd(T_for_ptct, T_next)
        T_for_ptct, T_for_ptct_level = self._ensure_nsd1(
            T_for_ptct, T_next_level)
        term = self._ptct_mul_scalar(T_for_ptct, c_n, T_for_ptct_level)
        # Rescale term (nsd=2) to nsd=1 before adding to result
        term, term_level = self._ensure_nsd1(term, T_for_ptct_level)
        result, result_level = self._ensure_nsd1(result, result_level)
        result, result_level = self._ct_add_aligned(
            result, result_level, term, term_level)

      # Shift: T_prev <- T_cur, T_cur <- T_next
      T_prev_ct = T_cur_ct
      T_prev_level = T_cur_level
      T_cur_ct = T_next
      T_cur_level = T_next_level

    return result, result_level

  def _eval_chebyshev_ps(self, ct, level: int):
    """Evaluate Chebyshev series via Paterson-Stockmeyer.

    Full port of OpenFHE's internalEvalChebyPolysPS + InnerEvalChebyshevPS
    (ckksrns-advancedshe.cpp). Uses Chebyshev long division for proper
    block recombination instead of naive T_m^b powers.

    Depth: ceil(log2(k)) + m, where (k, m) = ComputeDegreesPS(degree).

    Returns (result_ct, result_level).
    """
    coeffs_raw = [float(c) for c in self._cheby_coeffs]
    degree = _ps_degree(coeffs_raw)
    coeffs_raw = coeffs_raw[:degree + 1]

    if degree == 0:
      return self._ptct_mul_scalar(copy.copy(ct), float(coeffs_raw[0]) * 0.5,
                                   level), level

    if degree < 5:
      # Fall back to direct evaluation for very small degrees.
      return self._eval_chebyshev_direct(ct, level)

    k, m = compute_degrees_ps(degree)
    k2m2k = k * (1 << (m - 1)) - k  # k * 2^{m-1} - k

    # ----------------------------------------------------------------
    # Phase 1: Build Chebyshev basis T[0..k-1] via binary tree
    # T[i] = T_{i+1}(x) for i=0..k-1
    # ----------------------------------------------------------------
    T, T_level = self._build_chebyshev_baby_steps(ct, level, k)
    # Barrier on the whole baby-step basis (k degree-N ciphertexts built via a
    # multiply tree) so their construction intermediates are freed before T2.
    self._stage_checkpoint("cheby-babies", [t for t in T if t is not None])

    # Mode C alignment: adjust baby steps above baby_level to T[k-1]'s
    # actual polynomial scale via _adjust_ct_to_match_scale (CRT multiply
    # + ModReduce).  The ~0.003% cross-level scale mismatch must be
    # corrected — absorbing it loses ~24 bits of PS precision.
    baby_level = T_level[k - 1]
    for i in range(k - 1):
      if T[i] is None:
        continue
      if T_level[i] > baby_level:
        # Guard: ensure same nsd before _adjust_ct_to_match_scale.
        # Cross-NSD (nsd=2 source vs nsd=1 target) makes the CRT
        # correction factor round to 1, losing all precision.
        if self._get_nsd(T[i]) != self._get_nsd(T[k - 1]):
          T[i], T_level[i] = self._ensure_nsd1(T[i], T_level[i])
        T[i], T_level[i] = self._adjust_ct_to_match_scale(
            T[i], T_level[i], T[k - 1])
        self._copy_nsd(T[i], T[k - 1])
        if T_level[i] > baby_level:
          T[i], T_level[i] = self._level_reduce(T[i], T_level[i], baby_level)
      self._set_scale(T[i], self._get_scale(T[k - 1]))

    # ----------------------------------------------------------------
    # Phase 2: Build T2[0..m-1] via squaring and T2km1 via accumulation.
    # T2[0] = T_k, T2[i] = T_{2^i * k} for i=1..m-1
    # T2km1 = T_{k*(2^m - 1)}
    # ----------------------------------------------------------------
    T2 = [None] * m
    T2_level = [0] * m
    T2[0] = copy.copy(T[k - 1])
    T2[0].polynomial = T[k - 1].polynomial.copy()
    T2_level[0] = baby_level

    T2km1 = copy.copy(T[k - 1])
    T2km1.polynomial = T[k - 1].polynomial.copy()
    T2km1_level = baby_level

    for i in range(1, m):
      # T2[i] = 2 * T2[i-1]^2 - 1  (= T_{2^i * k}(x))
      prev_lev = T2_level[i - 1]
      if prev_lev < 1:
        break
      # Copy to avoid corrupting T2[i-1]
      prev_a = copy.copy(T2[i - 1])
      prev_a.polynomial = T2[i - 1].polynomial.copy()
      prev_b = copy.copy(T2[i - 1])
      prev_b.polynomial = T2[i - 1].polynomial.copy()
      # Pass prev_lev (not prev_lev-1) so _he_op_mul does not level-
      # reduce nsd=1 inputs.  The explicit ModReduce below is the
      # single level consumed, matching OpenFHE EvalSquare + ModReduce.
      sq = self._he_op_mul(prev_lev, prev_a, prev_b)
      sq = self._ct_add(sq, sq, prev_lev)  # 2x (free, nsd=2)
      sq = self._he_op_rescale(prev_lev, prev_lev - 1, sq)
      self._stage_checkpoint(f"cheby-T2-{i}", sq)
      self._add_scalar(sq, -1.0, prev_lev - 1,
                        noise_scale_deg=self._get_nsd(sq))
      T2[i] = sq
      T2_level[i] = prev_lev - 1

      # T2km1 = 2 * T2km1 * T2[i] - T2[0]  (= T_{k*(2^{i+1} - 1)}(x))
      if T2km1_level < 1:
        break
      tgt_lev = min(T2km1_level, T2_level[i])
      # Copy T2km1 for level matching
      t2km1_copy = copy.copy(T2km1)
      t2km1_copy.polynomial = T2km1.polynomial.copy()
      if T2km1_level > tgt_lev:
        t2km1_copy, _ = self._level_reduce(t2km1_copy, T2km1_level, tgt_lev)
      t2i_copy = copy.copy(T2[i])
      t2i_copy.polynomial = T2[i].polynomial.copy()
      if T2_level[i] > tgt_lev:
        t2i_copy, _ = self._level_reduce(t2i_copy, T2_level[i], tgt_lev)
      # Pass tgt_lev (not tgt_lev-1) so _he_op_mul does not level-
      # reduce nsd=1 inputs.  The explicit ModReduce below is the
      # single level consumed, matching OpenFHE EvalMult + ModReduce.
      prod = self._he_op_mul(tgt_lev, t2km1_copy, t2i_copy)
      prod = self._ct_add(prod, prod, tgt_lev)  # 2x (free, nsd=2)
      # Explicit ModReduce matching OpenFHE line 836: nsd=2→1
      prod = self._he_op_rescale(tgt_lev, tgt_lev - 1, prod)
      # Subtract T2[0] = T_k via scale-aligned subtraction.
      t0_copy = copy.copy(T2[0])
      t0_copy.polynomial = T2[0].polynomial.copy()
      self._copy_nsd(t0_copy, T2[0])
      self._copy_scale(t0_copy, T2[0])
      prod, prod_lev = self._ct_sub_aligned(
          prod, tgt_lev - 1, t0_copy, T2_level[0])
      T2km1 = prod
      T2km1_level = prod_lev

    # ----------------------------------------------------------------
    # Phase 3: Extend coefficients by adding T_{k*(2^m - 1)}
    # ----------------------------------------------------------------
    f2 = list(coeffs_raw)
    target_len = 2 * k2m2k + k + 1
    while len(f2) < target_len:
      f2.append(0.0)
    f2[-1] = 1.0  # Add coefficient for T_{k*(2^m - 1)}

    # ----------------------------------------------------------------
    # Phase 4: Recursive evaluation
    # ----------------------------------------------------------------
    result, result_level = self._inner_eval_chebyshev_ps(
        f2, k, m, T, T_level, T2, T2_level, baby_level)

    # Combine 4: result - T2km1
    # Normalize nsd only if they disagree.  _adjust_ct_to_match_scale
    # cannot handle cross-nsd (nsd=2→nsd=1 target makes the integer
    # correction round to 1), so we still need _ensure_nsd1 for that case.
    if self._get_nsd(result) != self._get_nsd(T2km1):
      result, result_level = self._ensure_nsd1(result, self._infer_level(result))
      T2km1, T2km1_level = self._ensure_nsd1(T2km1, self._infer_level(T2km1))
    else:
      result_level = self._infer_level(result)
      T2km1_level = self._infer_level(T2km1)
    res_actual = self._infer_level(result)
    km1_actual = self._infer_level(T2km1)

    if km1_actual > res_actual:
      T2km1, T2km1_level = self._adjust_ct_to_match_scale(
          T2km1, km1_actual, result)
      self._copy_nsd(T2km1, result)
      if self._infer_level(T2km1) > res_actual:
        T2km1, T2km1_level = self._level_reduce(
            T2km1, self._infer_level(T2km1), res_actual)
    elif res_actual > km1_actual:
      result, result_level = self._adjust_ct_to_match_scale(
          result, res_actual, T2km1)
      self._copy_nsd(result, T2km1)
      if self._infer_level(result) > km1_actual:
        result, result_level = self._level_reduce(
            result, self._infer_level(result), km1_actual)
    else:
      # Same level, same nsd: use level-free scale correction matching
      # OpenFHE AdjustLevelsAndDepthInPlace for same-level operands.
      # CRT multiply T2km1 by (sf_result / sf_T2km1 * SF) to correct
      # scale, incrementing nsd.  Then CRT multiply result by SF to
      # match nsd.  Both end at nsd+1, same level, matched scales.
      # NO _he_op_rescale — 0 levels consumed.
      sf_r = self._get_scale(result)
      sf_k = self._get_scale(T2km1)
      if sf_r and sf_k and sf_r > 0 and sf_k > 0:
        if abs(sf_r / sf_k - 1.0) > 1e-12:
          cd = self._composite_degree
          sf_rec = self._openfhe_scale_at_level(km1_actual)
          actual_nq = T2km1.polynomial.shape[-1]
          moduli = self._cache.q_towers[:actual_nq]
          # CRT multiply T2km1 by round(sf_r/sf_k * sf_rec) mod each q_i
          correction = sf_r / sf_k
          factors_km1 = self._int_crt_factors(correction, sf_rec, moduli)
          f_arr = jnp.array(factors_km1, dtype=jnp.uint64).reshape(1,1,1,1,actual_nq)
          m_arr = jnp.array(moduli, dtype=jnp.uint64).reshape(1,1,1,1,actual_nq)
          self._ensure_canonical(T2km1)
          T2km1.polynomial = ((T2km1.polynomial.astype(jnp.uint64) * f_arr) % m_arr).astype(jnp.uint32)
          self._set_scale(T2km1, sf_r * sf_rec)
          self._set_nsd(T2km1, self._get_nsd(T2km1) + 1)
          # CRT multiply result by round(sf_rec) to match nsd
          actual_nq_r = result.polynomial.shape[-1]
          moduli_r = self._cache.q_towers[:actual_nq_r]
          factors_r = self._int_crt_factors(1.0, sf_rec, moduli_r)
          f_arr_r = jnp.array(factors_r, dtype=jnp.uint64).reshape(1,1,1,1,actual_nq_r)
          m_arr_r = jnp.array(moduli_r, dtype=jnp.uint64).reshape(1,1,1,1,actual_nq_r)
          self._ensure_canonical(result)
          result.polynomial = ((result.polynomial.astype(jnp.uint64) * f_arr_r) % m_arr_r).astype(jnp.uint32)
          self._set_scale(result, sf_r * sf_rec)
          self._set_nsd(result, self._get_nsd(result) + 1)
          result_level = res_actual
          T2km1_level = km1_actual

    out_lev = min(self._infer_level(result), self._infer_level(T2km1))
    result, _ = self._level_reduce(
        result, self._infer_level(result), out_lev
    )
    T2km1, _ = self._level_reduce(
        T2km1, self._infer_level(T2km1), out_lev
    )
    self._ensure_canonical(result, out_lev)
    self._ensure_canonical(T2km1, out_lev)
    result = self._ct_sub(result, T2km1, out_lev)
    result_level = out_lev

    return result, result_level

  def _build_chebyshev_baby_steps(self, ct, level, k):
    """Build T[0..k-1] = T_1(x), ..., T_k(x) using OpenFHE's binary tree.

    For even i:  T_i = 2*T_{i/2}^2 - 1
    For odd i:   T_i = 2*T_{(i-1)/2}*T_{(i+1)/2} - T_1

    Returns (T_list, T_levels) where T_list[i] is a ciphertext for T_{i+1}(x).
    """
    T = [None] * k
    T_level = [0] * k

    # T[0] = T_1(x) = x
    T[0] = copy.copy(ct)
    T[0].polynomial = ct.polynomial.copy()
    T_level[0] = self._infer_level(ct)

    for i in range(2, k + 1):
      if i & 1:  # odd: T_i = 2*T_a*T_b - T_1
        a_idx = i // 2 - 1
        b_idx = i // 2
        a_lev, b_lev = T_level[a_idx], T_level[b_idx]
        a_ct_m = copy.copy(T[a_idx])
        a_ct_m.polynomial = T[a_idx].polynomial.copy()
        self._copy_scale(a_ct_m, T[a_idx]); self._copy_nsd(a_ct_m, T[a_idx])
        b_ct_m = copy.copy(T[b_idx])
        b_ct_m.polynomial = T[b_idx].polynomial.copy()
        self._copy_scale(b_ct_m, T[b_idx]); self._copy_nsd(b_ct_m, T[b_idx])
        # Mode C: only when operands differ in level (different chains).
        # _adjust_ct_to_match_scale requires same nsd, so ensure_nsd1
        # only the operands that need Mode C alignment.
        # When levels match, go straight to _he_op_mul (handles nsd=2).
        if a_lev != b_lev:
          # Different levels → need Mode C.  Ensure nsd=1 for
          # cross-chain scale adjustment to work correctly.
          a_ct_m, a_lev = self._ensure_nsd1(a_ct_m, a_lev)
          b_ct_m, b_lev = self._ensure_nsd1(b_ct_m, b_lev)
          if a_lev > b_lev:
            a_ct_m, a_lev = self._adjust_ct_to_match_scale(
                a_ct_m, a_lev, b_ct_m)
            self._copy_nsd(a_ct_m, b_ct_m)
            if a_lev > b_lev:
              a_ct_m, a_lev = self._level_reduce(a_ct_m, a_lev, b_lev)
          else:
            b_ct_m, b_lev = self._adjust_ct_to_match_scale(
                b_ct_m, b_lev, a_ct_m)
            self._copy_nsd(b_ct_m, a_ct_m)
            if b_lev > a_lev:
              b_ct_m, b_lev = self._level_reduce(b_ct_m, b_lev, a_lev)
        tgt = min(a_lev, b_lev)
        nsd_max = max(self._get_nsd(a_ct_m), self._get_nsd(b_ct_m))
        mul_lev = tgt - 1 if nsd_max == 2 else tgt
        if mul_lev < 1:
          break
        prod = self._he_op_mul(mul_lev, a_ct_m, b_ct_m)
        prod = self._ct_add(prod, prod, mul_lev)  # 2x (free, nsd=2)
        # ModReduce before subtracting T[0]; odd Chebyshev recurrence is
        # T_i = 2*T_a*T_b - T_1 (no scalar -1 term).
        prod = self._he_op_rescale(mul_lev, mul_lev - 1, prod)
        mul_lev = mul_lev - 1
        # Subtract T[0] — ensure T[0] copy is nsd=1 (matches prod)
        t1_copy = copy.copy(T[0])
        t1_copy.polynomial = T[0].polynomial.copy()
        self._copy_nsd(t1_copy, T[0])
        self._copy_scale(t1_copy, T[0])
        t1_copy, t1_lev = self._ensure_nsd1(t1_copy, T_level[0])
        prod, mul_lev = self._ct_sub_aligned(
            prod, mul_lev, t1_copy, t1_lev)
        T[i - 1] = prod
        T_level[i - 1] = mul_lev
      else:  # even: T_i = 2*T_{i/2}^2 - 1
        half_idx = i // 2 - 1
        half_ct = copy.copy(T[half_idx])
        half_ct.polynomial = T[half_idx].polynomial.copy()
        self._copy_nsd(half_ct, T[half_idx])
        self._copy_scale(half_ct, T[half_idx])
        # _he_op_mul handles nsd=2 pre-rescale internally.
        half_lev = T_level[half_idx]
        half_nsd = self._get_nsd(half_ct)
        mul_lev = half_lev - 1 if half_nsd == 2 else half_lev
        if mul_lev < 1:
          break
        sq = self._he_op_mul(mul_lev, half_ct, half_ct)
        sq = self._ct_add(sq, sq, mul_lev)  # 2x (free, nsd=2)
        # Rescale nsd=2→1 then add_scalar at nsd=1 (per-level SF).
        # Baby steps must be nsd=1 for wsum tracked-scale encoding.
        sq = self._he_op_rescale(mul_lev, mul_lev - 1, sq)
        mul_lev = mul_lev - 1
        self._add_scalar(sq, -1.0, mul_lev,
                         noise_scale_deg=self._get_nsd(sq))
        T[i - 1] = sq
        T_level[i - 1] = mul_lev
      # PER-BABY-STEP barrier: force this step's he_op_mul key-switch
      # intermediates to compute+free before building the next.  Without it,
      # the k baby steps' async buffers accumulate (the N=65536 ApproxMod OOM
      # site -- it died here, BEFORE the phase-level 'cheby-babies' checkpoint).
      if self._low_mem and T[i - 1] is not None:
        self._stage_checkpoint(f"cheby-baby-{i}", T[i - 1])

    return T, T_level

  def _eval_partial_linear_wsum(self, T, T_level, coeffs, n, target_level):
    """sum_{i=0}^{n-1} coeffs[i+1] * T[i].

    Matches OpenFHE's EvalPartialLinearWSum (ckksrns-advancedshe.cpp L143).
    With OpenFHE-consistent recursive SF, all baby steps at the same level
    have matching polynomial scales (the recursive SF chain is self-consistent).
    Direct accumulation is exact.

    Returns (result_ct, result_level).
    """
    first = copy.copy(T[0])
    first.polynomial = T[0].polynomial.copy()
    self._copy_nsd(first, T[0])
    self._copy_scale(first, T[0])
    first, first_level = self._level_reduce(first, T_level[0], target_level)
    first, first_level = self._ensure_nsd1(first, first_level)
    result = self._ptct_mul_scalar(first, float(coeffs[1]), first_level)

    for i in range(1, n):
      if i >= len(T) or T[i] is None:
        break
      if abs(coeffs[i + 1]) < _PS_DELTA:
        continue
      ti = copy.copy(T[i])
      ti.polynomial = T[i].polynomial.copy()
      self._copy_nsd(ti, T[i])
      self._copy_scale(ti, T[i])
      ti, ti_level = self._level_reduce(ti, T_level[i], target_level)
      ti, ti_level = self._ensure_nsd1(ti, ti_level)
      if ti_level > first_level:
        ti, ti_level = self._level_reduce(ti, ti_level, first_level)
      term = self._ptct_mul_scalar(ti, float(coeffs[i + 1]), first_level)
      result_level = self._infer_level(result)
      term_level = self._infer_level(term)
      first_level = min(result_level, term_level)
      result, _ = self._level_reduce(result, result_level, first_level)
      term, _ = self._level_reduce(term, term_level, first_level)
      result = self._ct_add(result, term, first_level)

    # ModReduce at end — matches OpenFHE's ModReduceInPlace
    result = self._he_op_rescale(first_level, first_level - 1, result)
    return result, first_level - 1

  def _inner_eval_chebyshev_ps(self, coefficients, k, m, T, T_level,
                                T2, T2_level, baby_level):
    """Recursive InnerEvalChebyshevPS (OpenFHE ckksrns-advancedshe.cpp L650).

    coefficients: Chebyshev coefficient list (c_0 NOT halved).
    k, m: PS parameters.
    T[0..k-1]: baby-step ciphertexts for T_1..T_k.
    T2[0..m-1]: giant-step ciphertexts for T_k, T_{2k}, ..., T_{2^{m-1}*k}.
    baby_level: common level of T[0..k-1].

    Returns (result_ct, result_level).
    """
    k2m2k = k * (1 << (m - 1)) - k

    # Divide coefficients by T_{k*2^{m-1}}
    Tkm = [0.0] * (k2m2k + k + 1)
    Tkm[-1] = 1.0
    divqr_q, divqr_r = long_division_chebyshev(coefficients, Tkm)

    # Subtract T_{k*(2^{m-1} - 1)} from r
    r2 = list(divqr_r)
    n_r2 = _ps_degree(r2)
    if int(k2m2k) - int(n_r2) <= 0:
      while len(r2) <= n_r2:
        r2.append(0.0)
      r2[k2m2k] -= 1.0
    else:
      while len(r2) <= k2m2k:
        r2.append(0.0)
      r2[k2m2k] = -1.0

    # Trim r2
    n_r2_new = _ps_degree(r2)
    r2 = r2[:n_r2_new + 1]

    # Divide r2 by q
    divcs_q, divcs_r = long_division_chebyshev(r2, divqr_q)

    # ---- Evaluate qu ----
    # Matches OpenFHE InnerEvalChebyshevPS lines 680-701.
    # CRITICAL: Baby-step T[i] are at nsd=2 (from _he_op_mul in baby step
    # construction). EvalPartialLinearWSum produces nsd=1 (ends with rescale).
    # OpenFHE resolves this via AdjustLevelsAndDepthToOneInPlace before add.
    # We must rescale nsd=2 ciphertexts to nsd=1 before adding to nsd=1 results.
    dq = _ps_degree(divqr_q)
    if dq > k:
      qu, qu_level = self._inner_eval_chebyshev_ps(
          divqr_q, k, m - 1, T, T_level, T2, T2_level, baby_level)
    else:
      # Leading coefficient is always a power of 2 from Chebyshev rule.
      # Use free doubling instead of ptct_mul to avoid S^2 -> S^3.
      qu = copy.copy(T[k - 1])
      qu.polynomial = T[k - 1].polynomial.copy()
      qu, _ = self._level_reduce(qu, T_level[k - 1], baby_level)
      qu_level = baby_level
      self._copy_nsd(qu, T[k - 1])
      lead = float(divqr_q[dq])
      if abs(lead) > _PS_DELTA:
        # Free doubling: lead is always a power of 2
        n_doubles = int(round(math.log2(abs(lead))))
        for _ in range(n_doubles):
          qu = self._ct_add(qu, qu, qu_level)
        if lead < 0:
          # Negate: (q - a) for each coefficient
          moduli = self._cache.q_moduli_at_level(qu_level)
          num_q = len(moduli)
          self._ensure_canonical(qu, qu_level)
          mod_arr = jnp.array([m for m in moduli], dtype=jnp.uint64).reshape(1, 1, 1, 1, num_q)
          qu.polynomial = (mod_arr - qu.polynomial) % mod_arr

      # Add remaining terms q[1..k-1] via EvalPartialLinearWSum (OpenFHE L699-701)
      divqr_q_trunc = list(divqr_q[:k])
      n_q_trunc = _ps_degree(divqr_q_trunc) if len(divqr_q_trunc) > 0 else 0
      if n_q_trunc > 0:
        wsum, wsum_level = self._eval_partial_linear_wsum(
            T, T_level, divqr_q_trunc, n_q_trunc, baby_level)
        # wsum is nsd=1 (after rescale in linear_wsum).
        # qu may be nsd=2 (from baby step). Rescale qu to nsd=1 first.
        qu, qu_level = self._ensure_nsd1(qu, qu_level)
        wsum, wsum_level = self._ensure_nsd1(wsum, wsum_level)
        qu, qu_level = self._ct_add_aligned(
            qu, qu_level, wsum, wsum_level)
      else:
        # No wsum terms, but still need to normalize qu to nsd=1
        # for correct add_scalar and subsequent operations.
        qu, qu_level = self._ensure_nsd1(qu, qu_level)

      # Add free term q[0]/2 (OpenFHE L695)
      # qu is now nsd=1, so noise_scale_deg=1
      if abs(divqr_q[0]) > _PS_DELTA:
        self._add_scalar(qu, divqr_q[0] / 2.0, qu_level,
                         noise_scale_deg=self._get_nsd(qu))

    # ---- Evaluate su ----
    # Matches OpenFHE InnerEvalChebyshevPS lines 705-729.
    s2 = list(divcs_r)
    while len(s2) <= k2m2k:
      s2.append(0.0)
    s2[k2m2k] = 1.0

    ds = _ps_degree(s2)
    if ds > k:
      su, su_level = self._inner_eval_chebyshev_ps(
          s2, k, m - 1, T, T_level, T2, T2_level, baby_level)
    else:
      # Leading coefficient is always 1 — just clone, no ptct_mul needed
      su = copy.copy(T[k - 1])
      su.polynomial = T[k - 1].polynomial.copy()
      su, _ = self._level_reduce(su, T_level[k - 1], baby_level)
      su_level = baby_level
      self._copy_nsd(su, T[k - 1])

      # Add remaining terms via EvalPartialLinearWSum (OpenFHE L720-722)
      s2_trunc = list(s2[:k])
      n_trunc = _ps_degree(s2_trunc) if len(s2_trunc) > 0 else 0
      if n_trunc > 0:
        wsum, wsum_level = self._eval_partial_linear_wsum(
            T, T_level, s2_trunc, n_trunc, baby_level)
        # wsum is nsd=1. Rescale su (nsd=2) to nsd=1 before add.
        su, su_level = self._ensure_nsd1(su, su_level)
        wsum, wsum_level = self._ensure_nsd1(wsum, wsum_level)
        su, su_level = self._ct_add_aligned(
            su, su_level, wsum, wsum_level)
      else:
        su, su_level = self._ensure_nsd1(su, su_level)

      # Add free term s2[0]/2 (OpenFHE L725)
      if abs(s2[0]) > _PS_DELTA:
        self._add_scalar(su, s2[0] / 2.0, su_level,
                         noise_scale_deg=self._get_nsd(su))

      # Match su to T2[m-1]'s scale and level (OpenFHE L728)
      if su_level > T2_level[m - 1]:
        su, su_level = self._adjust_ct_to_match_scale(
            su, su_level, T2[m - 1])
        self._copy_nsd(su, T2[m - 1])
        if su_level > T2_level[m - 1]:
          su, su_level = self._level_reduce(su, su_level, T2_level[m - 1])

    # ---- Evaluate cu ----
    # Matches OpenFHE InnerEvalChebyshevPS lines 732-752.
    n_c = _ps_degree(divcs_q) if len(divcs_q) > 0 else 0
    cu = None
    cu_level = baby_level
    if n_c >= 1:
      if n_c == 1:
        if _ps_is_not_equal_one(float(divcs_q[1])):
          # OpenFHE L734-736: EvalMult + ModReduce
          # T[0] may be at nsd=2; ensure nsd=1 before ptct_mul
          cu_base = copy.copy(T[0])
          cu_base.polynomial = T[0].polynomial.copy()
          self._copy_nsd(cu_base, T[0])
          cu_base, cu_base_level = self._level_reduce(
              cu_base, T_level[0], baby_level)
          cu_base, cu_base_level = self._ensure_nsd1(
              cu_base, cu_base_level)
          cu = self._ptct_mul_scalar(
              cu_base, float(divcs_q[1]), cu_base_level)
          cu = self._he_op_rescale(
              cu_base_level, cu_base_level - 1, cu)
          cu_level = cu_base_level - 1
        else:
          # OpenFHE L738-739: coefficient is 1, just clone
          cu = copy.copy(T[0])
          cu.polynomial = T[0].polynomial.copy()
          self._copy_nsd(cu, T[0])
          cu, cu_level = self._level_reduce(cu, T_level[0], baby_level)
          # Normalize to nsd=1 for consistent addition
          cu, cu_level = self._ensure_nsd1(cu, cu_level)
      else:
        # OpenFHE L743: EvalPartialLinearWSum (produces nsd=1)
        cu, cu_level = self._eval_partial_linear_wsum(
            T, T_level, divcs_q, n_c, baby_level)
      # Add free term c_0/2 (OpenFHE L747)
      if abs(divcs_q[0]) > _PS_DELTA:
        self._add_scalar(cu, divcs_q[0] / 2.0, cu_level,
                         noise_scale_deg=self._get_nsd(cu))

      # Match cu to T2[m-1]'s scale and level (OpenFHE L750-752)
      if cu_level > T2_level[m - 1]:
        cu, cu_level = self._adjust_ct_to_match_scale(
            cu, cu_level, T2[m - 1])
        self._copy_nsd(cu, T2[m - 1])
        if cu_level > T2_level[m - 1]:
          cu, cu_level = self._level_reduce(cu, cu_level, T2_level[m - 1])

    # cu_total = T2[m-1] + cu  (or T2[m-1] + c_0/2 if cu was None)
    t2m1 = copy.copy(T2[m - 1])
    t2m1.polynomial = T2[m - 1].polynomial.copy()
    t2m1_level = T2_level[m - 1]
    self._copy_nsd(t2m1, T2[m - 1])
    self._ensure_canonical(t2m1, t2m1_level)

    if cu is not None:
      # Combine 1: cu_total = T2[m-1] + cu
      # Only normalize nsd when the two operands disagree.
      # When both are nsd=1 this is a no-op.  When both are nsd=2
      # (giant step output + cu at matching nsd) skipping avoids an
      # extra ModReduce level that OpenFHE's EvalAdd doesn't spend.
      nsd_t = self._get_nsd(t2m1)
      nsd_c = self._get_nsd(cu)
      if nsd_t != nsd_c:
        t2m1, t2m1_level = self._ensure_nsd1(t2m1, t2m1_level)
        cu, cu_level = self._ensure_nsd1(cu, cu_level)
      t2m1_actual = self._infer_level(t2m1)
      cu_actual = self._infer_level(cu)
      if t2m1_actual > cu_actual:
        t2m1, t2m1_level = self._adjust_ct_to_match_scale(
            t2m1, t2m1_actual, cu)
        self._copy_nsd(t2m1, cu)
        if self._infer_level(t2m1) > cu_actual:
          t2m1, t2m1_level = self._level_reduce(
              t2m1, self._infer_level(t2m1), cu_actual)
      elif cu_actual > t2m1_actual:
        cu, cu_level = self._adjust_ct_to_match_scale(
            cu, cu_actual, t2m1)
        self._copy_nsd(cu, t2m1)
        if self._infer_level(cu) > t2m1_actual:
          cu, cu_level = self._level_reduce(
              cu, self._infer_level(cu), t2m1_actual)
      out_lev = min(self._infer_level(t2m1), self._infer_level(cu))
      t2m1, _ = self._level_reduce(
          t2m1, self._infer_level(t2m1), out_lev
      )
      cu, _ = self._level_reduce(cu, self._infer_level(cu), out_lev)
      self._ensure_canonical(t2m1, out_lev)
      self._ensure_canonical(cu, out_lev)
      cu_total = self._ct_add(t2m1, cu, out_lev)
      cu_total_level = out_lev
    else:
      # No cu: just add free term c_0/2 to T2[m-1]
      c0_half = divcs_q[0] / 2.0 if len(divcs_q) > 0 else 0.0
      self._add_scalar(t2m1, c0_half, t2m1_level,
                       noise_scale_deg=self._get_nsd(t2m1))
      cu_total = t2m1
      cu_total_level = t2m1_level

    # result = cu_total * qu + su (OpenFHE L757-759)
    # _he_op_mul handles NSD internally: nsd=2 inputs get pre-rescaled,
    # nsd=1 inputs are used directly. The key is that cu_total and qu
    # must be at the same level AND same nsd for _he_op_mul to handle them.
    #
    # After our NSD fixes above, cu_total and qu should already be at nsd=1.
    # But ensure consistency:
    # Only normalize nsd when operands disagree.
    if self._get_nsd(cu_total) != self._get_nsd(qu):
      cu_total, cu_total_level = self._ensure_nsd1(cu_total, cu_total_level)
      qu, qu_level = self._ensure_nsd1(qu, qu_level)

    # Match levels via Mode C: adjust the higher-level operand's polynomial
    # scale to match the lower one, then LevelReduce for nq alignment.
    mul_tgt = min(cu_total_level, qu_level)
    if cu_total_level > qu_level:
      cu_total, cu_total_level = self._adjust_ct_to_match_scale(
          cu_total, cu_total_level, qu)
      self._copy_nsd(cu_total, qu)
      if cu_total_level > mul_tgt:
        cu_total, cu_total_level = self._level_reduce(
            cu_total, cu_total_level, mul_tgt)
    elif qu_level > cu_total_level:
      qu, qu_level = self._adjust_ct_to_match_scale(
          qu, qu_level, cu_total)
      self._copy_nsd(qu, cu_total)
      if qu_level > mul_tgt:
        qu, qu_level = self._level_reduce(qu, qu_level, mul_tgt)
    self._ensure_canonical(cu_total, mul_tgt)
    self._ensure_canonical(qu, mul_tgt)

    # _he_op_mul(output_level, ct1_nsd1, ct2_nsd1): inputs at level, output at level, nsd=2
    mul_out = mul_tgt
    if mul_out < 0:
      return qu, qu_level  # degenerate: no levels left
    result = self._he_op_mul(mul_out, cu_total, qu)
    # result is nsd=2 from _he_op_mul, at mul_out level

    # Combine 3: result + su — only normalize nsd if they disagree
    if self._get_nsd(result) != self._get_nsd(su):
      result, result_level = self._ensure_nsd1(result, mul_out)
      su, su_level = self._ensure_nsd1(su, su_level)
    else:
      result_level = mul_out
    res_actual = self._infer_level(result)
    su_actual = self._infer_level(su)
    if res_actual > su_actual:
      result, result_level = self._adjust_ct_to_match_scale(
          result, res_actual, su)
      self._copy_nsd(result, su)
      if self._infer_level(result) > su_actual:
        result, result_level = self._level_reduce(
            result, self._infer_level(result), su_actual)
    elif su_actual > res_actual:
      su, su_level = self._adjust_ct_to_match_scale(
          su, su_actual, result)
      self._copy_nsd(su, result)
      if self._infer_level(su) > res_actual:
        su, su_level = self._level_reduce(
            su, self._infer_level(su), res_actual)
    out_lev = min(self._infer_level(result), self._infer_level(su))
    result, _ = self._level_reduce(
        result, self._infer_level(result), out_lev
    )
    su, _ = self._level_reduce(su, self._infer_level(su), out_lev)
    self._ensure_canonical(result, out_lev)
    self._ensure_canonical(su, out_lev)
    result = self._ct_add(result, su, out_lev)
    result_level = out_lev

    # Barrier per recursion level: bound the cu*qu he_op_mul + recursive-call
    # intermediates (part of the ApproxMod live set at N=65536).
    if self._low_mem:
      self._stage_checkpoint("cheby-inner", result)

    return result, result_level

  @staticmethod
  def _int_crt_factors(scalar: float, sf: float, moduli):
    """Encode scalar * sf as integer CRT residues (matches OpenFHE GetElementForEvalMult).

    Returns a list of integers, one per modulus, such that
    factors[i] = round(scalar * sf) mod moduli[i].
    Uses the same overflow-handling approach as OpenFHE: split into
    a base integer and a power-of-2 scale-back factor.
    """
    scaled = scalar * sf
    abs_scaled = abs(scaled)
    if abs_scaled < 0.5:
      return [0] * len(moduli)
    log_sf = math.ceil(math.log2(abs_scaled)) if abs_scaled >= 1 else 0
    log_valid = min(log_sf, 125)
    log_approx = max(0, log_sf - log_valid)
    approx_factor = float(1 << log_approx) if log_approx > 0 else 1.0
    large = round(scaled / approx_factor)
    factors = []
    for qi in moduli:
      qi_int = int(qi)
      r = large % qi_int
      if r < 0:
        r += qi_int
      if log_approx > 0:
        step = pow(2, log_approx, qi_int)
        r = (r * step) % qi_int
      factors.append(r)
    return factors

  def _ptct_mul_scalar(self, ct, scalar: float, level: int):
    """Multiply ciphertext by a scalar via ptct_mul (CKKS encoding).

    Encodes the scalar at the ciphertext's tracked scale for precision.
    """
    moduli = self._cache.q_moduli_at_level(level)
    num_q = len(moduli)
    self._ensure_canonical(ct, level)

    has_tracked = hasattr(ct, '_ckks_scale') and ct._ckks_scale is not None
    if has_tracked and math.isfinite(ct._ckks_scale) and ct._ckks_scale > 0:
      encode_sf = ct._ckks_scale
    else:
      encode_sf = self._openfhe_scale_at_level(level)

    if abs(scalar) > 0 and encode_sf > 0 and abs(scalar * encode_sf) < 0.5:
      result = copy.copy(ct)
      result.polynomial = jnp.zeros_like(ct.polynomial)
      self._set_scale(result, 0.0)
      self._set_nsd(result, 2)
      return result

    # Complex slots are host-side plaintext-encoder inputs. Creating this as a
    # TPU c128 array makes TPU7x attempt an unsupported 16-byte scalar
    # broadcast before `_encode_diagonal` immediately copies it back to host.
    import numpy as _np
    diag = _np.full(self._num_slots, complex(scalar, 0.0),
                    dtype=_np.complex128)
    pt_ntt = self._encode_diagonal(diag, moduli, num_q, scale=encode_sf)
    op = self.ctx.ptct_mul[level]
    result = op._mul_encoded(ct, pt_ntt)
    self._ensure_canonical(result, level)

    has_tracked = hasattr(ct, '_ckks_scale') and ct._ckks_scale is not None
    if has_tracked:
      new_sf = ct._ckks_scale * encode_sf
      if math.isfinite(new_sf):
        self._set_scale(result, new_sf)
      else:
        self._copy_scale(result, ct)
    # ptct_mul with tracked-scale encoding produces nsd=2.
    self._set_nsd(result, 2)
    return result

  def _add_scalar(self, ct, scalar: float, level: int, noise_scale_deg: int = 1):
    """Add a plaintext scalar constant to element 0 of a ciphertext.

    Encodes the scalar via CKKS encode (which correctly handles the
    negacyclic NTT transformation) and adds to element 0.

    Args:
      ct: ciphertext to modify in-place.
      scalar: the scalar value to add.
      level: the current level.
      noise_scale_deg: 1 if ct is at scale S, 2 if at scale S^2.
    """
    moduli = self._cache.q_moduli_at_level(level)
    num_q = len(moduli)
    self._ensure_canonical(ct, level)

    # nsd=2: encode at sqrt(tracked) = S.  Combined with the S² polynomial
    #   this effectively adds scalar at the nsd=1 base scale.
    # nsd=1: encode at tracked scale (matches the polynomial's actual
    #   encoding precisely). Falls back to recursive per-level SF only if
    #   no tracked scale is available. Using tracked scale avoids the
    #   0.07-0.5% mismatch between recursive SF and direct prime products
    #   that accumulates through PS baby-step construction at lower levels.
    if noise_scale_deg == 2:
      has_tracked = hasattr(ct, '_ckks_scale') and ct._ckks_scale is not None
      if has_tracked and math.isfinite(ct._ckks_scale) and ct._ckks_scale > 0:
        sf_base = math.sqrt(ct._ckks_scale)
      else:
        sf_base = self._openfhe_scale_at_level(level)
    else:
      has_tracked = hasattr(ct, '_ckks_scale') and ct._ckks_scale is not None
      if has_tracked and math.isfinite(ct._ckks_scale) and ct._ckks_scale > 0:
        sf_base = ct._ckks_scale
      else:
        sf_base = self._openfhe_scale_at_level(level)

    # Encode the constant via CKKS encode (handles negacyclic NTT correctly).
    const_slots = [complex(scalar, 0.0)] * self._num_slots
    from ckks_ctx import _ckks_encode as ckks_encode
    m = self._degree * 2
    encoded_rns = ckks_encode(
        slots=const_slots,
        cycl_order=m,
        q_towers=list(moduli),
        p_towers=[],
        scale=sf_base,
        max_bits_in_word=self.ctx.parameters.get("max_bits_in_word", 61),
        max_bits_value=self.ctx.parameters.get(
            "max_bits_value", (1 << 63) - (1 << 9) - 1),
    )
    r, c = self._cache.degree_layout
    pt_arr = jnp.array(encoded_rns, dtype=jnp.uint64).reshape(r, c, num_q)

    if noise_scale_deg == 2:
      # Scale up by S to go from encoding at S to S^2.
      sf_int = int(round(sf_base))
      moduli_red = jnp.array(moduli, dtype=jnp.uint64).reshape(1, 1, num_q)
      sf_mod_q = jnp.array([sf_int % q for q in moduli],
                            dtype=jnp.uint64).reshape(1, 1, num_q)
      pt_arr = (pt_arr * sf_mod_q) % moduli_red

    # The raw add below requires the plaintext in the ciphertext's
    # representation: encode into the backend's computation format exactly
    # once (identity for Barrett; Montgomery: pt*R mod q).
    pt_arr = jnp.asarray(
        self._cache.ff_q_max.slice(num_q).to_computation_format(
            pt_arr.astype(jnp.uint64)),
        jnp.uint64)

    pt_broadcast = pt_arr.reshape(1, 1, r, c, num_q)

    ct_poly = ct.polynomial.astype(jnp.uint64)
    moduli_arr = jnp.array(moduli, dtype=jnp.uint64).reshape(1, 1, 1, 1, num_q)

    elem0 = (ct_poly[:, 0:1, :, :, :] + pt_broadcast) % moduli_arr
    ct.polynomial = ct_poly.at[:, 0:1, :, :, :].set(elem0).astype(jnp.uint32)
    return ct

  def _ct_add(self, ct_a, ct_b, level):
    """Ciphertext-ciphertext modular addition. Returns a new Polynomial."""
    moduli = self._cache.q_moduli_at_level(level)
    num_q = len(moduli)
    self._ensure_canonical(ct_a, level)
    self._ensure_canonical(ct_b, level)
    moduli_arr = jnp.array(moduli, dtype=jnp.uint64).reshape(1, 1, 1, 1, num_q)
    a = ct_a.polynomial.astype(jnp.uint64)
    b = ct_b.polynomial.astype(jnp.uint64)
    sum_poly = ((a + b) % moduli_arr).astype(jnp.uint32)
    out = ct_a._clone_with_payload(sum_poly)
    # Propagate scale from ct_a (scales should match for addition)
    self._copy_scale(out, ct_a)
    # nsd = max(nsd_a, nsd_b) (matching OpenFHE EvalAddCore)
    self._set_nsd(out, max(self._get_nsd(ct_a), self._get_nsd(ct_b)))
    return out

  def _ct_sub(self, ct_a, ct_b, level):
    """Ciphertext-ciphertext modular subtraction (a - b). Returns a new Polynomial."""
    moduli = self._cache.q_moduli_at_level(level)
    num_q = len(moduli)
    self._ensure_canonical(ct_a, level)
    self._ensure_canonical(ct_b, level)
    moduli_arr = jnp.array(moduli, dtype=jnp.uint64).reshape(1, 1, 1, 1, num_q)
    a = ct_a.polynomial.astype(jnp.uint64)
    b = ct_b.polynomial.astype(jnp.uint64)
    diff_poly = ((a + moduli_arr - b) % moduli_arr).astype(jnp.uint32)
    out = ct_a._clone_with_payload(diff_poly)
    # Propagate scale from ct_a (scales should match for subtraction)
    self._copy_scale(out, ct_a)
    # nsd = max(nsd_a, nsd_b) (matching OpenFHE EvalSubCore)
    self._set_nsd(out, max(self._get_nsd(ct_a), self._get_nsd(ct_b)))
    return out

  def _ct_op_aligned(self, ct_a, ct_a_level, ct_b, ct_b_level, op='sub'):
    """Scale-safe ciphertext combine (add or sub).

    Matches OpenFHE's AdjustForAddOrSubInPlace: before combining, checks
    BOTH level AND tracked-scale chain. Adjusts the operand with more
    headroom (higher level or closer to max_level) to match the other.

    Three cases:
      1. Different levels → adjust higher-level operand (costs 1 level)
      2. Same level, different tracked scales (>0.01%) → adjust the one
         with more spare moduli (costs 1 level from spare)
      3. Same level, matching scales → direct combine (free)

    Returns: (result_ct, result_level).
    """
    combine = self._ct_sub if op == 'sub' else self._ct_add

    # Infer actual levels from polynomial shapes
    actual_a = self._infer_level(ct_a)
    actual_b = self._infer_level(ct_b)
    level_differs = (actual_a != actual_b)

    # Check tracked-scale mismatch
    sf_a = self._get_scale(ct_a)
    sf_b = self._get_scale(ct_b)
    scale_mismatch = False
    if sf_a and sf_b and sf_a > 0 and sf_b > 0:
      ratio = sf_a / sf_b
      if abs(ratio - 1.0) > 1e-4:  # > 0.01% → adjustment needed
        scale_mismatch = True

    need_adjust = level_differs or scale_mismatch

    if need_adjust:
      # Determine which operand to adjust: pick the one with MORE moduli
      # (higher actual level = more spare room for ptct_mul + ModReduce).
      if actual_b > actual_a:
        ct_b_adj, adj_level = self._adjust_ct_to_match_scale(
            ct_b, actual_b, ct_a)
        out_level = min(actual_a, adj_level)
        if ct_a.polynomial.shape[-1] != ct_b_adj.polynomial.shape[-1]:
          ct_a, _ = self._level_reduce(ct_a, actual_a, out_level)
          ct_b_adj, _ = self._level_reduce(ct_b_adj, adj_level, out_level)
        self._ensure_canonical(ct_a, out_level)
        self._ensure_canonical(ct_b_adj, out_level)
        return combine(ct_a, ct_b_adj, out_level), out_level
      elif actual_a > actual_b:
        ct_a_adj, adj_level = self._adjust_ct_to_match_scale(
            ct_a, actual_a, ct_b)
        out_level = min(actual_b, adj_level)
        if ct_a_adj.polynomial.shape[-1] != ct_b.polynomial.shape[-1]:
          ct_a_adj, _ = self._level_reduce(ct_a_adj, adj_level, out_level)
          ct_b, _ = self._level_reduce(ct_b, actual_b, out_level)
        self._ensure_canonical(ct_a_adj, out_level)
        self._ensure_canonical(ct_b, out_level)
        return combine(ct_a_adj, ct_b, out_level), out_level
      else:
        # Same actual level but tracked scales differ by >0.01%.  Neither
        # operand has a spare level to absorb a scale correction, so a direct
        # combine here would silently lose precision.  Surface it instead of
        # degrading the result.
        raise RuntimeError(
            f"cannot {op} ciphertexts at the same level {actual_a} with "
            f"mismatched scales (sf_a={sf_a:.6e}, sf_b={sf_b:.6e}, "
            f"ratio={sf_a / sf_b:.6f}): no spare level to align scales.")

    # Direct combine (same level, matching or near-matching scales)
    out_level = min(actual_a, actual_b)
    nq_a = ct_a.polynomial.shape[-1]
    nq_b = ct_b.polynomial.shape[-1]
    if nq_a != nq_b:
      if nq_a > nq_b:
        ct_a, _ = self._level_reduce(ct_a, actual_a, out_level)
      else:
        ct_b, _ = self._level_reduce(ct_b, actual_b, out_level)
    self._ensure_canonical(ct_a, out_level)
    self._ensure_canonical(ct_b, out_level)
    return combine(ct_a, ct_b, out_level), out_level

  def _ct_sub_aligned(self, ct_a, ct_a_level, ct_b, ct_b_level):
    """Scale-aligned ciphertext subtraction. See _ct_op_aligned."""
    return self._ct_op_aligned(ct_a, ct_a_level, ct_b, ct_b_level, op='sub')

  def _ct_add_aligned(self, ct_a, ct_a_level, ct_b, ct_b_level):
    """Scale-aligned ciphertext addition. See _ct_op_aligned."""
    return self._ct_op_aligned(ct_a, ct_a_level, ct_b, ct_b_level, op='add')

  def _rotate_by(self, ct, rot_index, level):
    """Slot rotation by rot_index via the he_rot accessor (scale/nsd kept).

    Requires the rotation key for rot_index to be registered (control_gen
    adds the sparse partial-sum / final-rotate indices to _all_rot_indices).
    """
    rot_input = copy.copy(ct)
    rot_input.polynomial = ct.polynomial
    self._ensure_canonical(rot_input, level)
    rotated = self.ctx.he_rot[level, int(rot_index)].rotate(rot_input)
    self._ensure_canonical(rotated, level)
    self._copy_scale(rotated, ct)
    self._copy_nsd(rotated, ct)
    return rotated

  def _conjugate(self, ct, level):
    """CKKS conjugation via Galois automorphism at index M-1 = 2*N-1.

    Maps slot[i] -> conj(slot[i]). Uses the same he_rot infrastructure
    as other rotations. Requires the conjugation key to be registered
    in setup_key().
    """
    conj_idx = self._m - 1  # M-1 = 2*degree - 1
    rot_input = copy.copy(ct)
    rot_input.polynomial = ct.polynomial
    # Rotation expects 5D input (batch, ne, r, c, nq)
    self._ensure_canonical(rot_input, level)
    rotated = self.ctx.he_rot[level, conj_idx].rotate(rot_input)
    self._ensure_canonical(rotated, level)
    # Scale and nsd unchanged by conjugation
    self._copy_scale(rotated, ct)
    self._copy_nsd(rotated, ct)
    return rotated

  def _mult_by_monomial(self, ct, k, level):
    """Multiply ciphertext polynomial by X^k mod (X^N + 1).

    With negacyclic NTT (ntt_mm now handles psi internally):
      1. to_coeffs_form()  → true negacyclic coefficients
      2. shift coefficients → X^k * p(X) in Z[X]/(X^N+1)
      3. to_ntt_form()     → back to NTT
    """
    moduli = self._cache.q_moduli_at_level(level)
    num_q = len(moduli)
    self._ensure_canonical(ct, level)
    N = self._degree
    M = 2 * N
    k = k % M

    if k == 0:
      out = copy.copy(ct)
      out.polynomial = ct.polynomial
      return out

    r_lay, c_lay = self._cache.degree_layout
    ne = ct.polynomial.shape[1]

    work = ct._clone_with_payload(ct.polynomial)

    # Step 1: iNTT → true negacyclic coefficients.
    # Montgomery INTT output is lazy in [0, 2q); the negation below computes
    # (q - a) % q in uint64, which is wrong for a >= q, so canonicalize
    # first (identity for Barrett).
    work.to_coeffs_form()
    work.polynomial = work.ntt_ctx.ff_ctx.strictify(work.polynomial)

    # Step 2: Shift coefficients (X^k in Z[X]/(X^N+1))
    moduli_arr = jnp.array(moduli, dtype=jnp.uint64).reshape(1, 1, 1, num_q)
    coeffs = work.polynomial.reshape(self._cache.batch, ne, N, num_q)

    q_quot = k // N
    r_shift = k % N

    old_indices = [(p - r_shift) % N for p in range(N)]
    signs = [1 if (p - r_shift) >= 0 else -1 for p in range(N)]
    if q_quot % 2 == 1:
      signs = [-s for s in signs]

    old_indices_jax = jnp.array(old_indices, dtype=jnp.int32)
    new_coeffs = coeffs[..., old_indices_jax, :]

    signs_arr = jnp.array(signs, dtype=jnp.int32).reshape(1, 1, N, 1)
    pos_mask = (signs_arr > 0)
    neg_coeffs = (moduli_arr - new_coeffs.astype(jnp.uint64)) % moduli_arr
    result_coeffs = jnp.where(pos_mask, new_coeffs.astype(jnp.uint64),
                               neg_coeffs).astype(jnp.uint32)

    work.polynomial = result_coeffs.reshape(
        self._cache.batch, ne, r_lay, c_lay, num_q)

    # Step 3: NTT → back to NTT form. Strictify the (possibly lazy
    # Montgomery) NTT output so the op-boundary contract (strict residues)
    # holds for downstream raw modular arithmetic; identity for Barrett.
    work.to_ntt_form()
    work.polynomial = work.ntt_ctx.ff_ctx.strictify(work.polynomial)

    # Scale and nsd unchanged by monomial multiplication
    self._copy_scale(work, ct)
    self._copy_nsd(work, ct)
    return work

  def _encode_diagonal(self, diag: jnp.ndarray,
                       moduli: Sequence[int],
                       num_q: int,
                       scale: float = None) -> jnp.ndarray:
    """Encode a complex slot vector into RNS/NTT-domain plaintext layout.

    Args:
      diag: Complex diagonal vector (one entry per slot).
      moduli: The RNS moduli at the target level.
      num_q: Number of Q-tower moduli at target level.
      scale: Scaling factor to use for encoding. If None, uses
        ctx.scaling_factor (fixed scale). For per-level encoding,
        pass cache.scaling_factor_at_level(level).
    """
    slots = list(diag)
    # Sparse C2S/S2C diagonals are 2*num_slots-length [d | i*d] patterns
    # (encoded at the doubled slot count, matching OpenFHE MakeAuxPlaintext
    # with rot.size() slots); everything else encodes at num_slots.
    if self._sparse and len(slots) == 2 * self._num_slots:
      ns = 2 * self._num_slots
    else:
      ns = self._num_slots
    if len(slots) < ns:
      slots = slots + [complex(0.0, 0.0)] * (ns - len(slots))
    elif len(slots) > ns:
      slots = slots[:ns]

    # Bootstrap DFT diagonals are evaluator-owned complex constants, so they
    # use the private complex-capable encoder rather than the real-only public
    # inference boundary.
    from ckks_ctx import _ckks_encode as ckks_encode
    encoded_rns = ckks_encode(
        slots=slots,
        cycl_order=self._degree * 2,
        q_towers=list(moduli),
        p_towers=[],
        scale=self.ctx.scaling_factor if scale is None else scale,
        max_bits_in_word=self.ctx.parameters.get("max_bits_in_word", 61),
        max_bits_value=self.ctx.parameters.get(
            "max_bits_value", (1 << 63) - (1 << 9) - 1),
    )
    arr = jnp.array(encoded_rns, dtype=jnp.uint64)
    return arr.reshape(*self._cache.degree_layout, num_q).astype(jnp.uint32)

  def _infer_level(self, ct) -> int:
    """Infer CKKS level from actual moduli count."""
    self._ensure_canonical(ct)
    actual_nq = ct.polynomial.shape[-1]
    cache = self._cache
    return max(0,
               cache.max_level - (cache.num_q - actual_nq) // cache.composite_degree)


def decrypt_decode(ctx, bs, ct, num_slots):
  """Decrypt a ciphertext at its current level and decode CKKS slot values.

  Self-contained test/validation utility. The ciphertext stays a canonical
  ``Polynomial`` through the public decrypt boundary.

  Args:
    ctx: CKKSContext holding the secret key and ring parameters.
    bs:  Bootstrap instance (for the tracked scale of ``ct``).
    ct:  ciphertext (Polynomial) to decrypt.
    num_slots: number of CKKS slots to decode.

  Returns:
    np.ndarray of complex slot values (length num_slots).
  """
  import numpy as np
  if not isinstance(ct, polynomial.Polynomial):
    raise TypeError(
        f'decrypt_decode requires Polynomial, got {type(ct).__name__}.'
    )
  ct.validate()
  scale = bs._get_scale(ct)
  decrypted = ctx.decrypt(ct)
  decrypted.validate()
  nq = decrypted.num_moduli
  degree = decrypted.degree
  pt_rns = np.asarray(decrypted.to_array()[0, 0]).reshape(degree, nq)
  q_slice = list(decrypted.moduli)

  # Exact CRT reconstruction followed by direct slot evaluation is kept for
  # this bootstrap diagnostic because it preserves complex error information.
  Q = math.prod(int(q) for q in q_slice)
  Qi = [Q // int(q) for q in q_slice]
  inv_crt = [
      pow(qi, -1, int(q)) for qi, q in zip(Qi, q_slice, strict=True)
  ]
  half = Q // 2
  coeffs = []
  for residues in pt_rns.tolist():
    combined = sum(
        int(residue) * qi * inv
        for residue, qi, inv in zip(residues, Qi, inv_crt, strict=True)
    ) % Q
    centered = combined - Q if combined > half else combined
    coeffs.append(float(centered) / scale)

  m = 2 * degree
  rot_group = []
  val_g = 1
  for _ in range(num_slots):
    rot_group.append(val_g)
    val_g = (val_g * 5) % m

  slots_out = []
  for rotation in rot_group:
    xi = np.exp(2j * np.pi * rotation / m)
    value = 0.0 + 0.0j
    for coefficient in reversed(coeffs):
      value = value * xi + coefficient
    slots_out.append(value)
  return np.asarray(slots_out)


__all__ = [
    "Bootstrap",
    "decrypt_decode",
    "_bitrev_permutation",
    "chebyshev_coefficients_for_sine",
    "compute_degrees_ps",
    "eval_chebyshev_numeric",
    "eval_linear_transform_numeric",
    "long_division_chebyshev",
]

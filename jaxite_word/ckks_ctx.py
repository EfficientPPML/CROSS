import cmath
import contextlib
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

import util
from he_ops import _HEAddAccessor
from he_ops import _HEBootstrapAccessor
from he_ops import _HEMulAccessor
from he_ops import _HEPtCtMulAccessor
from he_ops import _HELevelReduceAccessor, _HERescaleAccessor
from he_ops import _HERotAccessor
from he_ops import _HESubAccessor
from he_params import HEParameterCache
from he_params import normalize_noise_scale_degree
import key_gen as kg
import polynomial as poly

if __name__ == 'jaxite_word.ckks_ctx':
  sys.modules.setdefault('ckks_ctx', sys.modules[__name__])
elif __name__ == 'ckks_ctx':
  # If the historical flat spelling is imported after the package but before
  # the package submodule, reserve the canonical name for this same module.
  sys.modules.setdefault('jaxite_word.ckks_ctx', sys.modules[__name__])


sigma = 3.190000057220458984375

# Private. Mutated only by ``bypass_decode_stddev_check``; callers must not poke
# it directly. The reference decoder consults it in addition to its own
# ``validate_approximation`` argument, so a caller that cannot thread the
# argument through an intermediate layer still has a scoped, restoring escape
# hatch instead of a process-global flag nothing puts back.
_decode_stddev_check_enabled = True


@contextlib.contextmanager
def bypass_decode_stddev_check():
  """Disable the reference decoder's approximation guard for this block.

  Bootstrap intermediates legitimately carry approximation error above the
  decode limit. Prefer ``validate_approximation=False`` on the decode call when
  you can reach it; use this when the decode happens inside a helper you do not
  control. The previous setting is restored on exit, including on exception.
  """
  global _decode_stddev_check_enabled
  previous = _decode_stddev_check_enabled
  _decode_stddev_check_enabled = False
  try:
    yield
  finally:
    _decode_stddev_check_enabled = previous

# Decode fails closed above this conjugate-symmetry error, measured at unit
# (message) scale. Roughly five bits of surviving fractional precision. Both
# the reference and the vectorized decoder enforce it through
# ``_check_decode_stddev``.
_DECODE_STDDEV_LIMIT = 2.0 ** -5


class ScalingFactorTooSmall(ValueError):
  """Raised by the CKKS encode path when the scaling factor is too small.

  Every scaled coefficient rounds to zero (``max|scale * coeff| <= 0.5``), so
  the encoded plaintext would be identically zero. Subclasses ``ValueError`` so
  existing ``except ValueError`` handlers keep working, while callers that need
  to distinguish this specific condition (e.g. BSGS zero-encoding a
  sub-resolution diagonal) can catch it by type instead of string-matching the
  message.
  """


class ApproximationErrorTooHigh(ValueError):
  """Raised when CKKS decode detects loss of conjugate-symmetry precision."""


def _normalize_noise_std(value) -> float:
  """Return one Gaussian standard deviation supported by key generation."""
  return kg.normalize_noise_std(value)


def _resolve_codec_scale(value, explicit_scale, fallback_scale) -> float:
  """Resolve explicit, tracked, then fallback CKKS scale in that order."""
  if explicit_scale is not None:
    candidate = explicit_scale
  else:
    tracked = getattr(value, '_ckks_scale', None)
    candidate = fallback_scale if tracked is None else tracked
  try:
    resolved = float(candidate)
  except (TypeError, ValueError, OverflowError) as exc:
    raise ValueError("scale must be finite and positive") from exc
  if not math.isfinite(resolved) or resolved <= 0:
    raise ValueError("scale must be finite and positive")
  return resolved


########################
# Common Functions
########################
def _balanced_degree_layout(degree: int) -> tuple[int, int]:
  """Return the closest factor pair for a two-dimensional NTT layout."""
  if (
      isinstance(degree, bool)
      or not isinstance(degree, (int, np.integer))
      or int(degree) < 1
  ):
    raise ValueError('degree must be a positive int.')
  degree = int(degree)
  rows = math.isqrt(degree)
  while degree % rows:
    rows -= 1
  return rows, degree // rows


def _roots(m: int) -> List[complex]:
  return [cmath.exp(1j * 2 * math.pi * k / m) for k in range(m)]


def _rot_group(m: int, nh: int, g: int = 5) -> List[int]:
  # CKKS rotation subgroup (powers of 5 mod m)
  r = [1]
  for _ in range(1, nh):
    r.append((r[-1] * g) % m)
  return r


def _bitrev_inplace(a: List[complex]) -> None:
  n = len(a)
  j = 0
  for i in range(1, n):
    bit = n >> 1
    while j & bit:
      j ^= bit
      bit >>= 1
    j ^= bit
    if i < j:
      a[i], a[j] = a[j], a[i]


def FFTSpecialInv(vals: List[complex], cycl_order: int) -> None:
  """CKKS 'special' inverse FFT (DIF-style), in-place:

  - twiddles via rotation group,
  - per-stage idx = ((lenq - (rg[j] % lenq)) % lenq) * (m/lenq),
  - scale by 1/Nh,
  - *post* bit-reversal (to match OpenFHE ordering in your test).
  """
  m = cycl_order
  nh = len(vals)
  roots = _roots(m)
  rg = _rot_group(m, nh, g=5)

  length = nh
  while length >= 1:
    half = length >> 1
    lenq = length << 2
    step = m // lenq
    for i in range(0, nh, length):
      for j in range(half):
        mod = rg[j] % lenq
        idx = ((lenq - mod) % lenq) * step
        u = vals[i + j]
        t = vals[i + j + half]
        vals[i + j] = u + t
        vals[i + j + half] = (u - t) * roots[idx]
    length >>= 1

  inv = 1.0 / nh
  for k in range(nh):
    vals[k] *= inv

  _bitrev_inplace(vals)


def FFTSpecial(vals: List[complex], cycl_order: int) -> None:
  """CKKS 'special' forward FFT (DIT-style), in-place:

  - *pre* bit-reversal,
  - per-stage idx = (rg[j] % lenq) * (m/lenq),
  - no final scale.
  """
  m = cycl_order
  nh = len(vals)
  roots = _roots(m)
  rg = _rot_group(m, nh, g=5)

  _bitrev_inplace(vals)

  length = 2
  while length <= nh:
    half = length >> 1
    lenq = length << 2
    step = m // lenq
    for i in range(0, nh, length):
      for j in range(half):
        mod = rg[j] % lenq
        idx = mod * step
        u = vals[i + j]
        v = vals[i + j + half] * roots[idx]
        vals[i + j] = u + v
        vals[i + j + half] = u - v
    length <<= 1


def nearest_int(x: float) -> int:
  # matches OpenFHE's usual "nearest" behavior used for CKKS encode
  # ``` int(math.floor(x + 0.5)) if x >= 0 else -int(math.ceil(-x + 0.5)) ```
  # This code has issue for large value, like -8614798441835585 will be returned as -8614798441835586.
  # because internal data representation is not exact. So we end up using the following implementation.
  if x >= 0:
    f = math.floor(x)
    frac = x - f
    if frac > 0.5:
      return int(f + 1)
    elif frac < 0.5:
      return int(f)
    else:  # exactly half
      return int(f + 1)
  else:
    c = math.ceil(x)  # integer toward 0
    frac = c - x  # distance to c (>= 0)
    if frac > 0.5:
      return int(c - 1)
    elif frac < 0.5:
      return int(c)
    else:  # exactly half
      return int(c - 1)


def slot_to_coeffs(y: List[complex]) -> List[float]:
  """Slot -> real polynomial coefficients (length N=2*Nh).

  Adjust here if your build packs coefficients differently. Current:
  [Re(y_0..y_{Nh-1}), Im(y_0..y_{Nh-1})]
  """
  nh = len(y)
  re = [y[i].real for i in range(nh)]
  im = [y[i].imag for i in range(nh)]
  return re + im


def fit_to_native_vector(
    vec: List[int], big_bound: int, modulus: int, ring_dim: int
) -> List[int]:
  """Python equivalent of CKKSPackedEncoding::FitToNativeVector.

  - Places dslots values into a length-ring_dim vector at indices i*gap

    where gap = ring_dim // dslots.
  - Maps each input value n using bigBound and modulus:
      if n > bigBound/2: (n - (bigBound - modulus)) mod modulus
      else:               n mod modulus
  """
  dslots = len(vec)
  if dslots == 0:
    return [0] * ring_dim

  big_value_half = big_bound >> 1
  diff = big_bound - modulus
  gap = ring_dim // dslots

  native = [0] * ring_dim
  for i, val in enumerate(vec):
    n = int(val)
    if n > big_value_half:
      mapped = (n - diff) % modulus
    else:
      mapped = n % modulus
    native[gap * i] = mapped
  return native


# ===========================================================================
# Vectorized CKKS encrypt + encode (CPU/NumPy uint64).
#
# Embedded from the former `encrypt_fast.py` module. These are private
# kernels; callers use the corresponding CKKSContext codec methods.
#
# Noise sampling uses the kernel CSPRNG and OpenFHE's discrete Gaussian.
# OpenFHE parameter alignment:
#   * `v` ~ TernaryUniform {-1, 0, +1} matches `OpenFHE::TernaryUniformGenerator`
#     (`src/core/include/math/ternaryuniformgenerator-impl.h`).
#   * `e0, e1` use key_gen.sample_discrete_gaussian, a vectorized port of
#     OpenFHE 1.5.1's Peikert inversion sampler.
#
# `_encrypt_cache` is keyed by `(id(ctx), num_q)` so repeated encrypts pay
# the precomputation only once.
# ===========================================================================
def _csprng_ternary(n: int) -> np.ndarray:
  """n samples uniform on {-1, 0, 1} as int64 (matches OpenFHE TernaryUniform).

  2-bit rejection sampling: each byte produces 4 candidate 2-bit values
  in {0,1,2,3}; reject 3 (yields {0,1,2}); map {0→0, 1→+1, 2→-1}.
  Acceptance rate 3/4 → ~1.34 bytes/sample expected.
  """
  out = np.empty(n, dtype=np.int64)
  filled = 0
  while filled < n:
    need = n - filled
    n_bytes = max(need * 2 // 3 + 16, 16)
    raw = np.frombuffer(os.urandom(n_bytes), dtype=np.uint8)
    cand = np.concatenate([
        raw & 3, (raw >> 2) & 3, (raw >> 4) & 3, (raw >> 6) & 3,
    ])
    accepted = cand[cand < 3].astype(np.int64)
    signed = np.where(accepted == 2, -1, accepted)
    take = min(len(signed), need)
    if take > 0:
      out[filled:filled + take] = signed[:take]
      filled += take
  return out


class _EncryptCache:
  """Per-(ctx, num_q) precomputed tables for forward NTT + encrypt."""

  def __init__(self, q_towers: List[int], psi_pairs: List[int],
               public_key, degree: int):
    self.degree = int(degree)
    self.num_q = len(q_towers)
    M = self.num_q
    N = self.degree

    self.q = np.asarray(q_towers, dtype=np.uint64)              # (M,)
    self.q_int = [int(q) for q in q_towers]

    if public_key is None:
      self.pk0 = None
      self.pk1 = None
    else:
      pk_full_0 = np.asarray(public_key[0], dtype=np.uint64)
      pk_full_1 = np.asarray(public_key[1], dtype=np.uint64)
      self.pk0 = pk_full_0[:M].T.copy()                          # (N, M)
      self.pk1 = pk_full_1[:M].T.copy()                          # (N, M)

    psi_pow = np.zeros((N, M), dtype=np.uint64)
    for m in range(M):
      qm = self.q_int[m]
      psi_m = int(psi_pairs[m])
      acc = 1
      for i in range(N):
        psi_pow[i, m] = acc
        acc = (acc * psi_m) % qm
    self.psi_pow = psi_pow

    self.twiddles: Dict[int, np.ndarray] = {}
    L = N
    while L >= 2:
      half = L // 2
      tw = np.zeros((half, M), dtype=np.uint64)
      for m in range(M):
        qm = self.q_int[m]
        omega_m = pow(int(psi_pairs[m]), 2, qm)
        w_m = pow(omega_m, N // L, qm)
        acc = 1
        for j in range(half):
          tw[j, m] = acc
          acc = (acc * w_m) % qm
      self.twiddles[L] = tw
      L //= 2


_encrypt_cache: Dict[Tuple[int, int], _EncryptCache] = {}


def _get_encrypt_cache(ctx, num_q: int) -> _EncryptCache:
  key = (id(ctx), int(num_q))
  c = _encrypt_cache.get(key)
  public_key = getattr(ctx, 'public_key', None)
  # id() keys can be reused after a context is garbage-collected, handing a
  # stale cache (wrong degree/towers/keys) to an unrelated new context.
  # Validate the cheap invariants and the key-source identity before reuse.
  if c is not None and not (
      c.degree == int(ctx.degree)
      and c.q_int == [int(q) for q in ctx.q_towers[:num_q]]
      and getattr(c, '_pk_ref', None) is public_key
  ):
    c = None
  if c is None:
    psi_pairs = [
        util.root_of_unity(2 * ctx.degree, q) for q in ctx.q_towers[:num_q]
    ]
    c = _EncryptCache(
        q_towers=list(ctx.q_towers[:num_q]),
        psi_pairs=psi_pairs,
        public_key=public_key,
        degree=ctx.degree,
    )
    c._pk_ref = public_key
    _encrypt_cache[key] = c
  return c


def _vectorized_ntt(coeffs: np.ndarray, cache: _EncryptCache) -> np.ndarray:
  """Forward negacyclic NTT, vectorized over towers.

  Input  : coeffs in coefficient form, shape (N, M) uint64.
  Output : eval form in BIT-REVERSED order.
  """
  N, M = coeffs.shape
  qb = cache.q.reshape(1, M)
  a = (coeffs * cache.psi_pow) % qb
  L = N
  while L >= 2:
    half = L // 2
    n_blocks = N // L
    a3 = a.reshape(n_blocks, L, M)
    u = a3[:, :half, :]
    v = a3[:, half:, :]
    qb3 = qb.reshape(1, 1, M)
    diff = (u + qb3 - v) % qb3
    tw = cache.twiddles[L].reshape(1, half, M)
    new_v = (diff * tw) % qb3
    new_u = (u + v) % qb3
    a = np.concatenate([new_u, new_v], axis=1).reshape(N, M)
    L //= 2
  return a


def _fast_encrypt_from_plaintext_array(
    plaintext_eval: np.ndarray, ctx, v=None, e=None, sigma: float = None,
    noise_scale_degree: int = None
) -> np.ndarray:
  """Vectorized CKKS encrypt.

  Args:
    plaintext_eval     : (N, M) uint64 in NTT eval form (bit-reversed).
    ctx                : CKKSContext.
    v                  : (M, N) Python list of NTT-form polynomial; if None
                         a fresh ternary polynomial is sampled and NTT'd.
    e                  : (e0, e1) Python lists in NTT form; if None fresh
                         Gaussian polynomials are sampled and NTT'd.
    sigma              : Gaussian std for noise sampling (when e is None).
    noise_scale_degree : ns multiplier on noise term (matches OpenFHE).

  Returns: np.uint64 array of shape (2, N, M) — c0, c1 in eval form.
  """
  N, M = plaintext_eval.shape
  cache = _get_encrypt_cache(ctx, M)
  if cache.pk0 is None or cache.pk1 is None:
    raise ValueError('Public key is required for encryption.')
  qb = cache.q.reshape(1, M)
  if sigma is None:
    sigma = float(ctx.parameters.get("sigma", 3.190000057220458984375))
  if noise_scale_degree is None:
    noise_scale_degree = ctx.parameters.get("noise_scale_degree", 1)
  ns = normalize_noise_scale_degree(noise_scale_degree)
  ns_mod_q = np.fromiter(
      (ns % int(modulus) for modulus in cache.q),
      dtype=np.uint64,
      count=M,
  ).reshape(1, M)

  if v is None:
    v_signed = _csprng_ternary(N)
    q_int64 = cache.q.astype(np.int64).reshape(1, M)
    v_rns = np.ascontiguousarray(
        np.mod(v_signed[:, None], q_int64).astype(np.uint64))
    v_eval = _vectorized_ntt(v_rns, cache)
  else:
    v_eval = np.asarray(v, dtype=np.uint64).T.copy()

  if e is None:
    e_signed_all = kg.sample_discrete_gaussian(2 * N, sigma)
    e0_signed = e_signed_all[:N]
    e1_signed = e_signed_all[N:]
    q_int64 = cache.q.astype(np.int64).reshape(1, M)
    e0_rns = np.ascontiguousarray(
        np.mod(e0_signed[:, None], q_int64).astype(np.uint64))
    e1_rns = np.ascontiguousarray(
        np.mod(e1_signed[:, None], q_int64).astype(np.uint64))
    e0_eval = _vectorized_ntt(e0_rns, cache)
    e1_eval = _vectorized_ntt(e1_rns, cache)
  else:
    e0_eval = np.asarray(e[0], dtype=np.uint64).T.copy()
    e1_eval = np.asarray(e[1], dtype=np.uint64).T.copy()

  # c0 = (v * pk0 + ns * e0 + plaintext) mod q
  c0 = (v_eval * cache.pk0) % qb
  c0 = (c0 + (ns_mod_q * e0_eval) % qb) % qb
  c0 = (c0 + plaintext_eval) % qb
  # c1 = (v * pk1 + ns * e1) mod q
  c1 = (v_eval * cache.pk1) % qb
  c1 = (c1 + (ns_mod_q * e1_eval) % qb) % qb
  return np.stack([c0, c1], axis=0)


def _validated_slots(
    slots,
    *,
    name: str,
    expected_size: int | None,
    allow_complex: bool,
) -> list[complex]:
  """Normalize a finite numeric slot vector for one encoder boundary."""
  array = np.asarray(slots)
  if array.ndim != 1:
    raise ValueError(f'{name} must be a one-dimensional slot vector.')
  if expected_size is not None and array.size != expected_size:
    raise ValueError(
        f'{name} has length {array.size}, expected exactly {expected_size}.'
    )
  if array.dtype.kind == 'c':
    if not allow_complex and np.any(np.imag(array) != 0):
      raise TypeError(
          f'{name} must be real because CROSS decoding is real-valued.'
      )
    if not allow_complex:
      array = np.real(array)
  elif array.dtype.kind not in 'biuf':
    kind = 'numeric' if allow_complex else 'real numeric'
    raise TypeError(f'{name} must contain {kind} values.')
  if not np.all(np.isfinite(array)):
    raise ValueError(f'{name} must contain only finite values.')
  return [complex(value) for value in array]


def _fast_encode(slots, ctx, scale: float = None,
                 noise_scale_degree: int = 1, *,
                 allow_complex: bool = False) -> np.ndarray:
  """Vectorized CKKS encode (slots → plaintext eval form).

  Output: (N, M) np.uint64 in NTT eval form (bit-reversed) — directly usable
  as input to the private vectorized encryption kernel.
  """
  if scale is None:
    scale = ctx.scaling_factor
  N = ctx.degree
  M = len(ctx.q_towers)
  cache = _get_encrypt_cache(ctx, M)
  m = N * 2

  # 1) inverse special FFT on slot vector
  y = _validated_slots(
      slots,
      name='slots',
      expected_size=ctx.num_slots,
      allow_complex=allow_complex,
  )
  FFTSpecialInv(y, m)

  # 2) slot -> coefficients (length N, real)
  coeffs = slot_to_coeffs(y)

  # 3) scale + log-c bookkeeping
  scaled = [scale * v for v in coeffs]
  logc = -(10**9)
  for v in scaled:
    a = abs(v)
    if a != 0.0:
      logci = int(math.ceil(math.log2(a)))
      if logc < logci:
        logc = logci
  if logc == -(10**9):
    logc = 0
  if logc < 0:
    raise ScalingFactorTooSmall("Scaling factor too small")
  max_bits_in_word = ctx.parameters.get("max_bits_in_word", 61)
  max_bits_value = ctx.parameters.get(
      "max_bits_value", (1 << 63) - (1 << 9) - 1
  )
  log_valid = logc if logc <= max_bits_in_word else max_bits_in_word
  log_approx = logc - log_valid
  approx_factor = 2.0**log_approx
  ints_base = [nearest_int(v / approx_factor) for v in scaled]
  ints_base = [x + max_bits_value if x < 0 else x for x in ints_base]

  q_int = cache.q_int
  ints_per_tower = [
      fit_to_native_vector(ints_base, max_bits_value, q_int[m_id], N)
      for m_id in range(M)
  ]
  if log_approx > 0:
    step = 1 << log_approx
    ints_per_tower = [
        [(x * step) % q_int[m_id] for x in ints_per_tower[m_id]]
        for m_id in range(M)
    ]
  if noise_scale_degree > 1:
    int_pow_p = int(round(scale))
    if int_pow_p != 1:
      power = pow(int_pow_p, noise_scale_degree - 1)
      ints_per_tower = [
          [(x * power) % q_int[m_id] for x in ints_per_tower[m_id]]
          for m_id in range(M)
      ]

  coeffs_rns = np.empty((N, M), dtype=np.uint64)
  for m_id in range(M):
    coeffs_rns[:, m_id] = np.array(ints_per_tower[m_id], dtype=np.uint64)

  return _vectorized_ntt(coeffs_rns, cache)


def _ct_to_backend_format(ciphertext: poly.Polynomial, ctx) -> poly.Polynomial:
  """Encode a standard-residue ciphertext payload into the parameter cache's
  computation format (Montgomery: x*R mod q) exactly once at the encrypt
  boundary. Identity passthrough for standard-format backends and for
  contexts without a parameter cache."""
  cache = getattr(ctx, '_param_cache', None)
  if cache is None or cache.ff_q_max.computation_format_is_standard:
    return ciphertext
  ff = cache.ff_q_max.slice(ciphertext.num_moduli)
  return ciphertext._clone_with_payload(
      jnp.asarray(
          ff.to_computation_format(
              ciphertext.polynomial.astype(jnp.uint64)),
          jnp.uint32))


def _ct_to_standard_format(ciphertext: poly.Polynomial, ctx) -> poly.Polynomial:
  """Decode a computation-format ciphertext payload back to standard residues
  exactly once at the decrypt boundary (Montgomery: strip the R factor).
  Identity passthrough for standard-format backends; the caller's ciphertext
  is never mutated."""
  cache = getattr(ctx, '_param_cache', None)
  if cache is None or cache.ff_q_max.computation_format_is_standard:
    return ciphertext
  ff = cache.ff_q_max.slice(ciphertext.num_moduli)
  return ciphertext._clone_with_payload(
      jnp.asarray(
          ff.to_original_format(
              ciphertext.polynomial.astype(jnp.uint64)),
          jnp.uint32))


def _fast_encrypt_from_plaintext(
    plaintext: poly.Polynomial, ctx, v=None, e=None, sigma: float = None,
    noise_scale_degree: int = None
) -> poly.Polynomial:
  """Encrypts a canonical, single-batch plaintext Polynomial."""
  _validate_polynomial_for_context(
      plaintext, ctx, 'plaintext', num_elements=1
  )
  if plaintext.batch != 1:
    raise NotImplementedError(
        "Fast encryption currently supports exactly one plaintext batch."
    )
  plaintext_eval = np.asarray(
      plaintext.polynomial[0, 0].reshape(plaintext.degree,
                                         plaintext.num_moduli),
      dtype=np.uint64,
  )
  ct_array = _fast_encrypt_from_plaintext_array(
      plaintext_eval, ctx, v=v, e=e, sigma=sigma,
      noise_scale_degree=noise_scale_degree
  )
  payload = jnp.asarray(ct_array, dtype=jnp.uint32).reshape(
      1, 2, *plaintext.degree_layout, plaintext.num_moduli
  )
  return _ct_to_backend_format(
      plaintext._clone_with_payload(payload, num_elements=2), ctx)


def _fast_encode_encrypt(slots, ctx, scale: float = None) -> poly.Polynomial:
  """Encode + encrypt in one shot, returning a canonical Polynomial."""
  return _fast_encode_encrypt_batch([slots], ctx, scale=scale)


def _fast_encode_encrypt_batch(
    slots_batch, ctx, scale: float = None
) -> poly.Polynomial:
  """Encode and encrypt a host batch, transferring and wrapping it once."""
  slots_batch = list(slots_batch)
  if not slots_batch:
    raise ValueError('slots_batch must contain at least one slot vector.')
  ct_array = np.stack(
      [
          _fast_encode_encrypt_array(slots, ctx, scale=scale)
          for slots in slots_batch
      ],
      axis=0,
  )
  degree_layout = getattr(ctx, 'degree_layout', None)
  if degree_layout is None:
    degree_layout = _balanced_degree_layout(ctx.degree)
  ntt_ctx = getattr(ctx, '_polynomial_ntt_ctx', None)
  cache = getattr(ctx, '_param_cache', None)
  if cache is not None:
    ntt_ctx = cache.get_sliced_ntt_q(cache.max_level)
  result = _polynomial_from_flat_array(
      ct_array, ctx.q_towers, degree_layout, ntt_ctx=ntt_ctx
  )
  # Keep the actual encoding scale on the canonical wrapper.  In particular,
  # this must not be inferred from ``ctx.output_scale``: that value commonly
  # describes a later evaluator result and may differ from the fresh
  # ciphertext's scale.
  actual_scale = ctx.scaling_factor if scale is None else scale
  result._ckks_scale = float(actual_scale)
  if getattr(ctx, '_polynomial_ntt_ctx', None) is None:
    ctx._polynomial_ntt_ctx = result.ntt_ctx
  # The fused host encryptor produces standard RNS residues.  Convert once at
  # the public ciphertext boundary so backend-specific evaluators receive the
  # representation they advertise through the injected NTT context
  # (Montgomery: x -> x*R mod q; standard backends: identity).
  return _ct_to_backend_format(result, ctx)


def _fast_encode_encrypt_array(slots, ctx, scale: float = None) -> np.ndarray:
  """Private array form for IPC and other non-HE fused regions."""
  pt_eval = _fast_encode(slots, ctx, scale=scale)
  return _fast_encrypt_from_plaintext_array(
      pt_eval,
      ctx,
      sigma=ctx.parameters.get('sigma', 3.190000057220458984375),
      noise_scale_degree=ctx.parameters.get('noise_scale_degree', 1),
  )


class _LiteCtxForFast:
  """Minimal CKKSContext-shaped object accepted by `_fast_encode` /
  `_fast_encrypt_from_plaintext_array`. Encode-only callers use this without
  constructing a full CKKSContext.

  Encode-only paths may leave `public_key` unset because `_fast_encode` uses
  only the cache's twiddles, roots, and moduli.
  """

  def __init__(self, q_towers, degree, public_key, scaling_factor=None,
               parameters=None):
    self.q_towers = list(q_towers)
    self.degree = int(degree)
    self.num_slots = self.degree // 2
    self.public_key = public_key
    self.scaling_factor = float(scaling_factor) if scaling_factor is not None \
        else 1.0
    self.parameters = dict(parameters) if parameters is not None else {}


# Reuse one lite ctx per (q_towers, degree) for encode-only callers so
# `_get_encrypt_cache` (keyed by id(ctx)) doesn't rebuild twiddles
# + psi tables on every call. Critical for BSGS-encoding which calls
# `_ckks_encode` hundreds–thousands of times during cache build.
_lite_encode_context_cache: dict = {}


def _get_or_make_encode_ctx(q_towers, degree, scale, max_bits_in_word,
                              max_bits_value):
  key = (tuple(int(q) for q in q_towers), int(degree))
  lite = _lite_encode_context_cache.get(key)
  if lite is None:
    lite = _LiteCtxForFast(
        q_towers=q_towers,
        degree=int(degree),
        public_key=None,
        scaling_factor=scale,
        parameters={
            "max_bits_in_word": int(max_bits_in_word),
            "max_bits_value": int(max_bits_value),
        },
    )
    _lite_encode_context_cache[key] = lite
  else:
    # Scale may vary call-to-call; max_bits constants almost never do.
    lite.scaling_factor = float(scale)
    lite.parameters["max_bits_in_word"] = int(max_bits_in_word)
    lite.parameters["max_bits_value"] = int(max_bits_value)
  return lite


def _ckks_encrypt_list_reference(
    plaintext: List[List[int]],
    public_key: List[List[List[int]]],
    q_towers: List[int],
    noise_scale_degree: int = 1,
    sigma=3.190000057220458984375,
    v=None,
    e=None,
):
  """Pure-Python triple-loop reference implementation — bit-exact test reference."""
  # plaintext is now (degree, moduli)
  degree = len(plaintext)
  num_towers = len(q_towers)
  psi = None
  if v is None:
    psi = [
        util.root_of_unity(int(2 * degree), q_towers[t_id])
        for t_id in range(num_towers)
    ]
    v = kg.gen_ternary_uniform_polynomial(degree, q_towers).coeffs
    v = [
        util.bit_reverse_array(
            util.ntt_negacyclic_bit_reverse(v[t_id], q_towers[t_id], psi[t_id])
        )
        for t_id in range(num_towers)
    ]
  if e is None:
    # This noise is too large! I need to copy the OpenFHE's implementation to fix it
    psi = [
        util.root_of_unity(int(2 * degree), q_towers[t_id])
        for t_id in range(num_towers)
    ]
    e0 = kg.gen_gaussian_polynomial(degree, q_towers, sigma=sigma).coeffs
    e0 = [
        util.bit_reverse_array(
            util.ntt_negacyclic_bit_reverse(
                (e0[t_id]), q_towers[t_id], psi[t_id]
            )
        )
        for t_id in range(num_towers)
    ]
    e1 = kg.gen_gaussian_polynomial(degree, q_towers, sigma=sigma).coeffs
    e1 = [
        util.bit_reverse_array(
            util.ntt_negacyclic_bit_reverse(
                (e1[t_id]), q_towers[t_id], psi[t_id]
            )
        )
        for t_id in range(num_towers)
    ]
  else:
    e0, e1 = e[0], e[1]
  ns = noise_scale_degree
  if len(plaintext[0]) < len(public_key[0]):
    diff_length = len(public_key[0]) - len(plaintext[0])
    public_key[0] = public_key[0][:-diff_length]
    public_key[1] = public_key[1][:-diff_length]

  # Prepare c0, c1 accumulators with shape (degree, moduli)
  c0 = [[0] * num_towers for _ in range(degree)]
  c1 = [[0] * num_towers for _ in range(degree)]

  # We need to iterate carefully.
  # v is (moduli, degree)
  # public_key is (2, moduli, degree) -> public_key[0], public_key[1] are (moduli, degree)
  # e0, e1 are (moduli, degree)
  # But we want output (degree, moduli)

  for i in range(degree):
    # plaintext[i] is [m0, m1, ..., mk] corresponding to degree i
    p_i_moduli = plaintext[i]

    for j in range(num_towers):
      q_j = q_towers[j]
      v_ji = v[j][i]
      pk0_ji = public_key[0][j][i]
      e0_ji = e0[j][i]

      val0 = (v_ji * pk0_ji + ns * e0_ji) % q_j

      # Add plaintext
      val0 = (val0 + p_i_moduli[j]) % q_j
      c0[i][j] = val0

      pk1_ji = public_key[1][j][i]
      e1_ji = e1[j][i]

      val1 = (v_ji * pk1_ji + ns * e1_ji) % q_j
      c1[i][j] = val1

  return [c0, c1]


def _ckks_decrypt_list_reference(
    ciphertext: List[List[List[int]]],
    private_key: List[List[int]],
    q_towers: List[int],
):
  # ciphertext is list of elements. element 0 is c0, etc.
  # each element is (degree, moduli)
  num_elements = len(ciphertext)
  degree = len(ciphertext[0])
  num_towers = len(ciphertext[0][0])

  s = private_key  # (moduli, degree)

  if num_towers < len(s):
    diff_length = len(s) - num_towers
    s = s[:-diff_length]

  # Pre-transpose s for easier access or just index carefully
  # s is (moduli, degree)

  # We want to compute: M(X) = c0 + c1*s + ...
  # Result should be (degree, moduli) initially before NTT/CRT?
  # Actually decrypt returns coefficients.

  # Let's accumulate in (moduli, degree) for the final NTT part which expects that layout usually,
  # OR we adapt the rest of the function.
  # The original returned `first_element_coef` which was (moduli, degree).
  # But we want "ciphertext/plaintext in the layout of (degree, moduli)".
  # So we should probably return (degree, moduli).

  # Let's accumulate in (degree, moduli).

  res_poly = [[0] * num_towers for _ in range(degree)]

  # s_power starts as s^1. s is (moduli, degree).
  # We need s^k in (moduli, degree).

  cur_s_power = [list(row) for row in s]  # Copy s

  # c0
  for d in range(degree):
    for m in range(num_towers):
      res_poly[d][m] = ciphertext[0][d][m]

  for i in range(1, num_elements):
    ci = ciphertext[i]  # (degree, moduli)

    for d in range(degree):
      for m in range(num_towers):
        # + ci * s^i
        term = (ci[d][m] * cur_s_power[m][d]) % q_towers[m]
        res_poly[d][m] = (res_poly[d][m] + term) % q_towers[m]

    if i < num_elements - 1:
      # Update s_power to s^(i+1)
      # s^(i+1) = s^i * s
      new_s_power = [[0] * degree for _ in range(num_towers)]
      for m in range(num_towers):
        qi = q_towers[m]
        for d in range(degree):
          new_s_power[m][d] = (cur_s_power[m][d] * s[m][d]) % qi
      cur_s_power = new_s_power

  # Now res_poly is (degree, moduli)
  # We need to do inverse NTT.
  # Existing utils utilize (moduli, degree) usually?
  # util.bit_reverse_array takes 1D list.
  # util.intt_negacyclic_bit_reverse takes 1D list.

  # So we can process row by row if we transpose or col by col.
  # The original returned `first_element_coef` as list of lists (moduli, degree).
  # We want to return (degree, moduli).

  final_res = [[0] * num_towers for _ in range(degree)]

  for m in range(num_towers):
    # Extract column m
    col = [res_poly[d][m] for d in range(degree)]

    # bit reverse
    col_rev = util.bit_reverse_array(col)

    # intt
    coef = util.intt_negacyclic_bit_reverse(
        col_rev, q_towers[m], util.root_of_unity(2 * degree, q_towers[m])
    )

    for d in range(degree):
      final_res[d][m] = coef[d]

  return final_res


def _ckks_encode(
    slots: List[complex],
    cycl_order: int,
    q_towers: List[int],
    p_towers: List[int],
    scale: float,
    noise_scale_degree: int = 1,
    max_bits_in_word: int = 61,
    max_bits_value: int = (1 << 63) - (1 << 9) - 1,
) -> List[List[int]]:
  """Private fast CKKS encode kernel.

  Same signature + output format (list-of-list of ints, shape (degree,
  moduli)) as the historical pure-Python implementation, but routes to
  the vectorized NumPy path. `p_towers` is accepted for backward-compat
  but unused (encode emits Q-tower residues only — same as the legacy
  reference). Bit-exact versus `_ckks_encode_fall_back` for the same inputs.
  """
  degree = cycl_order // 2
  # Reuse one lite ctx per (q_towers, degree) so the per-ctx twiddle/psi
  # cache doesn't rebuild on every call. Encode path does NOT read
  # pk0/pk1 from the cache — only twiddles + psi_pow + q.
  lite = _get_or_make_encode_ctx(
      q_towers, degree, scale, max_bits_in_word, max_bits_value)
  arr = _fast_encode(
      slots,
      lite,
      scale=scale,
      noise_scale_degree=noise_scale_degree,
      allow_complex=True,
  )  # (N, M) uint64
  return arr.tolist()


def _ckks_encode_fall_back(
    slots: List[complex],
    cycl_order: int,
    q_towers: List[int],
    p_towers: List[int],
    scale: float,
    noise_scale_degree: int = 1,
    max_bits_in_word: int = 61,
    max_bits_value: int = (1 << 63) - (1 << 9) - 1,
) -> List[List[int]]:
  """Pure-Python reference implementation — kept for bit-exact testing.

  Encodes slots to DCRTPoly EVAL form with given (Q,P) towers, NATIVE_INT=64.
  Returns dict with residues for Q and P towers and the scaled integer coeffs.
  """
  nh = len(slots)
  N = 2 * nh
  m = cycl_order
  if m != 4 * nh:
    raise ValueError("cycl_order must be 4*Nh for CKKS special FFT size")

  # 1) inverse special FFT
  y = list(slots)
  FFTSpecialInv(y, m)

  # 2) slot->coeff packing
  coeffs = slot_to_coeffs(y)  # length N

  # 3) scale and determine bit length like OpenFHE (NATIVEINT==64)
  #    Find logc = ceil(log2(max(|scaled_real|, |scaled_imag|))) across slots
  scaled_vals = [scale * v for v in coeffs]

  logc = -(10**9)
  for v in scaled_vals:
    absv = abs(v)
    if absv != 0.0:
      logci = int(math.ceil(math.log2(absv)))
      if logc < logci:
        logc = logci
  if logc == -(10**9):
    logc = 0
  if logc < 0:
    raise ScalingFactorTooSmall("Scaling factor too small")
  # 4) approxFactor to keep values within 60-bit word, then quantize
  log_valid = logc if logc <= max_bits_in_word else max_bits_in_word
  log_approx = logc - log_valid
  approx_factor = 2.0**log_approx
  ints_base = [nearest_int(v / approx_factor) for v in scaled_vals]
  ints_base = [x + max_bits_value if x < 0 else x for x in ints_base]

  elements = [
      fit_to_native_vector(ints_base, max_bits_value, q_tower, N)
      for q_tower in q_towers
  ]

  # 5) Scale back up by approx_factor (power of two) in the ring
  if log_approx > 0:
    step = 1 << log_approx
    ints = [
        [x * step % q_towers[mod_id] for x in elements[mod_id]]
        for mod_id in range(len(q_towers))
    ]
  else:
    ints = elements

  # 6) If noise scale degree > 1, multiply by round(scale)^(d-1)
  if noise_scale_degree > 1:
    int_pow_p = int(round(scale))
    if int_pow_p != 1:
      power = pow(int_pow_p, noise_scale_degree - 1)
      for mod_id in range(len(q_towers)):
        ints[mod_id] = [x * power % q_towers[mod_id] for x in ints[mod_id]]

  # 4) residues per tower
  Q_res = [
      util.ntt_negacyclic_bit_reverse(
          ints[mod_id],
          q_towers[mod_id],
          util.root_of_unity(m, q_towers[mod_id]),
      )
      for mod_id in range(len(q_towers))
  ]
  # Current Q_res is (moduli, degree)
  # We want to return (degree, moduli)

  Q_res_T = [[0] * len(q_towers) for _ in range(N)]
  for mod_id in range(len(q_towers)):
    rev = util.bit_reverse_array(Q_res[mod_id])
    for deg in range(N):
      Q_res_T[deg][mod_id] = rev[deg]

  return Q_res_T


def _ckks_decode(
    plaintext: List[int],
    scaling_factor: float,
    slots: int,
    q: int,
    p: int,
    CKKS_M_FACTOR: int = 1,
    ADD_NOISE: bool = False,
    validate_approximation: bool = True,
):
  # Ported from notebook implementation
  degree = len(plaintext)
  q_half = q >> 1
  Nh = degree // 2
  gap = Nh // slots
  powP_positive = pow(2, p)
  powP = pow(2, -p)

  # Step 1: scale back to intermediate complex vector m(X)
  sf_pre = (1.0 / scaling_factor) * powP_positive

  real_part_list = []
  imag_part_list = []
  for i in range(slots):
    # real part from first half
    r_val = plaintext[i * gap]
    if r_val > q_half:
      real_part = -((q - r_val) * sf_pre)
    else:
      real_part = r_val * sf_pre
    real_part_list.append(int(real_part))

    # imag part from second half
    im_val = plaintext[i * gap + Nh]
    if im_val > q_half:
      imag_part = -((q - im_val) * sf_pre)
    else:
      imag_part = im_val * sf_pre
    imag_part_list.append(int(imag_part))

  curValues = [
      complex(real_part_list[i], imag_part_list[i]) for i in range(slots)
  ]

  # Step 2: conjugate vector and estimated error, through the shared decode
  # math. ``_decode_conjugate`` / ``_decode_approximation_stddev`` are the
  # vectorized form of the OpenFHE logic this path used to inline; both take
  # whatever scale their input carries, so the ``2 ** p`` working scale here is
  # normalized away at the guard below.
  values = np.asarray(curValues, dtype=np.complex128)
  conjugate = _decode_conjugate(values)

  stddev_dbl = _decode_approximation_stddev(values, conjugate)
  # ``_stddev`` reports at the ``2 ** p`` working scale used above; the shared
  # policy takes unit scale. ``stddev_dbl > 0`` reproduces the historical
  # ``log2`` guard, which silently passed non-positive and NaN estimates.
  if stddev_dbl > 0 and validate_approximation and _decode_stddev_check_enabled:
    _check_decode_stddev(math.ldexp(stddev_dbl, -p))
  if stddev_dbl < 0.125 * math.sqrt(degree):
    stddev_dbl = 0.125 * math.sqrt(degree)

  stddev = math.sqrt(CKKS_M_FACTOR + 1) * stddev_dbl
  scale = 0.5 * powP

  # Optional approximation-error perturbation; this is not a circuit-privacy
  # or noise-flooding security mechanism.
  rng = random.Random()

  def _gauss():
    return rng.gauss(0.0, stddev)

  if ADD_NOISE:
    curValues = [
        complex(
            real_part_list[i] * scale
            + conjugate[i].real * scale
            + powP * _gauss(),
            imag_part_list[i] * scale
            + conjugate[i].imag * scale
            + powP * _gauss(),
        )
        for i in range(slots)
    ]
  else:
    curValues = [
        complex(
            real_part_list[i] * scale + conjugate[i].real * scale,
            imag_part_list[i] * scale + conjugate[i].imag * scale,
        )
        for i in range(slots)
    ]

  # Step 3: Special forward FFT to slot values
  FFTSpecial(curValues, degree * 2)
  curValues = [complex(curValues[i].real, 0.0) for i in range(slots)]
  # Return real parts only
  return curValues


# ===========================================================================
# Vectorized CKKS decrypt + decode (CPU/NumPy uint64).
#
# Embedded from the former `decrypt_fast.py` module. Replaces the pure-Python
# triple-loop SK multiply and per-tower INTT in the list reference with NumPy
# uint64 vectorized ops; decode (CRT + FFT) is also vectorized.
#
# Exposes:
# These private kernels implement the context-owned decrypt/decode methods.
#
# `_decrypt_cache` is keyed by `(id(ctx), num_moduli)` so repeated decrypts
# pay the precomputation only once.
# ===========================================================================
def _bit_reverse_indices_np(n: int) -> np.ndarray:
  bits = n.bit_length() - 1
  idx = np.arange(n, dtype=np.uint64)
  rev = np.zeros(n, dtype=np.uint64)
  for i in range(bits):
    rev |= ((idx >> i) & 1) << (bits - 1 - i)
  return rev.astype(np.int64)


class _DecryptCache:
  """Per-(ctx, num_moduli) precomputed tables."""

  def __init__(self, q_towers: List[int], psi_pairs: List[int],
               secret_key: List[List[int]], degree: int):
    self.degree = int(degree)
    self.num_moduli = len(q_towers)
    M = self.num_moduli
    N = self.degree

    self.q = np.asarray(q_towers, dtype=np.uint64)              # (M,)
    self.q_int = [int(q) for q in q_towers]
    self.s = np.asarray(secret_key, dtype=np.uint64).T.copy()    # (N, M)

    # n^(-1) mod q  per tower
    self.n_inv = np.asarray(
        [pow(N, -1, q) for q in self.q_int], dtype=np.uint64
    )                                                            # (M,)

    # psi^(-i) mod q  per tower:  shape (N, M)
    psi_inv = [
        pow(int(psi), -1, q)
        for psi, q in zip(psi_pairs, self.q_int, strict=True)
    ]
    psi_inv_pow = np.zeros((N, M), dtype=np.uint64)
    for m in range(M):
      acc = 1
      qm = self.q_int[m]
      psi_im = psi_inv[m]
      for i in range(N):
        psi_inv_pow[i, m] = acc
        acc = (acc * psi_im) % qm
    self.psi_inv_pow = psi_inv_pow

    # GS twiddles per iteration length L = N, N/2, ..., 2.
    # For each L, half = L//2 entries: w_m = inv_omega^(N/L) mod q,
    # column j = w_m^j mod q.
    self.twiddles: Dict[int, np.ndarray] = {}
    for m in range(M):
      qm = self.q_int[m]
      omega_m = pow(int(psi_pairs[m]), 2, qm)         # primitive N-th root
      inv_omega = pow(omega_m, -1, qm)
    # Build per-tower per-length tables; assemble into (half, M) arrays.
    L = N
    while L >= 2:
      half = L // 2
      tw = np.zeros((half, M), dtype=np.uint64)
      for m in range(M):
        qm = self.q_int[m]
        inv_omega = pow(pow(int(psi_pairs[m]), 2, qm), -1, qm)
        w_m = pow(inv_omega, N // L, qm)
        acc = 1
        for j in range(half):
          tw[j, m] = acc
          acc = (acc * w_m) % qm
      self.twiddles[L] = tw
      L //= 2

    self.bit_rev_idx = _bit_reverse_indices_np(N)


_decrypt_cache: Dict[Tuple[int, int], _DecryptCache] = {}


def evict_context_caches(ctx) -> int:
  """Drop the encrypt/decrypt cache entries built for ``ctx``.

  Both caches are keyed by ``id(ctx)``, so entries outlive the context they
  were built for and would keep its device tables alive (or, once the id is
  reused, serve a stale table to a new context). ``Mapping.release`` calls
  this; returns the number of entries removed.
  """
  key_id = id(ctx)
  removed = 0
  for cache in (_encrypt_cache, _decrypt_cache):
    for key in [k for k in cache if k[0] == key_id]:
      cache.pop(key, None)
      removed += 1
  return removed


def _get_decrypt_cache(ctx, num_moduli: int) -> _DecryptCache:
  key = (id(ctx), int(num_moduli))
  c = _decrypt_cache.get(key)
  # Guard against id() reuse (see _get_encrypt_cache): a stale entry here
  # would silently decrypt with the wrong secret key.
  if c is not None and not (
      c.degree == int(ctx.degree)
      and c.q_int == [int(q) for q in ctx.q_towers[:num_moduli]]
      and getattr(c, '_sk_ref', None) is ctx.secret_key
  ):
    c = None
  if c is None:
    psi_pairs = [
        util.root_of_unity(2 * ctx.degree, q) for q in ctx.q_towers[:num_moduli]
    ]
    c = _DecryptCache(
        q_towers=list(ctx.q_towers[:num_moduli]),
        psi_pairs=psi_pairs,
        secret_key=[list(row) for row in ctx.secret_key[:num_moduli]],
        degree=ctx.degree,
    )
    c._sk_ref = ctx.secret_key
    _decrypt_cache[key] = c
  return c


def _vectorized_intt(eval_arr: np.ndarray, cache: _DecryptCache) -> np.ndarray:
  """Vectorized inverse negacyclic NTT.

  Input  : eval_arr in bit-reversed eval form, shape (N, M) uint64.
  Output : coefficient form, shape (N, M) uint64.
  """
  N = cache.degree
  M = cache.num_moduli
  q = cache.q.reshape(1, M)

  # Step 1: bit-reverse so subsequent GS sees natural order
  a = eval_arr[cache.bit_rev_idx, :]

  # Step 2: Gentleman-Sande DIF butterflies, length L = N down to 2
  L = N
  while L >= 2:
    half = L // 2
    n_blocks = N // L
    a3 = a.reshape(n_blocks, L, M)
    u = a3[:, :half, :]
    v = a3[:, half:, :]
    qb = q.reshape(1, 1, M)
    new_u = (u + v) % qb
    diff = (u + qb - v) % qb                # in [0, q)
    tw = cache.twiddles[L].reshape(1, half, M)
    new_v = (diff * tw) % qb
    a = np.concatenate([new_u, new_v], axis=1).reshape(N, M)
    L //= 2

  # Step 3: bit-reverse back to natural
  a = a[cache.bit_rev_idx, :]

  # Step 4: divide by n
  a = (a * cache.n_inv.reshape(1, M)) % q

  # Step 5: negacyclic post-twist by psi^(-i)
  a = (a * cache.psi_inv_pow) % q
  return a


def _fast_decrypt_to_rns_coeffs_array(
    ct_polynomial: np.ndarray, ctx
) -> np.ndarray:
  """Vectorized SK multiply + INTT.

  Args:
    ct_polynomial: shape (num_elements, N, M) uint32/uint64. CKKS ciphertext
      towers in bit-reversed eval form.
    ctx: CKKSContext (uses ctx.secret_key, ctx.q_towers, ctx.degree).

  Returns:
    np.uint64 array of shape (N, M) — RNS coefficients of M(X).
  """
  ct = np.asarray(ct_polynomial, dtype=np.uint64)
  num_elements, N, M = ct.shape
  cache = _get_decrypt_cache(ctx, M)

  q = cache.q.reshape(1, M)
  s = cache.s

  # Horner-style: m = c0 + c1*s + c2*s^2 + ...   (mod q per tower, eval form)
  m = ct[0].copy()
  if num_elements > 1:
    s_pow = s.copy()
    for i in range(1, num_elements):
      term = (ct[i] * s_pow) % q
      m = (m + term) % q
      if i < num_elements - 1:
        s_pow = (s_pow * s) % q

  return _vectorized_intt(m, cache)


def _crt_combine_rns(residues, moduli: List[int]) -> List[int]:
  """CRT combine per-tower residues into big-int coefficients.

  Args:
    residues: rows of per-tower residues, one row per polynomial coefficient.
      Accepts a ``(degree, num_moduli)`` array -- the fast decrypt path, where
      ``tolist()`` is the single host pull -- or an equivalent list of rows,
      which is what the plaintext decode path holds.
    moduli: the pairwise-coprime tower moduli, one per column.

  Returns:
    Python list of big ints, one per coefficient, reduced modulo the product.
  """
  # Coerce first: a numpy integer here would silently overflow the product.
  moduli = [int(modulus) for modulus in moduli]
  big = 1
  for modulus in moduli:
    big *= modulus
  weights = []
  for modulus in moduli:
    cofactor = big // modulus
    weights.append(cofactor * pow(cofactor, -1, modulus))

  rows = residues.tolist() if hasattr(residues, 'tolist') else residues
  out = []
  for row in rows:
    combined = 0
    for residue, modulus, weight in zip(
        row, moduli, weights, strict=True
    ):
      combined += (int(residue) % modulus) * weight
    out.append(combined % big)
  return out


def _decode_conjugate(values: np.ndarray) -> np.ndarray:
  """Return the CKKS conjugate-symmetry counterpart of coefficient slots."""
  conjugate = np.empty(values.size, dtype=np.complex128)
  conjugate[0] = complex(values[0].real, -values[0].imag)
  if values.size > 1:
    reversed_values = values[
        values.size - 1 - np.arange(0, values.size - 1)
    ]
    conjugate[1:] = (
        -reversed_values.imag - 1j * reversed_values.real
    )
  return conjugate


def _decode_approximation_stddev(
    values: np.ndarray, conjugate: np.ndarray
) -> float:
  """Mirror OpenFHE's conjugate-symmetry error estimate at unit scale."""
  slots = values.size
  if slots == 1:
    return abs(float(values[0].imag))
  complex_values = values[: slots // 2 + 1] - conjugate[: slots // 2 + 1]
  mean = 2.0 * float(np.sum(
      complex_values[1 : slots // 2].real
      + complex_values[1 : slots // 2].imag
  ))
  mean += float(complex_values[0].imag)
  mean += 2.0 * float(complex_values[slots // 2].real)
  mean /= 2 * slots - 1.0
  variance = 2.0 * float(np.sum(
      (complex_values[1 : slots // 2].real - mean) ** 2
      + (complex_values[1 : slots // 2].imag - mean) ** 2
  ))
  variance += float((complex_values[0].imag - mean) ** 2)
  variance += 2.0 * float(
      (complex_values[slots // 2].real - mean) ** 2
  )
  variance /= 2 * slots - 2.0
  return 0.5 * math.sqrt(variance)


def _check_decode_stddev(stddev: float) -> None:
  """Fail closed on a decode whose approximation error is too high.

  ``stddev`` is the conjugate-symmetry error estimate at unit (message) scale.
  This is the single decode-precision policy: the reference decoder rescales
  its own ``2 ** p`` estimate before calling in, and the vectorized decoder
  already works at unit scale.
  """
  if not math.isfinite(stddev) or stddev > _DECODE_STDDEV_LIMIT:
    raise ApproximationErrorTooHigh(
        'CKKS decryption failed because the approximation error is too high; '
        f'estimated stddev={stddev:.3e}, '
        f'limit={_DECODE_STDDEV_LIMIT:.3e}. Check the '
        'ring parameters, scale, and circuit depth.'
    )


def _validate_decode_approximation(
    values: np.ndarray, conjugate: np.ndarray
) -> None:
  _check_decode_stddev(_decode_approximation_stddev(values, conjugate))


def _fast_decode(
    coef_rns: np.ndarray,
    ctx,
    scale: float,
    slots_to_decode: int = None,
    *,
    validate_approximation: bool = True,
) -> np.ndarray:
  """CRT combine + CKKS FFT to slot values.

  Args:
    coef_rns        : (N, M) uint64 from `_fast_decrypt_to_rns_coeffs`.
    ctx             : CKKSContext.
    scale           : output scaling factor (typically ctx.output_scale).
    slots_to_decode : how many slot values to return (default = ctx.num_slots).
                      Decoding always runs on all slots; this only trims output.
  """
  N, M = coef_rns.shape
  q_int = [int(q) for q in ctx.q_towers[:M]]
  Big = 1
  for q in q_int:
    Big *= q
  Big_half = Big >> 1
  num_slots = ctx.num_slots
  slots_out = num_slots if slots_to_decode is None else min(
      slots_to_decode, num_slots
  )

  # CRT combine — returns N big ints
  combined = _crt_combine_rns(coef_rns, q_int)

  # Convert to signed reals (centered around 0), divide by scale.
  # Must process all num_slots positions (FFTSpecial needs full slot vector).
  Nh = N // 2
  gap = Nh // num_slots
  scale_inv = 1.0 / float(scale)
  reals = np.empty(num_slots, dtype=np.float64)
  imags = np.empty(num_slots, dtype=np.float64)
  for i in range(num_slots):
    r = combined[i * gap]
    if r > Big_half:
      r -= Big
    reals[i] = float(r) * scale_inv
    im = combined[i * gap + Nh]
    if im > Big_half:
      im -= Big
    imags[i] = float(im) * scale_inv

  cur = reals + 1j * imags

  if not isinstance(validate_approximation, bool):
    raise TypeError('validate_approximation must be a bool.')
  # Mirror `_ckks_decode`'s conjugate + average step.
  conj = _decode_conjugate(cur)
  if validate_approximation:
    _validate_decode_approximation(cur, conj)
  cur = 0.5 * (cur + conj)

  # CKKS special FFT (forward).
  buf = list(cur)
  FFTSpecial(buf, N * 2)
  arr = np.array([z.real for z in buf[:slots_out]], dtype=np.float64)
  return arr


def _ct_batch_to_host_array(
    ciphertext: poly.Polynomial, ctx, *, require_single_batch: bool = False
) -> np.ndarray:
  """Validate one ciphertext batch and pull it to the host in one transfer.

  The single Polynomial-to-host boundary behind every fast decrypt entry point.
  It validates the ciphertext against the context, strips the computation-format
  (Montgomery) factor, and materializes the whole batch with one device read --
  which is why the batched decoder is not a loop over the single-item one.

  Args:
    ciphertext: canonical two-element CKKS ciphertext Polynomial.
    ctx: CKKSContext the ciphertext must match.
    require_single_batch: reject a batched payload, for callers whose contract
      returns exactly one slot vector.

  Returns:
    uint64 array of shape (batch, num_elements, degree, num_moduli).
  """
  _validate_polynomial_for_context(
      ciphertext, ctx, 'ciphertext', num_elements=2
  )
  if require_single_batch and ciphertext.batch != 1:
    raise NotImplementedError(
        "Fast decryption currently supports exactly one ciphertext batch."
    )
  ciphertext = _ct_to_standard_format(ciphertext, ctx)
  return np.asarray(
      ciphertext.polynomial.reshape(
          ciphertext.batch,
          ciphertext.num_elements,
          ciphertext.degree,
          ciphertext.num_moduli,
      ),
      dtype=np.uint64,
  )


def _fast_decrypt_to_rns_coeffs(
    ciphertext: poly.Polynomial, ctx
) -> np.ndarray:
  """Decrypts a canonical, single-batch ciphertext to flat RNS coefficients."""
  ct_batch = _ct_batch_to_host_array(
      ciphertext, ctx, require_single_batch=True
  )
  return _fast_decrypt_to_rns_coeffs_array(ct_batch[0], ctx)


def _fast_decrypt_decode(
    ciphertext: poly.Polynomial,
    ctx,
    scale: float,
    slots_to_decode: int = None,
    *,
    validate_approximation: bool = True,
    require_single_batch: bool = False,
) -> np.ndarray:
  """End-to-end vectorized decrypt + decode for one ciphertext batch.

  Always returns a ``(batch, slots)`` float64 matrix, whatever the batch size:
  the batch dimension belongs in the shape, not in the return type. Callers
  whose own contract is a single slot vector index ``[0]`` and pass
  ``require_single_batch=True`` so a batched payload is rejected rather than
  silently truncated.

  The whole batch is pulled to the host in one transfer before any decoding
  starts, which is why this is not a loop over a single-ciphertext entry point.
  """
  return np.stack([
      _fast_decode(
          _fast_decrypt_to_rns_coeffs_array(ct_array, ctx),
          ctx,
          scale,
          slots_to_decode=slots_to_decode,
          validate_approximation=validate_approximation,
      )
      for ct_array in _ct_batch_to_host_array(
          ciphertext, ctx, require_single_batch=require_single_batch
      )
  ])


def _polynomial_from_flat_array(
    array: np.ndarray,
    moduli: List[int],
    degree_layout: tuple[int, int],
    ntt_ctx=None,
) -> poly.Polynomial:
  """Wraps local flat arithmetic output in the canonical rank-5 layout."""
  # All public CKKS Polynomial payloads use the 32-bit RNS representation.
  # The host-side fast kernels compute in uint64, but every residue is below
  # its 32-bit modulus at this boundary.
  payload = jnp.asarray(array, dtype=jnp.uint32)
  if payload.ndim != 4:
    raise ValueError(
        '_polynomial_from_flat_array expects shape '
        '(batch, num_elements, degree, num_moduli); '
        f'got rank {payload.ndim} shape {payload.shape}.'
    )
  batch, num_elements, degree, num_moduli = payload.shape
  if degree_layout[0] * degree_layout[1] != degree:
    raise ValueError(
        f"degree_layout {degree_layout} does not match degree {degree}."
    )
  shapes = {
      "batch": batch,
      "num_elements": num_elements,
      "num_moduli": num_moduli,
      "degree": degree,
      "precision": 32,
      "degree_layout": degree_layout,
  }
  parameters = {"moduli": list(moduli)}
  if ntt_ctx is not None:
    parameters['ntt_ctx'] = ntt_ctx
  return poly.Polynomial.from_array(
      payload.reshape(batch, num_elements, *degree_layout, num_moduli),
      shapes,
      parameters=parameters,
  )


def _validate_polynomial_for_context(
    value, ctx, name: str, num_elements: int | None = None
) -> poly.Polynomial:
  """Validate a Polynomial at an encrypt/decrypt context boundary."""
  if not isinstance(value, poly.Polynomial):
    raise TypeError(f'{name} must be a Polynomial.')
  value.validate()
  expected_dtype = jnp.dtype(value.modulus_dtype)
  if value.precision != 32 or expected_dtype != jnp.dtype(jnp.uint32):
    raise ValueError(
        f'{name} must use the canonical precision=32/uint32 Polynomial '
        f'representation; got precision={value.precision} and '
        f'modulus dtype {expected_dtype}.'
    )
  if value.polynomial.dtype != expected_dtype:
    raise ValueError(
        f'{name} payload dtype {value.polynomial.dtype} does not match its '
        f'{expected_dtype} ciphertext representation.'
    )
  if value.degree != ctx.degree:
    raise ValueError(
        f'{name} degree {value.degree} does not match context degree '
        f'{ctx.degree}.'
    )
  if num_elements is not None and value.num_elements != num_elements:
    raise ValueError(
        f'{name} must contain {num_elements} element(s), got '
        f'{value.num_elements}.'
    )
  expected_moduli = tuple(ctx.q_towers[:value.num_moduli])
  if tuple(value.moduli) != expected_moduli:
    raise ValueError(
        f'{name} moduli {tuple(value.moduli)} do not match context prefix '
        f'{expected_moduli}.'
    )
  ctx_layout = getattr(ctx, 'degree_layout', None)
  if ctx_layout is not None and tuple(value.degree_layout) != tuple(ctx_layout):
    raise ValueError(
        f'{name} layout {value.degree_layout} does not match context layout '
        f'{tuple(ctx_layout)}.'
    )
  return value


class _BSGSMatVecAccessor:
  """Provides the two supported ``ctx.bsgs_matvec[...]`` index forms.

  Returns a fresh private implementation configured at the requested level,
  dimension, and optional BSGS factors. A fresh instance is returned per
  access because matrix encoding mutates per-instance offline state. ``bsgs``
  is imported lazily to avoid the module cycle.
  """

  def __init__(self, ctx: "CKKSContext"):
    self.ctx = ctx

  def __getitem__(self, key):
    import bsgs  # lazy: breaks the import cycle
    if not isinstance(key, tuple):
      raise TypeError(
          'ctx.bsgs_matvec expects (level, n) or (level, n, n1, n2).'
      )
    if len(key) == 2:
      level, n = key
      n1, n2 = bsgs.compute_bsgs_params(n)
    elif len(key) == 4:
      level, n, n1, n2 = key
    else:
      raise ValueError(
          'ctx.bsgs_matvec expects (level, n) or (level, n, n1, n2); '
          f'got {len(key)} values.'
      )
    return bsgs._BSGSMatVecAtLevel(
        self.ctx, level=level, n=n, n1=n1, n2=n2
    )


# =============================================================================
# BEGIN: PRIVATE MAPPING BACKEND
#
# This block contains the low-level analysis and JAX materialization used by
# the network-level Mapping class. CKKSContext is only the cryptographic
# resource/evaluator facade; it owns no model or compilation lifecycle.
# =============================================================================

# -- Abstract planning -------------------------------------------------------


# Packing deliberately stores operators as plain immutable tuples rather than
# introducing another operation class. These accessors keep tuple layout local
# to this backend.


def _jsonable(value):
  if is_dataclass(value):
    return {
        field.name: _jsonable(getattr(value, field.name))
        for field in fields(value)
    }
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, tuple):
    return [_jsonable(item) for item in value]
  if isinstance(value, list):
    return [_jsonable(item) for item in value]
  if isinstance(value, dict):
    return {
        str(key): _jsonable(item)
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  return repr(value)


def _json_digest(value) -> str:
  data = json.dumps(_jsonable(value), sort_keys=True, separators=(',', ':'))
  return hashlib.sha256(data.encode()).hexdigest()


def _context_fingerprint(ctx, *, perf_test: bool = False) -> str:
  parameters = ctx.parameters
  return _json_digest({
      'degree': int(ctx.degree),
      'num_slots': int(ctx.num_slots),
      'q_towers': tuple(int(q) for q in ctx.q_towers),
      'p_towers': tuple(int(p) for p in ctx.p_towers),
      'composite_degree': int(ctx.composite_degree),
      'batch': int(ctx.batch),
      'degree_layout': tuple(int(x) for x in ctx.degree_layout),
      'dnum': None if ctx.dnum is None else int(ctx.dnum),
      'scaling_factor': float(ctx.scaling_factor),
      'output_scale': float(ctx.output_scale),
      'noise_std': _normalize_noise_std(parameters.get('sigma', sigma)),
      'noise_scale_degree': normalize_noise_scale_degree(
          parameters.get('noise_scale_degree', 1)
      ),
      'key_generation_version': int(kg.KEY_GENERATION_VERSION),
      'max_bits_in_word': int(parameters.get('max_bits_in_word', 61)),
      'max_bits_value': int(
          parameters.get('max_bits_value', (1 << 63) - (1 << 9) - 1)
      ),
      'ckks_m_factor': parameters.get('CKKS_M_FACTOR', 1),
      'perf_test': bool(perf_test),
  })


def _max_level(ctx) -> int:
  return (len(ctx.q_towers) - 1) // int(ctx.composite_degree)


def _moduli_at_level(ctx, level: int) -> tuple[int, ...]:
  max_level = _max_level(ctx)
  if not 0 <= level <= max_level:
    raise ValueError(f'level {level} outside [0, {max_level}].')
  count = len(ctx.q_towers) - (max_level - level) * ctx.composite_degree
  return tuple(int(q) for q in ctx.q_towers[:count])


def _validate_context(ctx) -> None:
  for name in ('degree', 'num_slots', 'composite_degree', 'batch'):
    value = getattr(ctx, name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
      raise ValueError(f'context {name} must be a positive int.')
  if ctx.dnum is None:
    raise ValueError(
        'dnum is required to map a network; pass Mapping(..., dnum=...).'
    )
  if isinstance(ctx.dnum, bool) or not isinstance(ctx.dnum, int) or ctx.dnum <= 0:
    raise ValueError('context dnum must be a positive int.')
  layout = tuple(ctx.degree_layout)
  if len(layout) != 2 or math.prod(layout) != ctx.degree:
    raise ValueError(
        f'degree_layout {layout} must multiply to degree {ctx.degree}.'
    )
  if not ctx.q_towers:
    raise ValueError('q_towers must be non-empty.')
  if ctx.composite_degree >= len(ctx.q_towers):
    raise ValueError('composite_degree must be smaller than the Q chain.')
  if not math.isfinite(ctx.scaling_factor) or ctx.scaling_factor <= 0:
    raise ValueError('scaling_factor must be finite and positive.')
  if not math.isfinite(ctx.output_scale) or ctx.output_scale < 0:
    raise ValueError('output_scale must be finite and non-negative.')


# -- Binding and executable materialization ---------------------------------


# =============================================================================
# END: PRIVATE MAPPING BACKEND
# =============================================================================


# =============================================================================
# CKKSContext PUBLIC CODEC AND LOW-LEVEL EVALUATOR FACADE
# =============================================================================
class CKKSContext:

  def __init__(
      self,
      parameters: dict,
      *,
      batch: int = 1,
      dnum: Optional[int] = None,
  ):
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
      raise ValueError('batch must be a positive int.')
    if (
        dnum is not None
        and (isinstance(dnum, bool) or not isinstance(dnum, int) or dnum <= 0)
    ):
      raise ValueError('dnum must be None or a positive int.')
    self.parameters = parameters
    self.batch = batch
    self.dnum = dnum
    self.degree = parameters["degree"]
    num_slots = parameters.get("num_slots", self.degree // 2)
    if (
        isinstance(num_slots, bool)
        or not isinstance(num_slots, (int, np.integer))
        or not util.is_power_of_two(int(num_slots))
        or (self.degree // 2) % int(num_slots)
    ):
      raise ValueError(
          "num_slots must be a power-of-two divisor of degree // 2."
      )
    self.num_slots = int(num_slots)
    self.scaling_factor = parameters.get("scaling_factor", 0.0)
    self.output_scale = parameters.get("output_scale", 0.0)
    self.q_towers = parameters["q_towers"]
    self.p_towers = parameters.get("p_towers", [])
    self.p = parameters.get("p", 0)
    self.CKKS_M_FACTOR = parameters.get("CKKS_M_FACTOR", 1)
    self.moduli = self.q_towers
    degree_layout = parameters.get("degree_layout")
    if degree_layout is None:
      requested_rows = parameters.get("r")
      if requested_rows is None:
        degree_layout = _balanced_degree_layout(self.degree)
      else:
        requested_rows = int(requested_rows)
        degree_layout = (requested_rows, self.degree // requested_rows)
    self.degree_layout = tuple(degree_layout)
    if (len(self.degree_layout) != 2
        or self.degree_layout[0] * self.degree_layout[1] != self.degree):
      raise ValueError(
          f"degree_layout {self.degree_layout} must be a two-dimensional "
          f"factorization of degree {self.degree}."
      )

    self.public_key = parameters.get("public_key", None)
    self.secret_key = parameters.get("secret_key", None)
    self.rotation_key = parameters.get("rotation_key", None)
    self.evaluation_key = parameters.get("evaluation_key", None)

    # Composite rescaling support
    self.composite_degree = parameters.get("composite_degree", 1)
    self._compute_composite_scale_factor()

    # Static network ownership deliberately lives in Mapping. A standalone
    # context remains a codec and direct low-level evaluator resource.

  def _compute_composite_scale_factor(self):
    """Compute the effective scale factor for composite rescaling.

    For composite_degree=k, the last k moduli form the scale factor. This
    enables higher precision on fixed word-size architectures by grouping
    multiple <31-bit moduli to achieve effective scales of k*30 bits.
    """
    if self.composite_degree >= len(self.q_towers):
      raise ValueError(
          f"composite_degree ({self.composite_degree}) must be < number of"
          f" moduli ({len(self.q_towers)})"
      )

    # Product of last composite_degree moduli = effective scale per level
    self.composite_scale_factor = 1
    for i in range(self.composite_degree):
      self.composite_scale_factor *= self.q_towers[-(i + 1)]

  def encrypt(
      self, plaintext: poly.Polynomial, v=None, e=None
  ) -> poly.Polynomial:
    if self.public_key is None:
      raise ValueError("Public key is not set in the context.")
    result = _fast_encrypt_from_plaintext(
        plaintext,
        self,
        v=v,
        e=e,
        sigma=self.parameters.get("sigma", 3.190000057220458984375),
        noise_scale_degree=self.parameters.get("noise_scale_degree", 1),
    )
    if getattr(result, '_ckks_scale', None) is None:
      result._ckks_scale = float(self.scaling_factor)
    return result

  def encrypt_slots(
      self, slots, scale: Optional[float] = None
  ) -> poly.Polynomial:
    """Encode and encrypt one slot vector through the context facade."""
    if self.public_key is None:
      raise ValueError('Public key is not set in the context.')
    return _fast_encode_encrypt(slots, self, scale=scale)

  def encrypt_slots_batch(
      self, slots_batch, scale: Optional[float] = None
  ) -> poly.Polynomial:
    """Encode and encrypt a host batch with one device transfer."""
    if self.public_key is None:
      raise ValueError('Public key is not set in the context.')
    return _fast_encode_encrypt_batch(slots_batch, self, scale=scale)

  def decrypt(self, ciphertext: poly.Polynomial) -> poly.Polynomial:
    if self.secret_key is None:
      raise ValueError("Secret key is not set in the context.")
    if not isinstance(ciphertext, poly.Polynomial):
      raise TypeError("ciphertext must be a Polynomial.")
    ciphertext.validate()
    if ciphertext.batch != 1:
      raise NotImplementedError(
          "CKKSContext.decrypt currently supports exactly one ciphertext batch."
      )
    decrypted_poly_rns = _fast_decrypt_to_rns_coeffs(
        ciphertext, self
    )
    payload = jnp.asarray(decrypted_poly_rns, dtype=jnp.uint32).reshape(
        1, 1, *ciphertext.degree_layout, ciphertext.num_moduli
    )
    return ciphertext._clone_with_payload(payload, num_elements=1)

  def decrypt_slots(
      self,
      ciphertext: poly.Polynomial,
      scale: Optional[float] = None,
      slots_to_decode: Optional[int] = None,
      *,
      validate_approximation: bool = True,
  ) -> np.ndarray:
    """Decrypt and decode one ciphertext through the context facade.

    Returns a single slot vector; use :meth:`decrypt_slots_batch` for a batched
    payload rather than relying on truncation.
    """
    if self.secret_key is None:
      raise ValueError('Secret key is not set in the context.')
    scale = _resolve_codec_scale(ciphertext, scale, self.output_scale)
    return _fast_decrypt_decode(
        ciphertext,
        self,
        scale,
        slots_to_decode=slots_to_decode,
        validate_approximation=validate_approximation,
        require_single_batch=True,
    )[0]

  def decrypt_slots_batch(
      self,
      ciphertext: poly.Polynomial,
      scale: Optional[float] = None,
      slots_to_decode: Optional[int] = None,
      *,
      validate_approximation: bool = True,
  ) -> list[np.ndarray]:
    """Decrypt and decode a ciphertext batch after one host transfer.

    Returns one slot vector per batch entry. The list is a facade over the
    ``(batch, slots)`` matrix the decoder produces; the split exists because
    demo callers outside this package expect a sequence.
    """
    if self.secret_key is None:
      raise ValueError('Secret key is not set in the context.')
    scale = _resolve_codec_scale(ciphertext, scale, self.output_scale)
    return list(_fast_decrypt_decode(
        ciphertext,
        self,
        scale,
        slots_to_decode=slots_to_decode,
        validate_approximation=validate_approximation,
    ))

  def encode(self, slots: List[complex], shift: int = 0) -> poly.Polynomial:
    del shift  # Retained for API compatibility; encoding has no shift variant.
    encoded_rns = _fast_encode(
        slots,
        self,
        scale=self.scaling_factor,
    )

    ntt_ctx = getattr(self, '_polynomial_ntt_ctx', None)
    cache = getattr(self, '_param_cache', None)
    if cache is not None:
      ntt_ctx = cache.get_sliced_ntt_q(cache.max_level)
    result = _polynomial_from_flat_array(
        np.asarray(encoded_rns, dtype=np.uint64)[None, None, ...],
        self.q_towers,
        self.degree_layout,
        ntt_ctx=ntt_ctx,
    )
    result._ckks_scale = float(self.scaling_factor)
    if getattr(self, '_polynomial_ntt_ctx', None) is None:
      self._polynomial_ntt_ctx = result.ntt_ctx
    return result

  def encode_at_level(
      self,
      slots: List[complex],
      level: int,
      scale: Optional[float] = None,
  ) -> poly.Polynomial:
    """Encode one slot vector against an initialized logical level.

    The result is a batch-one, one-element NTT plaintext using exactly the Q
    prefix and tiled NTT context owned by ``level``. This is the canonical
    constant-construction route for a bound static program.
    """
    cache = getattr(self, '_param_cache', None)
    if cache is None:
      raise RuntimeError(
          'Call CKKSContext.program_initialization(...) before '
          'encode_at_level(...).'
      )
    if not isinstance(level, int):
      raise TypeError(f'level must be an int, got {type(level).__name__}.')
    if not 0 <= level <= cache.max_level:
      raise ValueError(
          f'level {level} out of range [0, {cache.max_level}].'
      )

    actual_scale = self.scaling_factor if scale is None else scale
    q_at_level = cache.q_moduli_at_level(level)
    encode_ctx = _get_or_make_encode_ctx(
        q_towers=q_at_level,
        degree=self.degree,
        scale=actual_scale,
        max_bits_in_word=self.parameters.get('max_bits_in_word', 61),
        max_bits_value=self.parameters.get(
            'max_bits_value', (1 << 63) - (1 << 9) - 1
        ),
    )
    encoded_rns = _fast_encode(slots, encode_ctx, scale=actual_scale)
    result = _polynomial_from_flat_array(
        np.asarray(encoded_rns, dtype=np.uint64)[None, None, ...],
        q_at_level,
        cache.degree_layout,
        ntt_ctx=cache.get_sliced_ntt_q(level),
    )
    result._ckks_scale = float(actual_scale)
    return result

  def decode(
      self,
      encoded_plaintext: poly.Polynomial,
      is_ntt: bool = False,
      *,
      scale: float | None = None,
      level: int | None = None,
      validate_approximation: bool = True,
  ) -> jnp.ndarray:
    """Decode a CKKS plaintext with explicit, tracked, or canonical scale.

    ``level`` validates the plaintext's CROSS logical level and supplies the
    canonical fallback scale. A valid scale tracked on the plaintext takes
    precedence over that fallback, and an explicit ``scale`` takes precedence
    over both.
    """
    _validate_polynomial_for_context(
        encoded_plaintext, self, 'encoded_plaintext', num_elements=1
    )
    if encoded_plaintext.batch != 1:
      raise NotImplementedError(
          "CKKSContext.decode currently supports exactly one plaintext batch."
      )
    rns_poly = encoded_plaintext.polynomial[0, 0].reshape(
        encoded_plaintext.degree, encoded_plaintext.num_moduli
    ).tolist()
    num_towers = len(encoded_plaintext.moduli)
    fallback_scale = self.output_scale
    if level is not None:
      if (
          isinstance(level, bool)
          or not isinstance(level, (int, np.integer))
      ):
        raise TypeError('decode level must be an int.')
      level = int(level)
      if not hasattr(self, "_param_cache"):
        raise ValueError(
            "level-aware decode requires program_initialization"
        )
      if not 0 <= level <= self._param_cache.max_level:
        raise ValueError(
            f"decode level {level} out of range "
            f"[0, {self._param_cache.max_level}]"
        )
      expected_towers = self._param_cache.num_q_at_level(level)
      if expected_towers != num_towers:
        raise ValueError(
            f"level {level} has {expected_towers} Q limbs, but plaintext has "
            f"{num_towers}"
        )
      if self._param_cache.composite_degree >= 2:
        fallback_scale = self._param_cache.scaling_factor_recursive(level)
    scale = _resolve_codec_scale(
        encoded_plaintext, scale, fallback_scale
    )

    if is_ntt:
      # rns_poly is (degree, moduli).
      new_poly = [[0] * num_towers for _ in range(self.degree)]
      for t_id in range(num_towers):
        # get column
        col = [rns_poly[d][t_id] for d in range(self.degree)]

        rev = util.bit_reverse_array(col)
        intt_vals = util.intt_negacyclic_bit_reverse(
            rev,
            self.q_towers[t_id],
            util.root_of_unity(2 * self.degree, self.q_towers[t_id]),
        )

        for d in range(self.degree):
          new_poly[d][t_id] = intt_vals[d]
      rns_poly = new_poly

    plain_combined = _crt_combine_rns(
        rns_poly, self.q_towers[:num_towers]
    )

    big_q = 1
    for qi in self.q_towers[:num_towers]:
      big_q *= qi

    res = _ckks_decode(
        plaintext=plain_combined,
        scaling_factor=scale,
        slots=self.num_slots,
        q=big_q,
        p=self.p,
        CKKS_M_FACTOR=self.CKKS_M_FACTOR,
        validate_approximation=validate_approximation,
    )

    return jnp.array(res)

  def program_initialization(
      self,
      total_rotation_indices: List[int],
      dnum: int,
      r: int,
      c: int,
      degree_layout: Optional[tuple] = None,
      batch: int = 1,
      perf_test: bool = False,
      pregenerated_rotation_keys: Optional[dict] = None,
      cache_rotation_keys: bool = True,
      finite_field_context=None,
  ):
    """Offline one-time setup.

    Creates the shared parameter/key cache and level-indexed accessor
    factories. Exact per-level operator controls are materialized when an
    accessor is obtained; ``Mapping`` performs all required lookups before it
    publishes its static executable. Advanced direct users must likewise
    obtain their exact accessors during offline setup.

    ``cache_rotation_keys=False`` defers rotation-key generation until an
    accessor first needs each index and does not retain raw or formatted keys
    in the parameter cache. The operator accessor holds only its active key
    until that accessor is cleared.

    After calling this, use:
      ctx.he_add[level].add(ct1, ct2)
      ctx.he_sub[level].sub(ct1, ct2)
      ctx.he_mul[level].mul(ct1, ct2)
      ctx.he_mul[level].hemul_no_relin(ct1, ct2)
      ctx.he_mul[level].relinearize(ct_3elem)
      ctx.he_rot[level, rot_index].rotate(ct)
      ctx.he_rescale[src_level, dst_level].rescale(ct)
      ctx.ptct_mul[level].mul(ct, plaintext)
      matvec = ctx.bsgs_matvec[level, n]
      matvec.preprocess(matrix)
      matvec.matvec(ct)
      ctx.he_bootstrap.configure(...).setup()

    ``ctx.he_mul[output_level]`` expects normal multiplication operands at
    ``output_level + 1``. Its explicit ``hemul_no_relin`` and ``relinearize``
    controls both preserve that input level; callers may then invoke
    ``ctx.he_rescale[output_level + 1, output_level]`` explicitly.

    Args:
        total_rotation_indices: List of rotation indices to support.
        dnum: Key-switch decomposition parameter. r, c: Matrix NTT dimensions
          (degree = r * c).
        degree_layout: Tuple (r, c). Defaults to (r, c).
        batch: Batch size for ciphertext operations.
        perf_test: Use random params for benchmarking.
        pregenerated_rotation_keys: Optional dict of max-level rotation keys
          keyed by rotation index. Missing keys are generated as usual when
          ``cache_rotation_keys`` is true. The resolved mapping is then
          retained on ``self._raw_rotation_keys`` for serialization; the
          non-retaining mode intentionally ignores this input.
    """
    if getattr(self, '_param_cache', None) is not None:
      raise RuntimeError(
          'this CKKSContext is already initialized and cannot be reinitialized.'
      )
    if not isinstance(cache_rotation_keys, bool):
      raise TypeError('cache_rotation_keys must be a bool.')
    degree_layout = tuple(degree_layout or (r, c))
    if r * c != self.degree:
      raise ValueError(f"r*c ({r}*{c}={r*c}) must equal degree ({self.degree})")
    if degree_layout != (r, c):
      raise ValueError(
          f'degree_layout must match program NTT layout {(r, c)}, got '
          f'{degree_layout}.'
      )
    self.degree_layout = degree_layout
    self.batch = batch
    self.dnum = dnum

    # Require exact P coverage and the uint31/uint64 arithmetic envelope used
    # by the current Barrett, Montgomery, and BConv kernels.
    util.validate_barrett_bconv_moduli(
        self.q_towers, self.p_towers, dnum
    )

    noise_std = _normalize_noise_std(self.parameters.get("sigma", sigma))
    noise_scale = normalize_noise_scale_degree(
        self.parameters.get("noise_scale_degree", 1)
    )

    # 1. Generate eval key if not provided
    if self.evaluation_key is None:
      if self.secret_key is None:
        raise ValueError("secret_key required for key generation")
      ek = kg.gen_evaluation_key(
          self.secret_key,
          q=self.q_towers,
          P=self.p_towers,
          noise_std=noise_std,
          noise_scale=1,
          dnum=dnum,
      )
      eval_key_a = jnp.array(ek["a"], dtype=jnp.uint32).transpose(0, 2, 1)
      eval_key_b = jnp.array(ek["b"], dtype=jnp.uint32).transpose(0, 2, 1)
    else:
      eval_key_a, eval_key_b = self.evaluation_key

    # 2. Generate rotation keys (or reuse pre-generated ones from a cache).
    # coef_maps are cheap permutation arrays and are always recomputed.
    # CROSS_SKIP_TOPLEVEL_ROTKEYS=1 skips max-level rotation-key allocation;
    # runtime rotation operators regenerate an ephemeral per-level key from the
    # secret key (per_level_rotation_keys mode on the cache). Large-N memory.
    retain_toplevel_keys = (
        cache_rotation_keys
        and os.environ.get("CROSS_SKIP_TOPLEVEL_ROTKEYS") != "1"
    )
    rot_keys = (
        dict(pregenerated_rotation_keys or {})
        if retain_toplevel_keys
        else {}
    )
    coef_maps = {}
    for rot_idx in total_rotation_indices:
      coef_maps[rot_idx] = util.precompute_auto_map(
          self.degree,
          kg.find_automorphism_index_2n_complex(rot_idx, 2 * self.degree),
      )
      if not retain_toplevel_keys:
        continue  # regenerated per level at run time (get_rot_key); never read.
      if (pregenerated_rotation_keys is not None
          and rot_idx in pregenerated_rotation_keys):
        rot_keys[rot_idx] = pregenerated_rotation_keys[rot_idx]
        continue
      rot_ek = kg.gen_rotation_key(
          self.secret_key,
          self.q_towers,
          self.p_towers,
          rot_idx,
          dnum=dnum,
          noise_std=noise_std,
          noise_scale=noise_scale,
      )
      rot_keys[rot_idx] = rot_ek[rot_idx]

    # Expose the resolved rotation keys so callers can serialize them
    # (used by the demos' cache freeze path).
    self._raw_rotation_keys = rot_keys

    # 3. Create parameter cache
    self._param_cache = HEParameterCache(
        q_towers=self.q_towers,
        p_towers=self.p_towers,
        r=r,
        c=c,
        dnum=dnum,
        composite_degree=self.composite_degree,
        batch=batch,
        perf_test=perf_test,
        rotation_key_noise_std=noise_std,
        rotation_key_noise_scale=noise_scale,
        finite_field_context=finite_field_context,
        cache_rotation_keys=cache_rotation_keys,
    )
    self._param_cache.initialize(
        eval_key_a, eval_key_b, rot_keys, coef_maps, secret_key=self.secret_key
    )

    # 4. Create accessors (lazy — instances created on first access)
    self._he_add = _HEAddAccessor(self._param_cache)
    self._he_sub = _HESubAccessor(self._param_cache)
    self._he_mul = _HEMulAccessor(self._param_cache)
    self._he_rot = _HERotAccessor(self._param_cache)
    self._he_rescale = _HERescaleAccessor(self._param_cache)
    self._he_level_reduce = _HELevelReduceAccessor(self._param_cache)

    # Polynomial-plaintext multiplication accessor
    self._ptct_mul = _HEPtCtMulAccessor(self._param_cache)

    # BSGS matrix-vector accessor: ctx.bsgs_matvec[level, n, n1, n2]
    self._bsgs_matvec = _BSGSMatVecAccessor(self)

    # One context-owned route to the repository's single bootstrap engine.
    self._he_bootstrap = _HEBootstrapAccessor(self)

  @property
  def he_add(self):
    return self._he_add

  @property
  def he_sub(self):
    return self._he_sub

  @property
  def he_mul(self):
    return self._he_mul

  @property
  def he_rot(self):
    return self._he_rot

  @property
  def he_rescale(self):
    return self._he_rescale

  @property
  def he_level_reduce(self):
    """Drop modulus limbs without dividing the scale.

    ``ctx.he_level_reduce[src_level, dst_level].level_reduce(ct)`` brings a
    ciphertext down to meet another for addition. Use ``he_rescale`` instead
    after a multiplication, where the scale must come down with the moduli.
    """
    return self._he_level_reduce

  @property
  def ptct_mul(self):
    return self._ptct_mul

  @property
  def bsgs_matvec(self):
    return self._bsgs_matvec

  @property
  def he_bootstrap(self):
    return self._he_bootstrap

  @property
  def max_level(self) -> int:
    if hasattr(self, "_param_cache"):
      return self._param_cache.max_level
    return (len(self.q_towers) - 1) // self.composite_degree


__all__ = ['CKKSContext', 'ScalingFactorTooSmall']

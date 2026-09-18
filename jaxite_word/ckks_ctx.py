import cmath
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

import util
from he_ops import HEMulAccessor
from he_ops import HEBsgsMatVecAccessor
from he_ops import HEPtCtMulAccessor
from he_ops import HERescaleAccessor
from he_ops import HERotAccessor
from he_params import HEParameterCache
import key_gen as kg
import polynomial as poly


sigma = 3.190000057220458984375


########################
# Common Functions
########################
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
# Embedded from the former `encrypt_fast.py` module. Exposes:
#   * `vectorized_ntt(coeffs, cache)`             - forward negacyclic NTT
#   * `fast_encrypt_from_plaintext(pt, ctx, ...)` - asymmetric encrypt
#   * `fast_encode(slots, ctx, ...)`              - slots → plaintext-eval
#   * `fast_encode_encrypt(slots, ctx, scale)`    - combined entry point
#
# Noise sampling uses kernel CSPRNG via `os.urandom` + vectorized Box-Muller.
# OpenFHE alignment:
#   * `v` ~ TernaryUniform {-1, 0, +1} matches `OpenFHE::TernaryUniformGenerator`
#     (`src/core/include/math/ternaryuniformgenerator-impl.h`).
#   * `e0, e1` ~ continuous Gaussian (Box-Muller) rounded to nearest int.
#     OpenFHE uses Peikert's CDF inversion or Karney's algorithm to draw from
#     the *discrete* Gaussian directly. At σ ≈ 3.19 the two distributions
#     agree to within parts per thousand and produce identical end-to-end
#     correctness (decrypt ≈ plaintext within CKKS noise tolerance).
#
# `_ENCRYPT_CACHE` is keyed by `(id(ctx), num_q)` so repeated encrypts pay
# the precomputation only once.
# ===========================================================================
_TWO_PI = 2.0 * math.pi
_INV_2_53 = 1.0 / (1 << 53)


def _csprng_uniform_open(n: int) -> np.ndarray:
  """n float64 uniform on (0, 1] from kernel CSPRNG (one syscall)."""
  raw = os.urandom(8 * n)
  u64 = np.frombuffer(raw, dtype=np.uint64)
  return (u64 >> 11).astype(np.float64) * _INV_2_53 + _INV_2_53


def _csprng_gaussian(n: int, sigma: float) -> np.ndarray:
  """n float64 ~ N(0, sigma²) via vectorized Box-Muller (one syscall)."""
  n_pairs = (n + 1) // 2
  u = _csprng_uniform_open(2 * n_pairs).reshape(n_pairs, 2)
  radius = np.sqrt(-2.0 * np.log(u[:, 0]))
  angle = _TWO_PI * u[:, 1]
  z = np.empty(2 * n_pairs, dtype=np.float64)
  z[0::2] = radius * np.cos(angle)
  z[1::2] = radius * np.sin(angle)
  return (z[:n] if 2 * n_pairs > n else z) * sigma


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

    pk_full_0 = np.asarray(public_key[0], dtype=np.uint64)
    pk_full_1 = np.asarray(public_key[1], dtype=np.uint64)
    self.pk0 = pk_full_0[:M].T.copy()                            # (N, M)
    self.pk1 = pk_full_1[:M].T.copy()                            # (N, M)

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


_ENCRYPT_CACHE: Dict[Tuple[int, int], _EncryptCache] = {}


def _get_encrypt_cache(ctx, num_q: int) -> _EncryptCache:
  key = (id(ctx), int(num_q))
  c = _ENCRYPT_CACHE.get(key)
  if c is None:
    psi_pairs = [
        util.root_of_unity(2 * ctx.degree, q) for q in ctx.q_towers[:num_q]
    ]
    c = _EncryptCache(
        q_towers=list(ctx.q_towers[:num_q]),
        psi_pairs=psi_pairs,
        public_key=ctx.public_key,
        degree=ctx.degree,
    )
    _ENCRYPT_CACHE[key] = c
  return c


def vectorized_ntt(coeffs: np.ndarray, cache: _EncryptCache) -> np.ndarray:
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


def fast_encrypt_from_plaintext(plaintext_eval: np.ndarray, ctx,
                                v=None, e=None, sigma: float = None,
                                noise_scale_degree: int = None) -> np.ndarray:
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
  qb = cache.q.reshape(1, M)
  if sigma is None:
    sigma = float(ctx.parameters.get("sigma", 3.190000057220458984375))
  if noise_scale_degree is None:
    noise_scale_degree = int(ctx.parameters.get("noise_scale_degree", 1))
  ns = int(noise_scale_degree)

  if v is None:
    v_signed = _csprng_ternary(N)
    q_int64 = cache.q.astype(np.int64).reshape(1, M)
    v_rns = np.ascontiguousarray(
        np.mod(v_signed[:, None], q_int64).astype(np.uint64))
    v_eval = vectorized_ntt(v_rns, cache)
  else:
    v_eval = np.asarray(v, dtype=np.uint64).T.copy()

  if e is None:
    e_signed_all = np.round(_csprng_gaussian(2 * N, sigma)).astype(np.int64)
    e0_signed = e_signed_all[:N]
    e1_signed = e_signed_all[N:]
    q_int64 = cache.q.astype(np.int64).reshape(1, M)
    e0_rns = np.ascontiguousarray(
        np.mod(e0_signed[:, None], q_int64).astype(np.uint64))
    e1_rns = np.ascontiguousarray(
        np.mod(e1_signed[:, None], q_int64).astype(np.uint64))
    e0_eval = vectorized_ntt(e0_rns, cache)
    e1_eval = vectorized_ntt(e1_rns, cache)
  else:
    e0_eval = np.asarray(e[0], dtype=np.uint64).T.copy()
    e1_eval = np.asarray(e[1], dtype=np.uint64).T.copy()

  # c0 = (v * pk0 + ns * e0 + plaintext) mod q
  c0 = (v_eval * cache.pk0) % qb
  c0 = (c0 + (ns * e0_eval) % qb) % qb
  c0 = (c0 + plaintext_eval) % qb
  # c1 = (v * pk1 + ns * e1) mod q
  c1 = (v_eval * cache.pk1) % qb
  c1 = (c1 + (ns * e1_eval) % qb) % qb
  return np.stack([c0, c1], axis=0)


def fast_encode(slots, ctx, scale: float = None,
                noise_scale_degree: int = 1) -> np.ndarray:
  """Vectorized CKKS encode (slots → plaintext eval form).

  Output: (N, M) np.uint64 in NTT eval form (bit-reversed) — directly usable
  as input to fast_encrypt_from_plaintext.
  """
  if scale is None:
    scale = ctx.scaling_factor
  N = ctx.degree
  M = len(ctx.q_towers)
  cache = _get_encrypt_cache(ctx, M)
  m = N * 2

  # 1) inverse special FFT on slot vector
  y = list(slots)
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
    raise ValueError("Scaling factor too small")
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

  return vectorized_ntt(coeffs_rns, cache)


def fast_encode_encrypt(slots, ctx, scale: float = None) -> np.ndarray:
  """Encode + encrypt in one shot. Returns (2, N, M) uint64."""
  pt_eval = fast_encode(slots, ctx, scale=scale)
  return fast_encrypt_from_plaintext(pt_eval, ctx)


class _LiteCtxForFast:
  """Minimal CKKSContext-shaped object accepted by `fast_encode` /
  `fast_encrypt_from_plaintext`. Built on the fly inside `ckks_encrypt` /
  `ckks_encode` so those free-function APIs can route to the vectorized
  fast path without forcing callers to construct a full CKKSContext.

  When used by encode-only paths, `public_key` may be a zero-shaped dummy —
  `_get_encrypt_cache` stores it but `fast_encode` itself never reads
  `cache.pk0/pk1`.
  """

  def __init__(self, q_towers, degree, public_key, scaling_factor=None,
               parameters=None):
    self.q_towers = list(q_towers)
    self.degree = int(degree)
    self.public_key = public_key
    self.scaling_factor = float(scaling_factor) if scaling_factor is not None \
        else 1.0
    self.parameters = dict(parameters) if parameters is not None else {}


# Reuse one lite ctx per (q_towers, degree) for encode-only callers so
# `_get_encrypt_cache` (keyed by id(ctx)) doesn't rebuild twiddles
# + psi tables on every call. Critical for BSGS-encoding which calls
# ckks_encode hundreds–thousands of times during cache build.
_LITE_ENCODE_CTX_CACHE: dict = {}


def _get_or_make_encode_ctx(q_towers, degree, scale, max_bits_in_word,
                              max_bits_value):
  key = (tuple(int(q) for q in q_towers), int(degree))
  lite = _LITE_ENCODE_CTX_CACHE.get(key)
  if lite is None:
    dummy_pk = np.zeros((2, len(q_towers), int(degree)), dtype=np.uint64)
    lite = _LiteCtxForFast(
        q_towers=q_towers,
        degree=int(degree),
        public_key=dummy_pk,
        scaling_factor=scale,
        parameters={
            "max_bits_in_word": int(max_bits_in_word),
            "max_bits_value": int(max_bits_value),
        },
    )
    _LITE_ENCODE_CTX_CACHE[key] = lite
  else:
    # Scale may vary call-to-call; max_bits constants almost never do.
    lite.scaling_factor = float(scale)
    lite.parameters["max_bits_in_word"] = int(max_bits_in_word)
    lite.parameters["max_bits_value"] = int(max_bits_value)
  return lite


def ckks_encrypt(
    plaintext: List[List[int]],
    public_key,
    q_towers: List[int],
    noise_scale_degree: int = 1,
    sigma=3.190000057220458984375,
    v=None,
    e=None,
):
  """Fast CKKS encrypt — wraps `encrypt_fast.fast_encrypt_from_plaintext`.

  Same signature + output format as the historical pure-Python triple-loop
  implementation, but routes to the vectorized NumPy path. When `(v, e)` are
  None this samples noise via the kernel CSPRNG (`os.urandom` + Box-Muller).
  When `(v, e)` are supplied, output is bit-exact equivalent to
  `ckks_encrypt_fall_back` with the same inputs (the bit-exact test suite
  in `ckks_ctx_test` pins this).
  """
  degree = len(plaintext)
  num_q = len(q_towers)
  pt_arr = np.asarray(plaintext, dtype=np.uint64)             # (N, M)
  # Normalize public_key into a numpy array of shape (2, num_q_full, degree).
  pk_arr = np.asarray(public_key, dtype=np.uint64)
  if pk_arr.ndim == 3 and pk_arr.shape[1] > num_q:
    # Caller passed full Q+P pk; the fast path slices the first M rows.
    pass
  lite = _LiteCtxForFast(
      q_towers=q_towers,
      degree=degree,
      public_key=pk_arr,
      parameters={"sigma": sigma, "noise_scale_degree": noise_scale_degree},
  )
  ct_arr = fast_encrypt_from_plaintext(
      pt_arr, lite, v=v, e=e, sigma=sigma,
      noise_scale_degree=noise_scale_degree)
  # Convert (2, N, M) uint64 → [c0, c1] list-of-list of ints (legacy format).
  return [ct_arr[0].tolist(), ct_arr[1].tolist()]


def ckks_encrypt_fall_back(
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


def ckks_decrypt(
    ciphertext: List[List[List[int]]],
    private_key: List[List[int]],
    q_towers: List[int],
):
  """Standard CKKS decrypt: M(X) = sum_i c_i * s^i (mod q per tower) + INTT.

  Layout convention: ciphertext is (num_elements, degree, moduli);
  private_key is (moduli, degree); returned coefficients are (degree, moduli).
  """
  num_elements = len(ciphertext)
  degree = len(ciphertext[0])
  num_towers = len(ciphertext[0][0])

  s = private_key  # (moduli, degree)
  if num_towers < len(s):
    diff_length = len(s) - num_towers
    s = s[:-diff_length]

  # Accumulate M(X) in (degree, moduli).
  res_poly = [[0] * num_towers for _ in range(degree)]
  cur_s_power = [list(row) for row in s]  # s^1 in (moduli, degree)

  # c0
  for d in range(degree):
    for m in range(num_towers):
      res_poly[d][m] = ciphertext[0][d][m]

  for i in range(1, num_elements):
    ci = ciphertext[i]
    for d in range(degree):
      for m in range(num_towers):
        term = (ci[d][m] * cur_s_power[m][d]) % q_towers[m]
        res_poly[d][m] = (res_poly[d][m] + term) % q_towers[m]
    if i < num_elements - 1:
      new_s_power = [[0] * degree for _ in range(num_towers)]
      for m in range(num_towers):
        qi = q_towers[m]
        for d in range(degree):
          new_s_power[m][d] = (cur_s_power[m][d] * s[m][d]) % qi
      cur_s_power = new_s_power

  # INTT each tower (1D over the degree axis).
  final_res = [[0] * num_towers for _ in range(degree)]
  for m in range(num_towers):
    col = [res_poly[d][m] for d in range(degree)]
    col_rev = util.bit_reverse_array(col)
    coef = util.intt_negacyclic_bit_reverse(
        col_rev, q_towers[m], util.root_of_unity(2 * degree, q_towers[m])
    )
    for d in range(degree):
      final_res[d][m] = coef[d]
  return final_res


def ckks_encode(
    slots: List[complex],
    cycl_order: int,
    q_towers: List[int],
    p_towers: List[int],
    scale: float,
    noise_scale_degree: int = 1,
    max_bits_in_word: int = 61,
    max_bits_value: int = (1 << 63) - (1 << 9) - 1,
) -> List[List[int]]:
  """Fast CKKS encode — wraps `encrypt_fast.fast_encode`.

  Same signature + output format (list-of-list of ints, shape (degree,
  moduli)) as the historical pure-Python implementation, but routes to
  the vectorized NumPy path. `p_towers` is accepted for backward-compat
  but unused (encode emits Q-tower residues only — same as the legacy
  reference). bit-exact vs `ckks_encode_fall_back` for the same inputs.
  """
  degree = cycl_order // 2
  # Reuse one lite ctx per (q_towers, degree) so the per-ctx twiddle/psi
  # cache doesn't rebuild on every call. Encode path does NOT read
  # pk0/pk1 from the cache — only twiddles + psi_pow + q.
  lite = _get_or_make_encode_ctx(
      q_towers, degree, scale, max_bits_in_word, max_bits_value)
  arr = fast_encode(slots, lite, scale=scale,
                      noise_scale_degree=noise_scale_degree)  # (N, M) uint64
  return arr.tolist()


def ckks_encode_fall_back(
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
  assert m == 4 * nh, "cycl_order must be 4*Nh for CKKS special FFT size"

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
    raise ValueError("Scaling factor too small")
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


def ckks_decode(
    plaintext: List[int],
    scaling_factor: float,
    slots: int,
    q: int,
    p: int,
    CKKS_M_FACTOR: int = 1,
    ADD_NOISE: bool = False,
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
    r_val = plaintext[i]
    if r_val > q_half:
      real_part = -((q - r_val) * sf_pre)
    else:
      real_part = r_val * sf_pre
    real_part_list.append(int(real_part))

    # imag part from second half
    im_val = plaintext[i + Nh]
    if im_val > q_half:
      imag_part = -((q - im_val) * sf_pre)
    else:
      imag_part = im_val * sf_pre
    imag_part_list.append(int(imag_part))

  curValues = [
      complex(real_part_list[i], imag_part_list[i]) for i in range(slots)
  ]

  # Step 2: compute conjugate vector and estimated stddev (per OpenFHE logic)
  def _conjugate(vec: List[complex]) -> List[complex]:
    n = len(vec)
    result: List[complex] = [complex(0, 0)] * n
    for idx in range(1, n):
      z = vec[n - idx]
      result[idx] = complex(-z.imag, -z.real)
    z0 = vec[0]
    result[0] = complex(z0.real, -z0.imag)
    return result

  def _stddev(vec: List[complex], conjugate: List[complex]) -> float:
    import math as math

    s = len(vec)
    if s == 1:
      return vec[0].imag
    dslots = s * 2
    complex_values = [vec[i] - conjugate[i] for i in range(s // 2 + 1)]
    mean = 2 * sum((cv.real + cv.imag) for cv in complex_values[1 : (s // 2)])
    mean += complex_values[0].imag
    mean += 2 * complex_values[s // 2].real
    mean /= dslots - 1.0
    variance = 2 * sum(
        ((cv.real - mean) ** 2 + (cv.imag - mean) ** 2)
        for cv in complex_values[1 : (s // 2)]
    )
    variance += (complex_values[0].imag - mean) ** 2
    variance += 2 * (complex_values[s // 2].real - mean) ** 2
    variance /= dslots - 2.0
    return 0.5 * math.sqrt(variance)

  conjugate = _conjugate(curValues)

  stddev_dbl = _stddev(curValues, conjugate)
  logstd = math.log2(stddev_dbl) if stddev_dbl > 0 else float("-inf")
  if stddev_dbl < 0.125 * math.sqrt(degree):
    stddev_dbl = 0.125 * math.sqrt(degree)
  if logstd > p - 5.0:
    import ckks_ctx as _self_mod
    if not getattr(_self_mod, "BYPASS_DECODE_STDDEV_CHECK", False):
      raise Exception(
          "The decryption failed because the approximation error is too high."
          " Check the parameters. "
      )

  stddev = math.sqrt(CKKS_M_FACTOR + 1) * stddev_dbl
  scale = 0.5 * powP

  # For security, add tiny Gaussian noise scaled by 2^{-p}; it doesn't affect ~1e-3 accuracy
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


def _crt_combine_rns_plaintext(
    rns_plaintext: List[List[int]], moduli: List[int]
) -> List[int]:
  """Combine residues modulo pairwise-coprime moduli using the standard CRT formula.

  rns_plaintext is (degree, moduli).
  """
  M = 1
  for q in moduli:
    M *= q
  Mi_list = [M // qi for qi in moduli]
  inv_list = [pow(Mi, -1, qi) for Mi, qi in zip(Mi_list, moduli)]

  degree = len(rns_plaintext)
  num_moduli = len(moduli)

  result = []
  for d in range(degree):
    X = 0
    # rns_plaintext[d] is [r0, r1, ...] for degree d
    residues = rns_plaintext[d]

    for i in range(num_moduli):
      ri = residues[i]
      qi = moduli[i]
      Mi = Mi_list[i]
      inv = inv_list[i]
      X += (int(ri) % int(qi)) * int(Mi) * int(inv)

    result.append(X % M)
  return result


# ===========================================================================
# Vectorized CKKS decrypt + decode (CPU/NumPy uint64).
#
# Embedded from the former `decrypt_fast.py` module. Replaces the pure-Python
# triple-loop SK multiply and per-tower INTT in `ckks_decrypt` with NumPy
# uint64 vectorized ops; decode (CRT + FFT) is also vectorized.
#
# Exposes:
#   * `fast_decrypt_to_rns_coeffs(ct, ctx)` - vectorized SK·INTT → (N, M)
#   * `fast_decode(coef_rns, ctx, scale)`   - CRT combine + special FFT
#   * `fast_decrypt_decode(ct, ctx, scale)` - combined entry point
#
# `_DECRYPT_CACHE` is keyed by `(id(ctx), num_moduli)` so repeated decrypts
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
    psi_inv = [pow(int(psi), -1, q) for psi, q in zip(psi_pairs, self.q_int)]
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


_DECRYPT_CACHE: Dict[Tuple[int, int], _DecryptCache] = {}


def _get_decrypt_cache(ctx, num_moduli: int) -> _DecryptCache:
  key = (id(ctx), int(num_moduli))
  c = _DECRYPT_CACHE.get(key)
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
    _DECRYPT_CACHE[key] = c
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


def fast_decrypt_to_rns_coeffs(ct_polynomial: np.ndarray, ctx) -> np.ndarray:
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


def _crt_combine_rns(rns_coeffs: np.ndarray, q_int: List[int]) -> List[int]:
  """CRT combine from per-tower residues to big-int coefficients.

  rns_coeffs : np.uint64 (N, M).
  Returns Python list of length N (big ints).
  """
  N, M = rns_coeffs.shape
  Big = 1
  for q in q_int:
    Big *= q
  Mi_list = [Big // qi for qi in q_int]
  inv_list = [pow(Mi, -1, qi) for Mi, qi in zip(Mi_list, q_int)]
  weights = [Mi * inv for Mi, inv in zip(Mi_list, inv_list)]   # big ints

  rns_py = rns_coeffs.tolist()                                 # one host pull
  out = [0] * N
  for d in range(N):
    row = rns_py[d]
    X = 0
    for i in range(M):
      X += int(row[i]) * weights[i]
    out[d] = X % Big
  return out


def fast_decode(coef_rns: np.ndarray, ctx, scale: float,
                slots_to_decode: int = None) -> np.ndarray:
  """CRT combine + CKKS FFT to slot values.

  Args:
    coef_rns        : (N, M) uint64 from fast_decrypt_to_rns_coeffs.
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
  scale_inv = 1.0 / float(scale)
  reals = np.empty(num_slots, dtype=np.float64)
  imags = np.empty(num_slots, dtype=np.float64)
  for i in range(num_slots):
    r = combined[i]
    if r > Big_half:
      r -= Big
    reals[i] = float(r) * scale_inv
    im = combined[i + Nh]
    if im > Big_half:
      im -= Big
    imags[i] = float(im) * scale_inv

  cur = reals + 1j * imags

  # Mirror ckks_decode's _conjugate + average step.
  conj = np.empty(num_slots, dtype=np.complex128)
  conj[0] = complex(cur[0].real, -cur[0].imag)
  if num_slots > 1:
    z = cur[num_slots - 1 - np.arange(0, num_slots - 1)]
    conj[1:num_slots] = -z.imag - 1j * z.real
  cur = 0.5 * (cur + conj)

  # CKKS special FFT (forward).
  buf = list(cur)
  FFTSpecial(buf, N * 2)
  arr = np.array([z.real for z in buf[:slots_out]], dtype=np.float64)
  return arr


def fast_decrypt_decode(ct_polynomial: np.ndarray, ctx, scale: float,
                        slots_to_decode: int = None) -> np.ndarray:
  """End-to-end vectorized decrypt + decode."""
  rns = fast_decrypt_to_rns_coeffs(ct_polynomial, ctx)
  return fast_decode(rns, ctx, scale, slots_to_decode=slots_to_decode)


########################
# CKKS Context Class
########################
class CKKSContext:

  def __init__(self, parameters: dict):
    self.parameters = parameters
    self.degree = parameters["degree"]
    self.num_slots = parameters.get("num_slots", self.degree // 2)
    self.scaling_factor = parameters.get("scaling_factor", 0.0)
    self.output_scale = parameters.get("output_scale", 0.0)
    self.q_towers = parameters["q_towers"]
    self.p_towers = parameters.get("p_towers", [])
    self.p = parameters.get("p", 0)
    self.CKKS_M_FACTOR = parameters.get("CKKS_M_FACTOR", 1)
    self.moduli = self.q_towers

    self.public_key = parameters.get("public_key", None)
    self.secret_key = parameters.get("secret_key", None)
    self.rotation_key = parameters.get("rotation_key", None)
    self.evaluation_key = parameters.get("evaluation_key", None)

    # Composite rescaling support
    self.composite_degree = parameters.get("composite_degree", 1)
    self._compute_composite_scale_factor()

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

    element = plaintext.get_element(0)[0]  # Shape: (degree, num_moduli)
    # element is already (degree, moduli), no transpose needed
    encoded_values = element.tolist()

    c_poly = ckks_encrypt(
        plaintext=encoded_values,
        public_key=self.public_key,
        q_towers=self.q_towers,
        noise_scale_degree=self.parameters.get("noise_scale_degree", 1),
        sigma=self.parameters.get("sigma", 3.190000057220458984375),
        v=v,
        e=e,
    )

    shapes = {
        "batch": 1,
        "num_elements": 2,
        "num_moduli": len(self.q_towers),
        "degree": self.degree,
        "precision": 32,
    }

    res_ct = poly.Polynomial(shapes, parameters={"moduli": self.q_towers})

    # c0, c1 are (degree, moduli) naturally now
    c0 = jnp.expand_dims(
        jnp.array(c_poly[0], dtype=jnp.uint64), axis=0
    )  # (1, degree, moduli)
    c1 = jnp.expand_dims(jnp.array(c_poly[1], dtype=jnp.uint64), axis=0)

    res_ct.set_element(0, c0)
    res_ct.set_element(1, c1)

    return res_ct

  def decrypt(self, ciphertext: poly.Polynomial) -> poly.Polynomial:
    if self.secret_key is None:
      raise ValueError("Secret key is not set in the context.")
    c_list = [
        ciphertext.polynomial[0, 0].tolist(),  # c0
        ciphertext.polynomial[1, 0].tolist(),  # c1
    ]
    num_elems = ciphertext.num_elements
    c_list = []
    for i in range(num_elems):
      c_list.append(ciphertext.polynomial[0, i].tolist())  # (degree, moduli)
    num_moduli_ct = ciphertext.num_moduli
    current_q_towers = self.q_towers[:num_moduli_ct]
    decrypted_poly_rns = ckks_decrypt(
        ciphertext=c_list,
        private_key=self.secret_key,
        q_towers=current_q_towers,
    )
    shapes = {
        "batch": 1,
        "num_elements": 1,
        "num_moduli": len(current_q_towers),
        "degree": self.degree,
        "precision": 32,
    }

    res_ct = poly.Polynomial(shapes, parameters={"moduli": current_q_towers})
    # decrypted_poly_rns is (degree, moduli)
    elem = jnp.expand_dims(
        jnp.array(decrypted_poly_rns, dtype=jnp.uint32), axis=0
    )
    res_ct.set_element(0, elem)

    return res_ct

  def encode(self, slots: List[complex], shift: int = 0) -> poly.Polynomial:
    m = self.degree * 2
    encoded_rns = ckks_encode(
        slots=slots,
        cycl_order=m,
        q_towers=self.q_towers,
        p_towers=self.p_towers,
        scale=self.scaling_factor,
        max_bits_in_word=self.parameters.get("max_bits_in_word", 61),
        max_bits_value=self.parameters.get(
            "max_bits_value", (1 << 63) - (1 << 9) - 1
        ),
    )

    shapes = {
        "batch": 1,
        "num_elements": 1,
        "num_moduli": len(self.q_towers),
        "degree": self.degree,
        "precision": 32,
    }

    res_ct = poly.Polynomial(shapes, parameters={"moduli": self.q_towers})
    # encoded_rns is (degree, moduli)
    elem = jnp.expand_dims(jnp.array(encoded_rns, dtype=jnp.uint64), axis=0)
    res_ct.set_element(0, elem)

    return res_ct

  def decode(
      self, encoded_plaintext: poly.Polynomial, is_ntt: bool = False
  ) -> jnp.ndarray:
    rns_poly = encoded_plaintext.polynomial[0, 0].tolist()  # (degree, moduli)
    num_towers = len(encoded_plaintext.moduli)

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

    plain_combined = _crt_combine_rns_plaintext(
        rns_poly, self.q_towers[:num_towers]
    )

    big_q = 1
    for qi in self.q_towers[:num_towers]:
      big_q *= qi

    res = ckks_decode(
        plaintext=plain_combined,
        scaling_factor=self.output_scale,
        slots=self.num_slots,
        q=big_q,
        p=self.p,
        CKKS_M_FACTOR=self.CKKS_M_FACTOR,
    )

    return jnp.array(res)

  def program_initialization(
      self,
      total_hemul_levels: int,
      total_rotation_indices: List[int],
      dnum: int,
      r: int,
      c: int,
      degree_layout: Optional[tuple] = None,
      batch: int = 1,
      perf_test: bool = False,
      pregenerated_rotation_keys: Optional[dict] = None,
  ):
    """Offline one-time setup.

    Creates parameter cache and all operator wrappers.

    After calling this, use:
      ctx.he_mul[level].mul(ct1, ct2)
      ctx.he_mul[level].hemul_no_relin(ct1, ct2)
      ctx.he_mul[level].relinearize(ct_3elem)
      ctx.he_rot[level, rot_index].rotate(ct)
      ctx.he_rescale[src_level, dst_level].rescale(ct)

    Args:
        total_hemul_levels: Maximum multiplication level (max_level).
        total_rotation_indices: List of rotation indices to support.
        dnum: Key-switch decomposition parameter. r, c: Matrix NTT dimensions
          (degree = r * c).
        degree_layout: Tuple (r, c). Defaults to (r, c).
        batch: Batch size for ciphertext operations.
        perf_test: Use random params for benchmarking.
    """
    degree_layout = degree_layout or (r, c)
    if r * c != self.degree:
      raise ValueError(f"r*c ({r}*{c}={r*c}) must equal degree ({self.degree})")
    noise_scale = self.parameters.get("noise_scale_degree", 1)

    # 1. Generate eval key if not provided
    if self.evaluation_key is None:
      if self.secret_key is None:
        raise ValueError("secret_key required for key generation")
      ek = kg.gen_evaluation_key(
          self.secret_key,
          q=self.q_towers,
          P=self.p_towers,
          noise_std=sigma,
          noise_scale=1,
          dnum=dnum,
      )
      eval_key_a = jnp.array(ek["a"], dtype=jnp.uint32).transpose(0, 2, 1)
      eval_key_b = jnp.array(ek["b"], dtype=jnp.uint32).transpose(0, 2, 1)
    else:
      eval_key_a, eval_key_b = self.evaluation_key

    # 2. Generate rotation keys (or use pre-generated ones from a cache).
    rot_keys = {}
    coef_maps = {}
    for rot_idx in total_rotation_indices:
      coef_maps[rot_idx] = util.precompute_auto_map(
          self.degree,
          kg.find_automorphism_index_2n_complex(rot_idx, 2 * self.degree),
      )
      if (pregenerated_rotation_keys is not None
          and rot_idx in pregenerated_rotation_keys):
        rot_keys[rot_idx] = pregenerated_rotation_keys[rot_idx]
      else:
        rot_ek = kg.gen_rotation_key(
            self.secret_key,
            self.q_towers,
            self.p_towers,
            rot_idx,
            dnum=dnum,
            noise_std=sigma,
            noise_scale=noise_scale,
        )
        rot_keys[rot_idx] = rot_ek[rot_idx]
    # Expose the raw rotation keys + secret-key-derived eval key for callers
    # that want to persist them to disk (cache).
    self._raw_rotation_keys = dict(rot_keys)

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
    )
    self._param_cache.initialize(
        eval_key_a, eval_key_b, rot_keys, coef_maps, secret_key=self.secret_key
    )

    # 4. Create accessors (lazy — instances created on first access)
    self.he_mul = HEMulAccessor(self._param_cache)
    self.he_rot = HERotAccessor(self._param_cache)
    self.he_rescale = HERescaleAccessor(self._param_cache)

    # Polynomial-plaintext multiplication accessor
    self.ptct_mul = HEPtCtMulAccessor(self._param_cache)

    # BSGS matvec accessor. Usage:
    #   mv = ctx.bsgs_matvec[level, n]        # auto-split n1, n2
    #   mv = ctx.bsgs_matvec[level, n, n1, n2]
    #   mv.encode_matrix(W)
    #   y_ct = mv.mul(ct_in)
    # Rotation keys for the baby [1..n1-1] and giant [n1, 2n1, ..., (n2-1)n1]
    # indices must be present in `total_rotation_indices` at init time.
    self.bsgs_matvec = HEBsgsMatVecAccessor(self)

    # Convenience aliases matching the spec
    self.he_mul_no_relin = self.he_mul
    self.relin = self.he_mul

  @property
  def max_level(self) -> int:
    if hasattr(self, "_param_cache"):
      return self._param_cache.max_level
    return (len(self.q_towers) - 1) // self.composite_degree

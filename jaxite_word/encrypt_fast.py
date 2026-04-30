"""Vectorized CKKS encrypt + encode (CPU/NumPy uint64).

Replaces the pure-Python triple-loop pieces of `ckks_ctx.ckks_encode` /
`ckks_ctx.ckks_encrypt`:

  * Forward per-tower NTT (negacyclic, bit-reversed output) — currently a
    Python `for d in range(degree): for m in range(num_towers): ...` triple
    loop. Vectorized with the same DIF butterfly + cached twiddle tables as
    `decrypt_fast.fast_decrypt_to_rns_coeffs`, but with forward twiddles and
    a pre-twist by `psi^i`.
  * `c0 = (v * pk0 + e0 + plaintext) mod q` and `c1 = (v * pk1 + e1) mod q`
    over (N, M) uint64.
  * Slot encode (inverse special FFT + scale + quantize + per-tower NTT).

Cache keyed by `(id(ctx), num_q)` so repeated encrypts pay the precomputation
only once.
"""
from __future__ import annotations
import math
import secrets
from typing import Dict, List, Tuple

import numpy as np

import util
import decrypt_fast as df


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

    # public_key has shape (2, num_q_full, N) — typically with extra P-moduli
    # appended for key switching. Encryption only uses the first M Q-moduli.
    pk_full_0 = np.asarray(public_key[0], dtype=np.uint64)
    pk_full_1 = np.asarray(public_key[1], dtype=np.uint64)
    self.pk0 = pk_full_0[:M].T.copy()                            # (N, M)
    self.pk1 = pk_full_1[:M].T.copy()                            # (N, M)

    # psi^i mod q  per tower:  shape (N, M).  Used for forward NTT pre-twist.
    psi_pow = np.zeros((N, M), dtype=np.uint64)
    for m in range(M):
      qm = self.q_int[m]
      psi_m = int(psi_pairs[m])
      acc = 1
      for i in range(N):
        psi_pow[i, m] = acc
        acc = (acc * psi_m) % qm
    self.psi_pow = psi_pow

    # Forward DIF twiddles per length L = N, N/2, ..., 2.
    # For each L, twiddle column j = (omega^(N/L))^j mod q where omega = psi^2.
    self.twiddles: Dict[int, np.ndarray] = {}
    L = N
    while L >= 2:
      half = L // 2
      tw = np.zeros((half, M), dtype=np.uint64)
      for m in range(M):
        qm = self.q_int[m]
        omega_m = pow(int(psi_pairs[m]), 2, qm)         # primitive N-th root
        w_m = pow(omega_m, N // L, qm)                  # forward direction
        acc = 1
        for j in range(half):
          tw[j, m] = acc
          acc = (acc * w_m) % qm
      self.twiddles[L] = tw
      L //= 2


_CACHE: Dict[Tuple[int, int], _EncryptCache] = {}


def _get_cache(ctx, num_q: int) -> _EncryptCache:
  key = (id(ctx), int(num_q))
  c = _CACHE.get(key)
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
    _CACHE[key] = c
  return c


def vectorized_ntt(coeffs: np.ndarray, cache: _EncryptCache) -> np.ndarray:
  """Forward negacyclic NTT, vectorized over towers.

  Input  : coeffs in coefficient form, shape (N, M) uint64.
  Output : eval form in BIT-REVERSED order (matches `bit_reverse_array(
           ntt_negacyclic_bit_reverse(...))` in the reference).
  """
  N, M = coeffs.shape
  qb = cache.q.reshape(1, M)

  # Negacyclic pre-twist: a[i] *= psi^i mod q
  a = (coeffs * cache.psi_pow) % qb

  # DIF butterflies (length L: N → 2). Output natively in bit-reversed order.
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


def _vectorized_ntt_with_bitreverse(coeffs: np.ndarray,
                                     cache: _EncryptCache) -> np.ndarray:
  """Forward NTT mirroring the reference (extra final bit-reverse).

  The reference does: pre_twist → bit_rev → DIT → bit_rev. This is the
  composition we want to match BIT-EQUAL. Using DIF natively gives natural
  → bit-rev. Then the reference's final `bit_reverse_array(...)` re-orders
  to bit-reversed too — but the inner DIT result is in NATURAL order, so the
  final bit_reverse turns natural to bit-reversed. Same as DIF.
  """
  return vectorized_ntt(coeffs, cache)


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

  Returns:
    ct   : np.uint64 array of shape (2, N, M) — c0, c1 in eval form.
  """
  N, M = plaintext_eval.shape
  cache = _get_cache(ctx, M)
  qb = cache.q.reshape(1, M)
  if sigma is None:
    sigma = float(ctx.parameters.get("sigma", 3.190000057220458984375))
  if noise_scale_degree is None:
    noise_scale_degree = int(ctx.parameters.get("noise_scale_degree", 1))
  ns = int(noise_scale_degree)

  # Sample v ~ {0, 1}^N (matches gen_ternary_uniform_polynomial).
  if v is None:
    v_coeffs = np.fromiter(
        (secrets.randbelow(2) for _ in range(N)), dtype=np.uint64, count=N
    )
    v_rns = np.broadcast_to(v_coeffs[:, None], (N, M)) % qb   # (N, M)
    v_rns = np.ascontiguousarray(v_rns, dtype=np.uint64)
    v_eval = vectorized_ntt(v_rns, cache)
  else:
    # v is provided as (M, N) list of NTT-domain residues.
    v_eval = np.asarray(v, dtype=np.uint64).T.copy()           # (N, M)

  # Sample Gaussian noise e0, e1.
  if e is None:
    prng = secrets.SystemRandom()
    e0_coeffs_signed = np.array(
        [round(prng.normalvariate(0, sigma)) for _ in range(N)],
        dtype=np.int64)
    e1_coeffs_signed = np.array(
        [round(prng.normalvariate(0, sigma)) for _ in range(N)],
        dtype=np.int64)
    e0_rns = np.empty((N, M), dtype=np.uint64)
    e1_rns = np.empty((N, M), dtype=np.uint64)
    for m in range(M):
      qm = cache.q_int[m]
      e0_rns[:, m] = np.array(
          [int(c) % qm for c in e0_coeffs_signed], dtype=np.uint64)
      e1_rns[:, m] = np.array(
          [int(c) % qm for c in e1_coeffs_signed], dtype=np.uint64)
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

  return np.stack([c0, c1], axis=0)                            # (2, N, M)


def fast_encode(slots, ctx, scale: float = None,
                noise_scale_degree: int = 1) -> np.ndarray:
  """Vectorized CKKS encode (slots → plaintext eval form).

  Output: (N, M) np.uint64 in NTT eval form (bit-reversed) — directly usable
  as input to fast_encrypt_from_plaintext.
  """
  from ckks_ctx import (
      FFTSpecialInv, slot_to_coeffs, nearest_int, fit_to_native_vector,
  )

  if scale is None:
    scale = ctx.scaling_factor
  N = ctx.degree
  M = len(ctx.q_towers)
  cache = _get_cache(ctx, M)
  m = N * 2

  # 1) inverse special FFT on slot vector
  y = list(slots)
  FFTSpecialInv(y, m)

  # 2) slot -> coefficients (length N, real)
  coeffs = slot_to_coeffs(y)

  # 3) scale + log-c bookkeeping (mirror ckks_encode)
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

  # Reduce per tower.
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

  # Stack to (N, M) uint64
  coeffs_rns = np.empty((N, M), dtype=np.uint64)
  for m_id in range(M):
    coeffs_rns[:, m_id] = np.array(ints_per_tower[m_id], dtype=np.uint64)

  return vectorized_ntt(coeffs_rns, cache)


def fast_encode_encrypt(slots, ctx, scale: float = None) -> np.ndarray:
  """Encode + encrypt in one shot. Returns (2, N, M) uint64."""
  pt_eval = fast_encode(slots, ctx, scale=scale)
  return fast_encrypt_from_plaintext(pt_eval, ctx)

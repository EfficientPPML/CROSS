"""Vectorized CKKS decrypt + decode.

Replaces the pure-Python triple-loop SK multiply and per-tower INTT in
`ckks_ctx.ckks_decrypt` with NumPy uint64 vectorized ops. Decode (CRT + FFT)
is also vectorized via NumPy (for reasonable slot counts) but kept compact.

Module-level cache keyed by (id(ctx), num_moduli) so repeated decrypts pay
the precomputation only once.
"""
from __future__ import annotations
import numpy as np
import math
from typing import Dict, Tuple, List

import util


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


# Cache keyed by (id(ctx), num_moduli)
_CACHE: Dict[Tuple[int, int], _DecryptCache] = {}


def _get_cache(ctx, num_moduli: int) -> _DecryptCache:
  key = (id(ctx), int(num_moduli))
  c = _CACHE.get(key)
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
    _CACHE[key] = c
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
  cache = _get_cache(ctx, M)

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


# ---------------- decode ----------------

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
  from ckks_ctx import FFTSpecial
  buf = list(cur)
  FFTSpecial(buf, N * 2)
  arr = np.array([z.real for z in buf[:slots_out]], dtype=np.float64)
  return arr


def fast_decrypt_decode(ct_polynomial: np.ndarray, ctx, scale: float,
                        slots_to_decode: int = None) -> np.ndarray:
  """End-to-end vectorized decrypt + decode."""
  rns = fast_decrypt_to_rns_coeffs(ct_polynomial, ctx)
  return fast_decode(rns, ctx, scale, slots_to_decode=slots_to_decode)

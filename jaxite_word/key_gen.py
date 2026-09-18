from functools import lru_cache
import math
import os
import secrets
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import rns
import util

RnsPolynomial = rns.RnsPolynomial
RnsParams = rns.RnsParams
gen_rns_polynomial = rns.gen_rns_polynomial

sigma = 3.190000057220458984375

# Bump whenever persisted key material becomes incompatible or its security
# distribution changes. Cache producers should fingerprint this value and
# reject older keys rather than silently reusing them.
KEY_GENERATION_VERSION = 3

_OPENFHE_DGG_TAIL_MULTIPLIER = 12.00610553538285
_OPENFHE_KARNEY_THRESHOLD = 300.0
_INV_2_53 = 1.0 / (1 << 53)


def normalize_noise_std(value) -> float:
  """Return a standard deviation supported by the Peikert sampler."""
  try:
    normalized = float(value)
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("sigma must be a finite positive number") from error
  if not math.isfinite(normalized) or normalized <= 0:
    raise ValueError("sigma must be a finite positive number")
  if normalized >= _OPENFHE_KARNEY_THRESHOLD:
    raise ValueError(
        "the OpenFHE Peikert sampler requires sigma < 300; "
        "Karney sampling is not implemented"
    )
  return normalized


########################
# Key Generation
########################
def gen_ternary_uniform_polynomial(
    degree: int, moduli: list[int]
) -> RnsPolynomial:
  """Generate a uniformly random ternary {-1, 0, 1} RNS polynomial.

  Matches OpenFHE's `TernaryUniformGenerator` — each coefficient is
  sampled uniformly from {-1, 0, 1} with probability 1/3 each.
  `gen_rns_polynomial` reduces each value mod q_j per tower so -1 → q_j - 1.

  (Historical note: this previously sampled from {0, 1}, which is BinaryUniform
  and a longstanding deviation from OpenFHE's CKKS encryption noise.)
  """
  coeffs_q = [(-1 if r == 2 else r)
              for r in (secrets.randbelow(3) for _ in range(degree))]
  return gen_rns_polynomial(degree, coeffs_q, moduli)


def gen_uniform_polynomial(degree: int, moduli: list[int]) -> RnsPolynomial:
  """Generate a uniformly random RNS polynomial in R_Q = Z[X] / (Q, X^N+1)."""
  coeffs_q = []
  for q in moduli:
    if int(q) < (1 << 32):
      coeffs_q.append(_sample_uniform_mod_np(degree, int(q)).tolist())
    else:
      coeffs_q.append([secrets.randbelow(q) for _ in range(degree)])
  return RnsPolynomial(degree, moduli, coeffs_q, is_ntt=False)


def _sample_uniform_mod_np(size: int, modulus: int) -> np.ndarray:
  """Samples unbiased residues modulo ``modulus`` from the system CSPRNG.

  This is the vectorized counterpart of ``secrets.randbelow`` for native
  OpenFHE-sized moduli.  Rejection sampling is required: reducing arbitrary
  uint32 values directly would bias residues unless the modulus divides
  2**32.
  """
  size = int(size)
  modulus = int(modulus)
  if size < 0:
    raise ValueError("size must be non-negative")
  if modulus <= 0 or modulus >= (1 << 32):
    raise ValueError("modulus must satisfy 0 < modulus < 2**32")
  if size == 0:
    return np.empty(0, dtype=np.uint64)

  sample_space = 1 << 32
  accept_bound = sample_space - sample_space % modulus
  result = np.empty(size, dtype=np.uint64)
  filled = 0
  while filled < size:
    # At least half of the uint32 range is accepted for every supported
    # modulus.  The small cushion makes a second draw exceptionally rare for
    # the 28--31-bit primes used by composite scaling.
    draw_count = 2 * (size - filled) + 16
    draws = np.frombuffer(
        secrets.token_bytes(4 * draw_count), dtype="<u4"
    ).astype(np.uint64)
    accepted = draws[draws < np.uint64(accept_bound)]
    take = min(size - filled, int(accepted.size))
    if take:
      result[filled:filled + take] = (
          accepted[:take] % np.uint64(modulus)
      )
      filled += take
  return result


@lru_cache(maxsize=32)
def _discrete_gaussian_cdf(sigma: float) -> tuple[float, np.ndarray]:
  """Build OpenFHE's Peikert inversion table for one standard deviation."""
  sigma = normalize_noise_std(sigma)
  fin = int(math.ceil(sigma * _OPENFHE_DGG_TAIL_MULTIPLIER))
  cumulative_values = []
  cumulative_sum = 0.0
  denominator = 2.0 * sigma * sigma
  for value in range(1, fin + 1):
    cumulative_sum += math.exp(-(value * value) / denominator)
    cumulative_values.append(cumulative_sum)
  probability_zero = 1.0 / (2.0 * cumulative_sum + 1.0)
  cumulative = np.asarray(
      [value * probability_zero for value in cumulative_values],
      dtype=np.float64,
  )
  cumulative.setflags(write=False)
  return probability_zero, cumulative


def _sample_discrete_gaussian_from_words(
    random_words: np.ndarray, sigma: float
) -> np.ndarray:
  """Map uint64 CSPRNG words through OpenFHE's Peikert inversion table."""
  random_words = np.asarray(random_words)
  if random_words.ndim != 1 or random_words.dtype != np.uint64:
    raise TypeError("random_words must be a one-dimensional uint64 array")
  probability_zero, cumulative = _discrete_gaussian_cdf(float(sigma))
  uniforms = (random_words >> 11).astype(np.float64) * _INV_2_53
  centered = uniforms - 0.5
  search = np.abs(centered) - probability_zero / 2.0

  samples = np.zeros(random_words.size, dtype=np.int64)
  nonzero = search > 0.0
  nonzero_search = np.minimum(search[nonzero], cumulative[-1])
  magnitudes = (
      np.searchsorted(cumulative, nonzero_search, side="left") + 1
  ).astype(np.int64)
  samples[nonzero] = np.where(
      centered[nonzero] > 0.0, magnitudes, -magnitudes
  )
  return samples


def sample_discrete_gaussian(size: int, sigma: float) -> np.ndarray:
  """Sample OpenFHE-compatible centered discrete-Gaussian integers.

  This is a vectorized port of OpenFHE 1.5.1
  ``DiscreteGaussianGeneratorImpl::GenerateInt``. It uses the same
  ``ceil(12.00610553538285 * sigma)`` support bound and normalized Peikert
  inversion table, with 53-bit uniforms sourced directly from the kernel
  CSPRNG.
  """
  if isinstance(size, (bool, np.bool_)) or not isinstance(
      size, (int, np.integer)
  ):
    raise TypeError("size must be an integer")
  size = int(size)
  if size < 0:
    raise ValueError("size must be non-negative")
  if size == 0:
    _discrete_gaussian_cdf(float(sigma))
    return np.empty(0, dtype=np.int64)

  random_words = np.frombuffer(os.urandom(8 * size), dtype=np.uint64)
  return _sample_discrete_gaussian_from_words(random_words, sigma)


def gen_gaussian_polynomial(
    degree: int, moduli: list[int], sigma: float
) -> RnsPolynomial:
  """Generate a discrete-Gaussian polynomial in R_Q = Z[X] / (Q, X^N+1).

  Each coefficient is independently sampled using OpenFHE's Peikert inversion
  distribution with parameter ``sigma``.

  Args:
    degree: The degree N of the ring R_Q.
    moduli: The list of prime moduli q_i's whose product is Q.
    sigma: The standard deviation of the Gaussian distribution.

  Returns:
    An RNS polynomial with coefficients sampled from a Gaussian distribution.
  """
  coeffs = sample_discrete_gaussian(degree, sigma).tolist()
  return gen_rns_polynomial(degree, coeffs, moduli)


def _validate_private_key(private_key: List[List[int]]) -> Tuple[int, int]:

  if not isinstance(private_key, list) or not private_key:
    raise ValueError("private_key must be a non-empty 2D list")
  num_elements = len(private_key)
  degree = None
  for row in private_key:
    if not isinstance(row, list) or not row:
      raise ValueError(
          "private_key must be a non-empty 2D list with non-empty rows"
      )
    if degree is None:
      degree = len(row)
    elif len(row) != degree:
      raise ValueError(
          "All rows in private_key must have the same length (degree)"
      )
    for coeff in row:
      if not isinstance(coeff, int):
        raise TypeError("All coefficients in private_key must be integers")
  return num_elements, degree  # type: ignore[return-value]


def _mod_q(x: int, q: int) -> int:
  r = x % q
  return r if r >= 0 else r + q


def modulus_switch(
    coefficient: int | List[int],
    cur_moduli: int = 524353,
    target_moduli: int = 1152921504606845473,
) -> int | List[int]:
  """Switch coefficients from cur_moduli to target_moduli using centered representation.

  If c < (cur_moduli + 1) // 2, it is unchanged. Otherwise, it is treated as
  a negative representative (c - cur_moduli) and lifted to Z_{target_moduli}
  by computing target_moduli + (c - cur_moduli).

  Args:
      coefficient: An integer or a list of integers in Z_{cur_moduli}.
      cur_moduli: Current modulus (default: 524353).
      target_moduli: Target modulus (default: 1152921504606845473).

  Returns:
      The switched coefficient with the same container type as input
      (int for int input, List[int] for list input).
  """
  threshold = (cur_moduli + 1) // 2

  def _switch_one(value: int) -> int:
    v = int(value)
    return v if v < threshold else target_moduli + v - cur_moduli

  if isinstance(coefficient, list):
    return [_switch_one(c) for c in coefficient]
  else:
    return _switch_one(int(coefficient))


def gen_evaluation_key(
    private_key: List[List[int]],
    q: int | List[int],
    P: int | List[int] = 1,
    noise_std: float = 3.190000057220458984375,
    noise_scale: int = 1,
    a: Optional[List[List[List[int]]]] = None,
    e: Optional[List[List[List[int]]]] = None,
    dnum: int = 3,
) -> Dict[str, Any]:

  num_elements, degree = _validate_private_key(private_key)

  q_list: List[int] = list(q)
  p_list: List[int] = list(P) if isinstance(P, list) else [int(P)]
  if len(private_key) != len(q_list):
    raise ValueError("private_key must have one row per q modulus (len(q))")
  size_q = len(q_list)
  # size_p = len(p_list)
  # size_qp = size_q + size_p

  sk_q = private_key
  # sOld is s^2 mod q for Q part
  sOld = [
      [_mod_q(sk_q[i][j] * sk_q[i][j], q_list[i]) for j in range(degree)]
      for i in range(size_q)
  ]

  return key_switch_gen(
      sOld=sOld,
      sNew=sk_q,
      q_list=q_list,
      p_list=p_list,
      noise_std=noise_std,
      noise_scale=noise_scale,
      a=a,
      e=e,
      dnum=dnum,
  )


def key_switch_gen(
    sOld: List[List[int]],
    sNew: List[List[int]],
    q_list: List[int],
    p_list: List[int],
    noise_std: float = 3.190000057220458984375,
    noise_scale: int = 1,
    a: List[List[List[int]]] | None = None,
    e: List[List[List[int]]] | None = None,
    dnum: int = 3,
) -> Dict[str, Any]:
  """Construct evaluation key parts given secret forms sOld (Q) and sNew (time domain).

  Args:
      sOld: Secret squared residues modulo each q in Q, shape [|Q|][N].
      sNew: Secret in time domain for Q basis, shape [|Q|][N].
      q_list: List of moduli forming Q.
      p_list: List of moduli forming P.
      noise_std: Standard deviation for error sampling.
      noise_scale: Integer multiplier applied to error samples.
      a: Optional pre-specified a samples per part and limb.
      e: Optional pre-specified e samples per part and limb.
      dnum: Number of partitions over Q for HYBRID scheme.

  Returns:
      Dict with keys: "a", "b", "modulus", "P", and "shape".
  """
  degree = len(sNew[0])
  moduli_list = list(q_list) + list(p_list)
  pack_uint32 = util._correct_check(moduli_list, degree)
  size_q = len(q_list)
  size_qp = len(moduli_list)
  use_numpy = (
      os.environ.get("CROSS_NP_KEYGEN", "1") != "0"
      and pack_uint32
  )

  # Convert the new secret to coefficient form. Public transform helpers
  # preserve the input container and select their own exact backend.
  s_out = []
  for limb, q in zip(sNew, q_list, strict=True):
    modulus = int(q)
    values = (
        util.uint64_residues_np(limb, modulus)
        if use_numpy
        else [int(value) for value in limb]
    )
    psi = util.root_of_unity(2 * degree, modulus)
    reversed_values = util.bit_reverse_array(values)
    s_out.append(
        util.intt_negacyclic_bit_reverse(reversed_values, modulus, psi)
    )

  # Extend the centered q0 representatives to P. The NumPy path reduces
  # before narrowing so negative lifts cannot wrap through 2**64.
  q0 = int(q_list[0])
  threshold = (q0 + 1) // 2
  if use_numpy:
    base0 = np.asarray(s_out[0], dtype=np.int64)
    s_qp = list(s_out)
  else:
    base0 = [int(value) for value in s_out[0]]
    s_qp = [[int(value) for value in row] for row in s_out]
  for q_p in p_list:
    modulus = int(q_p)
    if use_numpy:
      lifted = np.where(base0 < threshold, base0, base0 - q0 + modulus)
      s_qp.append(util.uint64_residues_np(lifted, modulus))
    else:
      s_qp.append(list(modulus_switch(base0, q0, modulus)))

  s_qp_eva = []
  for limb, q in zip(s_qp, moduli_list, strict=True):
    modulus = int(q)
    psi = util.root_of_unity(2 * degree, modulus)
    transformed = util.ntt_negacyclic_bit_reverse(limb, modulus, psi)
    s_qp_eva.append(util.bit_reverse_array(transformed))

  p_product = math.prod(int(p) for p in p_list)
  p_mod_q = [p_product % int(q) for q in q_list]

  num_per_part_q = (size_q + dnum - 1) // dnum
  num_part_q = math.ceil(size_q / num_per_part_q)

  a_parts = []
  b_parts = []
  noise_scale = int(noise_scale)
  for part in range(num_part_q):
    start_idx = num_per_part_q * part
    end_idx = min(size_q, start_idx + num_per_part_q)
    if a is not None:
      a_rows = (
          np.stack([
              util.uint64_residues_np(a[part][i], modulus)
              for i, modulus in enumerate(moduli_list)
          ])
          if use_numpy
          else a[part]
      )
    elif use_numpy:
      # OpenFHE samples independent evaluation-domain residues per CRT limb.
      a_rows = np.stack([
          _sample_uniform_mod_np(degree, int(modulus))
          for modulus in moduli_list
      ])
    else:
      # DUG samples independent residues for each CRT limb directly in the
      # evaluation representation; no coefficient-domain NTT is applied.
      a_rows = gen_uniform_polynomial(degree, moduli_list).coeffs

    if e is not None:
      e_rows = (
          np.stack([
              util.uint64_residues_np(e[part][i], modulus)
              for i, modulus in enumerate(moduli_list)
          ])
          if use_numpy
          else e[part]
      )
    else:
      base_e = sample_discrete_gaussian(degree, noise_std).tolist()
      if use_numpy:
        e_rows = np.empty((size_qp, degree), dtype=np.uint64)
        for i, modulus in enumerate(moduli_list):
          modulus = int(modulus)
          row = util.uint64_residues_np(base_e, modulus)
          psi = util.root_of_unity(2 * degree, modulus)
          e_rows[i] = util.bit_reverse_array(
              util.ntt_negacyclic_bit_reverse(row, modulus, psi)
          )
      else:
        e_sample = gen_rns_polynomial(degree, base_e, moduli_list)
        e_rows = [
            util.bit_reverse_array(
                util.ntt_negacyclic_bit_reverse(
                    e_sample.coeffs[i],
                    int(modulus),
                    util.root_of_unity(2 * degree, int(modulus)),
                )
            )
            for i, modulus in enumerate(moduli_list)
        ]

    b_rows = (
        np.empty((size_qp, degree), dtype=np.uint64)
        if use_numpy
        else []
    )
    for i, modulus_i in enumerate(moduli_list):
      modulus_i = int(modulus_i)
      a_row = a_rows[i]
      e_row = e_rows[i]
      s_row = s_qp_eva[i]
      # HYBRID key switching splits the Q basis across dnum parts. Only the
      # Q limbs assigned to this part receive the P*sOld gadget term; other Q
      # limbs and every P limb contain only -a*sNew plus the error term.
      in_partition = start_idx <= i < end_idx
      if use_numpy:
        modulus = np.uint64(modulus_i)
        a_np = a_row % modulus
        s_np = s_row % modulus
        e_np = e_row % modulus
        ns = np.uint64(noise_scale % modulus_i)
        neg_as = ((modulus - a_np) % modulus * s_np) % modulus
        acc = (neg_as + (ns * e_np) % modulus) % modulus
        if in_partition:
          p_i = np.uint64(p_mod_q[i])
          sold_np = util.uint64_residues_np(sOld[i], modulus_i)
          acc = (acc + (p_i * sold_np) % modulus) % modulus
        b_rows[i] = acc
      else:
        b_row = []
        for j in range(degree):
          value = (
              _mod_q(-int(a_row[j]) * int(s_row[j]), modulus_i)
              + _mod_q(noise_scale * int(e_row[j]), modulus_i)
          )
          if in_partition:
            value += _mod_q(
                p_mod_q[i] * int(sOld[i][j]), modulus_i
            )
          b_row.append(_mod_q(value, modulus_i))
        b_rows.append(b_row)
    a_parts.append(a_rows)
    b_parts.append(b_rows)

  # Compact the guarded native path. The scalar fallback deliberately retains
  # Python integers so unsupported wider moduli cannot be truncated to uint32.
  result_a = a_parts if a is None else a
  result_b = b_parts
  if pack_uint32:
    result_a = np.stack([
        np.stack([
            util.uint64_residues_np(part[i], modulus)
            for i, modulus in enumerate(moduli_list)
        ])
        for part in result_a
    ]).astype(np.uint32)
    result_b = np.asarray(result_b, dtype=np.uint32)
  else:
    result_a = [
        [
            [
                int(value) % int(modulus)
                for value in part[i]
            ]
            for i, modulus in enumerate(moduli_list)
        ]
        for part in result_a
    ]
    result_b = [
        [
            [
                int(value) % int(modulus)
                for value in part[i]
            ]
            for i, modulus in enumerate(moduli_list)
        ]
        for part in result_b
    ]
  return {
      "a": result_a,
      "b": result_b,
      "modulus": {"Q": list(q_list), "P": list(p_list)},
      "P": list(p_list),
      "shape": (num_part_q, size_qp, degree),
  }


# Backwards-compatible public names for the canonical utility implementations.
find_automorphism_index_2n_complex = util.find_automorphism_index_2n_complex
precompute_rotation_key_map = util.precompute_auto_map


def gen_rotation_key(
    sk,
    original_moduli,
    extend_moduli,
    rot_index: int,
    dnum=3,
    noise_std=3.190000057220458984375,
    noise_scale=1,
    a=None,
    e=None,
) -> Dict[int, Any]:
  n = len(sk[0])
  result = find_automorphism_index_2n_complex(rot_index, 2 * n)
  key_map_idx = util.modinv(result, 2 * n)
  target_order = precompute_rotation_key_map(n, key_map_idx)

  # Object dtype preserves arbitrary-precision secret coefficients while the
  # permutation avoids a degree*|Q| Python-int round trip per rotation key.
  sk_rot = np.asarray(sk, dtype=object)[
      :, np.asarray(target_order, dtype=np.intp)
  ]

  ek = key_switch_gen(
      sk,
      sNew=sk_rot,
      q_list=original_moduli,
      p_list=extend_moduli,
      noise_std=noise_std,
      noise_scale=noise_scale,
      a=a,
      e=e,
      dnum=dnum,
  )
  return {rot_index: ek}


def gen_pke_pair(
    q_towers: List[int],
    p_towers: List[int],
    degree: int,
    noise_std: float = 3.190000057220458984375,
    noise_scale: int = 1,
    a_ref=None,
    s_ref=None,
    e_ref=None,
) -> Dict[str, Any]:
  """Generate a PKE pair.

  Args:
      q_towers: List of moduli forming Q.
      p_towers: List of moduli forming P.
      degree: The degree N of the ring R_Q.
      noise_std: Standard deviation for error sampling.
      noise_scale: Integer multiplier applied to error samples.

  Returns:
      Dict with keys: "public_key", "secret_key".
  """
  moduli_list = q_towers + p_towers
  s = gen_ternary_uniform_polynomial(degree, moduli_list)
  s = [
      util.bit_reverse_array(
          util.ntt_negacyclic_bit_reverse(
              s.coeffs[i],
              modulus_i,
              util.root_of_unity(int(degree << 1), modulus_i),
          )
      )
      for i, modulus_i in enumerate(moduli_list)
  ]
  s = s_ref if s_ref is not None else s
  a = gen_uniform_polynomial(degree, moduli_list)
  a = [
      util.bit_reverse_array(
          util.ntt_negacyclic_bit_reverse(
              a.coeffs[i],
              modulus_i,
              util.root_of_unity(int(degree << 1), modulus_i),
          )
      )
      for i, modulus_i in enumerate(moduli_list)
  ]
  a = a_ref if a_ref is not None else a
  e = gen_gaussian_polynomial(degree, moduli_list, sigma=noise_std)
  e = [
      util.bit_reverse_array(
          util.ntt_negacyclic_bit_reverse(
              e.coeffs[i],
              modulus_i,
              util.root_of_unity(int(degree << 1), modulus_i),
          )
      )
      for i, modulus_i in enumerate(moduli_list)
  ]
  e = e_ref if e_ref is not None else e

  b = [
      [
          _mod_q(
              _mod_q(e[i][j] * noise_scale, moduli_list[i])
              - _mod_q(a[i][j] * s[i][j], moduli_list[i]),
              moduli_list[i],
          )
          for j in range(degree)
      ]
      for i in range(len(moduli_list))
  ]
  s = s[: len(q_towers)]
  return {
      "public_key": [b, a],
      "secret_key": s,
  }

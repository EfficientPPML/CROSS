"""Global Configuration for running jaxite_word.

The default input data type is 64 bit integer.
"""

import math
import operator
import jax
import numpy as np
import re
import os
import json
import gzip
import jax.numpy as jnp
from typing import Any, Callable, List, Tuple
import copy

gcd = math.gcd

# Capture existing profile run directories so we can identify the new one.
profile_root = os.path.join("./log/xprof", "plugins", "profile")
try:
  pre_existing_dirs = set(os.listdir(profile_root)) if os.path.isdir(profile_root) else set()
except Exception:
  pre_existing_dirs = set()

####################################
# Utility Functions
####################################
def _square_like_mesh_shape(device_count: int) -> Tuple[int, int]:
  """Return a near-square 2D mesh shape that covers all available devices."""
  if device_count <= 0:
    raise ValueError("At least one device is required to build a mesh.")
  sqrt_devices = math.isqrt(device_count)
  for dim0 in range(sqrt_devices, 0, -1):
    if device_count % dim0 == 0:
      return dim0, device_count // dim0
  return 1, device_count


def create_sharding(shard_dim=None):
  """Create default batch and replicated shardings for the current device mesh.

  Args:
    shard_dim: length of the array axis that will carry the ``('x', 'y')``
      partition, when known. The mesh is capped to the largest device count
      that divides it, because a spec naming both mesh axes splits that axis
      ``x * y`` ways and JAX rejects a partition that does not divide evenly.
      Without this, an 8-chip host turns a length-4 axis into
      ``... does not evenly divide the dimension size 4`` while the same code
      works on a 4-chip host. ``None`` keeps the full device mesh.
  """
  available_devices = jax.devices()
  if not available_devices:
    raise RuntimeError("No devices available for sharding test.")
  device_count = len(available_devices)
  if shard_dim is not None:
    if (
        isinstance(shard_dim, bool)
        or not isinstance(shard_dim, int)
        or shard_dim < 1
    ):
      raise ValueError(f"shard_dim must be a positive int, got {shard_dim!r}.")
    device_count = max(
        count for count in range(1, device_count + 1) if shard_dim % count == 0
    )
  if device_count == 8:
    mesh_shape = (2, 4)
  elif device_count == 4:
    mesh_shape = (2, 2)
  elif device_count == 2:
    mesh_shape = (2, 1)
  else:
    mesh_shape = _square_like_mesh_shape(len(available_devices))

  # Auto axis types: XLA infers the sharding of intermediate reshapes.
  # jax.make_mesh defaults to Explicit as of JAX 0.9, which rejects any
  # reshape whose output sharding it cannot derive -- including the one
  # conv_general_dilated's batching rule emits when vmapped over a
  # sharded batch axis (ntt_mm.matmul_conv_flexible_kernel).
  mesh = jax.make_mesh(
      mesh_shape,
      ('x', 'y'),
      axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_shape),
      devices=available_devices[:mesh_shape[0] * mesh_shape[1]],
  )
  # Deliberately NOT jax.sharding.set_mesh(mesh): that sets a PROCESS-GLOBAL
  # mesh that is never restored, and every later jax.pmap in the same
  # process then fails with a context-mesh mismatch. Callers receive the
  # mesh and pass it explicitly where they need it.

  partition_spec = jax.sharding.PartitionSpec
  return mesh, partition_spec


def num_bits(x: int) -> int:
  """Returns the number of bits in x."""
  return x.bit_length() - 1


def is_power_of_two(x: int) -> bool:
  """Returns True if x is a power of two."""
  return x > 0 and (x & (x - 1)) == 0


def _valid_ntt_parameters(modulus, degree, psi):
  """Return whether parameters define a radix-2 negacyclic NTT."""
  return (
      modulus > 2
      and is_power_of_two(degree)
      and math.gcd(degree, modulus) == 1
      and pow(psi, degree, modulus) == modulus - 1
  )


def _correct_check(moduli=None, degree=None, psi=None):
  """Return whether inputs fit the exact NumPy host-arithmetic envelope.

  The vectorized NTT and key-generation arithmetic uses ``numpy.uint64``.
  Keeping every modulus below 2**31 makes all products smaller than 2**62.
  A supplied degree must be a positive power of two, and a supplied ``psi``
  must be a primitive negacyclic root for that degree and modulus.

  Unsupported values return ``False`` rather than leaking conversion or
  modular-arithmetic exceptions to callers performing a dispatch check.
  """

  try:
    if moduli is None:
      checked_moduli = ()
    else:
      try:
        checked_moduli = (operator.index(moduli),)
      except TypeError:
        checked_moduli = tuple(operator.index(value) for value in moduli)
    checked_degree = None if degree is None else operator.index(degree)
    checked_psi = None if psi is None else operator.index(psi)
  except (TypeError, ValueError, OverflowError):
    return False

  if any(modulus <= 2 or modulus >= (1 << 31)
         for modulus in checked_moduli):
    return False
  if checked_degree is not None and (
      isinstance(degree, bool) or not is_power_of_two(checked_degree)
  ):
    return False
  if checked_psi is None:
    return True
  if checked_degree is None or len(checked_moduli) != 1:
    return False

  return _valid_ntt_parameters(
      checked_moduli[0], checked_degree, checked_psi
  )


def to_tuple(a):
  """Create to convert numpy array into tuple."""
  try:
    return tuple(to_tuple(i) for i in a)
  except TypeError:
    return a


####################################
# BAT (Basis Aligned Transformation)
####################################

NUM_BYTES = 4
BYTE_SHIFTS = jnp.arange(NUM_BYTES, dtype=jnp.uint32) * 8


def to_bytes(values):
  """Bitcast uint32 values to a trailing little-endian byte dimension."""
  return jax.lax.bitcast_convert_type(
      jnp.asarray(values, dtype=jnp.uint32), jnp.uint8
  )


def reconstruct(byte_partials):
  """Reconstruct integer values from trailing byte-position partials."""
  shifts = BYTE_SHIFTS.reshape(
      (1,) * (byte_partials.ndim - 1) + (NUM_BYTES,)
  )
  return jnp.sum(
      byte_partials.astype(jnp.uint64) << shifts, axis=-1
  )


def matmul(lhs, rhs, subscripts, *, flatten_lhs_bytes=False):
  """Run a uint8 BAT einsum and reconstruct uint64 outputs.

  Set ``flatten_lhs_bytes`` when the logical contraction axis combines the
  input's final modulus and byte axes, as in basis conversion.
  """
  lhs = jnp.asarray(lhs, dtype=jnp.uint32)
  if flatten_lhs_bytes:
    # Array.view expresses the combined modulus-byte contraction directly and
    # avoids a separate reshape in StableHLO.
    lhs_bytes = lhs.view(jnp.uint8)
  else:
    # NTT contracts the byte lane separately, so preserve it as a new axis.
    lhs_bytes = jax.lax.bitcast_convert_type(lhs, jnp.uint8)
  products = jnp.einsum(
      subscripts,
      lhs_bytes,
      rhs,
      preferred_element_type=jnp.uint32,
  )
  return jnp.sum(
      products.astype(jnp.uint64) << BYTE_SHIFTS, axis=-1
  )


def shifted_mod_bytes(values, moduli):
  """Return raw ``(input_byte, *values.shape, output_byte)`` BAT data."""
  return to_bytes(_shifted_mod_words(values, moduli, jnp))


def _shifted_mod_words(values, moduli, array_module):
  """Shared BAT-control arithmetic for NumPy and JAX array namespaces."""
  values = array_module.asarray(values, dtype=array_module.uint64)
  moduli = array_module.asarray(moduli, dtype=array_module.uint64)
  shifted = array_module.stack(
      [values << (8 * byte_index) for byte_index in range(NUM_BYTES)],
      axis=0,
  )
  return (shifted % moduli).astype(array_module.uint32)


def shifted_mod_bytes_host(values, moduli):
  """Build static BAT controls on the host without consuming device memory."""
  words = _shifted_mod_words(values, moduli, np)
  # Force a little-endian word view so the control format is host-independent
  # and byte-identical to JAX's uint32 bitcast.
  words = np.ascontiguousarray(words.astype('<u4', copy=False))
  return words.view(np.uint8).reshape(
      words.shape + (NUM_BYTES,)
  )


def slice_first_k_along_axis0(arrays, k):
  """
  Given an iterable of array-like or sequence objects, return a tuple where
  each element is the slice of the original object taking the first k entries
  along axis 0 (i.e., obj[:k]).

  Example:
    (s_tuple, s_w_tuple, w_tuple, m_tuple) ->
    (s_tuple[:k], s_w_tuple[:k], w_tuple[:k], m_tuple[:k])
  """
  return tuple(arr[:k] for arr in arrays)


def slice_k_to_end_along_axis0(arrays, k):
  """
  Given an iterable of array-like or sequence objects, return a tuple where
  each element is the slice of the original object taking the k to end entries
  along axis 0 (i.e., obj[k:]).

  Example:
    (s_tuple, s_w_tuple, w_tuple, m_tuple) ->
    (s_tuple[k:], s_w_tuple[k:], w_tuple[k:], m_tuple[k:])
  """
  return tuple(arr[k:] for arr in arrays)


def slice_kth_along_axis0(arrays, k):
  """
  Given an iterable of array-like or sequence objects, return a tuple where
  each element is the slice of the original object taking the first k entries
  along axis 0 (i.e., obj[k]).

  Example:
    (s_tuple, s_w_tuple, w_tuple, m_tuple) ->
    (s_tuple[k], s_w_tuple[k], w_tuple[k], m_tuple[k])
  """
  return tuple(arr[k] for arr in arrays)


def slice_0_to_k0_to_k1_along_axis0(arrays, k0, k1):
  """
  Given an iterable of array-like or sequence objects, return a tuple where
  each element is the slice of the original object from k0 to k1 along axis 0 (i.e., obj[k0:k1]).

  Example:
    (s_tuple, s_w_tuple, w_tuple, m_tuple) ->
    (s_tuple[k0:k1], s_w_tuple[k0:k1], w_tuple[k0:k1], m_tuple[k0:k1])
  """
  if isinstance(arrays[0], jnp.ndarray):
    return tuple(jnp.concatenate([x[:k0], x[k1:]]) for x in arrays)
  elif isinstance(arrays, tuple):
    return tuple([arr[:k0] + arr[k1:] for arr in arrays])
  elif isinstance(arrays, list):
    return [arr[:k0] + arr[k1:] for arr in arrays]
  else:
    raise ValueError(f"Unsupported type: {type(arrays)}")


def slice_k0_to_k1_axis0(arrays, k0, k1):
  """
  Given an iterable of array-like or sequence objects, return a tuple where
  each element is the slice of the original object from k0 to k1 along axis 0 (i.e., obj[k0:k1]).

  Example:
    (s_tuple, s_w_tuple, w_tuple, m_tuple) ->
    (s_tuple[k0:k1], s_w_tuple[k0:k1], w_tuple[k0:k1], m_tuple[k0:k1])
  """
  return tuple(arr[k0:k1] for arr in arrays)

####################################
# Math Functions
####################################
def extended_gcd(a, b):
  """Return a tuple of (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
  if b == 0:
    return (a, 1, 0)
  else:
    g, x, y = extended_gcd(b, a % b)
    return (g, y, x - (a // b) * y)


def modinv_manual(x, q):
  """Returns the inverse of x mod q."""
  g, x, _ = extended_gcd(x, q)
  if g != 1:
    raise Exception(f'Modular inverse does not exist for {x} modulo {q}')
  else:
    return x % q


def modinv(x: int, q: int) -> int:
  """Returns the inverse of x mod q."""
  return int(pow(x, -1, q))


def prime_factors(n):
  """Return the set of prime factors of n."""
  factors = set()
  # Divide out factors of 2
  while n % 2 == 0:
    factors.add(2)
    n //= 2
  # Check odd factors from 3 to sqrt(n)
  p = 3
  while p**2 <= n:
    while n % p == 0:
      factors.add(p)
      n //= p
    p += 2
  if n > 1:
    factors.add(n)
  return factors


def find_generator(q):
  """Find a primitive root modulo q.

  Args:
    q (int): The prime modulus.

  Returns:
    A generator of GF(q)^*.

  Raises:
    ValueError: If no generator is found, indicating q is not prime.
  """
  phi = q - 1
  factors = prime_factors(phi)

  # Test candidates from 2 to q-1.
  for g in range(2, q):
    is_generator = all(pow(g, phi // p, q) != 1 for p in factors)
    if is_generator:
      return g
  raise ValueError("No generator found, check that q is prime.")


####################################
# Parameters Generation
####################################
_root_of_unity_cache = {}


def root_of_unity(m: int, q: int) -> int:
    """Canonical primitive m-th root of unity modulo q that **works with NTT**.

    Args:
      m (int): The order of the root of unity.
      q (int): The prime modulus.

    Returns:
      int: The canonical primitive m-th root of unity modulo q.

    Usage:
      root_of_unity(16, 134219681) # This works with NTT.
      computed_psi = [root_of_unity(m, q) for q in original_modulus]

    The result is deterministic per (m, q) and the candidate scan is O(m)
    modexps, so it is memoized -- key generation calls this once per tower
    per sample per key (tens of thousands of identical calls at large N).
    """
    cached = _root_of_unity_cache.get((m, q))
    if cached is not None:
      return cached
    if m <= 0 or (q - 1) % m != 0:
      raise ValueError("q-1 must be divisible by positive m")
    # Step 1: multiplicative generator of Z_q^*
    g = find_generator(q)
    # Step 2: raise to (q-1)/m to get an m-th root candidate
    r = pow(g, (q - 1) // m, q)
    # Step 3: among r^k with gcd(k,m)=1, pick the minimal value whose order is exactly m
    # For m=2^t, order check is psi^(m/2) == q-1 (i.e., == -1 mod q)
    candidates = []
    half = m // 2
    for k in range(1, m):
        if gcd(k, m) != 1:
            continue
        psi = pow(r, k, q)
        if pow(psi, half, q) == q - 1 and pow(psi, m, q) == 1:
            candidates.append(psi)
    if not candidates:
      raise ValueError("No primitive m-th root found")
    result = min(candidates)
    _root_of_unity_cache[(m, q)] = result
    return result


def any_primitive_root_of_unity(n, q):
  """Canonical primitive m-th root of unity modulo q that **may not work with NTT**.

    Args:
      m (int): The order of the root of unity.
      q (int): The prime modulus.

    Returns:
      int: The canonical primitive m-th root of unity modulo q.

    Usage:
      root_of_unity(16, 134219681) # This may not work with NTT.
      computed_psi = [root_of_unity(m, q) for q in original_modulus]
    """
  if (q - 1) % n != 0:
    raise ValueError(
        "n must divide q-1 for a primitive n-th root of unity to exist."
    )

  # Find a generator g of GF(q)^* (a primitive element).
  g = find_generator(q)
  # Compute omega = g^((q-1)/n) mod q.
  exponent = (q - 1) // n
  omega = pow(g, exponent, q)

  # Optional: Verify that omega is indeed of order n.
  if pow(omega, n, q) != 1:
    raise ValueError("Something went wrong: omega^n != 1")
  # Check that no smaller positive exponent gives 1.
  for d in range(1, n):
    if n % d == 0 and pow(omega, d, q) == 1:
      raise ValueError(
          "Found an exponent d < n with omega^d == 1, so omega is not"
          " primitive."
      )

  return omega


def compute_barrett_mu(modulus):
  """Compute the Barrett reduction constant mu for a given modulus m.

  Args:
    modulus (int): The modulus.

  Returns:
    tuple: (mu, k) where mu is the precomputed constant and k is the number of
    digits in base b.
  """
  # k is the smallest integer such that m < b^k.
  b = 2 ** (math.floor(math.log(modulus)) + 1)
  # For m < 2^64, k will be 1.
  k_val = math.floor(math.log(modulus, b)) + 1

  # Compute mu = floor(b^(2k) / m)
  barrett_mu = (b ** (2 * k_val)) // modulus
  return barrett_mu, k_val


def compute_QHatInvModq_QHatModp(original_moduli, target_moduli, perf_test=False):
  """
  Given a list of moduli original_moduli, compute QHatInvModq.
  Input:
    - original_moduli (list[int]):
            The list of primes (moduli) defining the original CRT basis (Q).
    - target_moduli (list[int]):
            The list of primes (moduli) defining the target CRT basis (P).

  For each modulus q_i, compute:
    - Qhat_i = Q // q_i
    - QHatInvModq[i] = modular inverse of Qhat_i modulo q_i
    - QHatModp: Precomputed Q̂ modulo each prime in P. Used in approximate basis switching.
  """
  if perf_test:
      sizeP = len(original_moduli)
      sizeQ = len(target_moduli)
      # Random arrays with matching shapes/dtypes
      PInvModq = random_parameters((sizeQ,), target_moduli, dtype=jnp.uint32).tolist()
      QHatInvModq = random_parameters((sizeP,), target_moduli, dtype=jnp.uint32).tolist()
      QHatModp = random_parameters((sizeP, sizeQ), [min(target_moduli + original_moduli)], dtype=jnp.uint32).tolist()
      return to_tuple((QHatInvModq, QHatModp))
  else:
    Q = 1
    for qi in original_moduli:
      Q *= qi

    QHatInvModq = []
    QHat = []
    for qi in original_moduli:
      Qhat_i = Q // qi
      inv = modinv(Qhat_i, qi)
      QHat.append(Qhat_i)
      QHatInvModq.append(inv)

    QHatModp = []
    for i in range(len(original_moduli)):
      QHatModp_sgl = []
      for j in range(len(target_moduli)):
        QHatModp_sgl.append(QHat[i] % target_moduli[j])
      QHatModp.append(QHatModp_sgl)

  return QHatInvModq, QHatModp


def approx_mod_down_control_generation(current_moduli, target_moduli, perf_test=False):
  if perf_test:
    PInvModq = random_parameters((len(target_moduli),), target_moduli, dtype=jnp.uint32).tolist()
  else:
    P = 1
    for moduli in current_moduli:
      P *= moduli
    PInvModq = [modinv(P, q) for q in target_moduli]
  overall_moduli =  current_moduli + target_moduli
  QHatInvModq, QHatModp = compute_QHatInvModq_QHatModp(current_moduli, target_moduli, perf_test=perf_test)

  return PInvModq, len(overall_moduli) - len(target_moduli), len(overall_moduli) - len(current_moduli), QHatInvModq, QHatModp


def compute_powers_of_psi(ring_dim, moduli, perf_test=False):
  """Computes powers of psi for the given moduli."""
  if perf_test:
    return random_parameters((len(moduli), ring_dim), moduli, dtype=jnp.uint64)
  else:
    psi = [root_of_unity(2 * ring_dim, q) for q in moduli]
    return jnp.array(
        [
            [pow(psi[idx], i, moduli[idx]) for i in range(ring_dim)]
            for idx in range(len(moduli))
        ],
        jnp.uint64,
    )


def is_prime_deterministic(n):
    """
    Deterministic primality test for n < 2^64.
    Uses Trial Division for speed + Deterministic Miller-Rabin for correctness.
    """
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False

    # 1. SPEED OPTIMIZATION: Trial Division
    # Check divisibility by small primes to fail fast on obvious composites.
    # This filters out ~85% of candidates without expensive modular exponentiation.
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    for p in small_primes:
        if n == p: return True
        if n % p == 0: return False

    # 2. DETERMINISTIC MILLER-RABIN
    # For n < 2^64, verifying these specific bases guarantees primality.
    # No randomness involved.
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # Bases required for deterministic check up to 2^64
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    for a in bases:
        if a >= n: break

        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue

        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False # Composite

    return True # Prime


def find_moduli_ntt(total_number, precision, ntt_length):
    """
    Deterministically finds the largest valid NTT moduli.

    Args:
      total_number: Number of moduli to find.
      precision: Bit-width (e.g., 60 for < 2^60).
      ntt_length: The required N-th root of unity (e.g., 1024).
    """
    overall_moduli = []

    # Upper bound
    limit = 2**precision

    # Start search from the largest possible k
    # P = k * ntt_length + 1
    k = (limit - 1) // ntt_length

    while len(overall_moduli) < total_number and k > 0:
        candidate_p = k * ntt_length + 1

        # Check candidate
        if is_prime_deterministic(candidate_p):
            overall_moduli.append(candidate_p)

        k -= 1

    return overall_moduli


def _integer(name, value):
  """Return an exact integer, rejecting booleans and lossy coercions."""
  if isinstance(value, (bool, np.bool_)):
    raise TypeError(f"{name} must be an integer")
  try:
    return int(operator.index(value))
  except TypeError as error:
    raise TypeError(f"{name} must be an integer") from error


def _moduli(name, values):
  try:
    result = [_integer(f"{name} entries", value) for value in values]
  except TypeError as error:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
      raise TypeError(f"{name} must be an iterable of integers") from error
    raise
  if any(value <= 1 for value in result):
    raise ValueError(f"{name} must contain integers greater than 1")
  return result


def q_partition_products(q_towers, dnum):
  """Return exact products for consecutive HYBRID Q partitions."""
  q_towers = _moduli("q_towers", q_towers)
  dnum = _integer("dnum", dnum)
  if dnum <= 0:
    raise ValueError(f"dnum must be positive, got {dnum}")
  if not q_towers:
    return []
  alpha = (len(q_towers) + dnum - 1) // dnum
  return [
      math.prod(q_towers[start:start + alpha])
      for start in range(0, len(q_towers), alpha)
  ]


def validate_barrett_bconv_moduli(q_towers, p_towers, dnum):
  """Validate static invariants shared by HYBRID BConv implementations."""
  q_towers = _moduli("q_towers", q_towers)
  p_towers = _moduli("p_towers", p_towers)
  dnum = _integer("dnum", dnum)
  if dnum <= 0:
    raise ValueError(f"dnum must be positive, got {dnum}")
  if not q_towers or not p_towers:
    raise ValueError("q_towers and p_towers must be non-empty")
  if len(set(q_towers + p_towers)) != len(q_towers) + len(p_towers):
    raise ValueError("Q and P moduli must be pairwise distinct")
  if any(modulus >= 1 << 31 for modulus in q_towers + p_towers):
    raise ValueError("Barrett/BConv requires every modulus < 2^31")

  partitions = q_partition_products(q_towers, dnum)
  if math.prod(p_towers) < max(partitions):
    raise ValueError(
        "P-tower product does not cover the largest Q partition"
    )


def compute_num_p_towers(q_towers, dnum, aux_bits=None):
  """Compute the required number of P-tower primes for HYBRID key-switching.

  In the HYBRID scheme, the Q modulus chain is split into `dnum` partitions.
  The approximate basis extension during key-switching requires:

      P_product >= max_partition(product of Q-moduli in partition)

  This function computes the number of P-tower primes (each of `aux_bits`
  size) from the exact bit length of the largest Q partition.

  Matches OpenFHE's EstimateLogP logic (rns-cryptoparameters.cpp:407-466).

  Args:
    q_towers: List of Q-tower moduli (integers).
    dnum: Number of key-switch partitions (numLargeDigits / numPartQ).
    aux_bits: Bit size of each P-tower prime. If None, uses one less than
        the largest Q-limb storage width, matching OpenFHE's
        ``registerWordSize - 1`` convention.

  Returns:
    sizeP: Number of P-tower primes required.
  """
  q_towers = _moduli("q_towers", q_towers)
  dnum = _integer("dnum", dnum)
  partitions = q_partition_products(q_towers, dnum)
  if not partitions:
    return 0

  if aux_bits is None:
    aux_bits = max(q.bit_length() for q in q_towers) - 1
  else:
    aux_bits = _integer("aux_bits", aux_bits)
  if aux_bits <= 0:
    raise ValueError(f"aux_bits must be positive, got {aux_bits}")

  return math.ceil(max(partitions).bit_length() / aux_bits)


def generate_p_towers(q_towers, dnum, degree, aux_bits=None):
  """Generate P-tower primes for HYBRID key-switching.

  Automatically computes the required number of P-tower primes based on
  the Q-tower configuration and dnum, then generates NTT-friendly primes
  that don't overlap with Q-towers.

  Matches OpenFHE's P-prime generation (rns-cryptoparameters.cpp:151-177).

  Args:
    q_towers: List of Q-tower moduli (integers).
    dnum: Number of key-switch partitions.
    degree: Ring polynomial degree (N). Primes must satisfy p ≡ 1 (mod 2N).
    aux_bits: Bit size of each P-tower prime. If None, uses one less than
        the largest Q-limb storage width.

  Returns:
    p_towers: List of P-tower primes.
  """
  q_towers = _moduli("q_towers", q_towers)
  if not q_towers:
    raise ValueError("q_towers must be non-empty")
  if len(set(q_towers)) != len(q_towers):
    raise ValueError("q_towers must contain pairwise distinct moduli")
  if any(modulus >= 1 << 31 for modulus in q_towers):
    raise ValueError("P-tower generation requires every Q modulus < 2^31")
  dnum = _integer("dnum", dnum)
  degree = _integer("degree", degree)
  if degree <= 0 or not is_power_of_two(degree):
    raise ValueError("degree must be a positive power of two")
  if aux_bits is None:
    aux_bits = max(q.bit_length() for q in q_towers) - 1
  else:
    aux_bits = _integer("aux_bits", aux_bits)
  if aux_bits <= 0:
    raise ValueError(f"aux_bits must be positive, got {aux_bits}")
  if aux_bits > 31:
    raise ValueError("use at most 31-bit P primes")

  size_p = compute_num_p_towers(q_towers, dnum, aux_bits)
  target_product = max(q_partition_products(q_towers, dnum))
  ntt_length = 2 * degree  # Primes must satisfy p ≡ 1 (mod 2N)
  q_set = set(q_towers)

  # Generate primes: search downward from 2^aux_bits, skip Q-tower primes
  p_towers = []
  p_product = 1
  limit = 2 ** aux_bits
  k = (limit - 1) // ntt_length

  while (len(p_towers) < size_p or p_product < target_product) and k > 0:
    candidate = k * ntt_length + 1
    if candidate not in q_set and is_prime_deterministic(candidate):
      p_towers.append(candidate)
      p_product *= candidate
    k -= 1

  if len(p_towers) < size_p or p_product < target_product:
    raise ValueError(
        "Could not generate P towers covering the largest Q partition: "
        f"initial lower-bound estimate {size_p}, found {len(p_towers)}, "
        f"product(P)={p_product}, required>={target_product}. Try increasing "
        f"aux_bits (currently {aux_bits})."
    )

  return p_towers


def gamma_beta_calculation(moduli_list, perf_test=False):
  """Computes gamma and beta parameters for approximate modulus switching.

  These parameters are used in the "modulus down" operation within a CRT
  context, specifically for the last modulus in the `moduli_list`.

  Args:
    moduli_list: A list of prime moduli. The last modulus in the list is
      treated as the target modulus `q_l`, and the product of the preceding
      moduli forms `Q`.
    perf_test: If True, returns random parameters for performance testing
      instead of computing the actual values.

  Returns:
    A tuple containing two JAX arrays:
      - gammas: An array of gamma_i values, one for each modulus in
        `moduli_list[:-1]`.
      - betas: An array of beta_i values, one for each modulus in
        `moduli_list[:-1]`.
  """
  if len(moduli_list) <= 1:
    raise ValueError("moduli_list must have at least 2 moduli")
  if perf_test:
    # Shapes: gammas: (len(moduli_list)-1,), betas: (len(moduli_list)-1,)
    gamma_rand = random_parameters(
        (len(moduli_list) - 1,), moduli_list[:-1], dtype=jnp.uint64
    )
    beta_rand = random_parameters(
        (len(moduli_list) - 1,), moduli_list[:-1], dtype=jnp.uint64
    )
    return jnp.array(gamma_rand, jnp.uint64), jnp.array(beta_rand, jnp.uint64)
  # Compute Q as the product of the moduli for the remaining towers.
  q_prod = 1
  for m in moduli_list[:-1]:
    q_prod *= m

  num_towers = len(moduli_list)
  q_l = moduli_list[-1]
  # Compute q_inv_mod_ql: the inverse of Q modulo q_l.
  q_inv_mod_ql = modinv(q_prod, q_l)

  # Compute gamma_common such that:
  # Q * q_inv_mod_ql = 1 + gamma_common * q_l.
  # Hence, gamma_common = (Q * q_inv_mod_ql - 1) // q_l.
  gamma_common = (q_prod * q_inv_mod_ql - 1) // q_l

  # For each remaining tower compute gamma_i and beta_i.
  gammas = []
  betas = []
  for i in range(num_towers - 1):
    mod_i = moduli_list[i]
    gamma_i = gamma_common % mod_i
    beta_i = modinv(q_l, mod_i)
    gammas.append(gamma_i)
    betas.append(beta_i)
  return jnp.array(gammas, jnp.uint64), jnp.array(betas, jnp.uint64)


####################################
# Random Functions
####################################
def random_batched_ciphertext(shape, modulus_list, dtype=jnp.int32):
  if not shape or len(modulus_list) != shape[-1]:
    raise ValueError("modulus_list length must match the final shape dimension")
  random_key = jax.random.key(0)
  if len(shape) == 5:
    limb_shape = (shape[0], shape[1], shape[2], shape[3], 1)
    concat_axis = 4
  elif len(shape) == 4:
    limb_shape = (shape[0], shape[1], shape[2], 1)
    concat_axis = 3
  else:
    raise ValueError(
        'random_batched_ciphertext expects rank-4 '
        '(batch, elements, degree, moduli) or rank-5 '
        '(batch, elements, r, c, moduli) shape; '
        f'got {shape}.'
    )
  return jnp.concatenate(
      [
          jax.random.randint(
              random_key,
              shape=limb_shape,
              minval=0,
              maxval=bound,
              dtype=dtype,
          )
          for bound in modulus_list
      ],
      axis=concat_axis,
  )


def random_ciphertext(shape, modulus_list, dtype=jnp.int32):
  if not shape or len(modulus_list) != shape[-1]:
    raise ValueError("modulus_list length must match the final shape dimension")
  random_key = jax.random.key(0)
  return jnp.concatenate(
      [
          jax.random.randint(
              random_key,
              shape=(shape[0],shape[1], 1),
              minval=0,
              maxval=bound,
              dtype=dtype,
          )
          for bound in modulus_list
      ],
      axis=2,
  )


def random_parameters(shape, modulus_list, dtype=jnp.int32):
  random_key = jax.random.key(0)
  min_modulus = 2**127
  for modulus in modulus_list:
    if modulus < min_modulus:
      min_modulus = modulus
  return jax.random.randint(
      random_key, shape=shape, minval=0, maxval=min_modulus - 1, dtype=dtype
  )


####################################
# Parse Functions
####################################
def parse_ciphertext_string(input_str, transpose_last_two=True):
  """Parses the input string into two objects.

    - data: a list of element groups, each a list of evaluations (list of lists
    of numbers).
            Shape: (num_element, num_numbers, num_eval)
    - modulus: a one-dimensional list of modulus values corresponding to each
    evaluation index.
                All element groups are assumed to share the same modulus per
                evaluation.

  Args:
    input_str (str): The string containing the input data.
    transpose_last_two (bool): Whether to transpose the last two dimensions of
      each element group in the data.

  Returns:
      tuple: (data, modulus) as described (with swapped inner dimensions for
      data).
  """
  data = []
  global_modulus = []  # This will store the modulus once per evaluation index.

  # Holds the current element's data evaluations.
  current_data_group = []

  # Process the input line by line.
  for line in input_str.strip().splitlines():
    line = line.strip()

    # Check for an "Element" header.
    if line.startswith("Element"):
      # Start a new element group.
      current_data_group = []
      data.append(current_data_group)

      # Check if there is extra content on the same line after the header.
      header_match = re.match(r"^Element\s+\d+:\s*(.*)", line)
      if header_match:
        remainder = header_match.group(1).strip()
        if remainder:
          # Process an evaluation if it appears on the same line.
          if eval_match := re.match(
              r"^(\d+):\s*EVAL:\s*\[(.*?)\]\s*modulus:\s*(\d+)", remainder
          ):
            numbers_str = eval_match.group(2)
            mod_val = int(eval_match.group(3))
            numbers = [
                int(num) for num in numbers_str.split()
            ]
            current_data_group.append(numbers)
            # For the first element group, record the modulus;
            # otherwise, check consistency.
            eval_idx = len(current_data_group) - 1
            if len(data) == 1:
              global_modulus.append(mod_val)
            else:
              if (
                  eval_idx < len(global_modulus)
                  and global_modulus[eval_idx] != mod_val
              ):
                raise ValueError(
                    f"Inconsistent modulus at evaluation index {eval_idx}"
                )

    # Otherwise, check if the line is an evaluation line.
    elif eval_match := re.match(
        r"^(\d+):\s*EVAL:\s*\[(.*?)\]\s*modulus:\s*(\d+)", line
    ):
      numbers_str = eval_match.group(2)
      mod_val = int(eval_match.group(3))
      numbers = [int(num) for num in numbers_str.split()]
      current_data_group.append(numbers)
      eval_idx = len(current_data_group) - 1
      # For the first element group, record the modulus;
      # for subsequent groups, check consistency.
      if len(data) == 1:
        global_modulus.append(mod_val)
      else:
        if (
            eval_idx < len(global_modulus)
            and global_modulus[eval_idx] != mod_val
        ):
          raise ValueError(
              f"Inconsistent modulus at evaluation index {eval_idx}"
          )

  # Transpose the last two dimensions for each element group.
  # Current shape per element: (num_eval, num_numbers)
  # Target shape per element: (num_numbers, num_eval)
  if transpose_last_two:
    for i in range(len(data)):
      if data[i]:
        data[i] = [list(x) for x in zip(*data[i], strict=True)]

  return data, global_modulus


####################################
# Bit Reverse Functions
####################################
def bit_reverse(x, bits):
  """Compute the bit-reversal of integer x with the given number of bits."""
  result = 0
  for i in range(bits):
    if (x >> i) & 1:  # if i-th bit of x is 1
      result |= 1 << (bits - 1 - i)  # set the corresponding reversed bit
  return result


_bit_reverse_permutation_cache = {}


def _bit_reverse_perm(n):
  perm = _bit_reverse_permutation_cache.get(n)
  if perm is None:
    bits = n.bit_length() - 1
    perm = [bit_reverse(i, bits) for i in range(n)]
    _bit_reverse_permutation_cache[n] = perm
  return perm


def bit_reverse_array(in_tower):
  """Bit-reverse a tower, vectorizing one-dimensional NumPy arrays."""
  import numpy as _np

  n = len(in_tower)
  if is_power_of_two(n):
    if isinstance(in_tower, _np.ndarray) and in_tower.ndim == 1:
      perm = _np.asarray(_bit_reverse_perm(n), dtype=_np.intp)
      return in_tower[perm]
    if (
        isinstance(in_tower, list)
        and in_tower
        and not isinstance(in_tower[0], (list, dict))
    ):
      return [in_tower[index] for index in _bit_reverse_perm(n)]

  # Preserve the generic/non-power-of-two behavior for callers outside the
  # optimized flat-array contract.
  x = copy.deepcopy(in_tower)
  bits = len(x).bit_length() - 1
  for i in range(len(x)):
    j = bit_reverse(i, bits)
    if i < j:
      x[i], x[j] = x[j], x[i]
  return x


def bit_reverse_indices(n: int) -> jnp.ndarray:
    """
    Compute an array rev_idx of shape (n,) such that rev_idx[i] is the bit-reversal
    of i over log2(n) bits.
    """
    bits = int(math.log2(n))
    idx = jnp.arange(n)
    # build the reversed index by summing shifted bits
    rev = sum(
        ((idx >> i) & 1) << (bits - 1 - i)
        for i in range(bits)
    )
    return rev


####################################
# Automorphism Functions
####################################
def precompute_auto_map(n: int, k: int) -> List[int]:
    m = n << 1  # cyclOrder
    logm = int(round(math.log2(m)))
    logn = int(round(math.log2(n)))

    precomp: List[int] = [0] * n
    for j in range(n):
        j_tmp = (j << 1) + 1
        t = j_tmp * k
        # ((t % m) >> 1) but written to mirror the C++ bit ops exactly
        idx = (t - ((t >> logm) << logm)) >> 1

        j_rev = bit_reverse(j, logn)
        idx_rev = bit_reverse(idx, logn)
        precomp[j_rev] = idx_rev

    return precomp


def find_automorphism_index_2n_complex(i: int, m: int) -> int:
    """Python translation of nbtheory2.cpp FindAutomorphismIndex2nComplex (243-263).

    Mirrors the C++ logic including early exits, power-of-two validation, and
    modulus via bitmask for m being a power of two.
    """
    if i == 0:
        return 1
    if i == (m - 1):
        return int(i)

    if not is_power_of_two(m):
        raise ValueError("m should be a power of two.")

    # Conjugation automorphism generator
    g0 = pow(5, -1, m) if i < 0 else 5
    g = g0
    i_unsigned = abs(i)
    mask = m - 1
    for _ in range(1, i_unsigned):
        # Equivalent to (g * g0) % m since m is a power of two
        g = (g * g0) & mask
    return int(g)


####################################
# Number Theory Transformation
# Negacyclic NTT is used in CKKS
####################################
def ntt_bit_reverse(a, q, omega):
  """Compute cyclic Number Theoretic Transform of array a modulo q using a given primitive omega of unity."""
  n = len(a)
  # Ensure that omega^n ≡ 1 (mod q) and n divides q-1 for validity.
  # (This should be true if omega is a correct n-th omega of unity.)
  # Bit-reverse the input array indices
  bits = n.bit_length() - 1  # number of bits needed for indexes 0..n-1
  for i in range(n):
    j = bit_reverse(i, bits)
    if i < j:
      a[i], a[j] = a[j], a[i]  # swap to achieve bit-reversed order
  # Cooley-Tukey iterative FFT (NTT)
  length = 2
  while length <= n:
    # Compute twiddle factor step: use omega^(n/length) as the increment
    w_m = pow(omega, n // length, q)
    half = length // 2
    for i in range(0, n, length):  # loop over sub-FFT blocks
      w = 1
      for j in range(i, i + half):  # loop within each block
        u = a[j]
        v = a[j + half] * w % q  # multiply by current twiddle factor
        a[j] = (u + v) % q  # butterfly: combine top part
        a[j + half] = (u - v) % q  # butterfly: combine bottom part
        w = w * w_m % q  # advance twiddle factor for next element
    length *= 2
  return a


def intt_bit_reverse(a, q, omega):
  """Compute the Inverse Number Theoretic Transform of array a modulo p using the given primitive root."""
  n = len(a)
  inv_root = pow(omega, -1, q)  # modular inverse of root
  # Decimation-in-frequency (Gentleman-Sande) butterfly operations
  length = n
  while length >= 2:
    w_m = pow(inv_root, n // length, q)
    half = length // 2
    for i in range(0, n, length):
      w = 1
      for j in range(i, i + half):
        u = a[j]
        v = a[j + half]
        a[j] = (u + v) % q  # combine pairs (top value)
        a[j + half] = (
            ((u - v) % q) * w % q
        )  # combine pairs (bottom), then multiply by twiddle
        w = w * w_m % q  # advance twiddle factor
    length //= 2
  # Bit-reverse the result (to invert the initial bit-reversal
  # permutation in NTT)
  bits = n.bit_length() - 1
  for i in range(n):
    j = bit_reverse(i, bits)
    if i < j:
      a[i], a[j] = a[j], a[i]
  # Divide by n (multiply by n^{-1} mod p) to finish the inverse transform
  inv_n = pow(n, -1, q)
  for i in range(n):
    a[i] = a[i] * inv_n % q
  return a


####################################
# Vectorized exact NTT fast path (numpy uint64)
#
# The pure-Python NTTs below cost ~seconds per transform at degree >= 32768,
# and key generation runs O(10^4) of them (hours at N=32768, ~day at 65536).
# For RNS primes q < 2^31 every butterfly product fits uint64 exactly
# (a*b < 2^62), so the numpy path is bit-identical to the Python path.
# Tables (bit-reversal permutation, twiddle powers, psi powers) are cached
# per (n, q, psi).
####################################
_numpy_ntt_table_cache = {}


def uint64_residues_np(values, modulus):
  """Return exact canonical residues without narrowing Python integers first.

  NumPy cannot directly cast arbitrary-precision Python integers to uint64.
  Object arrays therefore have to be reduced with Python's exact modulo before
  entering the uint64 fast path. Native integer arrays retain a vectorized
  conversion.
  """
  import numpy as _np
  try:
    modulus = operator.index(modulus)
  except TypeError as error:
    raise ValueError("modulus must be an integer") from error
  if modulus <= 0 or modulus >= (1 << 64):
    raise ValueError("modulus must satisfy 0 < modulus < 2**64")

  values_np = _np.asarray(values)
  if values_np.dtype.kind == "u":
    return values_np.astype(_np.uint64, copy=False) % _np.uint64(modulus)
  if (
      values_np.dtype.kind in ("i", "b")
      and modulus <= _np.iinfo(_np.int64).max
  ):
    return _np.remainder(values_np, modulus).astype(_np.uint64, copy=False)

  values_obj = _np.asarray(values, dtype=object)
  try:
    residues = _np.fromiter(
        (operator.index(value) % modulus for value in values_obj.flat),
        dtype=_np.uint64,
        count=values_obj.size,
    )
  except (TypeError, ValueError, OverflowError) as error:
    raise ValueError("values must contain integers") from error
  return residues.reshape(values_obj.shape)


def _ntt_fast_tables(n, q, psi):
  key = (n, q, psi)
  t = _numpy_ntt_table_cache.get(key)
  if t is not None:
    return t
  import numpy as _np
  bits = n.bit_length() - 1
  rev = _np.array([bit_reverse(i, bits) for i in range(n)], dtype=_np.int64)

  def _powers(base, count):
    out = _np.zeros(count, dtype=_np.uint64)
    x = 1
    for i in range(count):
      out[i] = x
      x = x * base % q
    return out

  psis = _powers(psi, n)
  psis_inv = _powers(pow(psi, -1, q), n)
  omega = pow(psi, 2, q)

  fwd = []
  length = 2
  while length <= n:
    w_m = pow(omega, n // length, q)
    fwd.append(_powers(w_m, length // 2))
    length *= 2

  inv_root = pow(omega, -1, q)
  inv = []
  length = n
  while length >= 2:
    w_m = pow(inv_root, n // length, q)
    inv.append(_powers(w_m, length // 2))
    length //= 2

  inv_n = pow(n, -1, q)
  t = (rev, psis, psis_inv, fwd, inv, inv_n)
  _numpy_ntt_table_cache[key] = t
  return t


def ntt_negacyclic_bit_reverse_np(a, q, psi):
  """Numpy-container variant of ntt_negacyclic_bit_reverse (q < 2^31 only).

  Identical values to the list-returning function; returns np.uint64 so
  large-N key generation avoids materializing millions of Python ints per
  call (the per-rotation-key churn OOMs N=65536 bootstrap otherwise).
  """
  import numpy as _np
  values = _np.asarray(a)
  if values.ndim != 1:
    raise ValueError("a must be a one-dimensional integer array")
  n = len(values)
  if not _correct_check(moduli=q, degree=n, psi=psi):
    raise ValueError(
        "NumPy NTT correctness envelope requires 2 < q < 2**31, a "
        "power-of-two degree, and a primitive negacyclic root"
    )
  q = operator.index(q)
  psi = operator.index(psi)
  rev, psis, _, fwd, _, _ = _ntt_fast_tables(n, q, psi)
  qq = _np.uint64(q)
  v = (uint64_residues_np(a, q) * psis) % qq  # pre-twist by psi^i
  v = v[rev]                                           # bit-reversal permute
  for s, ws in enumerate(fwd):
    length = 2 << s
    half = length >> 1
    v = v.reshape(-1, length)
    u = v[:, :half]
    tt = (v[:, half:] * ws) % qq
    v = _np.concatenate([(u + tt) % qq, (u + qq - tt) % qq], axis=1)
  return v.reshape(-1)


def intt_negacyclic_bit_reverse_np(a, q, psi):
  """Numpy-container variant of intt_negacyclic_bit_reverse (q < 2^31 only).

  Identical values to the list-returning function; returns np.uint64.
  """
  import numpy as _np
  values = _np.asarray(a)
  if values.ndim != 1:
    raise ValueError("a must be a one-dimensional integer array")
  n = len(values)
  if not _correct_check(moduli=q, degree=n, psi=psi):
    raise ValueError(
        "NumPy inverse NTT correctness envelope requires 2 < q < 2**31, "
        "a power-of-two degree, and a primitive negacyclic root"
    )
  q = operator.index(q)
  psi = operator.index(psi)
  rev, _, psis_inv, _, inv, inv_n = _ntt_fast_tables(n, q, psi)
  qq = _np.uint64(q)
  v = uint64_residues_np(a, q)
  length = n
  for ws in inv:
    half = length >> 1
    v = v.reshape(-1, length)
    u = v[:, :half]
    w = v[:, half:]
    top = (u + w) % qq
    bot = (((u + qq - w) % qq) * ws) % qq
    v = _np.concatenate([top, bot], axis=1)
    length >>= 1
  v = v.reshape(-1)[rev]                       # undo bit-reversal
  v = (v * _np.uint64(inv_n)) % qq             # divide by n
  v = (v * psis_inv) % qq                      # post-twist by psi^-i
  return v


def ntt_negacyclic_bit_reverse(a, q, psi):
  """Compute the negacyclic NTT of array a (length n) modulo q.

  Args:
    a: list (or 1D array) of integers (length n).
    q: prime modulus.
    psi: an element in GF(q) such that psi^(2*n) = 1 and psi^n = -1 mod q.
          (That is, psi is a primitive 2n-th root of unity; note that then ω =
          psi^2
          is a primitive n-th root of unity.)
    rows: Number of rows in the matrix.
    cols: Number of columns in the matrix.

  Returns:
    The negacyclic NTT of ``a``. List inputs return Python integers. NumPy
    inputs return ``uint64`` on the fast path and Python-object arrays on the
    arbitrary-precision fallback.

  Process:
    1. Pre-twist: multiply each coefficient a[i] by psi^i.
    2. Compute the vanilla NTT (for example, using ntt_bit_reverse) with ω =
    psi^2.
  """
  import numpy as _np

  n = len(a)
  q = operator.index(q)
  psi = operator.index(psi)
  return_array = isinstance(a, _np.ndarray)
  if not _valid_ntt_parameters(q, n, psi):
    raise ValueError(
        "negacyclic NTT requires q > 2, a power-of-two degree, and a "
        "primitive 2N-th root psi"
    )

  # Vectorized exact fast path; preserve the caller's container contract.
  if _correct_check(moduli=q, degree=n, psi=psi):
    result = ntt_negacyclic_bit_reverse_np(a, q, psi)
    return result if return_array else [int(value) for value in result]

  # Convert NumPy scalars before the arbitrary-precision fallback: otherwise
  # their fixed-width multiplication can overflow before Python applies % q.
  values = [operator.index(value) for value in a]
  a_twisted = [(values[i] * pow(psi, i, q)) % q for i in range(n)]

  # Compute vanilla NTT using ω = psi².
  omega = pow(psi, 2, q)

  result = ntt_bit_reverse(a_twisted, q, omega)
  return _np.asarray(result, dtype=object) if return_array else result


def intt_negacyclic_bit_reverse(a, q, psi):
  """Compute the inverse negacyclic NTT of array a (length n) modulo q.

  Args:
    a   : list (or 1D array) of integers (length n) in the negacyclic evaluation
    domain.
    q   : prime modulus.
    psi : an element in GF(q) such that psi^(2*n) = 1 and psi^n = -1 mod q.
          (That is, psi is a primitive 2n-th root of unity; note that then ω =
          psi^2
          is a primitive n-th root of unity.)
  Returns:
    The inverse transform. The output container and dtype follow the forward
    transform contract above.

  Process:
    1. Compute the inverse vanilla NTT using ω = psi².
    2. Post-twist: multiply the result by psi^(–i) for coefficient index i.
  """
  import numpy as _np

  n = len(a)
  q = operator.index(q)
  psi = operator.index(psi)
  return_array = isinstance(a, _np.ndarray)
  if not _valid_ntt_parameters(q, n, psi):
    raise ValueError(
        "inverse negacyclic NTT requires q > 2, a power-of-two degree, and a "
        "primitive 2N-th root psi"
    )

  # Vectorized exact fast path; preserve the caller's container contract.
  if _correct_check(moduli=q, degree=n, psi=psi):
    result = intt_negacyclic_bit_reverse_np(a, q, psi)
    return result if return_array else [int(value) for value in result]

  omega = pow(psi, 2, q)

  # Compute the inverse vanilla NTT.
  values = [operator.index(value) for value in a]
  a_inv = intt_bit_reverse(values, q, omega)

  # Post-twisting: multiply a_inv[i] by psi^(–i).
  psi_inv = pow(psi, -1, q)
  result = [(a_inv[i] * pow(psi_inv, i, q)) % q for i in range(n)]
  return _np.asarray(result, dtype=object) if return_array else result


####################################
# Precision Lowering Functions (outside Google)
####################################
def chunk_decomposition(x, chunkwidth=8):
  """Precision-level data conversion.

  Args:
      x: The input data.
      chunkwidth: The chunkwidth.

  Returns:
      The decomposed data.
  """
  dtype = jnp.uint8
  if chunkwidth == 16:
    dtype = jnp.uint16
  elif chunkwidth == 32:
    dtype = jnp.uint32

  elements = []
  mask = (1 << chunkwidth) - 1
  # Mask to extract the lower bits (e.g., 32 bits -> 0xFFFFFFFF)

  # Extract each element from the integer
  while x > 0:
    elements.append(x & mask)  # Extract the lower bits
    x >>= chunkwidth  # Shift to remove the extracted bits

  # Convert the list to a JAX array
  return jnp.array(elements, dtype=dtype)


####################################
# Performance Profiler Functions (outside Google)
####################################
def dump_hlo_from_lowered(lowered: Any, out_dir: str, out_filename: str) -> str:
  """
  Extract HLO (or StableHLO) textual IR from a lowered computation and write it to a file.
  Returns the full output file path.
  """
  # Prefer XLA HLO; fall back to StableHLO if necessary
  try:
    ir_obj = lowered.compiler_ir(dialect="hlo")
    # Handle XLA computation objects and MLIR modules
    if hasattr(ir_obj, "as_hlo_text"):
      hlo_text = ir_obj.as_hlo_text()
    elif hasattr(ir_obj, "operation"):
      try:
        hlo_text = ir_obj.operation.get_asm(enable_debug_info=True)
      except Exception:
        hlo_text = ir_obj.operation.get_asm()
    else:
      try:
        hlo_text = ir_obj.as_text()
      except Exception:
        hlo_text = str(ir_obj)
  except Exception:
    ir_obj = lowered.compiler_ir(dialect="stablehlo")
    if hasattr(ir_obj, "operation"):
      try:
        hlo_text = ir_obj.operation.get_asm(enable_debug_info=True)
      except Exception:
        hlo_text = ir_obj.operation.get_asm()
    else:
      try:
        hlo_text = ir_obj.as_text()
      except Exception:
        hlo_text = str(ir_obj)
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, out_filename)
  with open(out_path, "w") as f:
    f.write(hlo_text)
  return out_path


def dump_llo_from_lowered(lowered: Any, out_dir: str, out_filename: str) -> str:
  """
  Extract a lower-level XLA IR (LMHLO / LLVM if available) from a lowered computation
  and write it to a file. Returns the full output file path.
  """
  ir_text = None
  dialect_candidates = ["llvm", "lmhlo", "mhlo"]
  # Try both precompiled lowered and compiled executable views
  sources = [lowered]
  try:
    compiled = lowered.compile()
    sources.append(compiled)
  except Exception:
    pass
  for source in sources:
    for dialect in dialect_candidates:
      try:
        ir_obj = source.compiler_ir(dialect=dialect)
        if hasattr(ir_obj, "operation"):
          try:
            ir_text = ir_obj.operation.get_asm(enable_debug_info=True)
          except Exception:
            ir_text = ir_obj.operation.get_asm()
        else:
          # Some dialects (e.g., llvm) may expose textual interfaces differently
          if hasattr(ir_obj, "as_text"):
            ir_text = ir_obj.as_text()
          else:
            ir_text = str(ir_obj)
        if ir_text and len(ir_text) > 0:
          break
      except Exception:
        continue
    if ir_text:
      break
  if ir_text is None:
    # As a last resort, try to stringify generic compiler_ir without dialect hints
    try:
      generic = lowered.compiler_ir()
      if hasattr(generic, "operation"):
        ir_text = generic.operation.get_asm(enable_debug_info=True)
      elif hasattr(generic, "as_text"):
        ir_text = generic.as_text()
      else:
        ir_text = str(generic)
    except Exception:
      raise RuntimeError("Unable to extract LLO/LMHLO/LLVM IR from lowered/compiled computation.")
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, out_filename)
  with open(out_path, "w") as f:
    f.write(ir_text)
  return out_path


def dump_hlo_and_llo_from_tasks(
  tasks: List[Tuple[Callable[..., Any], Tuple[Any, ...]]],
  profile_name: str,
  kernel_name: str
):
    try:
      lowered = tasks[0][0].lower(
        *tasks[0][1]
      )
      repo_root = os.path.dirname(__file__)
      out_dir = os.path.join(repo_root, "log_analysis", f"{profile_name}")
      out_path = dump_hlo_from_lowered(lowered, out_dir, f"{kernel_name}_hlo.txt")
      print(f"HLO dumped to: {out_path}")
      # Also dump lower-level IR (LMHLO/LLVM) as LLO for deeper inspection
      try:
        out_path_llo = dump_llo_from_lowered(lowered, out_dir, f"{kernel_name}_llo.txt")
        print(f"LLO dumped to: {out_path_llo}")
      except Exception as le:
        print(f"Failed to dump LLO: {le}")
    except Exception as e:
      print(f"Failed to dump HLO: {e}")


def profile_jax_functions_xprof(
    tasks: List[Tuple[Callable[..., Any], Tuple[Any, ...]]],
    profile_name: str = "jax_profile",
    kernel_name: str = "kernel_name",
):
  """Profiles a list of JAX functions.

  Args:
    tasks: A list of tuples, where each tuple contains a JAX function and its
      arguments.
    profile_name: The name of the profile, written in log/xprof/plugins/profile.
    kernel_name: The name of the kernel, used to find the latency of the kernel in the trace file.
  Usage:
    tasks = [
        (jit_pdul_barrett_xyzz_pack, (point_a_jax,)),
    ]
    profile_name = "jit_pdul_barrett_xyzz_pack"
    profile_jax_functions(tasks, profile_name, kernel_name="jit_pdul_barrett_xyzz_pack")
  """
  latency = 0
  n = 1 # number of times running the kernel
  final_folder_name = os.path.join(profile_root, profile_name)
  options = jax.profiler.ProfileOptions()
  options.python_tracer_level = 3
  options.host_tracer_level = 3 # https://docs.jax.dev/en/latest/profiling.html#general-options
  options.advanced_configuration = {"tpu_trace_mode" : "TRACE_COMPUTE_AND_SYNC", "tpu_num_chips_to_profile_per_task" : 4}

  repo_root = os.path.dirname(__file__)
  xprof_dir = os.path.join(repo_root, "log/xprof")
  with jax.profiler.trace(xprof_dir):
    # Launch all JAX computations
    results = []
    for func, args_tuple in tasks:
      result = func(*args_tuple)
      results.append(result)

    # Wait for all computations launched in the loop to complete
    if results:
      jax.block_until_ready(results)

  # Rename the newly created timestamped directory to the designated profile_name.
  try:
    if os.path.isdir(profile_root):
      post_dirs = set(os.listdir(profile_root))
      created_dirs = [d for d in (post_dirs - pre_existing_dirs) if os.path.isdir(os.path.join(profile_root, d))]

      target_dir = None
      if created_dirs:
        # Choose the most recently modified among the newly created ones.
        target_dir = max(created_dirs, key=lambda d: os.path.getmtime(os.path.join(profile_root, d)))
      else:
        # Fallback: pick the most recent dir in case set diff failed (e.g., pre list failed).
        all_dirs = [d for d in post_dirs if os.path.isdir(os.path.join(profile_root, d))]
        if all_dirs:
          target_dir = max(all_dirs, key=lambda d: os.path.getmtime(os.path.join(profile_root, d)))

      if target_dir:

        # Avoid overwriting existing destination; add numeric suffix if necessary.
        if os.path.exists(final_folder_name):
          suffix = 1
          while os.path.exists(f"{final_folder_name}_{suffix}"):
            suffix += 1
          final_folder_name = f"{final_folder_name}_{suffix}"
        os.rename(os.path.join(profile_root, target_dir), final_folder_name)
  except Exception as e:
      print(f"Profile rename failed: {e}")

  # Read the trace file and print the latency of the kernel
  # Find the file that ends with 'trace.json.gz' in the destination directory
  trace_file = None
  if os.path.exists(final_folder_name):
    for fname in os.listdir(final_folder_name):
      if fname.endswith("trace.json.gz"):
        trace_file = os.path.join(final_folder_name, fname)
        break
    if trace_file:
      with gzip.open(trace_file, 'rt') as f:
        jtrace = json.loads(f.read())
        if jtrace:
          if "NVIDIA" in  jax.devices()[0].device_kind:
            for e in jtrace["traceEvents"]:
              if 'args' in e and 'tf_op' in e['args']:
                if kernel_name in e['args']["hlo_module"]:
                  latency += e['dur']
          elif "TPU" in  jax.devices()[0].device_kind:
            pid = 999999 # an invalid PID
            for e in jtrace["traceEvents"]:
              if 'args' in e and 'name' in e['args'] and 'TPU:0' in e['args']['name']:
                pid = e['pid']
              if 'args' in e and 'tf_op' in e['args'] and kernel_name in e['args']['tf_op']:
                if e['pid'] == pid:
                  latency += e['dur']
              if 'args' in e and 'hlo_category' in e['args'] and 'copy' in e['args']['hlo_category']:
                if e['pid'] == pid:
                  latency += e['dur']
    else:
      print(f"Trace file not found: {trace_file}")
  else:
    print(f"Final folder name not found: {final_folder_name}")
  return latency


# paper full case evaluation.
original_moduli_51_limbs = [1073753729, 1073738977, 1073753281, 1073739041, 1073753089, 1073747137, 1073752417, 1073739169, 1073745697, 1073739361, 1073752129, 1073746337, 1073748737, 1073746529, 1073748289, 1073747393, 1073749889, 1073748449, 1073751713, 1073749153, 1073750593, 1073749409, 1073751521, 1073750017, 1073751169, 1073750497, 1073751073, 1073750113, 1073750849, 1073739617, 1073746273, 1073745473, 1073745889, 1073742881, 1073745377, 1073739649, 1073745121, 1073741953, 1073744993, 1073739937, 1073744417, 1073742913, 1073744257, 1073742113, 1073743457, 1073742209, 1073743393, 1073740609, 1073742721, 1073741441, 1073741857, 524353]
original_psi_51_limbs = [1093151, 90892563, 108899655, 56634236, 235160291, 12265314, 191995239, 21404433, 40083131, 3916344, 113671079, 34500367, 61894143, 20463380, 13205216, 60050555, 145308815, 87067229, 10533116, 133048918, 13697511, 47895671, 14807533, 10994638, 25005605, 44429319, 77617905, 22756112, 21182116, 46947055, 41148497, 163086225, 60397627, 176334344, 30766686, 77429283, 67466901, 67653750, 4536048, 135444559, 63788661, 110966687, 9716122, 12174708, 49591386, 81862273, 51874541, 12155428, 60746932, 68809976, 28870916, 19017]
extend_moduli_51_limbs = [1152921504606845473, 1152921504606844513, 1152921504606844417, 1152921504606844289, 1152921504606843233, 1152921504606843073, 1152921504606842753, 1152921504606841793, 1152921504606841441, 1152921504606840929]


NTT_PARAMETERS_BY_DEGREE = {
  16: {
    "moduli": [1073759809, 1073759041, 1073759777, 1073758337, 1073759329, 1073758849, 1073759233, 1073738273, 1073754113, 1073738753, 1073753729, 1073738977, 1073753281, 1073739041, 1073753089, 1073747137, 1073752417, 1073739169, 1073745697, 1073739361, 1073752129, 1073746337, 1073748737, 1073746529, 1073748289, 1073747393, 1073749889, 1073748449, 1073751713, 1073749153, 1073750593, 1073749409, 1073751521, 1073750017, 1073751169, 1073750497, 1073751073, 1073750113, 1073750849, 1073739617, 1073746273, 1073745473, 1073745889, 1073742881, 1073745377, 1073739649, 1073745121, 1073741953, 1073744993, 1073739937, 1073744417, 1073742913, 1073744257, 1073742113, 1073743457, 1073742209, 1073743393, 1073740609, 1073742721, 1073741441, 1073741857, 524353],
    "root_of_unity": [149761193, 17168328, 145519847, 68042513, 3491826, 21109149, 48183983, 49547540, 15369996, 12935385, 1093151, 90892563, 108899655, 56634236, 235160291, 12265314, 191995239, 21404433, 40083131, 3916344, 113671079, 34500367, 61894143, 20463380, 13205216, 60050555, 145308815, 87067229, 10533116, 133048918, 13697511, 47895671, 14807533, 10994638, 25005605, 44429319, 77617905, 22756112, 21182116, 46947055, 41148497, 163086225, 60397627, 176334344, 30766686, 77429283, 67466901, 67653750, 4536048, 135444559, 63788661, 110966687, 9716122, 12174708, 49591386, 81862273, 51874541, 12155428, 60746932, 68809976, 28870916, 19017],
  },
  4096: {
    "moduli": [268730369, 268689409, 268361729, 268582913, 268369921, 268460033, 557057, 1152921504606830593, 1152921504606748673],
    "root_of_unity": [8801, 19068, 58939, 11033, 62736, 77090, 474, 116777451583545, 271802498405390],
  },
  8192: {
    "moduli": [269402113, 268091393, 268730369, 268271617, 269221889, 268664833, 268861441, 268369921, 268582913, 557057, 1152921504606830593, 1152921504606748673],
    "root_of_unity": [18987, 2826, 1678, 18925, 2446, 31335, 40892, 65274, 15787, 268, 25959043411404, 100406242475323],
  },
  16384: {
    "moduli": [274726913, 272760833, 274628609, 267059201, 270499841, 267550721, 270237697, 267943937, 268861441, 268042241, 268730369, 268238849, 269844481, 268271617, 269221889, 268369921, 268664833, 557057, 1152921504606748673, 1152921504606683137, 1152921504606584833],
    "root_of_unity": [9358, 15613, 1976, 5381, 15236, 9622, 5177, 2469, 792, 63914, 9742, 12308, 3704, 7216, 7564, 10360, 2023, 19, 62213374832584, 212089012217363, 92166579128688],
  },
  65536: {
    "moduli": [384040961, 376569857, 371458049, 375521281, 371589121, 383778817, 377880577, 379453441, 323092481, 351797249, 349962241, 351404033, 260702209, 308150273, 304742401, 307888129, 302776321, 306708481, 304218113, 347996161, 319291393, 347078657, 323223553, 337248257, 323878913, 336855041, 329515009, 332660737, 329777153, 335413249, 325844993, 330301441, 327548929, 332267521, 328728577, 344850433, 336068609, 340000769, 261488641, 302252033, 297664513, 299499521, 261881857, 295305217, 263323649, 277086209, 263454721, 292159489, 279838721, 291373057, 284950529, 290455553, 281935873, 285474817, 283508737, 288882689, 264634369, 276430849, 270532609, 274726913, 272760833, 276037633, 265420801, 270794753, 268042241, 269221889, 786433],
    "root_of_unity": [1197, 4622, 9335, 5748, 719, 1497, 2281, 3163, 3548, 80, 6577, 4942, 435, 3498, 316, 4503, 1433, 5766, 440, 2739, 1792, 13, 545, 7539, 7418, 7033, 32540, 1301, 4354, 16962, 10301, 289, 4195, 3322, 1005, 1747, 13384, 7659, 2200, 1035, 2142, 6961, 2774, 910, 43, 1949, 4343, 6648, 787, 2879, 4743, 563, 3385, 5648, 5875, 9494, 2122, 852, 6279, 1335, 712, 2017, 929, 142, 5274, 3264, 8],
  },
}

moduli_28_list = {
  degree: params["moduli"]
  for degree, params in NTT_PARAMETERS_BY_DEGREE.items()
}

roof_of_unity = {
  degree: params["root_of_unity"]
  for degree, params in NTT_PARAMETERS_BY_DEGREE.items()
}


# ============================================================================
# Bootstrap Chebyshev coefficient computation
# ============================================================================

def compute_bootstrap_chebyshev_coefficients(K, R, degree):
  """Compute Chebyshev coefficients for the CKKS bootstrap seed function.

  The bootstrap's approximate modular reduction uses a Chebyshev polynomial
  followed by R double-angle iterations to compute sin(2*pi*K*x)/(2*pi*K).

  The seed function approximated by Chebyshev is:

    f(x) = (2*pi)^{-1/2^R} * cos(2*pi/2^R * (x - 0.25))

  projected onto the Chebyshev basis on the interval [-K, K].

  After R double-angle iterations (y -> 2*y^2 + scalar_i), this seed
  produces the full modular reduction function sin(2*pi*K*x)/(2*pi*K).

  This matches OpenFHE's coefficient generation (see ckksrns-fhe.h and
  ckksrns-schemeswitching.cpp for the formula comments).

  Args:
    K: Overflow count parameter (512 for uniform ternary, 28 for sparse).
    R: Number of double-angle iterations (6 for uniform, 3 for sparse).
    degree: Chebyshev polynomial degree (88 for uniform, 44 for sparse).

  Returns:
    List of degree+1 Chebyshev coefficients (c_0 NOT halved).
    The coefficients are for the RESCALED interval [-1, 1] (mapped from [-K, K]).
  """
  import numpy as np

  n = degree + 1
  a, b = -float(K), float(K)
  bMinusA = 0.5 * (b - a)
  bPlusA = 0.5 * (a + b)
  PiByN = np.pi / n

  # The seed function
  c0 = (2 * np.pi) ** (-1.0 / (2 ** R))
  freq = 2 * np.pi / (2 ** R)

  def f(x):
    return c0 * np.cos(freq * (x - 0.25))

  # Evaluate at Chebyshev nodes mapped to [a, b]
  fpts = np.array([f(np.cos(PiByN * (i + 0.5)) * bMinusA + bPlusA) for i in range(n)])

  # Chebyshev projection (matches OpenFHE's EvalChebyshevCoefficients exactly)
  multFactor = 2.0 / n
  coeffs = np.zeros(n)
  for i in range(n):
    for j in range(n):
      coeffs[i] += fpts[j] * np.cos(PiByN * i * (j + 0.5))
    coeffs[i] *= multFactor

  return coeffs.tolist()


# ============================================================================
# D18 bootstrap primes (38 Q-towers, 39 P-towers, 0.024% SF drift)
#
# Generated from OpenFHE with firstModSize=61, scalingModSize=60,
# compositeDegree=2, depth=18. All primes < 2^31 (fits CROSS uint32 NTT).
# ============================================================================

CROSS_Q_TOWERS = [
    2147473409, 1073563649, 1073916929, 1073918977, 1073569793,
    1073579009, 1073907713, 1073600513, 1073605633, 1073620993,
    1073873921, 1073882113, 1073630209, 1073636353, 1073846273,
    1073872897, 1073843201, 1073643521, 1073820673, 1073842177,
    1073651713, 1073652737, 1073815553, 1073655809, 1073658881,
    1073668097, 1073775617, 1073814529, 1073682433, 1073692673,
    1073754113, 1073759233, 1073698817, 1073707009, 1073750017,
    1073753089, 1073732609, 1073738753,
]

CROSS_P_TOWERS = [
    1073499137, 1073497601, 1073493505, 1073480193, 1073475073,
    1073474561, 1073448449, 1073443841, 1073443329, 1073442817,
    1073440769, 1073435137, 1073431553, 1073430529, 1073412097,
    1073406977, 1073394689, 1073391617, 1073387009, 1073385473,
    1073379841, 1073372161, 1073370113, 1073358337, 1073356289,
    1073354753, 1073350657, 1073344001, 1073330177, 1073525249,
    1073527297, 1073530369, 1073539073, 1073545729, 1073551361,
    1073559041, 1073560577, 1073561089, 1073568257,
]

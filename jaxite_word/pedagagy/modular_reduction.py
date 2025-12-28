import functools
import jax
import jax.numpy as jnp
import util
import math

########################
# Control Generation Functions
########################


def barrett_control_generation(q_list):
  q_list = [q_list] if not isinstance(q_list, list) else q_list

  s_list, m_list = [], []
  for q in q_list:
    s = 2 * math.ceil(math.log2(q))
    m = math.floor(2**s / q)
    s_list.append(s)
    m_list.append(m)

  if len(q_list) == 1:
    s_list = s_list[0]
    m_list = m_list[0]
  return s_list, m_list


def barrett_control_generation_s_w(q_list):
  q_list = [q_list] if not isinstance(q_list, list) else q_list

  s_w_list, w_list, m_list = [], [], []
  for q in q_list:
    s = 2 * math.ceil(math.log2(q))
    w = min(s, 32)
    s_w = s-w
    m = math.floor(2**s / q)
    s_w_list.append(s_w)
    w_list.append(w)
    m_list.append(m)

  if len(q_list) == 1:
    s_w_list = s_w_list[0]
    w_list = w_list[0]
    m_list = m_list[0]
  return s_w_list, w_list, m_list


def barrett_reduction_overall_control_generation(moduli_list, perf_test=False):
  if perf_test:
    # Generate random parameters with the same shapes and expected dtypes
    s_w_tuple = util.random_parameters((len(moduli_list),), moduli_list, dtype=jnp.uint16).tolist()
    w_tuple = util.random_parameters((len(moduli_list),), moduli_list, dtype=jnp.uint16).tolist()
    m_tuple = util.random_parameters((len(moduli_list),), moduli_list, dtype=jnp.uint64).tolist()
    return util.to_tuple(s_w_tuple), util.to_tuple(w_tuple), util.to_tuple(m_tuple)

  s_w_tuple, w_tuple, m_tuple = [], [], []
  for q in moduli_list:
    s_w, w, m = barrett_control_generation_s_w(q)
    (s_w_tuple.append(s_w), w_tuple.append(w), m_tuple.append(m))
  return util.to_tuple(s_w_tuple), util.to_tuple(w_tuple), util.to_tuple(m_tuple)


def montgomery_control_generation(original_moduli, r, c):
  w = 32
  w_inv = util.modinv(1 << w, original_moduli)
  prime_inv_32 = util.modinv(original_moduli, 2**32)
  prime_low16 = original_moduli & 0xFFFF
  prime_high16 = original_moduli >> 16
  static_params = (prime_low16, prime_high16, prime_inv_32, original_moduli)
  return util.to_tuple(static_params)


def original_format_to_montgomery_computation_format(a: jax.Array, q: int):
  shape = a.shape 
  a = a.flatten()
  a_list = a.tolist()
  w = 32
  a_computation_format = [(a_i * (1 << w)) % q for a_i in a_list]
  a_computation_format = jnp.array(a_computation_format, dtype=jnp.uint64)
  a_computation_format = a_computation_format.reshape(*shape)
  return a_computation_format

def montgomery_computation_format_to_original_format(a: jax.Array, q: int):
  shape = a.shape
  a = a.flatten()
  a_list = a.tolist()
  w = 32
  w_inv = util.modinv(1 << w, q)
  a_original_format = [(a_i * w_inv) % q for a_i in a_list]
  a_original_format = jnp.array(a_original_format, dtype=jnp.uint64)
  a_original_format = a_original_format.reshape(*shape)
  return a_original_format

########################
# Online Generation Functions
########################

@functools.partial(
  jax.jit, 
  # static_argnames=("s_w", "w", "m")
)
def barrett_reduction_u64_multi_moduli(
    z,      # Input tensor, expected shape (M, ...)
    moduli, # 1D array of moduli, shape (M,)
    s_w,    # 1D array of s_w values, shape (M,)
    w,      # 1D array of w values, shape (M,)
    m,      # 1D array of m values, shape (M,)
):
  """Vectorized Barrett reduction for multi-dimension tensor with per-slice moduli.

  Reduces tensor `z` where the leading dimension `M` corresponds to different
  moduli and precomputed Barrett parameters. For each index `idx` from 0 to M-1,
  the slice `z[idx, ...]` is reduced modulo `moduli[idx]` using parameters
  `s_w[idx]`, `w[idx]`, and `m[idx]`.

  Assumes individual modulus `q` (elements of `moduli`) are less than 31 bits.

  Args:
    z: The input tensor with shape (M, ...), dtype typically uint64.
    moduli: The RNS moduli, 1D JAX array of shape (M,), dtype uint64.
    s_w: The bit width of moduli (`s = ceil(log2(moduli))`),
         1D JAX array of shape (M,), dtype uint16.
    w: The internal shift width (often `min(s, 32)` or `2*s`),
         1D JAX array of shape (M,), dtype uint16.
    m: The precomputed value floor(2^(s+w) / moduli),
         1D JAX array of shape (M,), dtype uint64.

  Returns:
    The result of the Barrett reduction, same shape as z, dtype uint32.
  """
  z = jnp.asarray(z, dtype=jnp.uint64)
  moduli = jnp.asarray(moduli, dtype=jnp.uint64)
  s_w = jnp.asarray(s_w, dtype=jnp.uint16)
  w = jnp.asarray(w, dtype=jnp.uint16)
  m = jnp.asarray(m, dtype=jnp.uint64)

  # Reshape parameters for broadcasting against z
  # Target shape: (M, 1, 1, ..., 1) with (z.ndim - 1) singleton dimensions
  num_singleton_dims = z.ndim - 1
  broadcast_shape = (-1,) + (1,) * num_singleton_dims # -1 infers M

  moduli_b = moduli.reshape(broadcast_shape)
  s_w_b = s_w.reshape(broadcast_shape)
  w_b = w.reshape(broadcast_shape)
  m_b = m.reshape(broadcast_shape)

  # Perform Barrett reduction using broadcasted parameters
  z1 = z & 0xFFFFFFFF # Lower 32 bits of z (still shape (M, ...))
  z2 = z >> w_b       # Upper bits shifted by corresponding w[m]
  t = ((z1 * m_b) >> w_b) + (z2 * m_b)
  t = t >> s_w_b
  reduced_z = z - t * moduli_b
  pred = reduced_z >= moduli_b
  result = jnp.where(pred, reduced_z - moduli_b, reduced_z)
  return result.astype(jnp.uint32)


@functools.partial(
  jax.jit, 
  # static_argnames=("s_w", "w", "m")
)
def barrett_reduction_u64_multi_element_multi_moduli(
    z,      # Input tensor, expected shape (, M, ...)
    moduli, # 1D array of moduli, shape (, M,)
    s_w,    # 1D array of s_w values, shape (, M,)
    w,      # 1D array of w values, shape (, M,)
    m,      # 1D array of m values, shape (, M,)
):
  """Vectorized Barrett reduction for multi-dimension tensor with per-slice moduli.

  Reduces tensor `z` where the leading dimension `M` corresponds to different
  moduli and precomputed Barrett parameters. For each index `idx` from 0 to M-1,
  the slice `z[idx, ...]` is reduced modulo `moduli[idx]` using parameters
  `s_w[idx]`, `w[idx]`, and `m[idx]`.

  Assumes individual modulus `q` (elements of `moduli`) are less than 31 bits.

  Args:
    z: The input tensor with shape (element/batch, M, ...), dtype typically uint64.
    moduli: The RNS moduli, 1D JAX array of shape (M,), dtype uint64.
    s_w: The bit width of moduli (`s = ceil(log2(moduli))`),
         1D JAX array of shape (M,), dtype uint16.
    w: The internal shift width (often `min(s, 32)` or `2*s`),
         1D JAX array of shape (M,), dtype uint16.
    m: The precomputed value floor(2^(s+w) / moduli),
         1D JAX array of shape (M,), dtype uint64.

  Returns:
    The result of the Barrett reduction, same shape as z, dtype uint32.
  """
  z = jnp.asarray(z, dtype=jnp.uint64)
  moduli = jnp.asarray(moduli, dtype=jnp.uint64)
  s_w = jnp.asarray(s_w, dtype=jnp.uint16)
  w = jnp.asarray(w, dtype=jnp.uint16)
  m = jnp.asarray(m, dtype=jnp.uint64)

  # Reshape parameters for broadcasting against z
  # Target shape: (1, M, 1, 1, ..., 1) with (z.ndim - 1) singleton dimensions
  num_singleton_dims = z.ndim - 2
  broadcast_shape = (1,) + (-1,) + (1,) * num_singleton_dims # -1 infers M

  moduli_b = moduli.reshape(broadcast_shape)
  s_w_b = s_w.reshape(broadcast_shape)
  w_b = w.reshape(broadcast_shape)
  m_b = m.reshape(broadcast_shape)

  # Perform Barrett reduction using broadcasted parameters
  z1 = z & 0xFFFFFFFF # Lower 32 bits of z (still shape (M, ...))
  z2 = z >> w_b       # Upper bits shifted by corresponding w[m]
  t = ((z1 * m_b) >> w_b) + (z2 * m_b)
  t = t >> s_w_b
  reduced_z = z - t * moduli_b
  pred = reduced_z >= moduli_b
  result = jnp.where(pred, reduced_z - moduli_b, reduced_z)
  return result.astype(jnp.uint32)


@functools.partial(
  jax.jit, 
  static_argnames=("s_w", "w", "m")
)
def barrett_reduction_u64_multi_moduli_multi_element(
    z,      # Input tensor, expected shape (M, ...)
    moduli, # 1D array of moduli, shape (M,)
    s_w,    # 1D array of s_w values, shape (M,)
    w,      # 1D array of w values, shape (M,)
    m,      # 1D array of m values, shape (M,)
):
  """Vectorized Barrett reduction for multi-dimension tensor with per-slice moduli.

  Reduces tensor `z` where the leading dimension `M` corresponds to different
  moduli and precomputed Barrett parameters. For each index `idx` from 0 to M-1,
  the slice `z[idx, ...]` is reduced modulo `moduli[idx]` using parameters
  `s_w[idx]`, `w[idx]`, and `m[idx]`.

  Assumes individual modulus `q` (elements of `moduli`) are less than 31 bits.

  Args:
    z: The input tensor with shape (M, ...), dtype typically uint64.
    moduli: The RNS moduli, 1D JAX array of shape (M,), dtype uint64.
    s_w: The bit width of moduli (`s = ceil(log2(moduli))`),
         1D JAX array of shape (M,), dtype uint16.
    w: The internal shift width (often `min(s, 32)` or `2*s`),
         1D JAX array of shape (M,), dtype uint16.
    m: The precomputed value floor(2^(s+w) / moduli),
         1D JAX array of shape (M,), dtype uint64.

  Returns:
    The result of the Barrett reduction, same shape as z, dtype uint32.
  """
  z = jnp.asarray(z, dtype=jnp.uint64)
  moduli = jnp.asarray(moduli, dtype=jnp.uint64)
  s_w = jnp.asarray(s_w, dtype=jnp.uint16)
  w = jnp.asarray(w, dtype=jnp.uint16)
  m = jnp.asarray(m, dtype=jnp.uint64)

  # Reshape parameters for broadcasting against z
  # Target shape: (M, 1, 1, ..., 1) with (z.ndim - 2) singleton dimensions
  num_singleton_dims = z.ndim - 1
  broadcast_shape = (-1,) + (1,) * num_singleton_dims # -1 infers M

  moduli_b = moduli.reshape(broadcast_shape)
  s_w_b = s_w.reshape(broadcast_shape)
  w_b = w.reshape(broadcast_shape)
  m_b = m.reshape(broadcast_shape)

  # Perform Barrett reduction using broadcasted parameters
  z1 = z & 0xFFFFFFFF # Lower 32 bits of z (still shape (M, ...))
  z2 = z >> w_b       # Upper bits shifted by corresponding w[m]
  t = ((z1 * m_b) >> w_b) + (z2 * m_b)
  t = t >> s_w_b
  reduced_z = z - t * moduli_b
  pred = reduced_z >= moduli_b
  result = jnp.where(pred, reduced_z - moduli_b, reduced_z)
  return result.astype(jnp.uint32)


@functools.partial(
  jax.jit, 
  # static_argnames=("s_w", "w", "m")
)
def barrett_reduction_u64_multi_moduli_multi_element_no_static(
    z,      # Input tensor, expected shape (M, ...)
    moduli, # 1D array of moduli, shape (M,)
    s_w,    # 1D array of s_w values, shape (M,)
    w,      # 1D array of w values, shape (M,)
    m,      # 1D array of m values, shape (M,)
):
  """Vectorized Barrett reduction for multi-dimension tensor with per-slice moduli.

  Reduces tensor `z` where the leading dimension `M` corresponds to different
  moduli and precomputed Barrett parameters. For each index `idx` from 0 to M-1,
  the slice `z[idx, ...]` is reduced modulo `moduli[idx]` using parameters
  `s_w[idx]`, `w[idx]`, and `m[idx]`.

  Assumes individual modulus `q` (elements of `moduli`) are less than 31 bits.

  Args:
    z: The input tensor with shape (M, ...), dtype typically uint64.
    moduli: The RNS moduli, 1D JAX array of shape (M,), dtype uint64.
    s_w: The bit width of moduli (`s = ceil(log2(moduli))`),
         1D JAX array of shape (M,), dtype uint16.
    w: The internal shift width (often `min(s, 32)` or `2*s`),
         1D JAX array of shape (M,), dtype uint16.
    m: The precomputed value floor(2^(s+w) / moduli),
         1D JAX array of shape (M,), dtype uint64.

  Returns:
    The result of the Barrett reduction, same shape as z, dtype uint32.
  """
  z = jnp.asarray(z, dtype=jnp.uint64)
  moduli = jnp.asarray(moduli, dtype=jnp.uint64)
  s_w = jnp.asarray(s_w, dtype=jnp.uint16)
  w = jnp.asarray(w, dtype=jnp.uint16)
  m = jnp.asarray(m, dtype=jnp.uint64)

  # Reshape parameters for broadcasting against z
  # Target shape: (M, 1, 1, ..., 1) with (z.ndim - 2) singleton dimensions
  num_singleton_dims = z.ndim - 1
  broadcast_shape = (-1,) + (1,) * num_singleton_dims # -1 infers M

  moduli_b = moduli.reshape(broadcast_shape)
  s_w_b = s_w.reshape(broadcast_shape)
  w_b = w.reshape(broadcast_shape)
  m_b = m.reshape(broadcast_shape)

  # Perform Barrett reduction using broadcasted parameters
  z1 = z & 0xFFFFFFFF # Lower 32 bits of z (still shape (M, ...))
  z2 = z >> w_b       # Upper bits shifted by corresponding w[m]
  t = ((z1 * m_b) >> w_b) + (z2 * m_b)
  t = t >> s_w_b
  reduced_z = z - t * moduli_b
  pred = reduced_z >= moduli_b
  result = jnp.where(pred, reduced_z - moduli_b, reduced_z)
  return result.astype(jnp.uint32)


@functools.partial(
    jax.jit,
    # static_argnames=("s_w", "w", "m"),
)
def barrett_reduction_u64(
    z,
    moduli,
    s_w,
    w,
    m,
):
  """Vectorized implementation of the Barrett reduction.

  Works for modulus `q` less than 31 bits.

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
  m = jnp.array(m, dtype=jnp.uint64)
  moduli = jnp.array(moduli, dtype=jnp.uint64)
  w = jnp.array(w, dtype=jnp.uint16)
  s_w = jnp.array(s_w, dtype=jnp.uint16)
  z1 = z & 0xFFFFFFFF
  z2 = z >> w
  t = ((z1 * m) >> w) + (z2 * m)
  t = t >> s_w
  z = z - t * moduli
  pred = z >= moduli
  return jnp.where(pred, z - moduli, z).astype(jnp.uint32)


@functools.partial(
    jax.jit,
    static_argnames=("moduli", "s_w", "w", "m"),
)
def barrett_reduction_u32(
    z,
    moduli,
    s_w,
    w,
    m,
):
  """Vectorized implementation of the Barrett reduction.

  Works for modulus `q` less than 31 bits.

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
  m = jnp.array(m, dtype=jnp.uint64)
  moduli = jnp.array(moduli, dtype=jnp.uint64)
  w = jnp.array(w, dtype=jnp.uint16)
  s_w = jnp.array(s_w, dtype=jnp.uint16)
  z = z.astype(jnp.uint64)
  t = (z * m) >> w
  t = t >> s_w
  z = z - t * moduli
  pred = z >= moduli
  return jnp.where(pred, z - moduli, z).astype(jnp.uint32)


@functools.partial(
    jax.jit,
    static_argnames=("q", "s", "m"),
)
def barrett_reduction(z, q, s, m):
  """Vectorized implementation of the Barrett reduction. Works for modulus `q` less than 31 bits.

  This implementation sets the internal shift width `w` to `min(s, 32)` so it
  works with small modulus `q < 2^16`.

  Args:
    z: The input value.
    q: The modulus.
    s: The bit width of q.
    m: The precomputed value for Barrett reduction.

  Returns:
    The result of the Barrett reduction.
  """
  w = min(s, 32)
  s_w = s - w
  z1 = z & (2**w - 1)
  z2 = z >> w
  t = ((z1 * m) >> w) + (z2 * m)
  t = t >> s_w
  z = (z - t * q).astype(jnp.uint32)
  pred = z >= q
  return jnp.where(pred, z - q, z)


def vec_barrett_u32_lshift32_mod64(u32, q0, q1, s, m0, m1):
  '''Vectorized implementation of the Barrett reduction of inputs left shifted by 32 bits: `(u32 * 2^32) % q`.

  Only for large modulus: `2^51 <= q < 2^60` is known to work.

  q and m are decomposed into lower and higher 32-bit integers
  '''
  u32_in_u64 = u32.astype("uint64")
  t1 = u32_in_u64 * m1 # 64-bit mul
  t2 = (u32_in_u64 * m0) >> 32 # 64-bit mul
  t = (t1 + t2) >> (s - 64)
  t_mul_q = t * q0 + ((t * q1) << 32) # t * q1 is within 32-bit
  r = (u32_in_u64 << 32) - t_mul_q
  return r


def vec_barrett_u33_lshift32_mod64(u33, q0, q1, s, m0, m1):
  '''Vectorized implementation of the Barrett reduction of *33-bit* inputs left shifted by 32 bits: `(u33 * 2^32) % q`.

  Only for large modulus: `2^51 <= q < 2^60` is known to work.

  q and m are decomposed into lower and higher 32-bit integers
  '''
  u33_in_u64 = u33.astype("uint64")
  t1 = u33_in_u64 * m1 # 64-bit mul
  t2 = (u33_in_u64 * m0) >> 30 # 64-bit mul
  t = (t1 + t2) >> (s - 62)
  t_mul_q = t * q0 + ((t * q1) << 32) # t * q1 is within 32-bit
  r = (u33_in_u64 << 32) - t_mul_q
  return r


def vec_barrett_u32_lshift64_mod64(u32, q0, q1, s, m0, m1):
  '''Vectorized implementation of the Barrett reduction of inputs left shifted by 64 bits: (u32 * 2^64) % q

  Only for large modulus: `2^51 <= q < 2^60` is known to work.

  q and m are decomposed into lower and higher 32-bit integers
  '''
  u32_in_u64 = u32.astype("uint64")
  t = ((u32_in_u64 * m1) + (u32_in_u64 * m0 >> 32)) >> (s - 96) # t can be more than 32-bit

  '''
  only need the lower 64-bit of t_mul_q
  because u32 << 64 is larger than 2^64, but `(u32 << 64) - t_mul_q` is guaranteed to be within 64bit:
  xxxxyyyy_00000000_00000000      # u32 << 64
  zzzzwwww_--------_--------      # t_mul_q
  '''
  t0 = t & 0xFFFF_FFFF
  t1 = t >> 32
  t_mul_q = t0 * q0 + (t1 * q0 << 32) + (t0 * q1 << 32) # 64-bit overflow is okay
  r = 0 - t_mul_q
  return r


@jax.jit
def montgomery_reduce_u64_to_u32(z, q_low, q_high, q_inv_32, q):
    """
    Montgomery reduction from u64 to u32 optimized version using only 32-bit operations
    Args:
        z:
            - is u64 array of shape (B, M)
            - input
    parameters:
        - Tuple parameters constants
        - is u32 array of shape (M)
        - modular or moduli
    Returns:
        - is u32 array of shape (B, M) 
        - output
        - reduced value
    """

    #Local constants
    MASK32 = 0xFFFFFFFF
    MASK16 = 0xFFFF
    SHIFT16 = 16
    SHIFT32 = 32

    # Constants parameters convert to jax array
    q_low, q_high, q_inv_32, q = jnp.array(q_low, dtype=jnp.uint32), jnp.array(q_high, dtype=jnp.uint32), jnp.array(q_inv_32, dtype=jnp.uint32), jnp.array(q, dtype=jnp.uint32)

    # Computation
    z_low = z.astype(jnp.uint32)
    z_high = (z >> SHIFT32).astype(jnp.uint32)
    t = (z_low * q_inv_32) & MASK32
    t_low = t & MASK16
    t_high = (t >> SHIFT16) & MASK16

    prod_high = t_high * q_high  # This contributes directly to upper 32 bits
    prod_mid_high = t_high * q_low  # Upper 16 bits go to upper 32 bits  
    prod_mid_low = t_low * q_high   # Upper 16 bits go to upper 32 bits
    prod_low = t_low * q_low        # Upper 16 bits contribute to middle part
    mid_low = (prod_mid_high & MASK16) + (prod_mid_low & MASK16) + (prod_low >> SHIFT16)
    mid_high = (prod_mid_high >> SHIFT16) + (prod_mid_low >> SHIFT16) + (mid_low >> SHIFT16)

    # Final upper 32 bits
    t_final = prod_high + mid_high
    b = z_high + q - t_final
    return b.astype(jnp.uint32)


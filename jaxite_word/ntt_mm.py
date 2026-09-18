"""
This script is specifically designed for NTT/INTT used for ciphertext.
Main difference to ntt_o.py is this script supports 
(1) multiple moduli
(2) multiple batch
(3) distributed sharding
"""
import collections
import os

import numpy as np
import util
import concurrent.futures
import finite_field as ff_context
import jax
import jax.numpy as jnp

_is_nvidia = "NVIDIA" in jax.devices()[0].device_kind


def matmul_conv_flexible_kernel(
    x: jnp.ndarray, y: jnp.ndarray, subscripts: tuple[str, str, str]
) -> jnp.ndarray:
  if x.dtype != jnp.uint32 or y.dtype != jnp.uint32:
    raise TypeError(
        f'matmul operands must both be uint32, got {x.dtype} and {y.dtype}'
    )

  lhs: jax.Array = jax.lax.bitcast_convert_type(x, new_dtype=jnp.uint8)  # bnmp
  rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)  # nk1q
  # https://github.com/google/jax/issues/11483
  rhs = jax.lax.rev(rhs, [2])

  if _is_nvidia:
    u8_products = jax.lax.conv_general_dilated(
        lhs.astype(
            jnp.int16
        ),  # NVIDIA GPU does not support uint8 as input type
        rhs.astype(
            jnp.int16
        ),  # NVIDIA GPU does not support uint8 as input type
        window_strides=(1,),
        padding=((3, 3),),
        dimension_numbers=subscripts,
        preferred_element_type=jnp.float32,  # NVIDIA GPU does not support uint32 as output type
    )
  else:
    u8_products = jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1,),
        padding=((3, 3),),
        dimension_numbers=subscripts,
        preferred_element_type=jnp.uint32,
    )

  shift_factors = jnp.array([0, 8, 16, 24, 32, 40, 48], dtype=jnp.uint32)
  return jnp.sum(u8_products.astype(jnp.uint64) << shift_factors, axis=(2,))


########################
# Parameter Generation Functions
########################
def _pow_matrix_mod(base, q, exponents):
  """Exact elementwise base**exponents mod q via square-and-multiply.

  q < 2^31 so every intermediate product fits uint64 exactly. Replaces the
  per-cell Python pow() (which, with a thread pool spun up per call, made
  the twiddle precompute the dominant setup cost at large N).
  """
  q = int(q)
  e = np.asarray(exponents, dtype=np.uint64)
  result = np.ones(e.shape, dtype=np.uint64)
  qq = np.uint64(q)
  b = int(base) % q
  emax = int(e.max()) if e.size else 0
  bit = 0
  while (1 << bit) <= emax:
    mask = ((e >> np.uint64(bit)) & np.uint64(1)).astype(bool)
    if mask.any():
      result[mask] = (result[mask] * np.uint64(b)) % qq
    b = (b * b) % q
    bit += 1
  return result.astype(np.int64)


def gen_twiddle_matrix(rows, cols, q, omega):
  """Precompute the twiddle matrix T of shape (rows, cols), where T[r, c] = omega^(r*c) mod q.

  Args:
    rows: The number of rows in the matrix.
    cols: The number of columns in the matrix.
    q: The modulus.
    omega: The primitive root of unity.

  Returns:
    The twiddle matrix.
  """
  r_idx = np.arange(rows, dtype=np.int64)[:, None]
  c_idx = np.arange(cols, dtype=np.int64)[None, :]
  exponents = r_idx * c_idx  # shape (rows, cols)
  if util._correct_check([q]):
    return _pow_matrix_mod(omega, q, exponents)
  twiddle_matrix = np.zeros((rows, cols), dtype=int)

  def compute_row(r):
    for c in range(cols):
      twiddle_matrix[r, c] = pow(int(omega), int(exponents[r, c]), int(q))

  with concurrent.futures.ThreadPoolExecutor() as executor:
    list(executor.map(compute_row, range(rows)))
  return twiddle_matrix


def gen_twiddle_matrix_inv(rows, cols, q, omega):
  """Precompute the inverse twiddle matrix T_inv of shape (rows, cols).

  T_inv[r, c] = omega^{- (r*c)} mod q.

  Args:
    rows: The number of rows in the matrix.
    cols: The number of columns in the matrix.
    q: The modulus.
    omega: The primitive root of unity.

  Returns:
    The inverse twiddle matrix.
  """
  inv_omega = pow(int(omega), -1, int(q))
  return gen_twiddle_matrix(rows, cols, q, inv_omega)


# Shared assembled-context cache: operator instances (key-switch part
# Polynomials in hemul/herot/etc.) construct IDENTICAL NTT contexts for the
# same (r, c, moduli) at a given level — at N=65536 each costs ~2 GB of BAT
# arrays, and per-diagonal operator instances OOM the bootstrap holding
# private copies.  Contexts are immutable on the runtime path (the only
# mutator, Polynomial.drop_last_modulus, has no production callers), so
# sharing is safe.  Bounded LRU; evicted entries free once holders drop.
# Configure with CROSS_BARRETT_NTT_CONTEXT_CACHE_ENABLED and
# CROSS_BARRETT_NTT_CONTEXT_CACHE_CAPACITY.
_barrett_ntt_context_cache: "collections.OrderedDict" = (
    collections.OrderedDict()
)
_barrett_ntt_context_cache_capacity = max(
    1,
    int(os.environ.get("CROSS_BARRETT_NTT_CONTEXT_CACHE_CAPACITY", "8")),
)
_barrett_ntt_context_cache_enabled = (
    os.environ.get("CROSS_BARRETT_NTT_CONTEXT_CACHE_ENABLED", "1") != "0"
)


def get_shared_barrett_ntt_context(moduli, parameters):
  """Return a shared NTTCiphertextBarrettContext for (r, c, tuple(moduli))."""
  if not _barrett_ntt_context_cache_enabled:
    return NTTCiphertextBarrettContext(moduli=moduli, parameters=parameters)
  key = (
      int(parameters["r"]),
      int(parameters["c"]),
      tuple(int(m) for m in moduli),
  )
  ctx = _barrett_ntt_context_cache.get(key)
  if ctx is None:
    ctx = NTTCiphertextBarrettContext(moduli=moduli, parameters=parameters)
    _barrett_ntt_context_cache[key] = ctx
    while (
        len(_barrett_ntt_context_cache)
        > _barrett_ntt_context_cache_capacity
    ):
      _barrett_ntt_context_cache.popitem(last=False)
  else:
    _barrett_ntt_context_cache.move_to_end(key)
  return ctx


# Per-modulus twiddle cache: the fused, MAT-permuted per-modulus twiddle
# blocks depend only on (direction, r, c, modulus) — psi/omega and the MAT
# bit-reversal permutations are deterministic functions of those — so they
# are shared across every NTT context that includes the modulus (max-level
# Q/P contexts, per-level key-switch partition contexts, operator-instance
# rebuilds under BOOTSTRAP_LOW_MEM eviction). Reusing them avoids repeated
# modular-exponentiation setup; a hit only copies the cached matrices.
# Entries live for the process lifetime and grow with the distinct
# (direction, r, c, modulus) combinations used.
_ntt_twiddle_cache: dict = {}


########################
# NTT Context with different modular reduction methods
########################
class NTTCiphertextContextBase:
  """Base class for NTT Context with different modular reduction methods

  This class implements the numpy version of three-step NTT algorithm.
  Args:
      moduli: The modulus.
      transform_length: The transform length.
      parameters: The parameters.

  Returns:
      The NTT Context.
  """

  def __init__(self, moduli, parameters: dict, perf_test=False):
    self.ff_ctx = parameters.get(
        "finite_field_context", ff_context.BarrettContext(moduli))
    self.num_bytes = 4

    self.moduli = moduli
    self.parameters = parameters
    # Each backend's validate_moduli enforces its exact/wrap-safe modulus envelope (Montgomery lazy REDC: q < 2^31).
    self.ff_ctx.validate_moduli(moduli)
    self.r = parameters.get("r", 0)
    self.c = parameters.get("c", 0)
    if self.r <= 0 or self.c <= 0:
      raise ValueError(f'r and c must be positive, got r={self.r}, c={self.c}')
    self.transform_length = self.r * self.c

    # u32 einsum accumulator sums 4*max(r,c) u8*u8 partials (< 2^16 each): need max(r,c) < 2^14 or the NTT silently wraps.
    max_dim = max(self.r, self.c)
    if 4 * max_dim * (1 << 16) >= (1 << 32):
      raise ValueError(
          f"NTT transform dim max(r, c)={max_dim} overflows the uint32 BAT "
          "accumulator: need 4 * max(r, c) * 2^16 < 2^32 (max(r, c) < 2^14)"
      )
    self.psi_list = [
        util.root_of_unity(2 * self.transform_length, q) for q in self.moduli
    ]
    self.omega_list = [
        (psi**2) % q
        for psi, q in zip(self.psi_list, self.moduli, strict=True)
    ]
    if perf_test:
      # Use random data for performance testing to avoid expensive precomputation
      key = jax.random.PRNGKey(0)
      self.ntt_bat_tf_step1 = jax.random.bits(
          key, (self.r, 4, self.r, 4, len(moduli)), dtype=jnp.uint8
      )
      self.ntt_tf_step2 = jax.random.bits(
          key, (self.r, self.c, len(moduli)), dtype=jnp.uint64
      )
      self.ntt_bat_tf_step3 = jax.random.bits(
          key, (self.c, 4, self.c, 4, len(moduli)), dtype=jnp.uint8
      )
      self.intt_bat_tf_step1 = jax.random.bits(
          key, (self.c, 4, self.c, 4, len(moduli)), dtype=jnp.uint8
      )
      self.intt_tf_step2 = jax.random.bits(
          key, (self.r, self.c, len(moduli)), dtype=jnp.uint64
      )
      self.intt_bat_tf_step3 = jax.random.bits(
          key, (self.r, 4, self.r, 4, len(moduli)), dtype=jnp.uint8
      )

      if type(self).__name__ == 'NTTCiphertextShoupContext':
        self.ntt_tf_step1 = jax.random.bits(
            key, (self.r, self.r, len(moduli)), dtype=jnp.uint32
        )
        self.ntt_tf_step3 = jax.random.bits(
            key, (self.c, self.c, len(moduli)), dtype=jnp.uint32
        )
        self.intt_tf_step1 = jax.random.bits(
            key, (self.c, self.c, len(moduli)), dtype=jnp.uint32
        )
        self.intt_tf_step3 = jax.random.bits(
            key, (self.r, self.r, len(moduli)), dtype=jnp.uint32
        )

        self.ntt_tf_step1_shoup = jax.random.bits(
            key, (self.r, self.r, len(moduli)), dtype=jnp.uint32
        )
        self.ntt_tf_step2_shoup = jax.random.bits(
            key, (self.r, self.c, len(moduli)), dtype=jnp.uint64
        )
        self.ntt_tf_step3_shoup = jax.random.bits(
            key, (self.c, self.c, len(moduli)), dtype=jnp.uint32
        )
        self.intt_tf_step1_shoup = jax.random.bits(
            key, (self.c, self.c, len(moduli)), dtype=jnp.uint32
        )
        self.intt_tf_step2_shoup = jax.random.bits(
            key, (self.r, self.c, len(moduli)), dtype=jnp.uint64
        )
        self.intt_tf_step3_shoup = jax.random.bits(
            key, (self.r, self.r, len(moduli)), dtype=jnp.uint32
        )
    else:
      self.memory_aligned_transformation()
      self.ntt_tf_step1, self.ntt_tf_step2, self.ntt_tf_step3 = (
          self.ntt_coefficients_precompute()
      )
      self.intt_tf_step1, self.intt_tf_step2, self.intt_tf_step3 = (
          self.intt_coefficients_precompute()
      )
      self.ntt_bat_tf_step1 = self.basis_aligned_transformation(
          self.to_computation_format(self.ntt_tf_step1)
      )
      self.ntt_tf_step2 = self.to_computation_format(self.ntt_tf_step2).astype(
          jnp.uint64
      )
      self.ntt_bat_tf_step3 = self.basis_aligned_transformation(
          self.to_computation_format(self.ntt_tf_step3)
      )
      self.intt_bat_tf_step1 = self.basis_aligned_transformation(
          self.to_computation_format(self.intt_tf_step1)
      )
      self.intt_tf_step2 = self.to_computation_format(
          self.intt_tf_step2
      ).astype(jnp.uint64)
      self.intt_bat_tf_step3 = self.basis_aligned_transformation(
          self.to_computation_format(self.intt_tf_step3)
      )

  ########################
  # Offline Functions
  ########################
  def ntt_coefficients_precompute(self):
    """R = self.r, C = self.c, M = len(self.moduli)

    Negacyclic NTT twiddles with psi factors fused:
    - ntt_tf_step1[k, i, m] = omega_col^(ki) * psi^(ci)  (psi^(ci) fused)
    - ntt_tf_step2[k, j, m] = omega^(kj) * psi^j          (psi^j fused)
    - ntt_tf_step3[j, l, m] = omega_row^(jl)               (unchanged)

    This eliminates the runtime psi_diag pre-multiply in ntt().
    """
    tf_step1_list, tf_step2_list, tf_step3_list = [], [], []
    for idx, modulus in enumerate(self.moduli):
      cache_key = ("ntt", self.r, self.c, int(modulus))
      twiddles = _ntt_twiddle_cache.get(cache_key)
      if twiddles is None:
        psi_m = self.psi_list[idx]
        omega_col = pow(self.omega_list[idx], self.c, modulus)
        omega_row = pow(self.omega_list[idx], self.r, modulus)
        tf_step1_one_modulus = gen_twiddle_matrix(
            self.r, self.r, modulus, omega_col
        )
        # Fuse psi^(c*i) into step1: multiply column i by psi^(c*i)
        for i in range(self.r):
          factor = pow(psi_m, self.c * i, modulus)
          tf_step1_one_modulus[:, i] = (
              (tf_step1_one_modulus[:, i] * factor) % modulus
          )
        tf_step2_one_modulus = gen_twiddle_matrix(
            self.r, self.c, modulus, self.omega_list[idx]
        )
        # Fuse psi^j into step2: multiply column j by psi^j
        for j in range(self.c):
          factor = pow(psi_m, j, modulus)
          tf_step2_one_modulus[:, j] = (
              (tf_step2_one_modulus[:, j] * factor) % modulus
          )
        tf_step3_one_modulus = gen_twiddle_matrix(
            self.c, self.c, modulus, omega_row
        )
        # Memory Aligned Transformation; values < modulus < 2^31 so uint32
        # storage is exact. Cached arrays are never mutated afterwards
        # (jnp.array below copies).
        twiddles = (
            tf_step1_one_modulus[self.perm_r, :].astype(np.uint32),
            tf_step2_one_modulus[self.perm_r, :].astype(np.uint32),
            tf_step3_one_modulus[:, self.perm_c].astype(np.uint32),
        )
        _ntt_twiddle_cache[cache_key] = twiddles
      tf_step1_list.append(twiddles[0])
      tf_step2_list.append(twiddles[1])
      tf_step3_list.append(twiddles[2])
    tf_step1 = jnp.array(tf_step1_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    tf_step2 = jnp.array(tf_step2_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    tf_step3 = jnp.array(tf_step3_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    return tf_step1, tf_step2, tf_step3

  def intt_coefficients_precompute(self):
    """R = self.r, C = self.c, M = len(self.moduli)

    Negacyclic iNTT twiddles with inv_psi factors fused:
    - intt_tf_step1[l, j, m] = inv_omega_row^(lj)             (unchanged)
    - intt_tf_step2[k, j, m] = omega_inv^(kj)/c * inv_psi^j   (inv_psi^j fused)
    - intt_tf_step3[i, k, m] = inv_omega_col^(ik)/r * inv_psi^(ci) (inv_psi^(ci) fused)

    This eliminates the runtime inv_psi_diag post-multiply in intt().
    """
    intt_tf_step1_list, intt_tf_step2_list, intt_tf_step3_list = [], [], []
    for idx, modulus in enumerate(self.moduli):
      cache_key = ("intt", self.r, self.c, int(modulus))
      twiddles = _ntt_twiddle_cache.get(cache_key)
      if twiddles is None:
        inv_psi_m = pow(self.psi_list[idx], -1, modulus)
        omega_col = pow(self.omega_list[idx], self.c, modulus)
        omega_row = pow(self.omega_list[idx], self.r, modulus)
        inv_omega_col = pow(omega_col, -1, modulus)
        inv_omega_row = pow(omega_row, -1, modulus)
        intt_tf_step1_one_modulus = gen_twiddle_matrix(
            self.c, self.c, modulus, inv_omega_row
        )
        intt_tf_step2_one_modulus = gen_twiddle_matrix_inv(
            self.r, self.c, modulus, self.omega_list[idx]
        )
        intt_tf_step3_one_modulus = gen_twiddle_matrix(
            self.r, self.r, modulus, inv_omega_col
        )
        # Fuse inv_psi^j into step2: multiply column j by inv_psi^j
        for j in range(self.c):
          factor = pow(inv_psi_m, j, modulus)
          intt_tf_step2_one_modulus[:, j] = (
              (intt_tf_step2_one_modulus[:, j] * factor) % modulus
          )
        # Fuse inv_psi^(c*i) into step3: multiply row i by inv_psi^(c*i)
        for i in range(self.r):
          factor = pow(inv_psi_m, self.c * i, modulus)
          intt_tf_step3_one_modulus[i, :] = (
              (intt_tf_step3_one_modulus[i, :] * factor) % modulus
          )
        intt_tf_step1_one_modulus = intt_tf_step1_one_modulus[
            self.perm_c, :
        ]  # Memory Aligned Transformation
        intt_tf_step2_one_modulus = intt_tf_step2_one_modulus[
            self.perm_r, :
        ]  # Memory Aligned Transformation
        intt_tf_step3_one_modulus = intt_tf_step3_one_modulus[
            :, self.perm_r
        ]  # Memory Aligned Transformation
        col_inv = pow(self.c, -1, modulus)
        row_inv = pow(self.r, -1, modulus)
        intt_tf_step2_one_modulus = (
            intt_tf_step2_one_modulus * col_inv
        ) % modulus
        intt_tf_step3_one_modulus = (
            intt_tf_step3_one_modulus * row_inv
        ) % modulus
        # Values < modulus < 2^31: uint32 storage is exact. Cached arrays
        # are never mutated afterwards (jnp.array below copies).
        twiddles = (
            intt_tf_step1_one_modulus.astype(np.uint32),
            intt_tf_step2_one_modulus.astype(np.uint32),
            intt_tf_step3_one_modulus.astype(np.uint32),
        )
        _ntt_twiddle_cache[cache_key] = twiddles
      intt_tf_step1_list.append(twiddles[0])
      intt_tf_step2_list.append(twiddles[1])
      intt_tf_step3_list.append(twiddles[2])
    intt_tf_step1 = jnp.array(intt_tf_step1_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    intt_tf_step2 = jnp.array(intt_tf_step2_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    intt_tf_step3 = jnp.array(intt_tf_step3_list, dtype=jnp.uint32).transpose(
        1, 2, 0
    )  # Make moduli the last dimension
    return intt_tf_step1, intt_tf_step2, intt_tf_step3

  def to_computation_format(self, a):
    return self.ff_ctx.to_computation_format(a.astype(jnp.uint64)).astype(
        jnp.uint32
    )

  def to_original_format(self, a: np.ndarray):
    return self.ff_ctx.to_original_format(a.astype(jnp.uint64)).astype(
        jnp.uint32
    )

  def basis_aligned_transformation(self, matrix):
    return util.shifted_mod_bytes_host(
        matrix, self.moduli
    ).transpose(1, 0, 2, 4, 3)

  def memory_aligned_transformation(self):
    """Memory Aligned Transformation (MAT)

    Must run after gen_twiddle_matrix()
    """

    def get_bit_reverse_perm(n):
      """Generates a list of indices for bit-reversal permutation of size n."""
      if n <= 0:
        return []
      bits = n.bit_length() - 1
      perm = [0] * n
      for i in range(n):
        # Reverse bits of i
        r = 0
        temp = i
        for _ in range(bits):
          r = (r << 1) | (temp & 1)
          temp >>= 1
        perm[i] = r
      return perm

    self.perm_r = get_bit_reverse_perm(self.r)
    self.perm_c = get_bit_reverse_perm(self.c)

  def get_jax_parameters(self):
    return {
        "ntt_bat_tf_step1": util.to_tuple(self.ntt_bat_tf_step1),
        "ntt_tf_step2": util.to_tuple(self.ntt_tf_step2),
        "ntt_bat_tf_step3": util.to_tuple(self.ntt_bat_tf_step3),
        "intt_bat_tf_step1": util.to_tuple(self.intt_bat_tf_step1),
        "intt_tf_step2": util.to_tuple(self.intt_tf_step2),
        "intt_bat_tf_step3": util.to_tuple(self.intt_bat_tf_step3),
        "finite_field_parameters": self.ff_ctx.get_jax_parameters(),
        "rows": self.r,
        "cols": self.c,
    }

  ########################
  # Online Functions
  ########################
  def ntt_limb(self, v: jax.Array, limb_index: int):
    """Negacyclic NTT for a single limb.

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)

    Args:
        v: u32 array of shape (B, R, C) - will be casted into u8 array of
          shape (B, R, C, Q)
        ntt_bat_tf_step1: u8 array of shape (R, 4, R, 4)
        ntt_tf_step2: u32 array of shape (R, C)
        ntt_bat_tf_step3: u8 array of shape (C, 4, C, 4)
    Returns:
        u32 array of shape (B, R, C)
    """
    result_step1 = util.matmul(
        v, self.ntt_bat_tf_step1[..., limb_index], "brcq,zqrp->bzcp"
    )
    result_step1_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step1, limb_index
    )
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64),
        self.ntt_tf_step2[..., limb_index],
    )
    result_step2_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step2, limb_index
    )
    result_step3 = util.matmul(
        result_step2_reduced,
        self.ntt_bat_tf_step3[..., limb_index],
        "brcq,cqnp->brnp",
    )
    result_step3_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step3, limb_index
    )
    return result_step3_reduced

  def intt_limb(self, v: jax.Array, limb_index: int):
    """Negacyclic INTT for a single limb.

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)

    Args:
        v: u32 array of shape (B, R, C) - will be casted into u8 array of
          shape (B, R, C, Q)
        intt_bat_tf_step1: u8 array of shape (C, 4, C, 4)
        intt_tf_step2: u32 array of shape (R, C)
        intt_bat_tf_step3: u8 array of shape (R, 4, R, 4)
    Returns:
        u32 array of shape (B, R, C)
    """
    result_step1 = util.matmul(
        v, self.intt_bat_tf_step1[..., limb_index], "brcq,cqlp->brlp"
    )
    result_step1_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step1, limb_index
    )
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64),
        self.intt_tf_step2[..., limb_index],
    )
    result_step2_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step2, limb_index
    )
    result_step3 = util.matmul(
        result_step2_reduced,
        self.intt_bat_tf_step3[..., limb_index],
        "brcq,lqrp->blcp",
    )
    result_step3_reduced = self.ff_ctx.modular_reduction_single_modulus(
        result_step3, limb_index
    )
    return result_step3_reduced

  def ntt(self, v: jax.Array):
    """Negacyclic NTT with modular u32.

    psi^n factors are fused into twiddle tables (step1 has psi^(ci),
    step2 has psi^j), so no runtime psi pre-multiply is needed.

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)
    M = len(self.moduli)

    Args:
        v: u32 array of shape (B, R, C, M) - will be casted into u8 array
          of shape (B, R, C, M, Q)
        ntt_bat_tf_step1: u8 array of shape (R, 4, R, 4, M)
        ntt_tf_step2: u32 array of shape (R, C, M)
        ntt_bat_tf_step3: u8 array of shape (C, 4, C, 4, M)

    Returns:
        u32 array of shape (B, R, C, M) — negacyclic NTT of v
    """
    result_step1 = util.matmul(
        v, self.ntt_bat_tf_step1, "brcmq,zqrpm->bzcmp"
    )
    result_step1_reduced = self.ff_ctx.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_step2
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
    result_step3 = util.matmul(
        result_step2_reduced, self.ntt_bat_tf_step3, "brcmq,cqnpm->brnmp"
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
    return result_step3_reduced

  def intt(self, v: jax.Array):
    """Negacyclic INTT with modular u32.

    inv_psi^n factors are fused into twiddle tables (step2 has inv_psi^j,
    step3 has inv_psi^(ci)), so no runtime inv_psi post-multiply is needed.

    B = Batch size, R = self.r, C = self.c
    Q = 4 (number of bytes per element)
    M = len(self.moduli)

    Args:
        v: u32 array of shape (B, R, C, M) - will be casted into u8 array
          of shape (B, R, C, M, Q)
        intt_bat_tf_step1: u8 array of shape (C, 4, C, 4, M)
        intt_tf_step2: u32 array of shape (R, C, M)
        intt_bat_tf_step3: u8 array of shape (R, 4, R, 4, M)

    Returns:
        u32 array of shape (B, R, C, M) — negacyclic INTT of v
    """
    result_step1 = util.matmul(
        v, self.intt_bat_tf_step1, "brcmq,cqlpm->brlmp"
    )
    result_step1_reduced = self.ff_ctx.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_step2
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
    result_step3 = util.matmul(
        result_step2_reduced, self.intt_bat_tf_step3, "brcmq,lqrpm->blcmp"
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
    return result_step3_reduced

  ########################
  # Modulus Dropping Functions
  ########################
  def drop_last_modulus(self):
    self.ntt_bat_tf_step1 = self.ntt_bat_tf_step1[..., :-1]
    self.ntt_tf_step2 = self.ntt_tf_step2[..., :-1]
    self.ntt_bat_tf_step3 = self.ntt_bat_tf_step3[..., :-1]
    self.intt_bat_tf_step1 = self.intt_bat_tf_step1[..., :-1]
    self.intt_tf_step2 = self.intt_tf_step2[..., :-1]
    self.intt_bat_tf_step3 = self.intt_bat_tf_step3[..., :-1]
    self.ff_ctx.drop_last_modulus()

  def slice(self, num_moduli: int, sliced_ff_ctx=None):
    """Return a view over the first `num_moduli` twiddle factors.

    Twiddle factors are independent per modulus (stacked along the last
    dimension), so slicing produces a valid NTT context for a moduli prefix.
    JAX array slicing shares memory.

    Args:
        num_moduli: Number of moduli to keep.
        sliced_ff_ctx: Finite field context for the sliced moduli. If None,
          slices self.ff_ctx via ff_ctx.slice().

    Returns:
        A new NTT context with sliced twiddle factors and ff_ctx.
    """
    if num_moduli > len(self.moduli):
      raise ValueError(
          f"num_moduli ({num_moduli}) exceeds moduli count ({len(self.moduli)})"
      )
    if sliced_ff_ctx is None:
      sliced_ff_ctx = self.ff_ctx.slice(num_moduli)
    ctx = object.__new__(type(self))
    ctx.ff_ctx = sliced_ff_ctx
    ctx.r = self.r
    ctx.c = self.c
    ctx.transform_length = self.transform_length
    ctx.moduli = self.moduli[:num_moduli]
    ctx.ntt_bat_tf_step1 = self.ntt_bat_tf_step1[..., :num_moduli]
    ctx.ntt_tf_step2 = self.ntt_tf_step2[..., :num_moduli]
    ctx.ntt_bat_tf_step3 = self.ntt_bat_tf_step3[..., :num_moduli]
    ctx.intt_bat_tf_step1 = self.intt_bat_tf_step1[..., :num_moduli]
    ctx.intt_tf_step2 = self.intt_tf_step2[..., :num_moduli]
    ctx.intt_bat_tf_step3 = self.intt_bat_tf_step3[..., :num_moduli]
    return ctx


class NTTCiphertextBarrettContext(NTTCiphertextContextBase):

  def __init__(self, moduli, parameters: dict, perf_test=False):
    super().__init__(moduli, parameters, perf_test=perf_test)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    if self.ff_ctx is None:
      self.ff_ctx = ff_context.BarrettContext(moduli)
    if self.ff_ctx is None:
      raise ValueError("finite_field_context must be provided")
    if self.moduli != self.ff_ctx.moduli:
      raise ValueError(
          "moduli must be the same as the moduli of the finite_field_context"
      )


class NTTCiphertextMontgomeryContext(NTTCiphertextContextBase):

  def __init__(self, moduli, parameters: dict, perf_test=False):
    super().__init__(moduli, parameters, perf_test=perf_test)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    if self.ff_ctx is None:
      self.ff_ctx = ff_context.MontgomeryContext(moduli)
    if self.ff_ctx is None:
      raise ValueError("finite_field_context must be provided")
    if self.moduli != self.ff_ctx.moduli:
      raise ValueError(
          "moduli must be the same as the moduli of the finite_field_context"
      )


class NTTCiphertextShoupContext(NTTCiphertextContextBase):
  """NTT with Shoup's Modular Reduction

  Note that Shoup's Reduction is NOT compatible with Basis Aligned
  Transformation (BAT).
  We use 1-d convolution to perform matrix multiplication for Shoup.
  """

  def __init__(self, moduli, parameters: dict, perf_test=False):
    super().__init__(moduli, parameters, perf_test=perf_test)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    if self.ff_ctx is None:
      self.ff_ctx = ff_context.ShoupContext(moduli)
    if self.ff_ctx is None:
      raise ValueError("finite_field_context must be provided")
    if self.moduli != self.ff_ctx.moduli:
      raise ValueError(
          "moduli must be the same as the moduli of the finite_field_context"
      )

    if not perf_test:
      self.ntt_bat_tf_step1 = self.to_computation_format(
          self.ntt_tf_step1
      ).astype(jnp.uint32)
      self.ntt_tf_step2 = self.to_computation_format(self.ntt_tf_step2).astype(
          jnp.uint64
      )
      self.ntt_bat_tf_step3 = self.to_computation_format(
          self.ntt_tf_step3
      ).astype(jnp.uint32)
      self.intt_bat_tf_step1 = self.to_computation_format(
          self.intt_tf_step1
      ).astype(jnp.uint32)
      self.intt_tf_step2 = self.to_computation_format(
          self.intt_tf_step2
      ).astype(jnp.uint64)
      self.intt_bat_tf_step3 = self.to_computation_format(
          self.intt_tf_step3
      ).astype(jnp.uint32)

      self.ntt_tf_step1_shoup = self.to_shoup_computation_format(
          self.ntt_tf_step1
      ).astype(jnp.uint32)
      self.ntt_tf_step2_shoup = self.to_shoup_computation_format(
          self.ntt_tf_step2
      ).astype(jnp.uint64)
      self.ntt_tf_step3_shoup = self.to_shoup_computation_format(
          self.ntt_tf_step3
      ).astype(jnp.uint32)
      self.intt_tf_step1_shoup = self.to_shoup_computation_format(
          self.intt_tf_step1
      ).astype(jnp.uint32)
      self.intt_tf_step2_shoup = self.to_shoup_computation_format(
          self.intt_tf_step2
      ).astype(jnp.uint64)
      self.intt_tf_step3_shoup = self.to_shoup_computation_format(
          self.intt_tf_step3
      ).astype(jnp.uint32)


  def to_computation_format(self, a):
    return self.ff_ctx.to_computation_format(a.astype(jnp.uint64))

  def to_shoup_computation_format(self, a):
    return self.ff_ctx.precompute_constant_operand(a.astype(jnp.uint64))

  def to_original_format(self, a: np.ndarray):
    return self.ff_ctx.to_original_format(a.astype(jnp.uint64))

  def get_jax_parameters(self):
    return {
        "ntt_tf_step1": util.to_tuple(self.ntt_tf_step1),
        "ntt_tf_step2": util.to_tuple(self.ntt_tf_step2),
        "ntt_tf_step3": util.to_tuple(self.ntt_tf_step3),
        "intt_tf_step1": util.to_tuple(self.intt_tf_step1),
        "intt_tf_step2": util.to_tuple(self.intt_tf_step2),
        "intt_tf_step3": util.to_tuple(self.intt_tf_step3),
        "finite_field_parameters": self.ff_ctx.get_jax_parameters(),
        "rows": self.r,
        "cols": self.c,
        "ntt_tf_step1_shoup": util.to_tuple(self.ntt_tf_step1_shoup),
        "ntt_tf_step2_shoup": util.to_tuple(self.ntt_tf_step2_shoup),
        "ntt_tf_step3_shoup": util.to_tuple(self.ntt_tf_step3_shoup),
        "intt_tf_step1_shoup": util.to_tuple(self.intt_tf_step1_shoup),
        "intt_tf_step2_shoup": util.to_tuple(self.intt_tf_step2_shoup),
        "intt_tf_step3_shoup": util.to_tuple(self.intt_tf_step3_shoup),
    }

  def ntt(self, v: jax.Array):
    """Negacyclic NTT with Shoup's modular reduction.

    Args:
        v: u32 array of shape (B, R, C, M) — input in computation format

    Returns:
        u32 array of shape (B, R, C, M) — negacyclic NTT output
    """
    conv_over_rns = jax.vmap(
        lambda x, y: matmul_conv_flexible_kernel(x, y, ("NCW", "IOW", "NCW")),
        in_axes=(-1, -1),
        out_axes=-1,
    )
    batched_conv_step1 = jax.vmap(
        lambda x, y_b: conv_over_rns(x, y_b),
        in_axes=(None, 0),
        out_axes=0,
    )

    conv_over_rns_step3 = jax.vmap(
        lambda x, y: matmul_conv_flexible_kernel(x, y, ("NCW", "IOW", "CNW")),
        in_axes=(-1, -1),
        out_axes=-1,
    )
    batched_conv_step3 = jax.vmap(
        lambda x_b, y: conv_over_rns_step3(x_b, y),
        in_axes=(0, None),
        out_axes=0,
    )
    result_step1 = batched_conv_step1(self.ntt_tf_step1, v)
    result_step1_shoup = batched_conv_step1(self.ntt_tf_step1_shoup, v)
    result_step1_reduced = self.ff_ctx.modular_reduction(
        result_step1, result_step1_shoup
    )
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_step2
    )
    result_step2_shoup = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_step2_shoup
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(
        result_step2, result_step2_shoup
    )
    result_step3 = batched_conv_step3(result_step2_reduced, self.ntt_tf_step3)
    result_step3_shoup = batched_conv_step3(
        result_step2_reduced, self.ntt_tf_step3_shoup
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(
        result_step3, result_step3_shoup
    )
    result_step3_reduced = result_step3_reduced.transpose(0, 2, 1, 3)
    return result_step3_reduced

  def intt(self, v: jax.Array):
    """Negacyclic INTT with Shoup's modular reduction.

    Args:
        v: u32 array of shape (B, R, C, M) — input in computation format

    Returns:
        u32 array of shape (B, R, C, M) — negacyclic INTT output
    """
    # computation
    conv_over_rns = jax.vmap(
        lambda x, y: matmul_conv_flexible_kernel(x, y, ("CNW", "IOW", "NCW")),
        in_axes=(-1, -1),
        out_axes=-1,
    )
    batched_conv_step1 = jax.vmap(
        lambda x_b, y: conv_over_rns(x_b, y),
        in_axes=(0, None),  # x is shared across B, y_b iterates over axis 0
        out_axes=0,
    )

    conv_over_rns_step3 = jax.vmap(
        lambda x, y: matmul_conv_flexible_kernel(x, y, ("NCW", "IOW", "NCW")),
        in_axes=(-1, -1),
        out_axes=-1,
    )
    batched_conv_step3 = jax.vmap(
        lambda x, y_b: conv_over_rns_step3(x, y_b),
        in_axes=(None, 0),
        out_axes=0,
    )
    v = v.transpose(0, 2, 1, 3)
    result_step1 = batched_conv_step1(v, self.intt_tf_step1)
    result_step1_shoup = batched_conv_step1(v, self.intt_tf_step1_shoup)
    result_step1_reduced = self.ff_ctx.modular_reduction(
        result_step1, result_step1_shoup
    )
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_step2
    )
    result_step2_shoup = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_step2_shoup
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(
        result_step2, result_step2_shoup
    )
    result_step3 = batched_conv_step3(self.intt_tf_step3, result_step2_reduced)
    result_step3_shoup = batched_conv_step3(
        self.intt_tf_step3_shoup, result_step2_reduced
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(
        result_step3, result_step3_shoup
    )
    return result_step3_reduced


class NTTCiphertextBATLazyContext(NTTCiphertextContextBase):

  def __init__(self, moduli, parameters: dict, perf_test=False):
    super().__init__(moduli, parameters, perf_test=perf_test)
    if type(self.moduli) is int:
      self.moduli = [self.moduli]
    # Steps 2/3 need STRICT reduction: always Barrett, even when BATLazyContext was injected (its lazy reduce drives step 1 only).
    # Swap is safe: Barrett and BATLazy share identity to_computation_format, so the twiddle precompute in super().__init__ is unaffected.
    self.ff_ctx = ff_context.BarrettContext(self.moduli)
    self.ff_ctx_bat_lazy = ff_context.BATLazyContext(self.moduli)

  def ntt(self, v: jax.Array):
    """Negacyclic NTT with BAT lazy reduction.

    B = Batch size, R = self.r, C = self.c, M = len(self.moduli)

    Args:
        v: u32 array of shape (B, R, C, M) - will be casted into u8 array
          of shape (B, R, C, M, Q)
        ntt_bat_tf_step1: u8 array of shape (R, 4, R, 4, M)
        ntt_tf_step2: u32 array of shape (R, C, M)
        ntt_bat_tf_step3: u8 array of shape (C, 4, C, 4, M)

    Returns:
        u32 array of shape (B, R, C, M) — negacyclic NTT output
    """
    result_step1 = util.matmul(
        v, self.ntt_bat_tf_step1, "brcmq,zqrpm->bzcmp"
    )
    result_step1_reduced = self.ff_ctx_bat_lazy.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.ntt_tf_step2
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
    result_step3 = util.matmul(
        result_step2_reduced.astype(jnp.uint32),
        self.ntt_bat_tf_step3,
        "brcmq,cqnpm->brnmp",
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
    return result_step3_reduced

  def intt(self, v: jax.Array):
    """Negacyclic INTT with BAT lazy reduction.

    B = Batch size, R = self.r, C = self.c, M = len(self.moduli)

    Args:
        v: u32 array of shape (B, R, C, M) - will be casted into u8 array
          of shape (B, R, C, M, Q)
        intt_bat_tf_step1: u8 array of shape (C, 4, C, 4, M)
        intt_tf_step2: u32 array of shape (R, C, M)
        intt_bat_tf_step3: u8 array of shape (R, 4, R, 4, M)

    Returns:
        u32 array of shape (B, R, C, M) — negacyclic INTT output
    """
    result_step1 = util.matmul(
        v, self.intt_bat_tf_step1, "brcmq,cqlpm->brlmp"
    )
    result_step1_reduced = self.ff_ctx_bat_lazy.modular_reduction(result_step1)
    result_step2 = jnp.multiply(
        result_step1_reduced.astype(jnp.uint64), self.intt_tf_step2
    )
    result_step2_reduced = self.ff_ctx.modular_reduction(result_step2)
    result_step3 = util.matmul(
        result_step2_reduced, self.intt_bat_tf_step3, "brcmq,lqrpm->blcmp"
    )
    result_step3_reduced = self.ff_ctx.modular_reduction(result_step3)
    return result_step3_reduced


def ntt_ciphertext_context_for(finite_field_context):
  """Return the NTT ciphertext context class named by the ff ctx's ntt_ciphertext_context_cls hook; raise if the hook is None (e.g. Shoup)."""
  name = getattr(finite_field_context, "ntt_ciphertext_context_cls", None)
  if name is None:
    raise ValueError(
        f"{type(finite_field_context).__name__} declares no "
        "ntt_ciphertext_context_cls hook; construct its NTT context directly "
        "and inject it as 'ntt_ctx'"
    )
  return globals()[name]

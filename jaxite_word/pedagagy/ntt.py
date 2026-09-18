"""JAX implementation of Gentalman Sande NTT."""

from curses import tigetflag
import functools
import concurrent.futures

import bat
import modular_reduction as modred
import jax
import jax.numpy as jnp
import numpy as np
import util

########################
# Offline Compilation Functions
########################
def mat_shuffle_matmul_right_param_matrix(param_matrix, shuffle_pattern):
  """
  vec_original -> shuffle_pattern -> vec -> (vec @ param_matrix) -> result
                                  ||
                                  vv
  vec_original ->  (vec_original @ shuffled_param_matrix) -> result
  This function is used for analyzing the applicability of MAT.

  Args:
    param_matrix: The input matrix.
    shuffle_pattern: The shuffle pattern.

  Returns:
    shuffled_param_matrix: The shuffled matrix, reorder rows according to the shuffle pattern.
  """
  shuffled_param_matrix = []
  for i in range(len(shuffle_pattern)):
    shuffled_param_matrix.append(param_matrix[shuffle_pattern[i]])
  return shuffled_param_matrix


########################################################
# Twiddle Matrix Generation
########################################################
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
  # Vectorized modular exponentiation via exponent bit-decomposition
  r_idx = np.arange(rows, dtype=np.int64)[:, None]
  c_idx = np.arange(cols, dtype=np.int64)[None, :]
  exponents = r_idx * c_idx  # shape (rows, cols)
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
  twiddle_matrix_inv = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    for c in range(cols):
      twiddle_matrix_inv[r, c] = pow(int(omega), int(-r * c), int(q))
  return twiddle_matrix_inv


########################################################
# NTT (INTT) Original Form O(N^2)
########################################################
def ntt_original_form(v, q, omega):
  length = len(v)
  coef_mat = gen_twiddle_matrix(length, length, q, omega)
  result = [0] * length
  for k in range(length):
    acc = 0
    for j in range(length):
      acc = (acc + int(v[j]) * int(coef_mat[j, k])) % q
    result[k] = acc
  return result


def intt_original_form(v, q, omega):
  """Compute the Inverse NTT (naive O(length^2) algorithm) of vector v of length length over GF(q).

  omega_inv is a primitive L-th root of unity for the inverse transform, i.e.

    if the forward NTT uses omega, then we use omega_inv = omega^{-1} mod q.
  The result is normalized by multiplying by the modular inverse of L.

  Args:
    v: The input vector.
    q: The prime modulus.
    omega: The primitive L-th root of unity.

  Returns:
    The inverse NTT of v.
  """

  length = len(v)
  omega_inv = pow(omega, -1, q)  # modular inverse of root
  coef_mat = gen_twiddle_matrix(length, length, q, omega_inv)
  result = [0] * length
  # Compute the modular inverse of L modulo q
  length_inv = pow(length, -1, q)
  for j in range(length):
    acc = 0
    for k in range(length):
      # Using omega_inv^(j*k)
      acc = (acc + int(v[k]) * int(coef_mat[j, k])) % q
    result[j] = (acc * length_inv) % q
  return result


########################################################
# Cyclic NTT (INTT) -- Algorithm Illustration
# -- Original Form O(N^2)
# -- Bit-Reverse NTT (INTT) O(N log N)
# -- 4-Step NTT (INTT) O(N^{3/2})
########################################################
def ntt_bit_reverse(a, q, omega):
  """Compute the Number Theoretic Transform of array a modulo p using a given primitive omega of unity."""
  n = len(a)
  # Ensure that omega^n ≡ 1 (mod p) and n divides p-1 for validity.
  # (This should be true if omega is a correct n-th omega of unity.)
  # Bit-reverse the input array indices
  bits = n.bit_length() - 1  # number of bits needed for indexes 0..n-1
  for i in range(n):
    j = util.bit_reverse(i, bits)
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


def ntt_bit_reverse_jax(a, q, omega, stage_w_pows=None, br_idx=None, s_w=None, w_barr=None, m_barr=None, total_iter=None):
  """Pure JAX implementation of Cooley-Tukey NTT with bit-reversed input order.

  Args:
    a: 1D list/array of length n (coefficients).
    q: modulus.
    omega: primitive n-th root of unity modulo q.

  Returns:
    List[int]: NTT(a) in bit-reversed order.
  """
  n = a.shape[-1]
  q64 = jnp.uint64(q)
  # Bit-reverse permutation (match the exact swap semantics of the reference)
  # Precompute twiddle powers per stage
  if stage_w_pows is None:
    stage_w_pows, br_idx = ntt_bit_reverse_control_gen_jax(n, q, omega)
  else:
    stage_w_pows = stage_w_pows
    br_idx = br_idx
  # Stack per-stage twiddles into an array [num_stages, n] for JAX-friendly indexing
  stage_twiddles = jnp.stack(stage_w_pows, axis=0) if isinstance(stage_w_pows, (list, tuple)) else stage_w_pows

  curr = jnp.take(a, br_idx, axis=-1)
  # Keep loop carry in uint32 to satisfy JAX fori_loop carry type invariants
  curr = curr.astype(jnp.uint32)
  # Barrett reduction parameters
  if s_w is None:
    s_w, w_barr, m_barr = modred.barrett_control_generation_s_w(q)
  else:
    s_w = s_w
    w_barr = w_barr
    m_barr = m_barr

  # Determine number of stages if not provided: log2(n)
  if total_iter is None:
    total_iter = (n.bit_length() - 1)
  # Precompute index vector once
  idx = jnp.arange(n, dtype=jnp.int32)
  # Loop over stages with jax.lax.fori_loop
  def body(i, state):
    half = jnp.int32(1 << i)
    # Select stage twiddles with dynamic indexing to avoid Python indexing on tracers
    w_stage = jax.lax.dynamic_index_in_dim(stage_twiddles, i, axis=0, keepdims=False)
    partner_idx = idx ^ half
    mask_left = (idx & half) == 0
    # Multiply vector by twiddles mod q
    state_u64 = state.astype(jnp.uint64)
    mul_vec = (state_u64 * w_stage).astype(jnp.uint64)
    w_times_current = modred.barrett_reduction_u64(mul_vec, q, s_w, w_barr, m_barr)
    # Gather needed terms for butterflies (support 1D and batched [B, n])
    w_times_right_for_left = jnp.take(w_times_current, partner_idx, axis=-1)
    left_vals = jnp.where(mask_left, state, jnp.take(state, partner_idx, axis=-1))
    left_vals_u64 = left_vals.astype(jnp.uint64)
    # Compute top/bottom
    sum_top = (left_vals_u64 + w_times_right_for_left.astype(jnp.uint64)).astype(jnp.uint64)
    top = modred.barrett_reduction_u64(sum_top, q, s_w, w_barr, m_barr)
    sum_bottom = (left_vals_u64 + q64 - w_times_current.astype(jnp.uint64)).astype(jnp.uint64)
    bottom = modred.barrett_reduction_u64(sum_bottom, q, s_w, w_barr, m_barr)
    return jnp.where(mask_left, top, bottom)
  curr = jax.lax.fori_loop(0, int(total_iter), body, curr)

  return curr


def ntt_bit_reverse_control_gen_jax(n, q, omega):
  """Generate per-stage twiddle vectors w_pows for Cooley-Tukey NTT.

  For each stage with span `length`, produce a full-length vector of size `n`
  whose entries repeat the per-butterfly twiddle powers across the entire array:
    stage_twiddle = tile([w^0..w^(half-1), w^0..w^(half-1)], n/length),
  where w = omega^(n/length) mod q and half = length // 2.
  """
  q64 = jnp.uint64(q)
  params = []
  length = 2
  while length <= n:
    w_m = pow(omega, n // length, q)
    half = length // 2
    # Build base twiddle powers for one half via modular multiplication (avoid overflow)
    w = jnp.uint64(1)
    w_pows_list = []
    w_m64 = jnp.uint64(w_m)
    for _ in range(half):
      w_pows_list.append(w)
      w = (w * w_m64) % q64
    base_half = jnp.array(w_pows_list, dtype=jnp.uint64)
    # Repeat for both halves within a length-block, then tile across blocks
    per_block = jnp.concatenate([base_half, base_half], axis=0)
    num_blocks = n // length
    full_stage = jnp.tile(per_block, reps=(num_blocks,))
    params.append(full_stage)
    length *= 2
  br_idx = util.bit_reverse_indices(n)
  return params, br_idx


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
    j = util.bit_reverse(i, bits)
    if i < j:
      a[i], a[j] = a[j], a[i]
  # Divide by n (multiply by n^{-1} mod p) to finish the inverse transform
  inv_n = pow(n, -1, q)
  for i in range(n):
    a[i] = a[i] * inv_n % q
  return a


def ntt_four_step(x, q, omega, rows, cols):
  """Compute the 4-step NTT of the input vector x (length N = rows * cols) over GF(q).

  Args:
    x: list or 1D numpy array (length N).
    q: prime modulus.
    omega: the primitive N-th root of unity.
    rows: factors of N, so that N = rows * cols.
    cols: factors of N, so that N = rows * cols.

  Returns:
    A list representing the NTT result.

  Process:
    1. Columns:  NTT on each column (length rows) using omega_col = omega^cols.
    2. Twiddle:  Multiply by T[r,c] = omega^(r*c).
    3. Rows:     NTT on each row (length cols) using omega_row = omega^rows.
    4. Reordering: Final output is flatten(transpose(Z)).
  """
  num_elements = rows * cols
  if len(x) != num_elements:
    raise ValueError("Length of x must equal rows * cols")
  omega_row = pow(omega, rows, q)
  omega_col = pow(omega, cols, q)
  matrix_a = np.array(x, dtype=int).reshape((rows, cols))
  y = np.zeros((rows, cols), dtype=int)
  for c in range(cols):
    col = matrix_a[:, c].tolist()
    y[:, c] = ntt_original_form(col, q, omega_col)

  twiddle = gen_twiddle_matrix(rows, cols, q, omega)
  y = (y * twiddle) % q

  matrix_z = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    row = y[r, :].tolist()
    matrix_z[r, :] = ntt_original_form(row, q, omega_row)
  matrix_x = np.array(
      matrix_z.T
  ).flatten()  # forward transform reorders via transpose flattening
  return matrix_x.tolist()


def intt_four_step(x, q, omega, rows, cols):
  """Compute the 4-step Inverse NTT of the input vector X (length N = rows * cols) over GF(q).

  Forward transform recap:
    - Columns:  NTT on each column (length rows) using omega_col = omega^cols.
    - Twiddle:  Multiply by T[r,c] = omega^(r*c).
    - Rows:     NTT on each row (length cols) using omega_row = omega^rows.
    - Reordering: Final output is flatten(transpose(Z)).

  To invert, we perform:
    0. Compute the appropriate inverse roots.
    1. Undo the reordering.
    2. Inverse row transform (length cols) on each row.
    3. Multiply by the inverse twiddle matrix T_inv[r,c] = omega^(-r*c).
    4. Inverse column transform (length rows) on each column.
    5. Reassemble the final result.

  Note: The naive inverse NTT (intt_original_form) already divides by the
  transform length.
  Hence, the two stages provide an overall normalization of 1/(rows·cols) = 1/N.

  Args:
    x     : list or 1D numpy array (length N) that is the forward NTT result.
    q     : prime modulus.
    omega : the primitive N-th root of unity used in the forward transform.
            (Forward transform used:
            rowNTT with omega_row = omega^R and columnNTT with omega_col =
            omega^C,
            plus twiddle multiplication T[r,c] = omega^(r*c).)
    rows  : factors of N, so that N = rows * cols.
    cols  : factors of N, so that N = rows * cols.

  Returns:
    A list representing the inverse NTT result (the original vector).
  """
  num_elements = rows * cols
  if len(x) != num_elements:
    raise ValueError("Length of X must equal rows * cols")

  # Step 0: Compute necessary inverse roots and normalization factors.
  # For the inverse column transform (of length rows):
  omega_col = pow(omega, cols, q)
  # For the inverse row transform (of length cols):
  omega_row = pow(omega, rows, q)

  # Step 1: Undo the final reordering of the forward transform.
  # The forward transform did: X = flatten(transpose(Z)) with Z of shape
  # (rows, cols).
  # To recover Z, first reshape X into shape (cols, rows) then transpose.
  matrix_z = np.array(x, dtype=int).reshape((cols, rows)).T
  # Now Z is an rows x cols matrix.
  # Step 2: Inverse row transform.
  # For each row of Y (length cols), compute the inverse NTT using omega_row.
  y = np.zeros((rows, cols), dtype=int)
  for r in range(rows):
    row = matrix_z[r, :].tolist()
    # intt on each row of length cols using inv_omega_row
    # (inverse happens inside intt_original_form)
    y[r, :] = intt_original_form(row, q, omega_row)

  # Step 3: Multiply by the inverse twiddle factor matrix.
  # The forward twiddle matrix was T[r,c] = omega^(r*c). Its inverse is:
  # T_inv[r,c] = omega^{-r*c} mod q.
  twiddle_inv = gen_twiddle_matrix_inv(rows, cols, q, omega)
  y = (y * twiddle_inv) % q

  # Step 4: Inverse column transform.
  # For each column of Z (length rows), compute the inverse NTT using
  # inv_omega_col.
  matrix_a = np.zeros((rows, cols), dtype=int)
  for c in range(cols):
    col = y[:, c].tolist()
    # intt on each column of length rows using inv_omega_col
    # (inverse happens inside intt_original_form).
    matrix_a[:, c] = intt_original_form(col, q, omega_col)

  # Step 5: Reassemble the final result.
  # The forward transform mapped the original vector x to X using a reordering.
  # Here, we flatten A (row-major order) to obtain the original x.
  x_recovered = np.array(matrix_a).flatten()
  return x_recovered.tolist()


########################################################
# Negacyclic NTT (INTT) -- Algorithm Illustration
# -- Bit-Reverse NTT (INTT) O(N log N)
# -- 4-Step NTT (INTT) O(N^{3/2})
# -- TPU Algorithm O(N^{3/2})
# -- Layout Invariant Algorithm O(N^{3/2}), 3-Step NTT
########################################################
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
    The negacyclic NTT of a.

  Process:
    1. Pre-twist: multiply each coefficient a[i] by psi^i.
    2. Compute the vanilla NTT (for example, using ntt_bit_reverse) with ω =
    psi^2.
  """
  n = len(a)
  # Check that psi^n = -1 mod q.
  if pow(psi, n, q) != q - 1:
    raise ValueError(
        "psi is not a valid 2n-th root of unity for negacyclic NTT (psi^n must"
        " equal -1 mod q)."
    )

  # Pre-twisting: multiply a[i] by psi^i.
  a_twisted = [(a[i] * pow(psi, i, q)) % q for i in range(n)]

  # Compute vanilla NTT using ω = psi².
  omega = pow(psi, 2, q)

  return ntt_bit_reverse(a_twisted.copy(), q, omega)


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
    The original input vector (i.e. the inverse transform).

  Process:
    1. Compute the inverse vanilla NTT using ω = psi².
    2. Post-twist: multiply the result by psi^(–i) for coefficient index i.
  """
  n = len(a)
  omega = pow(psi, 2, q)

  # Compute the inverse vanilla NTT.
  a_inv = intt_bit_reverse(a.copy(), q, omega)

  # Post-twisting: multiply a_inv[i] by psi^(–i).
  psi_inv = pow(psi, -1, q)
  return [(a_inv[i] * pow(psi_inv, i, q)) % q for i in range(n)]


def ntt_negacyclic_four_step(a, q, psi, rows, cols):
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
    The negacyclic NTT of a.

  Process:
    1. Pre-twist: multiply each coefficient a[i] by psi^i.
    2. Compute the vanilla NTT (for example, using ntt_bit_reverse) with ω =
    psi^2.
  """
  n = len(a)
  # Check that psi^n = -1 mod q.
  if pow(psi, n, q) != q - 1:
    raise ValueError(
        "psi is not a valid 2n-th root of unity for negacyclic NTT (psi^n must"
        " equal -1 mod q)."
    )

  # Pre-twisting: multiply a[i] by psi^i.
  a_twisted = [(a[i] * pow(psi, i, q)) % q for i in range(n)]

  # Compute vanilla NTT using ω = psi².
  omega = pow(psi, 2, q)

  # a_transformed = ntt_bit_reverse(a_twisted.copy(), q, omega)
  return ntt_four_step(a_twisted.copy(), q, omega, rows, cols)


def intt_negacyclic_four_step(a, q, psi, rows, cols):
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
    The original input vector (i.e. the inverse transform).

  Process:
    1. Compute the inverse vanilla NTT using ω = psi².
    2. Post-twist: multiply the result by psi^(–i) for coefficient index i.
  """
  n = len(a)
  omega = pow(psi, 2, q)

  # Compute the inverse vanilla NTT.
  # a_inv = intt_bit_reverse(a.copy(), q, omega)
  a_inv = intt_four_step(a.copy(), q, omega, rows, cols)

  # Post-twisting: multiply a_inv[i] by psi^(–i).
  psi_inv = pow(psi, -1, q)
  return [(a_inv[i] * pow(psi_inv, i, q)) % q for i in range(n)]


def ntt_negacyclic_three_step(
    a, q, psi, rows, cols, tf_step1, coef_step2, tf_step3
):
  """Compute the negacyclic NTT of array a (length n) modulo q.

  Args:
    a   : list (or 1D array) of integers (length n).
    q   : prime modulus.
    psi : an element in GF(q) such that psi^(2*n) = 1 and psi^n = -1 mod q.
          (That is, psi is a primitive 2n-th root of unity; note that then ω =
          psi^2
          is a primitive n-th root of unity.)
    rows: Number of rows in the matrix.
    cols: Number of columns in the matrix.
    tf_step1: The twiddle factor matrix for step 1.
    coef_step2: The twiddle factor matrix for step 2 (element-wise multiplication).
    tf_step3: The twiddle factor matrix for step 3.

  Returns:
    The negacyclic NTT of a.

  Process:
    1. Pre-twist: multiply each coefficient a[i] by psi^i.
    2. Compute the vanilla NTT (for example, using ntt_bit_reverse) with ω =
    psi^2.
  """
  n = len(a)
  # Check that psi^n = -1 mod q.
  if pow(psi, n, q) != q - 1:
    raise ValueError(
        "psi is not a valid 2n-th root of unity for negacyclic NTT (psi^n must"
        " equal -1 mod q)."
    )

  # Pre-twisting: multiply a[i] by psi^i.
  a_twisted = [(a[i] * pow(psi, i, q)) % q for i in range(n)]

  num_elements = rows * cols
  if len(a_twisted) != num_elements:
    raise ValueError("Length of a_twisted must equal rows * cols")
  matrix_a = np.array(a_twisted, dtype=int).reshape((rows, cols))
  y = np.matmul(tf_step1, matrix_a)
  y = y % q

  y = y * coef_step2
  y = y % q

  z = np.matmul(y, tf_step3)
  z = z % q
  x = np.array(
      z.T
  ).flatten()  # forward transform reorders via transpose flattening
  return x.tolist()


def intt_negacyclic_three_step(
    a, q, psi, rows, cols, inv_tf_step1, inv_coef_step2, inv_tf_step3
):
  """Compute the inverse negacyclic NTT of array a (length n) modulo q using TPU-friendly operations.

  Args:
    a   : list (or 1D array) of integers (length n) in the negacyclic evaluation
    domain.
    q   : prime modulus.
    psi : an element in GF(q) such that psi^(2*n) = 1 and psi^n = -1 mod q.
          (That is, psi is a primitive 2n-th root of unity; note that then ω =
          psi^2
          is a primitive n-th root of unity.)
    rows: Number of rows in the matrix.
    cols: Number of columns in the matrix.
    inv_tf_step1: The inverse of the first transform matrix.
    inv_coef_step2: The inverse of the second coefficient matrix.
    inv_tf_step3: The inverse of the third transform matrix.

  Returns:
    The original input vector (i.e. the inverse transform).

  Process:
    1. Compute the inverse vanilla NTT using ω = psi².
    2. Post-twist: multiply the result by psi^(–i) for coefficient index i.
  """
  n = len(a)

  num_elements = rows * cols
  if len(a) != num_elements:
    raise ValueError("Length of a must equal rows * cols")

  # Step 1: Undo the final reordering of the forward transform.
  # The forward transform did: X = flatten(transpose(Z)) with Z of shape
  # (rows, cols).
  # To recover Z, first reshape X into shape (cols, rows) then transpose.
  z = np.array(a, dtype=int).reshape((cols, rows)).T
  # Now z is an rows x cols matrix.

  # Step 2: Inverse row transform.
  # For each row of Y (length cols), compute the inverse NTT using omega_row.
  y = np.matmul(z, inv_tf_step1) % q


  cols_inv = pow(cols, -1, q)
  inv_coef_step2 = inv_coef_step2 * cols_inv % q

  # Step 3: Multiply by the inverse twiddle factor matrix.
  # The forward twiddle matrix was T[r,c] = omega^(r*c). Its inverse is:
  # T_inv[r,c] = omega^{-r*c} mod q.
  y = (y * inv_coef_step2) % q

  # Step 4: Inverse column transform.
  # For each column of Z (length rows), compute the inverse NTT using
  # inv_omega_col.
  a = np.matmul(inv_tf_step3, y) % q

  ### rows_inv = pow(rows, -1, q)  # This is already fused into the parameters step3
  ### a = a * rows_inv % q # This is already fused into the parameters step3
  # Step 5: Reassemble the final result.
  # The forward transform mapped the original vector x to X using a reordering.
  # Here, we flatten A (row-major order) to obtain the original x.
  x_recovered = np.array(a).flatten()

  # Post-twisting: multiply x_recovered[i] by psi^(–i).
  psi_inv = pow(psi, -1, q)
  return [(x_recovered[i] * pow(psi_inv, i, q)) % q for i in range(n)]

################################################################################################
# Cyclic/Negacyclic NTT (INTT) -- Illustration Above
#  --------------------------------------------Divider---------------------------------------  #
# Negacyclic NTT (INTT) -- Deployment Version below
################################################################################################

########################################################
# Negacyclic NTT (INTT) -- Control Generation
########################################################
def ntt_negacyclic_three_step_control_generation(q_list_in, r, c):
  """
  This function splits each byte into an independent logical axis.
  """
  n = r * c
  coef_step2_list, tf_mat_step1_list, tf_mat_step3_list = [], [], []

  q_list = [q_list_in] if not isinstance(q_list_in, list) else q_list_in
  # psi_list = [psi_list_in] if not isinstance(psi_list_in, list) else psi_list_in
  psi_list = [util.root_of_unity(2*r*c, q) for q in q_list]

  for (q, psi) in zip(q_list, psi_list):
    assert pow(psi, r*c*2, q) == 1 # This version is defined for 32-bit input.
    if psi is not None:
      omega = (psi ** 2) % q
      psi = psi
    else:
      # omega = nth_primitive_root(n, q)
      psi =  util.root_of_unity(2*n, q)
      omega = (psi ** 2) % q
      

    omega_col = pow(omega, c, q)
    omega_row = pow(omega, r, q)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      f1 = executor.submit(gen_twiddle_matrix, r, r, q, omega_col)
      f2 = executor.submit(gen_twiddle_matrix, r, c, q, omega)
      f3 = executor.submit(gen_twiddle_matrix, c, c, q, omega_row)
      tf_mat_step1 = jnp.array(f1.result(), dtype=jnp.uint32)
      coef_step2 = jnp.array(f2.result(), dtype=jnp.uint32)
      tf_mat_step3 = jnp.array(f3.result(), dtype=jnp.uint32)
    np.testing.assert_array_equal(tf_mat_step1.T, tf_mat_step1)
    np.testing.assert_array_equal(tf_mat_step3.T, tf_mat_step3)

    (tf_mat_step1_list.append(tf_mat_step1), coef_step2_list.append(coef_step2), tf_mat_step3_list.append(tf_mat_step3))

  if len(q_list) == 1:
    tf_mat_step1_list, coef_step2_list, tf_mat_step3_list = tf_mat_step1_list[0], coef_step2_list[0], tf_mat_step3_list[0]

  return  tf_mat_step1_list, coef_step2_list, tf_mat_step3_list


def intt_negacyclic_three_step_control_generation(q_list_in, r, c):
  """
  This function splits each byte into an independent logical axis.
  """
  q_list = [q_list_in] if not isinstance(q_list_in, list) else q_list_in
  psi_list = [util.root_of_unity(2*r*c, q) for q in q_list]
  inv_tf_mat_step1_list, inv_coef_step2_list, inv_tf_mat_step3_list = [], [], []
  for (q, psi) in zip(q_list, psi_list):
    assert pow(psi, r*c*2, q) == 1 # This version is defined for 32-bit input.

    omega = (psi ** 2) % q
    omega_col = pow(omega, c, q)
    omega_row = pow(omega, r, q)
    inv_omega_col = pow(omega_col, -1, q)
    # inverse primitive R-th root for columns
    inv_omega_row = pow(omega_row, -1, q)
    # inverse primitive C-th root for rows

    # intt needs to scale the corresponding coefficients.
    inv_r = pow(r, -1, q)

    with concurrent.futures.ThreadPoolExecutor() as executor:
      f1 = executor.submit(gen_twiddle_matrix, c, c, q, inv_omega_row)
      f2 = executor.submit(gen_twiddle_matrix_inv, r, c, q, omega)
      f3 = executor.submit(gen_twiddle_matrix, r, r, q, inv_omega_col)
      inv_tf_mat_step1 = jnp.array(f1.result(), dtype=int)
      inv_coef_step2_ori = jnp.array(f2.result(), dtype=int)
      inv_tf_mat_step3 = jnp.array(f3.result(), dtype=int)
      inv_tf_mat_step3 = (inv_tf_mat_step3 * inv_r) % q

    (inv_tf_mat_step1_list.append(inv_tf_mat_step1), inv_coef_step2_list.append(inv_coef_step2_ori), inv_tf_mat_step3_list.append(inv_tf_mat_step3))

  if len(q_list) == 1:
    inv_tf_mat_step1_list, inv_coef_step2_list, inv_tf_mat_step3_list = inv_tf_mat_step1_list[0], inv_coef_step2_list[0], inv_tf_mat_step3_list[0]

  return inv_tf_mat_step1_list, inv_coef_step2_list, inv_tf_mat_step3_list


def ntt_three_step_bat_control_generation(q_list_in, r, c, perf_test=False):
  if perf_test:
    bat_tf_mat_step1_list, coef_step2_list, bat_tf_mat_step3_list = util.random_parameters((len(q_list_in), r, r, 4, 4), q_list_in, jnp.uint8), util.random_parameters((len(q_list_in), r, c), q_list_in, jnp.uint32), util.random_parameters((len(q_list_in), c, c, 4, 4), q_list_in, jnp.uint8)
    return jnp.array(bat_tf_mat_step1_list, jnp.uint8), jnp.array(coef_step2_list, jnp.uint32), jnp.array(bat_tf_mat_step3_list, jnp.uint8)

  tf_mat_step1_list, coef_step2_list, tf_mat_step3_list = ntt_negacyclic_three_step_control_generation(q_list_in, r, c)
  coef_step2_list = [coef_step2_list] if not isinstance(coef_step2_list, list) else coef_step2_list
  coef_step2_list = [jnp.array(coef_step2, jnp.uint32) for coef_step2 in coef_step2_list]
  bat_tf_mat_step1_list, bat_tf_mat_step3_list = [], []

  q_list = [q_list_in] if not isinstance(q_list_in, list) else q_list_in
  tf_mat_step1_list = [tf_mat_step1_list] if not isinstance(tf_mat_step1_list, list) else tf_mat_step1_list
  tf_mat_step3_list = [tf_mat_step3_list] if not isinstance(tf_mat_step3_list, list) else tf_mat_step3_list
  def _process_single(q, tf_mat_step1, tf_mat_step3):
    tf_mat_bat_step1 = bat.hpmatmul_offline_bat_deployment(
        tf_mat_step1.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    tf_mat_bat_step3 = bat.hpmatmul_offline_bat_deployment(
        tf_mat_step3.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    return tf_mat_bat_step1, tf_mat_bat_step3

  with concurrent.futures.ThreadPoolExecutor() as executor:
    args_iter = zip(q_list, tf_mat_step1_list, tf_mat_step3_list)
    results = list(executor.map(lambda args: _process_single(*args), args_iter))
  for tf_mat_bat_step1, tf_mat_bat_step3 in results:
    (bat_tf_mat_step1_list.append(tf_mat_bat_step1), bat_tf_mat_step3_list.append(tf_mat_bat_step3))

  if len(q_list) == 1:
    bat_tf_mat_step1_list, coef_step2_list,bat_tf_mat_step3_list = bat_tf_mat_step1_list[0], coef_step2_list[0], bat_tf_mat_step3_list[0]

  return jnp.array(bat_tf_mat_step1_list, jnp.uint8), jnp.array(coef_step2_list, jnp.uint32), jnp.array(bat_tf_mat_step3_list, jnp.uint8)


def intt_three_step_bat_control_generation(q_list_in, r, c, shuffle_pattern=None, perf_test=False):

  """Generate the control parameters for the intt layout invariant algorithm.

  Args:
    q_list_in: The list of moduli.
    r: The number of rows.
    c: The number of columns.
    shuffle_pattern: The shuffle pattern, meaning the input to intt needs to be shuffled: y[i] = x[shuffle_pattern[i]].

  Returns:
    The control parameters for the intt layout invariant algorithm.
  """
  if perf_test:
    num_limbs = len(q_list_in) if isinstance(q_list_in, list) else 1
    q_list_in = q_list_in if isinstance(q_list_in, list) else [q_list_in]
    if num_limbs == 1:
      bat_inv_tf_mat_step1_list, scaled_inv_coef_step2_list, bat_inv_tf_mat_step3_list, power_of_inv_psi_arr = util.random_parameters((c, c, 4, 4), q_list_in, jnp.uint8), util.random_parameters((r, c), q_list_in, jnp.uint32), util.random_parameters((r, c, 4, 4), q_list_in,  jnp.uint8), util.random_parameters((c*r), q_list_in, jnp.uint64)
    else:
      bat_inv_tf_mat_step1_list, scaled_inv_coef_step2_list, bat_inv_tf_mat_step3_list, power_of_inv_psi_arr = util.random_parameters((num_limbs, c, c, 4, 4), q_list_in, jnp.uint8), util.random_parameters((num_limbs, r, c), q_list_in, jnp.uint32), util.random_parameters((num_limbs, r, c, 4, 4), q_list_in,  jnp.uint8), util.random_parameters((num_limbs, c*r), q_list_in, jnp.uint64)
    return jnp.array(bat_inv_tf_mat_step1_list, jnp.uint8), jnp.array(scaled_inv_coef_step2_list, jnp.uint32), jnp.array(bat_inv_tf_mat_step3_list, jnp.uint8), jnp.array(power_of_inv_psi_arr, jnp.uint64)

  inv_tf_mat_step1_list, inv_coef_step2_list, inv_tf_mat_step3_list = intt_negacyclic_three_step_control_generation(q_list_in, r, c)
  if shuffle_pattern is not None:
    print(np.array(inv_tf_mat_step1_list).shape)
    inv_tf_mat_step1_list = mat_shuffle_matmul_right_param_matrix(inv_tf_mat_step1_list, shuffle_pattern)
    print(np.array(inv_tf_mat_step1_list).shape)

  q_list = [q_list_in] if not isinstance(q_list_in, list) else q_list_in
  psi_list = [util.root_of_unity(2*r*c, q) for q in q_list]
  inv_tf_mat_step1_list = [inv_tf_mat_step1_list] if not isinstance(inv_tf_mat_step1_list, list) else inv_tf_mat_step1_list
  inv_coef_step2_list = [inv_coef_step2_list] if not isinstance(inv_coef_step2_list, list) else inv_coef_step2_list
  inv_tf_mat_step3_list = [inv_tf_mat_step3_list] if not isinstance(inv_tf_mat_step3_list, list) else inv_tf_mat_step3_list

  inv_psi = [pow(psi, -1, q) for (q, psi) in zip(q_list, psi_list)]
  power_of_inv_psi_arr = [
      [pow(inv_psi[idx], i, q_list[idx]) for i in range(c*r)] for idx in range(len(psi_list))
  ]

  bat_inv_tf_mat_step1_list, scaled_inv_coef_step2_list, bat_inv_tf_mat_step3_list = [], [], []
  def _process_single(q, inv_tf_mat_step1, inv_coef_step2, inv_tf_mat_step3):
    inv_c = pow(c, -1, q)
    inv_tf_step1 = bat.hpmatmul_offline_bat_deployment(
        inv_tf_mat_step1.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    inv_coef_step2_scaled = (inv_c * inv_coef_step2) % q
    inv_coef_step2_scaled = inv_coef_step2_scaled.astype(jnp.uint32)
    inv_tf_step3 = bat.hpmatmul_offline_bat_deployment(
        inv_tf_mat_step3.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    return inv_tf_step1, inv_coef_step2_scaled, inv_tf_step3

  with concurrent.futures.ThreadPoolExecutor() as executor:
    args_iter = zip(q_list, inv_tf_mat_step1_list, inv_coef_step2_list, inv_tf_mat_step3_list)
    results = list(executor.map(lambda args: _process_single(*args), args_iter))
  for inv_tf_step1, inv_coef_step2_scaled, inv_tf_step3 in results:
    (bat_inv_tf_mat_step1_list.append(inv_tf_step1), scaled_inv_coef_step2_list.append(inv_coef_step2_scaled), bat_inv_tf_mat_step3_list.append(inv_tf_step3))

  if len(q_list) == 1:
    bat_inv_tf_mat_step1_list, scaled_inv_coef_step2_list, bat_inv_tf_mat_step3_list = bat_inv_tf_mat_step1_list[0], scaled_inv_coef_step2_list[0], bat_inv_tf_mat_step3_list[0]

  return jnp.array(bat_inv_tf_mat_step1_list, jnp.uint8), jnp.array(scaled_inv_coef_step2_list, jnp.uint32), jnp.array(bat_inv_tf_mat_step3_list, jnp.uint8), jnp.array(power_of_inv_psi_arr, jnp.uint64)


def intt_jax_precompute_control_generation(original_moduli, r, c, perf_test=False):
  if perf_test:
    ring_dim = r * c
    bat_inv_tf_mat_step1_arr = util.random_parameters((len(original_moduli), r, r, 4, 4), original_moduli, dtype=jnp.uint8)
    scaled_inv_coef_step2_arr = util.random_parameters((len(original_moduli), r, c), original_moduli, dtype=jnp.uint32)
    bat_inv_tf_mat_step3_arr = util.random_parameters((len(original_moduli), c, c, 4, 4), original_moduli, dtype=jnp.uint8)
    power_of_inv_psi_arr = util.random_parameters((len(original_moduli), ring_dim), original_moduli, dtype=jnp.uint64)
    intt_dynamic_parameters = (
      jnp.array(bat_inv_tf_mat_step1_arr, jnp.uint8),
      jnp.array(scaled_inv_coef_step2_arr, jnp.uint32),
      jnp.array(bat_inv_tf_mat_step3_arr, jnp.uint8),
      jnp.array(power_of_inv_psi_arr, jnp.uint64),
    )
    s_w_tuple = util.random_parameters((len(original_moduli),), original_moduli, dtype=jnp.uint16)
    w_tuple = util.random_parameters((len(original_moduli),), original_moduli, dtype=jnp.uint16)
    m_tuple = util.random_parameters((len(original_moduli),), original_moduli, dtype=jnp.uint64)
    intt_static_parameters = (original_moduli, s_w_tuple, w_tuple, m_tuple)
    return (intt_dynamic_parameters, util.to_tuple(intt_static_parameters))

  bat_inv_tf_mat_step1_list, scaled_inv_coef_step2_list, bat_inv_tf_mat_step3_list, power_of_inv_psi_arr = intt_three_step_bat_control_generation(original_moduli, r, c)

  # Dynamic parameters generation
  bat_inv_tf_mat_step1_arr = jnp.array(bat_inv_tf_mat_step1_list, jnp.uint8)
  scaled_inv_coef_step2_arr = jnp.array(scaled_inv_coef_step2_list, jnp.uint32)
  bat_inv_tf_mat_step3_arr = jnp.array(bat_inv_tf_mat_step3_list, jnp.uint8)
  intt_dynamic_parameters = (bat_inv_tf_mat_step1_arr, scaled_inv_coef_step2_arr, bat_inv_tf_mat_step3_arr, power_of_inv_psi_arr)

  # Static parameters generation
  s_w_tuple, w_tuple, m_tuple = [], [], []
  for q in original_moduli:
    s_w, w, m = modred.barrett_control_generation_s_w(q)
    (s_w_tuple.append(s_w), w_tuple.append(w), m_tuple.append(m))
  intt_static_parameters = (original_moduli, s_w_tuple, w_tuple, m_tuple)
  return (intt_dynamic_parameters, util.to_tuple(intt_static_parameters))


def ntt_montgomery_three_step_bat_control_generation(q_list_in, r, c):
  tf_mat_step1_list, coef_step2_list, tf_mat_step3_list = ntt_negacyclic_three_step_control_generation(q_list_in, r, c)
  coef_step2_list = [coef_step2_list] if not isinstance(coef_step2_list, list) else coef_step2_list
  coef_step2_list = [jnp.array(coef_step2, jnp.uint32) for coef_step2 in coef_step2_list]
  tf_mat_step1_list = [tf_mat_step1_list] if not isinstance(tf_mat_step1_list, list) else tf_mat_step1_list
  tf_mat_step3_list = [tf_mat_step3_list] if not isinstance(tf_mat_step3_list, list) else tf_mat_step3_list
  q_list = [q_list_in] if not isinstance(q_list_in, list) else q_list_in

  tf_mat_step1_montgomery_list = [modred.original_format_to_montgomery_computation_format(jnp.array(mat1, jnp.uint64), q) for (mat1, q) in zip(tf_mat_step1_list, q_list)]
  coef_step2_montgomery_list = [modred.original_format_to_montgomery_computation_format(jnp.array(coef_step2, jnp.uint64), q) for (coef_step2, q) in zip(coef_step2_list, q_list)]
  tf_mat_step3_montgomery_list = [modred.original_format_to_montgomery_computation_format(jnp.array(mat3, jnp.uint64), q) for (mat3, q) in zip(tf_mat_step3_list, q_list)]
  bat_tf_mat_step1_list, bat_tf_mat_step3_list = [], []

  def _process_single(q, tf_mat_step1, tf_mat_step3):
    tf_mat_bat_step1 = bat.hpmatmul_offline_bat_deployment(
        tf_mat_step1.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    tf_mat_bat_step3 = bat.hpmatmul_offline_bat_deployment(
        tf_mat_step3.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    return tf_mat_bat_step1, tf_mat_bat_step3

  with concurrent.futures.ThreadPoolExecutor() as executor:
    args_iter = zip(q_list, tf_mat_step1_montgomery_list, tf_mat_step3_montgomery_list)
    results = list(executor.map(lambda args: _process_single(*args), args_iter))
  for tf_mat_bat_step1, tf_mat_bat_step3 in results:
    (bat_tf_mat_step1_list.append(tf_mat_bat_step1), bat_tf_mat_step3_list.append(tf_mat_bat_step3))

  if len(q_list) == 1:
    bat_tf_mat_step1_list, coef_step2_list,bat_tf_mat_step3_list = bat_tf_mat_step1_list[0], coef_step2_list[0], bat_tf_mat_step3_list[0]

  return jnp.array(bat_tf_mat_step1_list, jnp.uint8), jnp.array(coef_step2_list, jnp.uint32), jnp.array(bat_tf_mat_step3_list, jnp.uint8)


def ntt_montgomery_three_step_bat_bmatmul_control_generation(q_list_in, r, c):
  tf_mat_step1_list, coef_step2_list, tf_mat_step3_list = ntt_negacyclic_three_step_control_generation(q_list_in, r, c)
  coef_step2_list = [coef_step2_list] if not isinstance(coef_step2_list, list) else coef_step2_list
  coef_step2_list = [jnp.array(coef_step2, jnp.uint32) for coef_step2 in coef_step2_list]
  tf_mat_step1_list = [tf_mat_step1_list] if not isinstance(tf_mat_step1_list, list) else tf_mat_step1_list
  tf_mat_step3_list = [tf_mat_step3_list] if not isinstance(tf_mat_step3_list, list) else tf_mat_step3_list
  q_list = [q_list_in] if not isinstance(q_list_in, list) else q_list_in

  tf_mat_step1_montgomery_list = [modred.original_format_to_montgomery_computation_format(jnp.array(mat1, jnp.uint64), q) for (mat1, q) in zip(tf_mat_step1_list, q_list)]
  coef_step2_montgomery_list = [modred.original_format_to_montgomery_computation_format(jnp.array(coef_step2, jnp.uint64), q) for (coef_step2, q) in zip(coef_step2_list, q_list)]
  tf_mat_step3_montgomery_list = [modred.original_format_to_montgomery_computation_format(jnp.array(mat3, jnp.uint64), q) for (mat3, q) in zip(tf_mat_step3_list, q_list)]
  bat_tf_mat_step1_list, bat_tf_mat_step3_list = [], []

  def _process_single(q, tf_mat_step1, tf_mat_step3):
    tf_mat_bat_step1 = bat.hpmatmul_offline_bat_deployment(
        tf_mat_step1.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    tf_mat_bat_step3 = bat.hpmatmul_offline_bat_deployment(
        tf_mat_step3.astype(jnp.uint32), q
    ).astype(jnp.uint8)
    return tf_mat_bat_step1, tf_mat_bat_step3

  with concurrent.futures.ThreadPoolExecutor() as executor:
    args_iter = zip(q_list, tf_mat_step1_montgomery_list, tf_mat_step3_montgomery_list)
    results = list(executor.map(lambda args: _process_single(*args), args_iter))
  for tf_mat_bat_step1, tf_mat_bat_step3 in results:
    (bat_tf_mat_step1_list.append(tf_mat_bat_step1), bat_tf_mat_step3_list.append(tf_mat_bat_step3))

  if len(q_list) == 1:
    bat_tf_mat_step1_list, coef_step2_list,bat_tf_mat_step3_list = bat_tf_mat_step1_list[0], coef_step2_list[0], bat_tf_mat_step3_list[0]

  return jnp.array(bat_tf_mat_step1_list, jnp.uint8), jnp.array(coef_step2_list, jnp.uint32), jnp.array(bat_tf_mat_step3_list, jnp.uint8)


########################################################
# Negacyclic NTT (INTT) -- Runtime Implementations
########################################################
@jax.jit
def hpmatmul_bat_coef_lhs_batch(lhs: jax.Array, y: jax.Array):
  """Input (m, k) Left Matrix -> (m, k, p, q) Left Matrix, where each element in the original (m, k) matrix is replaced by a (p, q) matrix.

  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.
  """
  rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "mkpq,bknq->bmnp",
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


@jax.jit
def hpmatmul_bat_coef_rhs_batch(y: jax.Array, rhs: jax.Array):
  """Input (k, n) right Matrix -> (k, n, p, q) right Matrix, where each element in the original (k, n) matrix is replaced by a (p, q) matrix.

  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.
  """

  lhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "bmkq,knpq->bmnp",
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


########################
# Three Step NTT - Barrett Reduction
########################

def ntt_three_step_bat_barrett_batch(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    q,
    s_w,
    w,
    m,
):
  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8

  result_step1 = hpmatmul_bat_coef_lhs_batch(tf_step1, poly_coef_2d)
  result_step1_mod_q = modred.barrett_reduction_u64(result_step1, q, s_w, w, m)
  result_step2 = jax.numpy.multiply(
      result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64)
  )
  result_step2_mod_q = modred.barrett_reduction_u64(result_step2, q, s_w, w, m)
  result_step3 = hpmatmul_bat_coef_rhs_batch(result_step2_mod_q, tf_step3)
  return modred.barrett_reduction_u64(result_step3, q, s_w, w, m)


def ntt_three_step_bat_barrett_square_batch(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    q,
    s_w,
    w,
    m,
):
  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32

  result_step1 = hpmatmul_bat_coef_lhs_batch(tf_step1, poly_coef_2d)
  result_step1_mod_q = modred.barrett_reduction_u64(result_step1, q, s_w, w, m)

  result_step2 = jax.numpy.multiply(
      result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64)
  )
  result_step2_mod_q = modred.barrett_reduction_u64(result_step2, q, s_w, w, m)

  result_step3 = hpmatmul_bat_coef_rhs_batch(result_step2_mod_q, tf_step1)

  return modred.barrett_reduction_u64(result_step3, q, s_w, w, m)


def intt_three_step_bat_barrett_batch(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    q,
    s_w,
    w,
    m,
):
  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8

  result_step1 = hpmatmul_bat_coef_rhs_batch(poly_coef_2d, tf_step1)
  result_step1_mod_q = modred.barrett_reduction_u64(result_step1, q, s_w, w, m)
  result_step2 = jax.numpy.multiply(result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64))
  result_step2_mod_q = modred.barrett_reduction_u64(result_step2, q, s_w, w, m)
  result_step3 = hpmatmul_bat_coef_lhs_batch(tf_step3, result_step2_mod_q)
  result_step3_mod_q = modred.barrett_reduction_u64(result_step3, q, s_w, w, m)
  return result_step3_mod_q


########################
# Three Step NTT -- Multiple Moduli Implementation
########################

@jax.jit
def hpmatmul_bat_coef_lhs_multi_moduli(lhs: jax.Array, y: jax.Array):
  """Input (m, k) Left Matrix -> (m, k, p, q) Left Matrix, where each element in the original (m, k) matrix is replaced by a (p, q) matrix.

  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.
  """
  rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "bmkpq,bknq->bmnp",
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


@jax.jit
def hpmatmul_bat_coef_rhs_multi_moduli(y: jax.Array, rhs: jax.Array):
  """Input (k, n) right Matrix -> (k, n, p, q) right Matrix, where each element in the original (k, n) matrix is replaced by a (p, q) matrix.

  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.
  """

  lhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "bmkq,bknpq->bmnp",
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


@functools.partial(
  jax.jit, 
  # static_argnames=("s_w", "w", "m")
)
def ntt_three_step_bat_barrett_multi_moduli(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    q,
    s_w,
    w,
    m,
):
  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8

  result_step1 = hpmatmul_bat_coef_lhs_multi_moduli(tf_step1, poly_coef_2d)
  result_step1_mod_q = modred.barrett_reduction_u64_multi_moduli(result_step1, q, s_w, w, m)

  result_step2 = jax.numpy.multiply(
      result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64)
  )
  result_step2_mod_q = modred.barrett_reduction_u64_multi_moduli(result_step2, q, s_w, w, m)
  result_step3 = hpmatmul_bat_coef_rhs_multi_moduli(result_step2_mod_q, tf_step3)

  return modred.barrett_reduction_u64_multi_moduli(result_step3, q, s_w, w, m)

@functools.partial(
  jax.jit, 
  static_argnames=("s_w", "w", "m")
)
def ntt_three_step_bat_barrett_square_multi_moduli(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    q,
    s_w,
    w,
    m,
):
  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32

  result_step1 = hpmatmul_bat_coef_lhs_multi_moduli(tf_step1, poly_coef_2d)
  result_step1_mod_q = modred.barrett_reduction_u64_multi_moduli(result_step1, q, s_w, w, m)

  result_step2 = jax.numpy.multiply(
      result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64)
  )
  result_step2_mod_q = modred.barrett_reduction_u64_multi_moduli(result_step2, q, s_w, w, m)

  result_step3 = hpmatmul_bat_coef_rhs_multi_moduli(result_step2_mod_q, tf_step1)

  return modred.barrett_reduction_u64_multi_moduli(result_step3, q, s_w, w, m)


@functools.partial(
  jax.jit, 
  # static_argnames=("s_w", "w", "m")
)
def intt_three_step_bat_barrett_multi_moduli(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    q,
    s_w,
    w,
    m,
):

  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8
  result_step1 = hpmatmul_bat_coef_rhs_multi_moduli(poly_coef_2d, tf_step1)
  result_step1_mod_q = modred.barrett_reduction_u64_multi_moduli(result_step1, q, s_w, w, m)
  result_step2 = jax.numpy.multiply(result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64))
  result_step2_mod_q = modred.barrett_reduction_u64_multi_moduli(result_step2, q, s_w, w, m)
  result_step3 = hpmatmul_bat_coef_lhs_multi_moduli(tf_step3, result_step2_mod_q)
  result_step3_mod_q = modred.barrett_reduction_u64_multi_moduli(result_step3, q, s_w, w, m)
  return result_step3_mod_q


########################
# Three Step NTT -- Batched Multiple Moduli Implementation 
# Batch first, moduli second dimension
########################
@jax.jit
def hpmatmul_bat_coef_lhs_batch_multi_moduli(lhs: jax.Array, y: jax.Array):
  """batch_multi_moduli implementation
      original u32 matrix multiplication:
      (L, m, k) * (E, L, k, n) -> (E, L, m, n)

      input:
      (E, L, k, n) input ciphertext -> (E, L, k, n, q), where q means number of chunks
      (L, m, k, p, q) input twiddle factor matrix (lhs)

      (E, L,    k, n, q)
         (L, m, k, p, q) twiddle factor matrix 
            |
            v
      (E, L, m, n, p)
  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.
  """
  rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "lmkpq,elknq->elmnp", #TODO: iterate each moduli and batch.
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


@jax.jit
def hpmatmul_bat_coef_rhs_batch_multi_moduli(y: jax.Array, rhs: jax.Array):
  """batch_multi_moduli implementation
      original u32 matrix multiplication:
      (E, L, m, k) * (L, k, n) -> (E, L, m, n)

      input:
      (E, L, m, k) input ciphertext -> (E, L, m, k, q), where q means number of chunks
      (L, k, n) -> (L, k, n, p, q) input twiddle factor matrix (rhs)

      (E, L, m, k,       q)
         (L,    k, n, p, q) twiddle factor matrix 
           |
           v
      (E, L, m, n, p)
  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.
  """
  lhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "elmkq,lknpq->elmnp",
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


@functools.partial(
  jax.jit, 
  # static_argnames=("s_w", "w", "m")
)
def ntt_three_step_bat_barrett_batch_multi_moduli(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    q,
    s_w,
    w,
    m,
):
  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8

  result_step1 = hpmatmul_bat_coef_lhs_batch_multi_moduli(tf_step1, poly_coef_2d)
  result_step1_mod_q = modred.barrett_reduction_u64_multi_element_multi_moduli(result_step1, q, s_w, w, m)

  result_step2 = jax.numpy.multiply(
      result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64)
  )
  result_step2_mod_q = modred.barrett_reduction_u64_multi_element_multi_moduli(result_step2, q, s_w, w, m)
  result_step3 = hpmatmul_bat_coef_rhs_batch_multi_moduli(result_step2_mod_q, tf_step3)

  return modred.barrett_reduction_u64_multi_element_multi_moduli(result_step3, q, s_w, w, m)


@functools.partial(
  jax.jit, 
  static_argnames=("s_w", "w", "m")
)
def intt_three_step_bat_barrett_batch_multi_moduli(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    q,
    s_w,
    w,
    m,
):

  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8
  result_step1 = hpmatmul_bat_coef_rhs_batch_multi_moduli(poly_coef_2d, tf_step1)
  result_step1_mod_q = modred.barrett_reduction_u64_multi_element_multi_moduli(result_step1, q, s_w, w, m)
  result_step2 = jax.numpy.multiply(result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64))
  result_step2_mod_q = modred.barrett_reduction_u64_multi_element_multi_moduli(result_step2, q, s_w, w, m)
  result_step3 = hpmatmul_bat_coef_lhs_batch_multi_moduli(tf_step3, result_step2_mod_q)
  result_step3_mod_q = modred.barrett_reduction_u64_multi_element_multi_moduli(result_step3, q, s_w, w, m)
  return result_step3_mod_q



########################
# Three Step NTT -- Multiple Moduli Batch Implementation
# Moduli first dimension, batch second dimension -- this is different from previous one
########################
@jax.jit
def hpmatmul_bat_coef_lhs_multi_moduli_batch(lhs: jax.Array, y: jax.Array):
  """multi_moduli_batch implementation
    
  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.
  """
  rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "lmkpq,lbknq->lbmnp", #TODO: iterate each moduli and batch.
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


@jax.jit
def hpmatmul_bat_coef_rhs_multi_moduli_batch(y: jax.Array, rhs: jax.Array):
  """batch_multi_moduli implementation

  Expect the dtype of `lhs` and `rhs` to be `jnp.uint32`.
  """
  lhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)
  i8_products = jnp.einsum(
      "lbmkq,lknpq->lbmnp",
      lhs,
      rhs,
      preferred_element_type=jnp.int32,
  )
  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


@functools.partial(
  jax.jit, 
  static_argnames=("s_w", "w", "m")
)
def ntt_three_step_bat_barrett_multi_moduli_batch(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    q,
    s_w,
    w,
    m,
):
  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8

  result_step1 = hpmatmul_bat_coef_lhs_multi_moduli_batch(tf_step1, poly_coef_2d)
  result_step1_mod_q = modred.barrett_reduction_u64_multi_moduli_multi_element(result_step1, q, s_w, w, m)
  result_step2 = jax.numpy.multiply(
      result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64)[:, None, :, :]
  )
  result_step2_mod_q = modred.barrett_reduction_u64_multi_moduli_multi_element(result_step2, q, s_w, w, m)
  result_step3 = hpmatmul_bat_coef_rhs_multi_moduli_batch(result_step2_mod_q, tf_step3)

  return modred.barrett_reduction_u64_multi_moduli_multi_element(result_step3, q, s_w, w, m)


@functools.partial(
  jax.jit, 
  static_argnames=("s_w", "w", "m")
)
def intt_three_step_bat_barrett_multi_moduli_batch(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    q,
    s_w,
    w,
    m,
):

  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8
  result_step1 = hpmatmul_bat_coef_rhs_multi_moduli_batch(poly_coef_2d, tf_step1)
  result_step1_mod_q = modred.barrett_reduction_u64_multi_moduli_multi_element(result_step1, q, s_w, w, m)
  result_step2 = jax.numpy.multiply(result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64))
  result_step2_mod_q = modred.barrett_reduction_u64_multi_moduli_multi_element(result_step2, q, s_w, w, m)
  result_step3 = hpmatmul_bat_coef_lhs_multi_moduli_batch(tf_step3, result_step2_mod_q)
  result_step3_mod_q = modred.barrett_reduction_u64_multi_moduli_multi_element(result_step3, q, s_w, w, m)
  return result_step3_mod_q


@functools.partial(
  jax.jit, 
  # static_argnames=("s_w", "w", "m")
)
def ntt_three_step_bat_barrett_multi_moduli_batch_no_static(
    poly_coef_2d,
    tf_step1,
    coef_step2,
    tf_step3,
    q,
    s_w,
    w,
    m,
):
  """Jax implementation of Gentalman Sande NTT, vectorized implementation on VPU."""
  assert poly_coef_2d.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8

  result_step1 = hpmatmul_bat_coef_lhs_multi_moduli_batch(tf_step1, poly_coef_2d)
  result_step1_mod_q = modred.barrett_reduction_u64_multi_moduli_multi_element_no_static(result_step1, q, s_w, w, m)
  result_step2 = jax.numpy.multiply(
      result_step1_mod_q.astype(jnp.uint64), coef_step2.astype(jnp.uint64)[:, None, :, :]
  )
  result_step2_mod_q = modred.barrett_reduction_u64_multi_moduli_multi_element_no_static(result_step2, q, s_w, w, m)
  result_step3 = hpmatmul_bat_coef_rhs_multi_moduli_batch(result_step2_mod_q, tf_step3)

  return modred.barrett_reduction_u64_multi_moduli_multi_element_no_static(result_step3, q, s_w, w, m)


########################
# Three Step NTT -- Multiple Moduli Montgomery Reduction Implementation
########################
@jax.jit
def ntt_three_step_bat_montgomery_batch(v: jax.Array, tf_step1, coef_step2, tf_step3, q_low, q_high, q_inv_32, q):
  """
    NTT with modular u32 and Montgomery reduction
  """
  assert v.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32
  assert tf_step3.dtype == jnp.uint8

  #computation
  result_step1 = hpmatmul_bat_coef_lhs_batch(tf_step1, v)
  result_step1_reduced = modred.montgomery_reduce_u64_to_u32(result_step1, q_low, q_high, q_inv_32, q)
  result_step2 = jnp.multiply(result_step1_reduced.astype(jnp.uint64), coef_step2)
  result_step2_reduced = modred.montgomery_reduce_u64_to_u32(result_step2, q_low, q_high, q_inv_32, q)
  result_step3 = hpmatmul_bat_coef_rhs_batch(result_step2_reduced, tf_step3)
  result_step3_reduced = modred.montgomery_reduce_u64_to_u32(result_step3, q_low, q_high, q_inv_32, q)
  return result_step3_reduced


@jax.jit
def ntt_three_step_bat_montgomery_square_batch(v, tf_step1, coef_step2, q_low, q_high, q_inv_32, q):
  """
    NTT with modular u32 and Montgomery reduction
  """
  assert v.dtype == jnp.uint32
  assert tf_step1.dtype == jnp.uint8
  assert coef_step2.dtype == jnp.uint32

  #computation
  result_step1 = hpmatmul_bat_coef_lhs_batch(tf_step1, v)
  result_step1_reduced = modred.montgomery_reduce_u64_to_u32(result_step1, q_low, q_high, q_inv_32, q)
  result_step2 = jnp.multiply(result_step1_reduced.astype(jnp.uint64), coef_step2)
  result_step2_reduced = modred.montgomery_reduce_u64_to_u32(result_step2, q_low, q_high, q_inv_32, q)
  result_step3 = hpmatmul_bat_coef_rhs_batch(result_step2_reduced, tf_step1)
  result_step3_reduced = modred.montgomery_reduce_u64_to_u32(result_step3, q_low, q_high, q_inv_32, q)
  return result_step3_reduced


########################
# Butterfly NTT -- Multiple Moduli Bit-Reverse Implementation
########################
def ntt_bit_reverse_negacyclic_control_generation(n, q, psi):
  """Generate twist factors [psi^0, psi^1, ..., psi^(n-1)] mod q as uint64."""
  q64 = jnp.uint64(q)
  twist_list = []
  w = jnp.uint64(1)
  psi64 = jnp.uint64(psi)
  for _ in range(n):
    twist_list.append(w)
    w = (w * psi64) % q64
  return jnp.array(twist_list, dtype=jnp.uint64)


def ntt_negacyclic_bit_reverse_jax(a, q, psi, omega, twist=None, stage_w_pows=None, br_idx=None, s_w=None, w_barr=None, m_barr=None, total_iter=None):
  """Negacyclic NTT using the pure JAX bit-reverse NTT implementation.

  Process:
    1) Pre-twist: a[i] <- a[i] * psi^i mod q
    2) Vanilla NTT with omega = psi^2 using ntt_bit_reverse_jax
  """
  n = a.shape[-1]

  # Barrett reduction parameters
  if s_w is None:
    s_w, w_barr, m_barr = modred.barrett_control_generation_s_w(q)
  else:
    s_w = s_w
    w_barr = w_barr
    m_barr = m_barr

  q64 = jnp.uint64(q)
  a_arr = jnp.array(a, dtype=jnp.uint64)
  # Build twist factors psi^i mod q iteratively to avoid overflow.
  if twist is None:
    twist = ntt_bit_reverse_negacyclic_control_generation(n, q, psi)
  else:
    twist = twist
  mul = (a_arr * twist).astype(jnp.uint64)
  a_twisted = modred.barrett_reduction_u64(mul, q, s_w, w_barr, m_barr)

  return ntt_bit_reverse_jax(a_twisted, q, omega, stage_w_pows, br_idx, s_w, w_barr, m_barr, total_iter)

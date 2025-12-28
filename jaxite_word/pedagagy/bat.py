"""Util file for operations over matrices."""

import concurrent.futures

import jax
import jax.numpy as jnp
import util


@jax.jit
def int32_to_int8_arr(arr: jnp.ndarray) -> jnp.ndarray:
  """Decompose an int32 matrix into u8s."""
  return jax.lax.bitcast_convert_type(arr, new_dtype=jnp.uint8)


from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_bat_einsum(lhs: jax.Array, rhs: jax.Array, subscripts: str):
    """Basis Aligned Transformation (BAT) based matrix multiplication

    Args:
        lhs (jax.Array): input
        rhs (jax.Array): twiddle factor matrix
        subscripts (str): einsum subscripts

    Returns:
        jax.Array: result
    """
    #preprocess
    lhs = jax.lax.bitcast_convert_type(lhs, new_dtype=jnp.uint8)
    shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)

    #computation
    i8_products = jnp.einsum(subscripts, lhs, rhs, preferred_element_type=jnp.uint32)
    return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(-1,))


def hpmatmul_conv_outer_product(x: jax.Array, y: jax.Array) -> jax.Array:
  """Interleaved u8 matmul with fused einsum kernels.

  Args:
      x: The left matrix.
      y: The right matrix.

  Returns:
      The result matrix.
  """
  assert x.dtype == jnp.uint32
  assert y.dtype == jnp.uint32

  lhs: jax.Array = int32_to_int8_arr(x)
  rhs: jax.Array = int32_to_int8_arr(y)

  i8_products = jnp.einsum(
      "mnp,nkq->mkpq",
      lhs,
      rhs,
      preferred_element_type=jnp.uint32,
  )
  shift_factors = jnp.array(
      [
          [0, 8, 16, 24],
          [8, 16, 24, 32],
          [16, 24, 32, 40],
          [24, 32, 40, 48],
      ],
      dtype=jnp.uint32,
  )
  return jnp.sum(i8_products.astype(jnp.uint64) << shift_factors, axis=(2, 3))


@jax.jit
def hpmatmul_conv_conv(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Interleaved u8 matmul with padded 1D convolution.

  (reformulated as 2D convolution)
  This is similar to the integer implementation in TensorFHE.

  How do we map workload into Conv?

                   Left Mat                     Right Mat        ->
             <- in channel (C)->          <-Output Channel(O)->  ->
       -     xxxxxxxxxxxxxxxxxx      -     xxxxxxxxxxxxxxxxxx    ->    -
       ^     xxxxxxxxxxxxxxxxxx      ^     xxxxxxxxxxxxxxxxxx    ->    ^
       |     xxxxxxxxxxxxxxxxxx      |     xxxxxxxxxxxxxxxxxx    ->    |
     batch   xxxxxxxxxxxxxxxxxx     In     xxxxxxxxxxxxxxxxxx    ->  batch
      (N)    xxxxxxxxxxxxxxxxxx   channel  xxxxxxxxxxxxxxxxxx    ->   (N)
       |     xxxxxxxxxxxxxxxxxx     (I)    xxxxxxxxxxxxxxxxxx    ->    |
       v     xxxxxxxxxxxxxxxxxx      v     xxxxxxxxxxxxxxxxxx    ->    v
       -     xxxxxxxxxxxxxxxxxx      -     xxxxxxxxxxxxxxxxxx    ->    -

            Result Mat
        <-Output channel(C)->
        xxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxx
        xxxxxxxxxxxxxxxxxx

      Each x in the above example is a 1DConv

      <---W--->     <---W--->     <---W--->
      xxxxxxxxx  @  xxxxxxxxx  =  xxxxxxxxx

  Args:
    x: The left matrix.
    y: The right matrix.

  Returns:
    The result matrix.
  """

  assert x.dtype == jnp.uint32
  assert y.dtype == jnp.uint32

  lhs: jax.Array = jax.lax.bitcast_convert_type(x, new_dtype=jnp.uint8)  # bnmp
  rhs: jax.Array = jax.lax.bitcast_convert_type(y, new_dtype=jnp.uint8)  # nk1q
  # https://github.com/google/jax/issues/11483
  rhs = jax.lax.rev(rhs, [2])
  # rhs = jlax.rev(rhs, dimensions=[3])

  # basically an einsum of "mnp,nkq->mk(p+q)" but jax einsum doesn't support
  # convolution yet
  u8_products = jax.lax.conv_general_dilated(
      lhs,
      rhs,
      window_strides=(1,),
      padding=((3, 3),),
      dimension_numbers=("NCW", "IOW", "NCW"),
      preferred_element_type=jnp.uint32 # preferrably using jnp.uint32 but it's not supported by cuDNN, so use jnp.float32 for cuDNN.
  )

  shift_factors = jnp.array([0, 8, 16, 24, 32, 40, 48], dtype=jnp.uint32)
  return jnp.sum(u8_products.astype(jnp.uint64) << shift_factors, axis=(2,))


########################################################
# Basis Align Transformation (BAT) Version
# This part includes version for illustration purpose to contrast with SotA GPU's implementation.
# Also includes version for deployment purpose, which completes in O(N) time for MatMul with total N elements.
# Refer to hpmatmul_offline_bat_deployment for deployment version.
########################################################
def rechunkify_after_chunkwise_add(arr_a, chunkwidth):
  """Rechunkify after chunkwise add.

  Args:
      arr_a: The input array.
      chunkwidth: The chunkwidth.

  Returns:
      The rechunkified array.
  """
  dtype_double_length = jnp.uint16
  if chunkwidth == 16:
    dtype_double_length = jnp.uint32
  elif chunkwidth == 32:
    dtype_double_length = jnp.uint64

  # assert isinstance(arr_a, jnp.array)
  # assume the precision of partial sum is <= 2 * precision of input value.
  bitmask = (1 << chunkwidth) - 1

  # # Data Type Illustration
  #     We need to accumulate these data
  #     - Could directly perform bitwidth concatenation to generate the final
  #       result if there is no overlap across each partial sum
  #          LSB             MSB
  #         |-----------------> bit
  #         |   a0
  #         |  ==--
  #         |     a1
  #         |    ==--
  #         |        a2
  #         |       ==--
  #         |          a3
  #         v         ==--

  #   whole        a0   a1   a2   a3
  # precision     ==-- ==-- ==-- ==--

  #   lower       a0 a1 a2 a3
  #    half       == == == ==

  #   upper       a0 a1 a2 a3
  #    half       -- -- -- --

  # # Chunk Splitting -> upper and lower half
  # padding to align
  #   lower       a0 a1 a2 a3 0
  #    half       == == == == ==

  #   upper       0  a0 a1 a2 a3
  #    half       -- -- -- -- --

  # # Vectorized Accumulation
  #   lower         a0    a1    a2    a3    0
  #    half         ==    ==    ==    ==    ==
  #                 +     +     +     +     +
  #   upper         0     a0    a1    a2    a3
  #    half         --    --    --    --    --

  # -> result       b0    b1    b2    b3    b4
  #                 --  1/0-- 1/0-- 1/0--   --
  #    (b1 and b4 does not have carry for sure.)

  #             Each result chunk might have one more bit for carry.
  #             Perform one more chunk decomposition and accumulation.

  # # One more Chunk Splitting for partial sum "b" to take care of carry bit.
  #    carry      b0  b1  b2  b3  b4
  #               0  1/0 1/0 1/0  0

  #    carry      b4  b0  b1  b2  b3
  #    right      0   0  1/0 1/0 1/0
  #    shift
  #    (wrap around rotation, b4 is always zero so will be correct)
  #               +   +   +   +   +
  #   lower       b0  b1  b2  b3  b4
  #    half       --  --  --  --  --
  #               =   =   =   =   =
  #               c0  c1  c2  c3  c4
  # ->            --  --  --  --  1/0--
  #    (! c4 might overflow, need one more chunk decomposition)

  #               c0  c1  c2  c3  c4  c5
  # ->            --  --  --  --  --  1/0

  # Chunk Splitting -> upper and lower half
  arr_a_lower_half = jnp.bitwise_and(arr_a, bitmask)
  arr_a_upper_half = jnp.right_shift(arr_a, chunkwidth)

  # Padding to align
  arr_a_lower_half_pad = jnp.pad(arr_a_lower_half, (0, 1))
  arr_a_upper_half_pad = jnp.pad(arr_a_upper_half, (1, 0))

  # Vectorized Accumulation
  arr_b = jnp.add(
      arr_a_lower_half_pad.astype(dtype_double_length),
      arr_a_upper_half_pad.astype(dtype_double_length),
  )

  while not jnp.all(arr_b <= bitmask):
    arr_b_lower_half = jnp.bitwise_and(arr_b, bitmask)
    arr_b_carry = jnp.right_shift(arr_b, chunkwidth)
    arr_b = jnp.roll(arr_b_carry, 1, axis=-1)
    arr_b = jnp.add(arr_b_lower_half, arr_b)

  # Vectorized Accumulation
  arr_c = arr_b

  # break top chunk into upper and lower to avoid overflow.
  arr_c = jnp.pad(arr_c, (0, 1))
  arr_c = arr_c.at[-1].set(jnp.right_shift(arr_c[-2], chunkwidth))
  arr_c = arr_c.at[-2].set(jnp.bitwise_and(arr_c[-2], bitmask))

  return arr_c


def basis_align_transformation_illustration(
    x, total_in_precision=32, chunkwidth=8, q=4294967291
):
  """This is the implementation of Basis Align Transformation (BAT);
  Major improvement to achieve dense matrix.

  Args:
    x: The input matrix.
    total_in_precision: The total precision of the input matrix.
    chunkwidth: The chunkwidth.
    q: The modulus.

  Returns:
    The dense matrix.

  Steps:
  1. break x into [x0, x1, x2, x3]
  2. reform [x0, x1, x2, x3] into the output
  [
  x0    r00    r00    r00    # 2^0
  x1   x0+r01  r01    r01    # 2^8
  x2   x1+r02 x0+r02  r02    # 2^16
  x3   x2+r03 x1+r03 x0+r03  # 2^24
  ]

  Note: prefilled value are just examples.
    We pick largest 2^32-1 to make sure that intermediate results might
    exceed 32-bit precision range, and expose potential precision overflow.
  """
  dtype_double_length = jnp.uint16
  chunk_upper_bound = (1 << 8) - 1
  if chunkwidth == 16:
    dtype_double_length = jnp.uint32
    chunk_upper_bound = (1 << 16) - 1
  elif chunkwidth == 32:
    dtype_double_length = jnp.uint64
    chunk_upper_bound = (1 << 32) - 1

  total_chunk_num = int(jnp.ceil(total_in_precision / chunkwidth))

  # the number of row in left matrix
  height = total_chunk_num + total_chunk_num - 1
  x_dtype = util.chunk_decomposition(x, chunkwidth)
  x_dense = jnp.zeros(
      (total_chunk_num + total_chunk_num - 1, total_chunk_num),
      dtype=dtype_double_length,
  )
  for j in range(total_chunk_num):
    upper_idx = min(total_chunk_num, x_dtype.shape[0] + j)
    x_dense = x_dense.at[j:upper_idx, j].set(x_dtype[0 : upper_idx - j])

  # [
  # x0              # 2^0
  # x1 x0           # 2^8
  # x2 x1 x0        # 2^16
  # x3 x2 x1 x0     # 2^24
  # -----------
  #    x3 x2 x1     # 2^32  iterate all elements in the bottom block
  #       x3 x2     # 2^40
  #          x3     # 2^48
  # ]

  # Perform BAT to the following block of the matrix
  # j    2  1  0
  #     x3 x2 x1   # 2^32  i=0
  #        x3 x2   # 2^40  i=1
  #           x3   # 2^48  i=2

  for i in range(x_dtype.shape[0] - 1):
    for j in range(x_dtype.shape[0] - 1 - i):
      basis = (total_chunk_num + i) * chunkwidth
      projected_data = (int(x_dtype[i + j + 1]) << basis) % q
      r = util.chunk_decomposition(projected_data, chunkwidth).astype(
          dtype_double_length
      )

      x_dense = x_dense.at[: len(r), total_chunk_num - 1 - j].set(
          jnp.add(r, x_dense[: len(r), total_chunk_num - 1 - j])
      )

  while not jnp.all(x_dense <= chunk_upper_bound) or not jnp.all(
      x_dense[total_chunk_num:, :] == 0
  ):
    for j in range(total_chunk_num - 1):
      # Iterate over different columns
      if not jnp.all(x_dense[:, total_chunk_num - 1 - j] <= chunk_upper_bound):
        arr_new_chunkified = rechunkify_after_chunkwise_add(
            x_dense[:, total_chunk_num - 1 - j], chunkwidth
        )
        x_dense = x_dense.at[:, total_chunk_num - 1 - j].set(
            arr_new_chunkified[:height]
        )

    # j    2  1  0
    #     x3 x2 x1   # 2^32  i=0
    #        x3 x2   # 2^40  i=1
    #           x3   # 2^48  i=2
    for i in range(x_dtype.shape[0] - 1):
      for j in range(x_dtype.shape[0] - 1 - i):
        data = x_dense[total_chunk_num + i, total_chunk_num - 1 - j]
        if data > 0:
          basis = (total_chunk_num + i) * chunkwidth
          projected_data = (int(data) << basis) % q
          r = util.chunk_decomposition(projected_data, chunkwidth).astype(
              dtype_double_length
          )

          x_dense = x_dense.at[: len(r), total_chunk_num - 1 - j].set(
              jnp.add(r, x_dense[: len(r), total_chunk_num - 1 - j])
          )
          x_dense = x_dense.at[
              total_chunk_num + i, total_chunk_num - 1 - j
          ].set(0)
  return x_dense[:total_chunk_num, :].astype(jnp.uint8)


def hpmatmul_offline_bat_illustration(mat_a, q):
  """Convert the input (m,n) matrix into (m,n,p,q), i.e.

  replace each element in the original matrix by a p*q matrix (p==q).

  Args:
    mat_a: The input matrix.
    q: The modulus.

  Returns:
    The converted matrix.
  """
  assert mat_a.dtype == jnp.uint32  # This version is defined for 32-bit input.
  m, n = mat_a.shape[0], mat_a.shape[1]
  total_in_precision = 32
  chunkwidth = 8
  # Convert left-side matrix
  total_chunk_num = int(jnp.ceil(total_in_precision / chunkwidth))

  left_mat = jnp.zeros(
      (m, n, total_chunk_num, total_chunk_num), dtype=jnp.uint16
  )

  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for i in range(m):
      for k in range(n):
        futures.append(
            executor.submit(
                basis_align_transformation_illustration,
                mat_a[i, k],
                total_in_precision,
                chunkwidth,
                q,
            )
        )

    for future, (i, k) in zip(
        futures, [(i, k) for i in range(m) for k in range(n)]
    ):
      left_mat = left_mat.at[i, k, :, :].set(future.result())

  return left_mat


def hpmatmul_offline_bat_deployment(mat_a, q):
  """Convert the input (m,n) matrix into (m,n,p,q), i.e.

  replace each element in the original matrix by a p*q matrix (p==q).
  Using direct mathematical transformation

  Args:
    mat_a: The input matrix.
    q: The modulus.

  Returns:
    The converted matrix.
  """
  assert q <= 2**31
  num_bytes = 4
  matrix_u64 = mat_a.astype(jnp.uint64)
  shift_amounts = jnp.arange(num_bytes, dtype=jnp.uint64) * 8
  matrix_u64_byteshifted = matrix_u64[:, :, None] << shift_amounts[None, None, :]
  matrix_u64_byteshifted_mod_modulus = (matrix_u64_byteshifted % q).astype(jnp.uint32)
  matrix_u8 = jax.lax.bitcast_convert_type(matrix_u64_byteshifted_mod_modulus, new_dtype=jnp.uint8).transpose(0, 1, 3, 2)

  return matrix_u8

def hpmatmul_golden(mat_a, mat_b, modulus_32):
  mat_reference_result = []
  for i in range(mat_a.shape[0]):
    mat_reference_result_row = []
    for j in range(mat_b.shape[1]):
      acc_res = 0
      for k in range(mat_a.shape[1]):
        acc_res += int(mat_a[i, k]) * int(mat_b[k, j])
      mat_reference_result_row.append(acc_res % modulus_32)
    mat_reference_result.append(mat_reference_result_row)
  return mat_reference_result


def matmul_bat_pallas(lhs: jax.Array, rhs: jax.Array, subscripts: str):
  """Pallas implementation of matmul_bat_einsum.

  Currently supports subscripts 'bnkq,mnpq->bmkp' corresponding to
  Batched LHS (B, N, K) and RHS (M, N, 4, 4) -> Output (B, M, K).

  Args:
      lhs (jax.Array): Input LHS (B, N, K) - uint32
      rhs (jax.Array): Input RHS (M, N, 4, 4) - uint8
      subscripts (str): Einsum subscripts (must be compatible)

  Returns:
      jax.Array: Result (B, M, K) - uint64
  """
  # Dimensions logic based on 'bnkq,mnpq->bmkp'
  # lhs: (B, N, K) -> bitcast -> (B, N, K, 4)
  # rhs: (M, N, P, Q) -> (M, N, 4, 4)
  # Output: (B, M, K) -> effectively P is contracted/shifted?
  # The original sum is over axis=-1, which corresponds to P (sized 4).

  B, N, K = lhs.shape
  M, _N, P, Q = rhs.shape

  # Basic validation
  # We assume inputs are compatible with the specific Pallas kernel logic
  # Specifically checking for the test case dimensions/structure
  assert N == _N, f"Dimension mismatch N: {N} vs {_N}"
  # For matmul_bat_einsum, standard usage has Q=4 (from bitcast)
  assert Q == 4, f"RHS Q dim must be 4, got {Q}"
  assert P == 4, f"RHS P dim must be 4, got {P}"

  shift_factors = jnp.array([0, 8, 16, 24], dtype=jnp.uint32)

  # Pad to power of 2 for Pallas Triton compatibility
  def next_pow2(v):
      if v == 0: return 1
      return 1 << (v - 1).bit_length()

  B_pad_val = next_pow2(B)
  if B_pad_val < 16: B_pad_val = 16
  N_pad = next_pow2(N)
  K_pad = next_pow2(K)
  M_pad = next_pow2(M)

  # Transpose inputs outside kernel for better memory layout in TPU
  # LHS: (B, N, K) -> (B, K, N)
  lhs_transposed = lhs.transpose(0, 2, 1)

  # RHS: (M, N, P, Q) -> (M, P, N, Q)
  rhs_transposed = rhs.transpose(0, 2, 1, 3)

  # Prepare inputs for Kernel (Bitcast and Flatten)
  # LHS: (B, K, N) uint32 -> (B, K, N, 4) uint8 -> (B, K, N*4) uint8
  lhs_u8 = jax.lax.bitcast_convert_type(lhs_transposed, jnp.uint8)
  # LHS is now (B, K, N, 4)
  B, K_dim, N_dim, _ = lhs_u8.shape
  lhs_flat = lhs_u8.reshape(B, K_dim, N_dim * 4)

  # RHS: (M, P, N, Q) uint8 -> (M, P, N*4) uint8
  M_dim, P_dim, N_dim_r, Q_dim = rhs_transposed.shape
  # Check if N dimensions match
  assert N_dim == N_dim_r
  assert Q_dim == 4

  rhs_flat = rhs_transposed.reshape(M_dim, P_dim, N_dim * 4)

  X_dim = N_dim * 4

  # Pad inputs
  # Pad inputs
  lhs_padded = jnp.pad(lhs, ((0, B_pad_val - B), (0, N_pad - N), (0, K_pad - K)))

  rhs_padded = jnp.pad(rhs, ((0, M_pad - M), (0, N_pad - _N), (0, 0), (0, 0)))

  def kernel(lhs_ref, rhs_ref, shift_ref, out_ref):
    # Load blocks
    L = lhs_ref[...] # (B_chunk, N, K) - uint32
    R = rhs_ref[...]
    shift = shift_ref[...]

    # helper for dimensions
    B_chunk, N_dim, K_dim = L.shape
    M_dim, _, P_dim, Q_dim = R.shape

    # 1. Bitcast conversion inside kernel using pltpu
    # Reshape to ensure 4D shape (B, N, K, 4)
    L_u8 = pltpu.bitcast(L, jnp.uint8).reshape(B_chunk, N_dim, K_dim, 4)

    # 2. Reshape for efficient dot product
    # We want to contract N and Q (last dim of L_u8, last dim of R).
    # L structure: B, N, K, Q
    # R structure: M, N, P, Q
    # Result target: B, M, K, P (before shift sum)

    # Flatten contracting dims (N, Q) -> X = N*4
    # L -> (B, K, N, Q) -> (B, K, N*4)
    L_flat = L_u8.transpose(0, 2, 1, 3).reshape(B_chunk, K_dim, N_dim * 4)

    # R -> (M, P, N, Q) -> (M, P, N*4)
    R_flat = R.transpose(0, 2, 1, 3).reshape(M_dim, P_dim, N_dim * 4)

    # 3. Dot product
    # (B, K, X) dot (M, P, X).T
    # reshape to 2D: (B*K, X) dot (X, M*P) -> (B*K, M*P)

    L_2d = L_flat.reshape(-1, N_dim * 4).astype(jnp.int32)
    R_2dt = R_flat.reshape(-1, N_dim * 4).T.astype(jnp.int32) # (X, M*P)

    prods = jnp.dot(L_2d, R_2dt, preferred_element_type=jnp.int32) # (B*K, M*P)

    # 4. Reshape back to (B, K, M, P)
    prods_4d = prods.reshape(B_chunk, K_dim, M_dim, P_dim)

    # 5. Shift and Sum
    # Shift factors passed as input

    prods_shifted = prods_4d.astype(jnp.uint64) << shift
    res = jnp.sum(prods_shifted, axis=-1) # (B, K, M)

    # 6. Store result (B, M, K)
    out_ref[...] = res.transpose(0, 2, 1)

  # Grid strategy: Tile B, keep M, K, N full
  # Process full batch in one block to avoid grid issues
  B_BLOCK = B_pad_val

  grid = (1,)

  in_specs = [
      pl.BlockSpec((B_BLOCK, N_pad, K_pad), lambda i: (i * B_BLOCK, 0, 0)),
      pl.BlockSpec((M_pad, N_pad, 4, 4), lambda i: (0, 0, 0, 0)),
      pl.BlockSpec((4,), lambda i: (0,))
  ]
  out_specs = pl.BlockSpec((B_BLOCK, M_pad, K_pad), lambda i: (i * B_BLOCK, 0, 0))

  out_shape = jax.ShapeDtypeStruct((B_pad_val, M_pad, K_pad), jnp.uint64)

  ret = pl.pallas_call(
      kernel,
      out_shape=out_shape,
      grid=grid,
      in_specs=in_specs,
      out_specs=out_specs
  )(lhs_padded, rhs_padded, shift_factors)

  return ret[:B, :M, :K]

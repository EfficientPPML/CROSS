import jax
import jax.numpy as jnp
import util
import bconv
from polynomial import Polynomial
import numpy as np

# enable 64-bit computation in jax
jax.config.update("jax_enable_x64", True)


def is_power_of_two(x: int) -> bool:
  """Returns True if x is a power of two."""
  return x > 0 and (x & (x - 1)) == 0


def mat_1d_shuffle_to_2d(coef_map: jnp.ndarray, r: int, c: int):
  """Memory Aligned Transformation.

  Perform 1D data shuffing of O(N) in matrix fashion with O(sqrt(N)) memory
  cost.
  Precomputes the 2D indices.

  Factor coef_map (length r*c) into row_perm (len r) and col_perm (len c) such
  that:
    coef_map.reshape(r,c)[i,j] == row_perm[i]*c + col_perm[j]

  Args:
    coef_map: The 1D permuted indices of shape (r*c,).
    r: The number of rows for the 2D matrix.
    c: The number of columns for the 2D matrix.

  Returns:
    row_perm: int32[r]
    col_perm: int32[c]
  """
  if coef_map.ndim != 1 or coef_map.shape[0] != r * c:
    raise ValueError(
        f'coef_map must be 1D of length r*c. Got shape {coef_map.shape},'
        f' r*c={r*c}.'
    )
  if r <= 0 or c <= 0:
    raise ValueError('r and c must be positive.')
  # (Recommended, since your degree is power-of-2)
  if not (is_power_of_two(r) and is_power_of_two(c)):
    raise ValueError('For your setting, r and c should be powers of two.')
  coef2d = coef_map.reshape(r, c)

  # If coef2d[i,j] = row_perm[i]*c + col_perm[j], then:
  row_perm = (coef2d[:, 0] // c).astype(jnp.int32)
  col_perm = (coef2d[0, :] % c).astype(jnp.int32)

  coef2d_h = np.asarray(jax.device_get(coef2d))
  row_h = np.asarray(jax.device_get(row_perm))
  col_h = np.asarray(jax.device_get(col_perm))
  expected_h = row_h[:, None] * c + col_h[None, :]
  if not np.array_equal(coef2d_h, expected_h):
    raise ValueError(
        'coef_map is NOT decomposable into a single global row permutation +'
        f' column permutation for r={r}, c={c}. (i.e., not P_row ⊗ P_col). Pick'
        ' a different (r,c) factorization, or keep using jnp.take(a, coef_map,'
        ' axis=2).'
    )

  return row_perm, col_perm


class HERot:
  """HERot class."""

  def __init__(self, r, c, dnum, rotate_in_ciphertext_moduli, extend_moduli):
    self.r = r
    self.c = c
    self.dnum = dnum
    self.rotate_in_ciphertext_moduli = rotate_in_ciphertext_moduli
    self.extend_moduli = extend_moduli
    self.overall_moduli_init = (
        self.rotate_in_ciphertext_moduli + self.extend_moduli
    )
    self.bconv = bconv.BConvBarrett(self.overall_moduli_init)
    self.evalkey_a_vector = jnp.zeros((dnum, 0), dtype=jnp.uint64)
    self.evalkey_b_vector = jnp.zeros((dnum, 0), dtype=jnp.uint64)
    self.coef_map = jnp.zeros((0,), dtype=jnp.int32)

  def control_gen(self, batch=1, degree_layout=None, perf_test=False):
    """Generates control parameters and precomputes values for rotations.

    This method sets up various parameters and precomputations required for
    the rotation operations, including roots of unity, inverse roots of unity,
    modulus switching parameters, and indices for basis changes within the
    BConvBarrett instance. It also pre-allocates Polynomial objects.

    Args:
      batch: The batch size of the ciphertexts.
      degree_layout: The layout of the polynomial degrees. Defaults to (r * c,).
      perf_test: If True, uses random parameters instead of computing actual
        roots of unity, useful for performance testing.
    """
    if degree_layout is None:
      degree_layout = (self.r * self.c,)
    self.degree_layout = degree_layout
    sizeQl_in = len(self.rotate_in_ciphertext_moduli)
    sizeQlP_in = len(self.extend_moduli) + sizeQl_in
    alpha = (sizeQl_in + self.dnum - 1) // self.dnum
    ring_dim = self.r * self.c
    overall_moduli = self.rotate_in_ciphertext_moduli + self.extend_moduli
    self.perf_test = perf_test


    ## parameters generation for approximation mod down
    current_moduli = self.extend_moduli
    target_moduli = [
        item for item in overall_moduli if item not in current_moduli
    ]

    P = 1
    for moduli in current_moduli:
      P *= moduli
    PInvModq_approx_down = [util.modinv(P, q) for q in target_moduli]

    self.overall_moduli = overall_moduli
    self.PInvModq = jnp.asarray(PInvModq_approx_down, dtype=jnp.uint32).reshape(
        sizeQl_in
    )
    self.sizeQlP, self.sizeQl = sizeQlP_in, sizeQl_in
    self.batch = batch

    # BConv control generation
    original_moduli_extract_index = []
    for i in range(sizeQl_in):
      if i % alpha == 0:
        original_moduli_extract_index.append([i])
      else:
        original_moduli_extract_index[-1].append(i)
    numPartQl = (sizeQl_in + alpha - 1) // alpha

    control_indices_list = []

    self.select_tower_index = []
    self.non_select_tower_index = []

    # 1. Basis change for key switch decomposition
    for part in range(numPartQl):
      sel_index = original_moduli_extract_index[part]
      non_sel_index = [
          i for i in range(len(overall_moduli)) if i not in sel_index
      ]
      self.select_tower_index.append(sel_index)
      self.non_select_tower_index.append(non_sel_index)
      control_indices_list.append((sel_index, non_sel_index))

    # Precompute restore indices for scatter optimization
    self.restore_indices = []
    for part in range(numPartQl):
      sel_index = self.select_tower_index[part]
      non_sel_index = self.non_select_tower_index[part]

      concat_order = sel_index + non_sel_index

      restore_index = [0] * len(concat_order)
      for pos, val in enumerate(concat_order):
        restore_index[val] = pos

      self.restore_indices.append(jnp.array(restore_index, dtype=jnp.uint16))

    # 2. Basis change for approximation modulus switch
    rotate_indices = list(range(sizeQl_in))
    extend_indices = list(range(sizeQl_in, sizeQlP_in))
    control_indices_list.append((extend_indices, rotate_indices))

    self.bconv.control_gen(control_indices_list, perf_test=perf_test)

    ct_in_shapes = {
        'batch': batch,
        'num_elements': 2,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': sizeQl_in,
        'degree_layout': degree_layout,
    }
    self.ct_in = Polynomial(
        ct_in_shapes,
        parameters={'moduli': overall_moduli[:sizeQl_in], 'BAT_lazy': False},
    )

    self.ct_parts = []
    for part in range(numPartQl):
      _target_indices_list = self.non_select_tower_index[part]
      _target_moduli_list = [overall_moduli[i] for i in _target_indices_list]
      _num_moduli_part = len(_target_moduli_list)
      ct_part_shapes = {
          'batch': batch,
          'num_elements': 2,
          'degree': ring_dim,
          'precision': 32,
          'num_moduli': _num_moduli_part,
          'degree_layout': degree_layout,
      }
      self.ct_parts.append(
          Polynomial(
              ct_part_shapes,
              parameters={'moduli': _target_moduli_list, 'BAT_lazy': False},
          )
      )

    ct_full_shapes = {
        'batch': batch,
        'num_elements': 2,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': sizeQlP_in,
        'degree_layout': degree_layout,
    }
    self.ct_full = Polynomial(
        ct_full_shapes,
        parameters={'moduli': self.overall_moduli, 'BAT_lazy': False},
    )

    ct_approx_shapes = {
        'batch': batch,
        'num_elements': 2,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': sizeQlP_in - sizeQl_in,
        'degree_layout': degree_layout,
    }
    self.ct_approx = Polynomial(
        ct_approx_shapes,
        parameters={'moduli': self.extend_moduli, 'BAT_lazy': False},
    )

  def setup_rotate(self, evalkey_a_vector, evalkey_b_vector, coef_map):
    self.evalkey_a_vector = evalkey_a_vector.astype(jnp.uint64)
    self.evalkey_b_vector = evalkey_b_vector.astype(jnp.uint64)
    self.coef_map = jnp.asarray(coef_map, dtype=jnp.int32)

  def rotate(
      self,
      in_ciphertexts,
  ):
    """Rotates the input ciphertext."""
    batch, r, c, dnum = self.batch, self.r, self.c, self.dnum

    sizeQlP, sizeQl = self.sizeQlP, self.sizeQl
    select_tower_index = self.select_tower_index
    if (
        not hasattr(self, '_jax_arrays_precomputed')
        or not self._jax_arrays_precomputed
    ):
      # Convert lists of lists/tuples to lists of JAX arrays once
      self._restore_indices_jax = [
          jnp.array(idx, jnp.uint16) for idx in self.restore_indices
      ]
      self._select_tower_index_jax = [
          jnp.array(idx, jnp.uint16) for idx in select_tower_index
      ]
      self._evalkey_b_vector_jax = jnp.stack(self.evalkey_b_vector, axis=0)
      self._evalkey_a_vector_jax = jnp.stack(self.evalkey_a_vector, axis=0)
      self._jax_arrays_precomputed = True

    ring_dim = r * c

    overall_moduli_jax = jnp.asarray(self.overall_moduli, dtype=jnp.uint32)
    original_moduli = jnp.expand_dims(
        overall_moduli_jax[:sizeQl], axis=(0, 1, 2, 3)
    )
    PInvModq_jax = jnp.expand_dims(
        jnp.array(self.PInvModq, jnp.uint64), axis=(0, 1, 2, 3)
    )
    in_tower = in_ciphertexts.polynomial[:, -1:, ..., :sizeQl]

    # ---------- Step 1: Keyswitch (Accumulation) ----------
    # Use the precomputed JAX arrays
    restore_indices_jax = self._restore_indices_jax
    select_tower_index_jax = self._select_tower_index_jax

    self.ct_in.polynomial = in_tower
    self.ct_in.to_coeffs_form()
    parts_ct_clone_coef = self.ct_in.polynomial

    res0 = jnp.zeros((batch, 1, r, c, sizeQlP), dtype=jnp.uint64)
    res1 = jnp.zeros((batch, 1, r, c, sizeQlP), dtype=jnp.uint64)

    def compute_parts_ct_ext(p):
      input_for_bconv = parts_ct_clone_coef[..., select_tower_index_jax[p]]
      parts_ct_clone_eval = self.bconv.basis_change_bat(
          input_for_bconv, control_index=p
      ).astype(jnp.uint64)

      ct_part = self.ct_parts[p]
      ct_part.polynomial = parts_ct_clone_eval.astype(jnp.uint32)
      ct_part.to_ntt_form()

      # Need to pad zeros.
      # The padding zeros should be sharded.
      # Hence borrow the shape from in_tower, which is already sharded.
      ct_part_aligned = ct_part.polynomial + (in_tower[..., :1] * 0)
      parts_ct_ext = jnp.concatenate(
          [in_tower[..., select_tower_index_jax[p]], ct_part_aligned],
          axis=-1,
      )
      return parts_ct_ext[..., restore_indices_jax[p]]

    if dnum > 0:
      # Reduce each product immediately to prevent Barrett overflow when
      # the accumulated sum across dnum partitions exceeds 2^(2*modulus_bits).
      rot_ff = self.ct_full.ntt_ctx.ff_ctx
      next_parts_ct_ext = compute_parts_ct_ext(0)

      for part in range(dnum - 1):
        cur_parts_ct_ext = next_parts_ct_ext

        next_parts_ct_ext = compute_parts_ct_ext(part + 1)

        prod_b = rot_ff.modular_reduction(
            cur_parts_ct_ext * self._evalkey_b_vector_jax[part]
        ).astype(jnp.uint64)
        prod_a = rot_ff.modular_reduction(
            cur_parts_ct_ext * self._evalkey_a_vector_jax[part]
        ).astype(jnp.uint64)
        res0 = res0 + prod_b
        res1 = res1 + prod_a

      prod_b = rot_ff.modular_reduction(
          next_parts_ct_ext * self._evalkey_b_vector_jax[dnum - 1]
      ).astype(jnp.uint64)
      prod_a = rot_ff.modular_reduction(
          next_parts_ct_ext * self._evalkey_a_vector_jax[dnum - 1]
      ).astype(jnp.uint64)
      res0 = res0 + prod_b
      res1 = res1 + prod_a

    # ---------- Steps 2-5: Sequential Component Processing ----------
    base0 = in_ciphertexts.polynomial[:, 0:1, ..., :sizeQl]
    final_components = []

    for i, comp in enumerate([res0, res1]):
      # Step 2: Modulus reduction (component-wise)
      self.ct_full.polynomial = comp
      self.ct_full.mod_reduce()
      reduced = self.ct_full.polynomial.astype(jnp.uint32)

      # Step 3: Approximation modulus switch
      self.ct_approx.polynomial = reduced[..., sizeQl:]
      self.ct_approx.to_coeffs_form()
      ql_from_p = self.bconv.basis_change_bat(
          self.ct_approx.polynomial, control_index=dnum
      )

      self.ct_in.polynomial = ql_from_p.astype(jnp.uint32)
      self.ct_in.to_ntt_form()

      ql_part = reduced[..., :sizeQl]
      diff = ql_part - self.ct_in.polynomial
      sub = jnp.where(
          ql_part < self.ct_in.polynomial,
          diff + original_moduli,
          diff,
      )

      self.ct_in.polynomial = sub
      self.ct_in.modmul(PInvModq_jax)
      res_i = self.ct_in.polynomial

      # Step 4: Add base component for the first ciphertext component
      if i == 0:
        res_i = res_i + base0
        res_i = jnp.where(
            res_i >= original_moduli, res_i - original_moduli, res_i
        )

      # Step 5: Automorphism (permutation)
      res_i_flat = res_i.reshape(batch, ring_dim, sizeQl)
      final_components.append(jnp.take(res_i_flat, self.coef_map, axis=1))

    ks_results = jnp.stack(final_components, axis=1)
    self.ct_in.polynomial = ks_results
    self.ct_in.element_count = ks_results.shape[1]
    return self.ct_in

  @staticmethod
  def make_rotate_fn(herot_instance):
    """Build a pure rotation function from a configured HERot instance.

    Returns a closure (ct_data, eval_a, eval_b, coef_map) -> rotated_data
    that performs the same computation as rotate() but takes per-rotation
    state as arguments (suitable for jax.lax.scan).

    The returned function captures all level-shared constants (NTT contexts,
    BConv matrices, twiddle factors, etc.) from herot_instance.

    Args:
        herot_instance: A fully configured HERot (control_gen + setup_rotate
            already called). Only the shared state is captured; eval keys
            and coef_map are NOT captured (they become function arguments).

    Returns:
        A pure function: (ct_data, eval_a, eval_b, coef_map) -> jnp.ndarray
        where ct_data is (batch, 2, *degree_layout, sizeQl) and the output
        is (batch, 2, ring_dim, sizeQl).
    """
    inst = herot_instance
    # Force lazy precomputation so JAX arrays exist
    if not hasattr(inst, '_jax_arrays_precomputed') or not inst._jax_arrays_precomputed:
      inst._restore_indices_jax = [
          jnp.array(idx, jnp.uint16) for idx in inst.restore_indices]
      inst._select_tower_index_jax = [
          jnp.array(idx, jnp.uint16) for idx in inst.select_tower_index]
      inst._jax_arrays_precomputed = True

    # ---- Capture shared constants ----
    batch = inst.batch
    r, c, dnum = inst.r, inst.c, inst.dnum
    sizeQl, sizeQlP = inst.sizeQl, inst.sizeQlP
    ring_dim = r * c

    restore_indices = inst._restore_indices_jax
    select_tower_index = inst._select_tower_index_jax

    original_moduli = jnp.expand_dims(
        jnp.asarray(inst.overall_moduli[:sizeQl], dtype=jnp.uint32),
        axis=(0, 1, 2, 3))
    PInvModq = jnp.expand_dims(
        jnp.array(inst.PInvModq, jnp.uint64), axis=(0, 1, 2, 3))

    # NTT/FF contexts from pre-allocated Polynomial objects
    ntt_ctx_in = inst.ct_in.ntt_ctx
    ff_ctx_in = ntt_ctx_in.ff_ctx
    ntt_ctx_parts = [p.ntt_ctx for p in inst.ct_parts]
    ntt_ctx_full = inst.ct_full.ntt_ctx
    ff_ctx_full = ntt_ctx_full.ff_ctx
    ntt_ctx_approx = inst.ct_approx.ntt_ctx

    # BConv: pre-extract per-partition constants to avoid list indexing
    bconv_inst = inst.bconv
    # Also capture the approx-mod-down BConv control (index = dnum)
    # bconv.basis_change_bat is already a pure function on arrays

    # ---- Define pure inline helpers ----
    def _ntt(data, ntt_ctx):
      shape = data.shape
      return ntt_ctx.ntt(data.reshape(-1, r, c, shape[-1])).reshape(shape)

    def _intt(data, ntt_ctx):
      shape = data.shape
      return ntt_ctx.intt(data.reshape(-1, r, c, shape[-1])).reshape(shape)

    def _modmul(data, other, ff_ctx):
      return ff_ctx.modular_reduction(
          data.astype(jnp.uint64) * other.astype(jnp.uint64)
      ).astype(jnp.uint32)

    def _mod_reduce(data, ff_ctx):
      return ff_ctx.modular_reduction(data.astype(jnp.uint64)).astype(jnp.uint32)

    # ---- The pure rotation function ----
    def rotate_fn(ct_data, eval_a, eval_b, coef_map):
      """Pure HERot rotation.

      Args:
          ct_data: (batch, 2, *degree_layout, sizeQl) uint32 input ciphertext
          eval_a: (dnum, *degree_layout, sizeQlP) uint64 rotation eval key A
          eval_b: (dnum, *degree_layout, sizeQlP) uint64 rotation eval key B
          coef_map: (ring_dim,) int32 coefficient permutation

      Returns:
          (batch, 2, ring_dim, sizeQl) uint32 rotated ciphertext
      """
      in_tower = ct_data[:, -1:, ..., :sizeQl]

      # Step 1: INTT on last element (negacyclic NTT handles psi internally)
      partCtCloneCoef = _intt(in_tower, ntt_ctx_in)

      res0 = jnp.zeros((batch, 1, r, c, sizeQlP), dtype=jnp.uint64)
      res1 = jnp.zeros((batch, 1, r, c, sizeQlP), dtype=jnp.uint64)

      # Keyswitch accumulation (dnum loop — small, stays unrolled)
      def _compute_parts_ct_ext(p):
        input_for_bconv = partCtCloneCoef[..., select_tower_index[p]]
        # BConv output fed into NTT must be u32 (NTT's BAT einsum bitcasts
        # u32 -> 4-byte lanes). The eager path casts back to u32 via
        # Polynomial.to_ntt_form; the pure path must do so explicitly.
        partCtCloneEval = bconv_inst.basis_change_bat(
            input_for_bconv, control_index=p).astype(jnp.uint32)
        ntted = _ntt(partCtCloneEval, ntt_ctx_parts[p])
        partsCtExt = jnp.concatenate(
            [in_tower[..., select_tower_index[p]], ntted], axis=-1)
        return partsCtExt[..., restore_indices[p]]

      if dnum > 0:
        next_ext = _compute_parts_ct_ext(0)
        for part in range(dnum - 1):
          cur_ext = next_ext
          next_ext = _compute_parts_ct_ext(part + 1)
          prod_b = ff_ctx_full.modular_reduction(
              cur_ext * eval_b[part]).astype(jnp.uint64)
          prod_a = ff_ctx_full.modular_reduction(
              cur_ext * eval_a[part]).astype(jnp.uint64)
          res0 = res0 + prod_b
          res1 = res1 + prod_a
        prod_b = ff_ctx_full.modular_reduction(
            next_ext * eval_b[dnum - 1]).astype(jnp.uint64)
        prod_a = ff_ctx_full.modular_reduction(
            next_ext * eval_a[dnum - 1]).astype(jnp.uint64)
        res0 = res0 + prod_b
        res1 = res1 + prod_a

      # Steps 2-5: Sequential component processing
      base0 = ct_data[:, 0:1, ..., :sizeQl]
      final_components = []

      for i, comp in enumerate([res0, res1]):
        # Step 2: Modulus reduction
        reduced = _mod_reduce(comp, ff_ctx_full).astype(jnp.uint32)

        # Step 3: Approximation modulus switch
        p_part = reduced[..., sizeQl:]
        p_coeffs = _intt(p_part, ntt_ctx_approx)
        ql_from_p = bconv_inst.basis_change_bat(
            p_coeffs, control_index=dnum)

        ql_ntt = _ntt(ql_from_p.astype(jnp.uint32), ntt_ctx_in)

        ql_part = reduced[..., :sizeQl]
        diff = ql_part - ql_ntt
        sub = jnp.where(ql_part < ql_ntt, diff + original_moduli, diff)

        # Step 4: modmul(PInvModq)
        res_i = _modmul(sub, PInvModq, ff_ctx_in)

        if i == 0:
          res_i = res_i + base0
          res_i = jnp.where(res_i >= original_moduli,
                            res_i - original_moduli, res_i)

        # Step 5: Coefficient permutation
        res_i_flat = res_i.reshape(batch, ring_dim, sizeQl)
        final_components.append(jnp.take(res_i_flat, coef_map, axis=1))

      return jnp.stack(final_components, axis=1)

    return rotate_fn

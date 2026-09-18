import jax
import jax.numpy as jnp

import bconv
import finite_field
import polynomial
import rescale
import util


BarrettContext = finite_field.BarrettContext
Polynomial = polynomial.Polynomial
BConvBarrett = bconv.BConvBarrett
HERescale = rescale.HERescale


jax.config.update("jax_enable_x64", True)


class HEMul:
  """HEMul class."""

  def __init__(
      self,
      batch,
      r,
      c,
      dnum,
      num_eval_mult,
      original_moduli,
      extend_moduli,
      composite_degree=1,
  ):
    self.batch = batch
    self.r = r
    self.c = c
    self.dnum = dnum
    self.num_eval_mult = num_eval_mult
    self.original_moduli = original_moduli
    self.composite_degree = composite_degree
    self.extend_moduli = extend_moduli
    self.last_tower_moduli = original_moduli[-1]
    self.evalkey_a_vector = jnp.zeros((dnum, 0), dtype=jnp.uint64)
    self.evalkey_b_vector = jnp.zeros((dnum, 0), dtype=jnp.uint64)
    self.ks_evk_a_precomp = jnp.zeros((dnum, 0), dtype=jnp.uint64)
    self.ks_evk_b_precomp = jnp.zeros((dnum, 0), dtype=jnp.uint64)

  def control_gen(
      self,
      degree_layout=None,
      composite_degree=None,
      skip_rescale=False,
      perf_test=False,
      keygen_sizeQ=None,
  ):
    """Generate control parameters and precompute values for the HEMul operation.

    Args:
      degree_layout: The layout of the polynomial degrees. Defaults to (r * c,).
      composite_degree: The degree of the composite modulus. Defaults to the
        composite degree of the HEMul object.
      skip_rescale: Whether to skip the rescaling step.
      perf_test: Whether to use random parameters instead of computing actual
        roots of unity, useful for performance testing.
      keygen_sizeQ: The number of Q towers that were used during key generation.
        When running at a reduced level, the eval key was generated with the
        full Q count, and its partition alpha = ceil(keygen_sizeQ / dnum).
        The hemul's key-switch partition must match this alpha, otherwise the
        BConv tower groups misalign with the `P*s` signal in the eval key parts.
        If None, falls back to the old behavior (alpha based on drop_last count),
        which only works when keygen_sizeQ equals the current tower count.
    """
    if degree_layout is not None:
      self.degree_layout = degree_layout
    else:
      self.degree_layout = (self.r, self.c)

    if composite_degree is None:
      composite_degree = self.composite_degree
    else:
      assert composite_degree == self.composite_degree
    assert composite_degree >= 1, 'composite_degree must be at least 1'

    # Handle composite_degree=0 case (no rescaling)
    if skip_rescale:
      self.drop_last_extend_moduli = self.original_moduli + self.extend_moduli
      self.drop_last_moduli = self.original_moduli
      self.last_tower_moduli = []
    else:
      self.drop_last_extend_moduli = (
          self.original_moduli[:-composite_degree] + self.extend_moduli
      )
      self.drop_last_moduli = self.original_moduli[:-composite_degree]
      self.last_tower_moduli = self.original_moduli[
          -composite_degree:
      ]  # List of last moduli for composite

    self.perf_test = perf_test
    self.skip_rescale = skip_rescale

    # ==========================================================================
    # 0. Configuration Derivation
    # ==========================================================================
    ring_dim = self.r * self.c
    original_moduli = self.original_moduli
    extend_moduli = self.extend_moduli

    sizeQ_in = len(original_moduli)
    overall_sizeQ_in = sizeQ_in
    overall_sizeP_in = len(extend_moduli)
    self.overall_sizeQ_in, self.overall_sizeP_in = (
        overall_sizeQ_in,
        overall_sizeP_in,
    )
    overall_sizeQ_in_no_last = (
        sizeQ_in if self.skip_rescale else sizeQ_in - self.composite_degree
    )

    # ==========================================================================
    # 1. Rescale Control Generation (via HERescale)
    # ==========================================================================
    if not skip_rescale:
      self._he_rescale = HERescale(
          batch=self.batch,
          num_elements=4,
          moduli=self.original_moduli,
          r=self.r,
          c=self.c,
          degree_layout=self.degree_layout,
      )
      self._he_rescale.control_gen(
          composite_degree=composite_degree, perf_test=perf_test
      )

    # ==========================================================================
    # 2. Instantiate Polynomial Objects
    # ==========================================================================
    ct_num_moduli = (
        overall_sizeQ_in
        if skip_rescale
        else overall_sizeQ_in - composite_degree
    )
    ct_shapes = {
        'batch': self.batch,
        'num_elements': 4,
        'degree': self.r * self.c,
        'num_moduli': ct_num_moduli,
        'precision': 32,
        'degree_layout': self.degree_layout,
    }
    self.ct_obj = Polynomial(
        ct_shapes,
        parameters={
            'moduli': self.drop_last_moduli,
            'finite_field_context': BarrettContext,
            'r': self.r,
            'c': self.c,
        },
    )

    # idx_cur_last_tower is the number of Q moduli
    # after dropping composite_degree limbs
    idx_cur_last_tower = overall_sizeQ_in_no_last

    ct_extend_shapes = {
        'batch': self.batch,
        'num_elements': 2,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': overall_sizeP_in,
        'degree_layout': self.degree_layout,
    }
    self.ct_extend = Polynomial(
        ct_extend_shapes, parameters={'moduli': self.extend_moduli}
    )

    # ==========================================================================
    # 3. Parameter Generation
    # ==========================================================================
    # NOTE: psi/inv_psi precomputation removed — the negacyclic NTT now
    # handles psi pre-multiply (in ntt()) and inv_psi post-multiply (in intt())
    # internally, so no external psi manipulation is needed.

    self.bconv = BConvBarrett(self.drop_last_extend_moduli)
    control_indices_list = []
    rotate_indices = list(range(idx_cur_last_tower))
    extend_indices = list(
        range(idx_cur_last_tower, idx_cur_last_tower + overall_sizeP_in)
    )
    control_indices_list.append((extend_indices, rotate_indices))

    current_moduli = self.extend_moduli
    target_moduli = [
        item for item in self.drop_last_moduli if item not in current_moduli
    ]
    P = 1
    for moduli in current_moduli:
      P *= moduli
    PInvModq_approx_down = [util.modinv(P, q) for q in target_moduli]
    self.PInvModq = jnp.asarray(PInvModq_approx_down, dtype=jnp.uint32).reshape(
        idx_cur_last_tower
    )

    # ==========================================================================
    # 4. Key Switch Control Generation
    # ==========================================================================
    drop_last_extend_moduli = self.drop_last_extend_moduli
    sizeQ_drop_last = len(self.drop_last_moduli)
    # Use keygen_sizeQ if provided so alpha matches the eval key's partition.
    # Falls back to the local drop_last count for backward compatibility.
    if keygen_sizeQ is not None:
        alpha = (keygen_sizeQ + self.dnum - 1) // self.dnum
    else:
        alpha = (sizeQ_drop_last + self.dnum - 1) // self.dnum
    self.ks_alpha = alpha
    self.ks_numPartQl = (sizeQ_drop_last + alpha - 1) // alpha

    original_moduli_extract_index = []
    for i in range(sizeQ_drop_last):
      if i % alpha == 0:
        original_moduli_extract_index.append([i])
      else:
        original_moduli_extract_index[-1].append(i)

    ks_select_tower_index_overall = []
    ks_non_select_tower_index_overall = []
    ks_restore_indices = []
    for part in range(self.ks_numPartQl):
      select_tower_overall_index = original_moduli_extract_index[part]
      non_select_tower_overall_index = [
          i
          for i in range(len(drop_last_extend_moduli))
          if i not in select_tower_overall_index
      ]
      concat_order = select_tower_overall_index + non_select_tower_overall_index
      restore_index = [0] * len(concat_order)
      for pos, val in enumerate(concat_order):
        restore_index[val] = pos
      ks_select_tower_index_overall.append(
          jnp.array(select_tower_overall_index, jnp.uint16)
      )
      ks_non_select_tower_index_overall.append(
          jnp.array(non_select_tower_overall_index, jnp.uint16)
      )
      ks_restore_indices.append(jnp.array(restore_index, jnp.uint16))

    self.ks_select_tower_index_overall = ks_select_tower_index_overall
    self.ks_non_select_tower_index_overall = ks_non_select_tower_index_overall
    self.ks_restore_indices = ks_restore_indices

    # Append KS BConv indices after the approx mod down index
    self.ks_control_start_idx = len(control_indices_list)
    for part in range(self.ks_numPartQl):
      control_indices_list.append((
          ks_select_tower_index_overall[part].tolist(),
          ks_non_select_tower_index_overall[part].tolist(),
      ))
    self.bconv.control_gen(control_indices_list, perf_test=perf_test)

    # KS part Polynomials
    ct_shapes_common = {
        'batch': self.batch,
        'num_elements': 2,
        'degree': ring_dim,
        'precision': 32,
        'degree_layout': self.degree_layout,
    }
    ct_params_common = {
        'r': self.r,
        'c': self.c,
        'finite_field_context': BarrettContext,
    }
    self.ks_ct_parts = []
    for part in range(self.ks_numPartQl):
      target_indices = ks_non_select_tower_index_overall[part].tolist()
      target_moduli = [drop_last_extend_moduli[i] for i in target_indices]
      shapes_part = ct_shapes_common.copy()
      shapes_part['num_moduli'] = len(target_moduli)
      params_part = ct_params_common.copy()
      params_part['moduli'] = target_moduli
      self.ks_ct_parts.append(Polynomial(shapes_part, params_part))

    # CT for KS result (drop_last + extend moduli)
    shapes_dle = ct_shapes_common.copy()
    shapes_dle['num_moduli'] = len(drop_last_extend_moduli)
    params_dle = ct_params_common.copy()
    params_dle['moduli'] = drop_last_extend_moduli
    self.ks_ct_drop_last_extend = Polynomial(shapes_dle, params_dle)

    # ==========================================================================
    # 5. Parameter Reshape
    # ==========================================================================
    self.drop_last_moduli_arr = jnp.array(
        self.drop_last_moduli, jnp.uint32
    ).reshape(1, 1, 1, -1)

    self.drop_last_extend_moduli_arr = jnp.array(
        self.drop_last_extend_moduli, jnp.uint32
    )
    self.q_correction = self.drop_last_extend_moduli_arr[
        :idx_cur_last_tower
    ].reshape(1, 1, 1, -1)
    self.post_rescale_shape = (
        self.batch,
        4,
        *self.degree_layout,
        len(self.drop_last_moduli),
    )

  def setup_relinearization(self, evalkey_a_vector, evalkey_b_vector):
    # Eval key preprocessing only (control gen moved to control_gen())
    idx_cur_last_tower = len(self.drop_last_moduli)
    overall_sizeP = len(self.extend_moduli)
    self.ks_evk_a_precomp = jnp.concatenate(
        [
            evalkey_a_vector[..., :idx_cur_last_tower],
            evalkey_a_vector[..., -overall_sizeP:],
        ],
        axis=-1,
    ).reshape(-1, *self.degree_layout, len(self.drop_last_extend_moduli))
    self.ks_evk_b_precomp = jnp.concatenate(
        [
            evalkey_b_vector[..., :idx_cur_last_tower],
            evalkey_b_vector[..., -overall_sizeP:],
        ],
        axis=-1,
    ).reshape(-1, *self.degree_layout, len(self.drop_last_extend_moduli))

  def rescale(self, in_ciphertext_data):
    """Perform composite rescaling and set the result on self.ct_obj.

    Args:
        in_ciphertext_data: Raw ciphertext ndarray from the input Polynomial
          object.
    """
    if self.skip_rescale:
      self.ct_obj.set_batch_polynomial(
          in_ciphertext_data.reshape(self.post_rescale_shape)
      )
    else:
      rescaled = self._he_rescale.rescale(in_ciphertext_data)
      self.ct_obj.set_batch_polynomial(
          rescaled.reshape(self.post_rescale_shape)
      )

  def key_switch(self, last_ele_post_mult):
    """Perform key switching on the last element of a tensor product.

    Args:
        last_ele_post_mult: The third element (mul2) from ciphertext_mult, shape
          (batch, 1, *degree_layout, num_q).

    Returns:
        Key-switched ciphertext ndarray on the extended (Q+P) moduli basis.
    """
    self.ct_obj.set_batch_polynomial(last_ele_post_mult)
    ks_input_ntt = self.ct_obj.get_batch_polynomial()

    self.ct_obj.to_coeffs_form()
    partCtCloneCoef = self.ct_obj.get_batch_polynomial()

    ks_res0 = None
    ks_res1 = None
    for part in range(self.ks_numPartQl):
      select_idxs = self.ks_select_tower_index_overall[part]
      partCtCloneEval = self.bconv.basis_change_bat(
          jnp.take(partCtCloneCoef, select_idxs, axis=-1),
          control_index=self.ks_control_start_idx + part,
      ).astype(jnp.uint64)
      ct_part = self.ks_ct_parts[part]
      ct_part.polynomial = partCtCloneEval.astype(jnp.uint32)
      ct_part.to_ntt_form()
      partsCtCompl_multi_moduli = ct_part.polynomial

      partsCtExt_cur_part = jnp.concatenate(
          [
              jnp.take(ks_input_ntt, select_idxs, axis=-1),
              partsCtCompl_multi_moduli,
          ],
          axis=-1,
      )
      partsCtExt_cur_part = jnp.take(
          partsCtExt_cur_part, self.ks_restore_indices[part], axis=-1
      )

      # Reduce each product immediately to prevent Barrett overflow.
      # Without this, the accumulated sum `partsCtExt * evk` over `numPartQl`
      # parts can exceed 2^(2*modulus_bits), which the Barrett reduction
      # applied at the end (via mod_reduce) cannot handle correctly.
      ks_ff = self.ks_ct_drop_last_extend.ntt_ctx.ff_ctx
      prod_b = (
          partsCtExt_cur_part
          * self.ks_evk_b_precomp.astype(jnp.uint64)[part][None, None, :, :]
      )
      prod_a = (
          partsCtExt_cur_part
          * self.ks_evk_a_precomp.astype(jnp.uint64)[part][None, None, :, :]
      )
      prod_b = ks_ff.modular_reduction(prod_b).astype(jnp.uint64)
      prod_a = ks_ff.modular_reduction(prod_a).astype(jnp.uint64)
      if ks_res0 is None:
        ks_res0 = prod_b
        ks_res1 = prod_a
      else:
        ks_res0 = ks_res0 + prod_b
        ks_res1 = ks_res1 + prod_a

    ks_result = jnp.concatenate([ks_res0, ks_res1], axis=1)
    self.ks_ct_drop_last_extend.set_batch_polynomial(ks_result)
    self.ks_ct_drop_last_extend.mod_reduce()
    return self.ks_ct_drop_last_extend.get_batch_polynomial()

  def mul(self, in_ciphertexts):
    ct_3elem = self.hemul_no_relin(in_ciphertexts)
    return self.relinearize(ct_3elem)

  def hemul_no_relin(self, in_ciphertexts):
    """Perform rescale + tensor multiply WITHOUT relinearization.

    This is steps 1-2 of the full mul() pipeline.
    Use relinearize() on the output to complete the multiplication.

    mul() calls hemul_no_relin + relinearize as the combined API.

    Args:
      in_ciphertexts: The input ciphertexts.

    Returns:
      A 3-element jnp.ndarray (mul0, mul1, mul2) concatenated along
      axis=1. Shape: (batch, 3, *degree_layout, num_q_after_rescale).
    """
    # ---------- Step 1: Rescale ----------
    self.rescale(in_ciphertexts.polynomial)

    # ---------- Step 2: Tensor multiply ----------
    ct_post_mult, last_ele_post_mult = self.ct_obj.polynomial_mult()

    # Return 3-element concatenation: (mul0, mul1, mul2)
    return jnp.concatenate([ct_post_mult, last_ele_post_mult], axis=1)

  def hemul_no_relin_square(self, in_ciphertexts):
    """Squaring shortcut: rescale + (a*a, 2*a0*a1, a1*a1) instead of full mul.

    For a CKKS multiplicative depth-1 squaring (`a == b`), polynomial_mult's
    cross term `a0*b1 + a1*b0` simplifies to `2 * a0 * a1`, saving one
    modmul (3 → 2 in the tensor stage). The caller is expected to pass an
    `in_ciphertexts` whose 4 elements are `[a0, a1, a0, a1]` (same shape /
    layout as the standard `hemul_no_relin` input — `_FastHEMulPath.square`
    already builds this).

    The rescale step still runs on the 4-element shell because rescale
    operates on the full layout the cached parameter expects.

    Args:
      in_ciphertexts: 4-element Polynomial wrapping [a0, a1, a0, a1].

    Returns:
      Same shape as `hemul_no_relin`: (batch, 3, *degree_layout, num_q).
    """
    # ---------- Step 1: Rescale (unchanged) ----------
    self.rescale(in_ciphertexts.polynomial)

    # ---------- Step 2: Squaring shortcut tensor ----------
    ct_post_mult, last_ele_post_mult = self.ct_obj.polynomial_square()

    return jnp.concatenate([ct_post_mult, last_ele_post_mult], axis=1)

  def relinearize(self, ct_3elem):
    """Perform relinearization on a 3-element intermediate result.

    Args:
      ct_3elem: 3-element jnp.ndarray (batch, 3, *degree_layout, num_q),
        containing (mul0, mul1, mul2) where mul2 is the third element from
        tensor multiply.

    Returns:
      Polynomial with 2 elements after key switching + approx mod down + add.

    This is steps 3-6 of the full mul() pipeline.
    """
    overall_sizeQ_in, overall_sizeP_in = (
        self.overall_sizeQ_in,
        self.overall_sizeP_in,
    )
    idx_cur_last_tower = (
        overall_sizeQ_in
        if self.skip_rescale
        else overall_sizeQ_in - self.composite_degree
    )

    ct_post_mult = ct_3elem[:, :2, ...]
    last_ele_post_mult = ct_3elem[:, 2:3, ...]

    # ---------- Step 3 & 4: Key switch ----------
    keyswitch_core_res = self.key_switch(last_ele_post_mult)

    # ---------- Step 5: Approximate modulus down (via Polynomial) ----------
    result_ciphertext_list = []
    overall_moduli_jax = jnp.asarray(self.drop_last_moduli, dtype=jnp.uint32)
    approx_down_in_jax = jnp.asarray(keyswitch_core_res, dtype=jnp.uint32)

    self.ct_extend.set_batch_polynomial(
        approx_down_in_jax[
            ..., idx_cur_last_tower : (idx_cur_last_tower + overall_sizeP_in)
        ]
    )
    self.ct_extend.to_coeffs_form()
    reduced_approx_down = self.ct_extend.get_batch_polynomial()
    ct_new_basis_coef = self.bconv.basis_change_bat(
        reduced_approx_down, control_index=0
    ).astype(jnp.uint64)

    for element_index in range(ct_new_basis_coef.shape[1]):
      tower_new_basis_coef = ct_new_basis_coef[
          :, element_index : element_index + 1, ...
      ]
      self.ct_obj.set_batch_polynomial(tower_new_basis_coef.astype(jnp.uint32))
      self.ct_obj.to_ntt_form()
      tower_new_basis_jax = self.ct_obj.get_batch_polynomial()

      current_approx_down_in = approx_down_in_jax[
          :, element_index : element_index + 1, ..., :idx_cur_last_tower
      ]
      sub_result = jnp.where(
          current_approx_down_in < tower_new_basis_jax,
          current_approx_down_in + overall_moduli_jax - tower_new_basis_jax,
          current_approx_down_in - tower_new_basis_jax,
      )

      self.ct_obj.set_batch_polynomial(sub_result)
      self.ct_obj.modmul(self.PInvModq)
      reduced_elem_modq = self.ct_obj.get_batch_polynomial()

      result_ciphertext_list.append(reduced_elem_modq)

    approx_mod_down_custom = jnp.concatenate(result_ciphertext_list, axis=1)

    # ---------- Step 6: Add and return ----------
    result = ct_post_mult + approx_mod_down_custom
    val = jnp.where(
        result >= self.q_correction, result - self.q_correction, result
    )

    self.ct_obj.set_batch_polynomial(val)
    return self.ct_obj

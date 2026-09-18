import jax
import jax.numpy as jnp

import bconv
import finite_field
import polynomial
import rescale
import util


BarrettContext = finite_field.BarrettContext
Polynomial = polynomial.Polynomial
_HERescaleKernel = rescale._HERescaleKernel


jax.config.update("jax_enable_x64", True)


class _HEMulKernel:
  """Private homomorphic-multiplication kernel.

  hemul_no_relin matches OpenFHE EvalMultCore (tensor product only).
  The full mul pipeline performs its rescale explicitly before calling the
  tensor operation, preserving the existing CKKS scale convention.
  """

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
      finite_field_context=BarrettContext,
  ):
    self.batch = batch
    self.r = r
    self.c = c
    self.dnum = dnum
    self.num_eval_mult = num_eval_mult
    self.original_moduli = original_moduli
    self.composite_degree = composite_degree
    self.extend_moduli = extend_moduli
    self.ff_context_cls = finite_field_context
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
    """Generate controls and precompute values for multiplication.

    Args:
      degree_layout: The tiled polynomial layout. Defaults to ``(r, c)``.
      composite_degree: The degree of the composite modulus. Defaults to the
        composite degree of the private multiplication kernel.
      skip_rescale: Whether the full mul pipeline skips its explicit rescale.
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
    self.degree_layout = finite_field.canonical_degree_layout(
        self.r, self.c, degree_layout, '_HEMulKernel.control_gen'
    )

    if composite_degree is None:
      composite_degree = self.composite_degree
    elif composite_degree != self.composite_degree:
      raise ValueError(
          f'composite_degree={composite_degree} does not match the kernel '
          f'composite_degree={self.composite_degree}'
      )
    if composite_degree < 1:
      raise ValueError('composite_degree must be at least 1')

    self.perf_test = perf_test
    self.skip_rescale = skip_rescale

    # Determine whether to use split-rescale (cd >= 2 and rescale enabled).
    # Split-rescale: 1 prime dropped before multiply, (cd-1) primes after.
    # Split-rescale disabled: output scale grows by q per he_mul,
    # incompatible with CKKS chain convention. Using standard cd=2 rescale.
    # See Bug #28 in bts_bug_tracker.md for details.
    self._use_split_rescale = False

    if skip_rescale:
      self.drop_last_extend_moduli = self.original_moduli + self.extend_moduli
      self.drop_last_moduli = self.original_moduli
      self.last_tower_moduli = []
    elif self._use_split_rescale:
      # Split rescale: key-switch operates on (NQ-1) towers (after pre-rescale).
      # Post-rescale drops the remaining (cd-1) primes after relinearization.
      self.drop_last_moduli = self.original_moduli[:-1]
      self.drop_last_extend_moduli = (
          self.original_moduli[:-1] + self.extend_moduli
      )
      self.last_tower_moduli = [self.original_moduli[-1]]
      # Track total output moduli count (NQ - cd)
      self._output_moduli = self.original_moduli[:-composite_degree]
    else:
      # cd=1: single rescale before multiply (original behavior)
      self.drop_last_extend_moduli = (
          self.original_moduli[:-composite_degree] + self.extend_moduli
      )
      self.drop_last_moduli = self.original_moduli[:-composite_degree]
      self.last_tower_moduli = self.original_moduli[
          -composite_degree:
      ]

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
    if skip_rescale:
      overall_sizeQ_in_no_last = sizeQ_in
    elif self._use_split_rescale:
      # After pre-rescale: NQ-1 towers
      overall_sizeQ_in_no_last = sizeQ_in - 1
    else:
      overall_sizeQ_in_no_last = sizeQ_in - composite_degree

    # ==========================================================================
    # 1. Rescale control generation.
    # ==========================================================================
    if not skip_rescale:
      if self._use_split_rescale:
        # Pre-rescale: cd=1 on the 4-element input, drops q[-1]
        self._he_rescale_pre = _HERescaleKernel(
            batch=self.batch,
            num_elements=4,
            moduli=self.original_moduli,
            r=self.r,
            c=self.c,
            degree_layout=self.degree_layout,
            finite_field_context=self.ff_context_cls,
        )
        self._he_rescale_pre.control_gen(
            composite_degree=1, perf_test=perf_test
        )
        # Post-rescale: cd=(composite_degree-1) on the 2-element result,
        # operates on original_moduli[:-1] and drops (cd-1) more primes
        post_cd = composite_degree - 1
        self._he_rescale_post = _HERescaleKernel(
            batch=self.batch,
            num_elements=2,
            moduli=self.original_moduli[:-1],
            r=self.r,
            c=self.c,
            degree_layout=self.degree_layout,
            finite_field_context=self.ff_context_cls,
        )
        self._he_rescale_post.control_gen(
            composite_degree=post_cd, perf_test=perf_test
        )
        self._post_rescale_cd = post_cd
      else:
        # cd=1: single rescale before multiply
        self._he_rescale = _HERescaleKernel(
            batch=self.batch,
            num_elements=4,
            moduli=self.original_moduli,
            r=self.r,
            c=self.c,
            degree_layout=self.degree_layout,
            finite_field_context=self.ff_context_cls,
        )
        self._he_rescale.control_gen(
            composite_degree=composite_degree, perf_test=perf_test
        )

    # ==========================================================================
    # 2. Instantiate Polynomial Objects
    # ==========================================================================
    # ct_num_moduli is the tower count used during tensor multiply and key-switch
    if skip_rescale:
      ct_num_moduli = overall_sizeQ_in
    elif self._use_split_rescale:
      ct_num_moduli = overall_sizeQ_in - 1  # after pre-rescale
    else:
      ct_num_moduli = overall_sizeQ_in - composite_degree

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
            'finite_field_context': self.ff_context_cls,
            'r': self.r,
            'c': self.c,
        },
    )
    ct_single_shapes = dict(ct_shapes)
    ct_single_shapes['num_elements'] = 1
    self.ct_single = Polynomial(
        ct_single_shapes,
        parameters={
            'moduli': self.drop_last_moduli,
            'finite_field_context': self.ff_context_cls,
            'r': self.r,
            'c': self.c,
        },
    )

    # Pre-build the 3-element hemul_no_relin output container once (compile
    # time); the per-call path only sets its .polynomial. ntt_ctx = ff_ctx keeps
    # it mod-reduce-only (relinearize never NTTs it).
    self.ct_no_relin_out = Polynomial(
        {
            'batch': self.batch,
            'num_elements': 3,
            'degree': self.r * self.c,
            'num_moduli': len(self.drop_last_moduli),
            'precision': 32,
            'degree_layout': self.degree_layout,
        },
        parameters={
            'moduli': self.drop_last_moduli,
            'ntt_ctx': self.ff_context_cls(moduli=self.drop_last_moduli),
        },
    )

    # idx_cur_last_tower is the number of Q moduli
    # after dropping limbs (1 for split, cd for non-split)
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
        ct_extend_shapes,
        parameters={
            'moduli': self.extend_moduli,
            'finite_field_context': self.ff_context_cls,
        },
    )

    # ==========================================================================
    # 3. Parameter Generation
    # ==========================================================================
    # NOTE: psi/inv_psi precomputation removed — the negacyclic NTT now
    # handles psi pre-multiply (in ntt()) and inv_psi post-multiply (in intt())
    # internally, so no external psi manipulation is needed.

    # BConv backend selected from the injected reduction algorithm (see bconv.make_bconv / BConvMontgomery).
    self.bconv = bconv.make_bconv(
        self.ff_context_cls, self.drop_last_extend_moduli
    )
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
    # Encode into computation format so modmul(PInvModq) keeps ciphertexts in
    # the op's representation (identity for Barrett; Montgomery: PInv * R).
    self.PInvModq = (
        self.ff_context_cls(moduli=target_moduli)
        .to_computation_format(self.PInvModq.astype(jnp.uint64))
        .astype(jnp.uint32)
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
        'finite_field_context': self.ff_context_cls,
    }
    self.ks_ct_parts = []
    for part in range(self.ks_numPartQl):
      target_indices = ks_non_select_tower_index_overall[part].tolist()
      target_moduli = [drop_last_extend_moduli[i] for i in target_indices]
      shapes_part = ct_shapes_common.copy()
      shapes_part['num_elements'] = 1
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
    # Keys arrive standard-form; encode to computation format so the key-switch product stays in the op's representation (Montgomery: evk*R).
    evk_ctx = self.ff_context_cls(moduli=self.drop_last_extend_moduli)
    self.ks_evk_a_precomp = evk_ctx.to_computation_format(
        self.ks_evk_a_precomp.astype(jnp.uint64)
    ).astype(jnp.uint64)
    self.ks_evk_b_precomp = evk_ctx.to_computation_format(
        self.ks_evk_b_precomp.astype(jnp.uint64)
    ).astype(jnp.uint64)

  def _rescale_array_unchecked(self, in_ciphertext_data):
    """Loads the multiply input after its public boundary was validated."""
    if self.skip_rescale:
      self.ct_obj.set_batch_polynomial(
          in_ciphertext_data
      )
    elif self._use_split_rescale:
      rescaled = self._he_rescale_pre._rescale_array_unchecked(
          in_ciphertext_data
      )
      self.ct_obj.set_batch_polynomial(
          rescaled
      )
    else:
      rescaled = self._he_rescale._rescale_array_unchecked(in_ciphertext_data)
      self.ct_obj.set_batch_polynomial(
          rescaled
      )

  def _key_switch_array(self, last_ele_post_mult):
    """Perform key switching on the last element of a tensor product.

    Args:
        last_ele_post_mult: The third element (mul2) from ciphertext_mult, shape
          (batch, 1, *degree_layout, num_q).

    Returns:
        Key-switched ciphertext ndarray on the extended (Q+P) moduli basis.
    """
    self.ct_single.set_batch_polynomial(last_ele_post_mult)
    ks_input_ntt = self.ct_single.get_batch_polynomial()

    self.ct_single.to_coeffs_form()
    partCtCloneCoef = self.ct_single.get_batch_polynomial()

    ks_ff = self.ks_ct_drop_last_extend.ntt_ctx.ff_ctx
    ks_res0 = None
    ks_res1 = None
    for part in range(self.ks_numPartQl):
      select_idxs = self.ks_select_tower_index_overall[part]
      partCtCloneEval = self.bconv.basis_change(
          jnp.take(partCtCloneCoef, select_idxs, axis=-1),
          control_index=self.ks_control_start_idx + part,
      ).astype(jnp.uint64)
      ct_part = self.ks_ct_parts[part]
      ct_part.replace_payload(partCtCloneEval.astype(jnp.uint32))
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
    # Canonicalize the sum of ks_numPartQl reduced (possibly lazy) products (see ff_ctx.strictify_after_accumulation).
    ks_result = ks_ff.strictify_after_accumulation(
        ks_result, self.ks_numPartQl
    )
    self.ks_ct_drop_last_extend.set_batch_polynomial(ks_result)
    return self.ks_ct_drop_last_extend.get_batch_polynomial()

  def mul(self, in_ciphertexts):
    """Explicitly rescale, tensor multiply, and relinearize."""
    finite_field.check_ct_operand(
        self.ff_context_cls,
        self.overall_sizeQ_in,
        in_ciphertexts,
        '_HEMulKernel.mul',
        batch=self.batch,
        num_elements=4,
        degree_layout=self.degree_layout,
        moduli=self.original_moduli,
    )
    return self._mul_array_to_polynomial_unchecked(in_ciphertexts.polynomial)

  def _mul_array_to_polynomial_unchecked(self, in_ciphertext_data):
    """Private full multiply kernel after the caller validates its boundary."""
    self._rescale_array_unchecked(in_ciphertext_data)
    ct_3elem = self._tensor_multiply()
    return self._relinearize_array_to_polynomial(ct_3elem.polynomial)

  def hemul_no_relin(self, in_ciphertexts):
    """Perform a tensor multiply without relinearization or rescaling.

    This matches OpenFHE EvalMultCore: the output remains on the same modulus
    towers as the inputs.
    Use relinearize() on the output to complete the multiplication.

    Args:
      in_ciphertexts: Inputs already at this kernel's configured
        tensor/key-switch modulus level. Use control_gen(skip_rescale=True)
        to operate on the unchanged full-Q input level.

    Returns:
      Polynomial (3-element: mul0, mul1, mul2) at the input towers.
    """
    finite_field.check_ct_operand(
        self.ff_context_cls,
        len(self.drop_last_moduli),
        in_ciphertexts,
        '_HEMulKernel.hemul_no_relin',
        batch=self.batch,
        num_elements=4,
        degree_layout=self.degree_layout,
        moduli=self.drop_last_moduli,
    )

    return self._hemul_no_relin_array_to_polynomial_unchecked(
        in_ciphertexts.polynomial
    )

  def _hemul_no_relin_array_to_polynomial_unchecked(
      self, in_ciphertext_data
  ):
    self.ct_obj.set_batch_polynomial(in_ciphertext_data)
    return self._tensor_multiply()

  def _hemul_no_relin_array(self, in_ciphertext_data):
    """Privately tensor-multiply a canonical rank-5 raw payload."""
    finite_field.check_rank5_array(
        in_ciphertext_data,
        '_HEMulKernel._hemul_no_relin_array',
        batch=self.batch,
        num_elements=4,
        degree_layout=self.degree_layout,
        num_moduli=len(self.drop_last_moduli),
    )
    return self._hemul_no_relin_array_to_polynomial_unchecked(
        in_ciphertext_data
    ).polynomial

  def _tensor_multiply(self):
    """Tensor multiply the 4-element value currently loaded in ct_obj."""
    # ---------- Tensor multiply ----------
    ct_post_mult, last_ele_post_mult = self.ct_obj._polynomial_mult_array()

    # Reuse the compile-time output context while returning a detached wrapper.
    ct_3elem = jnp.concatenate([ct_post_mult, last_ele_post_mult], axis=1)
    return self.ct_no_relin_out._clone_with_payload(
        ct_3elem.astype(jnp.uint32)
    )

  def relinearize(self, ct_3elem: Polynomial) -> Polynomial:
    """Perform relinearization on a 3-element intermediate result.

    Args:
      ct_3elem: 3-element Polynomial containing (mul0, mul1, mul2).

    Returns:
      Polynomial with 2 elements after key switching + approx mod down + add.
      For split rescale (cd >= 2): also applies post-rescale to drop the
      remaining (cd-1) primes.

    This is steps 3-6 of the full mul() pipeline.
    """
    finite_field.check_ct_operand(
        self.ff_context_cls,
        len(self.drop_last_moduli),
        ct_3elem,
        '_HEMulKernel.relinearize',
        batch=self.batch,
        num_elements=3,
        degree_layout=self.degree_layout,
        moduli=self.drop_last_moduli,
    )
    return self._relinearize_array_to_polynomial(ct_3elem.polynomial)

  def _relinearize_array(self, ct_3elem_data):
    """Privately relinearize a canonical rank-5 raw payload."""
    finite_field.check_rank5_array(
        ct_3elem_data,
        '_HEMulKernel._relinearize_array',
        batch=self.batch,
        num_elements=3,
        degree_layout=self.degree_layout,
        num_moduli=len(self.drop_last_moduli),
    )
    return self._relinearize_array_to_polynomial(ct_3elem_data).polynomial

  def _relinearize_array_to_polynomial(self, arr):
    """Internal array implementation returning a validated Polynomial."""
    overall_sizeQ_in, overall_sizeP_in = (
        self.overall_sizeQ_in,
        self.overall_sizeP_in,
    )
    if self.skip_rescale:
      idx_cur_last_tower = overall_sizeQ_in
    elif self._use_split_rescale:
      idx_cur_last_tower = overall_sizeQ_in - 1
    else:
      idx_cur_last_tower = overall_sizeQ_in - self.composite_degree

    ct_post_mult = arr[:, :2, ...]
    last_ele_post_mult = arr[:, 2:3, ...]

    # ---------- Step 3 & 4: Key switch ----------
    keyswitch_core_res = self._key_switch_array(last_ele_post_mult)

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
    ct_new_basis_coef = self.bconv.basis_change(
        reduced_approx_down, control_index=0
    ).astype(jnp.uint64)

    for element_index in range(ct_new_basis_coef.shape[1]):
      tower_new_basis_coef = ct_new_basis_coef[
          :, element_index : element_index + 1, ...
      ]
      self.ct_single.set_batch_polynomial(
          tower_new_basis_coef.astype(jnp.uint32)
      )
      self.ct_single.to_ntt_form()
      tower_new_basis_jax = self.ct_single.get_batch_polynomial()

      # NTT output may be lazy (Montgomery, [0, 2q)); the modular subtraction
      # below assumes both operands strict in [0, q).  Identity for Barrett.
      tower_new_basis_jax = self.ct_single.ntt_ctx.ff_ctx.strictify(
          tower_new_basis_jax
      )

      current_approx_down_in = approx_down_in_jax[
          :, element_index : element_index + 1, ..., :idx_cur_last_tower
      ]
      sub_result = jnp.where(
          current_approx_down_in < tower_new_basis_jax,
          current_approx_down_in + overall_moduli_jax - tower_new_basis_jax,
          current_approx_down_in - tower_new_basis_jax,
      )

      self.ct_single.set_batch_polynomial(sub_result)
      self.ct_single.replace_payload(
          self.ct_single.ntt_ctx.ff_ctx.modular_reduction(
              self.ct_single.to_array().astype(jnp.uint64)
              * self.PInvModq.astype(jnp.uint64)
          ).astype(self.ct_single.modulus_dtype)
      )
      reduced_elem_modq = self.ct_single.get_batch_polynomial()
      # modmul's reduction may be lazy (Montgomery); strictify for the final
      # modular add.  Identity for Barrett.
      reduced_elem_modq = self.ct_single.ntt_ctx.ff_ctx.strictify(
          reduced_elem_modq
      )

      result_ciphertext_list.append(reduced_elem_modq)

    approx_mod_down_custom = jnp.concatenate(result_ciphertext_list, axis=1)

    # ---------- Step 6: Add and return ----------
    # Strictify lazy tensor-multiply outputs so the single conditional subtract below suffices.
    ct_post_mult = self.ct_single.ntt_ctx.ff_ctx.strictify(ct_post_mult)
    result = ct_post_mult + approx_mod_down_custom
    val = jnp.where(
        result >= self.q_correction, result - self.q_correction, result
    )

    # ---------- Step 7: Post-rescale for split rescale (cd >= 2) ----------
    if self._use_split_rescale:
      # Apply (cd-1) rescale on the 2-element result to reach the
      # final output tower count (NQ - composite_degree).
      post_rescaled = self._he_rescale_post._rescale_array_unchecked(val)
      output_moduli = self._output_moduli
      return self.ct_obj._clone_with_payload(
          post_rescaled,
          num_elements=2,
          moduli=output_moduli,
          ntt_ctx=self._he_rescale_post.ct_work_list[-1].ntt_ctx,
      )

    # Return a detached wrapper while reusing the immutable context built
    # during control generation.  Bootstrap may lower the returned value's
    # level without corrupting this operator's reusable scratch wrapper.
    return self.ct_obj._clone_with_payload(val, num_elements=2)


__all__ = []

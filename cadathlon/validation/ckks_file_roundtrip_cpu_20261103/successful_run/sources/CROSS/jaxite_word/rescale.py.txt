import math

import jax
import jax.numpy as jnp

import finite_field
import polynomial
import util


BarrettContext = finite_field.BarrettContext
Polynomial = polynomial.Polynomial

jax.config.update("jax_enable_x64", True)


def centered_lift(last_coeffs, threshold, add_bias, lift_bctx, last_idx=None):
  """Balanced (centered) lift of a dropped tower into the remaining towers.

  Shared by the eager rescale and the matvec fused-rescale closures so they
  never diverge. Both steps are uint64-only (no int64 / integer modulo, which
  are emulated on TPU):
    (1) add_bias = ceil(q_last/q_j)*q_j - q_last shifts the balanced value to a
        NON-NEGATIVE representative congruent mod q_j for any modulus spread.
    (2) a plain Barrett reduce into [0, q_j) -- never Montgomery, the lift value
        is an ordinary integer, not a residue.
  last_idx selects the composite_degree>1 sub-context (matvec passes None).
  """
  sb = jnp.where(last_coeffs < threshold,
                 last_coeffs.astype(jnp.uint64),
                 last_coeffs.astype(jnp.uint64) + add_bias)
  bctx = lift_bctx if last_idx is None else lift_bctx.slice(last_idx)
  return bctx.modular_reduction(sb).astype(jnp.uint64)


class _HERescaleKernel:
  """Private rescale (modulus-switching) kernel for CKKS ciphertexts.

  Public callers use ``ctx.he_rescale``. ``_rescale_array`` is reserved for
  fused/JIT implementation regions.
  """

  def __init__(self, batch, num_elements, moduli, r, c, degree_layout=None,
               finite_field_context=BarrettContext):
    """Initialize the private rescale kernel.

    Args:
      batch: The batch size of the ciphertexts.
      num_elements: Number of ciphertext elements (typically 2 or 4).
      moduli: List of RNS moduli (full set before rescaling).
      r: Ring dimension layout factor.
      c: Ring dimension layout factor (degree = r * c).
      degree_layout: Optional explicit degree layout tuple; defaults to (r, c).
      finite_field_context: Modular reduction context class (BarrettContext or
        MontgomeryContext). With MontgomeryContext, ciphertext data is expected
        and produced in Montgomery computation format (x * 2^32 mod q), and all
        internal modular reductions are Montgomery reductions.
    """
    self.batch = batch
    self.num_elements = num_elements
    self.moduli = list(moduli)
    self.num_moduli = len(moduli)
    self.r = r
    self.c = c
    self.degree = r * c
    self.ff_context_cls = finite_field_context
    self.degree_layout = finite_field.canonical_degree_layout(
        r, c, degree_layout, '_HERescaleKernel'
    )

  def control_gen(self, composite_degree=1, perf_test=False):
    """Generate control parameters for composite rescaling.

    Args:
        composite_degree: Number of moduli to drop per rescale invocation.
        perf_test: If True, skip expensive root-of-unity computation (random
          params).
    """
    if (not isinstance(composite_degree, int)
        or isinstance(composite_degree, bool)
        or composite_degree < 1):
      raise ValueError('composite_degree must be a positive integer')
    if self.num_moduli <= composite_degree:
      raise ValueError(
          f'Cannot drop {composite_degree} moduli from {self.num_moduli}-limb'
          ' ciphertext.'
      )

    self.composite_degree = composite_degree
    self.output_moduli = self.moduli[:-composite_degree]
    self.output_num_moduli = len(self.output_moduli)

    ring_dim = self.degree
    num_moduli_full = self.num_moduli
    current_moduli_full = self.moduli

    self.moduli_threshold_all = jnp.array(
        [(m + 1) // 2 for m in current_moduli_full], jnp.uint32
    )

    all_gammas, all_betas, all_lift_add = [], [], []
    lift_reduction_modes = []
    for iter_idx in range(composite_degree):
      iter_moduli = current_moduli_full[: num_moduli_full - iter_idx]
      n = len(iter_moduli)
      gammas, betas = util.gamma_beta_calculation(
          iter_moduli, perf_test=perf_test
      )
      # add_bias for the branch-free uint64 centered lift; formula and non-negativity argument in centered_lift().
      last_mod = iter_moduli[-1]
      lift_add = [
          (-(-last_mod // qj)) * qj - last_mod for qj in iter_moduli[: n - 1]
      ]
      # Select the cheapest exact canonicalization for the centered lift.
      # When q_last < 2*q_j, every selected upper-half coefficient satisfies
      # x + q_j >= q_last, so evaluating ``x + q_j - q_last`` in that order is
      # non-negative and already in [0, q_j).  Preserve the generic Barrett
      # path for arbitrary, wider-magnitude modulus chains.
      lift_multiples = [
          -(-last_mod // qj) for qj in iter_moduli[: n - 1]
      ]
      max_lift_multiple = max(lift_multiples)
      lift_reduction_modes.append(
          'direct' if max_lift_multiple <= 2 else 'barrett'
      )
      padded_lift_add = (
          jnp.zeros((num_moduli_full - 1,), jnp.uint64)
          .at[: n - 1]
          .set(jnp.asarray(lift_add, jnp.uint64))
      )
      # gammas are used in coefficient domain (between to_coeffs_form and
      # to_ntt_form). Since both now handle negacyclic conversion internally,
      # gammas are used without psi factors.
      padded_gammas = (
          jnp.zeros((ring_dim, num_moduli_full - 1), jnp.uint64)
          .at[:, : n - 1]
          .set(
              jnp.broadcast_to(
                  gammas[: n - 1].astype(jnp.uint64)[None, :],
                  (ring_dim, n - 1),
              )
          )
      )
      padded_betas = (
          jnp.zeros((num_moduli_full - 1,), jnp.uint64)
          .at[: n - 1]
          .set(betas[: n - 1])
      )
      all_gammas.append(padded_gammas)
      all_betas.append(padded_betas)
      all_lift_add.append(padded_lift_add)

    self.gammas_stacked = jnp.stack(all_gammas).reshape(
        composite_degree, 1, *self.degree_layout, num_moduli_full - 1
    )
    self.betas_stacked = jnp.stack(all_betas)
    self.lift_add_stacked = jnp.stack(all_lift_add)
    self._lift_reduction_modes = tuple(lift_reduction_modes)
    # Plain Barrett context for centered_lift's strict reduce (never Montgomery -- see centered_lift).
    self.lift_bctx = BarrettContext(moduli=current_moduli_full[:-1])

    # Barrett lift is exact only below 2^(2*ceil(log2 q_j)): enforce sb < q_last + q_j fits, for every (dropped, remaining) pair.
    lift_s = [
        2 * math.ceil(math.log2(int(q))) for q in current_moduli_full[:-1]
    ]
    for iter_idx in range(composite_degree):
      last_idx = num_moduli_full - 1 - iter_idx
      q_last = current_moduli_full[last_idx]
      for j in range(last_idx):
        q_j = current_moduli_full[j]
        if q_last + q_j >= (1 << lift_s[j]):
          raise ValueError(
              f"Rescale centered lift not exact: q_last ({q_last}) + "
              f"q_j ({q_j}) >= 2^{lift_s[j]}"
          )

    # Encode rescale constants for the injected reduction (no-ops for Barrett; zero padding stays zero):
    # gammas multiply STANDARD-form switched data -> encode_prescale_constant (Montgomery: gamma*R^2);
    # betas multiply computation-format data -> to_computation_format (Montgomery: beta*R).
    conv_ctx = self.ff_context_cls(moduli=current_moduli_full[:-1])
    self.gammas_stacked = conv_ctx.encode_prescale_constant(
        self.gammas_stacked
    ).astype(jnp.uint64)
    self.betas_stacked = conv_ctx.to_computation_format(
        self.betas_stacked.astype(jnp.uint64)
    ).astype(jnp.uint64)

    # ct_last: single-modulus CTs for each iteration's last tower
    ct_last_shapes = {
        'batch': self.batch,
        'num_elements': self.num_elements,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': 1,
        'degree_layout': self.degree_layout,
    }
    ct_last_params_base = {
        'finite_field_context': self.ff_context_cls,
        'r': self.r,
        'c': self.c,
    }
    self.ct_last_list = [
        Polynomial(
            ct_last_shapes,
            {
                **ct_last_params_base,
                'moduli': [current_moduli_full[num_moduli_full - 1 - i]],
            },
        )
        for i in range(composite_degree)
    ]

    # Working CTs for NTT at each iteration (progressively fewer moduli)
    self.ct_work_list = []
    for iter_idx in range(composite_degree):
      work_num_moduli = num_moduli_full - 1 - iter_idx
      work_moduli = current_moduli_full[:work_num_moduli]
      work_shapes = {
          'batch': self.batch,
          'num_elements': self.num_elements,
          'degree': ring_dim,
          'precision': 32,
          'num_moduli': work_num_moduli,
          'degree_layout': self.degree_layout,
      }
      work_params = {
          'moduli': work_moduli,
          'finite_field_context': self.ff_context_cls,
          'r': self.r,
          'c': self.c,
      }
      self.ct_work_list.append(Polynomial(work_shapes, work_params))

  def _rescale_array(self, in_ciphertext_data):
    """Privately perform composite rescaling on a raw ciphertext array.

    Args:
        in_ciphertext_data: Input ciphertext ndarray with shape (batch,
          num_elements, *degree_layout, num_moduli).

    Returns:
        Rescaled ndarray with (composite_degree) fewer moduli in the last dim.
    """
    finite_field.check_rank5_array(
        in_ciphertext_data,
        '_HERescaleKernel._rescale_array',
        batch=self.batch,
        num_elements=self.num_elements,
        degree_layout=self.degree_layout,
        num_moduli=self.num_moduli,
    )
    return self._rescale_array_unchecked(in_ciphertext_data)

  def _rescale_array_unchecked(self, in_ciphertext_data):
    """Internal rescale kernel after a Polynomial/array boundary check."""
    rescale_input = in_ciphertext_data
    for iter_idx in range(self.composite_degree):
      n = self.num_moduli - iter_idx
      last_idx = n - 1

      gammas = self.gammas_stacked[iter_idx, ..., :last_idx]
      betas = self.betas_stacked[iter_idx, :last_idx]
      threshold = self.moduli_threshold_all[last_idx]
      add_bias = self.lift_add_stacked[iter_idx, :last_idx]

      ct_last = self.ct_last_list[iter_idx]
      last_tower = rescale_input[..., -1:].astype(jnp.uint32)
      last_shape = last_tower.shape
      last_coeffs = ct_last.ntt_ctx.intt(
          last_tower.reshape(-1, self.r, self.c, 1)
      ).reshape(last_shape)

      # The lift compares coefficient VALUES against (q_last+1)/2, so the dropped
      # tower must be strict standard form (Montgomery: exact even for lazy inputs).
      last_coeffs = ct_last.ntt_ctx.ff_ctx.to_original_format(
          last_coeffs.astype(jnp.uint64)
      ).astype(jnp.uint32)

      # Centered lift (see centered_lift): sb < q_last + q_j < 2^32 (moduli < 2^31);
      # the strict lift keeps switched*gammas < q_j^2, inside the op reduction's valid input range.
      lift_mode = self._lift_reduction_modes[iter_idx]
      if lift_mode == 'barrett':
        switched = centered_lift(
            last_coeffs, threshold, add_bias, self.lift_bctx, last_idx
        )
      else:
        # Reorder the arithmetic relative to the historical
        # ``q_j - q_last + x`` expression: for the selected upper half,
        # ``x + q_j`` is at least q_last, so uint64 subtraction cannot wrap.
        lift_moduli = self.lift_bctx.moduli_reduction[:last_idx]
        dropped_modulus = jnp.uint64(self.moduli[last_idx])
        switched = jnp.where(
            last_coeffs < threshold,
            last_coeffs.astype(jnp.uint64),
            last_coeffs.astype(jnp.uint64)
            + lift_moduli
            - dropped_modulus,
        )
      twisted = switched * gammas

      ct_work = self.ct_work_list[iter_idx]
      work_shape = twisted.shape
      twisted_reduced = ct_work.ntt_ctx.ff_ctx.modular_reduction(
          twisted.astype(jnp.uint64)
      ).astype(jnp.uint32)
      ntt_result = ct_work.ntt_ctx.ntt(
          twisted_reduced.reshape(-1, self.r, self.c, last_idx)
      ).reshape(work_shape)

      # Finalize: canonicalize(main*beta + twisted_ntt) in computation format (see ff_ctx.reduce_scaled_sum).
      main_branch = rescale_input[..., :-1]
      w = ct_work.ntt_ctx.ff_ctx.reduce_scaled_sum(
          main_branch, betas, ntt_result
      )
      rescale_input = w.astype(jnp.uint32)

    return rescale_input

  def rescale(self, in_ciphertext: Polynomial) -> Polynomial:
    """Rescales a canonical Polynomial and returns a lower-level Polynomial."""
    finite_field.check_ct_operand(
        self.ff_context_cls,
        self.num_moduli,
        in_ciphertext,
        '_HERescaleKernel.rescale',
        batch=self.batch,
        num_elements=self.num_elements,
        degree_layout=self.degree_layout,
        moduli=self.moduli,
    )
    payload = self._rescale_array_unchecked(in_ciphertext.polynomial)
    return in_ciphertext._clone_with_payload(
        payload,
        moduli=self.output_moduli,
        ntt_ctx=self.ct_work_list[-1].ntt_ctx,
    )


__all__ = []

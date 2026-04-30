import jax
import jax.numpy as jnp

import finite_field
import polynomial
import util


BarrettContext = finite_field.BarrettContext
Polynomial = polynomial.Polynomial

jax.config.update("jax_enable_x64", True)


class HERescale:
  """Standalone rescale (modulus switching) operator for CKKS ciphertexts.

  Usage:
      he_rescale = HERescale(batch=1, num_elements=2, moduli=q_towers, r=4, c=4)
      he_rescale.control_gen(composite_degree=1)
      rescaled_data = he_rescale.rescale(in_ciphertext_data)
  """

  def __init__(self, batch, num_elements, moduli, r, c, degree_layout=None):
    """Initialize HERescale.

    Args:
      batch: The batch size of the ciphertexts.
      num_elements: Number of ciphertext elements (typically 2 or 4).
      moduli: List of RNS moduli (full set before rescaling).
      r: Ring dimension layout factor.
      c: Ring dimension layout factor (degree = r * c).
      degree_layout: Optional explicit degree layout tuple; defaults to (r, c).
    """
    self.batch = batch
    self.num_elements = num_elements
    self.moduli = list(moduli)
    self.num_moduli = len(moduli)
    self.r = r
    self.c = c
    self.degree = r * c
    if degree_layout is not None:
      self.degree_layout = degree_layout
    else:
      self.degree_layout = (r, c)

  def control_gen(self, composite_degree=1, perf_test=False):
    """Generate control parameters for composite rescaling.

    Args:
        composite_degree: Number of moduli to drop per rescale invocation.
        perf_test: If True, skip expensive root-of-unity computation (random
          params).
    """
    assert composite_degree >= 1, 'composite_degree must be at least 1'
    assert self.num_moduli > composite_degree, (
        f'Cannot drop {composite_degree} moduli from {self.num_moduli}-limb'
        ' ciphertext.'
    )

    self.composite_degree = composite_degree
    self.output_moduli = self.moduli[:-composite_degree]
    self.output_num_moduli = len(self.output_moduli)

    ring_dim = self.degree
    num_moduli_full = self.num_moduli
    current_moduli_full = self.moduli


    self.full_moduli_arr = jnp.array(current_moduli_full, dtype=jnp.uint32)
    self.moduli_threshold_all = jnp.array(
        [(m + 1) // 2 for m in current_moduli_full], jnp.uint32
    )

    all_gammas, all_betas = [], []
    for iter_idx in range(composite_degree):
      iter_moduli = current_moduli_full[: num_moduli_full - iter_idx]
      n = len(iter_moduli)
      gammas, betas = util.gamma_beta_calculation(
          iter_moduli, perf_test=perf_test
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

    self.gammas_stacked = jnp.stack(all_gammas).reshape(
        composite_degree, 1, *self.degree_layout, num_moduli_full - 1
    )
    self.betas_stacked = jnp.stack(all_betas)

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
        'finite_field_context': BarrettContext,
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
          'finite_field_context': BarrettContext,
          'r': self.r,
          'c': self.c,
      }
      self.ct_work_list.append(Polynomial(work_shapes, work_params))

  def rescale(self, in_ciphertext_data):
    """Perform composite rescaling.

    Args:
        in_ciphertext_data: Input ciphertext ndarray with shape (batch,
          num_elements, *degree_layout, num_moduli).

    Returns:
        Rescaled ndarray with (composite_degree) fewer moduli in the last dim.
    """
    rescale_input = in_ciphertext_data
    for iter_idx in range(self.composite_degree):
      n = self.full_moduli_arr.shape[0] - iter_idx
      last_idx = n - 1

      gammas = self.gammas_stacked[iter_idx, ..., :last_idx]
      betas = self.betas_stacked[iter_idx, :last_idx]
      threshold = self.moduli_threshold_all[last_idx]
      drop_moduli = self.full_moduli_arr[:last_idx].astype(jnp.uint32)
      last_modulus = self.full_moduli_arr[last_idx]

      ct_last = self.ct_last_list[iter_idx]
      ct_last.set_batch_polynomial(rescale_input[..., -1:].astype(jnp.uint32))
      ct_last.to_coeffs_form()
      last_coeffs = ct_last.get_batch_polynomial().astype(jnp.uint32)

      switched = jnp.where(
          last_coeffs < threshold,
          last_coeffs,
          jnp.array(drop_moduli, jnp.uint64) - last_modulus + last_coeffs,
      )
      twisted = switched.astype(jnp.uint64) * gammas

      ct_work = self.ct_work_list[iter_idx]
      ct_work.set_batch_polynomial(twisted)
      ct_work.mod_reduce()
      ct_work.to_ntt_form()
      ntt_result = ct_work.get_batch_polynomial()

      ct_work.set_batch_polynomial(rescale_input[..., :-1])
      ct_work.mul(betas)
      ct_work.add(ntt_result.astype(jnp.uint64))
      ct_work.mod_reduce()

      rescale_input = ct_work.get_batch_polynomial()

    return rescale_input

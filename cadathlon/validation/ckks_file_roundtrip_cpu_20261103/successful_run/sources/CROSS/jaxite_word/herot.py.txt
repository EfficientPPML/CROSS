import math

import jax
import jax.numpy as jnp
import util
import bconv
import finite_field
from polynomial import Polynomial

BarrettContext = finite_field.BarrettContext

# enable 64-bit computation in jax
jax.config.update("jax_enable_x64", True)


class _HERotKernel:
  """Private homomorphic-rotation kernel.

  Lifecycle:

  * Offline: ``__init__`` -> ``control_gen`` -> ``setup_rotation``.
  * Runtime: ``rotate`` or a validated ``_<step>_array`` boundary.
  * Runtime methods never create or replace cached control/key state.

  Naming follows the other HE kernels:

  * ``rotate`` is the complete ``Polynomial`` operation.
  * ``_<step>_array`` is a validated raw/JAX composition boundary.
  * ``_<verb>_<object>`` is an implementation-only helper. Such a helper may
    manipulate arrays, but it is not a supported composition boundary.
  * lifecycle methods such as ``control_gen`` and ``setup_rotation`` do not
    transform ciphertext payloads and therefore omit ``_array``.

  The ordinary runtime path is ``rotate`` -> ``_rotate_array`` -> key switch
  -> ModDown -> add c0 -> automorphism. The hoisted path decomposes c1 once
  with ``_decompose_array``, applies ``_hoisted_rotate_array`` per key, keeps
  accumulation in QP, and invokes ``_mod_down_array`` at group boundaries.
  """

  # ---------------------------------------------------------------------------
  # Offline construction and configuration.
  # ---------------------------------------------------------------------------

  def __init__(self, r, c, dnum, rotate_in_ciphertext_moduli, extend_moduli,
               finite_field_context=BarrettContext):
    self.r = r
    self.c = c
    self.dnum = dnum
    self.rotate_in_ciphertext_moduli = rotate_in_ciphertext_moduli
    self.extend_moduli = extend_moduli
    self.overall_moduli_init = (
        self.rotate_in_ciphertext_moduli + self.extend_moduli
    )
    self.ff_context_cls = finite_field_context
    # BConv backend selected from the injected reduction algorithm (see bconv.make_bconv).
    self.bconv = bconv.make_bconv(
        self.ff_context_cls, self.overall_moduli_init
    )
    self.evalkey_a_vector = jnp.zeros((dnum, 0), dtype=jnp.uint64)
    self.evalkey_b_vector = jnp.zeros((dnum, 0), dtype=jnp.uint64)
    self.coef_map = jnp.zeros((0,), dtype=jnp.int32)
    self._controls_ready = False
    self._rotation_ready = False

  def control_gen(
      self,
      batch=1,
      degree_layout=None,
      perf_test=False,
      keygen_sizeQ=None,
  ):
    """Generates control parameters and precomputes values for rotations.

    This method sets up various parameters and precomputations required for
    the rotation operations, including roots of unity, inverse roots of unity,
    modulus switching parameters, and indices for basis changes within the
    BConvBarrett instance. It also pre-allocates Polynomial objects.

    Args:
      batch: The batch size of the ciphertexts.
      degree_layout: The tiled polynomial layout. Defaults to ``(r, c)``.
      perf_test: If True, uses random parameters instead of computing actual
        roots of unity, useful for performance testing.
      keygen_sizeQ: Q-tower count used to generate the rotation key. When set,
        preserves the key's decomposition partition boundaries at lower levels.
    """
    # Invalidating first makes repeated setup safe: runtime methods cannot
    # observe controls or default rotation state from an earlier layout.
    self._controls_ready = False
    self._rotation_ready = False
    self.evalkey_a_vector = jnp.zeros((self.dnum, 0), dtype=jnp.uint64)
    self.evalkey_b_vector = jnp.zeros((self.dnum, 0), dtype=jnp.uint64)
    self.coef_map = jnp.zeros((0,), dtype=jnp.int32)
    degree_layout = finite_field.canonical_degree_layout(
        self.r, self.c, degree_layout, '_HERotKernel.control_gen'
    )
    self.degree_layout = degree_layout
    sizeQl_in = len(self.rotate_in_ciphertext_moduli)
    sizeQlP_in = len(self.extend_moduli) + sizeQl_in
    partition_size_q = keygen_sizeQ if keygen_sizeQ is not None else sizeQl_in
    alpha = (partition_size_q + self.dnum - 1) // self.dnum
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
    # Encode into computation format so modmul(PInvModq) keeps ciphertexts in
    # the op's representation (identity for Barrett; Montgomery: PInv * R).
    self.PInvModq = (
        self.ff_context_cls(moduli=target_moduli)
        .to_computation_format(self.PInvModq.astype(jnp.uint64))
        .astype(jnp.uint32)
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
    self.numPartQl = numPartQl

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
        parameters={
            'moduli': overall_moduli[:sizeQl_in],
            'finite_field_context': self.ff_context_cls,
        },
    )
    ct_single_shapes = dict(ct_in_shapes)
    ct_single_shapes['num_elements'] = 1
    self.ct_single = Polynomial(
        ct_single_shapes,
        parameters={
            'moduli': overall_moduli[:sizeQl_in],
            'finite_field_context': self.ff_context_cls,
        },
    )

    self.ct_parts = []
    for part in range(numPartQl):
      _target_indices_list = self.non_select_tower_index[part]
      _target_moduli_list = [overall_moduli[i] for i in _target_indices_list]
      _num_moduli_part = len(_target_moduli_list)
      ct_part_shapes = {
          'batch': batch,
          'num_elements': 1,
          'degree': ring_dim,
          'precision': 32,
          'num_moduli': _num_moduli_part,
          'degree_layout': degree_layout,
      }
      self.ct_parts.append(
          Polynomial(
              ct_part_shapes,
              parameters={
                  'moduli': _target_moduli_list,
                  'finite_field_context': self.ff_context_cls,
              },
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
        parameters={
            'moduli': self.overall_moduli,
            'finite_field_context': self.ff_context_cls,
        },
    )

    ct_approx_shapes = {
        'batch': batch,
        'num_elements': 1,
        'degree': ring_dim,
        'precision': 32,
        'num_moduli': sizeQlP_in - sizeQl_in,
        'degree_layout': degree_layout,
    }
    self.ct_approx = Polynomial(
        ct_approx_shapes,
        parameters={
            'moduli': self.extend_moduli,
            'finite_field_context': self.ff_context_cls,
        },
    )
    # Materialize the control arrays here, outside every JIT/runtime path.
    self._restore_indices_jax = [
        jnp.asarray(idx, dtype=jnp.uint16) for idx in self.restore_indices
    ]
    self._select_tower_index_jax = [
        jnp.asarray(idx, dtype=jnp.uint16) for idx in self.select_tower_index
    ]
    self._controls_ready = True

  def setup_rotation(self, evalkey_a_vector, evalkey_b_vector, coef_map):
    """Bind one rotation's default key and automorphism map offline.

    Explicit evaluation-key operands supplied later to runtime array methods
    must already use this kernel's computation representation. State returned
    by ``_HERotAtLevel._rotation_state`` satisfies that contract.
    """
    if not self._controls_ready:
      raise RuntimeError(
          'HERot.setup_rotation requires control_gen to run first.'
      )
    self._rotation_ready = False
    evalkey_a_vector = jnp.asarray(evalkey_a_vector, dtype=jnp.uint64)
    evalkey_b_vector = jnp.asarray(evalkey_b_vector, dtype=jnp.uint64)
    # Keys arrive standard-form; encode to computation format so the key-switch product stays in the op's representation (Montgomery: evk*R).
    evk_ctx = self.ff_context_cls(moduli=self.overall_moduli_init)
    self.evalkey_a_vector = evk_ctx.to_computation_format(
        evalkey_a_vector
    ).astype(jnp.uint32)
    self.evalkey_b_vector = evk_ctx.to_computation_format(
        evalkey_b_vector
    ).astype(jnp.uint32)
    self.coef_map = self._resolve_automorphism_map(coef_map)
    # Validate and retain the fully materialized computation-format defaults
    # now, so runtime/JIT methods remain read-only with respect to this object.
    self.evalkey_a_vector, self.evalkey_b_vector = (
        self._resolve_evaluation_keys(
            self.evalkey_a_vector, self.evalkey_b_vector
        )
    )
    self._rotation_ready = True

  # ---------------------------------------------------------------------------
  # Complete runtime operations.
  # ---------------------------------------------------------------------------

  def rotate(
      self,
      in_ciphertexts: Polynomial,
  ) -> Polynomial:
    """Rotate a ciphertext through the canonical HYBRID algorithm steps."""
    self._require_controls()
    finite_field.check_ct_operand(
        self.ff_context_cls,
        self.sizeQl,
        in_ciphertexts,
        '_HERotKernel.rotate',
        batch=self.batch,
        num_elements=2,
        degree_layout=(self.r, self.c),
        moduli=self.overall_moduli[:self.sizeQl],
    )
    return self.ct_in._clone_with_payload(
        self._rotate_array(in_ciphertexts.polynomial)
    )

  def _rotate_array(
      self, ct_data, eval_a=None, eval_b=None, coef_map=None
  ):
    """Rotate a canonical rank-5 payload through the HYBRID steps.

    Evaluation keys and the automorphism map are explicit runtime operands
    for fused scans. Explicit keys must already use the kernel's computation
    representation. Any omitted state comes from ``setup_rotation``.
    """
    self._require_controls()
    finite_field.check_rank5_array(
        ct_data,
        '_HERotKernel._rotate_array',
        batch=self.batch,
        num_elements=2,
        degree_layout=(self.r, self.c),
        num_moduli=self.sizeQl,
    )
    coef_map = self._resolve_automorphism_map(coef_map)
    # Preserve the generalized path for Montgomery and full-width Barrett
    # moduli. The common CKKS/HERot Barrett envelope (all q < 2^31) can use a
    # fused streaming implementation: uint32 add/sub is wrap-safe there and
    # keeping the two key-switch components separate avoids materializing a
    # full two-component QP temporary before ModDown.
    if self.ff_context_cls is BarrettContext and all(
        modulus < 1 << 31 for modulus in self.overall_moduli
    ):
      return self._rotate_barrett_fused(ct_data, eval_a, eval_b, coef_map)
    switched_qp = self._key_switch_array(ct_data, eval_a, eval_b)
    switched_q = self._mod_down_array(switched_qp)
    with_first = self._add_first_component_array(switched_q, ct_data)
    return self._automorphism_array(with_first, coef_map)

  def _rotate_barrett_fused(self, ct_data, eval_a, eval_b, coef_map):
    """Run the low-memory Barrett rotation kernel for sub-31-bit moduli."""
    eval_a, eval_b = self._resolve_evaluation_keys(eval_a, eval_b)
    in_tower, coeffs = self._prepare_decomposition_inputs(ct_data)
    ff_full = self.ct_full.ntt_ctx.ff_ctx

    res0 = jnp.zeros(
        (self.batch, 1, self.r, self.c, self.sizeQlP), dtype=jnp.uint64
    )
    res1 = jnp.zeros_like(res0)

    # Stream one digit at a time. Keeping the next digit explicit lets the TPU
    # compiler overlap its NTT with the current digit's key products, matching
    # the original fast HERot schedule without storing all QP digits.
    if self.numPartQl:
      next_digit = self._decompose_part(in_tower, coeffs, 0)
      for part in range(self.numPartQl - 1):
        digit = next_digit
        next_digit = self._decompose_part(in_tower, coeffs, part + 1)
        digit = digit.astype(jnp.uint64)
        res0 = res0 + ff_full.modular_reduction(
            digit * eval_b[part].astype(jnp.uint64)
        ).astype(jnp.uint64)
        res1 = res1 + ff_full.modular_reduction(
            digit * eval_a[part].astype(jnp.uint64)
        ).astype(jnp.uint64)

      next_digit = next_digit.astype(jnp.uint64)
      res0 = res0 + ff_full.modular_reduction(
          next_digit * eval_b[self.numPartQl - 1].astype(jnp.uint64)
      ).astype(jnp.uint64)
      res1 = res1 + ff_full.modular_reduction(
          next_digit * eval_a[self.numPartQl - 1].astype(jnp.uint64)
      ).astype(jnp.uint64)

    q_moduli = jnp.asarray(
        self.overall_moduli[:self.sizeQl], dtype=jnp.uint32
    ).reshape(1, 1, 1, 1, self.sizeQl)
    p_inv = jnp.asarray(self.PInvModq, dtype=jnp.uint64).reshape(
        1, 1, 1, 1, self.sizeQl
    )
    base0 = ct_data[:, 0:1, ..., :self.sizeQl]
    components = []

    # Process each component completely before starting the next one. This
    # bounds live QP data and avoids a bandwidth-heavy QP concatenate/slice.
    for component_index, accumulated in enumerate((res0, res1)):
      reduced = ff_full.modular_reduction(accumulated).astype(jnp.uint32)
      p_part = reduced[..., self.sizeQl:]
      p_shape = p_part.shape
      p_coeffs = self.ct_approx.ntt_ctx.intt(
          p_part.reshape(-1, self.r, self.c, p_shape[-1])
      ).reshape(p_shape)
      q_from_p = self.bconv.basis_change_bat(
          p_coeffs, control_index=self.numPartQl
      ).astype(jnp.uint32)
      q_shape = q_from_p.shape
      q_from_p_ntt = self.ct_in.ntt_ctx.ntt(
          q_from_p.reshape(-1, self.r, self.c, self.sizeQl)
      ).reshape(q_shape)

      q_part = reduced[..., :self.sizeQl]
      diff = q_part - q_from_p_ntt
      diff = jnp.where(
          q_part < q_from_p_ntt, diff + q_moduli, diff
      )
      result = self.ct_in.ntt_ctx.ff_ctx.modular_reduction(
          diff.astype(jnp.uint64) * p_inv
      ).astype(jnp.uint32)

      if component_index == 0:
        result = result + base0
        result = jnp.where(result >= q_moduli, result - q_moduli, result)

      flat = result.reshape(
          self.batch, self.r * self.c, self.sizeQl
      )
      permuted = jnp.take(flat, coef_map, axis=1)
      components.append(
          permuted.reshape(
              self.batch, 1, self.r, self.c, self.sizeQl
          )
      )

    return jnp.concatenate(components, axis=1)

  # ---------------------------------------------------------------------------
  # Runtime state helpers. These intentionally omit the ``_array`` suffix.
  # ---------------------------------------------------------------------------

  def _require_controls(self):
    """Reject runtime use before the offline control phase is complete."""
    if not self._controls_ready:
      raise RuntimeError(
          'HERot runtime operations require control_gen to run first.'
      )

  def _rotation_state(self):
    """Return bound computation-format keys and the automorphism map."""
    if not self._rotation_ready:
      raise RuntimeError(
          'HERot rotation state requires setup_rotation to run first.'
      )
    return self.evalkey_a_vector, self.evalkey_b_vector, self.coef_map

  def _resolve_automorphism_map(self, coef_map):
    """Return an explicit map or the default bound by setup_rotation."""
    if coef_map is None:
      if not self._rotation_ready:
        raise RuntimeError(
            'HERot default automorphism requires setup_rotation to run first.'
        )
      coef_map = self.coef_map
    coef_map = jnp.asarray(coef_map, dtype=jnp.int32)
    expected = (self.r * self.c,)
    if coef_map.shape != expected:
      raise ValueError(
          f'HERot automorphism map must have shape {expected}; got '
          f'{coef_map.shape}.'
      )
    return coef_map

  # ---------------------------------------------------------------------------
  # Runtime key-switch and hoisting boundaries, with local implementation
  # helpers kept beside the stage that owns them.
  # ---------------------------------------------------------------------------

  def _decompose_array(self, ct_data):
    """Hoist the HYBRID digit decomposition of a ciphertext's c1 element.

    The returned digits remain in the extended QlP evaluation basis.  They
    can be reused with every automorphism key at this level, matching
    OpenFHE's ``EvalFastRotationPrecompute``.
    """
    self._require_controls()
    finite_field.check_rank5_array(
        ct_data,
        'HERot._decompose_array',
        batch=self.batch,
        num_elements=2,
        degree_layout=(self.r, self.c),
        num_moduli=self.sizeQl,
    )

    in_tower, coeffs = self._prepare_decomposition_inputs(ct_data)
    digits = [
        self._decompose_part(in_tower, coeffs, part)
        for part in range(self.numPartQl)
    ]
    return jnp.stack(digits, axis=0).astype(jnp.uint32)

  def _prepare_decomposition_inputs(self, ct_data):
    """Prepare internal evaluation- and coefficient-form c1 inputs."""
    in_tower = ct_data[:, -1:, ..., :self.sizeQl]
    shape = in_tower.shape
    coeffs = self.ct_in.ntt_ctx.intt(
        in_tower.reshape(-1, self.r, self.c, self.sizeQl)
    ).reshape(shape)
    return in_tower, coeffs

  def _decompose_part(self, in_tower, coeffs, part):
    """Extend one internal HYBRID part; not a raw composition boundary."""
    selected = self._select_tower_index_jax[part]
    switched = self.bconv.basis_change_bat(
        coeffs[..., selected], control_index=part
    ).astype(jnp.uint32)
    part_ctx = self.ct_parts[part].ntt_ctx
    switched_shape = switched.shape
    switched_ntt = part_ctx.ntt(
        switched.reshape(
            -1, self.r, self.c, switched_shape[-1]
        )
    ).reshape(switched_shape)
    extended = jnp.concatenate(
        [in_tower[..., selected], switched_ntt], axis=-1
    )
    return extended[..., self._restore_indices_jax[part]]

  def _key_switch_core_array(self, digits, eval_a=None, eval_b=None):
    """Apply an automorphism key to hoisted digits without QP->Q ModDown."""
    self._require_controls()
    digits = jnp.asarray(digits, dtype=jnp.uint64)
    expected = (
        self.numPartQl, self.batch, 1, self.r, self.c, self.sizeQlP
    )
    if digits.shape != expected:
      raise ValueError(
          f'HERot._key_switch_core_array digits must have shape '
          f'{expected}; got {digits.shape}.'
      )
    eval_a, eval_b = self._resolve_evaluation_keys(eval_a, eval_b)
    return self._accumulate_key_switch_parts(
        (digits[part] for part in range(self.numPartQl)),
        eval_a,
        eval_b,
    )

  def _key_switch_array(self, ct_data, eval_a=None, eval_b=None):
    """Apply HYBRID key switching directly to ciphertext data, retaining QP.

    This is the streaming equivalent of
    ``_key_switch_core_array(_decompose_array(ct_data), eval_a, eval_b)``.
    It avoids materializing all hoisted digits when they will be used once.
    """
    self._require_controls()
    finite_field.check_rank5_array(
        ct_data,
        'HERot._key_switch_array',
        batch=self.batch,
        num_elements=2,
        degree_layout=(self.r, self.c),
        num_moduli=self.sizeQl,
    )
    eval_a, eval_b = self._resolve_evaluation_keys(eval_a, eval_b)
    in_tower, coeffs = self._prepare_decomposition_inputs(ct_data)
    parts = (
        self._decompose_part(in_tower, coeffs, part)
        for part in range(self.numPartQl)
    )
    return self._accumulate_key_switch_parts(parts, eval_a, eval_b)

  def _resolve_evaluation_keys(self, eval_a, eval_b):
    """Validate paired computation-format keys or return bound defaults."""
    if (eval_a is None) != (eval_b is None):
      raise ValueError(
          'HERot key switching requires eval_a and eval_b together.'
      )
    if eval_a is None:
      if not self._rotation_ready:
        raise RuntimeError(
            'HERot default key switching requires setup_rotation to run first.'
        )
      eval_a = self.evalkey_a_vector
      eval_b = self.evalkey_b_vector
    eval_a = jnp.asarray(eval_a)
    eval_b = jnp.asarray(eval_b)
    expected_key_tail = (self.r, self.c, self.sizeQlP)
    if eval_a.shape != eval_b.shape or \
       eval_a.ndim != 4 or eval_b.ndim != 4 or \
       tuple(eval_a.shape[1:]) != expected_key_tail or \
       tuple(eval_b.shape[1:]) != expected_key_tail:
      raise ValueError(
          'HERot key switching evaluation keys must have shape '
          f'(partitions, {self.r}, {self.c}, {self.sizeQlP}); got '
          f'{eval_a.shape} and {eval_b.shape}.'
      )
    if eval_a.shape[0] < self.numPartQl or eval_b.shape[0] < self.numPartQl:
      raise ValueError(
          'HERot key switching received fewer evaluation-key '
          f'partitions than the active {self.numPartQl} digits.'
      )
    return eval_a, eval_b

  def _accumulate_key_switch_parts(self, parts, eval_a, eval_b):
    """Reduce an internal digit iterable against one evaluation key."""
    ff_full = self.ct_full.ntt_ctx.ff_ctx
    res0 = jnp.zeros(
        (self.batch, 1, self.r, self.c, self.sizeQlP), dtype=jnp.uint64
    )
    res1 = jnp.zeros_like(res0)
    for part, digit in enumerate(parts):
      digit = jnp.asarray(digit, dtype=jnp.uint64)
      res0 = res0 + ff_full.modular_reduction(
          digit * eval_b[part].astype(jnp.uint64)
      ).astype(jnp.uint64)
      res1 = res1 + ff_full.modular_reduction(
          digit * eval_a[part].astype(jnp.uint64)
      ).astype(jnp.uint64)
    terms = max(self.numPartQl, 1)
    res0 = ff_full.strictify_after_accumulation(res0, terms)
    res1 = ff_full.strictify_after_accumulation(res1, terms)
    return jnp.concatenate([res0, res1], axis=1).astype(jnp.uint32)

  # ---------------------------------------------------------------------------
  # Runtime hoisted-rotation construction in the QP basis.
  # ---------------------------------------------------------------------------

  def _key_switch_extend_array(self, ct_data, include_first=True):
    """Embed a Q ciphertext into QP using OpenFHE's P-scaled convention.

    Q limbs contain ``P * c`` while P limbs are zero. With ``include_first``
    false, c0 is zeroed but all later components are still embedded.  This
    is OpenFHE ``KeySwitchExt`` (despite that routine not applying a key).
    """
    self._require_controls()
    finite_field.check_rank5_array(
        ct_data,
        'HERot._key_switch_extend_array',
        batch=self.batch,
        degree_layout=(self.r, self.c),
        num_moduli=self.sizeQl,
    )
    num_elements = ct_data.shape[1]
    p_product = math.prod(int(p) for p in self.extend_moduli)
    p_mod_q = jnp.asarray(
        [p_product % int(q) for q in self.overall_moduli[:self.sizeQl]],
        dtype=jnp.uint64,
    ).reshape(1, 1, 1, 1, self.sizeQl)
    p_mod_q = self.ct_in.ntt_ctx.ff_ctx.to_computation_format(p_mod_q)
    q_scaled = self.ct_in.ntt_ctx.ff_ctx.modular_reduction(
        ct_data.astype(jnp.uint64) * p_mod_q.astype(jnp.uint64)
    )
    q_scaled = self.ct_in.ntt_ctx.ff_ctx.strictify(q_scaled).astype(jnp.uint32)
    if not include_first:
      keep = (jnp.arange(num_elements) > 0).reshape(1, num_elements, 1, 1, 1)
      q_scaled = jnp.where(keep, q_scaled, jnp.zeros_like(q_scaled))
    result = jnp.zeros(
        (self.batch, num_elements, self.r, self.c, self.sizeQlP),
        dtype=jnp.uint32,
    )
    return result.at[..., :self.sizeQl].set(q_scaled)

  def _automorphism_array(self, data, coef_map=None):
    """Apply one NTT-domain automorphism to Q or QP ciphertext data."""
    self._require_controls()
    coef_map = self._resolve_automorphism_map(coef_map)
    data = jnp.asarray(data)
    if data.ndim != 5 or data.shape[0] != self.batch or \
       tuple(data.shape[-3:-1]) != (self.r, self.c):
      raise ValueError(
          'HERot._automorphism_array expects shape '
          f'({self.batch}, elements, {self.r}, {self.c}, moduli); got '
          f'{data.shape}.'
      )
    num_elements = data.shape[1]
    num_moduli = data.shape[-1]
    flat = data.reshape(
        self.batch, num_elements, self.r * self.c, num_moduli
    )
    return jnp.take(flat, jnp.asarray(coef_map, dtype=jnp.int32), axis=2) \
        .reshape(data.shape)

  def _add_first_component_array(self, target, source):
    """Add the first source component to a Q- or QP-basis target."""
    self._require_controls()
    finite_field.check_rank5_array(
        target,
        'HERot._add_first_component_array(target)',
        batch=self.batch,
        degree_layout=(self.r, self.c),
    )
    finite_field.check_rank5_array(
        source,
        'HERot._add_first_component_array(source)',
        batch=self.batch,
        degree_layout=(self.r, self.c),
    )
    if target.shape[1] < 1 or source.shape[1] < 1 or \
       target.shape[2:] != source.shape[2:]:
      raise ValueError(
          'HERot._add_first_component_array requires compatible rank-5 '
          'ciphertexts; got '
          f'{target.shape} and {source.shape}.'
      )
    num_moduli = target.shape[-1]
    if num_moduli not in (self.sizeQl, self.sizeQlP):
      raise ValueError(
          'HERot._add_first_component_array expects Q or QP data; got '
          f'{num_moduli} moduli.'
      )
    moduli = jnp.asarray(
        self.overall_moduli[:num_moduli], dtype=jnp.uint64
    ).reshape(1, 1, 1, 1, num_moduli)
    first = (
        target[:, 0:1].astype(jnp.uint64)
        + source[:, 0:1].astype(jnp.uint64)
    )
    # Raw ciphertext boundaries carry canonical residues, so one subtraction
    # is sufficient and avoids a uint64 remainder operation in accelerator
    # kernels.
    first = jnp.where(first >= moduli, first - moduli, first)
    return target.at[:, 0:1].set(first.astype(jnp.uint32))

  def _hoisted_rotate_array(
      self, ct_data, digits, eval_a=None, eval_b=None, coef_map=None,
      include_first=True,
  ):
    """Rotate with hoisted digits and retain the result in the QP basis."""
    self._require_controls()
    coef_map = self._resolve_automorphism_map(coef_map)
    switched = self._key_switch_core_array(digits, eval_a, eval_b)
    if include_first:
      embedded_first = self._key_switch_extend_array(
          ct_data, include_first=True
      )
      switched = self._add_first_component_array(switched, embedded_first)
    return self._automorphism_array(switched, coef_map)

  # ---------------------------------------------------------------------------
  # Runtime QP accumulation and final down-conversion.
  # ---------------------------------------------------------------------------

  def _mul_plain_array(self, qp_data, plaintext):
    """Multiply QP ciphertext data by a standard-form QP NTT plaintext.

    ``plaintext`` must end in ``(r, c, sizeQlP)`` and contain standard
    residues. It is converted to this kernel's computation representation
    exactly once before multiplication. The result remains in QP.
    """
    self._require_controls()
    qp_data = jnp.asarray(qp_data, dtype=jnp.uint32)
    plaintext = jnp.asarray(plaintext, dtype=jnp.uint64)
    if qp_data.ndim != 5 or qp_data.shape[0] != self.batch or \
       tuple(qp_data.shape[-3:]) != (self.r, self.c, self.sizeQlP):
      raise ValueError(
          'HERot._mul_plain_array ciphertext has incompatible shape '
          f'{qp_data.shape}.'
      )
    if tuple(plaintext.shape[-3:]) != (self.r, self.c, self.sizeQlP):
      raise ValueError(
          'HERot._mul_plain_array plaintext must end in '
          f'{(self.r, self.c, self.sizeQlP)}; got {plaintext.shape}.'
      )
    ff_full = self.ct_full.ntt_ctx.ff_ctx
    plaintext = ff_full.to_computation_format(plaintext)
    result = ff_full.modular_reduction(
        qp_data.astype(jnp.uint64) * plaintext.astype(jnp.uint64)
    )
    return ff_full.strictify(result).astype(jnp.uint32)

  def _mod_down_array(self, qp_data):
    """ApproxModDown one or more QP components into the active Ql basis."""
    self._require_controls()
    qp_data = jnp.asarray(qp_data, dtype=jnp.uint32)
    if qp_data.ndim != 5 or qp_data.shape[0] != self.batch or \
       tuple(qp_data.shape[-3:]) != (self.r, self.c, self.sizeQlP):
      raise ValueError(
          'HERot._mod_down_array expects shape '
          f'({self.batch}, elements, {self.r}, {self.c}, {self.sizeQlP}); '
          f'got {qp_data.shape}.'
      )
    num_elements = qp_data.shape[1]
    if num_elements < 1:
      raise ValueError(
          'HERot._mod_down_array requires at least one ciphertext component.'
      )

    ff_full = self.ct_full.ntt_ctx.ff_ctx
    q_moduli = jnp.asarray(
        self.overall_moduli[:self.sizeQl], dtype=jnp.uint64
    ).reshape(1, 1, 1, 1, self.sizeQl)
    p_inv = jnp.asarray(self.PInvModq, dtype=jnp.uint64).reshape(
        1, 1, 1, 1, self.sizeQl
    )
    components = []
    for i in range(num_elements):
      # Keep the per-component loop to bound live QP intermediates without a
      # separate single-call helper.
      reduced = ff_full.strictify(
          qp_data[:, i:i + 1]
      ).astype(jnp.uint32)
      p_part = reduced[..., self.sizeQl:]
      p_shape = p_part.shape
      p_coeffs = self.ct_approx.ntt_ctx.intt(
          p_part.reshape(-1, self.r, self.c, p_shape[-1])
      ).reshape(p_shape)
      q_from_p = self.bconv.basis_change_bat(
          p_coeffs, control_index=self.numPartQl
      ).astype(jnp.uint32)
      q_shape = q_from_p.shape
      q_from_p_ntt = self.ct_in.ntt_ctx.ntt(
          q_from_p.reshape(-1, self.r, self.c, self.sizeQl)
      ).reshape(q_shape)
      q_from_p_ntt = self.ct_in.ntt_ctx.ff_ctx.strictify(q_from_p_ntt)

      q_part = reduced[..., :self.sizeQl]
      diff = q_part.astype(jnp.uint64) - q_from_p_ntt.astype(jnp.uint64)
      diff = jnp.where(
          q_part < q_from_p_ntt, diff + q_moduli, diff
      ).astype(jnp.uint32)
      result = self.ct_in.ntt_ctx.ff_ctx.modular_reduction(
          diff.astype(jnp.uint64) * p_inv
      )
      components.append(
          self.ct_in.ntt_ctx.ff_ctx.strictify(result).astype(jnp.uint32)
      )
    return jnp.concatenate(components, axis=1)

__all__ = []

"""
Level-indexed HE operation wrappers and accessor classes.

Provides the private implementations behind the level-indexed CKKSContext
facade.  Public callers obtain these objects only through ``ctx.he_*``;
constructible kernels and raw-array hooks intentionally stay private here.
"""
from functools import cached_property
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

import hemul
import herot
import finite_field
import headd
import hesub
import polynomial
import rescale
import ptct_mul
import he_params

HEParameterCache = he_params.HEParameterCache
Polynomial = polynomial.Polynomial


def _validate_level(cache: HEParameterCache, level: int) -> int:
  """Return a valid logical level or raise a stable facade error."""
  if not isinstance(level, int):
    raise TypeError(f'level must be an int, got {type(level).__name__}.')
  if not 0 <= level <= cache.max_level:
    raise ValueError(
        f'level {level} out of range [0, {cache.max_level}].'
    )
  return level


def _validate_additive_metadata(left, right, operation: str) -> None:
  """Require the scale/noise metadata needed for a valid ciphertext sum."""
  left_scale = getattr(left, '_ckks_scale', None)
  right_scale = getattr(right, '_ckks_scale', None)
  if (left_scale is None) != (right_scale is None):
    raise ValueError(f'{operation}: operands have incompatible scale metadata.')
  if left_scale is not None and (
      not math.isfinite(float(left_scale))
      or not math.isfinite(float(right_scale))
      or not math.isclose(
          float(left_scale), float(right_scale), rel_tol=1e-12, abs_tol=0.0
      )
  ):
    raise ValueError(
        f'{operation}: operands have incompatible scales '
        f'{left_scale} and {right_scale}.'
    )
  left_nsd = getattr(left, '_ckks_nsd', 1)
  right_nsd = getattr(right, '_ckks_nsd', 1)
  if left_nsd != right_nsd:
    raise ValueError(
        f'{operation}: operands have incompatible nsd '
        f'{left_nsd} and {right_nsd}.'
    )


jax.config.update("jax_enable_x64", True)


class _HEMulAtLevel:
  """HEMul configured for one exact output level.

  ``mul`` and ``_square_array`` accept ciphertexts at ``level + 1`` and
  always return ciphertexts at ``level``, for every composite degree.

  ``hemul_no_relin`` and ``relinearize`` expose the two explicit no-rescale
  stages for schedules that own rescaling. Both preserve the input level.
  """

  def __init__(self, cache: HEParameterCache, level: int):
    _validate_level(cache, level)
    if level == cache.max_level:
      raise ValueError(
          f'ctx.he_mul output level {level} requires input level {level + 1}, '
          f'but max_level is {cache.max_level}.'
      )
    self.cache = cache
    self.level = level
    self._cd = cache.composite_degree

    # Input is at level+1 (before rescale)
    input_level = level + 1
    num_q_input = cache.num_q_at_level(input_level)
    q_input = cache.q_towers[:num_q_input]

    use_skip_rescale = cache.composite_degree >= 2

    self._hemul = hemul._HEMulKernel(
        batch=cache.batch,
        r=cache.r,
        c=cache.c,
        dnum=cache.dnum,
        num_eval_mult=1,
        original_moduli=q_input,
        extend_moduli=cache.p_towers,
        composite_degree=cache.composite_degree,
        finite_field_context=cache.ff_context_cls,
    )
    self._hemul.control_gen(
        degree_layout=cache.degree_layout,
        perf_test=cache.perf_test,
        keygen_sizeQ=len(cache.q_towers),
        skip_rescale=use_skip_rescale,
    )

    eval_key_level = input_level if use_skip_rescale else level
    eval_a, eval_b = cache.get_eval_key(eval_key_level)
    self._hemul.setup_relinearization(eval_a, eval_b)

    self._num_q_output = num_q_input - self._cd
    self._use_skip_rescale = use_skip_rescale

    # For cd>=2: pre-create post-rescale operator
    if use_skip_rescale and self._cd >= 2:
      self._post_rescale = rescale._HERescaleKernel(
          batch=cache.batch,
          num_elements=2,
          moduli=q_input,
          r=cache.r,
          c=cache.c,
          degree_layout=cache.degree_layout,
          finite_field_context=cache.ff_context_cls,
      )
      self._post_rescale.control_gen(
          composite_degree=self._cd,
          perf_test=cache.perf_test,
      )

    self._input_level = input_level

    self._q_input = q_input

    # The explicit tensor/relinearization controls need a second skip-rescale
    # kernel for composite_degree=1. Resolve it while this accessor is
    # materialized so control generation cannot leak into a later request.
    if not self._use_skip_rescale:
      _ = self._no_relin_operator

  def mul(self, ct1: Polynomial, ct2: Polynomial) -> Polynomial:
    """Multiply, relinearize, and return exactly this operator's level.

    Inputs must be at ``level + 1``. For composite degree one the private
    multiply kernel performs the rescale. For larger composite degrees this
    facade applies its prebuilt composite post-rescale to the relinearized
    result. Thus the public result always drops one logical level.
    """
    combined = self._combine_inputs_array(ct1, ct2)
    result = self._hemul._mul_array_to_polynomial_unchecked(combined)
    if self._use_skip_rescale:
      result = self._post_rescale.rescale(result)
    left_scale = getattr(ct1, '_ckks_scale', None)
    right_scale = getattr(ct2, '_ckks_scale', None)
    if left_scale is not None and right_scale is not None:
      divisor = 1
      for modulus in self._q_input[-self._cd:]:
        divisor *= int(modulus)
      if self._use_skip_rescale:
        # cd >= 2 tensors at the input scale and then performs one composite
        # post-rescale.
        result._ckks_scale = (left_scale * right_scale) / divisor
      else:
        # cd == 1 preserves the historical pipeline: both operands are
        # pre-rescaled before their tensor product.
        result._ckks_scale = (
            (left_scale / divisor) * (right_scale / divisor)
        )
    return result

  def hemul_no_relin(
      self, ct1: Polynomial, ct2: Polynomial
  ) -> Polynomial:
    """Tensor multiply only; return three elements at the input level."""
    combined = self._combine_inputs_array(ct1, ct2)
    operator = self._no_relin_operator
    result = operator._hemul_no_relin_array_to_polynomial_unchecked(combined)
    left_scale = getattr(ct1, '_ckks_scale', None)
    right_scale = getattr(ct2, '_ckks_scale', None)
    if left_scale is not None and right_scale is not None:
      result._ckks_scale = left_scale * right_scale
    result._ckks_nsd = (
        getattr(ct1, '_ckks_nsd', 1) + getattr(ct2, '_ckks_nsd', 1)
    )
    return result

  def relinearize(self, ct_3elem: Polynomial) -> Polynomial:
    """Relinearize a three-element intermediate without rescaling."""
    result = self._no_relin_operator.relinearize(ct_3elem)
    if hasattr(ct_3elem, '_ckks_scale'):
      result._ckks_scale = ct_3elem._ckks_scale
    result._ckks_nsd = getattr(ct_3elem, '_ckks_nsd', 2)
    return result

  @property
  def input_num_moduli(self) -> int:
    """Number of Q moduli expected by the configured low-level multiply."""
    return self._hemul.overall_sizeQ_in

  def _square_array(self, ct_data):
    """Private fused square from ``level + 1`` to exactly ``level``."""
    finite_field.check_rank5_array(
        ct_data,
        '_HEMulAtLevel._square_array',
        batch=self.cache.batch,
        num_elements=2,
        degree_layout=self.cache.degree_layout,
        num_moduli=len(self._q_input),
    )
    result = self._hemul._mul_array_to_polynomial_unchecked(
        jnp.concatenate([ct_data, ct_data], axis=1)
    )
    if self._use_skip_rescale:
      return self._post_rescale._rescale_array_unchecked(result.polynomial)
    return result.polynomial

  def _mul_array(self, left_data, right_data):
    """Private fused binary multiply from ``level + 1`` to ``level``."""
    for name, value in (('left_data', left_data), ('right_data', right_data)):
      finite_field.check_rank5_array(
          value,
          f'_HEMulAtLevel._mul_array({name})',
          batch=self.cache.batch,
          num_elements=2,
          degree_layout=self.cache.degree_layout,
          num_moduli=len(self._q_input),
      )
    if left_data.shape != right_data.shape:
      raise ValueError(
          '_HEMulAtLevel._mul_array operands must have identical shapes; '
          f'got {left_data.shape} and {right_data.shape}.'
      )
    result = self._hemul._mul_array_to_polynomial_unchecked(
        jnp.concatenate([left_data, right_data], axis=1).astype(jnp.uint32)
    )
    if self._use_skip_rescale:
      return self._post_rescale._rescale_array_unchecked(result.polynomial)
    return result.polynomial

  @cached_property
  def _no_relin_operator(self) -> hemul._HEMulKernel:
    """Return the HEMul configured for no-rescale multiplication.

    Each HEMul holds one parameter set, so cd=1 needs a second instance for
    the unchanged input level.
    """
    if self._use_skip_rescale:
      return self._hemul

    # cd=1 full mul works after rescaling, while the split API must work at
    # the unchanged input level. ``__init__`` eagerly resolves this property
    # so construction and control generation stay in offline preparation.
    op = hemul._HEMulKernel(
        batch=self.cache.batch,
        r=self.cache.r,
        c=self.cache.c,
        dnum=self.cache.dnum,
        num_eval_mult=1,
        original_moduli=self._q_input,
        extend_moduli=self.cache.p_towers,
        composite_degree=self.cache.composite_degree,
        finite_field_context=self.cache.ff_context_cls,
    )
    op.control_gen(
        degree_layout=self.cache.degree_layout,
        perf_test=self.cache.perf_test,
        keygen_sizeQ=len(self.cache.q_towers),
        skip_rescale=True,
    )
    eval_a, eval_b = self.cache.get_eval_key(self._input_level)
    op.setup_relinearization(eval_a, eval_b)
    return op

  def _combine_inputs_array(self, ct1: Polynomial, ct2: Polynomial):
    """Validate two ciphertexts and concatenate their private payloads."""
    for name, ct in (('ct1', ct1), ('ct2', ct2)):
      finite_field.check_ct_operand(
          self._hemul.ff_context_cls,
          len(self._q_input),
          ct,
          f'ctx.he_mul._combine_inputs_array({name})',
          batch=self.cache.batch,
          num_elements=2,
          degree_layout=self.cache.degree_layout,
          moduli=self._q_input,
      )
    combined = jnp.concatenate(
        [ct1.polynomial, ct2.polynomial], axis=1
    ).astype(jnp.uint32)
    return combined


class _HERotAtLevel:
  """HERot configured for a specific level and rotation index.

  ``rotate`` is the supported ``Polynomial`` operation. Private
  ``_<step>_array`` methods expose raw rank-5 payload transforms only to
  internal fused/JIT compositions, matching the other level-bound operators.
  """

  def __init__(self, cache: HEParameterCache, level: int, rot_index: int):
    _validate_level(cache, level)
    self.cache = cache
    self.level = level
    self.rot_index = rot_index

    q_at_level = cache.q_moduli_at_level(level)
    num_q = len(q_at_level)
    # Preserve the max-level key's partition boundaries at lower levels.
    # Only the number of non-empty partitions shrinks with the Q tower count.
    # per_level_rotation_keys mode (CROSS_SKIP_TOPLEVEL_ROTKEYS) regenerates a
    # per-level-sized key instead of retaining the max-level key material.
    keygen_size_q = (
        num_q if cache.per_level_rotation_keys else len(cache.q_towers)
    )
    alpha = (keygen_size_q + cache.dnum - 1) // cache.dnum
    effective_dnum = (num_q + alpha - 1) // alpha
    self._herot = herot._HERotKernel(
        cache.r, cache.c, cache.dnum, q_at_level, cache.p_towers,
        finite_field_context=cache.ff_context_cls,
    )

    self._herot.control_gen(
        batch=cache.batch,
        degree_layout=cache.degree_layout,
        perf_test=cache.perf_test,
        keygen_sizeQ=keygen_size_q,
    )
    rot_a, rot_b, coef_map = cache.get_rot_key(rot_index, level)
    # Slice rotation key to effective_dnum partitions
    self._herot.setup_rotation(
        rot_a[:effective_dnum], rot_b[:effective_dnum], coef_map
    )

  def rotate(self, ct_in: Polynomial) -> Polynomial:
    result = self._herot.rotate(ct_in)
    # Rotation does not change CKKS scale/depth metadata.  The private kernel
    # returns a clone of its reusable template rather than of ``ct_in``, so
    # preserve the operand metadata explicitly at the facade boundary.
    for attribute in ('_ckks_scale', '_ckks_nsd'):
      if hasattr(ct_in, attribute):
        setattr(result, attribute, getattr(ct_in, attribute))
    return result

  def _rotation_state(self):
    """Return computation-format keys and map for a fused rotation scan."""
    return self._herot._rotation_state()

  def _rotate_array(self, ct_data, eval_a=None, eval_b=None, coef_map=None):
    """Private raw-array rotate hook for a fused composition region."""
    return self._herot._rotate_array(
        ct_data, eval_a=eval_a, eval_b=eval_b, coef_map=coef_map
    )

  def _take_explicit_rotation_fn(self):
    """Transfer a key-free explicit-state runtime function to a fused owner."""
    kernel = self._herot
    rotate = kernel._rotate_array
    kernel.evalkey_a_vector = None
    kernel.evalkey_b_vector = None
    kernel.coef_map = None
    kernel._rotation_ready = False
    return rotate

  @property
  def num_partitions(self) -> int:
    """Number of non-empty key-switch partitions at this level."""
    return self._herot.numPartQl

  # --- Private hoisted-QP hooks used by the bootstrapping engine. ---
  def _decompose_array(self, ct_data):
    """Hoist c1 digit decomposition for QP-domain fast rotations."""
    return self._herot._decompose_array(ct_data)

  def _key_switch_extend_array(self, ct_data, include_first=True):
    """Embed Q ciphertext data in OpenFHE's P-scaled QP convention."""
    return self._herot._key_switch_extend_array(
        ct_data, include_first=include_first
    )

  def _hoisted_rotate_array(
      self, ct_data, digits, eval_a=None, eval_b=None, coef_map=None,
      include_first=True,
  ):
    """Apply a hoisted rotation while retaining the result in QP."""
    return self._herot._hoisted_rotate_array(
        ct_data,
        digits,
        eval_a=eval_a,
        eval_b=eval_b,
        coef_map=coef_map,
        include_first=include_first,
    )

  def _mod_down_array(self, qp_data):
    """Apply HYBRID ApproxModDown to QP ciphertext components."""
    return self._herot._mod_down_array(qp_data)

  def _mul_plain_array(self, qp_data, plaintext):
    """Multiply by a standard-form QP NTT plaintext without ModDown."""
    return self._herot._mul_plain_array(qp_data, plaintext)

  def _automorphism_array(self, data, coef_map=None):
    """Apply this rotation's automorphism map without key switching."""
    return self._herot._automorphism_array(data, coef_map)

  @property
  def extended_moduli(self):
    return tuple(self._herot.overall_moduli)


class _HERescaleAtLevels:
  """Rescale from src_level to dst_level.

  Drops (src_level - dst_level) * composite_degree moduli total.
  """

  def __init__(self, cache: HEParameterCache, src_level: int, dst_level: int):
    _validate_level(cache, src_level)
    _validate_level(cache, dst_level)
    if src_level <= dst_level:
      raise ValueError(
          f'src_level ({src_level}) must be greater than dst_level '
          f'({dst_level}).'
      )
    self.cache = cache
    self.src_level = src_level
    self.dst_level = dst_level
    self.num_steps = src_level - dst_level
    total_drop = self.num_steps * cache.composite_degree

    self._num_q_src = cache.num_q_at_level(src_level)
    self._q_at_src = cache.q_towers[: self._num_q_src]

    # Pre-create HERescale (reusable — no internal mutation)
    self._he_rescale = rescale._HERescaleKernel(
        batch=cache.batch,
        num_elements=2,
        moduli=self._q_at_src,
        r=cache.r,
        c=cache.c,
        degree_layout=cache.degree_layout,
        finite_field_context=cache.ff_context_cls,
    )
    self._he_rescale.control_gen(composite_degree=total_drop)

  def rescale(self, ct_in: Polynomial) -> Polynomial:
    result = self._he_rescale.rescale(ct_in)
    tracked_scale = getattr(ct_in, '_ckks_scale', None)
    if tracked_scale is not None:
      divisor = 1
      for modulus in self._q_at_src[-self._he_rescale.composite_degree:]:
        divisor *= int(modulus)
      result._ckks_scale = tracked_scale / divisor
    return result

  def _rescale_array(self, ct_data):
    """Private raw-array rescale hook for a fused composition region."""
    return self._he_rescale._rescale_array(ct_data)


class _HELevelReduceAtLevels:
  """Drop modulus limbs to reach ``dst_level`` while preserving the scale.

  Deliberately distinct from rescale. Rescale divides the plaintext scale by
  the moduli it drops, which is what keeps noise in check after a
  multiplication. This one only truncates the CRT representation: the value
  and its tracked scale and noise-scale degree are unchanged, so a ciphertext
  can be brought down to meet another at a shared level and still be added to
  it. Same semantics as OpenFHE's LevelReduceInPlace.
  """

  def __init__(self, cache: HEParameterCache, src_level: int, dst_level: int):
    _validate_level(cache, src_level)
    _validate_level(cache, dst_level)
    if src_level <= dst_level:
      raise ValueError(
          f'src_level ({src_level}) must be greater than dst_level '
          f'({dst_level}).'
      )
    self.cache = cache
    self.src_level = src_level
    self.dst_level = dst_level
    self._target_num_q = cache.num_q_at_level(dst_level)

  def level_reduce(self, ct_in: Polynomial) -> Polynomial:
    result = ct_in._clone_with_payload(
        ct_in.polynomial[..., : self._target_num_q],
        moduli=self.cache.q_moduli_at_level(self.dst_level),
        ntt_ctx=self.cache.get_sliced_ntt_q(self.dst_level),
    )
    # Level reduction changes neither the represented value nor its scale.
    scale = getattr(ct_in, '_ckks_scale', None)
    if scale is not None:
      result._ckks_scale = scale
    result._ckks_nsd = getattr(ct_in, '_ckks_nsd', 1)
    return result

  def _level_reduce_array(self, ct_data):
    """Raw-array hook for a fused composition region."""
    return ct_data[..., : self._target_num_q]


class _HEPtCtMulAtLevel:
  """Polynomial-Plaintext multiply at a specific level.

  The sole public operation is stateless: ``mul`` receives both the ciphertext
  and a canonical plaintext ``Polynomial``. A private precompute hook retains
  the BAT benchmark path without exposing it as a second evaluator API.

  Usage:
      op = ctx.ptct_mul[level]
      result = op.mul(ciphertext, plaintext)
  """

  def __init__(self, cache: HEParameterCache, level: int):
    _validate_level(cache, level)
    self.cache = cache
    self.level = level
    q_at_level = cache.q_moduli_at_level(level)
    self._ptct = ptct_mul._HEPtCtMulKernel(
        batch=cache.batch,
        r=cache.r,
        c=cache.c,
        moduli=q_at_level,
        degree_layout=cache.degree_layout,
        finite_field_context=cache.ff_context_cls,
    )

  def _precompute_bat(self, pt_ntt: Optional[jnp.ndarray] = None):
    """Privately precompute a BAT plaintext for performance experiments.

    Args:
        pt_ntt: If provided, sets and precomputes. Otherwise uses previously set
          pt.
    """
    if pt_ntt is not None:
      self._ptct.precompute_plaintext_bat(pt_ntt)
    elif self._ptct.pt_ntt is not None:
      self._ptct.precompute_plaintext_bat(self._ptct.pt_ntt)
    else:
      raise RuntimeError(
          'No plaintext set. Provide pt_ntt to _precompute_bat.'
      )

  def mul(
      self,
      ct: Polynomial,
      plaintext: Polynomial,
  ) -> Polynomial:
    """Multiply a ciphertext by one explicit canonical plaintext.

    Args:
        ct: Input ciphertext at this level.
        plaintext: Batch-one, one-element plaintext Polynomial at this level.
    """
    finite_field.check_ct_operand(
        self._ptct.ff_context_cls,
        self._ptct.num_moduli,
        ct,
        'ctx.ptct_mul.mul(ct)',
        batch=self.cache.batch,
        num_elements=2,
        degree_layout=self.cache.degree_layout,
        moduli=self._ptct.moduli,
    )
    finite_field.check_ct_operand(
        self._ptct.ff_context_cls,
        self._ptct.num_moduli,
        plaintext,
        'ctx.ptct_mul.mul(plaintext)',
        batch=1,
        num_elements=1,
        degree_layout=self.cache.degree_layout,
        moduli=self._ptct.moduli,
    )
    plaintext_data = self.cache._prepare_plaintext_payload(
        plaintext.polynomial, self.level
    )
    result = ct._clone_with_payload(
        self._ptct._mul_array(ct.polynomial, plaintext_data)
    )
    ciphertext_scale = getattr(ct, '_ckks_scale', None)
    plaintext_scale = getattr(plaintext, '_ckks_scale', None)
    if ciphertext_scale is not None and plaintext_scale is not None:
      result._ckks_scale = float(ciphertext_scale) * float(plaintext_scale)
    else:
      result.__dict__.pop('_ckks_scale', None)
    result._ckks_nsd = getattr(ct, '_ckks_nsd', 1) + 1
    return result

  def _mul_encoded(self, ct: Polynomial, pt_ntt) -> Polynomial:
    """Private stateless multiply by an already-encoded raw plaintext.

    The raw plaintext arrives in standard residues; encode it into the
    backend's computation format once before the multiply (identity for
    Barrett; Montgomery: pt*R mod q) so the REDC-based reduction yields a
    computation-format product.
    """
    pt_ntt = self._ptct.ff_ctx.to_computation_format(
        jnp.asarray(pt_ntt, jnp.uint64))
    finite_field.check_ct_operand(
        self._ptct.ff_context_cls,
        self._ptct.num_moduli,
        ct,
        'ctx.ptct_mul._mul_encoded(ct)',
        batch=self.cache.batch,
        num_elements=2,
        degree_layout=self.cache.degree_layout,
        moduli=self._ptct.moduli,
    )
    result = ct._clone_with_payload(
        self._ptct._mul_array(ct.polynomial, pt_ntt)
    )
    result.__dict__.pop('_ckks_scale', None)
    return result

  def _mul_prepared(
      self, ct: Polynomial, *, use_bat: bool = False
  ) -> Polynomial:
    """Private stateful path retained only for BAT kernel experiments."""
    finite_field.check_ct_operand(
        self._ptct.ff_context_cls,
        self._ptct.num_moduli,
        ct,
        'ctx.ptct_mul._mul_prepared(ct)',
        batch=self.cache.batch,
        num_elements=2,
        degree_layout=self.cache.degree_layout,
        moduli=self._ptct.moduli,
    )
    result = self._ptct.mul(ct, use_bat=use_bat)
    result.__dict__.pop('_ckks_scale', None)
    return result

  def _mul_array(self, ct_data, pt_ntt):
    """Private raw-array pt-ct multiply hook for a fused region."""
    return self._ptct._mul_array(ct_data, pt_ntt)


class _HEAddAtLevel:
  """Ciphertext addition bound to one exact logical level."""

  def __init__(self, cache: HEParameterCache, level: int):
    _validate_level(cache, level)
    self.cache = cache
    self.level = level
    self._kernel = headd._HEAddKernel(
        cache.q_moduli_at_level(level),
        finite_field_context=cache.get_sliced_ff_q(level),
    )

  def add(self, ct1: Polynomial, ct2: Polynomial) -> Polynomial:
    """Add two ciphertexts at this exact level."""
    finite_field.check_binary_ct_operands(
        self._kernel.ff_context_cls,
        len(self._kernel.moduli),
        ct1,
        ct2,
        'ctx.he_add.add',
        batch=self.cache.batch,
        num_elements=2,
        degree_layout=self.cache.degree_layout,
        moduli=self._kernel.moduli,
    )
    _validate_additive_metadata(ct1, ct2, 'ctx.he_add.add')
    return self._kernel.add(ct1, ct2)

  def _add_array(self, ct1_data, ct2_data):
    """Private raw-array add hook for a fused composition region."""
    return self._kernel._add_array(ct1_data, ct2_data)

  def add_plain(
      self, ct: Polynomial, plaintext: Polynomial
  ) -> Polynomial:
    """Add one batch-one plaintext Polynomial to every ciphertext batch."""
    finite_field.check_ct_operand(
        self._kernel.ff_context_cls,
        len(self._kernel.moduli),
        ct,
        'ctx.he_add.add_plain(ct)',
        batch=self.cache.batch,
        num_elements=2,
        degree_layout=self.cache.degree_layout,
        moduli=self._kernel.moduli,
    )
    finite_field.check_ct_operand(
        self._kernel.ff_context_cls,
        len(self._kernel.moduli),
        plaintext,
        'ctx.he_add.add_plain(plaintext)',
        batch=1,
        num_elements=1,
        degree_layout=self.cache.degree_layout,
        moduli=self._kernel.moduli,
    )
    ciphertext_scale = getattr(ct, '_ckks_scale', None)
    plaintext_scale = getattr(plaintext, '_ckks_scale', None)
    if (
        ciphertext_scale is not None
        and plaintext_scale is not None
        and not math.isclose(
            float(ciphertext_scale),
            float(plaintext_scale),
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
      raise ValueError(
          'ctx.he_add.add_plain: plaintext scale '
          f'{plaintext_scale} does not match ciphertext scale '
          f'{ciphertext_scale}.'
      )
    plaintext_data = self.cache._prepare_plaintext_payload(
        plaintext.polynomial, self.level
    )
    return ct._clone_with_payload(
        self._add_plain_array(ct.polynomial, plaintext_data)
    )

  def _add_plain_array(self, ct_data, plaintext_data):
    """Private fused plaintext add with batch-one plaintext broadcasting."""
    finite_field.check_rank5_array(
        ct_data,
        '_HEAddAtLevel._add_plain_array(ct_data)',
        batch=self.cache.batch,
        num_elements=2,
        degree_layout=self.cache.degree_layout,
        num_moduli=len(self._kernel.moduli),
    )
    finite_field.check_rank5_array(
        plaintext_data,
        '_HEAddAtLevel._add_plain_array(plaintext_data)',
        batch=1,
        num_elements=1,
        degree_layout=self.cache.degree_layout,
        num_moduli=len(self._kernel.moduli),
    )
    c0_sum = (
        ct_data[:, :1].astype(jnp.uint64)
        + plaintext_data.astype(jnp.uint64)
    )
    c0 = self._kernel.ff_ctx.strictify_after_accumulation(
        c0_sum, 2
    ).astype(jnp.uint32)
    return jnp.concatenate([c0, ct_data[:, 1:2]], axis=1)


class _HESubAtLevel:
  """Ciphertext subtraction bound to one exact logical level."""

  def __init__(self, cache: HEParameterCache, level: int):
    _validate_level(cache, level)
    self.cache = cache
    self.level = level
    self._kernel = hesub._HESubKernel(
        cache.q_moduli_at_level(level),
        finite_field_context=cache.get_sliced_ff_q(level),
    )

  def sub(self, ct1: Polynomial, ct2: Polynomial) -> Polynomial:
    """Subtract two ciphertexts at this exact level."""
    finite_field.check_binary_ct_operands(
        self._kernel.ff_context_cls,
        len(self._kernel.moduli),
        ct1,
        ct2,
        'ctx.he_sub.sub',
        batch=self.cache.batch,
        num_elements=2,
        degree_layout=self.cache.degree_layout,
        moduli=self._kernel.moduli,
    )
    _validate_additive_metadata(ct1, ct2, 'ctx.he_sub.sub')
    return self._kernel.sub(ct1, ct2)

  def _sub_array(self, ct1_data, ct2_data):
    """Private raw-array subtract hook for a fused composition region."""
    return self._kernel._sub_array(ct1_data, ct2_data)


class _HEPtCtMulAccessor:
  """Provides ctx.ptct_mul[level] indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, level: int) -> _HEPtCtMulAtLevel:
    if level not in self._instances:
      self._instances[level] = _HEPtCtMulAtLevel(self.cache, level)
    return self._instances[level]

  def clear(self):
    self._instances = {}


class _HEMulAccessor:
  """Provides ctx.he_mul[level] indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, level: int) -> _HEMulAtLevel:
    if level not in self._instances:
      self._instances[level] = _HEMulAtLevel(self.cache, level)
    return self._instances[level]

  def clear(self):
    self._instances = {}


class _HERotAccessor:
  """Provides ctx.he_rot[level, rot_index] indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, key: Tuple[int, int]) -> _HERotAtLevel:
    level, rot_index = key
    if key not in self._instances:
      self._instances[key] = _HERotAtLevel(self.cache, level, rot_index)
    return self._instances[key]

  def take_explicit_rotation_fn(self, key: Tuple[int, int]):
    """Transfer one key-free runtime kernel and evict its keyed facade."""
    operation = self[key]
    rotate = operation._take_explicit_rotation_fn()
    del self._instances[key]
    return rotate

  def clear(self):
    """Evict all cached per-(level, rot_index) rotation instances.

    Each cached HERotAtLevel holds a regenerated rotation key (~tens of MB at
    reference width). In a streaming forward each layer uses one level, so
    calling this between layers bounds the resident rotation working set to the
    current layer instead of accumulating every layer's keys.
    """
    self._instances = {}


class _HERescaleAccessor:
  """Provides ctx.he_rescale[src_level, dst_level] indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, key: Tuple[int, int]) -> _HERescaleAtLevels:
    src, dst = key
    if key not in self._instances:
      self._instances[key] = _HERescaleAtLevels(self.cache, src, dst)
    return self._instances[key]

  def clear(self):
    self._instances = {}


class _HELevelReduceAccessor:
  """Provides ``ctx.he_level_reduce[src_level, dst_level]`` indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, key: Tuple[int, int]) -> _HELevelReduceAtLevels:
    src, dst = key
    if key not in self._instances:
      self._instances[key] = _HELevelReduceAtLevels(self.cache, src, dst)
    return self._instances[key]

  def clear(self):
    self._instances = {}


class _HEAddAccessor:
  """Provides ``ctx.he_add[level]`` indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, level: int) -> _HEAddAtLevel:
    if level not in self._instances:
      self._instances[level] = _HEAddAtLevel(self.cache, level)
    return self._instances[level]

  def clear(self):
    self._instances = {}


class _HESubAccessor:
  """Provides ``ctx.he_sub[level]`` indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, level: int) -> _HESubAtLevel:
    if level not in self._instances:
      self._instances[level] = _HESubAtLevel(self.cache, level)
    return self._instances[level]

  def clear(self):
    self._instances = {}


class _HEBootstrapAccessor:
  """Context-owned facade around the repository's single bootstrap engine."""

  def __init__(self, ctx):
    self._ctx = ctx
    self._engine = None
    self._configured = False
    self._is_setup = False
    self._input_specs = None
    self._input_level = None
    self._input_scale = None
    self._input_nsd = 1
    self._configuration = None

  def configure(
      self,
      level_budget=None,
      secret_key_dist: str = 'uniform_ternary',
      input_specs=None,
      input_level: Optional[int] = None,
      input_scale: Optional[float] = None,
      input_nsd: int = 1,
  ) -> '_HEBootstrapAccessor':
    """Configure the repository's Bootstrap engine and offline constants.

    ``input_specs`` is the frozen aggregate of every
    ``(level, scale, nsd)`` request shape that may reach this configuration;
    direct callers may instead supply the single input_level/input_scale/nsd
    form.
    """
    if self._configured:
      raise RuntimeError(
          'this CKKSContext already owns a configured bootstrap plan.'
      )
    import bootstrapping
    self._engine = bootstrapping.Bootstrap(self._ctx)
    self._engine.control_gen(
        level_budget=level_budget,
        secret_key_dist=secret_key_dist,
    )
    self._configuration = (
        tuple(self._engine._level_budget), secret_key_dist
    )
    self._input_specs = (
        None
        if input_specs is None
        else tuple(tuple(spec) for spec in input_specs)
    )
    self._input_level = input_level
    self._input_scale = input_scale
    self._input_nsd = input_nsd
    self._configured = True
    self._is_setup = False
    return self

  @property
  def required_rotation_indices(self) -> tuple[int, ...]:
    """Rotation indices required by the configured bootstrap plan."""
    self._require_configured()
    # CoeffToSlot's conjugate path uses the Galois index M-1 in addition to
    # the rotations discovered by control generation.  It must be advertised
    # during the planning phase so a keyless deployment can supply every key.
    conjugation_index = 2 * self._ctx.degree - 1
    return tuple(sorted({
        *self._engine._all_rot_indices,
        conjugation_index,
    }))

  def setup(self) -> '_HEBootstrapAccessor':
    """Bind configured bootstrap constants and keys to this context."""
    self._require_configured()
    self._engine.setup_key()
    cache = self._ctx._param_cache
    # Materialize every advertised max-level key now. This both warms the
    # formatted-key cache and makes setup fail immediately when a keyless
    # deployment omitted required pregenerated material.
    for rotation_index in self.required_rotation_indices:
      cache.get_rot_key(rotation_index, cache.max_level)
    self._is_setup = True
    return self

  def bootstrap(self, ciphertext: Polynomial) -> Polynomial:
    """Refresh one ciphertext with the configured single-pass engine."""
    self._require_setup()
    return self._engine.bootstrap(ciphertext)

  def meta_bootstrap(
      self, ciphertext: Polynomial, precision: int
  ) -> Polynomial:
    """Run the engine's two-pass precision-refinement mode."""
    self._require_setup()
    return self._engine.meta_bootstrap(ciphertext, precision)

  def _require_configured(self) -> None:
    if not self._configured:
      raise RuntimeError(
          'Configure ctx.he_bootstrap before requesting its plan or setup.'
      )

  def _require_setup(self) -> None:
    self._require_configured()
    if not self._is_setup:
      raise RuntimeError(
          'Call ctx.he_bootstrap.setup() before refreshing ciphertexts.'
      )


__all__ = []

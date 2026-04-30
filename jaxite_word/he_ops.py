"""
Level-indexed HE operation wrappers and accessor classes.

Provides ctx.he_mul[level], ctx.he_rot[level, rot_index], ctx.he_rescale[src, dst]
syntax by wrapping existing HEMul, HERot, and Polynomial.rescale() with parameters
from HEParameterCache.
"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

import hemul
import herot
import polynomial
import rescale
import ptct_mul
import he_params

HEParameterCache = he_params.HEParameterCache
HEMul = hemul.HEMul
HERot = herot.HERot
Polynomial = polynomial.Polynomial
HERescale = rescale.HERescale
HEPtCtMul = ptct_mul.HEPtCtMul


jax.config.update("jax_enable_x64", True)



class HEMulAtLevel:
  """HEMul configured for a specific output level.

  Expects input ciphertexts at level+1 (before rescale).
  Internally rescales to level, then multiplies.
  """

  def __init__(self, cache: HEParameterCache, level: int):
    assert (
        0 <= level <= cache.max_level
    ), f'level {level} out of range [0, {cache.max_level}]'
    self.cache = cache
    self.level = level

    # Input is at level+1 (before rescale)
    input_level = min(level + 1, cache.max_level)
    num_q_input = cache.num_q_at_level(input_level)
    q_input = cache.q_towers[:num_q_input]

    self._hemul = HEMul(
        batch=cache.batch,
        r=cache.r,
        c=cache.c,
        dnum=cache.dnum,
        num_eval_mult=1,
        original_moduli=q_input,
        extend_moduli=cache.p_towers,
        composite_degree=cache.composite_degree,
    )
    self._hemul.control_gen(
        degree_layout=cache.degree_layout,
        perf_test=cache.perf_test,
        keygen_sizeQ=len(cache.q_towers),
    )

    eval_a, eval_b = cache.get_eval_key(level)
    self._hemul.setup_relinearization(eval_a, eval_b)

    # Pre-create input Polynomial for _combine_inputs
    self._ct_in_shapes = {
        'batch': cache.batch,
        'num_elements': 4,
        'degree': cache.degree,
        'precision': 32,
        'num_moduli': num_q_input,
        'degree_layout': cache.degree_layout,
    }
    self._q_input = q_input

  def mul(self, ct1: Polynomial, ct2: Polynomial) -> Polynomial:
    """Full HEMul: rescale + tensor_multiply + relinearize."""
    ct_combined = self._combine_inputs(ct1, ct2)
    return self._hemul.mul(ct_combined)

  def hemul_no_relin(self, ct1: Polynomial, ct2: Polynomial) -> jnp.ndarray:
    """Tensor multiply only (with rescale). Returns 3-element array."""
    ct_combined = self._combine_inputs(ct1, ct2)
    return self._hemul.hemul_no_relin(ct_combined)

  def relinearize(self, ct_3elem: jnp.ndarray) -> Polynomial:
    """Relinearize a 3-element intermediate result."""
    return self._hemul.relinearize(ct_3elem)

  def _combine_inputs(self, ct1: Polynomial, ct2: Polynomial) -> Polynomial:
    """Concatenate two 2-element ciphertexts into a 4-element ciphertext."""
    combined = (
        jnp.concatenate([ct1.polynomial, ct2.polynomial], axis=1)
        .reshape(self.cache.batch, 4, self.cache.r, self.cache.c, -1)
        .astype(jnp.uint32)
    )
    ct_in = Polynomial(self._ct_in_shapes, parameters={'moduli': self._q_input})
    ct_in.polynomial = combined
    return ct_in


class HERotAtLevel:
  """HERot configured for a specific level and rotation index."""

  def __init__(self, cache: HEParameterCache, level: int, rot_index: int):
    assert (
        0 <= level <= cache.max_level
    ), f'level {level} out of range [0, {cache.max_level}]'
    self.cache = cache
    self.level = level
    self.rot_index = rot_index

    q_at_level = cache.q_moduli_at_level(level)
    num_q = len(q_at_level)
    # Effective dnum: at lower levels with fewer Q towers, dnum may
    # exceed the tower count, producing fewer key-switch partitions.
    # HERot.rotate() iterates over self.dnum, so we must cap dnum
    # to match the actual number of partitions control_gen will create.
    alpha = (num_q + cache.dnum - 1) // cache.dnum
    effective_dnum = (num_q + alpha - 1) // alpha
    self._herot = HERot(
        cache.r, cache.c, effective_dnum, q_at_level, cache.p_towers
    )

    self._herot.control_gen(
        batch=cache.batch,
        degree_layout=cache.degree_layout,
        perf_test=cache.perf_test,
    )
    rot_a, rot_b, coef_map = cache.get_rot_key(rot_index, level)
    # Slice rotation key to effective_dnum partitions
    self._herot.setup_rotate(
        rot_a[:effective_dnum], rot_b[:effective_dnum], coef_map
    )

  def rotate(self, ct_in: Polynomial) -> Polynomial:
    return self._herot.rotate(ct_in)

  def __call__(self, ct_in: Polynomial) -> Polynomial:
    return self.rotate(ct_in)


class HERescaleOp:
  """Rescale from src_level to dst_level.

  Drops (src_level - dst_level) * composite_degree moduli total.
  """

  def __init__(self, cache: HEParameterCache, src_level: int, dst_level: int):
    assert (
        src_level > dst_level >= 0
    ), f'src_level ({src_level}) must be > dst_level ({dst_level}) >= 0'
    assert (
        src_level <= cache.max_level
    ), f'src_level ({src_level}) > max_level ({cache.max_level})'
    self.cache = cache
    self.src_level = src_level
    self.dst_level = dst_level
    self.num_steps = src_level - dst_level
    total_drop = self.num_steps * cache.composite_degree

    self._num_q_src = cache.num_q_at_level(src_level)
    self._q_at_src = cache.q_towers[: self._num_q_src]
    self._total_drop = total_drop

    # Pre-create HERescale (reusable — no internal mutation)
    self._he_rescale = HERescale(
        batch=cache.batch,
        num_elements=2,
        moduli=self._q_at_src,
        r=cache.r,
        c=cache.c,
        degree_layout=cache.degree_layout,
    )
    self._he_rescale.control_gen(composite_degree=total_drop)

    # Pre-compute output Polynomial template
    self._output_moduli = self._he_rescale.output_moduli
    self._output_shapes = {
        'batch': cache.batch,
        'num_elements': 2,
        'degree': cache.degree,
        'num_moduli': len(self._output_moduli),
        'precision': 32,
        'degree_layout': cache.degree_layout,
    }
    self._output_params = {
        'moduli': self._output_moduli,
        'r': cache.r,
        'c': cache.c,
    }

  def rescale(self, ct_in: Polynomial) -> Polynomial:
    in_data = ct_in.polynomial.reshape(
        -1, ct_in.num_elements, *self.cache.degree_layout, self._num_q_src
    )
    rescaled = self._he_rescale.rescale(in_data)
    ct_out = Polynomial(self._output_shapes, self._output_params)
    ct_out.set_batch_polynomial(rescaled)
    return ct_out

  def __call__(self, ct_in: Polynomial) -> Polynomial:
    return self.rescale(ct_in)


class HEPtCtMulAtLevel:
  """Polynomial-Plaintext multiply at a specific level.

  Wraps HEPtCtMul with level-specific moduli.
  The plaintext must be pre-encoded and set before calling mul().

  Usage (VPU, default):
      op = ctx.ptct_mul[level]
      op.set_plaintext(pt_ntt)           # set plaintext in NTT form
      result = op.mul(ct)                # element-wise modmul on VPU

  Opt-in MXU path (BAT) — only beneficial when the surrounding workload
  is MXU-bound; for standalone ct*pt the VPU path is faster:
      op.precompute_bat(pt_ntt)
      result = op.mul(ct, use_bat=True)
  """

  def __init__(self, cache: HEParameterCache, level: int):
    assert 0 <= level <= cache.max_level
    self.cache = cache
    self.level = level
    q_at_level = cache.q_moduli_at_level(level)
    self._ptct = HEPtCtMul(
        batch=cache.batch,
        r=cache.r,
        c=cache.c,
        moduli=q_at_level,
        degree_layout=cache.degree_layout,
    )

  def set_plaintext(self, pt_ntt: jnp.ndarray):
    """Set plaintext polynomial in NTT domain.

    Args:
        pt_ntt: shape (*degree_layout, num_moduli_at_level) or broadcastable.
    """
    self._ptct.set_plaintext(pt_ntt)

  def precompute_bat(self, pt_ntt: Optional[jnp.ndarray] = None):
    """Precompute BAT representation of plaintext for MXU-accelerated path.

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
          'No plaintext set. Provide pt_ntt or call set_plaintext first.'
      )

  def mul(self, ct: Polynomial, use_bat: bool = False) -> Polynomial:
    """Multiply ciphertext by the set plaintext.

    Default is the VPU (element-wise modmul) path; BAT is opt-in.

    Args:
        ct: Input ciphertext at this level.
        use_bat: If True, use the MXU/BAT path (requires precompute_bat()).
          Default False — VPU is faster for standalone ct*pt at the ring
          dimensions used here.
    """
    return self._ptct.mul(ct, use_bat=use_bat)

  def __call__(self, ct: Polynomial, use_bat: bool = False) -> Polynomial:
    return self.mul(ct, use_bat=use_bat)


class HEPtCtMulAccessor:
  """Provides ctx.ptct_mul[level] indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, level: int) -> HEPtCtMulAtLevel:
    if level not in self._instances:
      self._instances[level] = HEPtCtMulAtLevel(self.cache, level)
    return self._instances[level]


class HEMulAccessor:
  """Provides ctx.he_mul[level] indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, level: int) -> HEMulAtLevel:
    if level not in self._instances:
      self._instances[level] = HEMulAtLevel(self.cache, level)
    return self._instances[level]


class HERotAccessor:
  """Provides ctx.he_rot[level, rot_index] indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, key: Tuple[int, int]) -> HERotAtLevel:
    level, rot_index = key
    if key not in self._instances:
      self._instances[key] = HERotAtLevel(self.cache, level, rot_index)
    return self._instances[key]


class HEBsgsMatVecAtLevel:
  """BSGS matrix-vector multiply at a specific input level.

  Wraps `bsgs.BSGSMatVec` so callers can do:
      mv = ctx.bsgs_matvec[level, n]        # or (level, n, n1, n2)
      mv.encode_matrix(W)
      y_ct = mv.mul(ct_in)                  # one level consumed

  Rotation keys (baby at input level, giant at output level) must have been
  registered in `total_rotation_indices` at `program_initialization` — use
  `bsgs.BSGSMatVec.required_rotation_indices(n, n1, n2)` to get the list.
  """

  def __init__(self, ctx, level: int, n: int,
               n1: Optional[int] = None, n2: Optional[int] = None):
    import bsgs as _bsgs  # lazy to avoid circular import at module load
    self._bsgs = _bsgs.BSGSMatVec(ctx, level, n, n1, n2)

  @property
  def n1(self) -> int:
    return self._bsgs.n1

  @property
  def n2(self) -> int:
    return self._bsgs.n2

  def encode_matrix(self, matrix, pt_scale: Optional[int] = None,
                    bsgs_ratio: Optional[float] = None) -> None:
    self._bsgs.encode_matrix(matrix, pt_scale=pt_scale, bsgs_ratio=bsgs_ratio)

  def mul(self, ct_in: Polynomial) -> Polynomial:
    return self._bsgs.mul(ct_in)

  def __call__(self, ct_in: Polynomial) -> Polynomial:
    return self.mul(ct_in)

  def serializable_state(self) -> dict:
    return self._bsgs.serializable_state()

  def load_serialized_state(self, state: dict) -> None:
    self._bsgs.load_serialized_state(state)


class HEBsgsMatVecAccessor:
  """Provides ctx.bsgs_matvec[level, n] (or [level, n, n1, n2]) indexing."""

  def __init__(self, ctx):
    self.ctx = ctx
    self._instances = {}

  def __getitem__(self, key) -> HEBsgsMatVecAtLevel:
    # Accept (level, n) or (level, n, n1, n2).
    if not isinstance(key, tuple) or len(key) not in (2, 4):
      raise TypeError(
          'Index with (level, n) or (level, n, n1, n2); got {!r}'.format(key))
    if key not in self._instances:
      if len(key) == 2:
        level, n = key
        self._instances[key] = HEBsgsMatVecAtLevel(self.ctx, level, n)
      else:
        level, n, n1, n2 = key
        self._instances[key] = HEBsgsMatVecAtLevel(self.ctx, level, n, n1, n2)
    return self._instances[key]


class HERescaleAccessor:
  """Provides ctx.he_rescale[src_level, dst_level] indexing syntax."""

  def __init__(self, cache: HEParameterCache):
    self.cache = cache
    self._instances = {}

  def __getitem__(self, key: Tuple[int, int]) -> HERescaleOp:
    src, dst = key
    if key not in self._instances:
      self._instances[key] = HERescaleOp(self.cache, src, dst)
    return self._instances[key]

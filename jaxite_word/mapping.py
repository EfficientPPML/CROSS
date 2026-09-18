"""Schedule one packed workload onto a CKKS context.

This is the plan. Given the immutable :class:`packing.Packing` that
:func:`packing.pack` returns, it
computes, before any ciphertext exists, every decision execution cannot make for
itself: each value's level, scale and noise-scale-degree; the BSGS factorization
and the exact set of rotation-key indices the program will use; the bootstrap
depth; virtual ciphertext buffers; and the partition into fusible regions. It
then binds each packed operation to a CKKSContext primitive or operator, lowers
the result through JAX, and runs it.

Homomorphic execution has no feedback loop -- the runtime can never inspect a
ciphertext -- so every one of those decisions has to be derived ahead of time or
not at all. ``Mapping`` is where that derivation lives.

Layering: this module depends on ``packing`` for the packed program, and on
``ckks_ctx`` for the primitives and operators it binds against. It does not
import the frontend at all. Nothing depends on it except callers.
"""

from __future__ import annotations

import contextlib
import hashlib
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import math
import sys
from typing import Any, ClassVar, Iterator, List, Optional

import jax.numpy as jnp
import numpy as np

import key_gen as kg
import polynomial as poly
from he_params import HEParameterCache, normalize_noise_scale_degree

if __package__:
  sys.modules.setdefault('mapping', sys.modules[__name__])
  from . import packing as packing_ir
  from .ckks_ctx import (
      CKKSContext, sigma, _context_fingerprint, _max_level, _moduli_at_level,
      _normalize_noise_std, _validate_context,
      _balanced_degree_layout, _json_digest as _ctx_json_digest,
  )
else:
  sys.modules.setdefault('jaxite_word.mapping', sys.modules[__name__])
  import packing as packing_ir
  from ckks_ctx import (
      CKKSContext, sigma, _context_fingerprint, _max_level, _moduli_at_level,
      _normalize_noise_std, _validate_context,
      _balanced_degree_layout, _json_digest as _ctx_json_digest,
  )


_BINARY_KINDS = frozenset(('add', 'sub', 'mul'))


_PLAIN_KINDS = frozenset(('add_plain', 'mul_plain'))


_SUPPORTED_KINDS = frozenset((
    'add',
    'sub',
    'mul',
    'square',
    'rotate',
    'rescale',
    'level_reduce',
    'add_plain',
    'mul_plain',
    'matvec',
    'bootstrap',
))


def _aot_pmap_input(payload, device_count: int):
  """Normalize automatic-mesh arrays before an AOT pmap call.

  JAX 0.9 cannot reshard an array carrying an Auto ``NamedSharding`` into an
  AOT pmap executable: its call path exposes an internal ``UnspecifiedValue``
  instead.  Supplying the same payload at the host boundary lets the compiled
  executable apply its required leading-device sharding.  Single-device calls
  keep their existing device array and avoid the transfer.
  """
  if device_count == 1:
    return payload
  return np.asarray(payload)


def _device_prefix_for_batch(global_batch: int, available_devices):
  """Choose the widest deterministic data-parallel placement for a batch.

  ``pmap`` requires an equal local batch on every participating device.  Use
  the largest prefix of the visible devices whose size divides the public
  batch, so a batch of 32 or 64 uses all eight devices on a TPUv6e-8 while a
  smaller batch never creates empty shards.  Falling back to one device keeps
  prime batch sizes usable when no wider equal partition exists.
  """
  devices = tuple(available_devices)
  if not devices:
    raise RuntimeError('JAX reported no local devices.')
  widest = min(int(global_batch), len(devices))
  for device_count in range(widest, 1, -1):
    if int(global_batch) % device_count == 0:
      return devices[:device_count]
  return devices[:1]


def ring_runtime_parameters(ring_config, *, keys=None) -> dict:
  """CKKSContext parameters for a ring config, exactly as validated.

  ``runtime_kwargs`` carries the security-relevant fields. Two more are
  runtime-only and have no bearing on the security plan: the output scale,
  which starts at the ring's scaling factor, and the degree layout, which is
  how the coefficient axis is tiled for the accelerator.
  """
  runtime = ring_config.runtime_kwargs
  kwargs = dict(runtime() if callable(runtime) else runtime)
  degree = int(kwargs['degree'])
  # The dense two-step NTT stores square row/column transform controls.
  # Keep those dimensions balanced: a legacy (4, degree/4) layout makes the
  # larger control quadratic in the full ring degree (for N=32768, an
  # 8192x8192 matrix per modulus) and exhausts TPU HBM during demo setup.
  rows, columns = _balanced_degree_layout(degree)
  q_towers = [int(value) for value in kwargs['q_towers']]
  p_towers = [int(value) for value in kwargs['p_towers']]
  sigma = float(kwargs['sigma'])
  if keys is None:
    # Freshly generated for this ring, at the error width the security plan
    # was validated against -- generating at a different sigma would mean the
    # analysis does not describe the keys in use. Cost grows steeply with the
    # degree, so a caller running several Mappings over one ring should
    # generate once and pass the pair in.
    keys = kg.gen_pke_pair(q_towers, p_towers, degree, noise_std=sigma)
  else:
    missing = [
        name for name in ('public_key', 'secret_key') if name not in keys
    ]
    if missing:
      raise ValueError(
          f'keys must contain {missing}; got {sorted(keys)}.'
      )
  return {
      'degree': degree,
      'num_slots': int(kwargs['num_slots']),
      'q_towers': q_towers,
      'p_towers': p_towers,
      'composite_degree': int(kwargs['composite_degree']),
      'scaling_factor': float(kwargs['scaling_factor']),
      'output_scale': float(kwargs['scaling_factor']),
      'degree_layout': (rows, columns),
      'public_key': keys['public_key'],
      'secret_key': keys['secret_key'],
  }


def _operation_id(operation) -> str:
  return operation[0]


def _operation_kind(operation) -> str:
  return operation[1]


def _operation_inputs(operation) -> tuple[str, ...]:
  return operation[2]


def _operation_argument(operation):
  return operation[3]


def _default_matvec_plaintext_scale(ctx, source) -> float:
  """Return the evaluator's implicit MatVec scale at one logical level.

  Composite scaling follows the ciphertext's tracked recursive scale rather
  than replacing it with a canonical level value. CD1 retains its historical
  last-limb default. Explicit ``packing_ir.Linear.pt_scale`` values bypass this helper
  in ``_plan_bsgs``.

  ``source.scale`` is never ``None`` here: ``_analyze_mapping`` seeds the input
  spec through ``_checked_scale`` and ``_replace_spec`` only propagates ``None``
  from a ``None`` source, so every ``_plan_step`` operand carries a tracked
  scale.
  """
  if int(ctx.composite_degree) >= 2:
    return float(source.scale)
  return float(source.moduli[-1])


@dataclass(frozen=True)
class _ValueSpec:
  """Compile-time logical and physical metadata for one ciphertext value."""

  level: int
  batch: int
  degree_layout: tuple[int, int]
  moduli: tuple[int, ...]
  shape: tuple[int, ...]
  packing: str
  scale: Optional[float]
  nsd: Optional[int]
  elements: int = 2
  dtype: str = 'uint32'

  @property
  def num_moduli(self) -> int:
    return len(self.moduli)

  @property
  def payload_shape(self) -> tuple[int, int, int, int, int]:
    r, c = self.degree_layout
    return (self.batch, self.elements, r, c, self.num_moduli)


@dataclass(frozen=True)
class _MatVecPlan:
  """Minimal Mapping-owned schedule metadata for one BSGS invocation."""

  n: int
  n1: int
  active_diagonal_indices: tuple[int, ...]
  pt_scale: float

  @property
  def n2(self) -> int:
    return self.n // self.n1

  @property
  def required_rotation_indices(self) -> tuple[int, ...]:
    baby = {
        index % self.n1
        for index in self.active_diagonal_indices
        if index % self.n1
    }
    giant = {
        (index // self.n1) * self.n1
        for index in self.active_diagonal_indices
        if index // self.n1
    }
    # The current scan kernel needs one representative giant operator when
    # every active diagonal belongs to the identity giant step.
    if not giant:
      giant.add(self.n1)
    return tuple(sorted(baby | giant))


@dataclass(frozen=True)
class _ZeroMatVecPlan:
  """Explicit level/scale/layout transition for an identically zero matrix."""

  n: int
  pt_scale: float

  @property
  def active_diagonal_indices(self) -> tuple[int, ...]:
    return ()

  @property
  def required_rotation_indices(self) -> tuple[int, ...]:
    return ()


@dataclass(frozen=True)
class _BootstrapPlan:
  """Minimal Mapping metadata derived from the current Bootstrap API."""

  level_budget: tuple[int, int]
  secret_key_dist: str
  expected_output_level: int
  required_rotation_indices: tuple[int, ...]


def _same_layout(left: _ValueSpec, right: _ValueSpec, step_id: str) -> None:
  attributes = ('level', 'batch', 'degree_layout', 'moduli', 'shape', 'packing')
  mismatched = [
      name for name in attributes if getattr(left, name) != getattr(right, name)
  ]
  if mismatched:
    raise ValueError(
        f'step {step_id!r} inputs disagree on {mismatched}; alignment must be '
        'explicit in the model.'
    )


def _same_additive_metadata(
    left: _ValueSpec, right: _ValueSpec, step_id: str
) -> None:
  if (
      left.scale is not None
      and right.scale is not None
      and not math.isclose(
          left.scale, right.scale, rel_tol=1e-12, abs_tol=0.0
      )
  ):
    raise ValueError(
        f'step {step_id!r} inputs have incompatible scales '
        f'{left.scale} and {right.scale}.'
    )
  if (left.scale is None) != (right.scale is None):
    raise ValueError(
        f'step {step_id!r} inputs have incompatible known/unknown scales.'
    )
  if left.nsd is not None and right.nsd is not None and left.nsd != right.nsd:
    raise ValueError(
        f'step {step_id!r} inputs have incompatible nsd '
        f'{left.nsd} and {right.nsd}.'
    )
  if (left.nsd is None) != (right.nsd is None):
    raise ValueError(
        f'step {step_id!r} inputs have incompatible known/unknown nsd.'
    )


def _checked_scale(value: float, step_id: str, operation: str) -> float:
  value = float(value)
  if not math.isfinite(value) or value <= 0:
    raise ValueError(
        f'step {step_id!r} {operation} produces an invalid scale {value!r}.'
    )
  return value


def _rescale_divisor(source: _ValueSpec, ctx, steps: int) -> int:
  return math.prod(source.moduli[-steps * ctx.composite_degree:])


def _after_rescale_nsd(nsd: Optional[int], steps: int) -> Optional[int]:
  if nsd is None:
    return None
  return max(1, nsd - steps)


def _multiply_scale(
    left: _ValueSpec,
    right: _ValueSpec,
    ctx,
    step_id: str,
) -> Optional[float]:
  if left.scale is None or right.scale is None:
    return None
  divisor = _rescale_divisor(left, ctx, 1)
  if ctx.composite_degree == 1:
    result = (left.scale / divisor) * (right.scale / divisor)
  else:
    result = (left.scale / divisor) * right.scale
  return _checked_scale(result, step_id, 'multiply')


def _replace_spec(
    source: _ValueSpec,
    ctx,
    *,
    level: Optional[int] = None,
    shape: Optional[tuple[int, ...]] = None,
    packing: Optional[str] = None,
    scale: Any = ...,
    nsd: Any = ...,
) -> _ValueSpec:
  level = source.level if level is None else level
  return _ValueSpec(
      level=level,
      batch=source.batch,
      degree_layout=source.degree_layout,
      moduli=_moduli_at_level(ctx, level),
      shape=source.shape if shape is None else tuple(shape),
      packing=source.packing if packing is None else packing,
      scale=source.scale if scale is ... else scale,
      nsd=source.nsd if nsd is ... else nsd,
      elements=source.elements,
      dtype=source.dtype,
  )


def _plan_bsgs(constant, options, dimension, default_pt_scale):
  import bsgs

  n1, n2, bsgs_ratio, requested_pt_scale = options
  pt_scale = (
      float(default_pt_scale)
      if requested_pt_scale is None
      else float(requested_pt_scale)
  )
  if not math.isfinite(pt_scale) or pt_scale <= 0:
    raise ValueError('MatVec plaintext scale must be finite and positive.')
  if (n1 is None) != (n2 is None):
    raise ValueError('MatVec n1 and n2 must both be supplied or both omitted.')
  explicit_factors = n1 is not None
  if explicit_factors:
    if (
        isinstance(n1, bool)
        or isinstance(n2, bool)
        or not isinstance(n1, (int, np.integer))
        or not isinstance(n2, (int, np.integer))
        or int(n1) <= 0
        or int(n2) <= 0
    ):
      raise ValueError('MatVec n1 and n2 must be positive ints.')
    n1, n2 = int(n1), int(n2)
    if n1 * n2 != dimension:
      raise ValueError(
          f'MatVec n1 * n2 ({n1}*{n2}) != dimension ({dimension}).'
      )
  if bsgs_ratio is not None:
    if isinstance(bsgs_ratio, bool):
      raise TypeError('MatVec bsgs_ratio must be a real number.')
    bsgs_ratio = float(bsgs_ratio)
    if not math.isfinite(bsgs_ratio) or bsgs_ratio <= 0:
      raise ValueError('MatVec bsgs_ratio must be finite and positive.')
  maxima: dict[int, float] = {}
  constant_kind = getattr(constant, '_constant_kind', None)
  if constant_kind in ('sparse', 'lazy_sparse'):
    diagonal_maxima = getattr(constant, 'diagonal_maxima', None)
    if diagonal_maxima is not None:
      raw_maxima = diagonal_maxima()
      items = (
          raw_maxima.items()
          if hasattr(raw_maxima, 'items')
          else raw_maxima
      )
      for index, peak in items:
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or not 0 <= int(index) < dimension
        ):
          raise ValueError(
              'MatVec diagonal indices must be ints in '
              f'[0, {dimension}).'
          )
        peak = float(peak)
        if not math.isfinite(peak) or peak < 0:
          raise ValueError(
              'MatVec diagonal maxima must be finite and non-negative.'
          )
        maxima[int(index)] = peak
    elif constant_kind == 'sparse':
      maxima = {
          index: float(np.max(np.abs(diagonal)))
          for index, diagonal in constant.as_dict().items()
          if diagonal.size
      }
    else:
      raise TypeError(
          'lazy sparse MatVec constants must expose diagonal_maxima().'
      )
  else:
    rows, columns = np.nonzero(np.abs(constant.values) > 0)
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
      index = (column - row) % dimension
      maxima[index] = max(
          maxima.get(index, 0.0), float(abs(constant.values[row, column]))
      )
  if not maxima or not any(maxima.values()):
    plan = _ZeroMatVecPlan(dimension, pt_scale)
    return plan, plan.required_rotation_indices
  encode_floor = 0.5 / pt_scale
  subresolution = tuple(sorted(
      index
      for index, peak in maxima.items()
      if 0.0 < peak < encode_floor
  ))
  if subresolution:
    raise ValueError(
        f'{len(subresolution)} non-zero MatVec diagonal(s) are below the '
        f'plaintext encode floor {encode_floor:.3e} at pt_scale={pt_scale}. '
        'Refusing to drop non-zero coefficients; pick a larger pt_scale / '
        'scaling factor.'
    )
  active = tuple(sorted(
      index for index, peak in maxima.items() if peak >= encode_floor
  ))
  if not active:
    raise ValueError(
        'all MatVec diagonals are below the plaintext encode floor at '
        f'pt_scale={pt_scale}.'
    )
  if not explicit_factors:
    n1, n2 = bsgs.compute_bsgs_params(dimension)
  if bsgs_ratio is not None:
    adaptive = bsgs.compute_bsgs_params(
        dimension, num_diagonals=len(active), bsgs_ratio=bsgs_ratio
    )
    if explicit_factors and (n1, n2) != adaptive:
      raise ValueError(
          f'MatVec explicit factors {(n1, n2)} do not match adaptive '
          f'factors {adaptive}.'
      )
    n1, n2 = adaptive
  plan = _MatVecPlan(dimension, n1, active, pt_scale)
  return plan, plan.required_rotation_indices


def _plan_bootstrap(ctx, argument):
  import bootstrapping

  level_budget, secret_key_dist, _, _ = argument
  level_budget = tuple(level_budget)
  if secret_key_dist not in ('uniform_ternary', 'sparse_ternary'):
    raise ValueError(
        "secret_key_dist must be 'uniform_ternary' or 'sparse_ternary'."
    )
  maximum_budget = max(1, int(math.log2(ctx.num_slots)))
  if any(level > maximum_budget for level in level_budget):
    raise ValueError(
        f'level_budget={level_budget} exceeds log2(num_slots)='
        f'{maximum_budget}.'
    )
  depth = bootstrapping.Bootstrap.compute_bootstrap_nq(
      ctx.composite_degree,
      level_budget=level_budget,
      secret_key_dist=secret_key_dist,
  )['boot_depth']
  return _BootstrapPlan(
      level_budget=level_budget,
      secret_key_dist=secret_key_dist,
      expected_output_level=_max_level(ctx) - depth,
      required_rotation_indices=(
          bootstrapping.Bootstrap.required_rotation_indices(
              ctx.degree, ctx.num_slots, level_budget
          )
      ),
  )


def _plan_step(step, inputs, ctx):
  source = inputs[0]
  step_id = _operation_id(step)
  kind = _operation_kind(step)
  argument = _operation_argument(step)
  rotations = []
  operation_plan = None

  if kind in _BINARY_KINDS:
    _same_layout(inputs[0], inputs[1], step_id)
  if kind in ('add', 'sub'):
    _same_additive_metadata(inputs[0], inputs[1], step_id)
    return source, operation_plan, rotations
  if kind == 'mul':
    if any(value.nsd != 1 for value in inputs):
      raise ValueError(
          f'operator {step_id!r} multiply requires both inputs at nsd=1; '
          'insert an explicit Rescale after MulPlain or other depth-increasing '
          'operations.'
      )
    if source.level < 1:
      raise ValueError(f'operator {step_id!r} multiply underflows level 0.')
    return _replace_spec(
        source,
        ctx,
        level=source.level - 1,
        scale=_multiply_scale(inputs[0], inputs[1], ctx, step_id),
        nsd=1,
    ), operation_plan, rotations
  if kind == 'square':
    if source.nsd != 1:
      raise ValueError(
          f'operator {step_id!r} square requires input nsd=1; insert an explicit '
          'Rescale after MulPlain or other depth-increasing operations.'
      )
    if source.level < 1:
      raise ValueError(f'operator {step_id!r} square underflows level 0.')
    return _replace_spec(
        source,
        ctx,
        level=source.level - 1,
        scale=_multiply_scale(source, source, ctx, step_id),
        nsd=1,
    ), operation_plan, rotations
  if kind == 'rotate':
    index = int(argument) % ctx.num_slots
    if index:
      rotations.append(index)
    return source, index, rotations
  if kind == 'rescale':
    steps = int(argument)
    output_level = source.level - steps
    if output_level < 0:
      raise ValueError(
          f'operator {step_id!r} rescale underflows level {source.level} by '
          f'{steps}.'
      )
    output_scale = source.scale
    if output_scale is not None:
      output_scale = _checked_scale(
          output_scale / _rescale_divisor(source, ctx, steps),
          step_id,
          'rescale',
      )
    return _replace_spec(
        source,
        ctx,
        level=output_level,
        scale=output_scale,
        nsd=_after_rescale_nsd(source.nsd, steps),
    ), operation_plan, rotations
  if kind == 'level_reduce':
    steps = int(argument)
    output_level = source.level - steps
    if output_level < 0:
      raise ValueError(
          f'operator {step_id!r} level_reduce underflows level '
          f'{source.level} by {steps}.'
      )
    # Truncating the CRT representation leaves the represented value alone,
    # so unlike rescale this touches neither the scale nor the noise-scale
    # degree. That is the whole point: it lets a value meet another at a
    # shared level and still be added to it.
    return _replace_spec(
        source,
        ctx,
        level=output_level,
        scale=source.scale,
        nsd=source.nsd,
    ), operation_plan, rotations
  if kind in _PLAIN_KINDS:
    constant = argument
    if getattr(constant, '_constant_kind', None) != 'plain':
      raise TypeError(
          f'operator {step_id!r} requires an packing_ir.PlainSlots constant.'
      )
    if constant.shape != source.shape or constant.packing != source.packing:
      raise ValueError(
          f'operator {step_id!r} plaintext layout '
          f'{constant.shape}/{constant.packing!r} does not match input '
          f'{source.shape}/{source.packing!r}.'
      )
    if constant.values.size != ctx.num_slots:
      raise ValueError(
          f'operator {step_id!r} plaintext constant must contain exactly '
          f'{ctx.num_slots} physical slots, got {constant.values.size}.'
      )
    encoding_scale = source.scale if constant.scale is None else constant.scale
    if encoding_scale is None:
      encoding_scale = ctx.scaling_factor
    encoding_scale = _checked_scale(
        encoding_scale, step_id, 'plaintext encoding'
    )
    if (
        kind == 'add_plain'
        and source.scale is not None
        and not math.isclose(
            source.scale, encoding_scale, rel_tol=1e-12, abs_tol=0.0
        )
    ):
      raise ValueError(
          f'operator {step_id!r} AddPlain scale {encoding_scale} does not match '
          f'ciphertext scale {source.scale}.'
      )
    output = source
    if kind == 'mul_plain':
      output = _replace_spec(
          source,
          ctx,
          scale=(
              None
              if source.scale is None
              else _checked_scale(
                  source.scale * encoding_scale, step_id, 'MulPlain'
              )
          ),
          nsd=None if source.nsd is None else source.nsd + 1,
      )
    return output, float(encoding_scale), rotations
  if kind == 'matvec':
    constant, n1, n2, bsgs_ratio, pt_scale = argument
    constant_kind = getattr(constant, '_constant_kind', None)
    if constant_kind not in ('dense', 'sparse', 'lazy_sparse'):
      raise TypeError(
          f'operator {step_id!r} requires a packing matrix constant.'
      )
    if constant.input_shape != source.shape or constant.packing != source.packing:
      raise ValueError(
          f'operator {step_id!r} matrix input layout does not match its value.'
      )
    dimension = (
        constant.dimension
        if constant_kind in ('sparse', 'lazy_sparse')
        else ctx.num_slots
    )
    if dimension != ctx.num_slots:
      raise ValueError(
          f'operator {step_id!r} BSGS dimension {dimension} must equal the '
          f'context slot count {ctx.num_slots}.'
      )
    if source.level < 1:
      raise ValueError(f'operator {step_id!r} MatVec underflows level 0.')
    operation_plan, bsgs_rotations = _plan_bsgs(
        constant,
        (n1, n2, bsgs_ratio, pt_scale),
        dimension,
        _default_matvec_plaintext_scale(ctx, source),
    )
    rotations.extend(bsgs_rotations)
    output_scale = source.scale
    if output_scale is not None:
      output_scale = _checked_scale(
          (output_scale / _rescale_divisor(source, ctx, 1))
          * operation_plan.pt_scale,
          step_id,
          'MatVec',
      )
    return _replace_spec(
        source,
        ctx,
        level=source.level - 1,
        shape=constant.output_shape,
        packing=constant.output_packing,
        scale=output_scale,
        nsd=source.nsd,
    ), operation_plan, rotations
  if kind == 'bootstrap':
    level_budget, _, mode, precision = argument
    if source.scale is None:
      raise ValueError(
          f'operator {step_id!r} bootstrap requires a known input scale.'
      )
    if source.nsd != 1:
      raise ValueError(
          f'operator {step_id!r} bootstrap requires input nsd=1; insert an '
          'explicit Rescale after MulPlain or other depth-increasing '
          'operations.'
      )
    operation_plan = _plan_bootstrap(ctx, argument)
    output_level = operation_plan.expected_output_level
    if output_level < 0:
      raise ValueError(
          f'operator {step_id!r} bootstrap needs more levels than the context.'
      )
    # The engine resets bootstrap tracking to FLEXIBLEAUTO sf(0), which is the
    # product of the final composite group in the complete Q chain. Context
    # ``output_scale`` is only a decode fallback and is not an evaluator
    # contract.
    output_scale = _checked_scale(
        math.prod(ctx.q_towers[-ctx.composite_degree:]),
        step_id,
        'bootstrap',
    )
    if mode == 'meta':
      if output_level <= source.level:
        raise ValueError(
            f'operator {step_id!r} meta bootstrap requires post-bootstrap '
            f'headroom above input level {source.level}, but planned output '
            f'is {output_level}.'
        )
      try:
        output_scale = math.ldexp(output_scale, precision)
      except OverflowError as error:
        raise ValueError(
            f'operator {step_id!r} meta bootstrap precision overflows scale.'
        ) from error
      output_scale = _checked_scale(
          output_scale, step_id, 'meta bootstrap'
      )
    rotations.extend(operation_plan.required_rotation_indices)
    return _replace_spec(
        source,
        ctx,
        level=output_level,
        scale=output_scale,
        nsd=1,
    ), operation_plan, rotations
  raise AssertionError(f'unhandled primitive {kind!r}')


def _fusion_regions(operations, compile_mode):
  if compile_mode == 'operations':
    return tuple(
        (
            _operation_kind(operation) != 'bootstrap',
            (_operation_id(operation),),
        )
        for operation in operations
    )
  regions = []
  current = []
  for operation in operations:
    operation_id = _operation_id(operation)
    if _operation_kind(operation) == 'bootstrap':
      if current:
        regions.append((True, tuple(current)))
        current = []
      regions.append((False, (operation_id,)))
    else:
      current.append(operation_id)
  if current:
    regions.append((True, tuple(current)))
  return tuple(regions)


def _memory_schedule(packing, value_specs):
  """Assign deterministic virtual buffers from dependency liveness.

  These are Mapping-level ciphertext buffer identities. The JAX backend may
  further alias or place physical device buffers while preserving dependency
  and bootstrap-barrier semantics.
  """
  operations = packing.operations
  final_index = len(operations)
  birth = {'input': -1}
  last_use = {'input': -1}
  for index, operation in enumerate(operations):
    operation_id = _operation_id(operation)
    birth[operation_id] = index
    last_use.setdefault(operation_id, index)
    for source in _operation_inputs(operation):
      last_use[source] = max(last_use.get(source, -1), index)
  last_use[packing.output] = final_index

  free_buffers = []
  active = {}
  next_buffer = 0
  assigned = {}

  def allocate() -> int:
    nonlocal next_buffer
    if free_buffers:
      return free_buffers.pop(0)
    buffer_id = next_buffer
    next_buffer += 1
    return buffer_id

  assigned['input'] = allocate()
  active['input'] = assigned['input']
  for index, operation in enumerate(operations):
    operation_id = _operation_id(operation)
    # Inputs used by this operation remain live until after its output has
    # received a distinct virtual buffer.
    assigned[operation_id] = allocate()
    active[operation_id] = assigned[operation_id]
    expired = sorted(
        value_id
        for value_id in active
        if value_id != operation_id and last_use[value_id] == index
    )
    for value_id in expired:
      free_buffers.append(active.pop(value_id))
    free_buffers.sort()

  return tuple(
      (
          value_id,
          assigned[value_id],
          birth[value_id],
          last_use[value_id],
          value_specs[value_id].payload_shape,
      )
      for value_id in ('input',) + tuple(
          _operation_id(operation) for operation in operations
      )
  )


_LOW_LEVEL_PRIMITIVE = {
    'add': 'he_add',
    'sub': 'he_sub',
    'mul': 'he_mul',
    'square': 'he_mul.square',
    'rotate': 'he_rot',
    'rescale': 'he_rescale',
    'level_reduce': 'he_level_reduce',
    'add_plain': 'he_add.plain',
    'mul_plain': 'ptct_mul',
    'matvec': 'bsgs_matvec',
    'bootstrap': 'he_bootstrap',
}


def _compute_schedule(
    packing, value_specs, memory_trajectory, operation_plans
):
  buffer_by_value = {
      value_id: buffer_id
      for value_id, buffer_id, _, _, _ in memory_trajectory
  }

  def primitive(operation):
    operation_id = _operation_id(operation)
    if isinstance(operation_plans[operation_id], _ZeroMatVecPlan):
      return 'zero_matvec'
    return _LOW_LEVEL_PRIMITIVE[_operation_kind(operation)]

  return tuple(
      (
          index,
          primitive(operation),
          _operation_inputs(operation),
          _operation_id(operation),
          tuple(
              value_specs[source].level
              for source in _operation_inputs(operation)
          ),
          value_specs[_operation_id(operation)].level,
          tuple(
              buffer_by_value[source]
              for source in _operation_inputs(operation)
          ),
          buffer_by_value[_operation_id(operation)],
          operation_plans[_operation_id(operation)],
          _operation_kind(operation) == 'bootstrap',
      )
      for index, operation in enumerate(packing.operations)
  )


def _analyze_mapping(mapping, *, perf_test: bool = False) -> None:
  """Populate one Mapping's complete schedule before exact key setup."""
  if not isinstance(perf_test, bool):
    raise TypeError('perf_test must be a bool.')
  ctx = mapping.ctx
  packing = mapping.packing
  _validate_context(ctx)
  required = (
      'input_shape', 'input_packing', 'operations', 'output', 'fingerprint',
  )
  if any(not hasattr(packing, name) for name in required):
    raise TypeError('Mapping requires a complete Packing value.')
  shape = tuple(packing.input_shape)
  if not shape or any(
      isinstance(dimension, bool)
      or not isinstance(dimension, int)
      or dimension <= 0
      for dimension in shape
  ):
    raise ValueError('model input shape must contain positive ints.')
  if math.prod(shape) > ctx.num_slots:
    raise ValueError(
        f'input shape {shape} needs {math.prod(shape)} slots, but context has '
        f'{ctx.num_slots}.'
    )
  scale = (
      ctx.scaling_factor
      if mapping.input_scale is None
      else mapping.input_scale
  )
  scale = _checked_scale(scale, 'input', 'metadata')
  values = {
      'input': _ValueSpec(
          level=_max_level(ctx),
          batch=ctx.batch,
          degree_layout=tuple(ctx.degree_layout),
          moduli=_moduli_at_level(ctx, _max_level(ctx)),
          shape=shape,
          packing=packing.input_packing,
          scale=scale,
          nsd=mapping.input_nsd,
      )
  }
  operation_plans = {}
  rotations = []
  for operation in packing.operations:
    operation_id = _operation_id(operation)
    operation_kind = _operation_kind(operation)
    operation_inputs = _operation_inputs(operation)
    if operation_kind not in _SUPPORTED_KINDS:
      raise ValueError(f'unsupported packed operator {operation_kind!r}.')
    if operation_id in values:
      raise ValueError(f'duplicate value id {operation_id!r}.')
    try:
      inputs = tuple(values[source] for source in operation_inputs)
    except KeyError as error:
      raise ValueError(
          f'operator {operation_id!r} references an unavailable value: '
          f'{error.args[0]!r}.'
      ) from error
    output, operation_plan, operation_rotations = _plan_step(
        operation, inputs, ctx
    )
    values[operation_id] = output
    operation_plans[operation_id] = operation_plan
    # Rotate primitives are canonicalized by ``_plan_step``. BSGS plans are
    # already canonical, while bootstrap additionally advertises the distinct
    # conjugation Galois index (2 * degree - 1), which must not be reduced
    # modulo the slot count.
    rotations.extend(int(index) for index in operation_rotations)
  bootstrap_configurations = {
      (_operation_argument(operation)[0], _operation_argument(operation)[1])
      for operation in packing.operations
      if _operation_kind(operation) == 'bootstrap'
  }
  if len(bootstrap_configurations) > 1:
    raise ValueError(
        'one model cannot use multiple bootstrap configurations.'
    )
  if packing.output not in values:
    raise ValueError(
        f'Packing output references missing value {packing.output!r}.'
    )
  if values[packing.output].level < mapping.headroom:
    raise ValueError(
        f'mapped output reaches level {values[packing.output].level}, below '
        f'required headroom {mapping.headroom}.'
    )
  regions = _fusion_regions(
      packing.operations, compile_mode=mapping.compile_mode
  )
  context_fingerprint = _context_fingerprint(ctx, perf_test=perf_test)
  fingerprint = hashlib.sha256((
      packing.fingerprint
      + context_fingerprint
      + _ctx_json_digest({
          'input_scale': mapping.input_scale,
          'input_nsd': mapping.input_nsd,
          'headroom': mapping.headroom,
          'global_batch': mapping.global_batch,
          'device_count': mapping.device_count,
          'compile_mode': mapping.compile_mode,
          'bsgs_n_jobs': mapping.bsgs_n_jobs,
          'bsgs_streaming': mapping.bsgs_streaming,
      })
      + _ctx_json_digest(tuple(operation_plans.items()))
  ).encode()).hexdigest()
  memory_trajectory = _memory_schedule(packing, values)

  mapping.value_specs = dict(values)
  mapping.operation_plans = dict(operation_plans)
  mapping.required_rotation_indices = tuple(sorted(set(rotations)))
  mapping.regions = tuple(regions)
  mapping.context_fingerprint = context_fingerprint
  mapping.fingerprint = fingerprint
  mapping.perf_test = perf_test
  mapping.memory_trajectory = memory_trajectory
  mapping.compute_trajectory = _compute_schedule(
      packing, values, memory_trajectory, operation_plans
  )


def _apply_metadata(value, spec: _ValueSpec):
  if spec.scale is not None:
    value._ckks_scale = spec.scale
  if spec.nsd is not None:
    value._ckks_nsd = spec.nsd
  return value


def _bind_operation(ctx, mapping, operation, bootstrap_target):
  operation_id = _operation_id(operation)
  kind = _operation_kind(operation)
  inputs = _operation_inputs(operation)
  argument = _operation_argument(operation)
  input_spec = mapping.value_specs[inputs[0]]
  output_spec = mapping.value_specs[operation_id]
  operation_plan = mapping.operation_plans[operation_id]

  if kind == 'add':
    evaluator = ctx.he_add[input_spec.level]
    return evaluator._add_array
  if kind == 'sub':
    evaluator = ctx.he_sub[input_spec.level]
    return evaluator._sub_array
  if kind == 'mul':
    evaluator = ctx.he_mul[output_spec.level]
    return evaluator._mul_array
  if kind == 'square':
    evaluator = ctx.he_mul[output_spec.level]
    return evaluator._square_array
  if kind == 'rotate':
    index = operation_plan
    if index == 0:
      return lambda value: value
    evaluator = ctx.he_rot[input_spec.level, index]
    return evaluator._rotate_array
  if kind == 'rescale':
    evaluator = ctx.he_rescale[input_spec.level, output_spec.level]
    return evaluator._rescale_array
  if kind == 'level_reduce':
    evaluator = ctx.he_level_reduce[input_spec.level, output_spec.level]
    return evaluator._level_reduce_array
  if kind in _PLAIN_KINDS:
    constant = argument
    plaintext = ctx.encode_at_level(
        [complex(value) for value in constant.values],
        input_spec.level,
        scale=operation_plan,
    )
    plaintext_data = ctx._param_cache._prepare_plaintext_payload(
        plaintext.polynomial, input_spec.level
    )
    if kind == 'add_plain':
      evaluator = ctx.he_add[input_spec.level]
      return lambda value, evaluator=evaluator, plaintext_data=plaintext_data: (
          evaluator._add_plain_array(value, plaintext_data)
      )
    evaluator = ctx.ptct_mul[input_spec.level]
    return lambda value, evaluator=evaluator, plaintext_data=plaintext_data: (
        evaluator._mul_array(value, plaintext_data)
    )
  if kind == 'matvec':
    constant, _, _, _, _ = argument
    if isinstance(operation_plan, _ZeroMatVecPlan):
      shape = output_spec.payload_shape
      return lambda value, shape=shape: jnp.zeros(shape, dtype=value.dtype)
    bsgs_plan = operation_plan
    evaluator = ctx.bsgs_matvec[
        input_spec.level,
        bsgs_plan.n,
        bsgs_plan.n1,
        bsgs_plan.n2,
    ]
    if getattr(constant, '_constant_kind', None) == 'dense':
      matrix = constant.values
    elif (
        getattr(constant, '_constant_kind', None) == 'lazy_sparse'
        and mapping.bsgs_streaming
    ):
      matrix = constant
    else:
      matrix = constant.as_dict()
    evaluator.preprocess(
        matrix,
        pt_scale=bsgs_plan.pt_scale,
        n_jobs=mapping.bsgs_n_jobs,
        memory_bounded=mapping.bsgs_streaming,
        active_diagonal_indices=bsgs_plan.active_diagonal_indices,
    )
    return (
        evaluator.matvec
        if mapping.bsgs_streaming
        else evaluator._matvec_array
    )
  if kind == 'bootstrap':
    _, _, mode, precision = argument
    if mode == 'single':
      return bootstrap_target.bootstrap
    return lambda value, target=bootstrap_target, precision=precision: (
        target.meta_bootstrap(value, precision)
    )
  raise AssertionError(f'unhandled primitive {kind!r}')


def _prepare_bindings(ctx, mapping):
  bootstrap_groups = {}
  for operation in mapping.packing.operations:
    if _operation_kind(operation) != 'bootstrap':
      continue
    operation_id = _operation_id(operation)
    level_budget, secret_key_dist, _, _ = _operation_argument(operation)
    spec = mapping.value_specs[_operation_inputs(operation)[0]]
    if spec.scale is None or spec.nsd is None:
      raise ValueError(
          f'bootstrap operator {operation_id!r} requires known scale and nsd.'
      )
    configuration = (level_budget, secret_key_dist)
    bootstrap_groups.setdefault(configuration, []).append(operation)
  if len(bootstrap_groups) > 1:
    raise ValueError(
        'one Mapping cannot use multiple bootstrap configurations.'
    )

  bootstrap_target = None
  if bootstrap_groups:
    configuration, operations = next(iter(bootstrap_groups.items()))
    level_budget, secret_key_dist = configuration
    bootstrap_plan = mapping.operation_plans[_operation_id(operations[0])]
    if any(
        mapping.operation_plans[_operation_id(operation)] != bootstrap_plan
        for operation in operations[1:]
    ):
      raise ValueError('bootstrap operators disagree on their frozen plan.')
    input_specs = tuple(sorted({
        (
            mapping.value_specs[_operation_inputs(operation)[0]].level,
            mapping.value_specs[_operation_inputs(operation)[0]].scale,
            mapping.value_specs[_operation_inputs(operation)[0]].nsd,
        )
        for operation in operations
    }))
    facade = ctx.he_bootstrap
    if getattr(facade, '_configured', False):
      if (
          facade._configuration != (
              tuple(level_budget), secret_key_dist
          )
          or facade._input_specs != input_specs
      ):
        raise RuntimeError(
            'context bootstrap is already configured differently.'
        )
      if not facade._is_setup:
        facade.setup()
    else:
      facade.configure(
          level_budget=list(level_budget),
          secret_key_dist=secret_key_dist,
          input_specs=input_specs,
      ).setup()
    bootstrap_target = getattr(facade, '_engine', facade)

  return {
      _operation_id(operation): _bind_operation(
          ctx, mapping, operation, bootstrap_target
      )
      for operation in mapping.packing.operations
  }


class Mapping:
  """One complete context-specific compute and ciphertext-memory schedule.

  A Mapping is instantiated once for a whole Packing. Construction creates its
  CKKSContext, derives the low-level ``ctx.<op>`` trajectory, allocates virtual
  ciphertext buffers from dependency liveness, prepares exact key metadata,
  fixes an ordered device tuple, and selects the request-time execution mode.
  The public batch is split evenly across those devices; the owned context and
  physical value specifications use the derived per-device batch. When
  ``devices`` is omitted, whole-program bootstrap-free inference automatically
  uses the widest divisible prefix of the visible devices. Individual HE
  operations are trajectory entries, never Mapping objects.

  ``compile_mode='whole'`` binds constants and lowers one fused program during
  construction (the default). Bootstrap-free ``compile_mode='operations'``
  binds, lowers, and executes one operation at a time on each request, then
  releases its constants, keys, evaluator state, and dead intermediates. This
  reduces peak memory for large security-parameter profiles at the cost of
  repeated encoding/compilation and more dispatches. ``bsgs_streaming=True``
  further bounds sparse MatVec memory by encoding one giant step at a time.
  """


  def __init__(
      self,
      packing,
      parameters: Optional[dict] = None,
      *,
      global_batch: int = 1,
      devices=None,
      dnum: Optional[int] = None,
      input_scale: Optional[float] = None,
      input_nsd: Optional[int] = 1,
      headroom: int = 0,
      keys: Optional[dict] = None,
      pregenerated_rotation_keys: Optional[dict] = None,
      cache_rotation_keys: bool = True,
      compile_mode: str = 'whole',
      bsgs_n_jobs: int = 1,
      bsgs_streaming: bool = False,
      perf_test: bool = False,
  ):
    if not isinstance(packing, packing_ir.Packing):
      # By type, not by shape. Duck-typing here would let a hand-assembled
      # object carrying the right attribute names into the scheduler, and the
      # whole point of the restructure is that there is one producer of a
      # packed program.
      raise TypeError(
          f'Mapping takes the packing.Packing that packing.pack returns, got '
          f'{type(packing).__name__}.'
      )
    ring_config = packing.ring_config
    if parameters is not None and keys is not None:
      # The dict already carries a key pair, so honouring both would leave it
      # ambiguous which one the context actually used.
      raise ValueError(
          'pass either a parameters dict or keys, not both: an explicit '
          'parameters dict already carries its own key pair.'
      )
    if parameters is None:
      # The packed program already chose a secure ring for this model. Taking
      # its runtime fields is what keeps the executed parameters identical to
      # the ones the security plan was validated against.
      parameters = ring_runtime_parameters(ring_config, keys=keys)
    elif not isinstance(parameters, packing_ir.TestOnlyParameters):
      # A caller may still override, but it must be deliberate: silently
      # running a packed program on other parameters would mean the ring that
      # was validated is not the ring that executes.
      raise ValueError(
          'this Packing carries its own securely derived ring; passing '
          'a parameters dict would run it on different parameters than were '
          'validated. Drop the argument, or mark the dict as a deliberate '
          'test override with packing.test_only_parameters(...).'
      )
    if ring_config is not None:
      # The ring config fixes dnum: the key-switch digit count is part of the
      # parameter plan the security analysis validated, not a tuning knob to
      # be re-picked at execution. Deriving it is the normal path; a caller
      # may restate it, but not contradict it.
      planned_dnum = int(ring_config.dnum)
      if dnum is None:
        dnum = planned_dnum
      elif int(dnum) != planned_dnum:
        raise ValueError(
            f'this packed program was planned with dnum={planned_dnum}; '
            f'running it at dnum={int(dnum)} would execute parameters other '
            'than the ones validated. Pass dnum=None to derive it.'
        )
    if (
        isinstance(global_batch, bool)
        or not isinstance(global_batch, (int, np.integer))
        or int(global_batch) < 1
    ):
      raise ValueError('global_batch must be a positive int.')
    global_batch = int(global_batch)
    import jax

    if devices is None:
      # Operation-at-a-time execution and bootstrap barriers currently have
      # single-device implementations. Preserve those paths while making the
      # canonical whole-program model path data parallel by default.
      can_shard_automatically = (
          compile_mode == 'whole'
          and not any(
              _operation_kind(operation) == 'bootstrap'
              for operation in packing.operations
          )
      )
      if can_shard_automatically:
        devices = _device_prefix_for_batch(
            global_batch, jax.local_devices()
        )
      else:
        devices = (jax.local_devices()[0],)
    else:
      try:
        devices = tuple(devices)
      except TypeError as error:
        raise TypeError('devices must be an iterable of JAX devices.') from error
      if not devices:
        raise ValueError('devices must contain at least one JAX device.')
    if len(set(devices)) != len(devices):
      raise ValueError('devices must not contain duplicates.')
    device_count = len(devices)
    if global_batch % device_count:
      raise ValueError(
          f'global_batch={global_batch} must be divisible by '
          f'device_count={device_count}.'
      )
    if input_scale is not None:
      if isinstance(input_scale, bool):
        raise TypeError('input_scale must be None or a real number.')
      input_scale = float(input_scale)
      if not math.isfinite(input_scale) or input_scale <= 0:
        raise ValueError('input_scale must be finite and positive.')
    if input_nsd is not None and (
        isinstance(input_nsd, bool)
        or not isinstance(input_nsd, (int, np.integer))
        or int(input_nsd) < 1
    ):
      raise ValueError('input_nsd must be None or a positive int.')
    if (
        isinstance(headroom, bool)
        or not isinstance(headroom, (int, np.integer))
        or int(headroom) < 0
    ):
      raise ValueError('headroom must be a non-negative int.')
    if not isinstance(cache_rotation_keys, bool):
      raise TypeError('cache_rotation_keys must be a bool.')
    if (
        isinstance(bsgs_n_jobs, bool)
        or not isinstance(bsgs_n_jobs, (int, np.integer))
        or int(bsgs_n_jobs) < 1
    ):
      raise ValueError('bsgs_n_jobs must be a positive int.')
    if not isinstance(bsgs_streaming, bool):
      raise TypeError('bsgs_streaming must be a bool.')
    if compile_mode not in ('whole', 'operations'):
      raise ValueError(
          "compile_mode must be either 'whole' or 'operations'."
      )
    if compile_mode == 'operations' and device_count != 1:
      raise ValueError(
          "compile_mode='operations' currently requires exactly one device."
      )
    if bsgs_streaming and compile_mode != 'operations':
      raise ValueError(
          "bsgs_streaming=True requires compile_mode='operations'."
      )
    if bsgs_streaming:
      unsupported = [
          _operation_id(operation)
          for operation in packing.operations
          if (
              _operation_kind(operation) == 'matvec'
              and getattr(
                  _operation_argument(operation)[0], '_constant_kind', None
              ) not in ('sparse', 'lazy_sparse')
          )
      ]
      if unsupported:
        raise ValueError(
            'bsgs_streaming requires sparse MatVec constants; dense '
            f'operators: {unsupported}.'
        )
      if any(
          _operation_kind(operation) == 'bootstrap'
          for operation in packing.operations
      ):
        raise ValueError(
            'bsgs_streaming does not support bootstrap regions.'
        )
    self.packing = packing
    self.global_batch = global_batch
    self.devices = devices
    self.device_count = device_count
    self.per_device_batch = global_batch // device_count
    self.input_scale = input_scale
    self.input_nsd = None if input_nsd is None else int(input_nsd)
    self.headroom = int(headroom)
    self.cache_rotation_keys = cache_rotation_keys
    self.compile_mode = compile_mode
    self.bsgs_n_jobs = int(bsgs_n_jobs)
    self.bsgs_streaming = bsgs_streaming
    self.ctx = CKKSContext(
        parameters, batch=self.per_device_batch, dnum=dnum
    )
    self.bindings = {}
    self._compiled = None
    self._compiled_regions = None
    self._templates = {}

    _analyze_mapping(self, perf_test=perf_test)
    if self.device_count > 1 and self.has_bootstrap:
      raise ValueError(
          'multi-device Mapping does not yet support bootstrap barriers; '
          'use one device or a network without bootstrap.'
      )
    state_before_initialization = self.ctx.__dict__.copy()
    try:
      r, c = self.ctx.degree_layout
      self.ctx.program_initialization(
          total_rotation_indices=list(self.required_rotation_indices),
          dnum=self.ctx.dnum,
          r=r,
          c=c,
          degree_layout=self.ctx.degree_layout,
          batch=self.ctx.batch,
          perf_test=perf_test,
          pregenerated_rotation_keys=pregenerated_rotation_keys,
          cache_rotation_keys=cache_rotation_keys,
      )
      _materialize_mapping(self)
    except BaseException:
      # A build that fails part-way (an HBM RESOURCE_EXHAUSTED during compile
      # is the common case) has already handed its traced program, with the
      # bindings' device constants, to JAX's caches. The caller never gets a
      # Mapping to release, so discard that state here or the next build in
      # this process starts with less HBM than the one that just failed.
      _discard_program_state(self.ctx, getattr(self, 'bindings', None))
      self.ctx.__dict__.clear()
      self.ctx.__dict__.update(state_before_initialization)
      raise

  @property
  def context(self):
    """The cryptographic resource instantiated and owned by this Mapping."""
    self._require_live()
    return self.ctx

  def estimate_live_memory(self) -> dict[str, int]:
    """Estimate Mapping-owned ciphertext, constant, and key bytes.

    This is a logical deployment estimate, not an allocator measurement: JAX
    may alias, duplicate, or release physical buffers differently. Keeping the
    calculation here lets demos report one model-independent view without
    reading Mapping plans or CKKSContext caches directly.
    """
    self._require_live()
    input_spec = self.input_spec
    output_spec = self.output_spec
    degree = math.prod(input_spec.degree_layout)
    bytes_per_word = np.dtype(input_spec.dtype).itemsize
    full_num_moduli = len(self.ctx.q_towers) + len(self.ctx.p_towers)
    dnum = int(self.ctx.dnum)

    ciphertext_in_bytes = math.prod(input_spec.payload_shape) * bytes_per_word
    ciphertext_out_bytes = (
        math.prod(output_spec.payload_shape) * bytes_per_word
    )
    matvec_constants_bytes = sum(
        len(self.operation_plans[operation_id].active_diagonal_indices)
        * degree
        * self.value_specs[inputs[0]].num_moduli
        * bytes_per_word
        for operation_id, kind, inputs, _ in self.packing.operations
        if kind == 'matvec'
    )
    plaintext_constants_bytes = sum(
        degree
        * self.value_specs[inputs[0]].num_moduli
        * bytes_per_word
        for _, kind, inputs, _ in self.packing.operations
        if kind in _PLAIN_KINDS
    )
    evaluation_key_bytes = (
        2 * full_num_moduli * degree * bytes_per_word * dnum
    )
    rotation_key_count = len(self.required_rotation_indices)
    rotation_keys_bytes = rotation_key_count * evaluation_key_bytes
    total_bytes = sum((
        ciphertext_in_bytes,
        ciphertext_out_bytes,
        matvec_constants_bytes,
        plaintext_constants_bytes,
        evaluation_key_bytes,
        rotation_keys_bytes,
    ))
    return {
        'ciphertext_in_bytes': ciphertext_in_bytes,
        'ciphertext_out_bytes': ciphertext_out_bytes,
        'matvec_constants_bytes': matvec_constants_bytes,
        'plaintext_constants_bytes': plaintext_constants_bytes,
        'evaluation_key_bytes': evaluation_key_bytes,
        'rotation_keys_bytes': rotation_keys_bytes,
        'rotation_key_count': rotation_key_count,
        'total_bytes': total_bytes,
    }

  def pack_input(self, value) -> np.ndarray:
    """Lower one logical input to this Mapping's physical slot vector."""
    return self.packing.pack(value)

  def unpack_output(self, value) -> np.ndarray:
    """Restore one physical slot vector to the model's logical output."""
    return self.packing.unpack(value)

  def encrypt_input(self, value):
    """Pack and encrypt one logical input or this Mapping's global batch."""
    self._require_live()
    scale = self.input_spec.scale
    if self.global_batch == 1:
      return self.ctx.encrypt_slots(
          self.pack_input(value), scale=scale
      )
    try:
      values = list(value)
    except TypeError as error:
      raise TypeError(
          'batched Mapping input must be an iterable of logical inputs.'
      ) from error
    if len(values) != self.global_batch:
      raise ValueError(
          f'batched Mapping input contains {len(values)} values, expected '
          f'{self.global_batch}.'
      )
    return self.ctx.encrypt_slots_batch(
        [self.pack_input(item) for item in values],
        scale=scale,
    )

  def decrypt_output(
      self, value, *, validate_approximation: bool = True
  ):
    """Decrypt and unpack one logical output or the complete global batch."""
    self._require_live()
    self._validate_polynomial(value, self.output_spec, 'output')
    scale = self.output_spec.scale
    if self.global_batch == 1:
      physical = self.ctx.decrypt_slots(
          value,
          scale=scale,
          validate_approximation=validate_approximation,
      )
      return self.unpack_output(physical)
    physical_batch = self.ctx.decrypt_slots_batch(
        value,
        scale=scale,
        validate_approximation=validate_approximation,
    )
    if len(physical_batch) != self.global_batch:
      raise RuntimeError(
          'context returned a different decrypted batch size than Mapping.'
      )
    return [self.unpack_output(physical) for physical in physical_batch]

  def infer(
      self,
      value,
      *,
      validate_approximation: bool = True,
      trace_dir: str | None = None,
  ):
    """Run the complete trusted-demo inference boundary."""
    self._require_live()
    encrypted = self.encrypt_input(value)
    trace = nullcontext()
    if trace_dir:
      import jax

      trace = jax.profiler.trace(trace_dir)
    with trace:
      output = self.execute(encrypted)
    return self.decrypt_output(
        output, validate_approximation=validate_approximation
    )

  @property
  def input_spec(self) -> _ValueSpec:
    return replace(self.value_specs['input'], batch=self.global_batch)

  @property
  def output_spec(self) -> _ValueSpec:
    return replace(
        self.value_specs[self.packing.output], batch=self.global_batch
    )

  @property
  def has_bootstrap(self) -> bool:
    return any(
        _operation_kind(operation) == 'bootstrap'
        for operation in self.packing.operations
    )

  @staticmethod
  def _validate_polynomial(value, spec: _ValueSpec, name: str) -> None:
    import polynomial

    if not isinstance(value, polynomial.Polynomial):
      raise TypeError(f'{name} must be a Polynomial.')
    value.validate()
    if value.batch != spec.batch or value.num_elements != spec.elements:
      raise ValueError(f'{name} Polynomial batch/elements do not match.')
    if value.degree_layout != spec.degree_layout:
      raise ValueError(f'{name} Polynomial degree layout does not match.')
    if tuple(value.moduli) != spec.moduli:
      raise ValueError(f'{name} Polynomial modulus prefix does not match.')
    if (
        value.precision != 32
        or str(np.dtype(value.modulus_dtype)) != spec.dtype
        or str(value.polynomial.dtype) != spec.dtype
    ):
      raise ValueError(
          f'{name} must use the canonical precision=32/{spec.dtype} '
          'Polynomial representation.'
      )
    if spec.scale is not None:
      actual_scale = getattr(value, '_ckks_scale', None)
      if (
          actual_scale is None
          or not math.isclose(
              float(actual_scale), spec.scale, rel_tol=1e-12, abs_tol=0.0
          )
      ):
        raise ValueError(
            f'{name} Polynomial scale {actual_scale!r} does not match '
            f'planned scale {spec.scale}.'
        )
    if spec.nsd is not None and getattr(value, '_ckks_nsd', 1) != spec.nsd:
      raise ValueError(
          f'{name} Polynomial nsd does not match planned nsd {spec.nsd}.'
      )

  def _template(self, value_id: str):
    template = self._templates.get(value_id)
    if template is not None:
      return template
    import polynomial

    spec = self.value_specs[value_id]
    cache = self.ctx._param_cache
    template = polynomial.Polynomial(
        {
            'batch': spec.batch,
            'num_elements': spec.elements,
            'degree': self.ctx.degree,
            'num_moduli': spec.num_moduli,
            'precision': 32,
            'degree_layout': spec.degree_layout,
        },
        {
            'moduli': list(spec.moduli),
            'ntt_ctx': cache.get_sliced_ntt_q(spec.level),
        },
    )
    self._templates[value_id] = template
    return template

  def _wrap(self, value_id: str, payload, *, batch: Optional[int] = None):
    spec = self.value_specs[value_id]
    if batch is None:
      batch = self.global_batch
    value = self._template(value_id)._clone_with_payload(
        payload, batch=batch
    )
    return _apply_metadata(value, spec)

  def _prebuild_templates(self, value_ids) -> None:
    for value_id in value_ids:
      self._template(value_id)

  def _require_live(self) -> None:
    if getattr(self, '_released', False):
      raise RuntimeError(
          'this Mapping was released; construct a new Mapping to encrypt, '
          'execute or decrypt again.'
      )

  def release(self) -> None:
    """Return this Mapping's device memory to the runtime.

    A materialized Mapping pins gigabytes of HBM: the compiled executable with
    the evaluation keys and encoded constants it captured, the context's
    parameter caches, and the codec caches keyed by its context. Dropping the
    Python reference alone frees none of it, because JAX's tracing caches keep
    the compiled program's closure, and through it this object, alive.
    ``release`` drops the executable and bindings, evicts the codec caches
    built for this context, clears the JAX caches, and marks the Mapping
    unusable. Build a new Mapping for the next placement. Idempotent.
    """
    if getattr(self, '_released', False):
      return
    ctx = self.ctx
    bindings = self.bindings if isinstance(self.bindings, dict) else {}
    self._compiled = None
    self._compiled_regions = None
    self._templates = {}
    self.bindings = {}
    self.ctx = None
    self._released = True
    _discard_program_state(ctx, bindings)

  def execute(self, value):
    self._require_live()
    self._validate_polynomial(value, self.input_spec, 'input')
    if self._compiled is not None:
      local_input_spec = self.value_specs['input']
      device_payload = value.polynomial.reshape(
          (self.device_count,) + local_input_spec.payload_shape
      )
      device_output = self._compiled(
          _aot_pmap_input(device_payload, self.device_count)
      )
      payload = device_output.reshape(self.output_spec.payload_shape)
      result = self._wrap(
          self.packing.output, payload, batch=self.global_batch
      )
      self._validate_polynomial(result, self.output_spec, 'output')
      return result

    if self.compile_mode == 'operations' and not self.has_bootstrap:
      import gc

      operation_by_id = {
          _operation_id(operation): operation
          for operation in self.packing.operations
      }
      environment = {'input': value}
      for region_index, (
          node_ids,
          input_ids,
          output_ids,
          compiled,
      ) in enumerate(self._compiled_regions):
        if compiled is not None or len(node_ids) != 1 or output_ids != node_ids:
          raise RuntimeError(
              f'lazy operation region {node_ids} has invalid materialization.'
          )
        node_id = node_ids[0]
        operation = operation_by_id[node_id]
        try:
          values = tuple(
              environment[source] for source in input_ids
          )
          payload = _execute_lazy_operation(self, operation, values)
          result = self._wrap(node_id, payload)
          self._validate_polynomial(
              result, self.value_specs[node_id], node_id
          )
          environment[node_id] = result

          live_after = {self.packing.output}
          for _, future_inputs, _, _ in self._compiled_regions[
              region_index + 1 :
          ]:
            live_after.update(future_inputs)
          for value_id in tuple(environment):
            if value_id not in live_after:
              del environment[value_id]
        finally:
          _clear_transient_operation_accessors(self.ctx)
          gc.collect()

      result = environment[self.packing.output]
      self._validate_polynomial(result, self.output_spec, 'output')
      return result

    environment = {'input': value}
    for region_index, (
        node_ids,
        input_ids,
        output_ids,
        compiled,
    ) in enumerate(self._compiled_regions):
      if compiled is None:
        for node_id in node_ids:
          packed_operation = next(
              item
              for item in self.packing.operations
              if _operation_id(item) == node_id
          )
          operation = self.bindings[node_id]
          arguments = [
              environment[source]
              for source in _operation_inputs(packed_operation)
          ]
          result = operation(*arguments)
          spec = self.value_specs[node_id]
          self._validate_polynomial(result, spec, node_id)
          environment[node_id] = result
      else:
        payloads = tuple(
            environment[source].polynomial for source in input_ids
        )
        outputs = compiled(*payloads)
        if len(output_ids) == 1 and not isinstance(outputs, tuple):
          outputs = (outputs,)
        if len(outputs) != len(output_ids):
          raise RuntimeError(
              f'compiled region returned {len(outputs)} value(s), expected '
              f'{len(output_ids)}.'
          )
        for value_id, payload in zip(output_ids, outputs, strict=True):
          result = self._wrap(value_id, payload)
          self._validate_polynomial(
              result, self.value_specs[value_id], value_id
          )
          environment[value_id] = result

      # Segmented execution should not retain dead ciphertext intermediates.
      live_after = {self.packing.output}
      for _, future_inputs, _, _ in self._compiled_regions[
          region_index + 1 :
      ]:
        live_after.update(future_inputs)
      for value_id in tuple(environment):
        if value_id not in live_after:
          del environment[value_id]
    result = environment[self.packing.output]
    self._validate_polynomial(result, self.output_spec, 'output')
    return result


def _raw_program(mapping, bindings):
  def run(payload):
    environment = {'input': payload}
    for operation in mapping.packing.operations:
      operation_id = _operation_id(operation)
      fused = bindings[operation_id]
      arguments = [
          environment[source] for source in _operation_inputs(operation)
      ]
      environment[operation_id] = fused(*arguments)
    return environment[mapping.packing.output]

  return run


def _region_io(mapping, node_ids):
  """Return live external inputs/outputs for one maximal fused region."""
  region_set = set(node_ids)
  region_operations = tuple(
      operation
      for operation in mapping.packing.operations
      if _operation_id(operation) in region_set
  )
  input_ids = []
  for operation in region_operations:
    for source in _operation_inputs(operation):
      if source not in region_set and source not in input_ids:
        input_ids.append(source)
  needed_outside = {mapping.packing.output}
  for operation in mapping.packing.operations:
    if _operation_id(operation) not in region_set:
      needed_outside.update(_operation_inputs(operation))
  output_ids = tuple(
      _operation_id(operation)
      for operation in region_operations
      if _operation_id(operation) in needed_outside
  )
  return region_operations, tuple(input_ids), output_ids


def _compile_segmented(mapping, bindings) -> None:
  import jax
  import jax.numpy as jnp

  ctx = mapping.ctx
  operation_by_id = {
      _operation_id(operation): operation
      for operation in mapping.packing.operations
  }
  compiled_regions = []
  template_ids = set()
  for fusible, node_ids in mapping.regions:
    region_operations = tuple(
        operation_by_id[node_id] for node_id in node_ids
    )
    if not fusible:
      compiled_regions.append((
          node_ids,
          _operation_inputs(region_operations[0]),
          node_ids,
          None,
      ))
      continue

    region_operations, input_ids, output_ids = _region_io(mapping, node_ids)
    if not output_ids:
      raise RuntimeError(f'fusible region {node_ids} has no live output.')

    def raw_region(
        *payloads,
        input_ids=input_ids,
        output_ids=output_ids,
        region_operations=region_operations,
    ):
      environment = dict(zip(input_ids, payloads, strict=True))
      for operation in region_operations:
        operation_id = _operation_id(operation)
        fused = bindings[operation_id]
        arguments = [
            environment[source] for source in _operation_inputs(operation)
        ]
        environment[operation_id] = fused(*arguments)
      return tuple(environment[value_id] for value_id in output_ids)

    structs = tuple(
        jax.ShapeDtypeStruct(
            mapping.value_specs[value_id].payload_shape, jnp.uint32
        )
        for value_id in input_ids
    )
    compiled = jax.jit(raw_region).lower(*structs).compile()
    compiled_regions.append((
        node_ids,
        input_ids,
        output_ids,
        compiled,
    ))
    template_ids.update(output_ids)

  mapping.bindings = bindings
  mapping._compiled = None
  mapping._compiled_regions = tuple(compiled_regions)
  mapping._prebuild_templates(template_ids)


def _prepare_lazy_operation_regions(mapping) -> None:
  """Keep only region metadata for memory-bounded one-shot execution.

  Every region here is a single fusible operation. The caller only reaches this
  path when ``compile_mode == 'operations'`` and the network has no bootstrap,
  and those two conditions are exactly what make ``_fusion_regions`` emit one
  ``(True, (node_id,))`` region per operation.
  """
  regions = []
  for _, node_ids in mapping.regions:
    _, input_ids, output_ids = _region_io(mapping, node_ids)
    regions.append((node_ids, input_ids, output_ids, None))

  mapping.bindings = {}
  mapping._compiled = None
  mapping._compiled_regions = tuple(regions)


def _dynamic_binding_operands(mapping, operation, binding):
  """Return large per-operation state that must not become XLA constants."""
  kind = _operation_kind(operation)
  owner = getattr(binding, '__self__', None)
  if kind == 'matvec':
    operation_id = _operation_id(operation)
    if isinstance(
        mapping.operation_plans[operation_id], _ZeroMatVecPlan
    ):
      return (), owner
    if owner is None or not hasattr(owner, '_matvec_operands'):
      raise RuntimeError('matvec binding does not expose dynamic operands.')
    return tuple(owner._matvec_operands()), owner
  operation_id = _operation_id(operation)
  if kind == 'rotate' and mapping.operation_plans[operation_id] != 0:
    if owner is None or not hasattr(owner, '_rotation_state'):
      raise RuntimeError('rotation binding does not expose dynamic key state.')
    return tuple(owner._rotation_state()), owner
  return (), owner


def _execute_lazy_operation(mapping, operation, values):
  """Bind, lower, execute, and release exactly one packed operation."""
  import jax
  import jax.numpy as jnp

  payloads = tuple(value.polynomial for value in values)
  binding = None
  owner = None
  jitted = None
  compiled = None
  dynamic_operands = ()
  try:
    binding = _bind_operation(mapping.ctx, mapping, operation, None)
    if (
        _operation_kind(operation) == 'matvec'
        and mapping.bsgs_streaming
        and not isinstance(
            mapping.operation_plans[_operation_id(operation)],
            _ZeroMatVecPlan,
        )
    ):
      owner = getattr(binding, '__self__', None)
      result = binding(values[0])
      operation_id = _operation_id(operation)
      mapping._validate_polynomial(
          result, mapping.value_specs[operation_id], operation_id
      )
      return jax.block_until_ready(result.polynomial)
    dynamic_operands, owner = _dynamic_binding_operands(
        mapping, operation, binding
    )

    def raw_operation(runtime_payloads, runtime_operands):
      return binding(*runtime_payloads, *runtime_operands)

    payload_structs = tuple(
        jax.ShapeDtypeStruct(payload.shape, jnp.dtype(payload.dtype))
        for payload in payloads
    )
    operand_structs = jax.tree_util.tree_map(
        lambda value: jax.ShapeDtypeStruct(
            value.shape, jnp.dtype(value.dtype)
        ),
        dynamic_operands,
    )
    operation_sharding = jax.sharding.SingleDeviceSharding(
        mapping.devices[0]
    )
    jitted = jax.jit(
        raw_operation,
        in_shardings=(operation_sharding, operation_sharding),
        out_shardings=operation_sharding,
    )
    compiled = jitted.lower(payload_structs, operand_structs).compile()
    result = compiled(payloads, dynamic_operands)
    return jax.block_until_ready(result)
  finally:
    compiled = None
    if jitted is not None:
      jitted.clear_cache()
    jitted = None
    binding = None
    dynamic_operands = ()
    release = getattr(owner, '_release', None)
    if release is not None:
      release()
    owner = None


def _clear_transient_operation_accessors(ctx) -> None:
  """Evict key- and level-bearing evaluators after a one-shot operation."""
  for accessor in (
      ctx.ptct_mul,
      ctx.he_mul,
      ctx.he_rot,
      ctx.he_rescale,
      ctx.he_level_reduce,
      ctx.he_add,
      ctx.he_sub,
  ):
    accessor.clear()


def _discard_program_state(ctx, bindings) -> None:
  """Drop everything a (possibly half-built) program pinned on the devices.

  Releases the bindings' evaluator state and formatted keys, evicts the codec
  caches keyed by ``ctx``, and clears the JAX caches that hold the traced
  program's closure (there is no per-program handle to drop, so every cached
  program recompiles on its next call; this is why release belongs outside
  the request path). Safe to call more than once and with ``ctx`` None.
  """
  import gc

  import jax

  if ctx is not None:
    try:
      _release_aot_source_state(
          ctx, bindings if isinstance(bindings, dict) else {}
      )
    except AttributeError:
      # A build that failed before program initialization finished has no
      # operator accessors to clear; the cache eviction below still applies.
      pass
    sys.modules[CKKSContext.__module__].evict_context_caches(ctx)
  jax.clear_caches()
  gc.collect()


def _release_aot_source_state(ctx, bindings) -> None:
  """Drop binding buffers after a whole-program executable captures them.

  Lowering an AOT program embeds its constants in the executable. Keeping the
  bound evaluators alive after that point duplicates every encoded matvec and
  evaluator control on device 0. Large models can compile successfully and
  then fail while loading the executable because that redundant source state
  still occupies HBM. The executable no longer calls these Python bindings,
  so release their owned buffers and evict the accessor caches before serving.
  """
  released = set()
  for binding in tuple(bindings.values()):
    owner = getattr(binding, '__self__', None)
    release = getattr(owner, '_release', None)
    if release is not None and id(owner) not in released:
      release()
      released.add(id(owner))
  bindings.clear()
  _clear_transient_operation_accessors(ctx)
  cache = getattr(ctx, '_param_cache', None)
  if cache is not None:
    # Formatted keys are evaluator inputs already captured in the executable;
    # the raw host keys remain available if callers intentionally build a new
    # direct evaluator from ``mapping.ctx`` later.
    cache._formatted_rotation_keys.clear()
  import gc
  gc.collect()


def _materialize_mapping(mapping) -> None:
  """Prepare eager or one-shot Mapping execution after exact key setup."""
  ctx = mapping.ctx
  cache = getattr(ctx, '_param_cache', None)
  if cache is None:
    raise RuntimeError('context must be initialized before Mapping compilation.')
  if (
      _context_fingerprint(ctx, perf_test=mapping.perf_test)
      != mapping.context_fingerprint
  ):
    raise ValueError('initialized context no longer matches the Mapping.')
  planned_rotations = set(mapping.required_rotation_indices)
  initialized_rotations = set(cache.rot_indices)
  missing = sorted(planned_rotations - initialized_rotations)
  if missing:
    raise ValueError(f'context is missing planned rotation keys {missing}.')
  unexpected = sorted(initialized_rotations - planned_rotations)
  if unexpected:
    raise ValueError(
        f'context has unplanned rotation keys {unexpected}; Mapping '
        'materialization requires the exact scheduled key set.'
    )
  cache_contract = {
      'q_towers': tuple(int(q) for q in cache.q_towers),
      'p_towers': tuple(int(p) for p in cache.p_towers),
      'degree_layout': tuple(cache.degree_layout),
      'dnum': int(cache.dnum),
      'composite_degree': int(cache.composite_degree),
      'batch': int(cache.batch),
      'perf_test': bool(cache.perf_test),
      'noise_std': float(cache.noise_std),
      'noise_scale_degree': int(cache.noise_scale_degree),
      'key_generation_version': int(cache.key_generation_version),
  }
  expected_cache_contract = {
      'q_towers': tuple(int(q) for q in ctx.q_towers),
      'p_towers': tuple(int(p) for p in ctx.p_towers),
      'degree_layout': tuple(ctx.degree_layout),
      'dnum': int(ctx.dnum),
      'composite_degree': int(ctx.composite_degree),
      'batch': int(ctx.batch),
      'perf_test': mapping.perf_test,
      'noise_std': _normalize_noise_std(
          ctx.parameters.get('sigma', sigma)
      ),
      'noise_scale_degree': normalize_noise_scale_degree(
          ctx.parameters.get('noise_scale_degree', 1)
      ),
      'key_generation_version': int(kg.KEY_GENERATION_VERSION),
  }
  if cache_contract != expected_cache_contract:
    raise ValueError(
        'initialized parameter cache does not match the Mapping context.'
    )
  claimed_fingerprint = getattr(cache, '_static_model_fingerprint', None)
  if claimed_fingerprint not in (None, mapping.fingerprint):
    raise RuntimeError(
        'this initialized context is already sealed to a different Mapping.'
    )
  cache._static_model_fingerprint = mapping.fingerprint
  if mapping.compile_mode == 'operations' and not mapping.has_bootstrap:
    _prepare_lazy_operation_regions(mapping)
    return

  bindings = _prepare_bindings(ctx, mapping)
  if mapping.has_bootstrap or mapping.compile_mode == 'operations':
    _compile_segmented(mapping, bindings)
    return

  import jax
  import jax.numpy as jnp

  raw_program = _raw_program(mapping, bindings)
  input_struct = jax.ShapeDtypeStruct(
      (mapping.device_count,) + mapping.value_specs['input'].payload_shape,
      jnp.uint32,
  )
  compiled = jax.pmap(
      raw_program, devices=mapping.devices
  ).lower(input_struct).compile()
  mapping._compiled = compiled
  mapping._compiled_regions = None
  mapping._prebuild_templates((mapping.packing.output,))
  _release_aot_source_state(ctx, bindings)
  mapping.bindings = {}


__all__ = ['Mapping']

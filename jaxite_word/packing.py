"""Ciphertext slot layout and the packing phase over it.

Two things live here, both below the application model:

* Slot layout. A :class:`TensorLayout` is a coordinate-to-slot policy and a
  :class:`TensorSpec` binds one to a logical shape -- how an individual data
  element lands in a ciphertext slot.
* The packing phase. :func:`pack` takes the semantic
  :class:`nn.VectorizedProgram` the frontend produces and emits a
  :class:`Packing`: an ordered DAG of privacy-preserving operators over slots
  -- matvec, square, add, add_plain, mul_plain, rotate, rescale, level_reduce
  -- together with the constants they consume and the secure ring derived from
  the program's own slot demand and emitted depth.

A CKKS ciphertext is a SIMD vector, so an application operator written for
element-wise computation is not directly executable. Turning one into PP-ops
is what this module does, and it is the only thing that does it: the layer
descriptors and matrix-free recipes below are private, because a second way to
describe a model would have to be kept in agreement with the first by hand.

This module knows nothing about HE levels, keys or evaluators, and imports no
first-party module -- not even the frontend, which hands it a duck-typed
program. ``nn`` describes a model for it and ``mapping`` schedules what it
emits, so the dependency runs mapping -> here <- nn.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import hashlib
import json
import math
import sys
from typing import Any, Callable, ClassVar, Iterator, Optional, Union

import numpy as np


if __name__ == 'jaxite_word.packing':
  sys.modules.setdefault('packing', sys.modules[__name__])
elif __name__ == 'packing':
  sys.modules.setdefault('jaxite_word.packing', sys.modules[__name__])


def _json_digest(value) -> str:
  data = json.dumps(value, sort_keys=True, separators=(',', ':'))
  return hashlib.sha256(data.encode()).hexdigest()


def _logical_shape(value, description: str) -> tuple[int, ...]:
  try:
    shape = tuple(value)
  except TypeError as error:
    raise TypeError(
        f'{description} must be an iterable of positive ints.'
    ) from error
  normalized = []
  for dimension in shape:
    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, (int, np.integer))
        or int(dimension) <= 0
    ):
      raise ValueError(
          f'{description} dimensions must be positive ints, got {shape!r}.'
      )
    normalized.append(int(dimension))
  return tuple(normalized)


def _packing(value, description: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f'{description} must be a non-empty string.')
  return value


def _spatial_pair(
    value,
    description: str,
    *,
    allow_zero: bool = False,
) -> tuple[int, int]:
  if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
    pair = (int(value), int(value))
  else:
    if isinstance(value, (str, bytes)):
      raise TypeError(f'{description} must be an int or a pair of ints.')
    try:
      pair = tuple(value)
    except TypeError as error:
      raise TypeError(
          f'{description} must be an int or a pair of ints.'
      ) from error
    if len(pair) != 2:
      raise ValueError(f'{description} must contain exactly two ints.')
    if any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer))
        for item in pair
    ):
      raise TypeError(f'{description} must contain only ints.')
    pair = tuple(int(item) for item in pair)
  minimum = 0 if allow_zero else 1
  if any(item < minimum for item in pair):
    requirement = 'non-negative' if allow_zero else 'positive'
    raise ValueError(f'{description} values must be {requirement}.')
  return pair


def _logical_input_array(value, shape: tuple[int, ...]) -> np.ndarray:
  """Return one declared logical tensor, allowing only a flat compatibility form."""
  logical = np.asarray(value)
  flat_shape = (math.prod(shape),)
  if logical.shape == shape:
    return logical
  if logical.shape == flat_shape:
    return logical.reshape(shape)
  raise ValueError(
      f'input has shape {logical.shape}; expected logical shape {shape} '
      f'or flat compatibility shape {flat_shape}.'
  )


def _pack_coordinate_map(
    value,
    shape: tuple[int, ...],
    num_slots: int,
    physical_size: int,
    coordinate_map: tuple[int, ...],
    packing: str,
) -> np.ndarray:
  logical = _logical_input_array(value, shape)
  if physical_size > num_slots:
    raise ValueError(
        f'layout {packing!r} needs {physical_size} '
        f'slots, but only {num_slots} are available.'
    )
  slots = np.zeros(num_slots, dtype=logical.dtype)
  for coordinate, slot in zip(
      np.ndindex(shape), coordinate_map, strict=True
  ):
    slots[slot] = logical[coordinate]
  return slots


class TensorLayout:
  """Immutable coordinate-to-slot policy used by workload lowering."""

  def descriptor(self):
    """Return JSON-serializable metadata that identifies this layout policy."""
    raise NotImplementedError(
        'custom TensorLayout implementations must define descriptor().'
    )

  @property
  def packing(self) -> str:
    raise NotImplementedError

  def physical_size(self, shape: tuple[int, ...]) -> int:
    raise NotImplementedError

  def coordinate_to_slot(
      self,
      shape: tuple[int, ...],
      coordinate: tuple[int, ...],
  ) -> int:
    raise NotImplementedError

  def _validated_coordinate_map(
      self, shape: tuple[int, ...]
  ) -> tuple[int, tuple[int, ...]]:
    physical_size = self.physical_size(shape)
    if (
        isinstance(physical_size, bool)
        or not isinstance(physical_size, (int, np.integer))
        or int(physical_size) < 1
    ):
      raise ValueError('TensorLayout.physical_size must return a positive int.')
    physical_size = int(physical_size)
    indices = []
    seen = set()
    for coordinate in np.ndindex(shape):
      slot = self.coordinate_to_slot(shape, coordinate)
      if (
          isinstance(slot, bool)
          or not isinstance(slot, (int, np.integer))
      ):
        raise TypeError(
            'TensorLayout.coordinate_to_slot must return an int.'
        )
      slot = int(slot)
      if not 0 <= slot < physical_size:
        raise ValueError(
            f'layout slot {slot} is outside [0, {physical_size}).'
        )
      if slot in seen:
        raise ValueError(
            f'layout maps multiple logical coordinates to slot {slot}.'
        )
      seen.add(slot)
      indices.append(slot)
    try:
      json.dumps(
          self.descriptor(), sort_keys=True, separators=(',', ':')
      )
    except (TypeError, ValueError) as error:
      raise TypeError(
          'TensorLayout.descriptor() must be JSON-serializable.'
      ) from error
    return physical_size, tuple(indices)

  def pack(
      self,
      value,
      shape: tuple[int, ...],
      num_slots: int,
  ) -> np.ndarray:
    physical_size, indices = self._validated_coordinate_map(shape)
    return _pack_coordinate_map(
        value,
        shape,
        num_slots,
        physical_size,
        indices,
        self.packing,
    )


def _validate_coordinate(
    shape: tuple[int, ...],
    coordinate: tuple[int, ...],
) -> tuple[int, ...]:
  coordinate = tuple(coordinate)
  if len(coordinate) != len(shape):
    raise ValueError(
        f'coordinate rank {len(coordinate)} does not match shape {shape}.'
    )
  for index, (position, dimension) in enumerate(
      zip(coordinate, shape, strict=True)
  ):
    if (
        isinstance(position, bool)
        or not isinstance(position, (int, np.integer))
    ):
      raise TypeError(f'coordinate[{index}] must be an int.')
    if not 0 <= int(position) < dimension:
      raise ValueError(
          f'coordinate {coordinate} is outside shape {shape}.'
      )
  return tuple(int(position) for position in coordinate)


@dataclass(frozen=True)
class ChannelMajor(TensorLayout):
  """Contiguous row-major slots for unbatched ``CHW`` or flat tensors."""

  def descriptor(self):
    return {'kind': 'channel_major'}

  @property
  def packing(self) -> str:
    return 'channel_major'

  def physical_size(self, shape: tuple[int, ...]) -> int:
    return math.prod(shape)

  def coordinate_to_slot(
      self,
      shape: tuple[int, ...],
      coordinate: tuple[int, ...],
  ) -> int:
    coordinate = _validate_coordinate(shape, coordinate)
    return int(np.ravel_multi_index(coordinate, shape))


@dataclass(frozen=True)
class StrideMultiplexed(TensorLayout):
  """Residue-major CHW slots for a first strided spatial operator.

  A coordinate ``(c, h, w)`` is stored by spatial residue first, followed by
  channel and the quotient coordinates. This generalizes the first-convolution
  packing used by the demos to arbitrary channel counts and non-square strides.
  """

  stride: tuple[int, int]

  def __post_init__(self):
    object.__setattr__(
        self,
        'stride',
        _spatial_pair(self.stride, 'StrideMultiplexed.stride'),
    )

  @property
  def packing(self) -> str:
    return f'stride_multiplexed_{self.stride[0]}x{self.stride[1]}'

  def descriptor(self):
    return {
        'kind': 'stride_multiplexed',
        'stride': self.stride,
    }

  def physical_size(self, shape: tuple[int, ...]) -> int:
    if len(shape) != 3:
      raise ValueError('StrideMultiplexed requires a rank-3 CHW shape.')
    channels, height, width = shape
    stride_h, stride_w = self.stride
    base_h = math.ceil(height / stride_h)
    base_w = math.ceil(width / stride_w)
    return stride_h * stride_w * channels * base_h * base_w

  def coordinate_to_slot(
      self,
      shape: tuple[int, ...],
      coordinate: tuple[int, ...],
  ) -> int:
    if len(shape) != 3:
      raise ValueError('StrideMultiplexed requires a rank-3 CHW shape.')
    channel, height_index, width_index = _validate_coordinate(
        shape, coordinate
    )
    channels, height, width = shape
    stride_h, stride_w = self.stride
    base_h = math.ceil(height / stride_h)
    base_w = math.ceil(width / stride_w)
    residue = (
        (height_index % stride_h) * stride_w + width_index % stride_w
    )
    return (
        ((residue * channels + channel) * base_h
         + height_index // stride_h) * base_w
        + width_index // stride_w
    )


@dataclass(frozen=True)
class TensorSpec:
  """Immutable shape and layout of one unbatched logical tensor."""

  shape: tuple[int, ...]
  layout: TensorLayout = field(default_factory=ChannelMajor)
  physical_size: int = field(init=False)
  layout_fingerprint: str = field(init=False)
  packing: str = field(init=False)
  _coordinate_map: tuple[int, ...] = field(init=False, repr=False)

  def __post_init__(self):
    shape = _logical_shape(self.shape, 'TensorSpec.shape')
    if not shape:
      raise ValueError('TensorSpec.shape must contain at least one dimension.')
    layout = self.layout
    if layout == 'channel_major':
      layout = ChannelMajor()
    if not isinstance(layout, TensorLayout):
      raise TypeError(
          'TensorSpec.layout must be an packing.TensorLayout value.'
      )
    physical_size, coordinate_map = layout._validated_coordinate_map(shape)
    layout_fingerprint = _json_digest({
        'packing': layout.packing,
        'descriptor': layout.descriptor(),
        'shape': shape,
        'physical_size': physical_size,
        'coordinate_map': coordinate_map,
    })
    packing = (
        layout.packing
        if type(layout) in (ChannelMajor, StrideMultiplexed)
        else f'{layout.packing}:{layout_fingerprint[:16]}'
    )
    object.__setattr__(self, 'shape', shape)
    object.__setattr__(self, 'layout', layout)
    object.__setattr__(self, 'physical_size', physical_size)
    object.__setattr__(self, '_coordinate_map', coordinate_map)
    object.__setattr__(
        self, 'layout_fingerprint', layout_fingerprint
    )
    object.__setattr__(self, 'packing', packing)

  @property
  def size(self) -> int:
    return math.prod(self.shape)

  def coordinate_to_slot(self, coordinate: tuple[int, ...]) -> int:
    """Return the snapshotted physical slot for one logical coordinate."""
    coordinate = _validate_coordinate(self.shape, coordinate)
    logical_index = int(np.ravel_multi_index(coordinate, self.shape))
    return self._coordinate_map[logical_index]

  def pack(self, value, num_slots: int) -> np.ndarray:
    """Pack an exact-shape or explicitly flat tensor through this snapshot."""
    return _pack_coordinate_map(
        value,
        self.shape,
        num_slots,
        self.physical_size,
        self._coordinate_map,
        self.packing,
    )




# =============================================================
# Packed constants: the plaintext values a PP-op consumes. These live over
# ciphertext slots rather than over logical tensors -- an application
# operator becomes PP-ops, and its weights become these, only after
# vectorization.
# =============================================================


def _array_digest(array: np.ndarray) -> str:
  digest = hashlib.sha256()
  digest.update(str(array.dtype).encode())
  digest.update(repr(tuple(array.shape)).encode())
  digest.update(array.tobytes(order='C'))
  return digest.hexdigest()


def _readonly_array(value, *, ndim: Optional[int] = None) -> np.ndarray:
  array = np.array(value, copy=True)
  if ndim is not None and array.ndim != ndim:
    raise ValueError(f'expected a rank-{ndim} array, got shape {array.shape}.')
  if array.dtype.kind not in 'biufc':
    raise TypeError(f'constant arrays must be numeric, got dtype {array.dtype}.')
  if not np.all(np.isfinite(array)):
    raise ValueError('constant arrays must contain only finite values.')
  # A write-disabled owning ndarray can be made writable again by its caller.
  # Back the public view with immutable bytes so lazy recipes cannot change
  # after their fingerprint and security plan have been frozen.
  immutable = np.frombuffer(
      array.tobytes(order='C'), dtype=array.dtype
  ).reshape(array.shape)
  immutable.setflags(write=False)
  return immutable


def _readonly_real_array(value, *, ndim: int) -> np.ndarray:
  array = _readonly_array(value, ndim=ndim)
  if array.dtype.kind == 'c':
    raise TypeError('logical neural-network constants must be real.')
  if any(dimension <= 0 for dimension in array.shape):
    raise ValueError('logical neural-network constants cannot have empty axes.')
  return array


@dataclass(frozen=True)
class PlainSlots:
  """Immutable already-packed plaintext slots owned by model code."""

  values: np.ndarray
  shape: tuple[int, ...]
  packing: str = 'slots'
  scale: Optional[float] = None
  digest: str = field(init=False)

  _constant_kind: ClassVar[str] = 'plain'

  def __post_init__(self):
    values = _readonly_array(self.values, ndim=1)
    if values.dtype.kind == 'c':
      raise TypeError('PlainSlots.values must be real.')
    shape = _logical_shape(self.shape, 'PlainSlots.shape')
    packing = _packing(self.packing, 'PlainSlots.packing')
    scale = self.scale
    if scale is not None:
      if isinstance(scale, bool):
        raise TypeError('PlainSlots.scale must be a real number.')
      scale = float(scale)
      if not math.isfinite(scale) or scale <= 0:
        raise ValueError('PlainSlots.scale must be finite and positive.')
    if math.prod(shape) > values.size:
      raise ValueError(
          f'logical shape {shape} needs {math.prod(shape)} slots, '
          f'but constant has {values.size}.'
      )
    object.__setattr__(self, 'values', values)
    object.__setattr__(self, 'shape', shape)
    object.__setattr__(self, 'packing', packing)
    object.__setattr__(self, 'scale', scale)
    object.__setattr__(
        self,
        'digest',
        _json_digest({
            'values': _array_digest(values),
            'shape': shape,
            'packing': packing,
            'scale': scale,
        }),
    )


@dataclass(frozen=True)
class DenseMatrix:
  """Immutable dense logical transform owned by model code."""

  values: np.ndarray
  input_shape: tuple[int, ...]
  output_shape: tuple[int, ...]
  packing: str = 'slots'
  output_packing: Optional[str] = None
  digest: str = field(init=False)

  _constant_kind: ClassVar[str] = 'dense'

  def __post_init__(self):
    values = _readonly_array(self.values, ndim=2)
    if values.dtype.kind == 'c':
      raise TypeError('DenseMatrix.values must be real.')
    input_shape = _logical_shape(
        self.input_shape, 'DenseMatrix.input_shape'
    )
    output_shape = _logical_shape(
        self.output_shape, 'DenseMatrix.output_shape'
    )
    packing = _packing(self.packing, 'DenseMatrix.packing')
    output_packing = _packing(
        packing if self.output_packing is None else self.output_packing,
        'DenseMatrix.output_packing',
    )
    if values.shape != (
        math.prod(output_shape), math.prod(input_shape)
    ):
      raise ValueError(
          f'matrix shape {values.shape} does not match logical '
          f'{output_shape} x {input_shape}.'
      )
    object.__setattr__(self, 'values', values)
    object.__setattr__(self, 'input_shape', input_shape)
    object.__setattr__(self, 'output_shape', output_shape)
    object.__setattr__(self, 'packing', packing)
    object.__setattr__(self, 'output_packing', output_packing)
    object.__setattr__(
        self,
        'digest',
        _json_digest({
            'values': _array_digest(values),
            'input_shape': input_shape,
            'output_shape': output_shape,
            'packing': packing,
            'output_packing': output_packing,
        }),
    )


@dataclass(frozen=True)
class SparseDiagonals:
  """Immutable sparse diagonal transform owned by model code."""

  dimension: int
  diagonals: tuple[tuple[int, np.ndarray], ...]
  input_shape: tuple[int, ...]
  output_shape: tuple[int, ...]
  packing: str = 'slots'
  output_packing: Optional[str] = None
  digest: str = field(init=False)

  _constant_kind: ClassVar[str] = 'sparse'

  def __post_init__(self):
    if (
        isinstance(self.dimension, bool)
        or not isinstance(self.dimension, (int, np.integer))
        or int(self.dimension) <= 0
    ):
      raise ValueError('dimension must be a positive int.')
    dimension = int(self.dimension)
    input_shape = _logical_shape(
        self.input_shape, 'SparseDiagonals.input_shape'
    )
    output_shape = _logical_shape(
        self.output_shape, 'SparseDiagonals.output_shape'
    )
    packing = _packing(self.packing, 'SparseDiagonals.packing')
    output_packing = _packing(
        packing if self.output_packing is None else self.output_packing,
        'SparseDiagonals.output_packing',
    )
    copied = []
    seen = set()
    digest = hashlib.sha256()
    digest.update(_json_digest({
        'dimension': dimension,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'packing': packing,
        'output_packing': output_packing,
    }).encode())
    for index, diagonal in sorted(self.diagonals, key=lambda item: item[0]):
      if not isinstance(index, (int, np.integer)):
        raise TypeError(f'diagonal index must be an int, got {index!r}.')
      index = int(index)
      if not 0 <= index < dimension:
        raise ValueError(
            f'diagonal index {index} outside [0, {dimension}).'
        )
      if index in seen:
        raise ValueError(f'duplicate diagonal index {index}.')
      seen.add(index)
      array = _readonly_array(diagonal, ndim=1)
      if array.dtype.kind == 'c':
        raise TypeError('SparseDiagonals values must be real.')
      if array.shape != (dimension,):
        raise ValueError(
            f'diagonal {index} must have length {dimension}, got '
            f'{array.shape}.'
        )
      copied.append((index, array))
      digest.update(str(index).encode())
      digest.update(_array_digest(array).encode())
    if not copied:
      raise ValueError('SparseDiagonals must contain at least one diagonal.')
    if math.prod(input_shape) > dimension:
      raise ValueError('input_shape does not fit the diagonal dimension.')
    if math.prod(output_shape) > dimension:
      raise ValueError('output_shape does not fit the diagonal dimension.')
    object.__setattr__(self, 'dimension', dimension)
    object.__setattr__(self, 'diagonals', tuple(copied))
    object.__setattr__(self, 'input_shape', input_shape)
    object.__setattr__(self, 'output_shape', output_shape)
    object.__setattr__(self, 'packing', packing)
    object.__setattr__(self, 'output_packing', output_packing)
    object.__setattr__(self, 'digest', digest.hexdigest())

  def as_dict(self) -> dict[int, np.ndarray]:
    return {index: diagonal for index, diagonal in self.diagonals}

  def diagonal_maxima(self) -> dict[int, float]:
    return {
        index: float(np.max(np.abs(diagonal)))
        for index, diagonal in self.diagonals
    }


@dataclass(frozen=True)
class _LazySparseDiagonals:
  """Internal metadata plus a framework-owned matrix-free diagonal source.

  The Packing lowerer constructs this value only from its immutable transform
  recipes. Caller-defined matrices use ``SparseDiagonals``, which copies and
  fingerprints concrete values rather than trusting source-supplied identity
  or maxima.
  """

  dimension: int
  source: object
  input_shape: tuple[int, ...]
  output_shape: tuple[int, ...]
  packing: str = 'slots'
  output_packing: Optional[str] = None
  digest: str = field(init=False)

  _constant_kind: ClassVar[str] = 'lazy_sparse'

  def __post_init__(self):
    if (
        isinstance(self.dimension, bool)
        or not isinstance(self.dimension, (int, np.integer))
        or int(self.dimension) <= 0
    ):
      raise ValueError('dimension must be a positive int.')
    dimension = int(self.dimension)
    input_shape = _logical_shape(
        self.input_shape, '_LazySparseDiagonals.input_shape'
    )
    output_shape = _logical_shape(
        self.output_shape, '_LazySparseDiagonals.output_shape'
    )
    packing = _packing(self.packing, '_LazySparseDiagonals.packing')
    output_packing = _packing(
        packing if self.output_packing is None else self.output_packing,
        '_LazySparseDiagonals.output_packing',
    )
    if math.prod(input_shape) > dimension:
      raise ValueError('input_shape does not fit the diagonal dimension.')
    if math.prod(output_shape) > dimension:
      raise ValueError('output_shape does not fit the diagonal dimension.')
    source_digest = getattr(self.source, 'digest', None)
    if not isinstance(source_digest, str) or not source_digest:
      raise TypeError('source must expose a non-empty string digest.')
    for method_name in (
        'as_dict',
        'diagonal_maxima',
        'materialize_diagonals',
    ):
      if not callable(getattr(self.source, method_name, None)):
        raise TypeError(f'source must expose {method_name}().')
    object.__setattr__(self, 'dimension', dimension)
    object.__setattr__(self, 'input_shape', input_shape)
    object.__setattr__(self, 'output_shape', output_shape)
    object.__setattr__(self, 'packing', packing)
    object.__setattr__(self, 'output_packing', output_packing)
    object.__setattr__(
        self,
        'digest',
        _json_digest({
            'source': source_digest,
            'dimension': dimension,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'packing': packing,
            'output_packing': output_packing,
        }),
    )

  def _normalize_diagonals(
      self,
      diagonals,
      *,
      expected_indices: Optional[tuple[int, ...]] = None,
  ) -> dict[int, np.ndarray]:
    if not isinstance(diagonals, dict):
      raise TypeError('lazy sparse source must return a dict of diagonals.')
    normalized = {}
    for index, diagonal in diagonals.items():
      if (
          isinstance(index, bool)
          or not isinstance(index, (int, np.integer))
          or not 0 <= int(index) < self.dimension
      ):
        raise ValueError(
            f'lazy diagonal index {index!r} outside '
            f'[0, {self.dimension}).'
        )
      array = _readonly_array(diagonal, ndim=1)
      if array.dtype.kind == 'c':
        raise TypeError('lazy sparse diagonal values must be real.')
      if array.shape != (self.dimension,):
        raise ValueError(
            f'lazy diagonal {index} must have length {self.dimension}.'
        )
      normalized[int(index)] = array
    if expected_indices is not None:
      expected = set(expected_indices)
      actual = set(normalized)
      if actual != expected:
        raise ValueError(
            'lazy sparse source returned diagonal indices '
            f'{tuple(sorted(actual))}, expected {tuple(sorted(expected))}.'
        )
      normalized = {
          index: normalized[index] for index in expected_indices
      }
    elif not normalized:
      normalized[0] = _readonly_array(
          np.zeros(self.dimension, dtype=np.float64), ndim=1
      )
    return normalized

  def as_dict(self) -> dict[int, np.ndarray]:
    return self._normalize_diagonals(self.source.as_dict())

  def materialize_diagonals(
      self, indices
  ) -> dict[int, np.ndarray]:
    """Materialize exactly one requested subset of physical diagonals."""
    try:
      raw_indices = tuple(indices)
    except TypeError as error:
      raise TypeError('lazy diagonal indices must be iterable.') from error
    normalized_indices = []
    seen = set()
    for index in raw_indices:
      if (
          isinstance(index, bool)
          or not isinstance(index, (int, np.integer))
          or not 0 <= int(index) < self.dimension
      ):
        raise ValueError(
            f'lazy diagonal index {index!r} outside '
            f'[0, {self.dimension}).'
        )
      index = int(index)
      if index in seen:
        raise ValueError(f'duplicate lazy diagonal index {index}.')
      seen.add(index)
      normalized_indices.append(index)
    requested = tuple(normalized_indices)
    diagonals = self.source.materialize_diagonals(requested)
    return self._normalize_diagonals(
        diagonals, expected_indices=requested
    )

  def diagonal_maxima(self) -> dict[int, float]:
    maxima = self.source.diagonal_maxima()
    if not isinstance(maxima, dict):
      raise TypeError(
          'lazy sparse source diagonal_maxima() must return a dict.'
      )
    normalized = {}
    for index, maximum in maxima.items():
      if (
          isinstance(index, bool)
          or not isinstance(index, (int, np.integer))
          or not 0 <= int(index) < self.dimension
      ):
        raise ValueError(
            f'lazy diagonal index {index!r} outside '
            f'[0, {self.dimension}).'
        )
      maximum = float(maximum)
      if not math.isfinite(maximum) or maximum < 0:
        raise ValueError('lazy diagonal maxima must be finite and non-negative.')
      if maximum:
        normalized[int(index)] = maximum
    return normalized


PlainConstant = PlainSlots


MatrixConstant = Union[DenseMatrix, SparseDiagonals, _LazySparseDiagonals]


def _validate_name(name: Optional[str], description: str = 'name') -> None:
  if name is not None and (not isinstance(name, str) or not name.strip()):
    raise ValueError(f'{description} must be None or a non-empty string.')


def _validate_module(value, description: str) -> None:
  if not isinstance(value, _LogicalLayer):
    raise TypeError(
        f'{description} must be an packing.Module, got {type(value).__name__}.'
    )


def _validate_optional_int(value, description: str) -> None:
  if value is not None and (
      isinstance(value, bool) or not isinstance(value, (int, np.integer))
  ):
    raise TypeError(f'{description} must be None or an int.')


@dataclass(frozen=True, kw_only=True)
class _LogicalLayer:
  """Base class for one immutable, explicitly composed HE layer."""

  name: Optional[str] = None

  def __post_init__(self):
    _validate_name(self.name)


# The declarative module IR that used to live here -- Sequential, Rotate,
# MulPlain, ParallelSum and the rest -- described a graph a caller built by
# hand. packing.pack now derives the same operations from a vectorized
# program, so there is one way to describe a model and one way to pack it.


_PackedOperation = tuple[str, str, tuple[str, ...], Any]


_ARITY = {
    'add': 2,
    'sub': 2,
    'mul': 2,
    'square': 1,
    'rotate': 1,
    'rescale': 1,
    'level_reduce': 1,
    'add_plain': 1,
    'mul_plain': 1,
    'matvec': 1,
    'bootstrap': 1,
}


def _fingerprint_argument(value):
  kind = getattr(value, '_constant_kind', None)
  if kind is not None:
    return {'constant_kind': kind, 'digest': value.digest}
  if isinstance(value, tuple):
    return [_fingerprint_argument(item) for item in value]
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  return repr(value)


@dataclass(frozen=True)
class _Conv2d(_LogicalLayer):
  """Logical unbatched 2-D cross-correlation over a ``(C, H, W)`` tensor."""

  weight: np.ndarray
  bias: Optional[np.ndarray] = None
  stride: tuple[int, int] = (1, 1)
  padding: tuple[int, int] = (0, 0)

  def __post_init__(self):
    super().__post_init__()
    weight = _readonly_real_array(self.weight, ndim=4)
    bias = self.bias
    if bias is not None:
      bias = _readonly_real_array(bias, ndim=1)
      if bias.shape != (weight.shape[0],):
        raise ValueError(
            f'Conv2d.bias must have shape {(weight.shape[0],)}, '
            f'got {bias.shape}.'
        )
    object.__setattr__(self, 'weight', weight)
    object.__setattr__(self, 'bias', bias)
    object.__setattr__(
        self, 'stride', _spatial_pair(self.stride, 'Conv2d.stride')
    )
    object.__setattr__(
        self,
        'padding',
        _spatial_pair(
            self.padding, 'Conv2d.padding', allow_zero=True
        ),
    )


@dataclass(frozen=True)
class _AvgPool2d(_LogicalLayer):
  """Logical unbatched 2-D average pooling."""

  kernel_size: tuple[int, int]
  stride: Optional[tuple[int, int]] = None
  padding: tuple[int, int] = (0, 0)
  count_include_pad: bool = True

  def __post_init__(self):
    super().__post_init__()
    kernel_size = _spatial_pair(
        self.kernel_size, 'AvgPool2d.kernel_size'
    )
    stride = (
        kernel_size
        if self.stride is None
        else _spatial_pair(self.stride, 'AvgPool2d.stride')
    )
    padding = _spatial_pair(
        self.padding, 'AvgPool2d.padding', allow_zero=True
    )
    if not isinstance(self.count_include_pad, bool):
      raise TypeError('AvgPool2d.count_include_pad must be a bool.')
    if any(
        pad > kernel // 2
        for pad, kernel in zip(padding, kernel_size, strict=True)
    ):
      raise ValueError(
          'AvgPool2d.padding must not exceed half the kernel size.'
      )
    object.__setattr__(self, 'kernel_size', kernel_size)
    object.__setattr__(self, 'stride', stride)
    object.__setattr__(self, 'padding', padding)


@dataclass(frozen=True)
class _AdaptiveAvgPool2d(_LogicalLayer):
  """Logical unbatched adaptive average pooling."""

  output_size: tuple[int, int]

  def __post_init__(self):
    super().__post_init__()
    object.__setattr__(
        self,
        'output_size',
        _spatial_pair(
            self.output_size, 'AdaptiveAvgPool2d.output_size'
        ),
    )


@dataclass(frozen=True)
class _Flatten(_LogicalLayer):
  """Flatten every logical tensor dimension in channel-major order."""


@dataclass(frozen=True)
class _Dense(_LogicalLayer):
  """Logical dense transform over a flat feature vector."""

  weight: np.ndarray
  bias: Optional[np.ndarray] = None

  def __post_init__(self):
    super().__post_init__()
    weight = _readonly_real_array(self.weight, ndim=2)
    bias = self.bias
    if bias is not None:
      bias = _readonly_real_array(bias, ndim=1)
      if bias.shape != (weight.shape[0],):
        raise ValueError(
            f'Dense.bias must have shape {(weight.shape[0],)}, '
            f'got {bias.shape}.'
        )
    object.__setattr__(self, 'weight', weight)
    object.__setattr__(self, 'bias', bias)



def _array_digest(value: np.ndarray) -> str:
  value = np.asarray(value)
  digest = hashlib.sha256()
  digest.update(str(value.dtype).encode())
  digest.update(repr(tuple(value.shape)).encode())
  digest.update(value.tobytes(order='C'))
  return digest.hexdigest()


def _json_digest(value) -> str:
  return hashlib.sha256(
      json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
  ).hexdigest()


def _layer_payload(layer: _LogicalLayer | None):
  if layer is None:
    return None
  payload = {
      'type': type(layer).__name__,
      'name': layer.name,
  }
  if isinstance(layer, _Conv2d):
    payload.update({
        'weight': _array_digest(layer.weight),
        'stride': layer.stride,
        'padding': layer.padding,
    })
  elif isinstance(layer, _AvgPool2d):
    payload.update({
        'kernel_size': layer.kernel_size,
        'stride': layer.stride,
        'padding': layer.padding,
        'count_include_pad': layer.count_include_pad,
    })
  elif isinstance(layer, _AdaptiveAvgPool2d):
    payload['output_size'] = layer.output_size
  elif isinstance(layer, _Dense):
    payload['weight'] = _array_digest(layer.weight)
  return payload



def _channel_major(shape: tuple[int, ...]) -> TensorSpec:
  return TensorSpec(shape, ChannelMajor())


def _output_dimension(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> int:
  output = (input_size + 2 * padding - kernel_size) // stride + 1
  if output <= 0:
    raise ValueError(
        'kernel, stride, and padding produce a non-positive output size.'
    )
  return output


def _conv_output_spec(
    layer: _Conv2d,
    input_spec: TensorSpec,
) -> TensorSpec:
  if len(input_spec.shape) != 3:
    raise ValueError('Conv2d expects a rank-3 CHW input.')
  channels, height, width = input_spec.shape
  out_channels, weight_channels, kernel_h, kernel_w = layer.weight.shape
  if channels != weight_channels:
    raise ValueError(
        f'Conv2d weight expects {weight_channels} channels, got {channels}.'
    )
  output_h = _output_dimension(
      height, kernel_h, layer.stride[0], layer.padding[0]
  )
  output_w = _output_dimension(
      width, kernel_w, layer.stride[1], layer.padding[1]
  )
  return _channel_major((out_channels, output_h, output_w))


def _pool_output_spec(
    layer: _AvgPool2d,
    input_spec: TensorSpec,
) -> TensorSpec:
  if len(input_spec.shape) != 3:
    raise ValueError('AvgPool2d expects a rank-3 CHW input.')
  channels, height, width = input_spec.shape
  output_h = _output_dimension(
      height, layer.kernel_size[0], layer.stride[0], layer.padding[0]
  )
  output_w = _output_dimension(
      width, layer.kernel_size[1], layer.stride[1], layer.padding[1]
  )
  return _channel_major((channels, output_h, output_w))


def _adaptive_output_spec(
    layer: _AdaptiveAvgPool2d,
    input_spec: TensorSpec,
) -> TensorSpec:
  if len(input_spec.shape) != 3:
    raise ValueError('AdaptiveAvgPool2d expects a rank-3 CHW input.')
  return _channel_major(
      (input_spec.shape[0], *layer.output_size)
  )


def _pool_window(
    layer: _AvgPool2d,
    input_spec: TensorSpec,
    output_h: int,
    output_w: int,
) -> tuple[tuple[tuple[int, int], ...], int]:
  _, height, width = input_spec.shape
  start_h = output_h * layer.stride[0] - layer.padding[0]
  start_w = output_w * layer.stride[1] - layer.padding[1]
  positions = tuple(
      (height_index, width_index)
      for kernel_h in range(layer.kernel_size[0])
      for kernel_w in range(layer.kernel_size[1])
      if 0 <= (height_index := start_h + kernel_h) < height
      and 0 <= (width_index := start_w + kernel_w) < width
  )
  denominator = (
      math.prod(layer.kernel_size)
      if layer.count_include_pad
      else len(positions)
  )
  if denominator == 0:
    raise ValueError('AvgPool2d produced an empty averaging window.')
  return positions, denominator


def _adaptive_window(
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    output_h: int,
    output_w: int,
) -> tuple[tuple[tuple[int, int], ...], int]:
  start_h = math.floor(output_h * input_height / output_height)
  end_h = math.ceil((output_h + 1) * input_height / output_height)
  start_w = math.floor(output_w * input_width / output_width)
  end_w = math.ceil((output_w + 1) * input_width / output_width)
  positions = tuple(
      (height_index, width_index)
      for height_index in range(start_h, end_h)
      for width_index in range(start_w, end_w)
  )
  return positions, len(positions)


def _conv_entries(
    layer: _Conv2d,
    input_spec: TensorSpec,
    output_spec: TensorSpec,
) -> Iterator[tuple[int, int, float]]:
  channels, height, width = input_spec.shape
  out_channels, _, kernel_h, kernel_w = layer.weight.shape
  _, output_h, output_w = output_spec.shape
  stride_h, stride_w = layer.stride
  pad_h, pad_w = layer.padding
  for out_channel in range(out_channels):
    for out_h in range(output_h):
      for out_w in range(output_w):
        row = output_spec.coordinate_to_slot(
            (out_channel, out_h, out_w)
        )
        for channel in range(channels):
          for kernel_row in range(kernel_h):
            input_h = out_h * stride_h + kernel_row - pad_h
            if not 0 <= input_h < height:
              continue
            for kernel_column in range(kernel_w):
              input_w = out_w * stride_w + kernel_column - pad_w
              if not 0 <= input_w < width:
                continue
              value = layer.weight[
                  out_channel, channel, kernel_row, kernel_column
              ]
              if value:
                column = input_spec.coordinate_to_slot(
                    (channel, input_h, input_w)
                )
                yield row, column, float(value)


def _pool_entries(
    layer: _AvgPool2d,
    input_spec: TensorSpec,
    output_spec: TensorSpec,
) -> Iterator[tuple[int, int, float]]:
  channels, output_height, output_width = output_spec.shape
  for channel in range(channels):
    for output_h in range(output_height):
      for output_w in range(output_width):
        positions, denominator = _pool_window(
            layer, input_spec, output_h, output_w
        )
        row = output_spec.coordinate_to_slot(
            (channel, output_h, output_w)
        )
        for input_h, input_w in positions:
          column = input_spec.coordinate_to_slot(
              (channel, input_h, input_w)
          )
          yield row, column, 1.0 / denominator


def _adaptive_entries(
    layer: _AdaptiveAvgPool2d,
    input_spec: TensorSpec,
    output_spec: TensorSpec,
) -> Iterator[tuple[int, int, float]]:
  channels, input_height, input_width = input_spec.shape
  _, output_height, output_width = output_spec.shape
  for channel in range(channels):
    for output_h in range(output_height):
      for output_w in range(output_width):
        positions, denominator = _adaptive_window(
            input_height,
            input_width,
            output_height,
            output_width,
            output_h,
            output_w,
        )
        row = output_spec.coordinate_to_slot(
            (channel, output_h, output_w)
        )
        for input_h, input_w in positions:
          column = input_spec.coordinate_to_slot(
              (channel, input_h, input_w)
          )
          yield row, column, 1.0 / denominator


def _dense_entries(
    layer: _Dense,
    input_spec: TensorSpec,
    output_spec: TensorSpec,
) -> Iterator[tuple[int, int, float]]:
  for output_index in range(layer.weight.shape[0]):
    row = output_spec.coordinate_to_slot((output_index,))
    for input_index in range(layer.weight.shape[1]):
      value = layer.weight[output_index, input_index]
      if value:
        column = input_spec.coordinate_to_slot((input_index,))
        yield row, column, float(value)


def _flatten_entries(
    input_spec: TensorSpec,
    output_spec: TensorSpec,
) -> Iterator[tuple[int, int, float]]:
  for output_index, coordinate in enumerate(np.ndindex(input_spec.shape)):
    row = output_spec.coordinate_to_slot((output_index,))
    column = input_spec.coordinate_to_slot(coordinate)
    yield row, column, 1.0


def _fused_conv_pool_entries(
    convolution: _Conv2d,
    pool: _LogicalLayer,
    input_spec: TensorSpec,
    convolution_spec: TensorSpec,
    output_spec: TensorSpec,
) -> Iterator[tuple[int, int, float]]:
  channels, input_height, input_width = input_spec.shape
  out_channels, _, kernel_h, kernel_w = convolution.weight.shape
  _, convolution_height, convolution_width = convolution_spec.shape
  _, output_height, output_width = output_spec.shape
  stride_h, stride_w = convolution.stride
  pad_h, pad_w = convolution.padding
  for out_channel in range(out_channels):
    for output_h in range(output_height):
      for output_w in range(output_width):
        if isinstance(pool, _AvgPool2d):
          positions, denominator = _pool_window(
              pool, convolution_spec, output_h, output_w
          )
        else:
          positions, denominator = _adaptive_window(
              convolution_height,
              convolution_width,
              output_height,
              output_width,
              output_h,
              output_w,
          )
        row = output_spec.coordinate_to_slot(
            (out_channel, output_h, output_w)
        )
        coefficients: dict[int, float] = {}
        for convolution_h, convolution_w in positions:
          for channel in range(channels):
            for kernel_row in range(kernel_h):
              input_h = (
                  convolution_h * stride_h + kernel_row - pad_h
              )
              if not 0 <= input_h < input_height:
                continue
              for kernel_column in range(kernel_w):
                input_w = (
                    convolution_w * stride_w + kernel_column - pad_w
                )
                if not 0 <= input_w < input_width:
                  continue
                value = (
                    convolution.weight[
                        out_channel, channel, kernel_row, kernel_column
                    ] / denominator
                )
                if value:
                  column = input_spec.coordinate_to_slot(
                      (channel, input_h, input_w)
                  )
                  coefficients[column] = coefficients.get(column, 0.0) + value
        for column, value in coefficients.items():
          if value:
            yield row, column, float(value)


@dataclass(frozen=True)
class _TransformRecipe:
  """One logical layer as an implicit matrix over slot indices.

  Conv2d, Dense and the pooling layers are all linear maps, so each lowers to a
  single MatVec operator; ``he_ops`` records what a subclass becomes. Two access
  patterns are offered because their consumers differ: ``entries`` sweeps the
  whole matrix in row order, while ``_coefficient`` answers one cell in closed
  form so ``diagonal_entries`` can materialize a chosen subset of BSGS
  diagonals without enumerating anything.

  Subclasses supply ``kind``, ``he_ops``, ``entries`` and ``_coefficient``.
  """

  input_spec: TensorSpec
  output_spec: TensorSpec
  layer: _LogicalLayer | None = None
  second_layer: _LogicalLayer | None = None
  intermediate_spec: TensorSpec | None = None
  digest: str = field(init=False)
  _input_coordinate_index: tuple[
      tuple[int, ...] | None, ...
  ] = field(init=False, repr=False, compare=False)

  # ``kind`` stays in the digest payload, so its string must not change: the
  # recipe digest reaches Mapping.fingerprint, which seals an initialized
  # context to one model.
  kind: ClassVar[str] = ''
  he_ops: ClassVar[tuple[str, ...]] = ()

  def __post_init__(self):
    input_coordinate_index: list[tuple[int, ...] | None] = [
        None
    ] * self.input_spec.physical_size
    for coordinate in np.ndindex(self.input_spec.shape):
      input_coordinate_index[
          self.input_spec.coordinate_to_slot(coordinate)
      ] = coordinate
    object.__setattr__(
        self,
        '_input_coordinate_index',
        tuple(input_coordinate_index),
    )
    object.__setattr__(
        self,
        'digest',
        _json_digest({
            'kind': self.kind,
            'input_shape': self.input_spec.shape,
            'input_packing': self.input_spec.packing,
            'input_layout': self.input_spec.layout_fingerprint,
            'output_shape': self.output_spec.shape,
            'output_packing': self.output_spec.packing,
            'output_layout': self.output_spec.layout_fingerprint,
            'layer': _layer_payload(self.layer),
            'second_layer': _layer_payload(self.second_layer),
            'intermediate': (
                None
                if self.intermediate_spec is None
                else self.intermediate_spec.shape
            ),
        }),
    )

  def entries(self) -> Iterator[tuple[int, int, float]]:
    """Sweep the whole matrix as ``(row, column, coefficient)`` in row order."""
    raise NotImplementedError(
        f'{type(self).__name__} must define entries().'
    )

  def _coefficient(
      self,
      output_coordinate: tuple[int, ...],
      input_coordinate: tuple[int, ...],
  ) -> float:
    """Return one logical matrix coefficient without enumerating the matrix."""
    raise NotImplementedError(
        f'{type(self).__name__} must define _coefficient().'
    )

  def diagonal_entries(
      self,
      indices: tuple[int, ...],
      dimension: int,
  ) -> Iterator[tuple[int, int, float]]:
    """Generate requested BSGS diagonals without replaying ``entries()``."""
    if not indices:
      return
    for output_coordinate in np.ndindex(self.output_spec.shape):
      row = self.output_spec.coordinate_to_slot(output_coordinate)
      for diagonal in indices:
        column = (row + diagonal) % dimension
        if column >= len(self._input_coordinate_index):
          continue
        input_coordinate = self._input_coordinate_index[column]
        if input_coordinate is None:
          continue
        value = self._coefficient(output_coordinate, input_coordinate)
        if value:
          yield diagonal, row, value


@dataclass(frozen=True)
class Conv2dLowering(_TransformRecipe):
  """Lower ``Conv2d`` to one MatVec operator plus an optional AddPlain."""

  kind: ClassVar[str] = 'conv2d'
  he_ops: ClassVar[tuple[str, ...]] = ('matvec', 'add_plain')

  def entries(self) -> Iterator[tuple[int, int, float]]:
    return _conv_entries(self.layer, self.input_spec, self.output_spec)

  def _coefficient(self, output_coordinate, input_coordinate) -> float:
    output_channel, output_h, output_w = output_coordinate
    input_channel, input_h, input_w = input_coordinate
    kernel_h = (
        input_h - output_h * self.layer.stride[0] + self.layer.padding[0]
    )
    kernel_w = (
        input_w - output_w * self.layer.stride[1] + self.layer.padding[1]
    )
    if not (
        0 <= kernel_h < self.layer.weight.shape[2]
        and 0 <= kernel_w < self.layer.weight.shape[3]
    ):
      return 0.0
    return float(self.layer.weight[
        output_channel, input_channel, kernel_h, kernel_w
    ])


@dataclass(frozen=True)
class AvgPoolLowering(_TransformRecipe):
  """Lower ``AvgPool2d``. Averaging is linear, so it is one MatVec."""

  kind: ClassVar[str] = 'avg_pool2d'
  he_ops: ClassVar[tuple[str, ...]] = ('matvec',)

  def entries(self) -> Iterator[tuple[int, int, float]]:
    return _pool_entries(self.layer, self.input_spec, self.output_spec)

  def _coefficient(self, output_coordinate, input_coordinate) -> float:
    output_channel, output_h, output_w = output_coordinate
    input_channel, input_h, input_w = input_coordinate
    if output_channel != input_channel:
      return 0.0
    positions, denominator = _pool_window(
        self.layer, self.input_spec, output_h, output_w
    )
    return 1.0 / denominator if (input_h, input_w) in positions else 0.0


@dataclass(frozen=True)
class AdaptiveAvgPoolLowering(_TransformRecipe):
  """Lower ``AdaptiveAvgPool2d``; the window is derived from the ratio."""

  kind: ClassVar[str] = 'adaptive_avg_pool2d'
  he_ops: ClassVar[tuple[str, ...]] = ('matvec',)

  def entries(self) -> Iterator[tuple[int, int, float]]:
    return _adaptive_entries(self.layer, self.input_spec, self.output_spec)

  def _coefficient(self, output_coordinate, input_coordinate) -> float:
    output_channel, output_h, output_w = output_coordinate
    input_channel, input_h, input_w = input_coordinate
    if output_channel != input_channel:
      return 0.0
    _, input_height, input_width = self.input_spec.shape
    _, output_height, output_width = self.output_spec.shape
    positions, denominator = _adaptive_window(
        input_height,
        input_width,
        output_height,
        output_width,
        output_h,
        output_w,
    )
    return 1.0 / denominator if (input_h, input_w) in positions else 0.0


@dataclass(frozen=True)
class FullyConnectedLowering(_TransformRecipe):
  """Lower ``Dense`` to one MatVec operator plus an optional AddPlain."""

  kind: ClassVar[str] = 'dense'
  he_ops: ClassVar[tuple[str, ...]] = ('matvec', 'add_plain')

  def entries(self) -> Iterator[tuple[int, int, float]]:
    return _dense_entries(self.layer, self.input_spec, self.output_spec)

  def _coefficient(self, output_coordinate, input_coordinate) -> float:
    return float(self.layer.weight[
        output_coordinate[0], input_coordinate[0]
    ])


@dataclass(frozen=True)
class FlattenLowering(_TransformRecipe):
  """Lower ``Flatten``: a permutation, emitted only when the layout changes."""

  kind: ClassVar[str] = 'flatten'
  he_ops: ClassVar[tuple[str, ...]] = ('matvec',)

  def entries(self) -> Iterator[tuple[int, int, float]]:
    return _flatten_entries(self.input_spec, self.output_spec)

  def _coefficient(self, output_coordinate, input_coordinate) -> float:
    expected_input = tuple(np.unravel_index(
        output_coordinate[0], self.input_spec.shape
    ))
    return 1.0 if input_coordinate == expected_input else 0.0


@dataclass(frozen=True)
class FusedConvPoolLowering(_TransformRecipe):
  """Lower ``Conv2d`` immediately followed by pooling as one MatVec.

  Composing both linear maps into a single matrix saves a multiplicative
  level, which is the scarce resource in the modulus chain.
  """

  kind: ClassVar[str] = 'conv_pool'
  he_ops: ClassVar[tuple[str, ...]] = ('matvec', 'add_plain')

  def entries(self) -> Iterator[tuple[int, int, float]]:
    return _fused_conv_pool_entries(
        self.layer,
        self.second_layer,
        self.input_spec,
        self.intermediate_spec,
        self.output_spec,
    )

  def _coefficient(self, output_coordinate, input_coordinate) -> float:
    output_channel, output_h, output_w = output_coordinate
    input_channel, input_h, input_w = input_coordinate
    _, convolution_height, convolution_width = self.intermediate_spec.shape
    _, output_height, output_width = self.output_spec.shape
    if isinstance(self.second_layer, _AvgPool2d):
      positions, denominator = _pool_window(
          self.second_layer, self.intermediate_spec, output_h, output_w
      )
    else:
      positions, denominator = _adaptive_window(
          convolution_height,
          convolution_width,
          output_height,
          output_width,
          output_h,
          output_w,
      )
    coefficient = 0.0
    for convolution_h, convolution_w in positions:
      kernel_h = (
          input_h
          - convolution_h * self.layer.stride[0]
          + self.layer.padding[0]
      )
      kernel_w = (
          input_w
          - convolution_w * self.layer.stride[1]
          + self.layer.padding[1]
      )
      if (
          0 <= kernel_h < self.layer.weight.shape[2]
          and 0 <= kernel_w < self.layer.weight.shape[3]
      ):
        coefficient += float(self.layer.weight[
            output_channel, input_channel, kernel_h, kernel_w
        ]) / denominator
    return coefficient


@dataclass(frozen=True)
class _MatrixFreeSource:
  recipe: _TransformRecipe
  dimension: int
  digest: str = field(init=False)

  def __post_init__(self):
    object.__setattr__(
        self,
        'digest',
        _json_digest({
            'recipe': self.recipe.digest,
            'dimension': self.dimension,
        }),
    )

  def as_dict(self) -> dict[int, np.ndarray]:
    diagonals: dict[int, np.ndarray] = {}
    for row, column, value in self.recipe.entries():
      index = (column - row) % self.dimension
      diagonal = diagonals.get(index)
      if diagonal is None:
        diagonal = np.zeros(self.dimension, dtype=np.float64)
        diagonals[index] = diagonal
      diagonal[row] += value
    return {
        index: diagonal
        for index, diagonal in sorted(diagonals.items())
        if np.any(diagonal)
    }

  def materialize_diagonals(
      self, indices: tuple[int, ...]
  ) -> dict[int, np.ndarray]:
    """Materialize requested vectors through the recipe's diagonal index."""
    requested = tuple(indices)
    normalized = []
    for index in requested:
      if (
          isinstance(index, bool)
          or not isinstance(index, (int, np.integer))
          or not 0 <= int(index) < self.dimension
      ):
        raise ValueError(
            f'diagonal indices must be ints in [0, {self.dimension}).'
        )
      normalized.append(int(index))
    requested = tuple(normalized)
    requested_set = set(requested)
    if len(requested_set) != len(requested):
      raise ValueError('diagonal indices must be unique.')
    diagonals = {
        index: np.zeros(self.dimension, dtype=np.float64)
        for index in requested
    }
    diagonal_entries = getattr(self.recipe, 'diagonal_entries', None)
    if not callable(diagonal_entries):
      raise TypeError(
          'matrix-free recipes must expose '
          'diagonal_entries(indices, dimension).'
      )
    for diagonal, row, value in diagonal_entries(
        requested, self.dimension
    ):
      diagonals[diagonal][row] += value
    return diagonals

  def diagonal_maxima(self) -> dict[int, float]:
    """Return exact maxima after additive entries at each matrix cell coalesce."""
    maxima: dict[int, float] = {}
    current_row = None
    row_values: dict[int, float] = {}

    def flush_row() -> None:
      for diagonal, value in row_values.items():
        if value:
          maxima[diagonal] = max(
              maxima.get(diagonal, 0.0), abs(value)
          )

    for row, column, value in self.recipe.entries():
      if current_row is None:
        current_row = row
      elif row < current_row:
        raise ValueError(
            'matrix-free recipes must emit entries in non-decreasing row order.'
        )
      elif row != current_row:
        flush_row()
        row_values.clear()
        current_row = row
      diagonal = (column - row) % self.dimension
      row_values[diagonal] = row_values.get(diagonal, 0.0) + value
    flush_row()
    return maxima



def _build_conv2d(parameters, weights, role='weight'):
  return _Conv2d(
      name=parameters.get('name'),
      weight=weights[role],
      bias=weights.get('bias'),
      stride=parameters.get('stride', (1, 1)),
      padding=parameters.get('padding', (0, 0)),
  )


def _build_avg_pool(parameters, weights, role=None):
  del weights, role
  return _AvgPool2d(
      name=parameters.get('name'),
      kernel_size=parameters['kernel_size'],
      stride=parameters.get('stride'),
      padding=parameters.get('padding', (0, 0)),
      count_include_pad=parameters.get('count_include_pad', True),
  )


def _build_adaptive_pool(parameters, weights, role=None):
  del weights, role
  return _AdaptiveAvgPool2d(
      name=parameters.get('name'),
      output_size=parameters['output_size'],
  )


def _build_fused_pool(parameters, weights, role=None):
  """Rebuild the pool half of a fused conv+pool from ``pool_``-prefixed keys.

  The pool's parameters are flattened into the layer's own parameter tuple
  rather than nested, because a VectorizedLayer normalizes its parameters to
  hashable scalars and tuples -- a nested mapping has no stable digest.
  """
  pool = {
      name[len('pool_'):]: value
      for name, value in parameters.items()
      if name.startswith('pool_') and name != 'pool_kind'
  }
  pool.setdefault('name', parameters.get('name'))
  if parameters['pool_kind'] == 'avg_pool2d':
    return _build_avg_pool(pool, weights)
  return _build_adaptive_pool(pool, weights)


def _build_dense(parameters, weights, role='weight'):
  return _Dense(
      name=parameters.get('name'),
      weight=weights[role],
      bias=weights.get('bias'),
  )



# Which recipe each frontend template becomes, and how to rebuild the logical
# layer it needs. The frontend names a template; this table is what that name
# means physically, so the frontend never has to know.
# template id -> (version, recipe, layer builder, second-layer builder).
# The version is the contract: a layer captured one when it was built, and a
# program digested against that version must not be packed by a different
# implementation of it. Bumping an entry here means bumping the frontend
# template that names it.
@dataclass(frozen=True)
class TemplateLowering:
  """How one frontend template becomes slot arithmetic.

  A template names a lowering; this is what that name means physically. Two
  ways to supply one:

  * ``recipe_class`` plus optional layer builders, for a realization built
    from the matrix-free recipes here;
  * ``recipe_factory``, a callable ``(layer, weights) -> recipe``, for a
    genuinely new realization. Nothing of ours has to be involved: the object
    it returns need only satisfy the recipe protocol, which is three members.

    ``digest`` -- a non-empty ``str`` that is stable for identical arithmetic
    and different for different arithmetic. It reaches the packed program's
    fingerprint and from there the Mapping fingerprint that seals a context to
    one model, so deriving it from anything per-process (``id()``, an
    unsalted-hash of a repr) would make two runs of the same program disagree.

    ``entries() -> Iterable[(row, column, value)]`` -- the whole matrix in
    **non-decreasing row order**. The ordering is required: ``diagonal_maxima``
    coalesces additive contributions per row as it streams, and would
    under-report a maximum if a row reappeared later.

    ``diagonal_entries(indices, dimension) -> Iterable[(diagonal, row, value)]``
    -- only the requested BSGS diagonals, without replaying the whole matrix.
    This is the point of the protocol: a conv over 16384 slots is a matrix of
    16384**2 entries that is almost entirely zero, and BSGS asks for a handful
    of its diagonals.

  ``version`` is the contract. A layer captures the frontend template version
  it was built against, and packing refuses to lower a layer whose captured
  version is not the one registered here -- otherwise replacing a lowering
  would silently redefine a program that was already digested.
  """

  version: str
  recipe_class: Optional[type] = None
  layer_builder: Optional[Callable[..., Any]] = None
  second_layer_builder: Optional[Callable[..., Any]] = None
  recipe_factory: Optional[Callable[..., Any]] = None

  def __post_init__(self):
    if not isinstance(self.version, str) or not self.version:
      raise PackingError(
          f'a lowering needs a non-empty string version, got '
          f'{self.version!r}.'
      )
    if (self.recipe_class is None) == (self.recipe_factory is None):
      raise PackingError(
          'a lowering supplies exactly one of recipe_class or recipe_factory.'
      )


_LOWERINGS: dict[str, TemplateLowering] = {}


def register_template_lowering(
    template_id: str,
    lowering: Optional[TemplateLowering] = None,
    *,
    alias: Optional[str] = None,
    version: Optional[str] = None,
    replace: bool = False,
) -> TemplateLowering:
  """Register how ``template_id`` is realized physically.

  A frontend template is declarative: it says which lowering to run and with
  what parameters, never how. This is the other half, and it lives here
  because emitting slot arithmetic is the packing phase's job -- registering
  one does not put a recipe class back into the frontend.

  ``alias`` reuses an existing lowering under a new template id, which is the
  clean way to give a template a different condition or priority without
  duplicating arithmetic. Pass ``version`` alongside it to record a version
  different from the aliased one.
  """
  if not isinstance(template_id, str) or not template_id:
    raise PackingError(
        f'template_id must be a non-empty string, got {template_id!r}.'
    )
  if version is not None and (not isinstance(version, str) or not version):
    raise PackingError(
        f'version must be a non-empty string, got {version!r}.'
    )
  if alias is not None:
    if lowering is not None:
      raise PackingError('pass either a lowering or alias=, not both.')
    try:
      source = _LOWERINGS[alias]
    except KeyError:
      raise PackingError(
          f'cannot alias unknown lowering {alias!r}; registered: '
          f'{sorted(_LOWERINGS)}.'
      ) from None
    lowering = source if version is None else dataclasses.replace(
        source, version=version
    )
  if lowering is None:
    raise PackingError('register_template_lowering needs a lowering or alias=.')
  previous = _LOWERINGS.get(template_id)
  if previous is not None:
    if not replace:
      raise PackingError(
          f'lowering {template_id!r} is already registered; pass replace=True '
          'to override it.'
      )
    if previous.version == lowering.version:
      raise PackingError(
          f'lowering {template_id!r} is registered at version '
          f'{previous.version!r}; a replacement must declare a new version so '
          'programs built against the old one can detect it.'
      )
  _LOWERINGS[template_id] = lowering
  return lowering


def unregister_template_lowering(template_id: str) -> None:
  """Drop a registered lowering; mainly so tests can undo an override."""
  _LOWERINGS.pop(template_id, None)


def template_lowering(template_id: str) -> TemplateLowering:
  try:
    return _LOWERINGS[template_id]
  except KeyError:
    raise PackingError(
        f'no lowering for template {template_id!r}; registered: '
        f'{sorted(_LOWERINGS)}.'
    ) from None


register_template_lowering('conv2d', TemplateLowering(
    version='1', recipe_class=Conv2dLowering, layer_builder=_build_conv2d,
))
register_template_lowering('conv_pool', TemplateLowering(
    version='1', recipe_class=FusedConvPoolLowering,
    layer_builder=_build_conv2d, second_layer_builder=_build_fused_pool,
))
register_template_lowering('dense', TemplateLowering(
    version='1', recipe_class=FullyConnectedLowering,
    layer_builder=_build_dense,
))
register_template_lowering('avg_pool2d', TemplateLowering(
    version='1', recipe_class=AvgPoolLowering, layer_builder=_build_avg_pool,
))
register_template_lowering('adaptive_avg_pool2d', TemplateLowering(
    version='1', recipe_class=AdaptiveAvgPoolLowering,
    layer_builder=_build_adaptive_pool,
))
register_template_lowering('flatten', TemplateLowering(
    version='1', recipe_class=FlattenLowering,
))


def _fused_bias_factors(
    pool: _LogicalLayer,
    convolution_spec: TensorSpec,
    output_spec: TensorSpec,
) -> np.ndarray | None:
  if not isinstance(pool, _AvgPool2d) or not pool.count_include_pad:
    return None
  factors = np.ones(output_spec.shape, dtype=np.float64)
  channels, output_height, output_width = output_spec.shape
  for channel in range(channels):
    for output_h in range(output_height):
      for output_w in range(output_width):
        positions, denominator = _pool_window(
            pool, convolution_spec, output_h, output_w
        )
        factors[channel, output_h, output_w] = len(positions) / denominator
  return factors


def layer_recipe(layer, weights):
    """The matrix-free recipe a vectorized layer describes.

    ``layer`` is duck-typed: anything exposing template_id, parameters(),
    resolve_weights(), input_spec and output_spec will do. That is what keeps
    this module free of any import from the frontend.
    """
    entry = template_lowering(layer.template_id)
    captured = getattr(layer, 'template_version', entry.version)
    if captured != entry.version:
        raise PackingError(
            f'layer {layer.name!r} was built against template '
            f'{layer.template_id!r} version {captured!r}, but this packer '
            f'implements version {entry.version!r}. Packing it now would '
            'compute something other than what this program was digested as.'
        )
    if entry.recipe_factory is not None:
        return entry.recipe_factory(layer, weights)
    recipe_class = entry.recipe_class
    build_layer = entry.layer_builder
    build_second = entry.second_layer_builder
    parameters = layer.parameters()
    parameters.setdefault('name', layer.name)
    resolved = layer.resolve_weights(weights)
    intermediate = parameters.get('intermediate_shape')
    return recipe_class(
        input_spec=layer.input_spec,
        output_spec=layer.output_spec,
        layer=None if build_layer is None else build_layer(parameters, resolved),
        second_layer=(
            None if build_second is None else build_second(parameters, resolved)
        ),
        intermediate_spec=(
            None if intermediate is None else _channel_major(intermediate)
        ),
    )


def layer_matrix_source(layer, weights, dimension: int):
    """A matrix-free diagonal source over ``dimension`` slots."""
    return _MatrixFreeSource(layer_recipe(layer, weights), int(dimension))


def layer_bias_factors(layer):
    """Per-output correction for a bias added after a fused pool, or None."""
    entry = _LOWERINGS.get(layer.template_id)
    build_second = None if entry is None else entry.second_layer_builder
    if build_second is None:
        return None
    parameters = layer.parameters()
    if 'intermediate_shape' not in parameters:
        return None
    return _fused_bias_factors(
        build_second(parameters, {}),
        _channel_major(parameters['intermediate_shape']),
        layer.output_spec,
    )


# =============================================================
# Packing phase: a vectorized program becomes PP-ops on a ring.
#
# This is where slot layouts are reconciled globally and where the security
# parameters come from. The frontend describes one sample's arithmetic and
# names no HE concept; everything below -- ciphertext width, modulus chain
# depth, the rescales that keep a residual's two arms at one level -- is
# decided here, from the operations actually emitted.
# =============================================================


# CROSS runs composite-degree-2 CKKS: every logical modulus is a pair of
# machine-word primes, so one level costs two RNS limbs. These sizes match the
# registered profiles in he_params (60-bit scaling moduli under a 61-bit first
# modulus, 32-bit words) and are the shape its prime search is tuned for.
DEFAULT_SECURITY_BITS = 128
DEFAULT_SCALING_MOD_SIZE = 60
DEFAULT_FIRST_MOD_SIZE = 61
DEFAULT_REGISTER_WORD_SIZE = 32
DEFAULT_COMPOSITE_DEGREE = 2


class TestOnlyParameters(dict):
  """A parameters dict deliberately overriding a packed program's own ring.

  Running a packed program on parameters other than the ones its security
  plan was validated against is not a thing to do by accident, so Mapping
  refuses a plain dict. Tests that need a small insecure ring -- to keep a
  correctness check fast -- wrap it in this type to say so out loud.
  """


def test_only_parameters(parameters: dict) -> 'TestOnlyParameters':
  """Mark a parameters dict as a deliberate, test-only ring override."""
  return TestOnlyParameters(parameters)


# CKKS decoding of a real-valued program leaves a small imaginary residual.
# Judged relative to the magnitude of the result, so it scales with the data.
_IMAGINARY_RESIDUAL_TOLERANCE = 1e-6


class PackingError(Exception):
  """A vectorized program could not be packed onto a secure ring."""


@dataclass(frozen=True)
class PackingPolicy:
  """Knobs for the packing phase, all with documented defaults.

  There is no BSGS knob. The baby-step/giant-step split is chosen per
  operation by Mapping, from the diagonals each matvec actually has; one
  ratio fixed here would decide it for every matvec in the program at once.

  ``lazy_constants`` trades memory for eagerness. By default every matvec
  constant is a materialized SparseDiagonals: concrete, copied and digested
  from its own bytes. That is exact but quadratic in the ciphertext width --
  a 1024-diagonal matrix over 16384 slots is 134 MB -- so a large model can
  ask for the matrix-free source instead, which enumerates the same diagonals
  on demand and digests them through the recipe they came from.

  There is no bootstrap hook. Nothing here auto-inserts a bootstrap and no
  policy field pretends to: the planner and engine still disagree on the
  post-bootstrap noise-scale degree, so a program too deep for every
  supported degree is reported rather than silently segmented.
  """

  security_bits: int = DEFAULT_SECURITY_BITS
  scaling_mod_size: int = DEFAULT_SCALING_MOD_SIZE
  first_mod_size: int = DEFAULT_FIRST_MOD_SIZE
  register_word_size: int = DEFAULT_REGISTER_WORD_SIZE
  composite_degree: int = DEFAULT_COMPOSITE_DEGREE
  name: Optional[str] = None
  lazy_constants: bool = False


def _select_dnum(num_q: int) -> int:
  """A HYBRID digit count that is valid for ``num_q``, chosen deterministically.

  The usual convention is ceil(num_q/2), but at num_q=3 that gives dnum=2,
  where he_params raises `OpenFHE EstimateLogP mismatch: estimated 4, actual
  3` -- and that raise aborts generation rather than skipping to the next
  degree, so it would look like "no secure parameters exist". Small chains
  therefore use dnum=num_q, which is valid throughout.
  """
  if num_q < 2:
    raise PackingError(
        f'a modulus chain needs at least a first and one scaling modulus; '
        f'num_q={num_q} is not expressible.'
    )
  return num_q if num_q <= 3 else -(-num_q // 2)


# What each emitted PP-op costs in levels. Packing recomputes program depth
# from these over the operations it actually emitted, rather than trusting the
# frontend's declaration: a polynomial the frontend calls ceil(log2 d) deep
# must not turn into a deeper circuit here without the chain growing to match.
_LEVEL_COST = {
    'matvec': 1,
    'mul': 1,
    'square': 1,
    'add': 0,
    'sub': 0,
    'add_plain': 0,
    'mul_plain': 0,
    'rotate': 0,
}


def _operation_depth(kind: str, argument) -> int:
  if kind in ('rescale', 'level_reduce'):
    return int(argument)
  try:
    return _LEVEL_COST[kind]
  except KeyError:
    raise PackingError(
        f'no level cost is defined for PP-op {kind!r}.'
    ) from None


def _bias_slots(bias, output_spec, num_slots, factors=None) -> np.ndarray:
  """Broadcast a per-channel bias across the slots its channel occupies.

  ``factors`` corrects a bias that is added *after* a pool fused into the
  same matrix. Averaging a constant returns the constant only when the window
  is full; where a count_include_pad window overlaps padding, the convolution
  contributed the bias to fewer than ``denominator`` positions, so the bias
  arrives scaled by that fraction and must be pre-scaled to match.
  """
  values = np.zeros(num_slots, dtype=np.float64)
  for coordinate in np.ndindex(tuple(output_spec.shape)):
    value = float(bias[coordinate[0]])
    if factors is not None:
      value *= float(factors[coordinate])
    values[output_spec.coordinate_to_slot(coordinate)] = value
  return values


def _ring_config_payload(ring_config) -> dict:
  """A JSON-stable view of every field that defines the security plan."""
  to_dict = getattr(ring_config, 'to_dict', None)
  payload = to_dict() if callable(to_dict) else {
      name: getattr(ring_config, name)
      for name in sorted(dir(ring_config))
      if not name.startswith('_')
      and not callable(getattr(ring_config, name))
  }
  return json.loads(json.dumps(payload, sort_keys=True, default=str))


@dataclass(frozen=True)
class Packing:
  """A vectorized program materialized as PP-ops on one secure ring.

  ``operations`` is the ordered PP-op DAG :func:`pack` emits from a frontend
  :class:`nn.VectorizedProgram`: each entry is
  ``(value_id, kind, inputs, argument)``, and ``Mapping`` schedules exactly
  this. ``ring_config`` is the security parameter set derived from this
  program's own slot demand and emitted depth -- a result of packing, not an
  input to it.
  """

  operations: tuple[_PackedOperation, ...]
  ring_config: Any
  # Physical layout of every ciphertext: one full-width slot vector.
  input_shape: tuple[int, ...]
  input_packing: str
  # Logical shapes, carried as metadata for packing and unpacking a sample.
  logical_input_shape: tuple[int, ...]
  logical_output_shape: tuple[int, ...]
  # Where element j of the logical tensor lives in the slot vector. Slicing
  # 0..n instead would silently transpose anything that is not row-major, and
  # the packer now emits stride-multiplexed and repacked layouts.
  input_coordinate_map: tuple[int, ...]
  output_coordinate_map: tuple[int, ...]
  layer_shapes: tuple[tuple[str, tuple[tuple[int, ...], tuple[int, ...]]], ...]
  output: str
  num_slots: int
  depth: int
  fingerprint: str = field(init=False)

  def __post_init__(self):
    object.__setattr__(self, 'operations', tuple(self.operations))
    object.__setattr__(
        self, 'input_coordinate_map',
        tuple(int(slot) for slot in self.input_coordinate_map),
    )
    object.__setattr__(
        self, 'output_coordinate_map',
        tuple(int(slot) for slot in self.output_coordinate_map),
    )
    object.__setattr__(
        self,
        'fingerprint',
        _json_digest({
            'operations': [
                {
                    'id': value_id,
                    'kind': kind,
                    'inputs': list(inputs),
                    'argument': _fingerprint_argument(argument),
                }
                for value_id, kind, inputs, argument in self.operations
            ],
            'input_shape': list(self.input_shape),
            'input_packing': self.input_packing,
            'logical_input_shape': list(self.logical_input_shape),
            'logical_output_shape': list(self.logical_output_shape),
            'input_coordinate_map': list(self.input_coordinate_map),
            'output_coordinate_map': list(self.output_coordinate_map),
            'layer_shapes': [
                [name, [list(shape) for shape in shapes]]
                for name, shapes in self.layer_shapes
            ],
            'output': self.output,
            'num_slots': self.num_slots,
            'depth': self.depth,
            # The complete security plan, not just its shape. Two configs
            # that agree on degree and chain length can still differ in
            # target bits, cost model, moduli, dnum or scaling factor, and
            # they must not alias to one packed plan.
            'ring_config': _ring_config_payload(self.ring_config),
        }),
    )

  def kinds(self) -> tuple[str, ...]:
    return tuple(kind for _, kind, _, _ in self.operations)

  def pack(self, value) -> np.ndarray:
    """Scatter one logical input across this ring's slot vector.

    Element j goes to the slot the input layout assigns it, which is only
    j itself when that layout is row-major. Serving code therefore carries no
    model-specific packing callable.
    """
    array = np.asarray(value, dtype=np.float64)
    expected = len(self.input_coordinate_map)
    if array.size != expected:
      raise ValueError(
          f'input has {array.size} elements, expected {expected} for shape '
          f'{self.logical_input_shape}.'
      )
    if not np.all(np.isfinite(array)):
      raise ValueError('packed input slots must be finite.')
    slots = np.zeros(self.num_slots, dtype=np.float64)
    slots[list(self.input_coordinate_map)] = array.reshape(-1)
    return slots

  def unpack(self, value) -> np.ndarray:
    """Gather one physical slot vector back into the logical output.

    CKKS decoding of a real program leaves a small imaginary residual, so a
    complex array is accepted and its real part taken, provided the residual
    is negligible against the magnitude of the result. A large one means the
    program did not compute a real value and is reported rather than dropped.
    """
    array = np.asarray(value)
    if array.dtype.kind == 'c':
      imaginary = float(np.max(np.abs(np.imag(array)))) if array.size else 0.0
      scale = max(
          float(np.max(np.abs(np.real(array)))) if array.size else 0.0, 1.0
      )
      if not math.isfinite(imaginary):
        raise ValueError('unpacked output values must be finite.')
      if imaginary > _IMAGINARY_RESIDUAL_TOLERANCE * scale:
        raise ValueError(
            f'output carries an imaginary residual of {imaginary:.3e} against '
            f'a magnitude of {scale:.3e}, above the '
            f'{_IMAGINARY_RESIDUAL_TOLERANCE:g} tolerance for a real-valued '
            'program.'
        )
      array = np.real(array)
    elif array.dtype.kind not in 'biuf':
      raise TypeError('unpacked output values must be real numeric values.')
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
      raise ValueError('unpacked output values must be finite.')
    return array[list(self.output_coordinate_map)].reshape(
        self.logical_output_shape
    )

  def operation(self, value_id: str) -> _PackedOperation:
    for operation in self.operations:
      if operation[0] == value_id:
        return operation
    raise KeyError(f'no packed operation {value_id!r}.')


def _plan_operations(program, policy):
  """Emit the PP-op DAG and its per-value depth, without materializing values.

  Structure and depth do not depend on the ciphertext width, so this runs
  before a ring exists -- which is what keeps the ordering acyclic: the ring
  is chosen from the depth measured here, never the other way round.
  """
  plan: list[dict] = []
  depth: dict[str, int] = {'input': 0}
  produced: dict[str, str] = {}
  # The logical layout each value actually carries. Physical uniformity does
  # not make layouts interchangeable: a consumer's matrix reads slots through
  # its own input_spec, so feeding it a value laid out differently silently
  # computes a different function.
  layouts: dict[str, Any] = {'input': program.input_spec}
  counter = {'n': 0}

  def emit(kind, inputs, *, layer=None, argument=None, base=None, extra=None):
    counter['n'] += 1
    value_id = f'{base or kind}_{counter["n"]}'
    cost = _operation_depth(kind, argument)
    depth[value_id] = max(
        (depth[name] for name in inputs), default=0
    ) + cost
    plan.append({
        'id': value_id, 'kind': kind, 'inputs': tuple(inputs),
        'layer': layer, 'argument': argument, 'extra': extra or {},
    })
    return value_id

  def align(value_id, target_depth):
    """Consume levels on the shallow arm of a join, matching scale exactly.

    A join needs both operands at one level *and* one scale. level_reduce
    matches the level but keeps the old scale, whereas every MatVec and
    Square on the deep arm advanced it by s -> s**2 / q_pair. So the shallow
    arm consumes each missing level the same way: multiply by an all-ones
    plaintext, which Mapping encodes at the ciphertext's own scale and so
    squares it, then Rescale(1) to divide by the same dropped modulus pair.
    The represented value is multiplied by one throughout, and nsd goes
    1 -> 2 -> 1, so both arms arrive identical in level, scale and nsd.
    """
    gap = target_depth - depth[value_id]
    if gap <= 0:
      return value_id
    current = value_id
    for step in range(gap):
      current = emit(
          'mul_plain', (current,), base=f'align_ones',
          extra={'ones': True},
      )
      layouts[current] = layouts[value_id]
      current = emit(
          'rescale', (current,), argument=1, base=f'align_rescale',
      )
      layouts[current] = layouts[value_id]
    return current

  def reconcile(value_id, expected):
    """Make ``value_id`` carry ``expected``'s layout, repacking if needed."""
    current = layouts[value_id]
    if tuple(current._coordinate_map) == tuple(expected._coordinate_map):
      # The same element sits in the same slot under both descriptions, so
      # this is a relabeling; a flatten under row-major slots is the usual
      # case. Nothing has to move.
      layouts[value_id] = expected
      return value_id
    if current.physical_size != expected.physical_size:
      raise PackingError(
          f'cannot repack {current.shape}/{current.packing!r} into '
          f'{expected.shape}/{expected.packing!r}: they occupy '
          f'{current.physical_size} and {expected.physical_size} slots.'
      )
    repacked = emit(
        'matvec', (value_id,), base='repack',
        extra={'repack': (current, expected)},
    )
    layouts[repacked] = expected
    return repacked


  # Only what the declared output actually depends on. A layer beside or
  # after it contributes nothing, and letting one inflate the depth or the
  # slot demand would buy a larger, slower ring for a value nobody reads.
  by_name = {layer.name: layer for layer in program.layers}
  reachable: set[str] = set()
  frontier = [program.output]
  while frontier:
    name = frontier.pop()
    if name in reachable or name not in by_name:
      continue
    reachable.add(name)
    frontier.extend(by_name[name].inputs)

  for layer in program.layers:
    if layer.name not in reachable:
      continue
    sources = [
        produced.get(name, name) if name != 'input' else 'input'
        for name in (layer.inputs or ('input',))
    ]
    if layer.kind == 'linear_transform':
      value = emit(
          'matvec', (reconcile(sources[0], layer.input_spec),),
          layer=layer, base=f'{layer.name}',
      )
      if any(role == 'bias' for role, _ in layer.weights):
        value = emit(
            'add_plain', (value,), layer=layer, base=f'{layer.name}_bias',
            extra={'bias_factors': layer_bias_factors(layer)},
        )
    elif layer.kind == 'polynomial_activation':
      polynomial = layer.polynomial()
      if not polynomial.is_squaring_chain:
        # register_activation refuses these, so reaching here means a layer
        # was built around a polynomial that has since been replaced.
        raise PackingError(
            f'layer {layer.name!r} applies {polynomial.name!r}, which is not '
            'a squaring chain x**(2**k). Only repeated squaring keeps both '
            'operands of every multiplication at one level; any other '
            'exponent needs a mod-switch that the available primitives cannot '
            'express without also dividing the scale.'
        )
      value = reconcile(sources[0], layer.input_spec)
      # Measured from the reconciled input: a repack inserted to satisfy this
      # layer's layout is layout cost, not part of the polynomial's circuit.
      base_depth = depth[value]
      for step in range(polynomial.depth_cost):
        value = emit(
            'square', (value,), base=f'{layer.name}_pow{1 << (step + 1)}',
        )
      declared = layer.depth_cost
      emitted = depth[value] - base_depth
      if emitted != declared:
        raise PackingError(
            f'layer {layer.name!r} declared depth {declared} but packing '
            f'emitted a circuit of depth {emitted}; the modulus chain would '
            'be sized for the wrong program.'
        )
    elif layer.kind == 'add':
      sources = [reconcile(name, layer.input_spec) for name in sources]
      # Slotwise addition needs both operands on one level *and* one scale.
      # See align(): the shallow arm consumes each missing level the way the
      # deep arm did, so the two agree on both.
      target = max(depth[name] for name in sources)
      value = emit(
          'add', tuple(align(name, target) for name in sources),
          base=layer.name,
      )
    elif layer.kind == 'layout':
      # Usually a relabeling under row-major slots, in which case reconcile
      # emits nothing; a genuine permutation becomes a repack matrix.
      produced[layer.name] = reconcile(sources[0], layer.output_spec)
      continue
    else:
      raise PackingError(f'no PP-op lowering for layer kind {layer.kind!r}.')
    layouts[value] = layer.output_spec
    produced[layer.name] = value

  output = produced[program.output]
  return plan, depth, output, layouts, reachable


_PROGRAM_ATTRIBUTES = ('layers', 'weights', 'input_spec', 'output', 'digest')


def _validate_program(program) -> None:
  """Reject anything that is not a vectorized program, by name not by crash."""
  missing = [
      name for name in _PROGRAM_ATTRIBUTES if not hasattr(program, name)
  ]
  if missing or not callable(getattr(program, 'layer', None)):
    if not callable(getattr(program, 'layer', None)):
      missing = missing + ['layer()']
    raise TypeError(
        f'pack() takes the VectorizedProgram that nn.vectorize returns; got '
        f'{type(program).__name__}, which is missing {missing}.'
    )
  for index, layer in enumerate(program.layers):
    absent = [
        name for name in
        ('name', 'kind', 'inputs', 'input_spec', 'output_spec', 'depth_cost')
        if not hasattr(layer, name)
    ]
    if absent:
      raise TypeError(
          f'pack() expects vectorized layers; layer {index} is a '
          f'{type(layer).__name__}, which is missing {absent}.'
      )


def pack(program, policy: Optional[PackingPolicy] = None) -> Packing:
  """Materialize a vectorized program as PP-ops on a derived secure ring.

  The order is deliberately one-way. Layouts are reconciled and the PP-op DAG
  is emitted first, because neither depends on the ciphertext width; the ring
  is then chosen from the slot demand and the depth those operations actually
  consume; only then are constants materialized, at exactly the ring's slot
  count. Nothing loops back, so no bootstrap decision can depend on a ring
  that was itself sized around bootstraps.
  """
  _validate_program(program)
  policy = policy or PackingPolicy()
  # Imported here, not at module scope, so `import jaxite_word.packing` stays
  # free of the crypto stack for tooling that never builds a ciphertext.
  import he_params

  plan, depth, output, layouts, reachable = _plan_operations(program, policy)
  program_depth = max(depth.values(), default=0)

  # Every live value occupies one ciphertext, so the ring must hold the widest
  # layout any of them takes -- inputs and outputs alike.
  live = [layer for layer in program.layers if layer.name in reachable]
  slot_demand = max(
      [int(program.input_spec.physical_size)]
      + [int(layer.output_spec.physical_size) for layer in live]
      + [int(layer.input_spec.physical_size) for layer in live]
  )

  num_q = program_depth + 1
  spec = he_params.ModelSpec(
      name=policy.name or f'packed_{program.digest[:12]}',
      num_q=num_q,
      dnum=_select_dnum(num_q),
      scaling_mod_size=policy.scaling_mod_size,
      first_mod_size=policy.first_mod_size,
      register_word_size=policy.register_word_size,
      composite_degree=policy.composite_degree,
      min_slots=slot_demand,
  )
  target = he_params.SecurityTarget(
      policy.security_bits, he_params.CostModel.CLASSICAL
  )
  try:
    # he_params searches the supported degrees and returns the smallest that
    # satisfies every invariant, including num_slots == degree // 2. Deriving
    # a degree here instead -- next_pow2(slot_demand), say -- would be wrong by
    # a factor of two and would skip that validation entirely.
    ring_config = he_params.generate_ring_config(spec, target)
  except he_params.ParameterGenerationError as error:
    raise PackingError(
        f'no secure ring satisfies this program at {policy.security_bits}-bit '
        f'security: it needs {slot_demand} slots and multiplicative depth '
        f'{program_depth} (num_q={num_q}, dnum={spec.dnum}). '
        'Reduce the depth or the width; packing does not insert a bootstrap '
        'on its own, and no policy option turns that on today.'
    ) from error

  num_slots = int(ring_config.num_slots)
  if num_slots != ring_config.degree // 2:
    raise PackingError(
        f'ring reports {num_slots} slots for degree {ring_config.degree}; '
        'CKKS packs degree/2 slots.'
    )
  if num_slots < slot_demand:
    raise PackingError(
        f'ring holds {num_slots} slots but the program needs {slot_demand}.'
    )

  # Every ciphertext is one full-width vector, so a value's *physical* layout
  # is (num_slots,) from end to end. Mapping compares layouts for exact
  # equality, so advertising a matvec's logical output shape here would make
  # its own bias plaintext look like a different layout. Logical shapes stay
  # as metadata, which is all pack/unpack need them for.
  physical = (num_slots,)
  physical_packing = 'slots'
  layer_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}

  operations: list[_PackedOperation] = []
  for step in plan:
    layer = step['layer']
    argument = step['argument']
    if step['kind'] == 'matvec' and 'repack' in step['extra']:
      current, expected = step['extra']['repack']
      # A permutation: element j sits at current slot p and must sit at
      # expected slot c, so the matrix carries a single 1 at (c, p).
      diagonals: dict[int, np.ndarray] = {}
      for source_slot, target_slot in zip(
          current._coordinate_map, expected._coordinate_map, strict=True
      ):
        index = (int(source_slot) - int(target_slot)) % num_slots
        vector = diagonals.get(index)
        if vector is None:
          vector = np.zeros(num_slots, dtype=np.float64)
          diagonals[index] = vector
        vector[int(target_slot)] = 1.0
      operations.append((
          step['id'], 'matvec', step['inputs'],
          (
              SparseDiagonals(
                  dimension=num_slots,
                  diagonals=tuple(sorted(diagonals.items())),
                  input_shape=physical,
                  output_shape=physical,
                  packing=physical_packing,
                  output_packing=physical_packing,
              ),
              None, None, None, None,
          ),
      ))
      layer_shapes[step['id']] = (
          tuple(current.shape), tuple(expected.shape)
      )
      continue
    if step['kind'] == 'matvec':
      source = layer_matrix_source(layer, program.weights, num_slots)
      if policy.lazy_constants:
        matrix = _LazySparseDiagonals(
            dimension=num_slots,
            source=source,
            input_shape=physical,
            output_shape=physical,
            packing=physical_packing,
            output_packing=physical_packing,
        )
      else:
        matrix = SparseDiagonals(
            dimension=num_slots,
            diagonals=tuple(sorted(source.as_dict().items())),
            input_shape=physical,
            output_shape=physical,
            packing=physical_packing,
            output_packing=physical_packing,
        )
      layer_shapes[step['id']] = (
          tuple(layer.input_spec.shape), tuple(layer.output_spec.shape)
      )
      # n1/n2/ratio/pt_scale stay open: the baby-step split is a
      # scheduling choice Mapping makes per operation, from the
      # diagonals that matvec actually has. Freezing one ratio for a
      # whole program here would decide it for every matvec at once.
      argument = (matrix, None, None, None, None)
    elif step['kind'] == 'mul_plain' and step['extra'].get('ones'):
      # Encoded at the ciphertext's own scale (scale=None), so the multiply
      # squares the scale exactly as a MatVec or Square would.
      argument = PlainSlots(
          values=np.ones(num_slots, dtype=np.float64),
          shape=physical,
          packing=physical_packing,
      )
    elif step['kind'] == 'add_plain':
      bias = layer.resolve_weights(program.weights)['bias']
      argument = PlainSlots(
          values=_bias_slots(
              bias, layer.output_spec, num_slots,
              step['extra'].get('bias_factors'),
          ),
          shape=physical,
          packing=physical_packing,
      )
    operations.append(
        (step['id'], step['kind'], step['inputs'], argument)
    )

  return Packing(
      operations=tuple(operations),
      ring_config=ring_config,
      input_shape=physical,
      input_packing=physical_packing,
      logical_input_shape=tuple(program.input_spec.shape),
      logical_output_shape=tuple(
          program.layer(program.output).output_spec.shape
      ),
      input_coordinate_map=tuple(program.input_spec._coordinate_map),
      output_coordinate_map=tuple(
          layouts[output]._coordinate_map
      ),
      layer_shapes=tuple(sorted(layer_shapes.items())),
      output=output,
      num_slots=num_slots,
      depth=program_depth,
  )


# The packing phase's public surface. The logical-layer descriptors and the
# recipes built from them are deliberately absent: they are how a template is
# realized, not a way to describe a model. There is one way to do that, and it
# is a torch.nn.Module handed to nn.vectorize.
__all__ = [
    'ChannelMajor',
    'DEFAULT_COMPOSITE_DEGREE',
    'DEFAULT_FIRST_MOD_SIZE',
    'DEFAULT_REGISTER_WORD_SIZE',
    'DEFAULT_SCALING_MOD_SIZE',
    'DEFAULT_SECURITY_BITS',
    'Packing',
    'PackingError',
    'PackingPolicy',
    'PlainSlots',
    'SparseDiagonals',
    'StrideMultiplexed',
    'TensorLayout',
    'TensorSpec',
    'TestOnlyParameters',
    'TemplateLowering',
    'register_template_lowering',
    'unregister_template_lowering',
    'template_lowering',
    'pack',
    'test_only_parameters',
]

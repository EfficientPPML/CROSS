"""Parse a torch model into a semantic VectorizedProgram.

:func:`vectorize` traces a real ``torch.nn.Module`` and describes it: each
layer names a vectorization template and the parameters that template needs,
and each non-linear activation is substituted by a polynomial approximation.
The result is a :class:`VectorizedProgram` -- application semantics only, with
no slot constant, no diagonal and no HE operation anywhere in it.
:func:`packing.pack` is what turns that description into an ordered PP-op DAG.

The description is deliberately LOCAL: each layer is vectorized on its own
merits, and ``packing`` reconciles those local preferences into one coherent
global layout and one secure ring. Nothing here touches HE levels, keys or
ciphertexts, and neither torch nor JAX is imported at module scope, so this
module stays importable without either.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import operator
import sys
import warnings
from typing import Any, Callable, ClassVar, Iterator, Optional, Union

import numpy as np


if __name__ == 'jaxite_word.nn':
  sys.modules.setdefault('nn', sys.modules[__name__])
elif __name__ == 'nn':
  sys.modules.setdefault('jaxite_word.nn', sys.modules[__name__])


if __package__:
  from . import packing
  from .packing import (
      ChannelMajor, StrideMultiplexed, TensorLayout, TensorSpec, _spatial_pair,
  )
else:
  import packing
  from packing import (
      ChannelMajor, StrideMultiplexed, TensorLayout, TensorSpec, _spatial_pair,
  )


# =============================================================
# The frontend: a torch model becomes a vectorized program.
#
# Everything here is semantic. A layer is described, never lowered: no slot
# constant, no diagonal, no HE operation is built in this module. Packing owns
# all of that, which is what lets the dependency run one way -- packing never
# imports this module, and importing this one costs neither JAX nor torch.
# =============================================================

# =============================================================


# Application/vector semantics only. Rescale and bootstrap are HE realizations:
# packing materializes them and mapping schedules them, so neither is nameable
# here. Adding a kind means teaching packing what PP-ops it becomes.
LINEAR_TRANSFORM = 'linear_transform'
POLYNOMIAL_ACTIVATION = 'polynomial_activation'
ADD = 'add'
LAYOUT = 'layout'

# Operator families the frontend can read parameters from. A template must
# name one: it decides *how* a known family is vectorized, and the parser
# still has to interpret the torch layer to fill it in. Admitting a new family
# means teaching the parser to read it, not only registering a template.
SUPPORTED_FAMILIES = frozenset({
    'conv2d', 'dense', 'avg_pool2d', 'adaptive_avg_pool2d', 'flatten',
})

VECTORIZED_KINDS = frozenset({
    LINEAR_TRANSFORM, POLYNOMIAL_ACTIVATION, ADD, LAYOUT,
})

# Stands for the program's own input where a layer names its sources. A layer
# with no inputs at all also reads the program input; naming it explicitly
# matters only for a multi-input node, such as the shallow arm of a residual,
# where dropping the edge would hide that the node is a join.
#
# Spelled 'input' because that is the value id Packing seeds and Mapping
# reserves. One name across all three phases means a packed program's operands
# resolve without translation; a second spelling here would produce operations
# referring to a value the scheduler never initializes.
PROGRAM_INPUT = 'input'

# Weights are digested after being cast to this dtype, so an int32 and a
# float64 tensor holding the same values name the same entry.
_CANONICAL_WEIGHT_DTYPE = np.float64



def _digest_payload(value) -> str:
  """Digest a JSON-native payload. Frontend identities are built from this."""
  return hashlib.sha256(
      json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
  ).hexdigest()


def _channel_major(shape: tuple[int, ...]) -> TensorSpec:
  """The default layout for a traced activation: contiguous row-major slots."""
  return TensorSpec(tuple(shape), ChannelMajor())


def normalize_weight(value) -> np.ndarray:
  """Return ``value`` as a canonical, immutable, C-contiguous float64 array.

  The result never shares storage with ``value``. ``np.ascontiguousarray``
  would return the caller's own array whenever it already has the right dtype
  and order, which breaks this twice over: clearing the writable flag would
  mutate the caller's array, and a write through a base array the caller still
  holds would silently change bytes a digest -- and the security plan derived
  from it -- has already been computed over. Backing the result with an
  immutable ``bytes`` object removes the possibility rather than forbidding it.
  """
  if hasattr(value, 'detach'):  # torch.Tensor, without importing torch.
    value = value.detach().cpu().numpy()
  array = np.asarray(value)
  if array.dtype.kind not in 'biuf':
    raise TypeError(
        f'weights must be real numeric, got dtype {array.dtype}.'
    )
  array = np.asarray(array, dtype=_CANONICAL_WEIGHT_DTYPE)
  if not np.all(np.isfinite(array)):
    raise ValueError('weights must contain only finite values.')
  detached = np.frombuffer(
      array.tobytes(order='C'), dtype=_CANONICAL_WEIGHT_DTYPE
  ).reshape(array.shape)
  detached.setflags(write=False)
  return detached


def weight_digest(array: np.ndarray) -> str:
  """Digest a normalized weight over its dtype, shape and bytes."""
  digest = hashlib.sha256()
  digest.update(str(array.dtype).encode())
  digest.update(repr(tuple(array.shape)).encode())
  digest.update(array.tobytes(order='C'))
  return digest.hexdigest()


class WeightTable:
  """Immutable content-addressed weight store for one program.

  Identity is the set of digests, never the arrays: comparing ndarrays would
  raise, and hashing them is impossible. Equal digests mean equal bytes.

  There is no ``add``. A program's digest covers this table, so a table that
  could still grow afterwards would let the digest go stale while continuing
  to look authoritative. Accumulate with ``WeightTableBuilder`` and freeze.
  """

  def __init__(self, entries=None):
    self._entries: dict[str, np.ndarray] = {}
    for digest, array in dict(entries or {}).items():
      normalized = normalize_weight(array)
      actual = weight_digest(normalized)
      if actual != digest:
        # A key that does not name its own content is not an address; it
        # would let two different weights answer to one digest.
        raise ValueError(
            f'weight key {str(digest)[:16]}... does not match the digest of '
            f'the array it names ({actual[:16]}...).'
        )
      self._entries[actual] = normalized

  def __getitem__(self, digest: str) -> np.ndarray:
    try:
      return self._entries[digest]
    except KeyError:
      raise KeyError(
          f'weight {digest[:16]}... is not in this program.'
      ) from None

  def __contains__(self, digest) -> bool:
    return digest in self._entries

  def __len__(self) -> int:
    return len(self._entries)

  def digests(self) -> tuple[str, ...]:
    return tuple(sorted(self._entries))

  def as_dict(self) -> dict[str, np.ndarray]:
    return dict(self._entries)

  def __eq__(self, other):
    if not isinstance(other, WeightTable):
      return NotImplemented
    return self.digests() == other.digests()

  def __hash__(self):
    return hash(self.digests())

  def __reduce__(self):
    # Rebuild through __init__ rather than restoring __dict__. numpy pickles
    # an array's data but not its writeable flag, so a default restore would
    # hand back mutable arrays with no immutable buffer behind them: the
    # content could then drift from the digest naming it, silently, in a
    # process that had already derived a security plan from those bytes.
    return (type(self), (self._entries,))

  def __repr__(self):
    return f'WeightTable({len(self._entries)} weights)'


class WeightTableBuilder:
  """Accumulate weights, then ``freeze()`` them into a WeightTable."""

  def __init__(self):
    self._entries: dict[str, np.ndarray] = {}

  def add(self, value) -> str:
    """Store ``value`` if new and return the digest naming it."""
    array = normalize_weight(value)
    digest = weight_digest(array)
    self._entries.setdefault(digest, array)
    return digest

  def __contains__(self, digest) -> bool:
    return digest in self._entries

  def __len__(self) -> int:
    return len(self._entries)

  def freeze(self) -> WeightTable:
    return WeightTable(self._entries)

  def __reduce__(self):
    # Same reason as WeightTable.__reduce__: re-normalize on restore so the
    # arrays keep their immutable backing.
    return (_restore_weight_table_builder, (self._entries,))

  def __repr__(self):
    return f'WeightTableBuilder({len(self._entries)} weights)'


def _restore_weight_table_builder(entries) -> 'WeightTableBuilder':
  builder = WeightTableBuilder()
  for array in entries.values():
    builder.add(array)
  return builder


def _normalize_param(value):
  """Coerce one template parameter to a hashable, JSON-stable value.

  ndarrays are rejected rather than coerced: a weight belongs in the
  WeightTable, where it is deduplicated and digested by content.
  """
  if value is None or isinstance(value, (bool, str)):
    return value
  if isinstance(value, (int, np.integer)):
    return int(value)
  if isinstance(value, (float, np.floating)):
    return float(value)
  if isinstance(value, (tuple, list)):
    return tuple(_normalize_param(item) for item in value)
  raise TypeError(
      f'template parameters must be scalars or tuples of them, got '
      f'{type(value).__name__}; array-valued data belongs in the weight table.'
  )


def _normalize_params(params) -> tuple[tuple[str, Any], ...]:
  items = params.items() if isinstance(params, MappingABC) else params
  return tuple(
      (str(name), _normalize_param(value))
      for name, value in sorted(items, key=lambda item: str(item[0]))
  )


@dataclass(frozen=True)
class ActivationPolynomial:
  """A polynomial standing in for a non-linear activation.

  ``coefficients`` is ascending: ``(0.0, 0.0, 1.0)`` is x**2, the default
  substitution for ReLU. Depth is the Paterson-Stockmeyer bound ceil(log2 d),
  which is what the level accounting charges.
  """

  name: str
  coefficients: tuple[float, ...]

  def __post_init__(self):
    coefficients = tuple(float(value) for value in self.coefficients)
    while len(coefficients) > 1 and coefficients[-1] == 0.0:
      coefficients = coefficients[:-1]
    if len(coefficients) < 2:
      raise ValueError(
          f'activation {self.name!r} must have degree >= 1.'
      )
    if not all(math.isfinite(value) for value in coefficients):
      raise ValueError(
          f'activation {self.name!r} has non-finite coefficients.'
      )
    object.__setattr__(self, 'coefficients', coefficients)

  @property
  def degree(self) -> int:
    return len(self.coefficients) - 1

  @property
  def is_unit_monomial(self) -> bool:
    """True for x**d: one term, coefficient one, no constant."""
    return (
        self.coefficients[-1] == 1.0
        and not any(self.coefficients[:-1])
    )

  @property
  def is_squaring_chain(self) -> bool:
    """True for x**(2**k), the form packing can emit today.

    A power of two is reached by repeated squaring, so both operands of every
    multiplication are already at the same level. Any other exponent needs one
    operand mod-switched down to meet the other, and the only level-changing
    primitive available is Rescale, which also divides the scale -- so the
    product would carry a scale neither the planner nor the engine expects.
    """
    degree = self.degree
    return self.is_unit_monomial and degree >= 2 and not (
        degree & (degree - 1)
    )

  @property
  def depth_cost(self) -> int:
    """Levels evaluating this polynomial consumes.

    x**(2**k) is a squaring chain, so exactly k. Anything else additionally
    needs its terms scaled and aligned, which costs one further level;
    claiming ceil(log2 d) for those would under-provision the modulus chain.
    """
    chain = max(1, math.ceil(math.log2(self.degree)))
    return chain if self.is_unit_monomial else chain + 1


_ACTIVATIONS: dict[str, ActivationPolynomial] = {}
# Which registered polynomial each traced activation gets by default. Keyed by
# the frontend's activation name so registration needs no torch import.
_ACTIVATION_DEFAULTS: dict[str, str] = {}


def register_activation(
    activation: ActivationPolynomial,
    *,
    substitutes: tuple[str, ...] = (),
    replace: bool = False,
) -> ActivationPolynomial:
  """Register a polynomial, optionally as the default for named activations."""
  if not activation.is_squaring_chain:
    raise ValueError(
        f'activation {activation.name!r} is {activation.degree}-degree with '
        f'coefficients {activation.coefficients}; only a bare x**(2**k) can '
        'be evaluated today. Reaching any other exponent needs one operand '
        'mod-switched down to meet the other, and the only level-changing '
        'primitive available is Rescale, which also divides the scale. '
        'Registering this would let a model vectorize and then fail to pack.'
    )
  if not replace and activation.name in _ACTIVATIONS:
    raise ValueError(
        f'activation {activation.name!r} is already registered; pass '
        'replace=True to override it.'
    )
  _ACTIVATIONS[activation.name] = activation
  for source in substitutes:
    _ACTIVATION_DEFAULTS[source] = activation.name
  return activation


def activation(name: str) -> ActivationPolynomial:
  try:
    return _ACTIVATIONS[name]
  except KeyError:
    raise KeyError(
        f'no activation {name!r}; registered: '
        f'{sorted(_ACTIVATIONS)}.'
    ) from None


def default_activation_for(source: str) -> ActivationPolynomial:
  """Return the polynomial substituted for a traced activation by default."""
  try:
    return _ACTIVATIONS[_ACTIVATION_DEFAULTS[source]]
  except KeyError:
    raise KeyError(
        f'no default polynomial for {source!r}; register one with '
        f'register_activation(..., substitutes=({source!r},)). '
        f'Substitutions available: {sorted(_ACTIVATION_DEFAULTS)}.'
    ) from None


# x**2 is the repo's only activation today: every demo is a "Quad" network.
# ReLU and SiLU substitute to it by default; a better approximation is a
# register_activation call, not an edit here.
register_activation(
    ActivationPolynomial(name='square', coefficients=(0.0, 0.0, 1.0)),
    substitutes=('relu', 'silu', 'square'),
)


@dataclass(frozen=True)
class LayoutTemplate:
  """One way to vectorize one layer, plus the condition under which it applies.

  Purely declarative: a template decides *which* lowering runs and supplies
  the parameters that lowering needs, and contributes no arithmetic of its
  own. How ``template_id`` is realized physically is registered separately,
  with :func:`packing.register_template_lowering` -- which is why adding a template
  means two registrations, one per phase, and why no recipe class appears in
  this module. A new template without a matching lowering vectorizes and then
  fails to pack, naming the id it could not lower.
  """

  template_id: str
  kind: str
  # Bumped whenever the vectorization this template produces changes. A layer
  # captures it, so replacing a template cannot silently redefine a program
  # that was already built and digested against the previous one.
  version: str
  # The operator family whose parameters the frontend knows how to read. A
  # template is an alternative *vectorization* of a family, not a way to admit
  # a new one: the parser has to interpret the layer to fill the template in.
  family: str
  depth_cost: int = 0
  # Operators this template can vectorize, as ``(node_op, dotted_target)``.
  # Targets are spelled, not imported: keeping them strings is what lets this
  # module stay free of torch at import time. The frontend parser resolves
  # each dotted name to the real object once and matches by identity, which
  # it must -- ``F.avg_pool2d`` is ``torch._C._nn.avg_pool2d`` and carries no
  # usable __name__, and ``x * x`` traces to ``operator.mul``.
  targets: tuple[tuple[str, str], ...] = ()
  # Names of the weight roles this template consumes, in build order.
  weight_roles: tuple[str, ...] = ()
  # Selection among templates sharing a target; highest applicable wins.
  priority: int = 0
  # ``condition(parameters, input_spec) -> bool``; None means "always".
  condition: Optional[Callable[..., bool]] = None

  def applies(self, parameters=None, input_spec=None) -> bool:
    if self.condition is None:
      return True
    return bool(self.condition(dict(parameters or {}), input_spec))


_LAYOUT_TEMPLATES: dict[str, LayoutTemplate] = {}
# target -> template ids, kept ordered by descending priority so selection is
# a scan rather than a sort, and ties resolve deterministically by id.
_TEMPLATES_BY_TARGET: dict[tuple[str, str], list[str]] = {}


def _reindex_target(target: tuple[str, str]) -> None:
  candidates = [
      template_id
      for template_id, template in _LAYOUT_TEMPLATES.items()
      if target in template.targets
  ]
  candidates.sort(
      key=lambda name: (-_LAYOUT_TEMPLATES[name].priority, name)
  )
  if candidates:
    _TEMPLATES_BY_TARGET[target] = candidates
  else:
    _TEMPLATES_BY_TARGET.pop(target, None)


def register_layout_template(
    template: LayoutTemplate, *, replace: bool = False
) -> LayoutTemplate:
  """Register a vectorization template.

  Several templates may claim one target. ``select_layout_template`` takes the
  highest-priority one whose condition holds, so a user adding a better
  vectorization for a specific shape registers it above the shipped default
  and constrains it with a condition, instead of editing this module.

  A new ``template_id`` also needs a lowering, registered on the packing side
  with :func:`packing.register_template_lowering` -- by ``alias`` when it reuses an
  existing realization, or with its own recipe when it does not. Without one
  the template vectorizes and then fails to pack.
  """
  if template.kind not in VECTORIZED_KINDS:
    raise ValueError(
        f'template {template.template_id!r} has kind {template.kind!r}; '
        f'expected one of {sorted(VECTORIZED_KINDS)}.'
    )
  if not template.version:
    raise ValueError(
        f'template {template.template_id!r} must declare a version.'
    )
  if template.family not in SUPPORTED_FAMILIES:
    raise ValueError(
        f'template {template.template_id!r} declares family '
        f'{template.family!r}; the frontend can read parameters for '
        f'{sorted(SUPPORTED_FAMILIES)} only. A template chooses how a known '
        'family is vectorized; admitting a new family also needs a parameter '
        'extractor in the parser.'
    )
  previous = _LAYOUT_TEMPLATES.get(template.template_id)
  if previous is not None:
    if not replace:
      raise ValueError(
          f'template {template.template_id!r} is already registered; pass '
          'replace=True to override it.'
      )
    if previous.version == template.version:
      # Reusing an id and a version for different behaviour would let an
      # already-digested program silently mean something else.
      raise ValueError(
          f'template {template.template_id!r} is already registered at '
          f'version {previous.version!r}; a replacement must declare a new '
          'version so programs built against the old one can detect it.'
      )
  _LAYOUT_TEMPLATES[template.template_id] = template
  touched = set(template.targets)
  if previous is not None:
    touched |= set(previous.targets)
  for target in touched:
    _reindex_target(target)
  return template


def unregister_layout_template(template_id: str) -> None:
  """Drop a registered template; mainly so tests can undo an override."""
  template = _LAYOUT_TEMPLATES.pop(template_id, None)
  if template is None:
    return
  for target in template.targets:
    _reindex_target(target)


def layout_template(template_id: str) -> LayoutTemplate:
  try:
    return _LAYOUT_TEMPLATES[template_id]
  except KeyError:
    raise KeyError(
        f'no layout template {template_id!r}; registered: '
        f'{sorted(_LAYOUT_TEMPLATES)}.'
    ) from None


def layout_template_candidates(
    target: tuple[str, str]
) -> tuple[LayoutTemplate, ...]:
  """Every template claiming ``target``, highest priority first."""
  return tuple(
      _LAYOUT_TEMPLATES[name]
      for name in _TEMPLATES_BY_TARGET.get(tuple(target), ())
  )


def select_layout_template(
    target: tuple[str, str],
    parameters=None,
    input_spec=None,
) -> LayoutTemplate:
  """Return the highest-priority template that applies to ``target``."""
  candidates = layout_template_candidates(target)
  if not candidates:
    raise KeyError(
        f'no layout template vectorizes {tuple(target)!r}; registered '
        f'targets: {sorted(_TEMPLATES_BY_TARGET)}.'
    )
  for template in candidates:
    if template.applies(parameters, input_spec):
      return template
  raise KeyError(
      f'{len(candidates)} template(s) claim {tuple(target)!r} but none apply '
      f'to these parameters: '
      f'{[template.template_id for template in candidates]}.'
  )


register_layout_template(LayoutTemplate(
    template_id='conv2d',
    version='1',
    family='conv2d',
    kind=LINEAR_TRANSFORM,
    depth_cost=1,
    targets=(('call_module', 'torch.nn.Conv2d'),),
    weight_roles=('weight', 'bias'),
))
register_layout_template(LayoutTemplate(
    template_id='avg_pool2d',
    version='1',
    family='avg_pool2d',
    kind=LINEAR_TRANSFORM,
    depth_cost=1,
    targets=(
        ('call_module', 'torch.nn.AvgPool2d'),
        ('call_function', 'torch.nn.functional.avg_pool2d'),
    ),
))
register_layout_template(LayoutTemplate(
    template_id='adaptive_avg_pool2d',
    version='1',
    family='adaptive_avg_pool2d',
    kind=LINEAR_TRANSFORM,
    depth_cost=1,
    targets=(
        ('call_module', 'torch.nn.AdaptiveAvgPool2d'),
        ('call_function', 'torch.nn.functional.adaptive_avg_pool2d'),
    ),
))
register_layout_template(LayoutTemplate(
    template_id='dense',
    version='1',
    family='dense',
    kind=LINEAR_TRANSFORM,
    depth_cost=1,
    targets=(('call_module', 'torch.nn.Linear'),),
    weight_roles=('weight', 'bias'),
))
# A flatten is a relabeling of slot coordinates, so it consumes nothing here.
# Whether it costs a level is packing's answer, not the frontend's: only if
# global layout reconciliation cannot satisfy both sides by relabeling does it
# materialize a repack, and only then is a level charged.
register_layout_template(LayoutTemplate(
    template_id='flatten',
    version='1',
    family='flatten',
    kind=LAYOUT,
    depth_cost=0,
    targets=(
        ('call_module', 'torch.nn.Flatten'),
        ('call_function', 'torch.flatten'),
        ('call_method', 'flatten'),
        ('call_method', 'view'),
        ('call_method', 'reshape'),
    ),
))
# Fusing the pool into the convolution's matrix saves a level, so it outranks
# lowering the two layers separately wherever the frontend can pair them.
register_layout_template(LayoutTemplate(
    template_id='conv_pool',
    version='1',
    family='conv2d',
    kind=LINEAR_TRANSFORM,
    depth_cost=1,
    targets=(('call_module', 'torch.nn.Conv2d'),),
    weight_roles=('weight', 'bias'),
    priority=10,
    condition=lambda parameters, input_spec: 'pool_kind' in parameters,
))


@dataclass(frozen=True, eq=False)
class VectorizedLayer:
  """One vectorized layer, declared without holding any array.

  ``eq=False`` is deliberate. The generated ``__eq__`` would compare fields
  structurally, and this type is meant to be a stable identity: two layers are
  the same layer when their digests agree, which covers the weight bytes via
  the digests recorded in ``weights``.
  """

  name: str
  kind: str
  template_id: str
  input_spec: TensorSpec
  output_spec: TensorSpec
  # Ids of the layers feeding this one; empty means the program input.
  inputs: tuple[str, ...] = ()
  params: tuple[tuple[str, Any], ...] = ()
  # role -> weight digest, resolved against the program's WeightTable.
  weights: tuple[tuple[str, str], ...] = ()
  activation_name: Optional[str] = None
  # Captured from the registries at construction so the layer stands alone.
  # Resolving these lazily would make the layer's meaning depend on mutable
  # global state: replacing a template or an activation would change what an
  # already-digested program computes while its digest stayed identical.
  template_version: str = field(init=False, default='')
  activation_coefficients: tuple[float, ...] = field(init=False, default=())
  depth_cost: int = field(init=False, default=0)
  digest: str = field(init=False)

  def __post_init__(self):
    if self.kind not in VECTORIZED_KINDS:
      raise ValueError(
          f'layer {self.name!r} has kind {self.kind!r}; expected one of '
          f'{sorted(VECTORIZED_KINDS)}. Rescale and bootstrap are HE '
          'realizations chosen by packing, not vectorized layer kinds.'
      )
    object.__setattr__(self, 'inputs', tuple(str(i) for i in self.inputs))
    object.__setattr__(self, 'params', _normalize_params(self.params))
    object.__setattr__(self, 'weights', tuple(
        (str(role), str(digest))
        for role, digest in sorted(dict(self.weights).items())
    ))
    if self.kind == POLYNOMIAL_ACTIVATION:
      if self.activation_name is None:
        raise ValueError(
            f'layer {self.name!r} is a polynomial activation but names no '
            'polynomial.'
        )
      polynomial = activation(self.activation_name)
      object.__setattr__(
          self, 'activation_coefficients', polynomial.coefficients
      )
      object.__setattr__(self, 'depth_cost', polynomial.depth_cost)
    elif self.template_id:
      template = layout_template(self.template_id)
      object.__setattr__(self, 'template_version', template.version)
      object.__setattr__(self, 'depth_cost', template.depth_cost)
    object.__setattr__(
        self,
        'digest',
        _digest_payload({
            'name': self.name,
            'kind': self.kind,
            'template': self.template_id,
            'template_version': self.template_version,
            'activation': self.activation_name,
            'activation_coefficients': list(self.activation_coefficients),
            'depth_cost': self.depth_cost,
            'inputs': list(self.inputs),
            'params': [list(item) for item in self.params],
            'weights': [list(item) for item in self.weights],
            'input_shape': self.input_spec.shape,
            'input_packing': self.input_spec.packing,
            'input_layout': self.input_spec.layout_fingerprint,
            'output_shape': self.output_spec.shape,
            'output_packing': self.output_spec.packing,
            'output_layout': self.output_spec.layout_fingerprint,
        }),
    )

  def __eq__(self, other):
    if not isinstance(other, VectorizedLayer):
      return NotImplemented
    return self.digest == other.digest

  def __hash__(self):
    return hash(self.digest)

  def parameters(self) -> dict[str, Any]:
    return dict(self.params)

  def polynomial(self) -> ActivationPolynomial:
    """The activation this layer applies, from its own captured coefficients."""
    if self.kind != POLYNOMIAL_ACTIVATION:
      raise TypeError(f'layer {self.name!r} is not a polynomial activation.')
    return ActivationPolynomial(
        name=self.activation_name, coefficients=self.activation_coefficients
    )

  def resolve_weights(self, weights: WeightTable) -> dict[str, np.ndarray]:
    return {role: weights[digest] for role, digest in self.weights}





@dataclass(frozen=True, eq=False)
class VectorizedProgram:
  """A whole model, vectorized layer by layer and ready for packing.

  This is the frontend's entire output. It names no HE concept: no level, no
  modulus, no ring. Packing reconciles these layers into one global layout,
  derives a secure RingConfig from the reconciled slot count and this
  program's critical-path depth, and materializes the PP-ops.
  """

  layers: tuple[VectorizedLayer, ...]
  weights: WeightTable
  input_spec: TensorSpec
  output: str
  digest: str = field(init=False)

  def __post_init__(self):
    layers = tuple(self.layers)
    # Snapshot the weights. Accepting a builder, or holding a table someone
    # else still has a handle on, would let the weight set change after
    # ``digest`` was computed over it -- and packing derives a security
    # parameter set from these bytes.
    weights = self.weights
    if not isinstance(weights, WeightTable):
      weights = getattr(weights, 'freeze', lambda: WeightTable(weights))()
    else:
      weights = WeightTable(weights.as_dict())
    object.__setattr__(self, 'weights', weights)
    seen: dict[str, VectorizedLayer] = {}
    for layer in layers:
      if layer.name in seen:
        raise ValueError(f'duplicate layer name {layer.name!r}.')
      if layer.name == PROGRAM_INPUT:
        raise ValueError(
            f'{PROGRAM_INPUT!r} names the program input; a layer cannot take '
            'that name.'
        )
      for source in layer.inputs:
        if source != PROGRAM_INPUT and source not in seen:
          raise ValueError(
              f'layer {layer.name!r} consumes {source!r}, which is not a '
              'preceding layer; layers must be in topological order.'
          )
      for role, digest in layer.weights:
        if digest not in self.weights:
          raise ValueError(
              f'layer {layer.name!r} names weight {role!r} '
              f'({digest[:16]}...), absent from the program weight table.'
          )
      seen[layer.name] = layer
    if layers and self.output not in seen:
      raise ValueError(
          f'program output {self.output!r} is not a layer.'
      )
    object.__setattr__(self, 'layers', layers)
    object.__setattr__(
        self,
        'digest',
        _digest_payload({
            'layers': [layer.digest for layer in layers],
            'weights': list(self.weights.digests()),
            'input_shape': self.input_spec.shape,
            'input_packing': self.input_spec.packing,
            'input_layout': self.input_spec.layout_fingerprint,
            'output': self.output,
        }),
    )

  def __eq__(self, other):
    if not isinstance(other, VectorizedProgram):
      return NotImplemented
    return self.digest == other.digest

  def __hash__(self):
    return hash(self.digest)

  def layer(self, name: str) -> VectorizedLayer:
    for candidate in self.layers:
      if candidate.name == name:
        return candidate
    raise KeyError(f'no layer {name!r} in this program.')

  def critical_depth(self) -> int:
    """Longest depth-weighted path through the layer DAG.

    Max over branches, never the sum: two arms of a residual that rejoin cost
    what the deeper arm costs, because they are evaluated against the same
    modulus chain rather than one after the other. Packing turns this into
    ``ModelSpec.num_q``, so summing here would over-provision the ring and, at
    the top of the chain, make a satisfiable model look unsatisfiable.
    """
    depth: dict[str, int] = {}
    for layer in self.layers:
      arriving = max(
          (depth.get(source, 0) for source in layer.inputs), default=0
      )
      depth[layer.name] = arriving + layer.depth_cost
    return max(depth.values(), default=0)

  def max_live_slots(self) -> int:
    """Largest physical slot count any single value in the program occupies."""
    return max(
        [self.input_spec.physical_size]
        + [layer.output_spec.physical_size for layer in self.layers]
    )


# =============================================================
# torch.fx frontend: the one way a model enters this compiler.
#
# torch is imported inside these functions, never at module scope, so
# ``import jaxite_word.nn`` stays free of torch exactly as it stays free of
# JAX. That is also why LayoutTemplate.targets are dotted strings: they are
# resolved to real objects here, once, and matched by identity afterwards --
# ``F.avg_pool2d`` is ``torch._C._nn.avg_pool2d`` and ``x * x`` traces to
# ``operator.mul``, neither of which can be matched by name.
# =============================================================


class VectorizeError(Exception):
  """A model could not be vectorized. The message says what to change."""


class ActivationSubstitutionWarning(UserWarning):
  """A non-linear activation was replaced by a polynomial approximation."""


def _import_torch():
  try:
    import torch
    import torch.fx  # noqa: F401  (registers the fx submodule)
    return torch
  except ImportError as error:
    raise VectorizeError(
        'vectorize() needs PyTorch, which failed to import: '
        f'{error}. Only this entry point needs it; the rest of jaxite_word '
        'does not.'
    ) from error


def _resolve_dotted(dotted: str):
  """Resolve ``'torch.nn.functional.avg_pool2d'`` to the object itself."""
  import importlib
  parts = dotted.split('.')
  for split in range(len(parts) - 1, 0, -1):
    try:
      module = importlib.import_module('.'.join(parts[:split]))
    except ImportError:
      continue
    target = module
    try:
      for attribute in parts[split:]:
        target = getattr(target, attribute)
    except AttributeError:
      continue
    return target
  raise VectorizeError(f'cannot resolve template target {dotted!r}.')


def _lookup_by_type(index, module, default=()):
  """What ``index`` holds for ``module``, honouring subclasses via the MRO.

  A subclass of a supported layer -- a project's own Conv2d wrapper, say --
  resolves to the base class entry rather than falling off the registry.
  """
  for klass in type(module).__mro__:
    if klass in index:
      return index[klass]
  return default


def _selection_context(parameters, module=None, output_shape=None) -> dict:
  """Parameters a template condition may branch on.

  Wider than what the layer stores: a condition wants the weight and output
  shapes to decide whether its vectorization suits this layer, but those are
  derived, so keeping them out of the stored parameters keeps the digest
  minimal and canonical.
  """
  context = dict(parameters)
  weight = getattr(module, 'weight', None) if module is not None else None
  if weight is not None:
    context['weight_shape'] = tuple(int(size) for size in weight.shape)
    context['has_bias'] = getattr(module, 'bias', None) is not None
  if output_shape is not None:
    context['output_shape'] = tuple(output_shape)
  return context


def _choose_template(claimed, parameters=None, input_spec=None) -> str:
  """First claimed template whose condition holds, highest priority first."""
  for template_id in claimed:
    if layout_template(template_id).applies(parameters, input_spec):
      return template_id
  raise VectorizeError(
      f'templates {list(claimed)} claim this operator but none applies to '
      f'parameters {dict(parameters or {})}.'
  )


def _resolved_target_index():
  """Map live torch objects to the templates claiming them.

  Rebuilt per call: the registries are mutable, and a stale index would keep
  selecting a template that has since been replaced.
  """
  index: dict[str, dict] = {
      'call_module': {}, 'call_function': {}, 'call_method': {}
  }
  for template in _LAYOUT_TEMPLATES.values():
    for node_op, dotted in template.targets:
      if node_op == 'call_method':
        key = dotted
      else:
        key = _resolve_dotted(dotted)
      index[node_op].setdefault(key, []).append(template.template_id)
  for node_op, table in index.items():
    for key, ids in table.items():
      ids.sort(key=lambda name: (-_LAYOUT_TEMPLATES[name].priority, name))
  return index


# Activations that are not polynomials and must be substituted to run under
# HE, mapped to the registry name whose default substitution applies.
_ACTIVATION_TARGETS = (
    ('call_module', 'torch.nn.ReLU', 'relu'),
    ('call_module', 'torch.nn.SiLU', 'silu'),
    ('call_function', 'torch.nn.functional.relu', 'relu'),
    ('call_function', 'torch.nn.functional.silu', 'silu'),
    ('call_function', 'torch.relu', 'relu'),
)

# Reshapes the demos use. Under a row-major slot layout these permute nothing,
# so they are recorded as a change of logical shape and emit no computation.
_RESHAPE_METHODS = frozenset({'view', 'reshape', 'flatten'})
# Calls that produce a Python int rather than a tensor. They exist only to
# feed a reshape, so they are metadata and never become a layer.
_METADATA_METHODS = frozenset({'size', 'dim', 'numel'})


def _activation_index():
  index = {'call_module': {}, 'call_function': {}}
  for node_op, dotted, source in _ACTIVATION_TARGETS:
    try:
      key = _resolve_dotted(dotted)
    except VectorizeError:
      continue  # An activation this torch build does not ship.
    index[node_op][key] = source
  return index


def _trace(model, torch, leaf_types=()):
  """Symbolically trace ``model``, turning fx failures into guidance.

  ``leaf_types`` are the module classes the registries claim. fx only treats
  ``torch.nn`` modules as leaves by default, so a project's own subclass of a
  supported layer would be traced *through*, decomposing into get_attr and a
  raw functional call. Treating a claimed subclass as a leaf is what makes
  MRO-based template lookup actually reachable.
  """
  leaf_types = tuple(leaf_types)

  class _Tracer(torch.fx.Tracer):

    def is_leaf_module(self, module, qualified_name):
      if leaf_types and isinstance(module, leaf_types):
        return True
      return super().is_leaf_module(module, qualified_name)

  try:
    graph = _Tracer().trace(model)
    graph_module = torch.fx.GraphModule(model, graph)
  except torch.fx.proxy.TraceError as error:
    raise VectorizeError(
        f'{type(model).__name__} cannot be symbolically traced: {error}. '
        'This is usually data-dependent control flow -- an `if` on a shape or '
        'a value, such as `if x.shape[-1] == 32:`. Decide it in __init__ '
        'instead, so the forward pass is a fixed graph.'
    ) from error
  graph_module.graph.eliminate_dead_code()
  graph_module.recompile()
  return graph_module


def _propagate_shapes(graph_module, model, input_spec, torch):
  """Annotate every node with a concrete shape by running one example.

  This executes the graph, so it must run on a private copy: modules with
  buffers update them in place. A BatchNorm in training mode would fold this
  synthetic example into the caller's running statistics, silently altering a
  model the caller has not finished training.
  """
  from torch.fx.passes.shape_prop import ShapeProp
  try:
    parameter = next(model.parameters())
    dtype, device = parameter.dtype, parameter.device
  except StopIteration:
    dtype, device = torch.get_default_dtype(), torch.device('cpu')
  example = torch.zeros(
      (1,) + tuple(input_spec.shape), dtype=dtype, device=device
  )
  try:
    with torch.no_grad():
      ShapeProp(graph_module).propagate(example)
  except Exception as error:
    raise VectorizeError(
        f'could not infer shapes for {type(model).__name__} on input '
        f'{tuple(input_spec.shape)}: {error}. Check that input_spec matches '
        'what the model expects, without the batch dimension.'
    ) from error


def _unbatched_shape(node, description):
  meta = node.meta.get('tensor_meta')
  if meta is None:
    raise VectorizeError(
        f'{description} produced no tensor shape; vectorize() supports only '
        'tensor-valued operations.'
    )
  shape = tuple(int(size) for size in meta.shape)
  if len(shape) < 2 or shape[0] != 1:
    raise VectorizeError(
        f'{description} has shape {shape}; vectorize() maps one unbatched '
        'sample, so every activation must keep a leading batch of 1.'
    )
  return shape[1:]


def _spatial(value, fallback=None):
  if value is None:
    return fallback
  if isinstance(value, (tuple, list)):
    return tuple(int(item) for item in value)
  return (int(value), int(value))


def _argument(node, index, name, default=None):
  if len(node.args) > index:
    return node.args[index]
  return node.kwargs.get(name, default)


def _pool_parameters_from_module(module, torch):
  if isinstance(module, torch.nn.AdaptiveAvgPool2d):
    return 'adaptive_avg_pool2d', {
        'output_size': _spatial(module.output_size)
    }
  kernel = _spatial(module.kernel_size)
  return 'avg_pool2d', {
      'kernel_size': kernel,
      'stride': _spatial(module.stride, kernel),
      'padding': _spatial(module.padding, (0, 0)),
      'count_include_pad': bool(module.count_include_pad),
  }


def _pool_parameters_from_node(node, template_id):
  if template_id == 'adaptive_avg_pool2d':
    return 'adaptive_avg_pool2d', {
        'output_size': _spatial(_argument(node, 1, 'output_size'))
    }
  kernel = _spatial(_argument(node, 1, 'kernel_size'))
  return 'avg_pool2d', {
      'kernel_size': kernel,
      'stride': _spatial(_argument(node, 2, 'stride'), kernel),
      'padding': _spatial(_argument(node, 3, 'padding', 0), (0, 0)),
      'count_include_pad': bool(
          _argument(node, 5, 'count_include_pad', True)
      ),
  }


def vectorize(
    model,
    input_spec: TensorSpec,
    *,
    fuse_pooling: bool = True,
    warn: bool = True,
) -> VectorizedProgram:
  """Vectorize a real ``torch.nn.Module`` into a VectorizedProgram.

  ``model`` is traced with ``torch.fx``; every node is matched against the
  layout-template registry by the identity of its target, and becomes a layer
  carrying application semantics only. Nothing here names a level, a modulus
  or an HE operator: packing decides those.

  Non-linear activations cannot run under HE and are replaced by the
  polynomial the activation registry names, with a warning saying which. Any
  node the registry does not claim raises rather than being skipped, because
  skipping one would silently evaluate a different function.
  """
  torch = _import_torch()
  if not isinstance(model, torch.nn.Module):
    raise VectorizeError(
        f'vectorize() takes a torch.nn.Module, got {type(model).__name__}.'
    )
  if not isinstance(input_spec, TensorSpec):
    input_spec = TensorSpec(tuple(input_spec), ChannelMajor())

  # Everything downstream reads a private copy in eval mode. Shape inference
  # has to execute the graph, and the caller's model must come back untouched
  # -- same buffers, same training flags -- whether vectorize succeeds or
  # raises partway through.
  import copy
  templates = _resolved_target_index()
  activations = _activation_index()
  leaf_types = tuple({
      key for key in
      list(templates['call_module']) + list(activations['call_module'])
      if isinstance(key, type)
  })
  traced_model = copy.deepcopy(model).eval()
  graph_module = _trace(traced_model, torch, leaf_types)
  _propagate_shapes(graph_module, traced_model, input_spec, torch)
  submodules = dict(graph_module.named_modules())

  weights = WeightTableBuilder()
  layers: list[VectorizedLayer] = []
  # fx node name -> (producing layer name or None for the model input, spec)
  values: dict[str, tuple[Optional[str], TensorSpec]] = {}
  metadata: set[str] = set()
  fused: set[str] = set()
  substituted: list[tuple[str, str, str]] = []
  names: set[str] = set()
  output_layer: Optional[str] = None

  def unique(base: str) -> str:
    candidate, index = base, 1
    while candidate in names:
      index += 1
      candidate = f'{base}_{index}'
    names.add(candidate)
    return candidate

  def tensor_inputs(node) -> list:
    found = []
    for argument in node.all_input_nodes:
      if argument.name in metadata:
        continue
      if argument.name not in values:
        raise VectorizeError(
            f'node {node.name!r} consumes {argument.name!r}, which produced '
            'no vectorized value.'
        )
      found.append(argument)
    return found

  def emit(node, *, kind, template_id='', parameters=None, weight_map=None,
           activation_name=None, sources=None, output_shape=None):
    source_nodes = tensor_inputs(node) if sources is None else sources
    producers = tuple(
        values[item.name][0] or PROGRAM_INPUT for item in source_nodes
    )
    if not source_nodes:
      raise VectorizeError(f'node {node.name!r} has no tensor input.')
    shape = output_shape or _unbatched_shape(node, f'node {node.name!r}')
    layer = VectorizedLayer(
        name=unique(node.name),
        kind=kind,
        template_id=template_id,
        input_spec=values[source_nodes[0].name][1],
        output_spec=_channel_major(shape),
        inputs=producers,
        params=parameters or {},
        weights=weight_map or {},
        activation_name=activation_name,
    )
    layers.append(layer)
    values[node.name] = (layer.name, layer.output_spec)
    return layer

  def store_weights(module) -> dict[str, str]:
    stored = {'weight': weights.add(module.weight)}
    if getattr(module, 'bias', None) is not None:
      stored['bias'] = weights.add(module.bias)
    return stored

  def pool_after(node):
    """The single pooling consumer of ``node``, if it can be fused into it."""
    if not fuse_pooling or len(node.users) != 1:
      return None, None, None
    consumer = next(iter(node.users))
    if consumer.op == 'call_module':
      module = submodules.get(consumer.target)
      claimed = _lookup_by_type(templates['call_module'], module)
      if claimed and claimed[0] in ('avg_pool2d', 'adaptive_avg_pool2d'):
        kind, parameters = _pool_parameters_from_module(module, torch)
        return consumer, kind, parameters
    elif consumer.op == 'call_function':
      claimed = templates['call_function'].get(consumer.target, ())
      if claimed and claimed[0] in ('avg_pool2d', 'adaptive_avg_pool2d'):
        kind, parameters = _pool_parameters_from_node(consumer, claimed[0])
        return consumer, kind, parameters
    return None, None, None

  def substitute(node, source):
    polynomial = default_activation_for(source)
    substituted.append((node.name, source, polynomial.name))
    return emit(
        node,
        kind=POLYNOMIAL_ACTIVATION,
        activation_name=polynomial.name,
    )

  for node in graph_module.graph.nodes:
    description = f'{node.op} {node.target!r} (node {node.name!r})'

    if node.op == 'placeholder':
      if values:
        raise VectorizeError(
            'vectorize() maps a single-input model; '
            f'{type(model).__name__} takes more than one argument.'
        )
      values[node.name] = (None, input_spec)
      continue

    if node.op == 'output':
      sources = tensor_inputs(node)
      if len(sources) != 1:
        raise VectorizeError(
            'vectorize() maps a model with exactly one output tensor.'
        )
      output_layer = values[sources[0].name][0]
      if output_layer is None:
        raise VectorizeError(
            'the model returns its input unchanged; nothing to vectorize.'
        )
      continue

    if node.name in fused:
      continue

    if node.op == 'get_attr':
      raise VectorizeError(
          f'{description} reads a captured tensor constant. vectorize() maps '
          'parameters held by submodules; hoist this into an nn.Module '
          'attribute such as a Linear or Conv2d weight.'
      )

    if node.op == 'call_module':
      module = submodules.get(node.target)
      if module is None:
        raise VectorizeError(f'{description} names no submodule.')
      source = _lookup_by_type(activations['call_module'], module, None)
      if source is not None:
        substitute(node, source)
        continue
      claimed = _lookup_by_type(templates['call_module'], module)
      if not claimed:
        raise VectorizeError(
            f'{description} is a {type(module).__name__}, which no layout '
            'template vectorizes. Register one with '
            'register_layout_template(...), or replace the layer. '
            'BatchNorm in particular must be folded into the preceding '
            'Conv2d or Linear before tracing, as the demos do at export.'
        )
      families = {layout_template(name).family for name in claimed}
      if len(families) != 1:
        raise VectorizeError(
            f'{description} is claimed by templates from more than one '
            f'operator family ({sorted(families)}); a template must be an '
            'alternative vectorization of one family.'
        )
      family = families.pop()
      source_spec = values[tensor_inputs(node)[0].name][1]

      if family == 'flatten':
        values[node.name] = (
            values[tensor_inputs(node)[0].name][0],
            _channel_major(_unbatched_shape(node, description)),
        )
        continue

      if family == 'conv2d':
        if module.groups != 1 or tuple(module.dilation) != (1, 1):
          raise VectorizeError(
              f'{description} uses groups={module.groups} '
              f'dilation={tuple(module.dilation)}; the conv2d template '
              'vectorizes dense, undilated convolutions only.'
          )
        parameters = {
            'stride': _spatial(module.stride),
            'padding': _spatial(module.padding, (0, 0)),
        }
        output_shape = _unbatched_shape(node, description)
        pool_node, pool_kind, pool_parameters = pool_after(node)
        if pool_node is not None:
          # Composing both linear maps into one matrix saves a level, which
          # is the scarce resource; the pool then emits nothing of its own.
          parameters['intermediate_shape'] = output_shape
          parameters['pool_kind'] = pool_kind
          parameters.update(
              {f'pool_{key}': value for key, value in pool_parameters.items()}
          )
          output_shape = _unbatched_shape(pool_node, description)
        template_id = _choose_template(
            claimed,
            _selection_context(parameters, module, output_shape),
            source_spec,
        )
        if pool_node is not None and template_id != 'conv_pool':
          # A template that declined the fusion cannot consume the pool, so
          # leave the pool to be vectorized as its own layer.
          for key in [k for k in parameters if k.startswith('pool_')]:
            del parameters[key]
          parameters.pop('intermediate_shape', None)
          pool_node = None
          output_shape = _unbatched_shape(node, description)
        layer = emit(
            node,
            kind=LINEAR_TRANSFORM,
            template_id=template_id,
            parameters=parameters,
            weight_map=store_weights(module),
            output_shape=output_shape,
        )
        if pool_node is not None:
          fused.add(pool_node.name)
          values[pool_node.name] = (layer.name, layer.output_spec)
        continue

      if family == 'dense':
        emit(
            node,
            kind=LINEAR_TRANSFORM,
            template_id=_choose_template(
                claimed,
                _selection_context({}, module, None),
                source_spec,
            ),
            weight_map=store_weights(module),
        )
        continue

      pool_kind, parameters = _pool_parameters_from_module(module, torch)
      emit(
          node,
          kind=LINEAR_TRANSFORM,
          template_id=_choose_template(claimed, parameters, source_spec),
          parameters=parameters,
      )
      continue

    if node.op == 'call_function':
      source = activations['call_function'].get(node.target)
      if source is not None:
        substitute(node, source)
        continue
      if node.target is operator.mul:
        left, right = node.args[:2]
        if left is right:
          # x * x is a degree-2 polynomial, the one non-linearity HE runs
          # natively. It needs no substitution and no warning.
          substitute_free = emit(
              node,
              kind=POLYNOMIAL_ACTIVATION,
              activation_name='square',
              sources=[left],
          )
          del substitute_free
          continue
        raise VectorizeError(
            f'{description} multiplies two different values. Only x * x is a '
            'vectorized kind today; a general product of two activations '
            'would need a multiply kind that packing and mapping do not yet '
            'materialize.'
        )
      if node.target is operator.add:
        sources = tensor_inputs(node)
        if len(sources) != 2:
          raise VectorizeError(
              f'{description} adds a non-tensor operand. Fold constant '
              'offsets into a preceding Linear or Conv2d bias.'
          )
        left, right = (values[item.name][1] for item in sources)
        result_shape = _unbatched_shape(node, description)
        if left.shape != right.shape or tuple(result_shape) != left.shape:
          # Torch would broadcast; ciphertext addition is slotwise and cannot.
          # Encoding this as a plain SIMD add would compute something else.
          raise VectorizeError(
              f'{description} adds shapes {left.shape} and {right.shape} '
              f'giving {tuple(result_shape)}, which relies on broadcasting. '
              'Ciphertext addition is slotwise, so both operands must already '
              'have identical shape and layout; make the broadcast explicit '
              'with an expand or a matching layer.'
          )
        emit(node, kind=ADD, sources=sources)
        continue
      claimed = templates['call_function'].get(node.target, ())
      if not claimed:
        raise VectorizeError(
            f'{description} is not vectorized by any layout template. '
            'Register one with register_layout_template(...) naming this '
            'target, or express the model with a supported operation.'
        )
      source_spec = values[tensor_inputs(node)[0].name][1]
      template_id = _choose_template(claimed, {}, source_spec)
      if layout_template(template_id).family == 'flatten':
        values[node.name] = (
            values[tensor_inputs(node)[0].name][0],
            _channel_major(_unbatched_shape(node, description)),
        )
        continue
      pool_kind, parameters = _pool_parameters_from_node(node, template_id)
      emit(
          node,
          kind=LINEAR_TRANSFORM,
          template_id=_choose_template(claimed, parameters, source_spec),
          parameters=parameters,
      )
      continue

    if node.op == 'call_method':
      if node.target in _METADATA_METHODS:
        # `x.size(0)` exists only to feed a reshape. It is not a value.
        metadata.add(node.name)
        continue
      if node.target in _RESHAPE_METHODS:
        sources = tensor_inputs(node)
        shape = _unbatched_shape(node, description)
        if math.prod(shape) != math.prod(values[sources[0].name][1].shape):
          raise VectorizeError(
              f'{description} changes the element count from '
              f'{values[sources[0].name][1].shape} to {shape}.'
          )
        # Row-major slots make this a relabeling, not a permutation, so it
        # carries no computation -- only the new logical shape.
        values[node.name] = (
            values[sources[0].name][0], _channel_major(shape)
        )
        continue
      raise VectorizeError(
          f'{description} calls a tensor method vectorize() does not map. '
          f'Supported methods: {sorted(_RESHAPE_METHODS | _METADATA_METHODS)}.'
      )

    raise VectorizeError(f'{description} is not a node vectorize() maps.')

  if output_layer is None:
    raise VectorizeError('the traced graph has no output.')
  if warn and substituted:
    for node_name, source, polynomial_name in substituted:
      polynomial = activation(polynomial_name)
      warnings.warn(
          f'{node_name}: {source} is not evaluable under homomorphic '
          f'encryption and was replaced by the polynomial '
          f'{polynomial_name!r} (degree {polynomial.degree}, coefficients '
          f'{polynomial.coefficients}). Accuracy will differ from the '
          'cleartext model. Register a different approximation with '
          f'nn.register_activation(..., substitutes=({source!r},)).',
          ActivationSubstitutionWarning,
          stacklevel=2,
      )
  return VectorizedProgram(
      layers=tuple(layers),
      weights=weights,
      input_spec=input_spec,
      output=output_layer,
  )









__all__ = [
    'ADD',
    'ActivationPolynomial',
    'ActivationSubstitutionWarning',
    'ChannelMajor',
    'LAYOUT',
    'LINEAR_TRANSFORM',
    'LayoutTemplate',
    'POLYNOMIAL_ACTIVATION',
    'SUPPORTED_FAMILIES',
    'PROGRAM_INPUT',
    'StrideMultiplexed',
    'TensorLayout',
    'TensorSpec',
    'VECTORIZED_KINDS',
    'VectorizedLayer',
    'VectorizedProgram',
    'VectorizeError',
    'WeightTable',
    'WeightTableBuilder',
    'activation',
    'default_activation_for',
    'layout_template',
    'layout_template_candidates',
    'normalize_weight',
    'register_activation',
    'register_layout_template',
    'select_layout_template',
    'unregister_layout_template',
    'vectorize',
    'weight_digest',
]

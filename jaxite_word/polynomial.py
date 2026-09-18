import math
from typing import Any, List, Optional, Union

import jax
import jax.numpy as jnp

import util
import finite_field as ff_context
import ntt_mm as ntt


# Global registry for Polynomial pytree static state
_POLYNOMIAL_REGISTRY = {}

# Fields derived entirely from the structural metadata below.  They live in
# the registry because JAX arrays cannot be pytree auxiliary data, but they do
# not need to participate in the compilation key themselves.
_POLYNOMIAL_BASE_FIELDS = frozenset({
    'batch', 'num_elements', 'num_moduli', 'degree', 'precision', 'r', 'c',
    'degree_layout', 'modulus_dtype', 'moduli',
    'shape_in_ntt_all_limbs', 'ntt_ctx',
    'moduli_array', '_polynomial',
})


class Polynomial:
  """Polynomial class for FHE operations.

  Ciphertext payloads have one canonical representation:
  ``(batch, num_elements, r, c, num_moduli)``.
  """

  def __init__(
      self,
      shapes: dict[str, Any],
      parameters: Optional[dict[str, Any]] = None,
      *,
      payload: Optional[jnp.ndarray] = None,
  ):
    """Initialize the Polynomial object.

    Each polynomial is a rank-5 tensor with shape
    ``(batch, num_elements, r, c, num_moduli)``, where ``r * c == degree``.

    Args:
        shapes (dict[str, Any]): A dictionary containing the shapes of the
          polynomial. Expected keys include:
          - batch: The batch size of the polynomial.
          - num_elements: The number of elements in the polynomial.
          - degree: The degree of the polynomial.
          - num_moduli: The number of moduli in the polynomial.
          - precision: The precision of the polynomial.
          - degree_layout (Optional): Tuple (r, c) for NTT layout.
        parameters (Optional[dict[str, Any]], optional): A dictionary
          containing additional parameters. Expected keys include:
          - moduli: The moduli of the polynomial. Can be a single integer
            for a single modulus or a list of integers for a multi-modulus.
          - ntt_ctx: An externally provided NTT context. If provided,
            it must have an 'ff_ctx' attribute.
          - finite_field_context: A callable to create the finite field
            context (e.g., ff_context.BarrettContext). Inject BATLazyContext
            to select the BAT-lazy NTT.
        payload: Optional canonical rank-5 payload. Supplying it avoids an
          otherwise unused zero allocation when wrapping an existing array.
    """
    self.batch = shapes['batch']
    self.num_elements = shapes['num_elements']
    self.num_moduli = shapes['num_moduli']
    self.degree = shapes['degree']
    self.precision = shapes['precision']
    if self.degree <= 0:
      raise ValueError(f'degree must be positive, got {self.degree}.')
    if 'degree_layout' in shapes:
      degree_layout = tuple(shapes['degree_layout'])
      if len(degree_layout) != 2:
        raise ValueError(
            'degree_layout must contain exactly two dimensions (r, c); '
            f'got {degree_layout}.'
        )
      self.r, self.c = degree_layout
    else:
      # Pick the closest factor pair deterministically. This preserves the
      # familiar square layout for square degrees and remains valid for every
      # positive degree.
      self.r = math.isqrt(self.degree)
      while self.degree % self.r:
        self.r -= 1
      self.c = self.degree // self.r
    if self.r <= 0 or self.c <= 0 or self.r * self.c != self.degree:
      raise ValueError(
          f'degree_layout {(self.r, self.c)} does not multiply to degree '
          f'{self.degree}.'
      )
    self.degree_layout = (self.r, self.c)

    if self.precision <= 32:
      self.modulus_dtype = jnp.uint32
    else:
      self.modulus_dtype = jnp.uint64

    if parameters is not None and 'moduli' in parameters:
      self.moduli = parameters['moduli']
    else:
      self.moduli = util.find_moduli_ntt(
          self.num_moduli, self.precision, 2 * self.degree
      )
    if isinstance(self.moduli, int):
      self.moduli = [self.moduli]

    # NTT Parameters.  Bit-reversal state belongs to the NTT context; keeping
    # another degree-sized array on every wrapper was both unused and costly.
    self.shape_in_ntt_all_limbs = (-1, self.r, self.c, self.num_moduli)

    # Allow external NTT context injection
    # (used by HEParameterCache for shared contexts)
    if parameters is not None and 'ntt_ctx' in parameters:
      self.ntt_ctx = parameters['ntt_ctx']
      if not hasattr(self.ntt_ctx, 'ff_ctx'):
        raise ValueError("Injected ntt_ctx must have 'ff_ctx' attribute")
    else:
      if parameters is not None and 'finite_field_context' in parameters:
        finite_field_context = parameters['finite_field_context'](
            moduli=self.moduli
        )
      else:
        finite_field_context = ff_context.BarrettContext(moduli=self.moduli)

      ntt_params = {
          'r': self.r,
          'c': self.c,
          'finite_field_context': finite_field_context,
      }

      ntt_ctx_cls = ntt.ntt_ciphertext_context_for(finite_field_context)
      if ntt_ctx_cls is ntt.NTTCiphertextBarrettContext:
        # Operator instances repeatedly build identical Barrett contexts for
        # the same level. Share those immutable tables, while respecting the
        # declared dispatch hook for every other backend.
        self.ntt_ctx = ntt.get_shared_barrett_ntt_context(
            self.moduli, ntt_params
        )
      else:
        self.ntt_ctx = ntt_ctx_cls(
            moduli=self.moduli, parameters=ntt_params
        )

    self.moduli_array = jnp.array(self.moduli, dtype=self.modulus_dtype)
    if payload is None:
      payload = jnp.zeros(self._payload_shape, dtype=self.modulus_dtype)
    self.polynomial = payload
    self.validate()

  @property
  def _payload_shape(self) -> tuple[int, int, int, int, int]:
    return (self.batch, self.num_elements, self.r, self.c, self.num_moduli)

  @property
  def polynomial(self) -> jnp.ndarray:
    return self._polynomial

  @polynomial.setter
  def polynomial(self, payload: jnp.ndarray) -> None:
    payload = jnp.asarray(payload)
    self._validate_payload(payload)
    self._polynomial = payload

  def _validate_payload(self, payload: jnp.ndarray) -> None:
    if payload.ndim != 5:
      raise ValueError(
          'Polynomial payload must be rank 5 with shape '
          '(batch, num_elements, r, c, num_moduli); '
          f'got rank {payload.ndim} shape {payload.shape}.'
      )
    if tuple(payload.shape) != self._payload_shape:
      raise ValueError(
          f'Polynomial payload shape must be {self._payload_shape}; '
          f'got {tuple(payload.shape)}.'
      )

  @classmethod
  def from_array(
      cls,
      array: jnp.ndarray,
      shapes: dict[str, Any],
      parameters: Optional[dict[str, Any]] = None,
  ) -> 'Polynomial':
    """Constructs a Polynomial from an already-canonical payload."""
    return cls(shapes, parameters, payload=array)

  def replace_payload(self, payload: jnp.ndarray) -> 'Polynomial':
    """Replaces the payload after validating its canonical shape."""
    self.polynomial = payload
    return self

  def to_array(self) -> jnp.ndarray:
    """Returns the canonical rank-5 payload."""
    self.validate()
    return self.polynomial

  def batch_slice(self, batch_index: int) -> 'Polynomial':
    """Returns one batch item as another canonical Polynomial."""
    if not 0 <= batch_index < self.batch:
      raise IndexError(
          f'batch_index {batch_index} out of range for batch {self.batch}.'
      )
    return self._clone_with_payload(
        self.polynomial[batch_index : batch_index + 1],
        batch=1,
    )

  def _clone_with_payload(
      self,
      payload: jnp.ndarray,
      *,
      batch: Optional[int] = None,
      num_elements: Optional[int] = None,
      moduli: Optional[List[int]] = None,
      ntt_ctx=None,
  ) -> 'Polynomial':
    """Fast internal clone with an atomically validated payload/metadata set."""
    obj = object.__new__(type(self))
    obj.__dict__ = self.__dict__.copy()
    if batch is not None:
      obj.batch = batch
    if num_elements is not None:
      obj.num_elements = num_elements
    if moduli is not None:
      if ntt_ctx is None:
        raise ValueError('ntt_ctx is required when cloning with new moduli.')
      obj.moduli = list(moduli)
      obj.num_moduli = len(obj.moduli)
      obj.moduli_array = jnp.asarray(obj.moduli, dtype=obj.modulus_dtype)
      obj.ntt_ctx = ntt_ctx
      obj.shape_in_ntt_all_limbs = (-1, obj.r, obj.c, obj.num_moduli)
    elif ntt_ctx is not None:
      obj.ntt_ctx = ntt_ctx
    obj.polynomial = payload
    return obj.validate()

  def __copy__(self) -> 'Polynomial':
    """Return a metadata-preserving wrapper sharing immutable contexts."""
    obj = object.__new__(type(self))
    obj.__dict__ = self.__dict__.copy()
    return obj

  def validate(self) -> 'Polynomial':
    """Validates structural payload, layout, and modulus metadata invariants.

    Public ciphertext operations additionally require ``modulus_dtype``.
    Keeping dtype out of this structural check permits private wide arithmetic
    intermediates used by kernel-level tests.
    """
    if self.r * self.c != self.degree:
      raise ValueError(
          f'Polynomial layout {(self.r, self.c)} does not match degree '
          f'{self.degree}.'
      )
    if self.degree_layout != (self.r, self.c):
      raise ValueError(
          f'degree_layout {self.degree_layout} does not match '
          f'{(self.r, self.c)}.'
      )
    if len(self.moduli) != self.num_moduli:
      raise ValueError(
          f'Expected {self.num_moduli} moduli, got {len(self.moduli)}.'
      )
    if tuple(self.moduli_array.shape) != (self.num_moduli,):
      raise ValueError(
          f'moduli_array must have shape {(self.num_moduli,)}, got '
          f'{tuple(self.moduli_array.shape)}.'
      )
    expected_moduli = tuple(int(modulus) for modulus in self.moduli)
    for context_name, context in (
        ('ntt_ctx', self.ntt_ctx),
        ('ntt_ctx.ff_ctx', self.ntt_ctx.ff_ctx),
    ):
      if hasattr(context, 'moduli'):
        context_moduli = tuple(int(modulus) for modulus in context.moduli)
        if context_moduli != expected_moduli:
          raise ValueError(
              f'{context_name} moduli {context_moduli} do not match '
              f'Polynomial moduli {expected_moduli}.'
          )
    self._validate_payload(self.polynomial)
    return self

  def tree_flatten(self):
    children = (self.polynomial,)
    aux = {
        k: v
        for k, v in self.__dict__.items()
        if k != '_polynomial'
    }
    extras = []
    for name in sorted(set(self.__dict__) - _POLYNOMIAL_BASE_FIELDS):
      value = self.__dict__[name]
      try:
        hash(value)
        token = value
      except TypeError:
        # Unknown mutable/static extension metadata must not alias another
        # wrapper's registry entry.
        token = ('identity', id(value))
      extras.append((name, token))
    pytree_key = (
        type(self), self.batch, self.num_elements, self.num_moduli,
        self.degree, self.precision, self.r, self.c,
        tuple(int(modulus) for modulus in self.moduli),
        type(self.ntt_ctx), id(self.ntt_ctx), tuple(extras),
    )
    _POLYNOMIAL_REGISTRY[pytree_key] = aux
    return children, pytree_key

  @classmethod
  def tree_unflatten(cls, pytree_key, children):
    obj = object.__new__(cls)
    aux = _POLYNOMIAL_REGISTRY[pytree_key]
    for k, v in aux.items():
      setattr(obj, k, v)
    (obj.polynomial,) = children
    obj.validate()
    return obj

  def random_init(self):
    self.polynomial = util.random_batched_ciphertext(
        (self.batch, self.num_elements, *self.degree_layout, self.num_moduli),
        self.moduli,
        dtype=self.modulus_dtype,
    )

  #####################
  # Getter Functions
  #####################
  @property
  def shape(self):
    return self.polynomial.shape

  def get_batch_polynomial(self) -> jnp.ndarray:
    return self.polynomial

  def get_polynomial(self, batch_index) -> jnp.ndarray:
    return self.polynomial[batch_index]

  def get_element(self, element_index) -> jnp.ndarray:
    return self.polynomial[:, element_index]

  def get_limb(self, limb_index) -> jnp.ndarray:
    return self.polynomial[..., limb_index]

  #####################
  # Setter Functions
  # Note set_polynomial, set_element, set_limb are in place operations
  # and not recommended in JAX, as JAX implement immutable arrays.
  #####################
  def set_batch_polynomial(self, batch_polynomial: jnp.ndarray) -> None:
    self.replace_payload(batch_polynomial)

  def set_polynomial(self, batch_index: int, polynomial: jnp.ndarray) -> None:
    self.polynomial = self.polynomial.at[batch_index].set(polynomial)

  def set_element(self, element_index: int, element: jnp.ndarray) -> None:
    self.polynomial = self.polynomial.at[:, element_index].set(element)

  def set_limb(self, limb_index: int, limb: jnp.ndarray) -> None:
    self.polynomial = self.polynomial.at[..., limb_index].set(limb)

  def get_moduli_array(self) -> jnp.ndarray:
    return self.moduli_array

  def get_moduli(self) -> Union[List[int], int]:
    return self.moduli

  def get_modulus(self, index: int) -> int:
    return self.moduli[index]

  #####################
  # Domain Conversion Functions
  #####################
  def to_ntt_form(self):
    current_shape = self.polynomial.shape
    reshaped_in = self.polynomial.reshape(self.shape_in_ntt_all_limbs)
    ntt_result = self.ntt_ctx.ntt(reshaped_in)
    self.polynomial = ntt_result.reshape(current_shape)

  def to_coeffs_form(self):
    current_shape = self.polynomial.shape
    reshaped_in = self.polynomial.reshape(self.shape_in_ntt_all_limbs)
    intt_result = self.ntt_ctx.intt(reshaped_in)
    self.polynomial = intt_result.reshape(current_shape)

  def to_compute_format(self):
    self.polynomial = self.ntt_ctx.to_computation_format(self.polynomial)

  def to_original_format(self):
    self.polynomial = self.ntt_ctx.to_original_format(self.polynomial)

  #####################
  # Arithmetic Functions Entire Polynomial
  #####################
  def _require_polynomial_operand(self, other: 'Polynomial') -> jnp.ndarray:
    if not isinstance(other, Polynomial):
      raise TypeError(
          f'Polynomial arithmetic requires a Polynomial operand, got '
          f'{type(other).__name__}.'
      )
    self.validate()
    other.validate()
    if self._payload_shape != other._payload_shape:
      raise ValueError(
          f'Polynomial operand shapes differ: {self._payload_shape} and '
          f'{other._payload_shape}.'
      )
    if tuple(self.moduli) != tuple(other.moduli):
      raise ValueError('Polynomial operands must use the same moduli.')
    if type(self.ntt_ctx.ff_ctx) is not type(other.ntt_ctx.ff_ctx):
      raise ValueError(
          'Polynomial operands must use the same finite-field reduction '
          f'backend; got {type(self.ntt_ctx.ff_ctx).__name__} and '
          f'{type(other.ntt_ctx.ff_ctx).__name__}.'
      )
    return other.polynomial

  def _add_inplace(self, other: 'Polynomial'):
    self.polynomial = self.polynomial + self._require_polynomial_operand(other)

  def _sub_inplace(self, other: 'Polynomial'):
    self.polynomial = self.polynomial - self._require_polynomial_operand(other)

  def _mul_wide_inplace(self, other: 'Polynomial'):
    other_array = self._require_polynomial_operand(other)
    self.polynomial = self.polynomial.astype(jnp.uint64) * other_array.astype(
        jnp.uint64
    )

  def _modmul_inplace(self, other: 'Polynomial'):
    other_array = self._require_polynomial_operand(other)
    temp = self.polynomial.astype(jnp.uint64) * other_array.astype(jnp.uint64)
    reduced = self.ntt_ctx.ff_ctx.modular_reduction(temp)
    self.polynomial = reduced.astype(self.modulus_dtype)

  def _mod_reduce_inplace(self):
    reduced = self.ntt_ctx.ff_ctx.modular_reduction(
        self.polynomial.astype(jnp.uint64)
    )
    self.polynomial = reduced.astype(self.modulus_dtype)

  #####################
  # Modulus Dropping Functions
  #####################
  def drop_last_modulus(self) -> 'Polynomial':
    """Returns a lower-level Polynomial without mutating this wrapper."""
    if self.num_moduli <= 1:
      raise ValueError('Cannot drop modulus from a single-limb polynomial.')

    reduced_payload = self.polynomial[..., :-1]
    new_moduli = list(self.moduli[:-1])
    # Never mutate the existing context: shallow Polynomial copies and
    # parameter-cache wrappers may intentionally share it.
    try:
      new_ff_ctx = self.ntt_ctx.ff_ctx.slice(self.num_moduli - 1)
      new_ntt_ctx = self.ntt_ctx.slice(self.num_moduli - 1, new_ff_ctx)
    except (AttributeError, NotImplementedError, TypeError):
      new_ff_ctx = type(self.ntt_ctx.ff_ctx)(moduli=new_moduli)
      if self.ntt_ctx is self.ntt_ctx.ff_ctx:
        # Mod-reduce-only wrappers use the finite-field context directly.
        new_ntt_ctx = new_ff_ctx
      else:
        new_ntt_ctx = type(self.ntt_ctx)(
            moduli=new_moduli,
            parameters={
                'r': self.r,
                'c': self.c,
                'finite_field_context': new_ff_ctx,
            },
        )
    return self._clone_with_payload(
        reduced_payload, moduli=new_moduli, ntt_ctx=new_ntt_ctx
    )

  #####################
  # FHE Kernel Functions
  #####################
  def _polynomial_mult_array(self) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Internal tensor-product kernel over the canonical payload."""
    a0 = self.polynomial[:, 0].astype(jnp.uint64)
    a1 = self.polynomial[:, 1].astype(jnp.uint64)
    b0 = self.polynomial[:, 2].astype(jnp.uint64)
    b1 = self.polynomial[:, 3].astype(jnp.uint64)

    mul0_t = a0 * b0
    mul0 = self.ntt_ctx.ff_ctx.modular_reduction(mul0_t).astype(
        self.modulus_dtype
    )

    mul2_t = a1 * b1
    mul2 = self.ntt_ctx.ff_ctx.modular_reduction(mul2_t).astype(
        self.modulus_dtype
    )

    t1_t = a0 * b1 + a1 * b0
    mul1 = self.ntt_ctx.ff_ctx.modular_reduction(t1_t)
    return (
        jnp.concatenate([mul0[:, None], mul1[:, None]], axis=1),
        mul2[:, None],
    )

  def _polynomial_square_array(self) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Internal square kernel using the (a*a, 2*a0*a1, a1*a1) shortcut.

    Equivalent to `_polynomial_mult_array` when the input shell is built by
    duplicating ct (so a0==b0 and a1==b1), but uses 2 modmuls + 1 doubling
    instead of 3 modmuls + 1 add.

      mul0 = a0 * a0
      mul1 = 2 * a0 * a1   (1 modmul + 1 << 1, then mod-reduce)
      mul2 = a1 * a1

    Layout matches `_polynomial_mult_array` exactly: returns (3-elem packed as
    (batch, 2, ...) for [mul0, mul1] and (batch, 1, ...) for [mul2]).
    """
    a0 = self.polynomial[:, 0].astype(jnp.uint64)
    a1 = self.polynomial[:, 1].astype(jnp.uint64)

    mul0 = self.ntt_ctx.ff_ctx.modular_reduction(a0 * a0).astype(
        self.modulus_dtype
    )
    mul2 = self.ntt_ctx.ff_ctx.modular_reduction(a1 * a1).astype(
        self.modulus_dtype
    )
    # 2*a0*a1: a0*a1 fits in uint64 (each operand < 2^32); shifting left by 1
    # at most doubles, still fits. Then a single modular reduction.
    cross = (a0 * a1) << jnp.uint64(1)
    mul1 = self.ntt_ctx.ff_ctx.modular_reduction(cross)

    return (
        jnp.concatenate([mul0[:, None], mul1[:, None]], axis=1),
        mul2[:, None],
    )

jax.tree_util.register_pytree_node_class(Polynomial)


__all__ = ['Polynomial']

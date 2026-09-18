import math
from typing import Any, List, Optional, Union

import jax
import jax.numpy as jnp

import util
import finite_field as ff_context
import ntt_mm as ntt


# Global registry for Polynomial pytree static state
_POLYNOMIAL_REGISTRY = {}
_POLYNOMIAL_NEXT_ID = [0]


class Polynomial:
  """Polynomial class for FHE operations."""

  def __init__(
      self, shapes: dict[str, Any], parameters: Optional[dict[str, Any]] = None
  ):
    """Initialize the Polynomial object.

    Each polynomial is a 4D tensor with shape (batch, num_elements, num_moduli,
    degree).

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
            context (e.g., ff_context.BarrettContext).
          - BAT_lazy: Boolean indicating whether to use BATLazyContext.
    """
    self.batch = shapes['batch']
    self.num_elements = shapes['num_elements']
    self.num_moduli = shapes['num_moduli']
    self.degree = shapes['degree']
    log_degree = int(math.log2(self.degree))
    self.precision = shapes['precision']
    if 'degree_layout' in shapes:
      self.degree_layout = shapes['degree_layout']
    else:
      self.degree_layout = (self.degree,)

    if len(self.degree_layout) == 2:
      self.r = self.degree_layout[0]
      self.c = self.degree_layout[1]
    else:
      self.r = 1 << (log_degree // 2)
      self.c = self.degree // self.r

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

    # NTT Parameters
    self.bit_reverse_indices = jnp.array(
        util.bit_reverse_indices(self.degree), jnp.uint32
    )
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

      if (
          parameters is not None
          and 'BAT_lazy' in parameters
          and parameters['BAT_lazy']
      ):
        self.ntt_ctx = ntt.NTTCiphertextBATLazyContext(
            moduli=self.moduli, parameters=ntt_params
        )
      else:
        if isinstance(finite_field_context, ff_context.BarrettContext):
          self.ntt_ctx = ntt.NTTCiphertextBarrettContext(
              moduli=self.moduli, parameters=ntt_params
          )
        elif isinstance(finite_field_context, ff_context.MontgomeryContext):
          self.ntt_ctx = ntt.NTTCiphertextMontgomeryContext(
              moduli=self.moduli, parameters=ntt_params
          )
        elif isinstance(finite_field_context, ff_context.ShoupContext):
          self.ntt_ctx = ntt.NTTCiphertextShoupContext(
              moduli=self.moduli, parameters=ntt_params
          )
        else:
          raise ValueError(
              'Unsupported finite field context type:'
              f' {type(finite_field_context)}'
          )

    self.moduli_array = jnp.array(self.moduli, dtype=self.modulus_dtype)
    self.polynomial = jnp.zeros(
        (self.batch, self.num_elements, self.degree, self.num_moduli),
        dtype=self.modulus_dtype,
    )
    self.extend_polynomial = jnp.zeros(
        (self.batch, self.num_elements, self.degree, 1),
        dtype=self.modulus_dtype,
    )

    self._pytree_id = _POLYNOMIAL_NEXT_ID[0]
    _POLYNOMIAL_NEXT_ID[0] += 1

  def tree_flatten(self):
    children = (self.polynomial, self.extend_polynomial)
    aux = {
        k: v
        for k, v in self.__dict__.items()
        if k not in ('polynomial', 'extend_polynomial')
    }
    _POLYNOMIAL_REGISTRY[self._pytree_id] = aux
    return children, self._pytree_id

  @classmethod
  def tree_unflatten(cls, pytree_id, children):
    obj = object.__new__(cls)
    obj.polynomial, obj.extend_polynomial = children
    aux = _POLYNOMIAL_REGISTRY[pytree_id]
    for k, v in aux.items():
      setattr(obj, k, v)
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
    self.polynomial = batch_polynomial

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
  def add(self, other: Union['Polynomial', jnp.ndarray]):
    other_array = other.polynomial if isinstance(other, Polynomial) else other
    self.polynomial = self.polynomial + other_array

  def sub(self, other: Union['Polynomial', jnp.ndarray]):
    other_array = other.polynomial if isinstance(other, Polynomial) else other
    self.polynomial = self.polynomial - other_array

  def mul(self, other: Union['Polynomial', jnp.ndarray]):
    other_array = other.polynomial if isinstance(other, Polynomial) else other
    self.polynomial = self.polynomial.astype(jnp.uint64) * other_array.astype(
        jnp.uint64
    )

  def modmul(self, other: Union['Polynomial', jnp.ndarray]):
    other_array = other.polynomial if isinstance(other, Polynomial) else other
    temp = self.polynomial.astype(jnp.uint64) * other_array.astype(jnp.uint64)
    reduced = self.ntt_ctx.ff_ctx.modular_reduction(temp)
    self.polynomial = reduced.astype(self.modulus_dtype)

  def mod_reduce(self):
    reduced = self.ntt_ctx.ff_ctx.modular_reduction(
        self.polynomial.astype(jnp.uint64)
    )
    self.polynomial = reduced.astype(self.modulus_dtype)

  #####################
  # Modulus Dropping Functions
  #####################
  def drop_last_modulus(self) -> jnp.ndarray:
    """Drops the last modulus from the polynomial."""
    if self.num_moduli <= 1:
      raise ValueError('Cannot drop modulus from a single-limb polynomial.')

    # Drop polynomial limb and track the new modulus set.
    self.moduli = self.moduli[:-1]
    self.moduli_array = self.moduli_array[:-1]
    self.num_moduli -= 1

    # Update finite field context and rebuild NTT context for reduced limb set.
    self.shape_in_ntt_all_limbs = (-1, self.r, self.c, self.num_moduli)
    self.shape_in_ntt_last_limb = (-1, self.r, self.c)
    self.ntt_ctx.drop_last_modulus()
    return self.polynomial

  #####################
  # FHE Kernel Functions
  #####################
  def polynomial_mult(self) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Multiplies two polynomials."""
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

  def polynomial_square(self) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Squares a 2-element ciphertext using the (a*a, 2*a0*a1, a1*a1) shortcut.

    Equivalent to `polynomial_mult` when the input shell is built by
    duplicating ct (so a0==b0 and a1==b1), but uses 2 modmuls + 1 doubling
    instead of 3 modmuls + 1 add.

      mul0 = a0 * a0
      mul1 = 2 * a0 * a1   (1 modmul + 1 << 1, then mod-reduce)
      mul2 = a1 * a1

    Layout matches polynomial_mult exactly: returns (3-elem packed as
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

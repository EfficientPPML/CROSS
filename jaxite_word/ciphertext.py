from typing import List, Union, Optional
import jax
import util
import jax.numpy as jnp
import math
import ntt_mm as ntt
import finite_field as ff_context
from bconv import BConvBarrett


# Global registry for Ciphertext pytree static state
_CIPHERTEXT_REGISTRY = {}
_CIPHERTEXT_NEXT_ID = [0]


class Ciphertext:
  def __init__(self, shapes: dict, parameters: Optional[dict] = None):
    """
    Initialize the Ciphertext object.
    Each ciphertext is a 4D tensor with shape (batch, num_elements, num_moduli, degree).

    Args:
        shapes (dict): A dictionary containing the shapes of the ciphertext.
            - batch: The batch size of the ciphertext.
            - num_elements: The number of elements in the ciphertext.
            - degree: The degree of the ciphertext.
            - num_moduli: The number of moduli in the ciphertext.
            - precision: The precision of the ciphertext.
        parameters (Optional[dict], optional): A dictionary containing the parameters of the ciphertext.
            - moduli: The moduli of the ciphertext.
                - If the moduli is a single integer, the ciphertext will be a single modulus.
                - If the moduli is a list of integers, the ciphertext will be a multi-modulus.
        finite_field_context (Optional[object], optional): The finite field context to use.
            - If not provided, a default BarrettContext will be created.
        r (Optional[int], optional): The r parameter for the NTT context.
        c (Optional[int], optional): The c parameter for the NTT context.
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
      self.degree_layout = (self.degree, )

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
      self.moduli = util.find_moduli_ntt(self.num_moduli, self.precision, 2 * self.degree)

    # NTT Parameters
    self.bit_reverse_indices = jnp.array(util.bit_reverse_indices(self.degree), jnp.uint32)
    self.shape_in_ntt_all_limbs = (-1, self.r, self.c, self.num_moduli)

    # Allow external NTT context injection (used by HEParameterCache for shared contexts)
    if parameters is not None and 'ntt_ctx' in parameters:
      self.ntt_ctx = parameters['ntt_ctx']
      if not hasattr(self.ntt_ctx, 'ff_ctx'):
          raise ValueError("Injected ntt_ctx must have 'ff_ctx' attribute")
    else:
      if parameters is not None and 'finite_field_context' in parameters:
        finite_field_context = parameters['finite_field_context'](moduli=self.moduli)
      else:
        finite_field_context = ff_context.BarrettContext(moduli=self.moduli)

      ntt_params = {
          "r": self.r,
          "c": self.c,
          "finite_field_context": finite_field_context,
      }

      if parameters is not None and "BAT_lazy" in parameters and parameters["BAT_lazy"]:
        self.ntt_ctx = ntt.NTTCiphertextBATLazyContext(moduli=self.moduli, parameters=ntt_params)
      else:
        if isinstance(finite_field_context, ff_context.BarrettContext):
          self.ntt_ctx = ntt.NTTCiphertextBarrettContext(moduli=self.moduli, parameters=ntt_params)
        elif isinstance(finite_field_context, ff_context.MontgomeryContext):
          self.ntt_ctx = ntt.NTTCiphertextMontgomeryContext(moduli=self.moduli, parameters=ntt_params)
        elif isinstance(finite_field_context, ff_context.ShoupContext):
          self.ntt_ctx = ntt.NTTCiphertextShoupContext(moduli=self.moduli, parameters=ntt_params)
        else:
            raise ValueError(f"Unsupported finite field context type: {type(finite_field_context)}")

    self.moduli_array = jnp.array(self.moduli, dtype=self.modulus_dtype)
    self.ciphertext = jnp.zeros((self.batch, self.num_elements, self.degree, self.num_moduli), dtype=self.modulus_dtype)
    self.extend_ciphertext = jnp.zeros((self.batch, self.num_elements, self.degree, 1), dtype=self.modulus_dtype)
    self.bconv = None
    self.bconv_indices_list = []

    self._pytree_id = _CIPHERTEXT_NEXT_ID[0]
    _CIPHERTEXT_NEXT_ID[0] += 1

  def _create_bconv(self, moduli):
      if self.bconv is None:
          self.bconv = BConvBarrett(moduli)
      return self.bconv

  def tree_flatten(self):
      children = (self.ciphertext, self.extend_ciphertext)
      aux = {k: v for k, v in self.__dict__.items()
             if k not in ('ciphertext', 'extend_ciphertext')}
      _CIPHERTEXT_REGISTRY[self._pytree_id] = aux
      return children, self._pytree_id

  @classmethod
  def tree_unflatten(cls, pytree_id, children):
      obj = object.__new__(cls)
      obj.ciphertext, obj.extend_ciphertext = children
      aux = _CIPHERTEXT_REGISTRY[pytree_id]
      for k, v in aux.items():
          setattr(obj, k, v)
      return obj

  def random_init(self):
    self.ciphertext = util.random_batched_ciphertext((self.batch, self.num_elements, *self.degree_layout, self.num_moduli), self.moduli, dtype=self.modulus_dtype)

  #####################
  # Getter Functions
  #####################
  @property
  def shape(self):
    return self.ciphertext.shape

  def get_batch_ciphertext(self) -> jnp.ndarray:
    return self.ciphertext

  def get_ciphertext(self, batch_index) -> jnp.ndarray:
    return self.ciphertext[batch_index]

  def get_element(self, element_index) -> jnp.ndarray:
    return self.ciphertext[:, element_index]

  def get_limb(self, limb_index) -> jnp.ndarray:
    return self.ciphertext[..., limb_index]

  #####################
  # Setter Functions
  # Note set_ciphertext, set_element, set_limb are in place operations, not recommended in JAX
  #####################
  def set_batch_ciphertext(self, batch_ciphertext: jnp.ndarray) -> jnp.ndarray:
    self.ciphertext = batch_ciphertext

  def set_ciphertext(self, batch_index: int, ciphertext: jnp.ndarray) -> jnp.ndarray:
    self.ciphertext = self.ciphertext.at[batch_index].set(ciphertext)

  def set_element(self, element_index: int, element: jnp.ndarray) -> jnp.ndarray:
    self.ciphertext = self.ciphertext.at[:, element_index].set(element)

  def set_limb(self, limb_index: int, limb: jnp.ndarray) -> jnp.ndarray:
    self.ciphertext = self.ciphertext.at[..., limb_index].set(limb)

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
    current_shape = self.ciphertext.shape
    reshaped_in = self.ciphertext.reshape(self.shape_in_ntt_all_limbs)
    ntt_result = self.ntt_ctx.ntt(reshaped_in)
    self.ciphertext = ntt_result.reshape(current_shape)

  def to_coeffs_form(self):
    current_shape = self.ciphertext.shape
    reshaped_in = self.ciphertext.reshape(self.shape_in_ntt_all_limbs)
    intt_result = self.ntt_ctx.intt(reshaped_in)
    self.ciphertext = intt_result.reshape(current_shape)

  def to_compute_format(self):
    self.ciphertext = self.ntt_ctx.to_computation_format(self.ciphertext)

  def to_original_format(self):
    self.ciphertext = self.ntt_ctx.to_original_format(self.ciphertext)

  #####################
  # Arithmetic Functions Entire Ciphertext
  #####################
  def add(self, other: Union['Ciphertext', jnp.ndarray]):
    other_array = other.ciphertext if isinstance(other, Ciphertext) else other
    self.ciphertext = self.ciphertext + other_array

  def sub(self, other: Union['Ciphertext', jnp.ndarray]):
    other_array = other.ciphertext if isinstance(other, Ciphertext) else other
    self.ciphertext = self.ciphertext - other_array

  def mul(self, other: Union['Ciphertext', jnp.ndarray]):
    other_array = other.ciphertext if isinstance(other, Ciphertext) else other
    self.ciphertext = (
        self.ciphertext.astype(jnp.uint64) * other_array.astype(jnp.uint64)
    )

  def modmul(self, other: Union['Ciphertext', jnp.ndarray]):
    other_array = other.ciphertext if isinstance(other, Ciphertext) else other
    temp = self.ciphertext.astype(jnp.uint64) * other_array.astype(jnp.uint64)
    reduced = self.ntt_ctx.ff_ctx.modular_reduction(temp)
    self.ciphertext = reduced.astype(self.modulus_dtype)

  def mod_reduce(self):
    reduced = self.ntt_ctx.ff_ctx.modular_reduction(self.ciphertext.astype(jnp.uint64))
    self.ciphertext = reduced.astype(self.modulus_dtype)


  # The legacy `drop_last_modulus`, `modulus_switch_control_gen`, `rescale`,
  # `ciphertext_mult`, `key_switch_control_gen`, and `key_switch` methods
  # were removed (they relied on `power_of_inv_psi_all` / `power_of_psi_all`
  # / a cyclic-NTT rescale path). Negacyclic NTT bakes psi/psi^{-1} into
  # the NTT parameter generation, so the explicit psi modmuls are no-ops.
  # The production pipeline (HEMul / HERescale / HERot / BSGS / matvec)
  # uses the new classes in `hemul.py`, `rescale.py`, `herot.py` instead.




jax.tree_util.register_pytree_node_class(Ciphertext)

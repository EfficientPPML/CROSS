# Adapted from Google Jaxite PR147 for CROSS, 2026-10-27. See README.md and LICENSE.
"""Private trusted-setup records; online ciphertexts use CROSS Polynomial."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class Plaintext:
  """A CKKS Plaintext."""

  data: jax.Array  # Shape (degree, num_moduli)
  moduli: jax.Array

  def __post_init__(self):
    if not isinstance(self.data, jax.Array):
      object.__setattr__(self, "data", jnp.asarray(self.data))
    if not isinstance(self.moduli, jax.Array):
      object.__setattr__(self, "moduli", jnp.asarray(self.moduli))


@dataclasses.dataclass
class Ciphertext:
  """A CKKS Ciphertext."""

  data: jax.Array  # Shape (num_elements, degree, num_moduli)
  moduli: jax.Array

  def __post_init__(self):
    if not isinstance(self.data, jax.Array):
      object.__setattr__(self, "data", jnp.asarray(self.data))
    if not isinstance(self.moduli, jax.Array):
      object.__setattr__(self, "moduli", jnp.asarray(self.moduli))


@dataclasses.dataclass(frozen=True)
class PublicKey:
  """A CKKS Public Key."""

  data: np.ndarray  # Shape (2, degree, num_moduli)
  moduli: np.ndarray


@dataclasses.dataclass(frozen=True)
class SecretKey:
  """A CKKS Secret Key."""

  data: np.ndarray  # Shape (degree, num_moduli)
  moduli: np.ndarray


@dataclasses.dataclass(frozen=True)
class EvaluationKeys:
  """CKKS Evaluation Keys."""

  a: jax.Array = dataclasses.field(default_factory=lambda: np.empty((0, 0, 0)))  # pyrefly: ignore[bad-assignment]
  b: jax.Array = dataclasses.field(default_factory=lambda: np.empty((0, 0, 0)))  # pyrefly: ignore[bad-assignment]
  moduli: jax.Array = dataclasses.field(default_factory=lambda: np.empty((0,)))  # pyrefly: ignore[bad-assignment]


@dataclasses.dataclass(frozen=True)
class HMuxRotKey:
  """A key used in a single HMuxRot step.

  Consists of two ciphertexts symmetrically encrypted under the destination key
  sk modulo PQ:
    - key0: encrypts P * beta * sk(X^{5^{-j}})
    - key1: encrypts P * beta
  """

  key0: Ciphertext
  key1: Ciphertext


@dataclasses.dataclass(frozen=True)
class MuxRotationKey:
  """A set of HMuxRot keys for all bits of a secret rotation index.

  Contains a list of pairs of keys (hmrkey_jk_0, hmrkey_not_jk_1) for each bit k
  from 0 to log2(num_slots) - 1.
  """

  keys: list[tuple[HMuxRotKey, HMuxRotKey]]
  permutations: list[jax.Array]

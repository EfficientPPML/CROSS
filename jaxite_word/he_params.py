"""CKKS security generation, validation, and runtime parameter storage.

``HEParameterCache`` computes NTT twiddle factors, finite-field reduction
contexts, and psi/inv_psi arrays once at max level. Per-level BConv parameters
(which cannot be sliced) are computed separately. The module also owns
versioned ring-security generation and the corresponding command-line entry
point.

The reduction backend is injectable: pass `finite_field_context` (a
FiniteFieldContextBase subclass, default BarrettContext) and the cache derives
the paired NTT-ciphertext and BConv implementations through the existing
backend hooks (ntt_mm.ntt_ciphertext_context_for / bconv.make_bconv). Key
material (eval/rotation keys) is stored in standard form; the low-level
operators convert it to their computation format exactly once at setup.
"""
import argparse
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import json
import math
import operator
from numbers import Integral
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp

import bconv
import finite_field
import ntt_mm
import util
import key_gen as kg
from composite_prime_gen import composite_prime_gen
from util import (
    compute_num_p_towers,
    generate_p_towers,
    is_prime_deterministic,
    q_partition_products,
    validate_barrett_bconv_moduli,
)

if __name__ == 'jaxite_word.he_params':
  sys.modules.setdefault('he_params', sys.modules[__name__])
elif __name__ == 'he_params':
  sys.modules.setdefault('jaxite_word.he_params', sys.modules[__name__])

BarrettContext = finite_field.BarrettContext
NTTCiphertextBarrettContext = ntt_mm.NTTCiphertextBarrettContext
BConvBarrett = bconv.BConvBarrett

jax.config.update('jax_enable_x64', True)


def normalize_noise_scale_degree(value) -> int:
  """Return a positive integral key/encryption noise multiplier."""
  if isinstance(value, bool) or not isinstance(value, Integral):
    raise ValueError('noise_scale_degree must be a positive int.')
  value = int(value)
  if value < 1:
    raise ValueError('noise_scale_degree must be a positive int.')
  return value


@dataclass
class LevelBConvParams:
  """BConv parameters for a specific level's moduli configuration.

  PInvModq is kept in STANDARD form (plain modular inverses); consumers
  multiplying it against computation-format data must encode it via the
  backend's to_computation_format first (the level operators compute and
  encode their own copy, see herot.py/hemul.py).
  """

  level: int
  num_q: int
  bconv: object  # BConv subclass selected by the backend's bconv_cls hook
  select_tower_index: List[jnp.ndarray]
  non_select_tower_index: List[jnp.ndarray]
  restore_indices: List[jnp.ndarray]
  PInvModq: jnp.ndarray


@dataclass
class LevelParams:
  """All parameters for a specific level."""

  num_q: int
  sliced_ntt_q: object  # NTT context (sliced from max-level)
  sliced_ff_q: object  # finite-field context (sliced from max-level)
  ntt_qp: object  # mod_reduce-only NTT context for Q+P
  ff_qp: object  # finite-field context (Q concat P)
  bconv_params: LevelBConvParams
  psi_q: jnp.ndarray
  inv_psi_q: jnp.ndarray
  psi_qp: jnp.ndarray
  inv_psi_qp: jnp.ndarray

  @property
  def sliced_barrett_q(self):
    """Backwards-compatible alias for sliced_ff_q (Barrett default backend)."""
    return self.sliced_ff_q


class HEParameterCache:

  def __init__(
      self,
      q_towers,
      p_towers,
      r,
      c,
      dnum,
      composite_degree,
      batch=1,
      perf_test=False,
      rotation_key_noise_std=kg.sigma,
      rotation_key_noise_scale=1,
      finite_field_context=None,
      cache_rotation_keys=True,
      noise_scale_degree=None,
  ):
    self.q_towers: List[int] = list(q_towers)
    self.p_towers: List[int] = list(p_towers)
    self.r: int = r
    self.c: int = c
    self.dnum: int = dnum
    self.composite_degree: int = composite_degree
    self.batch: int = batch
    self.degree: int = r * c
    self.degree_layout: Tuple[int, int] = (r, c)
    self.num_q: int = len(q_towers)
    self.num_p: int = len(p_towers)
    self.max_level: int = (self.num_q - 1) // composite_degree
    self.perf_test: bool = perf_test
    self.noise_std = kg.normalize_noise_std(rotation_key_noise_std)
    legacy_noise_scale = normalize_noise_scale_degree(
        rotation_key_noise_scale
    )
    if noise_scale_degree is None:
      noise_scale_degree = legacy_noise_scale
    else:
      noise_scale_degree = normalize_noise_scale_degree(noise_scale_degree)
      if legacy_noise_scale != 1 and legacy_noise_scale != noise_scale_degree:
        raise ValueError(
            'rotation_key_noise_scale and noise_scale_degree disagree.'
        )
    self.noise_scale_degree = noise_scale_degree
    # Preserve main's attribute names for existing low-level callers.
    self.rotation_key_noise_std = self.noise_std
    self.rotation_key_noise_scale = self.noise_scale_degree
    self.key_generation_version: int = int(kg.KEY_GENERATION_VERSION)
    self.cache_rotation_keys: bool = bool(cache_rotation_keys)
    self.level_params: Dict[int, LevelParams] = {}

    # Reduction backend (a FiniteFieldContextBase subclass). The paired NTT
    # and BConv classes are derived through the backend hooks so every
    # context the cache hands out shares one reduction algorithm.
    self.ff_context_cls = finite_field_context or BarrettContext
    self._ntt_ctx_cls = ntt_mm.ntt_ciphertext_context_for(self.ff_context_cls)

    # Attributes populated by initialize() — placeholders use correct types
    # with minimal storage (single-modulus contexts, single-element arrays).
    # The ff-context instance must be passed explicitly in the parameters
    # dict: the NTT base defaults a missing entry to BarrettContext.
    _init_q_ff = self.ff_context_cls(moduli=[q_towers[0]])
    _init_q_ntt_params = {'r': r, 'c': c, 'finite_field_context': _init_q_ff}
    self.ff_q_max = _init_q_ff
    self.ntt_q_max = self._ntt_ctx_cls(
        moduli=[q_towers[0]], parameters=_init_q_ntt_params)
    _init_p_ff = self.ff_context_cls(moduli=[p_towers[0]])
    _init_p_ntt_params = {'r': r, 'c': c, 'finite_field_context': _init_p_ff}
    self.ff_p = _init_p_ff
    self.ntt_p = self._ntt_ctx_cls(
        moduli=[p_towers[0]], parameters=_init_p_ntt_params)
    self.psi_full: jnp.ndarray = jnp.zeros(1, dtype=jnp.uint64)
    self.inv_psi_full: jnp.ndarray = jnp.zeros(1, dtype=jnp.uint64)
    self.eval_key_a_full: jnp.ndarray = jnp.zeros(1, dtype=jnp.uint32)
    self.eval_key_b_full: jnp.ndarray = jnp.zeros(1, dtype=jnp.uint32)
    self.secret_key: list = []
    self.rot_indices: List[int] = []
    self.coef_maps: Dict[int, jnp.ndarray] = {}
    self.raw_rotation_keys: dict = {}
    self._formatted_rotation_keys: dict = {}
    self.per_level_rotation_keys: bool = False

  # Backwards-compatible aliases for the pre-parameterization attribute
  # names (the contexts are Barrett only under the default backend).
  @property
  def barrett_q_max(self):
    return self.ff_q_max

  @property
  def barrett_p(self):
    return self.ff_p

  @staticmethod
  def _validated_integral(value, name: str) -> int:
    """Return ``value`` as an int without accepting lossy coercions."""
    if isinstance(value, bool) or not isinstance(value, Integral):
      raise TypeError(f'{name} must be an int.')
    return int(value)

  @classmethod
  def _validated_index(cls, value, name: str, size: int) -> int:
    """Return an index in ``[0, size)`` or fail before array access."""
    value = cls._validated_integral(value, name)
    if value < 0 or value >= size:
      raise ValueError(f'{name} {value} outside [0, {size - 1}].')
    return value

  def _validated_level(self, level) -> int:
    """Return a configured logical level."""
    return self._validated_index(
        level, 'level', self.max_level + 1
    )

  def _validated_num_q(self, num_q) -> int:
    """Return a positive Q-limb count bounded by this cache's chain."""
    num_q = self._validated_integral(num_q, 'num_q')
    if num_q < 1 or num_q > self.num_q:
      raise ValueError(f'num_q {num_q} outside [1, {self.num_q}].')
    return num_q

  def num_q_at_level(self, level: int) -> int:
    level = self._validated_level(level)
    num_q = self.num_q - (
        self.max_level - level
    ) * self.composite_degree
    return self._validated_num_q(num_q)

  def q_moduli_at_level(self, level: int) -> List[int]:
    return self.q_towers[: self.num_q_at_level(level)]

  def level_for_num_q(self, num_q: int) -> int:
    """Return the CROSS logical level represented by ``num_q`` Q limbs."""
    num_q = self._validated_num_q(num_q)
    removed = self.num_q - num_q
    if removed % self.composite_degree:
      raise ValueError(
          f'num_q={num_q} is not a valid CD{self.composite_degree} level '
          f'for a {self.num_q}-limb Q chain'
      )
    level = self.max_level - removed // self.composite_degree
    if level < 0 or level > self.max_level:
      raise ValueError(
          f'num_q={num_q} maps outside level range [0, {self.max_level}]'
      )
    if self.num_q_at_level(level) != num_q:
      raise ValueError(
          f'num_q={num_q} is not represented by the configured level layout'
      )
    return level

  def _prepare_plaintext_payload(self, plaintext_data, level: int):
    """Return a canonical plaintext payload in this level's compute format.

    Encoders produce standard residues so their output remains valid for
    encryption and decoding. Evaluator kernels instead consume the injected
    backend's computation format. This private, non-mutating boundary must be
    applied exactly once before a freshly encoded plaintext reaches a raw
    evaluator kernel (identity for Barrett; ``x * R mod q`` for Montgomery).
    """
    level = self._validated_level(level)
    if level not in self.level_params:
      raise RuntimeError(
          'HEParameterCache must be initialized before preparing plaintext.'
      )
    finite_field.check_rank5_array(
        plaintext_data,
        'HEParameterCache._prepare_plaintext_payload',
        batch=1,
        num_elements=1,
        degree_layout=self.degree_layout,
        num_moduli=self.num_q_at_level(level),
    )
    # Lazy cache: index through the accessor so an uncached or evicted
    # level is built rather than raising KeyError.
    ff_ctx = self.get_sliced_ff_q(level)
    return jnp.asarray(
        ff_ctx.to_computation_format(
            jnp.asarray(plaintext_data, dtype=jnp.uint64)
        ),
        dtype=jnp.uint32,
    )

  def _precompute_scaling_factors(self):
    """Pre-compute OpenFHE's m_scalingFactorsReal recursively.

    Exact port of ckksrns-cryptoparameters.cpp lines 89-171.
    For COMPOSITESCALINGAUTO with cd=2:
      sf[0] = q[sizeQ-1] * q[sizeQ-2]  (product of LAST cd primes)
      sf[k] = sf[k-cd]^2 / product(q[sizeQ-k+j] for j in 0..cd-1)
              for k % cd == 0
      sf[k] = 1 for k % cd != 0

    This recursive formula tracks the exact scale evolution through
    multiply-rescale chains, accounting for the actual prime values.
    """
    sf = self._recursive_scaling_factors(
        self.q_towers, self.composite_degree
    )
    self._scaling_factors_real = sf

    # Also compute "Big" scaling factors (S² = sf[k]^2)
    size_q = len(sf)
    sfBig = [0.0] * max(size_q - 1, 1)
    sfBig[0] = sf[0] * sf[0]
    for k in range(1, size_q - 1):
      sfBig[k] = sf[k] * sf[k]
    self._scaling_factors_real_big = sfBig

    # Store individual moduli as float for GetModReduceFactor
    self._dmoduliQ = [float(qi) for qi in self.q_towers]

  @staticmethod
  def _recursive_scale_inputs(q_towers, composite_degree):
    """Validate inputs shared by pre-init recursive-scale computations."""
    if isinstance(q_towers, (str, bytes)):
      raise TypeError('q_towers must be an iterable of positive ints.')
    try:
      q_towers = tuple(q_towers)
    except TypeError as error:
      raise TypeError(
          'q_towers must be an iterable of positive ints.'
      ) from error
    if not q_towers:
      raise ValueError('q_towers must be non-empty.')
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        for value in q_towers
    ):
      raise TypeError('q_towers must contain only positive ints.')
    if any(int(value) <= 0 for value in q_towers):
      raise ValueError('q_towers must contain only positive ints.')
    if (
        isinstance(composite_degree, bool)
        or not isinstance(composite_degree, Integral)
    ):
      raise TypeError('composite_degree must be an int.')
    q_towers = tuple(int(value) for value in q_towers)
    composite_degree = int(composite_degree)
    if composite_degree < 1 or composite_degree >= len(q_towers):
      raise ValueError(
          'composite_degree must be positive and smaller than the Q chain.'
      )
    return q_towers, composite_degree

  @classmethod
  def _recursive_scaling_factors(cls, q_towers, composite_degree):
    """Compute OpenFHE's recursive scale table without initialized HE state."""
    q_towers, composite_degree = cls._recursive_scale_inputs(
        q_towers, composite_degree
    )
    size_q = len(q_towers)
    factors = [0.0] * size_q
    factors[0] = float(math.prod(q_towers[-composite_degree:]))
    for index in range(1, size_q):
      if index % composite_degree:
        factors[index] = 1.0
        continue
      factors[index] = factors[index - composite_degree] ** 2
      for offset in range(composite_degree):
        factors[index] /= float(q_towers[size_q - index + offset])
    return factors

  @classmethod
  def recursive_scaling_factor_for(
      cls, q_towers, composite_degree: int, level: int
  ) -> float:
    """Return the recursive scale without requiring cache initialization."""
    q_towers, composite_degree = cls._recursive_scale_inputs(
        q_towers, composite_degree
    )
    maximum_level = (len(q_towers) - 1) // composite_degree
    level = cls._validated_index(
        level, 'level', maximum_level + 1
    )
    factors = cls._recursive_scaling_factors(
        q_towers, composite_degree
    )
    num_q = (
        len(q_towers)
        - (maximum_level - level) * composite_degree
    )
    return factors[len(q_towers) - num_q]

  def scaling_factor_at_level(self, level: int) -> float:
    """Get the scaling factor at a given level.

    For COMPOSITESCALINGAUTO: always uses the DIRECT product of the
    top cd primes at the given level.  This is numerically exact and
    matches the rescale divisor, avoiding accumulated floating-point
    drift from the recursive sf[k] = sf[k-cd]^2 / (product of cd primes)
    formula, which diverges for deep chains with small (28-bit) primes.

    The recursive formula is still pre-computed for reference but is
    NOT used as the primary return value because:
    - With 28-bit primes and 40+ Q-towers, 20 levels of squaring and
      dividing produces O(1) accumulated relative error.
    - The direct product always matches the rescale divisor exactly,
      ensuring zero encode-rescale scale mismatch.
    """
    level = self._validated_level(level)
    if not hasattr(self, '_scaling_factors_real'):
      self._precompute_scaling_factors()

    cd = self.composite_degree
    size_q = len(self.q_towers)
    nq = self.num_q_at_level(level)

    # Direct product of the top cd primes at this level.
    # At level L, the ciphertext has nq primes: q[0] .. q[nq-1].
    # A rescale at this level drops q[nq-cd] .. q[nq-1], so the
    # scaling factor = product(q[nq-cd .. nq-1]).
    #
    # Special case: level 0. OpenFHE's sf[0] = product of the LAST
    # cd primes of the FULL chain (ckksrns-cryptoparameters.cpp line 107).
    # This is the base encoding scale. For levels > 0, the top-cd-primes
    # formula gives the rescale divisor (which IS the encoding scale at
    # that level in FLEXIBLEAUTO).
    if level == 0:
      # OpenFHE convention: sf[0] = product of LAST cd primes of full chain
      product = 1.0
      for i in range(cd):
        product *= float(self.q_towers[size_q - cd + i])
      return product
    if nq < cd:
      product = 1.0
      for i in range(nq):
        product *= float(self.q_towers[i])
      return product
    product = 1.0
    for i in range(cd):
      product *= float(self.q_towers[nq - cd + i])
    return product

  def scaling_factor_recursive(self, level: int) -> float:
    """Get the RECURSIVE scaling factor at a given level.

    Unlike scaling_factor_at_level (which uses the direct product of top
    cd primes), this returns the value from the recursive chain:
      sf[0] = product of last cd primes
      sf[k] = sf[k-cd]² / product(q[sizeQ-k+j] for j in 0..cd-1)

    The recursive SF tracks the ACTUAL scale of a ciphertext that has gone
    through k levels of multiply-then-rescale operations. It diverges
    exponentially from the direct product for non-uniform primes.

    Matches OpenFHE's GetScalingFactorReal(level).
    """
    level = self._validated_level(level)
    if not hasattr(self, '_scaling_factors_real'):
      self._precompute_scaling_factors()
    factor_index = self._validated_index(
        len(self.q_towers) - self.num_q_at_level(level),
        'scaling-factor index',
        len(self._scaling_factors_real),
    )
    return self._scaling_factors_real[factor_index]

  def scaling_factor_real_big(self, level: int) -> float:
    """Get the S² scaling factor (sf[k]^2) at a given level."""
    level = self._validated_level(level)
    if not hasattr(self, '_scaling_factors_real_big'):
      self._precompute_scaling_factors()

    factor_index = self._validated_index(
        len(self.q_towers) - self.num_q_at_level(level),
        'big scaling-factor index',
        len(self._scaling_factors_real_big),
    )
    return self._scaling_factors_real_big[factor_index]

  def mod_reduce_factor(self, prime_index: int) -> float:
    """Get the modulus at a given index (for rescale scale tracking).

    Matches OpenFHE GetModReduceFactor(l) = m_dmoduliQ[l].
    """
    prime_index = self._validated_index(
        prime_index, 'prime_index', len(self.q_towers)
    )
    if not hasattr(self, '_dmoduliQ'):
      self._precompute_scaling_factors()
    return self._dmoduliQ[prime_index]

  def initialize(
      self,
      eval_key_a,
      eval_key_b,
      rot_keys=None,
      coef_maps=None,
      secret_key=None,
  ):
    """Master offline initialization.

    Args:
        eval_key_a: Evaluation key part a, shape (dnum, degree, num_Q + num_P)
        eval_key_b: Evaluation key part b, same shape
        rot_keys: Max-level rotation keys keyed by rotation index.
        coef_maps: Dict[rot_index -> permutation array]
        secret_key: Secret key for per-level rotation key generation
    """
    ring_dim = self.degree
    all_q_moduli = self.q_towers

    # ================================================================
    # 1. Max-level finite-field and NTT context for Q moduli (sliceable)
    # ================================================================
    self.ff_q_max = self.ff_context_cls(moduli=all_q_moduli)
    ntt_params_q = {
        'r': self.r,
        'c': self.c,
        'finite_field_context': self.ff_q_max,
    }
    self.ntt_q_max = self._ntt_ctx_cls(
        moduli=all_q_moduli, parameters=ntt_params_q, perf_test=self.perf_test
    )

    # P-only finite-field context (fixed across all levels, used in .concat())
    self.ff_p = self.ff_context_cls(moduli=self.p_towers)

    # P-only NTT context (for approx-mod-down INTT on P-part)
    ntt_params_p = {
        'r': self.r,
        'c': self.c,
        'finite_field_context': self.ff_p,
    }
    self.ntt_p = self._ntt_ctx_cls(
        moduli=self.p_towers, parameters=ntt_params_p, perf_test=self.perf_test
    )

    # ================================================================
    # 2. Psi / inv_psi for all Q+P moduli
    # ================================================================
    all_moduli = all_q_moduli + self.p_towers
    if not self.perf_test:
      all_psi_roots = [util.root_of_unity(2 * ring_dim, q) for q in all_moduli]
      self.psi_full = jnp.array(
          [
              [
                  pow(all_psi_roots[idx], i, all_moduli[idx])
                  for i in range(ring_dim)
              ]
              for idx in range(len(all_moduli))
          ],
          jnp.uint64,
      ).T.reshape(*self.degree_layout, len(all_moduli))

      all_inv_psi_roots = [
          pow(psi, -1, q)
          for psi, q in zip(all_psi_roots, all_moduli, strict=True)
      ]
      self.inv_psi_full = jnp.array(
          [
              [
                  pow(all_inv_psi_roots[idx], i, all_moduli[idx])
                  for i in range(ring_dim)
              ]
              for idx in range(len(all_moduli))
          ],
          jnp.uint64,
      ).T.reshape(*self.degree_layout, len(all_moduli))
    else:
      self.psi_full = util.random_parameters(
          (*self.degree_layout, len(all_moduli)), all_moduli, dtype=jnp.uint64
      )
      self.inv_psi_full = util.random_parameters(
          (*self.degree_layout, len(all_moduli)), all_moduli, dtype=jnp.uint64
      )

    # ================================================================
    # 3. Store eval keys (full shape)
    # ================================================================
    self.eval_key_a_full = jnp.asarray(eval_key_a, dtype=jnp.uint32)
    self.eval_key_b_full = jnp.asarray(eval_key_b, dtype=jnp.uint32)

    # ================================================================
    # 4. Store rotation key material
    # ================================================================
    # Rotation keys use max-level partition boundaries, matching evaluation
    # keys. Lower levels slice Q+P moduli while preserving those boundaries.
    self.secret_key = secret_key
    self.raw_rotation_keys = rot_keys if rot_keys is not None else {}
    self._formatted_rotation_keys = {}
    # Large bootstraps can opt out of retaining every max-level key. In this
    # mode rotation operators generate a correctly partitioned key for their
    # active level and own it for the operator's lifetime.
    self.per_level_rotation_keys = (
        not self.cache_rotation_keys
        or os.environ.get("CROSS_SKIP_TOPLEVEL_ROTKEYS") == "1"
    )
    self.rot_indices = list(coef_maps.keys()) if coef_maps else []
    self.coef_maps = {}
    if coef_maps:
      for rot_idx in coef_maps:
        self.coef_maps[rot_idx] = jnp.asarray(
            coef_maps[rot_idx], dtype=jnp.int32
        )

    # ================================================================
    # 5. Per-level parameter generation
    # ================================================================
    # Lazy by default: the per-level sliced NTT tables are NOT zero-copy
    # (JAX slicing materializes) — eagerly building all levels holds ~32 GB
    # of (C,4,C,4,M) BAT slices at N=65536.  A small LRU keeps only the
    # levels in active use; rebuilds are deterministic re-slices (memcpy).
    # CROSS_EAGER_LEVEL_PARAMS=1 restores the old eager behavior.
    self._lazy_level_params = (
        os.environ.get("CROSS_EAGER_LEVEL_PARAMS") != "1"
    )
    self._level_params_max = max(
        1, int(os.environ.get("CROSS_LEVEL_PARAMS_MAX", "4"))
    )
    if not self._lazy_level_params:
      for level in range(self.max_level + 1):
        self.level_params[level] = self._build_level_params(level)

  def _level(self, level: int) -> LevelParams:
    """Return (building lazily if needed) the LevelParams for a level."""
    if level < 0 or level > self.max_level:
      raise ValueError(
          f"level must be in [0, {self.max_level}], got {level}."
      )
    lp = self.level_params.get(level)
    if lp is None:
      lp = self._build_level_params(level)
      self.level_params[level] = lp
      if self._lazy_level_params and len(self.level_params) > self._level_params_max:
        # Evict least-recently-inserted other levels (dict preserves order;
        # re-inserting on rebuild refreshes recency).  Live operator
        # instances keep their arrays alive until stage eviction.
        for old in list(self.level_params.keys()):
          if old != level and len(self.level_params) > self._level_params_max:
            del self.level_params[old]
    elif self._lazy_level_params:
      # Refresh recency on hits; regular dicts preserve insertion order.
      del self.level_params[level]
      self.level_params[level] = lp
    return lp

  def _build_level_params(self, level: int) -> LevelParams:
    """Build all parameters for a specific level."""
    num_q_l = self.num_q_at_level(level)
    q_at_level = self.q_towers[:num_q_l]
    qp_at_level = q_at_level + self.p_towers

    # A full-range context already exists. Lower-level NTT slices are deferred
    # until a Polynomial wrapper actually needs one; BConv-only consumers then
    # avoid materializing the large BAT arrays entirely.
    sliced_ff_q = self.ff_q_max.slice(num_q_l)
    if level == self.max_level:
      sliced_ntt_q = self.ntt_q_max
    elif self._lazy_level_params:
      sliced_ntt_q = None
    else:
      sliced_ntt_q = self.ntt_q_max.slice(num_q_l, sliced_ff_q)

    # Q+P context via concatenation: sliced-Q + fixed-P contexts of the same
    # reduction backend. No NTT twiddle factors needed — Q+P is only used for
    # mod_reduce. FiniteFieldContextBase.ff_ctx returns self, so the bare
    # context can serve directly as ntt_ctx.
    ff_qp = sliced_ff_q.concat(self.ff_p)

    # Psi slices
    psi_q = self.psi_full[..., :num_q_l]
    inv_psi_q = self.inv_psi_full[..., :num_q_l]
    psi_qp = jnp.concatenate(
        [self.psi_full[..., :num_q_l], self.psi_full[..., self.num_q :]],
        axis=-1,
    )
    inv_psi_qp = jnp.concatenate(
        [
            self.inv_psi_full[..., :num_q_l],
            self.inv_psi_full[..., self.num_q :],
        ],
        axis=-1,
    )

    # BConv params
    bconv_params = self._build_bconv_params(
        level, num_q_l, q_at_level, qp_at_level
    )

    return LevelParams(
        num_q=num_q_l,
        sliced_ntt_q=sliced_ntt_q,
        sliced_ff_q=sliced_ff_q,
        ntt_qp=ff_qp,  # bare ff context serves as ntt_ctx via .ff_ctx property
        ff_qp=ff_qp,
        bconv_params=bconv_params,
        psi_q=psi_q,
        inv_psi_q=inv_psi_q,
        psi_qp=psi_qp,
        inv_psi_qp=inv_psi_qp,
    )

  def _build_bconv_params(self, level, num_q_l, q_at_level, qp_at_level):
    """Build BConv parameters for key-switch decomposition + approx-mod-down."""
    sizeQl = num_q_l
    sizeP = self.num_p
    alpha = (sizeQl + self.dnum - 1) // self.dnum
    numPartQl = (sizeQl + alpha - 1) // alpha

    # Partition indices for key-switch decomposition
    original_moduli_extract_index = []
    for i in range(sizeQl):
      if i % alpha == 0:
        original_moduli_extract_index.append([i])
      else:
        original_moduli_extract_index[-1].append(i)

    select_tower_index = []
    non_select_tower_index = []
    restore_indices = []
    control_indices_list = []

    # Key-switch decomposition configs (dnum partitions)
    for part in range(numPartQl):
      sel_index = original_moduli_extract_index[part]
      non_sel_index = [i for i in range(len(qp_at_level)) if i not in sel_index]
      select_tower_index.append(jnp.array(sel_index, jnp.uint16))
      non_select_tower_index.append(jnp.array(non_sel_index, jnp.uint16))
      control_indices_list.append((sel_index, non_sel_index))

      concat_order = sel_index + non_sel_index
      restore_index = [0] * len(concat_order)
      for pos, val in enumerate(concat_order):
        restore_index[val] = pos
      restore_indices.append(jnp.array(restore_index, dtype=jnp.uint16))

    # Approx-mod-down config: P -> Q
    extend_indices = list(range(sizeQl, sizeQl + sizeP))
    rotate_indices = list(range(sizeQl))
    control_indices_list.append((extend_indices, rotate_indices))

    # Create BConv through the backend hook and generate controls
    bconv_obj = bconv.make_bconv(self.ff_context_cls, qp_at_level)
    bconv_obj.control_gen(control_indices_list, perf_test=self.perf_test)

    # P^{-1} mod q_i for each Q modulus at this level
    P = 1
    for p in self.p_towers:
      P *= p
    PInvModq = jnp.asarray(
        [util.modinv(P, q) for q in q_at_level], dtype=jnp.uint32
    ).reshape(num_q_l)

    return LevelBConvParams(
        level=level,
        num_q=num_q_l,
        bconv=bconv_obj,
        select_tower_index=select_tower_index,
        non_select_tower_index=non_select_tower_index,
        restore_indices=restore_indices,
        PInvModq=PInvModq,
    )

  # ================================================================
  # Getter Methods
  # ================================================================
  def get_psi_q(self, level: int) -> jnp.ndarray:
    return self._level(level).psi_q

  def get_inv_psi_q(self, level: int) -> jnp.ndarray:
    return self._level(level).inv_psi_q

  def get_psi_qp(self, level: int) -> jnp.ndarray:
    return self._level(level).psi_qp

  def get_inv_psi_qp(self, level: int) -> jnp.ndarray:
    return self._level(level).inv_psi_qp

  def get_sliced_ff_q(self, level: int):
    return self._level(level).sliced_ff_q

  def get_eval_key(self, level: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n = self.num_q_at_level(level)
    a = jnp.concatenate(
        [
            self.eval_key_a_full[..., :n],
            self.eval_key_a_full[..., -self.num_p :],
        ],
        axis=-1,
    )
    b = jnp.concatenate(
        [
            self.eval_key_b_full[..., :n],
            self.eval_key_b_full[..., -self.num_p :],
        ],
        axis=-1,
    )
    return a, b

  def get_rot_key(self, rot_index: int, level: int):
    """Return reusable rotation key arrays for the given level.

    Keys are generated once at max level. Lower levels retain the max-level
    decomposition partition boundaries and slice only the Q+P modulus axis.

    Returns (a, b, coefMap) where a and b have shape
    (effective_dnum, *degree_layout, num_Q_at_level + num_P).
    """

    n = self.num_q_at_level(level)
    if self.per_level_rotation_keys:
      if self.secret_key is None:
        raise ValueError(
            f'rotation key {rot_index} was not supplied and '
            'secret_key is unavailable for generation'
        )
      generated = kg.gen_rotation_key(
          self.secret_key[:n],
          self.q_towers[:n],
          self.p_towers,
          rot_index,
          dnum=self.dnum,
          noise_std=self.rotation_key_noise_std,
          noise_scale=self.rotation_key_noise_scale,
      )[rot_index]
      a = (
          jnp.asarray(generated['a'], dtype=jnp.uint32)
          .transpose(0, 2, 1)
          .reshape(-1, *self.degree_layout, n + self.num_p)
      )
      b = (
          jnp.asarray(generated['b'], dtype=jnp.uint32)
          .transpose(0, 2, 1)
          .reshape(-1, *self.degree_layout, n + self.num_p)
      )
      return a, b, self.coef_maps[rot_index]

    raw_key = self.raw_rotation_keys.get(rot_index)
    if raw_key is None:
      if self.secret_key is None:
        raise ValueError(
            f'rotation key {rot_index} was not supplied and '
            'secret_key is unavailable for generation'
        )
      generated = kg.gen_rotation_key(
          self.secret_key,
          self.q_towers,
          self.p_towers,
          rot_index,
          dnum=self.dnum,
          noise_std=self.noise_std,
          noise_scale=self.noise_scale_degree,
      )
      raw_key = generated[rot_index]
      if self.cache_rotation_keys:
        self.raw_rotation_keys[rot_index] = raw_key

    # Keys are held as uint32 (every residue < 2^32); herot widens to uint64
    # transiently at the key-switch multiply. Halves the resident rotation-key
    # working set and lets degree-32768 (128-bit) keys fit in host RAM.
    formatted_key = self._formatted_rotation_keys.get(rot_index)
    if formatted_key is None:
      formatted_key = (
          jnp.array(raw_key['a'], jnp.uint32).transpose(0, 2, 1),
          jnp.array(raw_key['b'], jnp.uint32).transpose(0, 2, 1),
      )
      if self.cache_rotation_keys:
        self._formatted_rotation_keys[rot_index] = formatted_key
    a_full, b_full = formatted_key
    a = jnp.concatenate(
        [a_full[..., :n], a_full[..., -self.num_p :]], axis=-1
    ).reshape(-1, *self.degree_layout, n + self.num_p)
    b = jnp.concatenate(
        [b_full[..., :n], b_full[..., -self.num_p :]], axis=-1
    ).reshape(-1, *self.degree_layout, n + self.num_p)
    return a, b, self.coef_maps[rot_index]

  def get_bconv_params(self, level: int) -> LevelBConvParams:
    return self._level(level).bconv_params

  def get_sliced_ntt_q(self, level: int):
    params = self._level(level)
    if params.sliced_ntt_q is None:
      params.sliced_ntt_q = self.ntt_q_max.slice(
          params.num_q, params.sliced_ff_q
      )
    return params.sliced_ntt_q


####################################
# Security parameter generation
####################################

SCHEMA_VERSION = 2
MAX_MODULUS_BITS = 31
MAX_KERNEL_MODULUS_BITS = 31
DEFAULT_SIGMA = 3.190000057220458984375
SECURITY_TABLE_SOURCE = "OpenFHE 1.5.1 stdlatticeparms.cpp, HEStd_ternary entries"
SECURITY_TABLE_URL = (
    "https://github.com/openfheorg/openfhe-development/blob/"
    "v1.5.1/src/core/lib/lattice/stdlatticeparms.cpp"
)
OPENFHE_REFERENCE_COMMIT = "1306d14f8c26bb6150d3e6ad54f28dfe1007689e"
OPENFHE_PARAMETER_GENERATOR_URL = (
    "https://github.com/openfheorg/openfhe-development/blob/"
    f"{OPENFHE_REFERENCE_COMMIT}/src/pke/lib/scheme/ckksrns/"
    "ckksrns-parametergeneration.cpp"
)
SECRET_DISTRIBUTION = "uniform_ternary"


class ParameterGenerationError(ValueError):
    """Raised when no supported, valid parameter set can be generated."""


def _integer(value: object, name: str) -> int:
    """Return an integer value without accepting lossy numeric coercions."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _moduli(
    values: Sequence[int] | Iterable[int],
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    try:
        result = tuple(
            _integer(value, f"{name}[{index}]") for index, value in enumerate(values)
        )
    except TypeError as exc:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{name} must be an integer sequence") from exc
        raise
    if not result and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    if any(value < 2 for value in result):
        raise ValueError(f"{name} values must be at least 2")
    return result


class CostModel(str, Enum):
    """Attack-cost model used by the OpenFHE security table."""

    CLASSICAL = "classical"
    QUANTUM = "quantum"


class InferenceProfile(str, Enum):
    """Purpose selected by a demo inference entry point."""

    RAW_REFERENCE = "raw_reference"
    ENCRYPTED_CORRECTNESS = "encrypted_correctness"
    ENCRYPTED_SECURE = "encrypted_secure"


class MappingExecutionPolicy(str, Enum):
    """Resource policy for Mapping; this does not assert ring security."""

    WHOLE = "whole"
    MEMORY_BOUNDED = "memory_bounded"


def inference_execution_policy(
    require_128bit: bool,
    requested: MappingExecutionPolicy | str | None = None,
) -> MappingExecutionPolicy:
    """Resolve a resource policy independently from ring scale semantics."""
    if not isinstance(require_128bit, bool):
        raise TypeError("require_128bit must be a bool")
    if requested is None:
        # The registered secure profiles are intentionally run through the
        # bounded-memory implementation. This is a deployment choice, not
        # evidence of their cryptographic strength.
        if require_128bit:
            return MappingExecutionPolicy.MEMORY_BOUNDED
        return MappingExecutionPolicy.WHOLE
    try:
        return (
            requested
            if isinstance(requested, MappingExecutionPolicy)
            else MappingExecutionPolicy(requested)
        )
    except ValueError as exc:
        choices = ", ".join(policy.value for policy in MappingExecutionPolicy)
        raise ValueError(
            f"execution policy must be one of: {choices}"
        ) from exc


def _cost_model(value: CostModel | str) -> CostModel:
    try:
        return value if isinstance(value, CostModel) else CostModel(value)
    except ValueError as exc:
        choices = ", ".join(model.value for model in CostModel)
        raise ValueError(f"cost_model must be one of: {choices}") from exc


# Exact OpenFHE HEStd_ternary ceilings keyed by ring dimension N. In OpenFHE's
# CKKS parameter generator, N = cyclOrder / 2 and cyclOrder is then set to 2*N.
# These include every ring dimension published in the pinned source; no value
# is linearly extrapolated.
_OPENFHE_CEILINGS_RAW = {
    (CostModel.CLASSICAL, 128): {
        1024: 27,
        2048: 54,
        4096: 109,
        8192: 218,
        16384: 438,
        32768: 881,
        65536: 1747,
        131072: 3523,
    },
    (CostModel.CLASSICAL, 192): {
        1024: 19,
        2048: 37,
        4096: 75,
        8192: 152,
        16384: 305,
        32768: 611,
        65536: 1199,
        131072: 2411,
    },
    (CostModel.CLASSICAL, 256): {
        1024: 14,
        2048: 29,
        4096: 58,
        8192: 118,
        16384: 237,
        32768: 476,
        65536: 929,
        131072: 1866,
    },
    (CostModel.QUANTUM, 128): {
        1024: 25,
        2048: 51,
        4096: 101,
        8192: 202,
        16384: 411,
        32768: 827,
        65536: 1663,
        131072: 3348,
    },
    (CostModel.QUANTUM, 192): {
        1024: 17,
        2048: 35,
        4096: 70,
        8192: 141,
        16384: 284,
        32768: 571,
        65536: 1145,
        131072: 2301,
    },
    (CostModel.QUANTUM, 256): {
        1024: 13,
        2048: 27,
        4096: 54,
        8192: 109,
        16384: 220,
        32768: 443,
        65536: 888,
        131072: 1784,
    },
}
OPENFHE_UNIFORM_TERNARY_CEILINGS: Mapping[tuple[CostModel, int], Mapping[int, int]] = (
    MappingProxyType(
        {
            key: MappingProxyType(dict(value))
            for key, value in _OPENFHE_CEILINGS_RAW.items()
        }
    )
)
del _OPENFHE_CEILINGS_RAW
SUPPORTED_SECURITY_BITS = tuple(
    sorted({bits for _, bits in OPENFHE_UNIFORM_TERNARY_CEILINGS})
)
SUPPORTED_DEGREES = tuple(
    sorted(
        set.intersection(
            *(set(ceilings) for ceilings in OPENFHE_UNIFORM_TERNARY_CEILINGS.values())
        )
    )
)


@dataclass(frozen=True)
class SecurityTarget:
    """An explicit security strength and attack-cost model."""

    bits: int
    cost_model: CostModel

    def __post_init__(self) -> None:
        object.__setattr__(self, "bits", _integer(self.bits, "security bits"))
        object.__setattr__(self, "cost_model", _cost_model(self.cost_model))
        if (self.cost_model, self.bits) not in OPENFHE_UNIFORM_TERNARY_CEILINGS:
            supported = ", ".join(str(bits) for bits in SUPPORTED_SECURITY_BITS)
            raise ValueError(f"security bits must be one of: {supported}")


def security_ceiling(degree: int, target: SecurityTarget) -> int:
    """Return OpenFHE's maximum ``GetMSB(Q*P)`` for ring dimension ``N``."""
    if not isinstance(target, SecurityTarget):
        raise TypeError("target must be a SecurityTarget")
    degree = _integer(degree, "degree")
    ceilings = OPENFHE_UNIFORM_TERNARY_CEILINGS[(target.cost_model, target.bits)]
    try:
        return ceilings[degree]
    except KeyError as exc:
        supported = ", ".join(str(value) for value in sorted(ceilings))
        raise ValueError(
            f"degree {degree} is not tabulated for {target.bits}-bit "
            f"{target.cost_model.value} security; supported: {supported}"
        ) from exc


def openfhe_logq_bits(modulus_product: int) -> int:
    """Return the integer bit length OpenFHE calls ``GetMSB()``.

    OpenFHE passes ``GetParamsQP()->GetModulus().GetMSB()`` to
    ``StdLatticeParm::FindRingDim``. For positive integers this is Python's
    ``int.bit_length()``, including the extra bit at exact powers of two.
    """
    modulus_product = _integer(modulus_product, "modulus_product")
    if modulus_product < 1:
        raise ValueError("modulus_product must be positive")
    return modulus_product.bit_length()


def modulus_product_is_secure(
    modulus_product: int, degree: int, target: SecurityTarget
) -> bool:
    """Mirror OpenFHE's inclusive ``GetMSB(QP) <= maxLogQ`` check."""
    return openfhe_logq_bits(modulus_product) <= security_ceiling(degree, target)


def qp_is_secure(
    q_towers: Sequence[int],
    p_towers: Sequence[int],
    degree: int,
    target: SecurityTarget,
) -> bool:
    """Return whether the complete Q*P evaluation-key modulus is secure."""
    q_values = _moduli(q_towers, "q_towers")
    p_values = _moduli(p_towers, "p_towers")
    return modulus_product_is_secure(
        math.prod((*q_values, *p_values)),
        degree,
        target,
    )


def p_covers_q(q_towers: Sequence[int], p_towers: Sequence[int], dnum: int) -> bool:
    """Return whether P covers every hybrid key-switch Q partition."""
    p_values = _moduli(p_towers, "p_towers", allow_empty=True)
    if not p_values:
        return False
    return math.prod(p_values) >= max(q_partition_products(q_towers, dnum))


def assert_p_coverage(
    q_towers: Sequence[int], p_towers: Sequence[int], dnum: int
) -> None:
    """Raise if P is smaller than any exact Q partition product."""
    required = max(q_partition_products(q_towers, dnum))
    actual = math.prod(_moduli(p_towers, "p_towers"))
    if actual < required:
        raise ValueError(
            f"P product ({actual.bit_length()} bits) does not cover the "
            f"largest Q partition ({required.bit_length()} bits) for "
            f"dnum={dnum}"
        )


def validate_kernel_moduli(
    q_towers: Sequence[int], p_towers: Sequence[int], dnum: int
) -> None:
    """Validate target-independent arithmetic invariants of CROSS kernels."""
    q_values = _moduli(q_towers, "q_towers")
    p_values = _moduli(p_towers, "p_towers")
    validate_barrett_bconv_moduli(q_values, p_values, dnum)
    for modulus in (*q_values, *p_values):
        if not is_prime(modulus):
            raise ValueError(f"modulus {modulus} is not prime")


is_prime = is_prime_deterministic


def _validate_degree(degree: int) -> int:
    degree = _integer(degree, "degree")
    if degree < 2 or degree & (degree - 1):
        raise ValueError(f"degree must be a power of two >= 2, got {degree}")
    return degree


def gen_ntt_primes(
    degree: int,
    bits: int,
    count: int,
    avoid: Iterable[int] = (),
) -> tuple[int, ...]:
    """Generate distinct, exactly ``bits``-bit primes congruent to 1 mod 2N."""
    degree = _validate_degree(degree)
    bits = _integer(bits, "bits")
    count = _integer(count, "count")
    if not 2 <= bits <= MAX_MODULUS_BITS:
        raise ValueError(f"modulus bits must be in [2, {MAX_MODULUS_BITS}], got {bits}")
    if count < 1:
        raise ValueError(f"prime count must be positive, got {count}")

    excluded = {_integer(value, "avoid value") for value in avoid}
    step = 2 * degree
    lower = 1 << (bits - 1)
    upper = 1 << bits
    multiplier = (upper - 2) // step
    primes: list[int] = []
    while multiplier > 0 and len(primes) < count:
        candidate = multiplier * step + 1
        if candidate < lower:
            break
        if candidate not in excluded and is_prime(candidate):
            primes.append(candidate)
            excluded.add(candidate)
        multiplier -= 1
    if len(primes) != count:
        raise ParameterGenerationError(
            f"found {len(primes)}/{count} distinct {bits}-bit NTT primes "
            f"for degree {degree}"
        )
    return tuple(primes)


def _gen_lowest_ntt_primes(
    degree: int,
    bits: int,
    count: int,
    avoid: Iterable[int] = (),
) -> tuple[int, ...]:
    """Generate the lowest distinct NTT primes in an exact bit-width."""
    degree = _validate_degree(degree)
    bits = _integer(bits, "bits")
    count = _integer(count, "count")
    if not 2 <= bits <= MAX_MODULUS_BITS:
        raise ValueError(f"modulus bits must be in [2, {MAX_MODULUS_BITS}], got {bits}")
    if count < 1:
        raise ValueError(f"prime count must be positive, got {count}")

    excluded = {_integer(value, "avoid value") for value in avoid}
    step = 2 * degree
    lower = 1 << (bits - 1)
    upper = 1 << bits
    multiplier = (lower + step - 1) // step
    primes: list[int] = []
    while len(primes) < count:
        candidate = multiplier * step + 1
        if candidate >= upper:
            break
        if candidate not in excluded and is_prime(candidate):
            primes.append(candidate)
            excluded.add(candidate)
        multiplier += 1
    if len(primes) != count:
        raise ParameterGenerationError(
            f"found {len(primes)}/{count} distinct {bits}-bit NTT primes "
            f"for degree {degree}"
        )
    return tuple(primes)


def estimate_openfhe_aux_prime_count(
    logical_num_q: int,
    dnum: int,
    first_mod_size: int,
    scaling_mod_size: int,
    aux_mod_size: int,
) -> int:
    """Mirror OpenFHE 1.5.1 ``EstimateLogP`` for CKKS HYBRID.

    This estimate runs on logical composite moduli, before OpenFHE expands Q
    into physical prime limbs.
    """
    logical_num_q = _integer(logical_num_q, "logical_num_q")
    dnum = _integer(dnum, "dnum")
    first_mod_size = _integer(first_mod_size, "first_mod_size")
    scaling_mod_size = _integer(scaling_mod_size, "scaling_mod_size")
    aux_mod_size = _integer(aux_mod_size, "aux_mod_size")
    if logical_num_q < 1 or dnum < 1:
        raise ValueError("logical_num_q and dnum must be positive")
    if first_mod_size <= scaling_mod_size:
        raise ValueError("first_mod_size must exceed scaling_mod_size")
    if aux_mod_size < 2:
        raise ValueError("aux_mod_size must be at least 2")

    num_part_q = min(dnum, logical_num_q)
    alpha = math.ceil(logical_num_q / num_part_q)
    if logical_num_q <= alpha * (num_part_q - 1):
        raise ParameterGenerationError(
            f"cannot distribute {logical_num_q} logical Q moduli into "
            f"{num_part_q} HYBRID digits"
        )
    widths = (first_mod_size,) + (scaling_mod_size,) * (logical_num_q - 1)
    max_bits = max(
        sum(widths[start : start + alpha])
        for start in range(0, logical_num_q, alpha)
    )
    # ParamsGenCKKSRNSInternal passes addOne=true.
    if max_bits != aux_mod_size:
        max_bits += 1
    return math.ceil(max_bits / aux_mod_size)


def estimate_openfhe_log_qp(spec: "ModelSpec") -> int:
    """Return OpenFHE's pre-generation QP security bound for ``spec``."""
    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a ModelSpec")
    q_bound = (
        spec.first_mod_size
        + (spec.num_q - 1) * spec.scaling_mod_size
    )
    if q_bound != spec.aux_mod_size:
        q_bound += 1
    p_count = estimate_openfhe_aux_prime_count(
        spec.num_q,
        spec.dnum,
        spec.first_mod_size,
        spec.scaling_mod_size,
        spec.aux_mod_size,
    )
    return q_bound + p_count * spec.aux_mod_size


@dataclass(frozen=True)
class ModelSpec:
    """OpenFHE CKKS composite-scaling inputs for one demo model.

    ``num_q`` is OpenFHE's logical ``numPrimes`` value before composite
    expansion. The generated physical RNS chain therefore contains
    ``num_q * composite_degree`` Q limbs.
    """

    name: str
    num_q: int
    dnum: int
    scaling_mod_size: int
    first_mod_size: int
    register_word_size: int
    composite_degree: int
    min_slots: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model name must be non-empty")
        for field in (
            "num_q",
            "dnum",
            "scaling_mod_size",
            "first_mod_size",
            "register_word_size",
            "composite_degree",
            "min_slots",
        ):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        if self.num_q < 1:
            raise ValueError("num_q must be positive")
        if not 1 <= self.dnum <= self.num_q:
            raise ValueError("dnum must be between 1 and num_q")
        if self.composite_degree != 2:
            raise ValueError("CROSS demo profiles require composite_degree=2")
        if self.scaling_mod_size < 2:
            raise ValueError("scaling_mod_size must be at least 2")
        if self.first_mod_size <= self.scaling_mod_size:
            raise ValueError("first_mod_size must exceed scaling_mod_size")
        if not 3 <= self.register_word_size <= MAX_KERNEL_MODULUS_BITS + 1:
            raise ValueError("register_word_size is outside the CROSS envelope")
        max_component_bits = max(
            math.ceil(self.first_mod_size / self.composite_degree),
            math.ceil(self.scaling_mod_size / self.composite_degree),
            self.register_word_size - 1,
        )
        if max_component_bits > MAX_KERNEL_MODULUS_BITS:
            raise ValueError("profile can generate a modulus wider than 31 bits")
        if self.min_slots < 1:
            raise ValueError("min_slots must be positive")

    @property
    def physical_num_q(self) -> int:
        return self.num_q * self.composite_degree

    @property
    def multiplicative_depth(self) -> int:
        return self.num_q - 1

    @property
    def aux_mod_size(self) -> int:
        return self.register_word_size - 1

    @property
    def q_bits(self) -> int:
        """Compatibility alias for the nominal physical scaling-limb width."""
        return math.ceil(self.scaling_mod_size / self.composite_degree)

    @property
    def p_bits(self) -> int:
        """Compatibility alias for OpenFHE's HYBRID auxiliary-prime width."""
        return self.aux_mod_size

    @property
    def scale_towers(self) -> int:
        """Compatibility alias; one logical scale contains this many limbs."""
        return self.composite_degree


# These values describe the active demo schedules, not retained sweep pools.
MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "AlexNetHE": ModelSpec(
            name="AlexNetHE",
            num_q=17,
            dnum=9,
            scaling_mod_size=60,
            first_mod_size=61,
            register_word_size=32,
            composite_degree=2,
            min_slots=1024,
        ),
        "LoLAHE": ModelSpec(
            name="LoLAHE",
            num_q=7,
            dnum=4,
            scaling_mod_size=60,
            first_mod_size=61,
            register_word_size=32,
            composite_degree=2,
            min_slots=1024,
        ),
        "LeNetHE": ModelSpec(
            name="LeNetHE",
            num_q=9,
            dnum=5,
            scaling_mod_size=60,
            first_mod_size=61,
            register_word_size=32,
            composite_degree=2,
            min_slots=1024,
        ),
        "AlexNetTinyHE": ModelSpec(
            name="AlexNetTinyHE",
            num_q=9,
            dnum=5,
            scaling_mod_size=60,
            first_mod_size=61,
            register_word_size=32,
            composite_degree=2,
            min_slots=1024,
        ),
    }
)


@dataclass(frozen=True)
class _CorrectnessProfileSpec:
    """Legacy CD1 schedule retained for fast encrypted correctness tests."""

    num_q: int
    num_p: int
    dnum: int


_CORRECTNESS_2048_SPECS: Mapping[str, _CorrectnessProfileSpec] = MappingProxyType(
    {
        "LoLAHE": _CorrectnessProfileSpec(num_q=7, num_p=3, dnum=3),
        "LeNetHE": _CorrectnessProfileSpec(num_q=9, num_p=4, dnum=4),
        "AlexNetTinyHE": _CorrectnessProfileSpec(num_q=9, num_p=4, dnum=4),
        "AlexNetHE": _CorrectnessProfileSpec(num_q=17, num_p=4, dnum=5),
    }
)


@lru_cache(maxsize=None)
def _correctness_2048_parameters(
    model: str,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Return the exact legacy CD1 chain and scale for a registered model."""
    profile = _CORRECTNESS_2048_SPECS[model]
    degree = 2048
    q_towers = _gen_lowest_ntt_primes(
        degree, bits=30, count=profile.num_q
    )
    p_towers = gen_ntt_primes(
        degree,
        bits=31,
        count=profile.num_p,
        avoid=q_towers,
    )
    return q_towers, p_towers, q_towers[0] * q_towers[1]


@dataclass(frozen=True)
class RuntimeRingConfig:
    """Immutable model-bound correctness config with no security claim."""

    name: str
    model: str
    degree: int
    num_slots: int
    q_towers: tuple[int, ...]
    p_towers: tuple[int, ...]
    scaling_factor: int
    sigma: float
    dnum: int
    composite_degree: int
    first_mod_size: int
    scaling_mod_size: int
    register_word_size: int
    target: None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.model, str):
            raise TypeError("config name and model must be strings")
        for field_name in (
            "degree",
            "num_slots",
            "scaling_factor",
            "dnum",
            "composite_degree",
            "first_mod_size",
            "scaling_mod_size",
            "register_word_size",
        ):
            object.__setattr__(
                self,
                field_name,
                _integer(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "q_towers", _moduli(self.q_towers, "q_towers"))
        object.__setattr__(self, "p_towers", _moduli(self.p_towers, "p_towers"))
        if isinstance(self.sigma, (bool, str, bytes)):
            raise TypeError("sigma must be a real number")
        try:
            object.__setattr__(self, "sigma", float(self.sigma))
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("sigma must be a real number") from exc
        if self.target is not None:
            raise ValueError("RuntimeRingConfig cannot carry a security target")
        validate_runtime_ring_config(self)

    @property
    def security_bits(self) -> None:
        """No security strength is claimed for a runtime-only config."""
        return None

    @property
    def cost_model(self) -> None:
        """No attack-cost model applies without a security target."""
        return None

    @property
    def secret_distribution(self) -> str:
        return SECRET_DISTRIBUTION

    @property
    def ring_dimension(self) -> int:
        return self.degree

    @property
    def cyclotomic_order(self) -> int:
        return 2 * self.degree

    @property
    def max_level(self) -> int:
        return (len(self.q_towers) - 1) // self.composite_degree

    def runtime_kwargs(self) -> dict[str, object]:
        """Return the common fields consumed by demo model constructors."""
        return {
            "name": self.name,
            "degree": self.degree,
            "num_slots": self.num_slots,
            "q_towers": self.q_towers,
            "p_towers": self.p_towers,
            "scaling_factor": self.scaling_factor,
            "sigma": self.sigma,
            "dnum": self.dnum,
            "composite_degree": self.composite_degree,
            "first_mod_size": self.first_mod_size,
            "scaling_mod_size": self.scaling_mod_size,
            "register_word_size": self.register_word_size,
        }


@dataclass(frozen=True)
class RingConfig:
    """Immutable CKKS config where ``degree`` is ring dimension ``N``."""

    name: str
    model: str
    degree: int
    num_slots: int
    q_towers: tuple[int, ...]
    p_towers: tuple[int, ...]
    scaling_factor: int
    sigma: float
    dnum: int
    composite_degree: int
    first_mod_size: int
    scaling_mod_size: int
    register_word_size: int
    target: SecurityTarget

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.model, str):
            raise TypeError("config name and model must be strings")
        for field in (
            "degree",
            "num_slots",
            "scaling_factor",
            "dnum",
            "composite_degree",
            "first_mod_size",
            "scaling_mod_size",
            "register_word_size",
        ):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        object.__setattr__(self, "q_towers", _moduli(self.q_towers, "q_towers"))
        object.__setattr__(self, "p_towers", _moduli(self.p_towers, "p_towers"))
        if isinstance(self.sigma, (bool, str, bytes)):
            raise TypeError("sigma must be a real number")
        try:
            object.__setattr__(self, "sigma", float(self.sigma))
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("sigma must be a real number") from exc
        if not isinstance(self.target, SecurityTarget):
            raise TypeError("target must be a SecurityTarget")
        validate_ring_config(self)

    @property
    def security_bits(self) -> int:
        return self.target.bits

    @property
    def cost_model(self) -> str:
        return self.target.cost_model.value

    @property
    def secret_distribution(self) -> str:
        return SECRET_DISTRIBUTION

    @property
    def ring_dimension(self) -> int:
        """Polynomial ring dimension ``N`` used by the security table."""
        return self.degree

    @property
    def cyclotomic_order(self) -> int:
        """Power-of-two cyclotomic order ``m = 2*N`` used by the NTT."""
        return 2 * self.degree

    @property
    def hes_ceiling(self) -> int:
        return security_ceiling(self.degree, self.target)

    @property
    def qp_product(self) -> int:
        return math.prod((*self.q_towers, *self.p_towers))

    @property
    def log2_qp(self) -> float:
        return sum(math.log2(value) for value in (*self.q_towers, *self.p_towers))

    @property
    def openfhe_log_qp(self) -> int:
        """Integer QP bit length passed to OpenFHE's security lookup."""
        return openfhe_logq_bits(self.qp_product)

    @property
    def logical_num_q(self) -> int:
        return len(self.q_towers) // self.composite_degree

    @property
    def multiplicative_depth(self) -> int:
        return self.logical_num_q - 1

    @property
    def max_level(self) -> int:
        return (len(self.q_towers) - 1) // self.composite_degree

    @property
    def openfhe_estimated_log_qp(self) -> int:
        spec = ModelSpec(
            name=self.model,
            num_q=self.logical_num_q,
            dnum=self.dnum,
            scaling_mod_size=self.scaling_mod_size,
            first_mod_size=self.first_mod_size,
            register_word_size=self.register_word_size,
            composite_degree=self.composite_degree,
            min_slots=self.num_slots,
        )
        return estimate_openfhe_log_qp(spec)

    @property
    def modulus_headroom_bits(self) -> float:
        return self.hes_ceiling - self.log2_qp

    def runtime_kwargs(self) -> dict[str, object]:
        """Return fields understood by existing duck-typed demo configs."""
        return {
            "name": self.name,
            "degree": self.degree,
            "num_slots": self.num_slots,
            "q_towers": self.q_towers,
            "p_towers": self.p_towers,
            "scaling_factor": self.scaling_factor,
            "sigma": self.sigma,
            "dnum": self.dnum,
            "composite_degree": self.composite_degree,
            "first_mod_size": self.first_mod_size,
            "scaling_mod_size": self.scaling_mod_size,
            "register_word_size": self.register_word_size,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize to the versioned JSON/Python schema."""
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "model": self.model,
            "security_bits": self.security_bits,
            "cost_model": self.cost_model,
            "secret_distribution": self.secret_distribution,
            "degree": self.degree,
            "ring_dimension": self.ring_dimension,
            "cyclotomic_order": self.cyclotomic_order,
            "num_slots": self.num_slots,
            "q_towers": list(self.q_towers),
            "p_towers": list(self.p_towers),
            "scaling_factor": self.scaling_factor,
            "sigma": self.sigma,
            "dnum": self.dnum,
            "composite_degree": self.composite_degree,
            "logical_num_q": self.logical_num_q,
            "multiplicative_depth": self.multiplicative_depth,
            "first_mod_size": self.first_mod_size,
            "scaling_mod_size": self.scaling_mod_size,
            "register_word_size": self.register_word_size,
            "hes_ceiling": self.hes_ceiling,
            "openfhe_log_QP": self.openfhe_log_qp,
            "openfhe_estimated_log_QP": self.openfhe_estimated_log_qp,
            "log2_QP": round(self.log2_qp, 6),
            "modulus_headroom_bits": round(self.modulus_headroom_bits, 6),
            "security_table_source": SECURITY_TABLE_SOURCE,
            "security_table_url": SECURITY_TABLE_URL,
            "openfhe_reference_commit": OPENFHE_REFERENCE_COMMIT,
            "openfhe_parameter_generator_url": OPENFHE_PARAMETER_GENERATOR_URL,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RingConfig":
        """Load and revalidate a serialized configuration."""
        if not isinstance(value, Mapping):
            raise TypeError("serialized config must be a mapping")
        version = value.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version!r}; " f"expected {SCHEMA_VERSION}"
            )
        config = cls(
            name=value["name"],  # type: ignore[arg-type]
            model=value["model"],  # type: ignore[arg-type]
            degree=value["degree"],  # type: ignore[arg-type]
            num_slots=value["num_slots"],  # type: ignore[arg-type]
            q_towers=value["q_towers"],  # type: ignore[arg-type]
            p_towers=value["p_towers"],  # type: ignore[arg-type]
            scaling_factor=value["scaling_factor"],  # type: ignore[arg-type]
            sigma=value["sigma"],  # type: ignore[arg-type]
            dnum=value["dnum"],  # type: ignore[arg-type]
            composite_degree=value["composite_degree"],  # type: ignore[arg-type]
            first_mod_size=value["first_mod_size"],  # type: ignore[arg-type]
            scaling_mod_size=value["scaling_mod_size"],  # type: ignore[arg-type]
            register_word_size=value["register_word_size"],  # type: ignore[arg-type]
            target=SecurityTarget(
                bits=value["security_bits"],  # type: ignore[arg-type]
                cost_model=_cost_model(value["cost_model"]),  # type: ignore[arg-type]
            ),
        )
        expected = config.to_dict()
        for derived in (
            "hes_ceiling",
            "ring_dimension",
            "cyclotomic_order",
            "openfhe_log_QP",
            "openfhe_estimated_log_QP",
            "logical_num_q",
            "multiplicative_depth",
            "log2_QP",
            "modulus_headroom_bits",
            "secret_distribution",
            "security_table_source",
            "security_table_url",
            "openfhe_reference_commit",
            "openfhe_parameter_generator_url",
        ):
            if derived in value and value[derived] != expected[derived]:
                raise ValueError(f"serialized {derived} does not match config")
        return config


@dataclass(frozen=True)
class InferenceSetup:
    """One explicit raw, correctness-only, or secure inference setup."""

    model: str
    profile: InferenceProfile
    config: RuntimeRingConfig | RingConfig | None

    def __post_init__(self) -> None:
        if self.model not in MODEL_SPECS:
            choices = ", ".join(MODEL_SPECS)
            raise ValueError(f"unknown model {self.model!r}; expected one of: {choices}")
        if not isinstance(self.profile, InferenceProfile):
            raise TypeError("profile must be an InferenceProfile")

        if self.profile is InferenceProfile.RAW_REFERENCE:
            if self.config is not None:
                raise ValueError("raw inference cannot carry encrypted ring parameters")
            return
        if self.profile is InferenceProfile.ENCRYPTED_CORRECTNESS:
            if not isinstance(self.config, RuntimeRingConfig):
                raise TypeError(
                    "encrypted correctness requires a RuntimeRingConfig"
                )
            if self.config.model != self.model or self.config.degree != 2048:
                raise ValueError(
                    "encrypted correctness config does not match its "
                    "model/profile"
                )
            if self.config.target is not None:
                raise ValueError(
                    "encrypted correctness must not claim a security target"
                )
            return
        if self.profile is not InferenceProfile.ENCRYPTED_SECURE:
            raise ValueError(f"unsupported inference profile {self.profile!r}")
        if not isinstance(self.config, RingConfig):
            raise TypeError("encrypted secure requires a RingConfig")
        if self.config.model != self.model:
            raise ValueError(
                "encrypted secure config is bound to a different model"
            )
        if (
            self.config.target.bits != 128
            or self.config.target.cost_model is not CostModel.CLASSICAL
        ):
            raise ValueError(
                "encrypted secure requires classical 128-bit validation"
            )

    @property
    def encrypted(self) -> bool:
        return self.profile is not InferenceProfile.RAW_REFERENCE

    @property
    def require_128bit(self) -> bool:
        """Whether model construction must enforce the secure profile."""
        return self.profile is InferenceProfile.ENCRYPTED_SECURE

    @property
    def security_bits(self) -> int | None:
        if isinstance(self.config, RingConfig):
            return self.config.security_bits
        return None

    @property
    def has_security_guarantee(self) -> bool:
        return self.security_bits is not None

    @property
    def execution_policy(self) -> MappingExecutionPolicy:
        """Resource policy selected by this registered inference profile."""
        return inference_execution_policy(self.require_128bit)

    def constructor_kwargs(self) -> dict[str, object]:
        """Return the common encrypted-demo constructor arguments."""
        if self.config is None:
            raise ValueError("raw inference does not construct an encrypted model")
        return {
            "config": self.config,
            "require_128bit": self.require_128bit,
            "execution_policy": self.execution_policy,
        }


_MISSING = object()


def _runtime_field(config: object, field: str, default: object = _MISSING) -> object:
    if isinstance(config, Mapping):
        if field in config:
            return config[field]
    else:
        try:
            return getattr(config, field)
        except AttributeError:
            pass
    if default is not _MISSING:
        return default
    raise TypeError(f"runtime config is missing {field!r}")


def _runtime_target(config: object, target: SecurityTarget | None) -> SecurityTarget:
    embedded = _runtime_field(config, "target", None)
    if embedded is None and isinstance(config, Mapping):
        if "security_bits" in config and "cost_model" in config:
            embedded = SecurityTarget(
                bits=config["security_bits"],  # type: ignore[arg-type]
                cost_model=_cost_model(config["cost_model"]),  # type: ignore[arg-type]
            )
    if embedded is not None and not isinstance(embedded, SecurityTarget):
        raise TypeError("embedded target must be a SecurityTarget")
    if target is None:
        target = embedded
    elif not isinstance(target, SecurityTarget):
        raise TypeError("target must be a SecurityTarget")
    elif embedded is not None and embedded != target:
        raise ValueError("explicit target does not match embedded target")
    if target is None:
        raise TypeError("target is required when the runtime config does not embed one")
    return target


def validate_runtime_ring_config(config: RuntimeRingConfig) -> None:
    """Validate the exact model-bound CD1 profile without a security claim."""
    if not isinstance(config, RuntimeRingConfig):
        raise TypeError("config must be a RuntimeRingConfig")
    if not config.name or not config.model:
        raise ValueError("config name and model must be non-empty")
    if config.model not in MODEL_SPECS:
        choices = ", ".join(MODEL_SPECS)
        raise ValueError(
            f"unknown model {config.model!r}; expected one of: {choices}"
        )
    degree = _validate_degree(config.degree)
    if config.num_slots != degree // 2:
        raise ValueError("num_slots must equal degree // 2")
    if not 1 <= config.dnum <= len(config.q_towers):
        raise ValueError("dnum must be between 1 and the Q tower count")
    if config.composite_degree < 1:
        raise ValueError("composite_degree must be positive")
    if len(config.q_towers) % config.composite_degree:
        raise ValueError("Q tower count must be divisible by composite_degree")
    if config.scaling_factor < 1:
        raise ValueError("scaling_factor must be positive")
    if not math.isfinite(config.sigma) or config.sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if config.first_mod_size < 2 or config.scaling_mod_size < 2:
        raise ValueError("modulus-size metadata must be at least 2 bits")
    if not 3 <= config.register_word_size <= MAX_KERNEL_MODULUS_BITS + 1:
        raise ValueError("register_word_size is outside the CROSS envelope")

    validate_kernel_moduli(config.q_towers, config.p_towers, config.dnum)
    ntt_order = 2 * degree
    for modulus in (*config.q_towers, *config.p_towers):
        if modulus % ntt_order != 1:
            raise ValueError(f"modulus {modulus} is not 1 mod {ntt_order}")

    profile = _CORRECTNESS_2048_SPECS[config.model]
    expected_q, expected_p, expected_scale = _correctness_2048_parameters(
        config.model
    )
    actual = (
        config.degree,
        config.num_slots,
        config.q_towers,
        config.p_towers,
        config.scaling_factor,
        config.sigma,
        config.dnum,
        config.composite_degree,
        config.first_mod_size,
        config.scaling_mod_size,
        config.register_word_size,
    )
    expected = (
        2048,
        1024,
        expected_q,
        expected_p,
        expected_scale,
        DEFAULT_SIGMA,
        profile.dnum,
        1,
        expected_q[0].bit_length(),
        expected_scale.bit_length(),
        MAX_KERNEL_MODULUS_BITS + 1,
    )
    if actual != expected:
        raise ValueError(
            f"{config.model} encrypted correctness requires its exact "
            "registered degree-2048 profile"
        )


def _validate_model_binding(
    model: object,
    *,
    q_towers: tuple[int, ...],
    p_towers: tuple[int, ...],
    dnum: int,
    composite_degree: int,
    first_mod_size: int,
    scaling_mod_size: int,
    register_word_size: int,
    num_slots: int,
    scaling_factor: int,
    sigma: float,
) -> None:
    if not isinstance(model, str) or not model:
        raise ValueError("config model must be a non-empty string")
    spec = MODEL_SPECS.get(model)
    if spec is None:
        return
    if len(q_towers) != spec.physical_num_q:
        raise ValueError(
            f"{model} requires exactly {spec.physical_num_q} physical Q "
            f"towers ({spec.num_q} logical x CD{spec.composite_degree}), "
            f"got {len(q_towers)}"
        )
    if dnum != spec.dnum:
        raise ValueError(f"{model} requires dnum={spec.dnum}, got {dnum}")
    for field, actual in (
        ("composite_degree", composite_degree),
        ("first_mod_size", first_mod_size),
        ("scaling_mod_size", scaling_mod_size),
        ("register_word_size", register_word_size),
    ):
        expected = getattr(spec, field)
        if actual != expected:
            raise ValueError(f"{model} requires {field}={expected}, got {actual}")
    if num_slots < spec.min_slots:
        raise ValueError(
            f"{model} requires at least {spec.min_slots} slots, got {num_slots}"
        )
    if any(modulus.bit_length() != spec.aux_mod_size for modulus in p_towers):
        raise ValueError(f"{model} requires {spec.aux_mod_size}-bit P towers")
    expected_scale = math.prod(q_towers[-spec.composite_degree :])
    if scaling_factor != expected_scale:
        raise ValueError(
            f"{model} scaling_factor must equal the product of its last "
            f"{spec.composite_degree} physical Q towers"
        )
    if sigma != DEFAULT_SIGMA:
        raise ValueError(f"{model} requires sigma={DEFAULT_SIGMA!r}")


@lru_cache(maxsize=None)
def _reference_composite_q_towers(
    degree: int,
    logical_num_q: int,
    composite_degree: int,
    first_mod_size: int,
    scaling_mod_size: int,
    register_word_size: int,
) -> tuple[int, ...]:
    """Regenerate OpenFHE's deterministic physical Q chain."""
    try:
        return tuple(
            composite_prime_gen(
                composite_degree=composite_degree,
                num_primes=logical_num_q * composite_degree,
                first_mod_size=first_mod_size,
                scaling_mod_size=scaling_mod_size,
                cycl_order=2 * degree,
                register_word_size=register_word_size,
            )
        )
    except ValueError as exc:
        raise ParameterGenerationError(
            f"OpenFHE composite-prime search failed for degree {degree}"
        ) from exc


def _first_mismatch(
    actual: Sequence[int], expected: Sequence[int]
) -> tuple[int, int | None, int | None] | None:
    """Return the first differing index and values, including length changes."""
    for index, (actual_value, expected_value) in enumerate(
        zip(actual, expected, strict=False)
    ):
        if actual_value != expected_value:
            return index, actual_value, expected_value
    if len(actual) != len(expected):
        index = min(len(actual), len(expected))
        return (
            index,
            actual[index] if index < len(actual) else None,
            expected[index] if index < len(expected) else None,
        )
    return None


def validate_runtime_config(
    config: object,
    target: SecurityTarget | None = None,
    *,
    model: str | None = None,
) -> None:
    """Explicitly revalidate a RingConfig or compatible cached runtime object.

    ``model`` binds legacy/duck-typed configs that do not carry their own model
    name to an exact registered demo schedule.
    """
    target = _runtime_target(config, target)
    distribution = _runtime_field(config, "secret_distribution", None)
    if distribution is not None and distribution != SECRET_DISTRIBUTION:
        raise ValueError(f"security table requires {SECRET_DISTRIBUTION!r} secrets")
    degree = _validate_degree(_runtime_field(config, "degree"))
    num_slots = _integer(_runtime_field(config, "num_slots"), "num_slots")
    q_towers = _moduli(
        _runtime_field(config, "q_towers"),  # type: ignore[arg-type]
        "q_towers",
    )
    p_towers = _moduli(
        _runtime_field(config, "p_towers"),  # type: ignore[arg-type]
        "p_towers",
    )
    scaling_factor = _integer(
        _runtime_field(config, "scaling_factor"), "scaling_factor"
    )
    dnum = _integer(_runtime_field(config, "dnum"), "dnum")
    composite_degree = _integer(
        _runtime_field(config, "composite_degree"), "composite_degree"
    )
    first_mod_size = _integer(
        _runtime_field(config, "first_mod_size"), "first_mod_size"
    )
    scaling_mod_size = _integer(
        _runtime_field(config, "scaling_mod_size"), "scaling_mod_size"
    )
    register_word_size = _integer(
        _runtime_field(config, "register_word_size"), "register_word_size"
    )
    sigma = _runtime_field(config, "sigma")

    if num_slots != degree // 2:
        raise ValueError("num_slots must equal degree // 2")
    if dnum < 1:
        raise ValueError("dnum must be positive")
    if composite_degree != 2:
        raise ValueError("CROSS secure profiles require composite_degree=2")
    if len(q_towers) % composite_degree:
        raise ValueError(
            "physical Q tower count must be divisible by composite_degree"
        )
    if len(q_towers) <= composite_degree:
        raise ValueError("Q must contain a first and a scaling composite modulus")
    if first_mod_size <= scaling_mod_size:
        raise ValueError("first_mod_size must exceed scaling_mod_size")
    if not 3 <= register_word_size <= MAX_KERNEL_MODULUS_BITS + 1:
        raise ValueError("register_word_size is outside the CROSS envelope")
    if scaling_factor < 1:
        raise ValueError("scaling_factor must be positive")
    if isinstance(sigma, (bool, str, bytes)):
        raise TypeError("sigma must be a real number")
    try:
        sigma_value = float(sigma)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("sigma must be a real number") from exc
    sigma_value = kg.normalize_noise_std(sigma_value)
    if sigma_value < DEFAULT_SIGMA:
        raise ValueError(f"OpenFHE security tables require sigma >= {DEFAULT_SIGMA!r}")

    logical_num_q = len(q_towers) // composite_degree
    aux_mod_size = register_word_size - 1
    estimated_p_count = estimate_openfhe_aux_prime_count(
        logical_num_q,
        dnum,
        first_mod_size,
        scaling_mod_size,
        aux_mod_size,
    )
    if len(p_towers) != estimated_p_count:
        raise ValueError(
            f"OpenFHE HYBRID profile requires exactly {estimated_p_count} "
            f"P towers, got {len(p_towers)}"
        )
    validate_kernel_moduli(q_towers, p_towers, dnum)
    expected_q_towers = _reference_composite_q_towers(
        degree,
        logical_num_q,
        composite_degree,
        first_mod_size,
        scaling_mod_size,
        register_word_size,
    )
    mismatch = _first_mismatch(q_towers, expected_q_towers)
    if mismatch is not None:
        index, actual, expected = mismatch
        raise ValueError(
            "Q towers do not match OpenFHE 1.5.1 "
            f"CompositePrimeModuliGen at physical index {index}: "
            f"got {actual}, expected {expected}"
        )
    expected_p_towers = tuple(
        generate_p_towers(
            expected_q_towers,
            dnum,
            degree,
            aux_mod_size,
        )
    )
    mismatch = _first_mismatch(p_towers, expected_p_towers)
    if mismatch is not None:
        index, actual, expected = mismatch
        raise ValueError(
            "P towers do not match OpenFHE 1.5.1 auxiliary-prime selection "
            f"at physical index {index}: got {actual}, expected {expected}"
        )
    initial_scale = math.prod(q_towers[-composite_degree:])
    if scaling_factor != initial_scale:
        raise ValueError(
            "scaling_factor must equal the product of the last "
            "composite_degree Q towers"
        )
    if any(modulus.bit_length() != aux_mod_size for modulus in p_towers):
        raise ValueError(
            f"HYBRID P towers must be exactly {aux_mod_size} bits"
        )
    actual_p_count = compute_num_p_towers(q_towers, dnum, aux_mod_size)
    if actual_p_count != estimated_p_count:
        raise ValueError(
            "OpenFHE EstimateLogP does not match the realized Q partitions"
        )
    moduli = (*q_towers, *p_towers)
    ntt_order = 2 * degree
    for modulus in moduli:
        if modulus % ntt_order != 1:
            raise ValueError(f"modulus {modulus} is not 1 mod {ntt_order}")

    ceiling = security_ceiling(degree, target)
    q_bound = first_mod_size + (logical_num_q - 1) * scaling_mod_size
    if q_bound != aux_mod_size:
        q_bound += 1
    estimated_log_qp = q_bound + estimated_p_count * aux_mod_size
    if estimated_log_qp > ceiling:
        raise ValueError(
            f"OpenFHE estimated Q*P bound {estimated_log_qp} exceeds the "
            f"{ceiling}-bit ceiling for degree {degree}"
        )
    if not qp_is_secure(q_towers, p_towers, degree, target):
        raise ValueError(
            f"Q*P exceeds the {ceiling}-bit ceiling for degree {degree}, "
            f"{target.bits}-bit {target.cost_model.value} security"
        )
    embedded_model = _runtime_field(config, "model", None)
    if model is not None:
        if embedded_model is not None and embedded_model != model:
            raise ValueError(
                f"{model} cannot consume a config bound to {embedded_model!r}"
            )
    else:
        model = embedded_model
    if model is not None:
        _validate_model_binding(
            model,
            q_towers=q_towers,
            p_towers=p_towers,
            dnum=dnum,
            composite_degree=composite_degree,
            first_mod_size=first_mod_size,
            scaling_mod_size=scaling_mod_size,
            register_word_size=register_word_size,
            num_slots=num_slots,
            scaling_factor=scaling_factor,
            sigma=sigma_value,
        )


def validate_ring_config(config: RingConfig) -> None:
    """Validate every structural, arithmetic, model, and security invariant."""
    if not isinstance(config, RingConfig):
        raise TypeError("config must be a RingConfig")
    if not config.name or not config.model:
        raise ValueError("config name and model must be non-empty")
    validate_runtime_config(config, config.target)


def generate_ring_config(
    spec: ModelSpec,
    target: SecurityTarget,
    *,
    name: str | None = None,
    sigma: float = DEFAULT_SIGMA,
    degrees: Sequence[int] = SUPPORTED_DEGREES,
) -> RingConfig:
    """Return the smallest supported degree satisfying every exact invariant."""
    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a ModelSpec")
    if not isinstance(target, SecurityTarget):
        raise TypeError("target must be a SecurityTarget")
    ordered_degrees = tuple(sorted({_validate_degree(degree) for degree in degrees}))
    if not ordered_degrees:
        raise ValueError("degrees must be non-empty")

    reference_log_qp = estimate_openfhe_log_qp(spec)
    estimated_p_count = estimate_openfhe_aux_prime_count(
        spec.num_q,
        spec.dnum,
        spec.first_mod_size,
        spec.scaling_mod_size,
        spec.aux_mod_size,
    )

    last_generation_error: ParameterGenerationError | None = None
    for degree in ordered_degrees:
        # Reject untabulated degrees rather than extrapolating.
        ceiling = security_ceiling(degree, target)
        if degree // 2 < spec.min_slots:
            continue
        # OpenFHE selects N using this conservative pre-generation bound.
        if reference_log_qp > ceiling:
            continue
        try:
            q_towers = _reference_composite_q_towers(
                degree,
                spec.num_q,
                spec.composite_degree,
                spec.first_mod_size,
                spec.scaling_mod_size,
                spec.register_word_size,
            )
        except ParameterGenerationError as exc:
            last_generation_error = exc
            continue
        actual_p_count = compute_num_p_towers(
            q_towers, spec.dnum, spec.aux_mod_size
        )
        if actual_p_count != estimated_p_count:
            raise ParameterGenerationError(
                "OpenFHE EstimateLogP mismatch: "
                f"estimated {estimated_p_count}, actual {actual_p_count}"
            )
        p_towers = tuple(
            generate_p_towers(
                q_towers,
                spec.dnum,
                degree,
                spec.aux_mod_size,
            )
        )
        validate_kernel_moduli(q_towers, p_towers, spec.dnum)
        if not qp_is_secure(q_towers, p_towers, degree, target):
            continue
        return RingConfig(
            name=name or f"{spec.name}_{target.bits}bit_{target.cost_model.value}",
            model=spec.name,
            degree=degree,
            num_slots=degree // 2,
            q_towers=q_towers,
            p_towers=p_towers,
            scaling_factor=math.prod(q_towers[-spec.composite_degree :]),
            sigma=sigma,
            dnum=spec.dnum,
            composite_degree=spec.composite_degree,
            first_mod_size=spec.first_mod_size,
            scaling_mod_size=spec.scaling_mod_size,
            register_word_size=spec.register_word_size,
            target=target,
        )

    raise ParameterGenerationError(
        f"no tabulated degree in {ordered_degrees} satisfies {spec.name} at "
        f"{target.bits}-bit {target.cost_model.value} security"
    ) from last_generation_error


@lru_cache(maxsize=None)
def _canonical_config(
    model: str, security_bits: int, cost_model: CostModel
) -> RingConfig:
    return generate_ring_config(
        MODEL_SPECS[model], SecurityTarget(security_bits, cost_model)
    )


def canonical_config(
    model: str,
    bits: int = 128,
    cost_model: CostModel | str = CostModel.CLASSICAL,
) -> RingConfig:
    """Return the authoritative immutable config for a registered demo model."""
    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if model not in MODEL_SPECS:
        choices = ", ".join(MODEL_SPECS)
        raise ValueError(f"unknown model {model!r}; expected one of: {choices}")
    target = SecurityTarget(bits, _cost_model(cost_model))
    return _canonical_config(model, target.bits, target.cost_model)


@lru_cache(maxsize=None)
def _correctness_2048_config(model: str) -> RuntimeRingConfig:
    profile = _CORRECTNESS_2048_SPECS[model]
    degree = 2048
    q_towers, p_towers, scaling_factor = _correctness_2048_parameters(
        model
    )
    return RuntimeRingConfig(
        name=f"{model}_default_insecure",
        model=model,
        degree=degree,
        num_slots=degree // 2,
        q_towers=q_towers,
        p_towers=p_towers,
        scaling_factor=scaling_factor,
        sigma=DEFAULT_SIGMA,
        dnum=profile.dnum,
        composite_degree=1,
        first_mod_size=q_towers[0].bit_length(),
        scaling_mod_size=scaling_factor.bit_length(),
        register_word_size=MAX_KERNEL_MODULUS_BITS + 1,
    )


def _inference_profile(value: InferenceProfile | str) -> InferenceProfile:
    try:
        return value if isinstance(value, InferenceProfile) else InferenceProfile(value)
    except ValueError as exc:
        choices = ", ".join(profile.value for profile in InferenceProfile)
        raise ValueError(f"profile must be one of: {choices}") from exc


def inference_setup(
    model: str,
    profile: InferenceProfile | str,
) -> InferenceSetup:
    """Return the authoritative setup for one demo inference profile.

    ``encrypted_correctness`` is an arithmetic-correctness profile only. Its
    current registered configuration has ring degree 2048. Its
    ``RuntimeRingConfig.target`` and ``InferenceSetup.security_bits`` are both
    ``None``; callers must not infer a security guarantee from its parameters.
    """
    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if model not in MODEL_SPECS:
        choices = ", ".join(MODEL_SPECS)
        raise ValueError(f"unknown model {model!r}; expected one of: {choices}")
    resolved_profile = _inference_profile(profile)
    if resolved_profile is InferenceProfile.RAW_REFERENCE:
        return InferenceSetup(
            model=model,
            profile=resolved_profile,
            config=None,
        )
    if resolved_profile is InferenceProfile.ENCRYPTED_CORRECTNESS:
        return InferenceSetup(
            model=model,
            profile=resolved_profile,
            config=_correctness_2048_config(model),
        )
    if resolved_profile is InferenceProfile.ENCRYPTED_SECURE:
        return InferenceSetup(
            model=model,
            profile=resolved_profile,
            config=canonical_config(
                model,
                bits=128,
                cost_model=CostModel.CLASSICAL,
            ),
        )
    raise ValueError(f"unsupported inference profile {resolved_profile!r}")


####################################
# Security parameter CLI and OpenFHE verification
####################################

_OPENFHE_PROBE_HEADER = {
    "openfhe_version": "1.5.1",
    "security": "HEStd_128_classic",
    "composite_degree": 2,
    "scaling_mod_size": 60,
    "first_mod_size": 61,
    "register_word_size": 32,
}
_OPENFHE_PROBE_MODELS = ("LoLAHE", "LeNetHE", "AlexNetTinyHE", "AlexNetHE")


class ProbeVerificationError(ValueError):
    """Raised when pinned OpenFHE output and CROSS disagree."""


def load_probe_records(lines: Iterable[str]) -> list[Mapping[str, object]]:
    """Load line-delimited JSON emitted by the OpenFHE reference probe."""
    records: list[Mapping[str, object]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProbeVerificationError(
                f"probe line {number} is not valid JSON"
            ) from exc
        if not isinstance(record, dict):
            raise ProbeVerificationError(f"probe line {number} is not an object")
        records.append(record)
    if not records:
        raise ProbeVerificationError("probe output is empty")
    return records


def _check_fields(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    context: str,
) -> None:
    for field, value in expected.items():
        if actual.get(field) != value:
            raise ProbeVerificationError(
                f"{context}: {field} mismatch: "
                f"expected {value!r}, got {actual.get(field)!r}"
            )


def verify_openfhe_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Compare exact OpenFHE 1.5.1 Q/P output with canonical configs."""
    if not records:
        raise ProbeVerificationError("probe output is empty")
    _check_fields(records[0], _OPENFHE_PROBE_HEADER, "probe header")
    by_model: dict[str, Mapping[str, object]] = {}
    for record in records[1:]:
        model = record.get("model")
        if not isinstance(model, str) or model not in _OPENFHE_PROBE_MODELS:
            raise ProbeVerificationError(f"unexpected probe model {model!r}")
        if model in by_model:
            raise ProbeVerificationError(f"duplicate probe record for {model}")
        by_model[model] = record
    missing = set(_OPENFHE_PROBE_MODELS) - set(by_model)
    if missing:
        raise ProbeVerificationError(
            f"probe output is missing model(s): {', '.join(sorted(missing))}"
        )

    for model in _OPENFHE_PROBE_MODELS:
        config = canonical_config(model)
        _check_fields(
            by_model[model],
            {
                "logical_q": config.logical_num_q,
                "num_large_digits": config.dnum,
                "ring_n": config.degree,
                "physical_q": len(config.q_towers),
                "physical_p": len(config.p_towers),
                "q_moduli": list(config.q_towers),
                "p_moduli": list(config.p_towers),
                "openfhe_log_qp": config.openfhe_log_qp,
            },
            model,
        )
    return _OPENFHE_PROBE_MODELS


def generate_grid(
    models: Sequence[str] | None = None,
    securities: Sequence[int] = (128,),
    cost_models: Sequence[CostModel | str] = (CostModel.CLASSICAL,),
) -> tuple[RingConfig, ...]:
    """Generate canonical configs for the requested parameter grid."""
    selected_models = tuple(MODEL_SPECS) if models is None else tuple(models)
    unknown = set(selected_models) - set(MODEL_SPECS)
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(sorted(unknown))}")
    return tuple(
        canonical_config(model, bits, cost)
        for model in selected_models
        for bits in securities
        for cost in cost_models
    )


def _format_table(configs: Sequence[RingConfig]) -> str:
    header = (
        f"{'model':14} {'sec':>3} {'cost':9} {'ring_N':>7} "
        f"{'Q':>2} {'P':>2} {'CD':>2} {'QP bits':>7} {'limit':>5}"
    )
    lines = [header, "-" * len(header)]
    lines.extend(
        f"{config.model:14} {config.security_bits:>3} "
        f"{config.cost_model:9} {config.degree:>7} "
        f"{len(config.q_towers):>2} {len(config.p_towers):>2} "
        f"{config.composite_degree:>2} {config.openfhe_log_qp:>7} "
        f"{config.hes_ceiling:>5}"
        for config in configs
    )
    return "\n".join((*lines, "", f"source: {SECURITY_TABLE_SOURCE}"))


def main(argv: Sequence[str] | None = None) -> tuple[RingConfig, ...]:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and validate canonical CKKS parameters for CROSS demos."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_SPECS),
        help="model to generate (repeatable; default: all)",
    )
    parser.add_argument(
        "--security",
        type=int,
        choices=SUPPORTED_SECURITY_BITS,
        default=128,
    )
    cost = parser.add_mutually_exclusive_group()
    cost.add_argument(
        "--cost-model",
        choices=tuple(item.value for item in CostModel),
        default=CostModel.CLASSICAL.value,
    )
    cost.add_argument(
        "--classical",
        action="store_const",
        dest="cost_model",
        const=CostModel.CLASSICAL.value,
    )
    cost.add_argument(
        "--quantum",
        action="store_const",
        dest="cost_model",
        const=CostModel.QUANTUM.value,
    )
    parser.add_argument("--json", metavar="FILE", help="write validated JSON")
    parser.add_argument(
        "--verify-openfhe",
        metavar="JSONL",
        help="compare a pinned OpenFHE probe output with canonical parameters",
    )
    args = parser.parse_args(argv)

    if args.verify_openfhe:
        selected_models = (
            _OPENFHE_PROBE_MODELS if args.model is None else tuple(args.model)
        )
        pinned_models = (
            len(selected_models) == len(_OPENFHE_PROBE_MODELS)
            and set(selected_models) == set(_OPENFHE_PROBE_MODELS)
        )
        if (
            args.security != 128
            or args.cost_model != CostModel.CLASSICAL.value
            or not pinned_models
        ):
            parser.error(
                "--verify-openfhe is pinned to all models at "
                "128-bit classical security"
            )

    configs = generate_grid(
        args.model,
        securities=(args.security,),
        cost_models=(args.cost_model,),
    )
    print(_format_table(configs))
    if args.json:
        output = Path(args.json)
        output.write_text(
            json.dumps([config.to_dict() for config in configs], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {len(configs)} config(s) to {output}")
    if args.verify_openfhe:
        records = load_probe_records(
            Path(args.verify_openfhe).read_text(encoding="utf-8").splitlines()
        )
        models = verify_openfhe_records(records)
        print("\nverified exact OpenFHE Q/P parameters for " + ", ".join(models))
    return configs


if __name__ == "__main__":
    main()

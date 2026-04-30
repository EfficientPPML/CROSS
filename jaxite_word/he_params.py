"""
HEParameterCache: Shared parameter storage for level-indexed HE operations.

Computes NTT twiddle factors, Barrett contexts, psi/inv_psi arrays once at max level.
Per-level BConv parameters (which cannot be sliced) are computed separately.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp

import bconv
import finite_field
import ntt_mm
import polynomial
import util
import key_gen as kg

Polynomial = polynomial.Polynomial
BarrettContext = finite_field.BarrettContext
NTTCiphertextBarrettContext = ntt_mm.NTTCiphertextBarrettContext
BConvBarrett = bconv.BConvBarrett

jax.config.update('jax_enable_x64', True)


@dataclass
class LevelBConvParams:
  """BConv parameters for a specific level's moduli configuration."""

  level: int
  num_q: int
  bconv: BConvBarrett
  select_tower_index: List[jnp.ndarray]
  non_select_tower_index: List[jnp.ndarray]
  restore_indices: List[jnp.ndarray]
  PInvModq: jnp.ndarray


@dataclass
class LevelPolynomialHelpers:
  """Pre-allocated Polynomial objects for a given level."""

  ct_q: Polynomial  # For Q-only operations (tensor multiply, modmul)
  ct_p: Polynomial  # For P-only operations (approx mod down INTT)
  ct_qp: Polynomial  # For Q+P operations (key-switch result)
  ct_ks_parts: List[Polynomial]  # One per dnum partition


@dataclass
class LevelParams:
  """All parameters for a specific level."""

  num_q: int
  sliced_ntt_q: object  # NTT context (sliced from max-level)
  sliced_barrett_q: object  # Barrett context (sliced from max-level)
  ntt_qp: object  # mod_reduce-only NTT context for Q+P
  barrett_qp: object  # Barrett context (Q concat P)
  bconv_params: LevelBConvParams
  ct_helpers: LevelPolynomialHelpers
  psi_q: jnp.ndarray
  inv_psi_q: jnp.ndarray
  psi_qp: jnp.ndarray
  inv_psi_qp: jnp.ndarray


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
    self.level_params: Dict[int, LevelParams] = {}

    # Attributes populated by initialize() — placeholders use correct types
    # with minimal storage (single-modulus contexts, single-element arrays).
    _init_q_barrett = BarrettContext(moduli=[q_towers[0]])
    _init_q_ntt_params = {'r': r, 'c': c, 'finite_field_context': _init_q_barrett}
    self.barrett_q_max: BarrettContext = _init_q_barrett
    self.ntt_q_max: NTTCiphertextBarrettContext = NTTCiphertextBarrettContext(
        moduli=[q_towers[0]], parameters=_init_q_ntt_params)
    _init_p_barrett = BarrettContext(moduli=[p_towers[0]])
    _init_p_ntt_params = {'r': r, 'c': c, 'finite_field_context': _init_p_barrett}
    self.barrett_p: BarrettContext = _init_p_barrett
    self.ntt_p: NTTCiphertextBarrettContext = NTTCiphertextBarrettContext(
        moduli=[p_towers[0]], parameters=_init_p_ntt_params)
    self.psi_full: jnp.ndarray = jnp.zeros(1, dtype=jnp.uint64)
    self.inv_psi_full: jnp.ndarray = jnp.zeros(1, dtype=jnp.uint64)
    self.eval_key_a_full: jnp.ndarray = jnp.zeros(1, dtype=jnp.uint32)
    self.eval_key_b_full: jnp.ndarray = jnp.zeros(1, dtype=jnp.uint32)
    self.secret_key: list = []
    self.rot_indices: List[int] = []
    self.coef_maps: Dict[int, jnp.ndarray] = {}

  def num_q_at_level(self, level: int) -> int:
    return self.num_q - (self.max_level - level) * self.composite_degree

  def q_moduli_at_level(self, level: int) -> List[int]:
    return self.q_towers[: self.num_q_at_level(level)]

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
        rot_keys: Dict[rot_index -> key dict with "a", "b" arrays]
        coef_maps: Dict[rot_index -> permutation array]
        secret_key: Secret key for per-level rotation key generation
    """
    ring_dim = self.degree
    all_q_moduli = self.q_towers

    # ================================================================
    # 1. Max-level Barrett and NTT context for Q moduli (sliceable)
    # ================================================================
    self.barrett_q_max = BarrettContext(moduli=all_q_moduli)
    ntt_params_q = {
        'r': self.r,
        'c': self.c,
        'finite_field_context': self.barrett_q_max,
    }
    self.ntt_q_max = NTTCiphertextBarrettContext(
        moduli=all_q_moduli, parameters=ntt_params_q, perf_test=self.perf_test
    )

    # P-only Barrett context (fixed across all levels, used in .concat())
    self.barrett_p = BarrettContext(moduli=self.p_towers)

    # P-only NTT context (for approx-mod-down INTT on P-part)
    ntt_params_p = {
        'r': self.r,
        'c': self.c,
        'finite_field_context': self.barrett_p,
    }
    self.ntt_p = NTTCiphertextBarrettContext(
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
          pow(psi, -1, q) for psi, q in zip(all_psi_roots, all_moduli)
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
    # Store secret key and rotation parameters for per-level key generation.
    # Rotation keys depend on the Q tower count (different dnum partitioning
    # at each level), so they must be regenerated per level rather than sliced.
    self.secret_key = secret_key
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
    for level in range(self.max_level + 1):
      self.level_params[level] = self._build_level_params(level)

  def _build_level_params(self, level: int) -> LevelParams:
    """Build all parameters for a specific level."""
    num_q_l = self.num_q_at_level(level)
    q_at_level = self.q_towers[:num_q_l]
    qp_at_level = q_at_level + self.p_towers

    # Sliced Q-only contexts (shared from max-level via array slicing)
    sliced_barrett_q = self.barrett_q_max.slice(num_q_l)
    sliced_ntt_q = self.ntt_q_max.slice(num_q_l, sliced_barrett_q)

    # Q+P context via concatenation: sliced-Q Barrett + fixed-P Barrett.
    # No NTT twiddle factors needed — Q+P is only used for mod_reduce.
    # BarrettContext.ff_ctx returns self, so it can serve directly as ntt_ctx.
    barrett_qp = sliced_barrett_q.concat(self.barrett_p)

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

    # Helper Polynomial objects
    ct_helpers = self._build_ct_helpers(
        level,
        num_q_l,
        q_at_level,
        qp_at_level,
        sliced_ntt_q,
        barrett_qp,
        bconv_params,
    )

    return LevelParams(
        num_q=num_q_l,
        sliced_ntt_q=sliced_ntt_q,
        sliced_barrett_q=sliced_barrett_q,
        ntt_qp=barrett_qp,  # Barrett context serves as ntt_ctx via .ff_ctx property
        barrett_qp=barrett_qp,
        bconv_params=bconv_params,
        ct_helpers=ct_helpers,
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

    # Create BConv and generate controls
    bconv = BConvBarrett(qp_at_level)
    bconv.control_gen(control_indices_list, perf_test=self.perf_test)

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
        bconv=bconv,
        select_tower_index=select_tower_index,
        non_select_tower_index=non_select_tower_index,
        restore_indices=restore_indices,
        PInvModq=PInvModq,
    )

  def _build_ct_helpers(
      self,
      level,
      num_q_l,
      q_at_level,
      qp_at_level,
      sliced_ntt_q,
      barrett_qp,
      bconv_params,
  ):
    """Build pre-allocated Polynomial helper objects for a level."""
    ring_dim = self.degree

    # Q-only ciphertext (for tensor multiply, modmul, NTT)
    ct_q = Polynomial(
        {
            'batch': self.batch,
            'num_elements': 4,
            'degree': ring_dim,
            'num_moduli': num_q_l,
            'precision': 32,
            'degree_layout': self.degree_layout,
        },
        {'moduli': q_at_level, 'ntt_ctx': sliced_ntt_q},
    )

    # P-only ciphertext (for approx mod down INTT) — uses cached P NTT
    ct_p = Polynomial(
        {
            'batch': self.batch,
            'num_elements': 2,
            'degree': ring_dim,
            'num_moduli': self.num_p,
            'precision': 32,
            'degree_layout': self.degree_layout,
        },
        {'moduli': self.p_towers, 'ntt_ctx': self.ntt_p},
    )

    # Q+P ciphertext (for key-switch result mod reduce only).
    # BarrettContext.ff_ctx returns self, so it serves directly as ntt_ctx.
    ct_qp = Polynomial(
        {
            'batch': self.batch,
            'num_elements': 2,
            'degree': ring_dim,
            'num_moduli': num_q_l + self.num_p,
            'precision': 32,
            'degree_layout': self.degree_layout,
        },
        {'moduli': qp_at_level, 'ntt_ctx': barrett_qp},
    )

    # Key-switch part ciphertexts (one per dnum partition)
    ct_ks_parts = []
    numPartQl = len(bconv_params.select_tower_index)
    for part in range(numPartQl):
      target_indices = bconv_params.non_select_tower_index[part].tolist()
      target_moduli = [qp_at_level[i] for i in target_indices]
      target_barrett = BarrettContext(moduli=target_moduli)
      target_ntt_params = {
          'r': self.r,
          'c': self.c,
          'finite_field_context': target_barrett,
      }
      target_ntt = NTTCiphertextBarrettContext(
          moduli=target_moduli,
          parameters=target_ntt_params,
          perf_test=self.perf_test,
      )
      ct_part = Polynomial(
          {
              'batch': self.batch,
              'num_elements': 2,
              'degree': ring_dim,
              'num_moduli': len(target_moduli),
              'precision': 32,
              'degree_layout': self.degree_layout,
          },
          {'moduli': target_moduli, 'ntt_ctx': target_ntt},
      )
      ct_ks_parts.append(ct_part)

    return LevelPolynomialHelpers(
        ct_q=ct_q, ct_p=ct_p, ct_qp=ct_qp, ct_ks_parts=ct_ks_parts
    )

  # ================================================================
  # Getter Methods
  # ================================================================
  def get_psi_q(self, level: int) -> jnp.ndarray:
    return self.level_params[level].psi_q

  def get_inv_psi_q(self, level: int) -> jnp.ndarray:
    return self.level_params[level].inv_psi_q

  def get_psi_qp(self, level: int) -> jnp.ndarray:
    return self.level_params[level].psi_qp

  def get_inv_psi_qp(self, level: int) -> jnp.ndarray:
    return self.level_params[level].inv_psi_qp

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
    """Generate rotation key arrays for the given level.

    Rotation keys depend on the Q tower count at each level (dnum partitioning
    changes), so they are generated per level using the stored secret key.

    Returns (a, b, coefMap) where a and b have shape
    (effective_dnum, *degree_layout, num_Q_at_level + num_P).
    """

    n = self.num_q_at_level(level)
    q_at_level = self.q_towers[:n]
    sk = self.secret_key[:n]  # slice secret key to match Q tower count

    rk = kg.gen_rotation_key(
        sk,
        q_at_level,
        self.p_towers,
        rot_index,
        dnum=self.dnum,
        noise_std=3.190000057220458984375,
        noise_scale=1,
    )
    ek = rk[rot_index]
    a = (
        jnp.array(ek['a'], jnp.uint64)
        .transpose(0, 2, 1)
        .reshape(-1, *self.degree_layout, n + self.num_p)
    )
    b = (
        jnp.array(ek['b'], jnp.uint64)
        .transpose(0, 2, 1)
        .reshape(-1, *self.degree_layout, n + self.num_p)
    )
    return a, b, self.coef_maps[rot_index]

  def get_bconv_params(self, level: int) -> LevelBConvParams:
    return self.level_params[level].bconv_params

  def get_ct_helpers(self, level: int) -> LevelPolynomialHelpers:
    return self.level_params[level].ct_helpers

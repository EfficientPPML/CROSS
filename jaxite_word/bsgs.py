"""Baby-Step Giant-Step (BSGS) homomorphic matrix-vector multiplication.

BSGS decomposes the diagonal-method matvec
    y = sum_{k=0}^{n-1} Rot_k(v) * diag_k(A)
into n1 baby rotations of v (cached) + n2 giant rotations of partial sums:
    y = sum_{j=0}^{n2-1} Rot_{j*n1}( sum_{i=0}^{n1-1} Rot_i(v) * diag'_{j*n1+i}(A) )
where diag'_k is the k-th diagonal pre-rotated by -j*n1 to compensate the
giant-step rotation. With n1 * n2 = n and n1 ~= sqrt(n), total rotations drop
from O(n) to O(sqrt(n)).

Public API:
    BSGSMatVec(ctx, level, n, n1=None, n2=None)
      .encode_matrix(W, pt_scale=None)  # accepts rectangular W, pads to n x n
      .mul(ct_in) -> Polynomial at level-1

One level is consumed by the inner rescale.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from herot import HERot
from matvec import make_ptct_rescale_fn
from polynomial import Polynomial


def compute_bsgs_params(
    n: int,
    num_diagonals: Optional[int] = None,
    bsgs_ratio: float = 2.0,
) -> Tuple[int, int]:
  """Factor n = n1 * n2 picking n1 adaptively.

  - If num_diagonals is None: n1 ≈ sqrt(n) (the dense default).
  - If num_diagonals is given: n1 ≈ sqrt(num_diagonals * bsgs_ratio), constrained
    to divide n. This matches ORION/Lattigo's BSGS rule, which trades baby vs
    giant rotation cost based on the actual diagonal count.
  """
  if n <= 0:
    raise ValueError(f'n must be positive, got {n}')
  if num_diagonals is None:
    n1 = int(math.isqrt(n))
    while n1 > 1 and n % n1 != 0:
      n1 -= 1
    return max(n1, 1), n // max(n1, 1)
  # Adaptive: pick the divisor of n closest to sqrt(num_diagonals * bsgs_ratio).
  target = max(1, int(round(math.sqrt(num_diagonals * bsgs_ratio))))
  divisors = [d for d in range(1, n + 1) if n % d == 0]
  n1 = min(divisors, key=lambda d: (abs(d - target), d))
  return n1, n // n1


def extract_diagonals(matrix: np.ndarray) -> List[np.ndarray]:
  """diag_k[i] = matrix[i, (i + k) % n]."""
  n = matrix.shape[0]
  return [
      np.array([matrix[i, (i + k) % n] for i in range(n)]) for k in range(n)
  ]


def pre_rotate_diagonal(diagonal: np.ndarray, j: int, n1: int) -> np.ndarray:
  """Compensate giant-step rotation by pre-rotating the diagonal.

  For CKKS's left-rotation convention Rot_s(x)[i] = x[(i + s) mod n], the
  identity  diag_k * Rot_k(v) = Rot_{j*n1}(diag'_k * Rot_i(v))  where k = j*n1
  + i requires  diag'_k = Rot_{-j*n1}(diag_k)  = shift diag RIGHT by j*n1.
  """
  return np.roll(diagonal, (j * n1) % len(diagonal))


def pad_matrix_square(a: np.ndarray, n: int) -> np.ndarray:
  """Zero-pad an (m, k) matrix into an (n, n) matrix (m, k <= n)."""
  m, k = a.shape
  if m > n or k > n:
    raise ValueError(f'Matrix shape ({m},{k}) does not fit in ({n},{n}).')
  padded = np.zeros((n, n), dtype=a.dtype)
  padded[:m, :k] = a
  return padded


class BSGSMatVec:
  """Homomorphic matrix-vector product using BSGS.

  Attributes:
      ctx: CKKSContext with program_initialization already called AND with
          baby/giant rotation keys registered (indices 1..n1-1 at level, and
          n1, 2*n1, ..., (n2-1)*n1 at level-1).
      level: input ciphertext level. Output ciphertext at level-1.
      n: logical slot dimension the matrix acts on.
      n1, n2: BSGS factors, n1 * n2 == n.
  """

  def __init__(
      self,
      ctx,
      level: int,
      n: int,
      n1: Optional[int] = None,
      n2: Optional[int] = None,
      bsgs_ratio: Optional[float] = None,
  ):
    self.ctx = ctx
    self.level = level
    self.n = n
    if n1 is None or n2 is None:
      n1, n2 = compute_bsgs_params(n)
    if n1 * n2 != n:
      raise ValueError(f'n1 * n2 ({n1}*{n2}) != n ({n})')
    self.n1, self.n2 = n1, n2
    # If bsgs_ratio is not None, encode_matrix will RE-PICK n1, n2 adaptively
    # based on the actual non-zero diagonal count (matches ORION/Lattigo).
    self.bsgs_ratio = bsgs_ratio

    cache = ctx._param_cache
    self.r, self.c = cache.r, cache.c
    self.num_q_in = cache.num_q_at_level(level)
    self.num_q_out = cache.num_q_at_level(level - 1)
    self.q_towers_in = list(ctx.q_towers[: self.num_q_in])
    self.q_towers_out = list(ctx.q_towers[: self.num_q_out])

    self._groups: Optional[List[Tuple]] = None
    self._mul_fn = None
    self._rotation_keys_ready = False

  # --------------------------------------------------------------
  # Offline: diagonals + encoding
  # --------------------------------------------------------------
  def encode_matrix(
      self, matrix: np.ndarray, pt_scale: Optional[int] = None,
      bsgs_ratio: Optional[float] = None,
  ) -> None:
    """Sparse-diagonal BSGS encoding with three optimizations:

    1. Whole-giant-step skip — drop giant steps j whose entire baby slot
       is zero.
    2. Per-diagonal skip — within an active giant step, drop individual
       zero diagonals; group active j's by their non-zero baby count so
       `mul_fn` can run one (num_j, inner_len) scan per group.
    3. **Active-baby precomputation** — only pre-rotate the input by
       baby indices that some active diagonal actually uses. For sparse
       matrices (e.g. conv1 Toeplitz) this collapses the n1-1 baby rotations
       down to just len(active_baby) - 1.

    If `adaptive=True`, n1 is also re-chosen as
    `argmin_{n1|n} |n1 - sqrt(num_active * bsgs_ratio)|` after the active
    diagonals are known — matching ORION/Lattigo's adaptive BSGS rule.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
      raise ValueError(f'matrix must be 2-D, got shape {matrix.shape}')
    padded = pad_matrix_square(matrix, self.n)
    diagonals = extract_diagonals(padded)

    # ---- Identify all non-zero diagonal indices (independent of n1) ----
    active_k = [k for k in range(self.n)
                if np.any(diagonals[k] != 0)]
    if not active_k:
      raise ValueError('All diagonals are zero; matrix is trivially zero.')

    # ---- Optionally re-choose n1, n2 adaptively for sparse matrices ----
    # Adaptive trigger: caller passed bsgs_ratio (here or in __init__).
    eff_bsgs_ratio = (
        bsgs_ratio if bsgs_ratio is not None else self.bsgs_ratio
    )
    if eff_bsgs_ratio is not None:
      n1, n2 = compute_bsgs_params(
          self.n, num_diagonals=len(active_k), bsgs_ratio=eff_bsgs_ratio)
      self.n1, self.n2 = n1, n2

    # ---- Group active diagonals by giant step (post-adaptive n1) ----
    active_i_per_j: Dict[int, List[int]] = defaultdict(list)
    for k in active_k:
      j = k // self.n1
      i = k - j * self.n1
      active_i_per_j[j].append(i)
    for j in active_i_per_j:
      active_i_per_j[j] = sorted(active_i_per_j[j])

    self._active_j = sorted(active_i_per_j.keys())
    self._active_i_per_j = dict(active_i_per_j)

    # ---- Active baby indices (include 0 to host the unrotated input) ----
    active_baby_set = {0}
    for j in self._active_j:
      active_baby_set.update(active_i_per_j[j])
    self._active_baby_indices = sorted(active_baby_set)
    # Map baby index -> position in compact `all_baby` array.
    self._baby_local_index = {
        b: pos for pos, b in enumerate(self._active_baby_indices)
    }

    # ---- Group giant steps by non-zero baby count ----
    groups_by_len: Dict[int, List[int]] = defaultdict(list)
    for j in self._active_j:
      groups_by_len[len(active_i_per_j[j])].append(j)

    import ckks_ctx as ckks_ctx_mod  # lazy import to avoid cycle
    params = dict(self.ctx.parameters)
    params['q_towers'] = self.q_towers_in
    if pt_scale is None:
      pt_scale = self.q_towers_in[-1]
    params['scaling_factor'] = pt_scale
    encode_ctx = ckks_ctx_mod.CKKSContext(params)
    num_slots = self.ctx.num_slots

    def _encode_diag(arr: np.ndarray) -> jnp.ndarray:
      vals = [complex(float(v)) for v in arr]
      if len(vals) < num_slots:
        vals = vals + [0 + 0j] * (num_slots - len(vals))
      elif len(vals) > num_slots:
        vals = vals[:num_slots]
      pt = encode_ctx.encode(vals)
      return (
          pt.polynomial[0, 0]
          .reshape(self.r, self.c, self.num_q_in)
          .astype(jnp.uint32)
      )

    # Build per-group stacked arrays. baby_idx now stores LOCAL positions
    # (indices into self._active_baby_indices), not the raw n1-mod indices.
    self._groups: List[Tuple] = []
    for L in sorted(groups_by_len.keys(), reverse=True):
      j_list = groups_by_len[L]
      group_pts = []
      group_idx = []
      for j in j_list:
        pts_j = []
        idx_j = []
        for i in active_i_per_j[j]:
          k = j * self.n1 + i
          rotated = pre_rotate_diagonal(diagonals[k], j, self.n1)
          pts_j.append(_encode_diag(rotated))
          idx_j.append(self._baby_local_index[i])    # LOCAL position
        group_pts.append(jnp.stack(pts_j, axis=0))
        group_idx.append(jnp.asarray(idx_j, dtype=jnp.int32))
      stacked_pts = jnp.stack(group_pts, axis=0)
      stacked_idx = jnp.stack(group_idx, axis=0)
      j_array = jnp.asarray(j_list, dtype=jnp.int32)
      self._groups.append(
          (L, j_list, stacked_pts, stacked_idx, j_array)
      )

    total_nz = sum(len(v) for v in active_i_per_j.values())
    n_active_baby = max(0, len(self._active_baby_indices) - 1)  # excl. identity
    print(
        f'[bsgs] n1={self.n1}, n2={self.n2}; '
        f'{total_nz} non-zero diagonals over {len(self._active_j)} '
        f'active giant steps in {len(self._groups)} group(s); '
        f'active baby pre-rotations = {n_active_baby} '
        f'(was n1-1={self.n1 - 1} before active-baby).'
    )

    self._prepare_rotation_keys()
    self._build_mul_fn()

  # --------------------------------------------------------------
  # Rotation key staging
  # --------------------------------------------------------------
  def _fetch_rot(self, level, idx):
    """Return (eval_a, eval_b, coef_map) for one (level, idx) from ctx."""
    op = self.ctx.he_rot[level, idx]
    inst = op._herot
    return (
        jnp.asarray(inst.evalkey_a_vector, dtype=jnp.uint64),
        jnp.asarray(inst.evalkey_b_vector, dtype=jnp.uint64),
        jnp.asarray(inst.coef_map, dtype=jnp.int32),
    )

  def _prepare_rotation_keys(self):
    n1, n2 = self.n1, self.n2

    # Active-baby precomputation: only the non-zero entries of
    # self._active_baby_indices (i.e. excluding the identity slot 0).
    baby_indices = [b for b in self._active_baby_indices if b != 0]
    baby_a: List[jnp.ndarray] = []
    baby_b: List[jnp.ndarray] = []
    baby_cm: List[jnp.ndarray] = []
    for i in baby_indices:
      ea, eb, cm = self._fetch_rot(self.level, i)
      baby_a.append(ea)
      baby_b.append(eb)
      baby_cm.append(cm)

    # Capture a sample HERot at input level for baby rotate pure fn.
    self._sample_baby_herot = (
        self.ctx.he_rot[self.level, baby_indices[0]]._herot
        if baby_indices else None
    )

    # Giant keys are per-GROUP. j==0 uses a dummy key because the scan
    # body skips rotation when j==0. Pick a dummy that's guaranteed
    # registered (smallest active non-zero giant index, else n1).
    nonzero_active_j = [j for j in self._active_j if j > 0]
    dummy_giant_idx = (nonzero_active_j[0] * n1) if nonzero_active_j else n1
    self._sample_giant_herot = self.ctx.he_rot[
        self.level - 1, dummy_giant_idx
    ]._herot

    updated_groups: List[Tuple] = []
    for L, j_list, stacked_pts, stacked_idx, j_array in self._groups:
      ga: List[jnp.ndarray] = []
      gb: List[jnp.ndarray] = []
      gcm: List[jnp.ndarray] = []
      for j in j_list:
        idx = j * n1 if j > 0 else dummy_giant_idx
        ea, eb, cm = self._fetch_rot(self.level - 1, idx)
        ga.append(ea)
        gb.append(eb)
        gcm.append(cm)
      updated_groups.append((
          L,
          j_list,
          stacked_pts,
          stacked_idx,
          j_array,
          jnp.stack(ga, axis=0),
          jnp.stack(gb, axis=0),
          jnp.stack(gcm, axis=0),
      ))
    self._groups = updated_groups

    self._baby_eval_a = jnp.stack(baby_a, axis=0) if baby_a else None
    self._baby_eval_b = jnp.stack(baby_b, axis=0) if baby_b else None
    self._baby_cm = jnp.stack(baby_cm, axis=0) if baby_cm else None
    self._rotation_keys_ready = True

  # --------------------------------------------------------------
  # JIT'd scan-based mul
  # --------------------------------------------------------------
  def _build_mul_fn(self):
    cache = self.ctx._param_cache
    n1, n2 = self.n1, self.n2
    r, c = self.r, self.c
    # Pull batch from the cached HE parameters so BSGS is consistent with
    # the rest of the pipeline (HEMul / HERescale / HERot were built using
    # the same `batch` that program_initialization passed in).
    batch = getattr(cache, "batch", 1)
    m_in = self.num_q_in
    m_out = self.num_q_out
    moduli_out = jnp.asarray(self.q_towers_out, dtype=jnp.uint32)

    pure_rot_baby = (
        HERot.make_rotate_fn(self._sample_baby_herot)
        if self._sample_baby_herot is not None
        else None
    )
    pure_rot_giant = HERot.make_rotate_fn(self._sample_giant_herot)

    ptct_op = self.ctx.ptct_mul[self.level]._ptct
    rescale_op = self.ctx.he_rescale[self.level, self.level - 1]._he_rescale
    # Note: matvec.py's make_ptct_rescale_fn used to need
    # `rescale_op.power_of_inv_psi_all` — that's now a vestigial no-op
    # (psi is baked into the negacyclic NTT). HERescale.control_gen no
    # longer creates it, and matvec.py no longer references it.
    pure_ptct_rescale = make_ptct_rescale_fn(ptct_op, rescale_op)

    def _modadd(a, b):
      s = a.astype(jnp.uint64) + b.astype(jnp.uint64)
      return jnp.where(s >= moduli_out, s - moduli_out, s).astype(jnp.uint32)

    def baby_body(carry, xs):
      ea, eb, cm = xs
      rotated = pure_rot_baby(carry, ea, eb, cm)
      return carry, rotated.reshape(batch, 2, r, c, m_in)

    # Closure-capture the static group layout so the Python loop over groups
    # unrolls at JIT trace time into multiple (num_j, L) scan pairs.
    groups_static = self._groups  # (L, j_list, pts, baby_idx, j_array, gs_ea, gs_eb, gs_cm)

    has_active_baby = self._baby_eval_a is not None

    def mul_fn(ct_data, baby_ea, baby_eb, baby_cm):
      # ---- Phase 1: pre-rotate input by ACTIVE baby indices only ----
      if has_active_baby:
        _, baby_rotated = jax.lax.scan(
            baby_body, ct_data, (baby_ea, baby_eb, baby_cm)
        )
        # all_baby ordering matches self._active_baby_indices:
        # position 0 = ct_data (identity / baby index 0)
        # positions 1..N = ct_data rotated by each non-zero active baby idx
        all_baby = jnp.concatenate([ct_data[None], baby_rotated], axis=0)
      else:
        all_baby = ct_data[None]
      # all_baby: (len(active_baby_indices), batch, 2, r, c, m_in)

      zero_inner = jnp.zeros((batch, 2, r, c, m_out), dtype=jnp.uint32)
      global_acc = jnp.zeros((batch, 2, r, c, m_out), dtype=jnp.uint32)

      # One nested scan pair per group — num groups is typically 1-4.
      for (L, _j_list, stacked_pts, stacked_idx, j_array,
           gs_ea, gs_eb, gs_cm) in groups_static:

        def inner_body(inner_sum, xs):
          i, pt = xs
          baby_ct = jnp.take(all_baby, i, axis=0)
          product = pure_ptct_rescale(baby_ct, pt)
          return _modadd(inner_sum, product), None

        def giant_body(acc, xs):
          j, pts_j, idx_j, gea, geb, gcm = xs
          inner_sum, _ = jax.lax.scan(
              inner_body, zero_inner, (idx_j, pts_j)
          )
          rotated = pure_rot_giant(inner_sum, gea, geb, gcm)
          rotated_5d = rotated.reshape(batch, 2, r, c, m_out)
          contribution = jnp.where(j > 0, rotated_5d, inner_sum)
          return _modadd(acc, contribution), None

        global_acc, _ = jax.lax.scan(
            giant_body,
            global_acc,
            (j_array, stacked_pts, stacked_idx, gs_ea, gs_eb, gs_cm),
        )

      return global_acc

    self._mul_fn = jax.jit(mul_fn)

  # --------------------------------------------------------------
  # Runtime
  # --------------------------------------------------------------
  def mul(self, ct_in: Polynomial) -> Polynomial:
    """Compute A . ct_in. Returns ciphertext at level-1."""
    if self._groups is None or self._mul_fn is None:
      raise RuntimeError(
          'Matrix not encoded. Call encode_matrix(...) before mul().'
      )

    ct_data = ct_in.polynomial
    if ct_data.ndim != 5:
      ct_data = ct_data.reshape(1, 2, self.r, self.c, self.num_q_in)

    # When no baby rotations are needed, pass empty arrays of compatible
    # shape (the JIT'd `mul_fn` skips the baby-scan branch via has_active_baby
    # captured at trace time).
    if self._baby_eval_a is not None:
      baby_ea, baby_eb, baby_cm = (
          self._baby_eval_a, self._baby_eval_b, self._baby_cm)
    else:
      baby_ea = jnp.zeros((0,), dtype=jnp.uint64)
      baby_eb = jnp.zeros((0,), dtype=jnp.uint64)
      baby_cm = jnp.zeros((0,), dtype=jnp.int32)

    result_data = self._mul_fn(
        ct_data, baby_ea, baby_eb, baby_cm,
    )

    out = Polynomial(
        {
            'batch': 1,
            'num_elements': 2,
            'degree': self.r * self.c,
            'num_moduli': self.num_q_out,
            'precision': 32,
            'degree_layout': (self.r, self.c),
        },
        {'moduli': self.q_towers_out},
    )
    out.polynomial = result_data
    return out

  # --------------------------------------------------------------
  # Persistence helpers (skip the expensive encode_matrix on reload)
  # --------------------------------------------------------------
  def serializable_state(self) -> dict:
    """Return a pickleable dict capturing all encoded state.

    The expensive part of `encode_matrix` is the 1024 (per-FC) plaintext
    encodings. By snapshotting the resulting `groups` arrays + active_j /
    active_i_per_j, a future LoLAHE instance can skip re-encoding entirely.
    """
    if self._groups is None:
      raise RuntimeError('No state to serialize: encode_matrix was not run.')
    return {
        'level': self.level,
        'n': self.n,
        'n1': self.n1,
        'n2': self.n2,
        'active_j': list(self._active_j),
        'active_i_per_j': {
            int(j): list(self._active_i_per_j[j]) for j in self._active_j
        },
        'active_baby_indices': list(self._active_baby_indices),
        # Each group entry is a tuple — store the JAX arrays & Python parts.
        'groups': [
            {
                'L': L,
                'j_list': list(j_list),
                'stacked_pts': np.asarray(stacked_pts),
                'stacked_idx': np.asarray(stacked_idx),
            }
            for (L, j_list, stacked_pts, stacked_idx, _ja, _ge_a, _ge_b, _ge_cm)
            in self._groups
        ],
    }

  def load_serialized_state(self, state: dict) -> None:
    """Reverse of `serializable_state`. Reconstructs `_groups` (with rotation
    keys re-fetched from ctx) and rebuilds the JIT'd `mul_fn`.

    Assumes `__init__` has already configured ctx, level, and shapes. Only
    skips the slow plaintext-encoding step.
    """
    self.n1 = int(state.get('n1', self.n1))
    self.n2 = int(state.get('n2', self.n2))
    self._active_j = list(state['active_j'])
    self._active_i_per_j = {int(j): list(v) for j, v in state['active_i_per_j'].items()}
    self._active_baby_indices = list(
        state.get('active_baby_indices',
                  sorted({0} | {i for v in self._active_i_per_j.values() for i in v}))
    )
    self._baby_local_index = {
        b: pos for pos, b in enumerate(self._active_baby_indices)
    }
    rebuilt_groups: List[Tuple] = []
    for grp in state['groups']:
      L = int(grp['L'])
      j_list = list(grp['j_list'])
      stacked_pts = jnp.asarray(grp['stacked_pts'], dtype=jnp.uint32)
      stacked_idx = jnp.asarray(grp['stacked_idx'], dtype=jnp.int32)
      j_array = jnp.asarray(j_list, dtype=jnp.int32)
      rebuilt_groups.append(
          (L, j_list, stacked_pts, stacked_idx, j_array)
      )
    self._groups = rebuilt_groups
    self._prepare_rotation_keys()
    self._build_mul_fn()

  # --------------------------------------------------------------
  # Convenience: list the rotation indices required for program_init
  # --------------------------------------------------------------
  @staticmethod
  def required_rotation_indices(n: int, n1: Optional[int] = None,
                                n2: Optional[int] = None) -> List[int]:
    """Return the rotation indices that must be pre-registered.

    Baby indices are at the input level; giant at the output level. This
    helper returns the UNION (the caller merges with other required indices
    and passes to program_initialization's total_rotation_indices).
    """
    if n1 is None or n2 is None:
      n1, n2 = compute_bsgs_params(n)
    baby = list(range(1, n1))
    giant = [j * n1 for j in range(1, n2)]
    return sorted(set(baby + giant))

"""Baby-step giant-step homomorphic matrix-vector multiplication.

External users configure one evaluator with ``preprocess`` and run the matrix-
vector multiplication with ``matvec``. Dense matrices, sparse diagonal maps,
and matrix-free sources are all preprocessing inputs; ``memory_bounded=True``
defers diagonal encoding by giant step. Raw-array hooks stay private.
"""

import math
import sys
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral, Real
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


if __name__ == 'jaxite_word.bsgs':
  sys.modules.setdefault('bsgs', sys.modules[__name__])
elif __name__ == 'bsgs':
  sys.modules.setdefault('jaxite_word.bsgs', sys.modules[__name__])

import finite_field
from polynomial import Polynomial


def compute_bsgs_params(
    n: int,
    num_diagonals: Optional[int] = None,
    bsgs_ratio: float = 2.0,
) -> Tuple[int, int]:
  """Return factors ``n1 * n2 == n`` for a dense or sparse workload."""
  if isinstance(n, bool) or not isinstance(n, Integral) or n <= 0:
    raise ValueError(f'n must be a positive int, got {n!r}.')
  if num_diagonals is not None and (
      isinstance(num_diagonals, bool)
      or not isinstance(num_diagonals, Integral)
      or num_diagonals < 0
  ):
    raise ValueError('num_diagonals must be a non-negative int.')
  if isinstance(bsgs_ratio, bool) or not isinstance(bsgs_ratio, Real):
    raise TypeError('bsgs_ratio must be a real number.')
  if not math.isfinite(float(bsgs_ratio)) or bsgs_ratio <= 0:
    raise ValueError('bsgs_ratio must be finite and positive.')
  target = math.isqrt(n) if num_diagonals is None else max(
      1, round(math.sqrt(num_diagonals * float(bsgs_ratio)))
  )
  divisors = []
  for divisor in range(1, math.isqrt(n) + 1):
    if n % divisor == 0:
      divisors.extend((divisor, n // divisor))
  n1 = min(set(divisors), key=lambda divisor: (abs(divisor - target), divisor))
  return n1, n // n1


def required_rotation_indices(
    n: int,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
) -> List[int]:
  """Return rotation indices needed before BSGS preprocessing."""
  if n1 is None or n2 is None:
    n1, n2 = compute_bsgs_params(n)
  if n1 * n2 != n:
    raise ValueError(f'n1 * n2 ({n1}*{n2}) != n ({n})')
  giant = [j * n1 for j in range(1, n2)] or [n1]
  return sorted(set(range(1, n1)) | set(giant))


def pre_rotate_diagonal(diagonal: np.ndarray, j: int, n1: int) -> np.ndarray:
  """Pre-rotate one diagonal to compensate its later giant rotation."""
  return np.roll(diagonal, (j * n1) % len(diagonal))


class _BSGSMatVecAtLevel:
  """Context-owned BSGS evaluator; obtain it through ``ctx.bsgs_matvec``."""

  def __init__(
      self,
      ctx,
      level: int,
      n: int,
      n1: Optional[int] = None,
      n2: Optional[int] = None,
      bsgs_ratio: Optional[float] = None,
  ):
    cache = ctx._param_cache
    if isinstance(level, bool) or not isinstance(level, int):
      raise TypeError('ctx.bsgs_matvec level must be an int.')
    if not 1 <= level <= cache.max_level:
      raise ValueError(
          f'ctx.bsgs_matvec input level {level} out of range '
          f'[1, {cache.max_level}].'
      )
    if n1 is None or n2 is None:
      n1, n2 = compute_bsgs_params(n)
    if n1 * n2 != n:
      raise ValueError(f'n1 * n2 ({n1}*{n2}) != n ({n})')
    if n > ctx.num_slots:
      raise ValueError(f'BSGS dimension {n} exceeds {ctx.num_slots} slots.')
    self.ctx, self.level, self.n = ctx, level, n
    self.n1, self.n2, self.bsgs_ratio = n1, n2, bsgs_ratio
    self.r, self.c = cache.r, cache.c
    self.num_q_in = cache.num_q_at_level(level)
    self.num_q_out = cache.num_q_at_level(level - 1)
    self.q_towers_in = list(ctx.q_towers[: self.num_q_in])
    self.q_towers_out = list(ctx.q_towers[: self.num_q_out])
    self._groups = None
    self._matvec_fn = None
    self._jit_matvec_fn = None
    self._baby_fn = None
    self._giant_step_fn = None
    self._add_out_fn = None
    self._baby_eval_a = None
    self._baby_eval_b = None
    self._baby_coefficient_map = None
    self._memory_bounded = False
    self._diagonal_map = None
    self._diagonal_loader = None
    self._active_i_per_j = None
    self._encoder = None
    self._scaling_error = None
    self._n_jobs = 1
    self._pt_scale = None
    self._encoded_pt_scale = None
    self._encode_floor = None
    self._out_template = None

  def preprocess(
      self,
      matrix,
      pt_scale: Optional[float] = None,
      bsgs_ratio: Optional[float] = None,
      n_jobs: int = 1,
      memory_bounded: bool = False,
      active_diagonal_indices: Optional[tuple[int, ...]] = None,
  ) -> None:
    """Prepare a dense, sparse-diagonal, or matrix-free workload."""
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, Integral)
        or n_jobs < 1
    ):
      raise ValueError('n_jobs must be a positive int.')
    if not isinstance(memory_bounded, bool):
      raise TypeError('memory_bounded must be a bool.')
    if self._groups is not None:
      self._release()
    n_jobs = int(n_jobs)

    # Normalize every concrete workload into its non-zero diagonal map.
    loader = None
    if isinstance(matrix, Mapping):
      diagonal_map = dict(matrix)
    elif callable(getattr(matrix, 'materialize_diagonals', None)):
      if getattr(matrix, 'dimension', None) != self.n:
        raise ValueError(
            f'matrix-free source dimension {getattr(matrix, "dimension", None)!r} '
            f'must equal BSGS dimension {self.n}.'
        )
      if not memory_bounded:
        raise ValueError(
            'matrix-free preprocessing requires memory_bounded=True.'
        )
      diagonal_map = None
      loader = matrix.materialize_diagonals
    else:
      dense = np.asarray(matrix)
      if dense.ndim != 2:
        raise ValueError(f'matrix must be 2-D, got shape {dense.shape}.')
      rows, columns = dense.shape
      if rows > self.n or columns > self.n:
        raise ValueError(
            f'Matrix shape ({rows},{columns}) does not fit in '
            f'({self.n},{self.n}).'
        )
      padded = np.zeros((self.n, self.n), dtype=dense.dtype)
      padded[:rows, :columns] = dense
      diagonal_map = {
          k: np.asarray(
              [padded[i, (i + k) % self.n] for i in range(self.n)]
          )
          for k in range(self.n)
      }
      diagonal_map = {
          k: diagonal for k, diagonal in diagonal_map.items()
          if np.any(diagonal)
      }

    # Validate concrete diagonals before any keys or encoder state are retained.
    if diagonal_map is not None:
      normalized = {}
      for key, diagonal in diagonal_map.items():
        if (
            isinstance(key, bool)
            or not isinstance(key, Integral)
            or not 0 <= int(key) < self.n
        ):
          raise ValueError(
              f'diagonal key {key!r} must be an int in [0, {self.n}).'
          )
        if not isinstance(diagonal, np.ndarray) or diagonal.shape != (self.n,):
          raise ValueError(
              f'diagonal {key} must be a 1-D np.ndarray of length {self.n}; '
              f'got {type(diagonal).__name__} with shape '
              f'{getattr(diagonal, "shape", None)}.'
          )
        if diagonal.dtype.kind == 'c':
          raise TypeError(f'diagonal {key} values must be real.')
        if diagonal.dtype.kind not in 'biuf':
          raise TypeError(f'diagonal {key} values must be numeric.')
        if not np.all(np.isfinite(diagonal)):
          raise ValueError(f'diagonal {key} contains non-finite values.')
        normalized[int(key)] = diagonal
      diagonal_map = normalized

    # Resolve the plaintext scale and reject coefficients it cannot represent.
    cache = self.ctx._param_cache
    if pt_scale is None:
      pt_scale = (
          cache.scaling_factor_recursive(self.level)
          if cache.composite_degree >= 2
          else self.q_towers_in[-1]
      )
    if isinstance(pt_scale, bool) or not isinstance(pt_scale, Real):
      raise TypeError('pt_scale must be a real number.')
    pt_scale_float = float(pt_scale)
    if not math.isfinite(pt_scale_float) or pt_scale_float <= 0:
      raise ValueError('pt_scale must be finite and positive.')
    encode_floor = 0.5 / pt_scale_float
    if diagonal_map is None:
      planned = tuple(active_diagonal_indices or ())
      if not planned:
        raise ValueError(
            'matrix-free preprocessing requires planned active diagonal '
            'indices.'
        )
      active = list(planned)
      nonzero = list(planned)
    else:
      maxima = {
          key: float(np.max(np.abs(diagonal))) if diagonal.size else 0.0
          for key, diagonal in diagonal_map.items()
      }
      nonzero = [key for key in sorted(maxima) if maxima[key] > 0.0]
      active = [key for key in nonzero if maxima[key] >= encode_floor]
      dropped = [key for key in nonzero if maxima[key] < encode_floor]
      if dropped:
        raise ValueError(
            f'{len(dropped)} of {len(nonzero)} non-zero MatVec diagonals are '
            f'below the plaintext encode floor {encode_floor:.3e} at '
            f'pt_scale={pt_scale}; pick a larger scale.'
        )
    if not nonzero:
      raise ValueError('All diagonals are zero; matrix is trivially zero.')
    if not active:
      raise ValueError(
          f'All {len(nonzero)} non-zero diagonals are sub-resolution at '
          f'pt_scale={pt_scale}; pick a larger scale.'
      )
    active_tuple = tuple(active)
    if (
        tuple(sorted(set(active_tuple))) != active_tuple
        or any(
            isinstance(key, bool)
            or not isinstance(key, Integral)
            or not 0 <= int(key) < self.n
            for key in active_tuple
        )
    ):
      raise ValueError(
          f'active diagonal indices must be sorted unique ints in '
          f'[0, {self.n}).'
      )
    if active_diagonal_indices is not None:
      if tuple(active_diagonal_indices) != active_tuple:
        raise ValueError(
            'encoded active diagonals do not match the Mapping schedule: '
            f'planned={tuple(active_diagonal_indices)}, actual={active_tuple}.'
        )
    else:
      ratio = bsgs_ratio if bsgs_ratio is not None else self.bsgs_ratio
      if ratio is not None:
        self.n1, self.n2 = compute_bsgs_params(
            self.n, len(active), ratio
        )

    # Group diagonals by giant step and compact the baby rotations they use.
    active_i_per_j = defaultdict(list)
    for key in active:
      j, i = divmod(key, self.n1)
      active_i_per_j[j].append(i)
    active_i_per_j = {
        j: sorted(indices) for j, indices in active_i_per_j.items()
    }
    active_j = sorted(active_i_per_j)
    active_baby = sorted(
        {0} | {i for indices in active_i_per_j.values() for i in indices}
    )
    baby_position = {index: pos for pos, index in enumerate(active_baby)}
    groups_by_length = defaultdict(list)
    for j in active_j:
      groups_by_length[len(active_i_per_j[j])].append(j)

    # Create the CKKS plaintext encoder used now or one giant step at a time.
    import ckks_ctx as ckks_ctx_mod  # Avoid the module import cycle.
    encode_parameters = dict(self.ctx.parameters)
    encode_parameters['q_towers'] = self.q_towers_in
    encode_parameters['scaling_factor'] = pt_scale
    encoder = ckks_ctx_mod.CKKSContext(encode_parameters)
    scaling_error = ckks_ctx_mod.ScalingFactorTooSmall
    encoded = {}
    if not memory_bounded:
      items = [
          (
              j * self.n1 + i,
              pre_rotate_diagonal(
                  diagonal_map[j * self.n1 + i], j, self.n1
              ),
          )
          for j in active_j
          for i in active_i_per_j[j]
      ]
      slots = [
          [complex(float(value), 0.0) for value in diagonal]
          + [0j] * (self.ctx.num_slots - self.n)
          for _, diagonal in items
      ]
      if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
          futures = [executor.submit(encoder.encode, values) for values in slots]
          for (key, _), future in zip(items, futures, strict=True):
            try:
              plaintext = future.result()
            except scaling_error as error:
              raise ValueError(
                  f'MatVec diagonal {key} underflowed CKKS encode at '
                  f'pt_scale={pt_scale}; pick a larger scale.'
              ) from error
            encoded[key] = np.asarray(
                plaintext.polynomial[0, 0], dtype=np.uint32
            ).reshape(self.r, self.c, self.num_q_in)
      else:
        for (key, _), values in zip(items, slots, strict=True):
          try:
            plaintext = encoder.encode(values)
          except scaling_error as error:
            raise ValueError(
                f'MatVec diagonal {key} underflowed CKKS encode at '
                f'pt_scale={pt_scale}; pick a larger scale.'
            ) from error
          encoded[key] = np.asarray(
              plaintext.polynomial[0, 0], dtype=np.uint32
          ).reshape(self.r, self.c, self.num_q_in)

    # Stack eager plaintexts and record the indices needed by both runtimes.
    groups = []
    for length in sorted(groups_by_length, reverse=True):
      j_list = groups_by_length[length]
      points = None if memory_bounded else jnp.asarray(np.stack([
          np.stack([
              encoded.pop(j * self.n1 + i)
              for i in active_i_per_j[j]
          ])
          for j in j_list
      ]))
      indices = jnp.asarray(np.stack([
          np.asarray(
              [baby_position[i] for i in active_i_per_j[j]],
              dtype=np.int32,
          )
          for j in j_list
      ]))
      groups.append((
          length,
          j_list,
          points,
          indices,
          jnp.asarray(j_list, dtype=jnp.int32),
      ))

    # Stage compact rotation-key arrays and discard their per-index facades.
    baby_a, baby_b, baby_map = [], [], []
    baby_indices = [index for index in active_baby if index]
    for index in baby_indices:
      eval_a, eval_b, coefficient_map = self.ctx.he_rot[
          self.level, index
      ]._rotation_state()
      baby_a.append(jnp.asarray(eval_a, dtype=jnp.uint32))
      baby_b.append(jnp.asarray(eval_b, dtype=jnp.uint32))
      baby_map.append(jnp.asarray(coefficient_map, dtype=jnp.int32))
    nonzero_j = [j for j in active_j if j]
    dummy_giant = nonzero_j[0] * self.n1 if nonzero_j else self.n1
    staged_groups = []
    for length, j_list, points, indices, j_array in groups:
      giant_a, giant_b, giant_map = [], [], []
      for j in j_list:
        index = j * self.n1 if j else dummy_giant
        eval_a, eval_b, coefficient_map = self.ctx.he_rot[
            self.level - 1, index
        ]._rotation_state()
        giant_a.append(jnp.asarray(eval_a, dtype=jnp.uint32))
        giant_b.append(jnp.asarray(eval_b, dtype=jnp.uint32))
        giant_map.append(jnp.asarray(coefficient_map, dtype=jnp.int32))
      staged_groups.append((
          length,
          j_list,
          points,
          indices,
          j_array,
          jnp.stack(giant_a),
          jnp.stack(giant_b),
          jnp.stack(giant_map),
      ))
    baby_rotation = (
        self.ctx.he_rot.take_explicit_rotation_fn(
            (self.level, baby_indices[0])
        )
        if baby_indices
        else None
    )
    giant_rotation = self.ctx.he_rot.take_explicit_rotation_fn(
        (self.level - 1, dummy_giant)
    )
    self.ctx.he_rot.clear()

    # Build one dynamic-operand kernel for eager and memory-bounded execution.
    batch = cache.batch
    ptct_mul = self.ctx.ptct_mul[self.level]._mul_array
    rescale = self.ctx.he_rescale[
        self.level, self.level - 1
    ]._rescale_array
    add_in = self.ctx.he_add[self.level]._add_array
    add_out = self.ctx.he_add[self.level - 1]._add_array
    has_baby = bool(baby_indices)

    def baby_body(carry, rotation_state):
      eval_a, eval_b, coefficient_map = rotation_state
      rotated = baby_rotation(carry, eval_a, eval_b, coefficient_map)
      return carry, rotated

    def baby_fn(ciphertext, eval_a, eval_b, coefficient_map):
      if has_baby:
        _, rotated = jax.lax.scan(
            baby_body, ciphertext, (eval_a, eval_b, coefficient_map)
        )
        return jnp.concatenate((ciphertext[None], rotated), axis=0)
      return ciphertext[None]

    def giant_step_fn(
        all_baby, points, indices, eval_a, eval_b, coefficient_map, j
    ):
      # Accumulate plaintext products before one rescale per giant step.
      def inner_body(total, inputs):
        index, plaintext = inputs
        product = ptct_mul(jnp.take(all_baby, index, axis=0), plaintext)
        return add_in(total, product), None

      total = jnp.zeros(
          (batch, 2, self.r, self.c, self.num_q_in), dtype=jnp.uint32
      )
      total, _ = jax.lax.scan(inner_body, total, (indices, points))
      total = rescale(total)
      rotated = giant_rotation(
          total, eval_a, eval_b, coefficient_map
      ).reshape(batch, 2, self.r, self.c, self.num_q_out)
      return jnp.where(j > 0, rotated, total)

    group_metadata = [(group[0], group[1]) for group in staged_groups]

    def matvec_fn(ciphertext, eval_a, eval_b, coefficient_map, arrays):
      all_baby = baby_fn(ciphertext, eval_a, eval_b, coefficient_map)
      total = jnp.zeros(
          (batch, 2, self.r, self.c, self.num_q_out), dtype=jnp.uint32
      )
      for (_, _), group in zip(group_metadata, arrays, strict=True):
        points, indices, j_array, giant_a, giant_b, giant_map = group

        # Scan the giant steps while keeping encoded arrays as JAX operands.
        def giant_body(carry, inputs):
          j, step_points, step_indices, step_a, step_b, step_map = inputs
          contribution = giant_step_fn(
              all_baby,
              step_points,
              step_indices,
              step_a,
              step_b,
              step_map,
              j,
          )
          return add_out(carry, contribution), None

        total, _ = jax.lax.scan(
            giant_body,
            total,
            (j_array, points, indices, giant_a, giant_b, giant_map),
        )
      return total

    self._groups = staged_groups
    self._baby_eval_a = (
        jnp.stack(baby_a) if baby_a else jnp.zeros((0,), dtype=jnp.uint32)
    )
    self._baby_eval_b = (
        jnp.stack(baby_b) if baby_b else jnp.zeros((0,), dtype=jnp.uint32)
    )
    self._baby_coefficient_map = (
        jnp.stack(baby_map) if baby_map else jnp.zeros((0,), dtype=jnp.int32)
    )
    self._matvec_fn = matvec_fn
    self._jit_matvec_fn = None if memory_bounded else jax.jit(matvec_fn)
    self._baby_fn = jax.jit(baby_fn)
    self._giant_step_fn = jax.jit(giant_step_fn)
    self._add_out_fn = jax.jit(add_out)
    self._memory_bounded = memory_bounded
    self._diagonal_map = diagonal_map
    self._diagonal_loader = loader
    self._active_i_per_j = active_i_per_j
    self._encoder = encoder
    self._scaling_error = scaling_error
    self._n_jobs = n_jobs
    self._pt_scale = pt_scale
    self._encoded_pt_scale = pt_scale_float
    self._encode_floor = encode_floor
    print(
        f'[bsgs] n1={self.n1}, n2={self.n2}; {len(active)} diagonals over '
        f'{len(active_j)} giant steps; {max(0, len(active_baby) - 1)} baby '
        'rotations.'
    )

  def _matvec_array(self, ciphertext, *operands):
    """Run the eager raw-array kernel for the program compiler."""
    if self._matvec_fn is None or self._memory_bounded:
      raise RuntimeError(
          'Call preprocess(...) in eager mode before _matvec_array.'
      )
    finite_field.check_rank5_array(
        ciphertext,
        'ctx.bsgs_matvec._matvec_array',
        batch=self.ctx._param_cache.batch,
        num_elements=2,
        degree_layout=(self.r, self.c),
        num_moduli=self.num_q_in,
    )
    return self._matvec_fn(
        ciphertext, *(operands or self._matvec_operands())
    )

  def _matvec_operands(self):
    """Return dynamic encoded-matrix and rotation operands for compilation."""
    if self._groups is None or self._memory_bounded:
      raise RuntimeError(
          'Call preprocess(...) in eager mode before requesting operands.'
      )
    arrays = [
        (points, indices, j_array, giant_a, giant_b, giant_map)
        for (
            _length,
            _j_list,
            points,
            indices,
            j_array,
            giant_a,
            giant_b,
            giant_map,
        ) in self._groups
    ]
    return (
        self._baby_eval_a,
        self._baby_eval_b,
        self._baby_coefficient_map,
        arrays,
    )

  def matvec(self, ciphertext: Polynomial) -> Polynomial:
    """Multiply the preprocessed matrix by one encrypted vector."""
    if self._groups is None:
      raise RuntimeError('Call preprocess(...) before matvec().')

    # Validate and unwrap the public ciphertext operand.
    cache = self.ctx._param_cache
    finite_field.check_ct_operand(
        type(cache.get_sliced_ff_q(self.level)),
        self.num_q_in,
        ciphertext,
        'ctx.bsgs_matvec.matvec',
        batch=cache.batch,
        num_elements=2,
        degree_layout=(self.r, self.c),
        moduli=self.q_towers_in,
    )
    ciphertext_data = ciphertext.polynomial

    if not self._memory_bounded:
      # Execute the fully encoded workload in one compiled call.
      result_data = self._jit_matvec_fn(
          ciphertext_data, *self._matvec_operands()
      )
    else:
      # Pre-rotate the ciphertext once, then encode one giant step at a time.
      all_baby = self._baby_fn(
          ciphertext_data,
          self._baby_eval_a,
          self._baby_eval_b,
          self._baby_coefficient_map,
      )
      result_data = jnp.zeros(
          (cache.batch, 2, self.r, self.c, self.num_q_out),
          dtype=jnp.uint32,
      )
      executor = (
          ThreadPoolExecutor(max_workers=self._n_jobs)
          if self._n_jobs > 1
          else None
      )
      try:
        for (
            _length,
            j_list,
            _points,
            indices,
            j_array,
            giant_a,
            giant_b,
            giant_map,
        ) in self._groups:
          for position, j in enumerate(j_list):
            requested = tuple(
                j * self.n1 + i for i in self._active_i_per_j[j]
            )
            step_map = (
                self._diagonal_map
                if self._diagonal_loader is None
                else self._diagonal_loader(requested)
            )

            # Verify that a matrix-free source still matches its planned indices.
            if self._diagonal_loader is not None:
              if not isinstance(step_map, Mapping):
                raise TypeError('matrix-free source must return a mapping.')
              if tuple(sorted(step_map)) != requested:
                raise ValueError(
                    'matrix-free source returned diagonal indices '
                    f'{tuple(sorted(step_map))}, expected {requested}.'
                )
            step_diagonals = []
            for key in requested:
              diagonal = step_map[key]
              if not isinstance(diagonal, np.ndarray) or diagonal.shape != (self.n,):
                raise ValueError(
                    f'diagonal {key} must be a 1-D np.ndarray of length '
                    f'{self.n}.'
                )
              if diagonal.dtype.kind == 'c':
                raise TypeError('matrix-free diagonal values must be real.')
              if diagonal.dtype.kind not in 'biuf' or not np.all(
                  np.isfinite(diagonal)
              ):
                raise ValueError('matrix-free diagonal values must be finite numbers.')
              if float(np.max(np.abs(diagonal))) < self._encode_floor:
                raise ValueError(
                    f'matrix-free diagonal {key} is below the plaintext '
                    f'encode floor {self._encode_floor:.3e}.'
                )
              step_diagonals.append(
                  pre_rotate_diagonal(diagonal, j, self.n1)
              )
            slots = [
                [complex(float(value), 0.0) for value in diagonal]
                + [0j] * (self.ctx.num_slots - self.n)
                for diagonal in step_diagonals
            ]
            futures = (
                [executor.submit(self._encoder.encode, values) for values in slots]
                if executor is not None
                else None
            )
            plaintexts = []
            for offset, values in enumerate(slots):
              try:
                plaintext = (
                    futures[offset].result()
                    if futures is not None
                    else self._encoder.encode(values)
                )
              except self._scaling_error as error:
                raise ValueError(
                    f'MatVec diagonal {requested[offset]} underflowed CKKS '
                    f'encode at pt_scale={self._pt_scale}; pick a larger scale.'
                ) from error
              plaintexts.append(np.asarray(
                  plaintext.polynomial[0, 0], dtype=np.uint32
              ).reshape(self.r, self.c, self.num_q_in))
            points = jnp.asarray(np.stack(plaintexts))
            contribution = self._giant_step_fn(
                all_baby,
                points,
                indices[position],
                giant_a[position],
                giant_b[position],
                giant_map[position],
                j_array[position],
            )
            result_data = self._add_out_fn(result_data, contribution)
            result_data = jax.block_until_ready(result_data)
      finally:
        if executor is not None:
          executor.shutdown(wait=True, cancel_futures=True)

    # Wrap the payload and propagate CKKS scale metadata once for both modes.
    if self._out_template is None:
      self._out_template = Polynomial(
          {
              'batch': cache.batch,
              'num_elements': 2,
              'degree': self.r * self.c,
              'num_moduli': self.num_q_out,
              'precision': 32,
              'degree_layout': (self.r, self.c),
          },
          {
              'moduli': self.q_towers_out,
              'ntt_ctx': cache.get_sliced_ntt_q(self.level - 1),
          },
      )
    result = self._out_template._clone_with_payload(result_data)
    input_scale = getattr(ciphertext, '_ckks_scale', None)
    if input_scale is not None:
      divisor = math.prod(
          self.q_towers_in[-cache.composite_degree:]
      )
      result._ckks_scale = (
          float(input_scale) / divisor
      ) * self._encoded_pt_scale
    result._ckks_nsd = getattr(ciphertext, '_ckks_nsd', 1)
    return result

  def _release(self) -> None:
    """Release encoded matrices, staged keys, and compiled kernels."""
    for name in ('_jit_matvec_fn', '_baby_fn', '_giant_step_fn', '_add_out_fn'):
      compiled = getattr(self, name, None)
      clear_cache = getattr(compiled, 'clear_cache', None)
      if clear_cache is not None:
        clear_cache()
      setattr(self, name, None)
    self._groups = None
    self._matvec_fn = None
    self._baby_eval_a = None
    self._baby_eval_b = None
    self._baby_coefficient_map = None
    self._memory_bounded = False
    self._diagonal_map = None
    self._diagonal_loader = None
    self._active_i_per_j = None
    self._encoder = None
    self._scaling_error = None
    self._n_jobs = 1
    self._pt_scale = None
    self._encoded_pt_scale = None
    self._encode_floor = None
    self._out_template = None


__all__ = [
    'compute_bsgs_params',
    'pre_rotate_diagonal',
    'required_rotation_indices',
]

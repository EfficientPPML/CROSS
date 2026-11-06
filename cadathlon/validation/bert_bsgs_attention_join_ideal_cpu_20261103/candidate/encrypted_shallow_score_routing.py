"""Experimental two-level full-slot BERT score routing.

This trades more rotations and simultaneous ciphertexts for seven fewer levels
than the independently measured bitwise route. It never refreshes or owns a
context. The caller supplies a keyless initialized CompiledArithmetic owner;
bound-aware arithmetic requires a separate propagated-bound design.
"""
from collections import Counter
from dataclasses import dataclass
import math
import numbers
import numpy as np

SLOTS, ROWS, FEATURES, HEADS, HEAD_WIDTH, KEYS, PAD_HEADS = 65536, 16, 4096, 12, 64, 128, 16


@dataclass(frozen=True)
class ShallowScoreLayout:
    key_tile: int
    offset: int

    def __post_init__(self):
        if (type(self.key_tile) is not int or not 0 <= self.key_tile < 8
                or type(self.offset) is not int or not 0 <= self.offset < ROWS):
            raise ValueError('public tile/offset outside the fixed128-key layout')

    @property
    def row_keys(self):
        return self.key_tile*ROWS + (np.arange(ROWS)+self.offset) % ROWS

    @property
    def source_indices(self):
        return (np.arange(ROWS)[:, None]*FEATURES + np.arange(HEADS)*HEAD_WIDTH).ravel()

    @property
    def target_indices(self):
        return (np.arange(ROWS)[:, None]*FEATURES + self.row_keys[:, None]*PAD_HEADS
                + np.arange(HEADS)).ravel()

    def rotation_indices(self, *, include_inverse=True):
        forward = [63*head for head in range(HEADS)]
        forward += [-PAD_HEADS*int(key) for key in self.row_keys]
        if include_inverse:
            forward += [-shift for shift in forward]
        return tuple(sorted({shift % SLOTS for shift in forward if shift % SLOTS}))

    def head_diagonals(self, *, inverse=False, gain=1.):
        for head in range(HEADS):
            mask = np.zeros(SLOTS)
            positions = np.arange(ROWS)*FEATURES + head*(HEAD_WIDTH if inverse else 1)
            mask[positions] = gain
            mask.setflags(write=False)
            yield ((-1 if inverse else 1)*63*head), mask

    def key_diagonals(self, *, inverse=False):
        for row, key in enumerate(self.row_keys):
            mask = np.zeros(SLOTS)
            positions = row*FEATURES + (0 if inverse else int(key)*PAD_HEADS) + np.arange(HEADS)
            mask[positions] = 1.
            mask.setflags(write=False)
            yield ((1 if inverse else -1)*PAD_HEADS*int(key)), mask


class ShallowScoreRouting:
    """Two actual public weighted sums, each followed by one composite rescale."""
    def __init__(self, arithmetic, layout, *, progress=None):
        from encrypted_packed_qwen import CompiledArithmetic
        if not isinstance(arithmetic, CompiledArithmetic):
            raise TypeError('routing requires a CompiledArithmetic owner')
        if callable(getattr(arithmetic, 'declare_bound', None)):
            raise TypeError('bound-aware arithmetic is outside this component contract')
        if not isinstance(layout, ShallowScoreLayout):
            raise TypeError('expected ShallowScoreLayout')
        if progress is not None and not callable(progress):
            raise TypeError('progress must be callable or None')
        self.arithmetic, self.layout, self.progress = arithmetic, layout, progress
        self.counts = Counter()
        self._borrowed_arithmetic = arithmetic
        self._borrowed_context = arithmetic.ctx
        self._borrowed_cache = getattr(arithmetic.ctx, '_param_cache', None)
        self._borrowed_scale = self._typed(arithmetic.scale)
        self._borrowed_layout = layout
        self._borrowed_layout_values = self._layout_fingerprint()
        self._borrowed_context_values = self._context_fingerprint()
        self._active_input = None
        self._guard_context()

    @staticmethod
    def _typed(value):
        if isinstance(value, (tuple, list)):
            return (type(value), tuple(ShallowScoreRouting._typed(x) for x in value))
        return (type(value), value)

    def _layout_fingerprint(self):
        return tuple(self._typed(getattr(self.layout, name, None)) for name in ('key_tile', 'offset'))

    def _context_fingerprint(self):
        context = self.arithmetic.ctx
        cache = getattr(context, '_param_cache', None)
        return tuple(self._typed(getattr(owner, name, None)) for owner, names in (
            (context, ('degree', 'num_slots', 'composite_degree', 'dnum', 'degree_layout')),
            (cache, ('batch', 'composite_degree', 'dnum', 'degree_layout', 'q_towers', 'p_towers')))
            for name in names)

    def _value_fingerprint(self, value):
        return (self._typed(self.arithmetic._level(value)),
                tuple(self._typed(getattr(value, name, None)) for name in
                      ('_ckks_nsd', '_ckks_scale', 'batch', 'num_elements', 'degree', 'degree_layout', 'moduli')),
                id(getattr(value, 'polynomial', None)), id(getattr(value, 'ntt_ctx', None)))

    @staticmethod
    def _finite_positive_scale(value):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError('finite positive real scale required')
        try:
            result = float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError('finite positive representable scale required') from error
        if not math.isfinite(result) or result <= 0:
            raise ValueError('finite positive representable scale required')
        return result

    def _guard_context(self):
        if (self.arithmetic is not self._borrowed_arithmetic
                or self.arithmetic.ctx is not self._borrowed_context
                or getattr(self.arithmetic.ctx, '_param_cache', None) is not self._borrowed_cache
                or self._typed(self.arithmetic.scale) != self._borrowed_scale
                or self._context_fingerprint() != self._borrowed_context_values
                or self.layout is not self._borrowed_layout
                or self._layout_fingerprint() != self._borrowed_layout_values):
            raise ValueError('borrowed routing owner, context or public layout changed')
        context = self.arithmetic.ctx
        cache = getattr(context, '_param_cache', None)
        if (cache is None or type(context.degree) is not int or context.degree != 2*SLOTS
                or type(context.num_slots) is not int or context.num_slots != SLOTS):
            raise ValueError('routing requires an initialized full65536-slot context')
        if (context.secret_key is not None or cache.secret_key is not None
                or context.parameters.get('secret_key') is not None):
            raise ValueError('routing requires a keyless context')
        if (type(cache.batch) is not int or cache.batch != 1
                or type(context.composite_degree) is not int or context.composite_degree != 2):
            raise ValueError('routing requires batchone and composite degree two')
        self._finite_positive_scale(self.arithmetic.scale)
        self.layout.__post_init__()
        if callable(getattr(self.arithmetic, 'declare_bound', None)):
            raise ValueError('routing owner became bound-aware')
        if getattr(self.arithmetic, 'refresher', None) is not None:
            raise ValueError('two-level route does not permit implicit refresh')

    def _validate_ciphertext(self, value, minimum_levels=2, *, canonical=True):
        self._guard_context()
        from jaxite_word import Polynomial
        from support import require_secure_execution
        if not isinstance(value, Polynomial):
            raise TypeError('routing requires a canonical ciphertext')
        if any(type(getattr(value, name)) is not int or getattr(value, name) != expected
               for name, expected in (('batch', 1), ('num_elements', 2), ('degree', 2*SLOTS))):
            raise ValueError('routing requires a full-slot two-component ciphertext')
        if type(getattr(value, '_ckks_nsd', None)) is not int or value._ckks_nsd != 1:
            raise ValueError('explicit integer noise-scale degree one required')
        require_secure_execution(self.arithmetic.ctx, value)
        level = self.arithmetic._level(value)
        if type(level) is not int or level < minimum_levels:
            raise ValueError('routing has insufficient or invalid levels')
        scale = self._finite_positive_scale(value._ckks_scale)
        if canonical and not math.isclose(scale, self.arithmetic.scale, rel_tol=1e-12):
            raise ValueError('routing requires canonical input scale')

    def _stage(self, value, diagonals, label):
        # The real weighted_sum aligns positive input scales to the owner's
        # canonical scale while performing this stage's public multiplication.
        self._validate_ciphertext(value, minimum_levels=1, canonical=False)
        level = self.arithmetic._level(value)
        values, weights = [], []
        for shift, mask in diagonals:
            normalized = int(shift) % SLOTS
            values.append(self.arithmetic.rotate(value, normalized) if normalized else value)
            weights.append(mask)
            self.counts['rotations'] += bool(normalized)
        result = self.arithmetic.weighted_sum(values, weights)
        self.counts['public_multiplications'] += len(values)
        self.counts['ciphertext_additions'] += len(values)-1
        self.counts['actual_composite_rescales'] += 1
        self.counts['public_multiplication_levels'] += 1
        if self.arithmetic._level(result) != level-1:
            raise RuntimeError('routing stage did not consume exactly one level')
        self._validate_ciphertext(result, minimum_levels=0)
        for name in tuple(vars(result)):
            if name.startswith('_public_'):
                delattr(result, name)
        if self.progress is not None:
            before = self._value_fingerprint(result)
            original, original_fingerprint = self._active_input
            self.progress(label, result, dict(self.counts))
            self._guard_context()
            if (self._value_fingerprint(result) != before
                    or self._value_fingerprint(original) != original_fingerprint):
                raise ValueError('progress callback changed routing ciphertext metadata or payload')
            self._validate_ciphertext(result, minimum_levels=0)
        return result

    def _evaluate(self, value, *, inverse, gain=1.):
        if self._active_input is not None:
            raise ValueError('routing evaluation cannot be reentered')
        self._validate_ciphertext(value)
        self._active_input = (value, self._value_fingerprint(value))
        try:
            if inverse:
                result = self._stage(value, self.layout.key_diagonals(inverse=True), 'key extraction')
                return self._stage(result, self.layout.head_diagonals(inverse=True), 'head expansion')
            result = self._stage(value, self.layout.head_diagonals(gain=gain), 'head compaction')
            return self._stage(result, self.layout.key_diagonals(), 'key placement')
        finally:
            self._active_input = None

    def forward(self, value, *, gain=1.):
        if (isinstance(gain, (bool, np.bool_)) or not isinstance(gain, numbers.Real)
                or not math.isfinite(float(gain)) or float(gain) not in (1., .125)):
            raise ValueError('supported public score gains are one and one eighth')
        return self._evaluate(value, inverse=False, gain=float(gain))

    def inverse(self, value):
        return self._evaluate(value, inverse=True)

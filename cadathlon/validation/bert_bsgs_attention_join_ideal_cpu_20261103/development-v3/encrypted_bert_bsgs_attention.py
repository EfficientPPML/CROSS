"""Full128-key BERT score and value assembly for one16-query tile.

Inputs are already projected, row-packed Q/K/V ciphertexts. This owner composes
real arithmetic primitives and the two-level BSGS route, with no codecs,
key generation, implicit refresh, nonlinear approximation or dense projection.
Common-ring numerical error remains a separate encrypted-proof obligation.
"""
from collections import Counter
from collections.abc import Mapping
from encrypted_factored_score_routing import FactoredScoreLayout
from encrypted_bsgs_score_routing import BSGSScoreRouting, required_rotation_indices, typed, accessor, same_accessors
import numpy as np

_QUARANTINE = []

SLOTS, ROWS, FEATURES, HEAD_WIDTH, KEY_TILES = 65536, 16, 4096, 64, 8


class BertBSGSAttentionTile:
    def __init__(self, arithmetic, *, progress=None):
        if progress is not None and not callable(progress):
            raise TypeError('progress must be callable or None')
        self.arithmetic = arithmetic
        self.progress = progress
        self._guard_owner = BSGSScoreRouting(arithmetic, FactoredScoreLayout(0, 0))
        self._bound_progress = progress
        self._multiply_accessor = accessor(arithmetic, "multiply")
        self._bound_arithmetic = arithmetic
        self._bound_guard = self._guard_owner
        self._active = False
        self.last_counts = None
        self._bound_keys = self._key_state()
        self._guard()

    @staticmethod
    def required_rotation_indices():
        keys = set(required_rotation_indices())
        keys.update(sign*(1 << bit) % SLOTS for bit in range(6) for sign in (-1, 1))
        keys.update(FEATURES*offset for offset in range(1, ROWS))
        return tuple(sorted(keys))

    def _key_state(self):
        cache = self.arithmetic.ctx._param_cache
        keys, maps = cache.raw_rotation_keys, cache.coef_maps
        if not isinstance(keys, Mapping) or not isinstance(maps, Mapping):
            raise ValueError('complete supplied attention key/map inventories required')
        result = []
        for key in self.required_rotation_indices():
            pair, permutation = keys.get(key), maps.get(key)
            if not isinstance(pair, Mapping) or permutation is None:
                raise ValueError('missing complete attention key/map')
            arrays = tuple(pair.get(n) for n in ('a', 'b'))
            for value in arrays:
                if (tuple(getattr(value, 'shape', ())) != (8, 114, 131072)
                        or np.dtype(getattr(value, 'dtype', None)) != np.dtype('uint32')
                        or isinstance(value, np.ndarray) and value.flags.writeable):
                    raise ValueError('immutable full attention key arrays required')
            if (tuple(getattr(permutation, 'shape', ())) != (131072,)
                    or np.dtype(getattr(permutation, 'dtype', None)) != np.dtype('int32')
                    or isinstance(permutation, np.ndarray) and permutation.flags.writeable):
                raise ValueError('immutable full attention maps required')
            result.append((key, id(pair), *(id(v) for v in arrays), id(permutation)))
        return tuple(result)

    def _guard(self):
        if _QUARANTINE:
            raise RuntimeError('failed BSGS attention requires dedicated-process exit')
        if (self.arithmetic is not self._bound_arithmetic or self._guard_owner is not self._bound_guard
                or self.progress is not self._bound_progress
                or not same_accessors((accessor(self.arithmetic, 'multiply'),), (self._multiply_accessor,))):
            raise ValueError('construction-bound attention owner/accessor changed')
        self._guard_owner._guard()
        if self._key_state() != self._bound_keys:
            raise ValueError('complete attention key/map identity changed')

    def _validate(self, value, levels):
        self._guard()
        self._guard_owner._validate(value, minimum=levels)

    def _fingerprint(self, value):
        return self._guard_owner._fingerprint(value)

    @staticmethod
    def _forget(value):
        for name in tuple(vars(value)):
            if name.startswith('_public_'):
                delattr(value, name)
        return value

    def _begin(self, primary, tiles, *, primary_levels, tile_levels):
        self._guard()
        if self._active:
            raise ValueError('attention tile evaluation cannot be reentered')
        if not isinstance(tiles, (tuple, list)) or len(tiles) != KEY_TILES:
            raise ValueError('exactly eight projected key/value tiles required')
        tiles = tuple(tiles)
        self._validate(primary, primary_levels)
        for tile in tiles:
            self._validate(tile, tile_levels)
        metadata = [(self.arithmetic._level(tile), typed(tile._ckks_scale)) for tile in tiles]
        if any(item != metadata[0] for item in metadata[1:]):
            raise ValueError('projected tiles must share actual level and scale')
        originals = (primary, *tiles)
        fingerprints = tuple(self._fingerprint(value) for value in originals)
        self._active = True
        self.last_counts = None
        return tiles, originals, fingerprints

    def _check_inputs(self, originals, fingerprints):
        self._guard()
        if tuple(self._fingerprint(value) for value in originals) != fingerprints:
            raise ValueError('attention source ciphertext changed')

    def _note(self, label, result, counts, originals, fingerprints):
        self._check_inputs(originals, fingerprints)
        self._validate(result, 0)
        self._forget(result)
        if self.progress is not None:
            current_payload = result.polynomial
            before = self._fingerprint(result)
            self.progress(label, result, dict(counts))
            self._check_inputs(originals, fingerprints)
            if self._fingerprint(result) != before:
                raise ValueError('attention callback changed result metadata or payload')
            self._validate(result, 0)
        self._forget(result)

    def _accumulate(self, accumulated, value, expected_level, counts):
        self._validate(value, 0)
        if self.arithmetic._level(value) != expected_level:
            raise RuntimeError('attention map returned unexpected level')
        if accumulated is None:
            return value
        if typed(accumulated._ckks_scale) != typed(value._ckks_scale):
            raise RuntimeError('attention maps have different actual scales')
        result = self.arithmetic.add(accumulated, value)
        counts['accumulator_additions'] += 1
        self._validate(result, 0)
        if (self.arithmetic._level(result) != expected_level
                or typed(result._ckks_scale) != typed(value._ckks_scale)):
            raise RuntimeError('attention addition changed level or scale')
        return result

    def scores(self, projected_query, projected_keys):
        """Return every raw q·k/8 score, with no public attention mask applied."""
        keys, originals, fingerprints = self._begin(projected_query, projected_keys, primary_levels=3, tile_levels=3)
        original_payloads = tuple(v.polynomial for v in originals)
        counts = Counter()
        output = None
        arithmetic = self.arithmetic
        try:
            level = min(arithmetic._level(projected_query), arithmetic._level(keys[0])) - 3
            for tile_index, key in enumerate(keys):
                for offset in range(ROWS):
                    self._check_inputs(originals, fingerprints)
                    aligned = arithmetic.rotate(key, offset*FEATURES) if offset else key
                    counts['row_alignment_rotations'] += bool(offset)
                    product = arithmetic.multiply(projected_query, aligned)
                    counts['ciphertext_products'] += 1
                    for bit in range(6):
                        product = arithmetic.add(product, arithmetic.rotate(product, 1 << bit))
                        counts['head_rotations'] += 1
                        counts['head_additions'] += 1
                    router = BSGSScoreRouting(arithmetic, FactoredScoreLayout(tile_index, offset))
                    mapped = router.forward(product, gain=.125)
                    for name, count in router.counts.items():
                        counts['route_' + name] += count
                    output = self._accumulate(output, mapped, level, counts)
                    self._note(f'score key_tile{tile_index} offset{offset}', output, counts, originals, fingerprints)
            self.last_counts = dict(counts)
            return output
        except BaseException as error:
            # Preserve every traceback-local temporary and borrowed backing owner
            # until dedicated-process exit after an entered arithmetic failure.
            _QUARANTINE.append((self, arithmetic, originals, original_payloads, output, error))
            raise
        finally:
            self._active = False

    def mix_values(self, probabilities, projected_values):
        """Sum all128 key probabilities times V for each query/head/feature."""
        values, originals, fingerprints = self._begin(probabilities, projected_values, primary_levels=3, tile_levels=1)
        original_payloads = tuple(v.polynomial for v in originals)
        counts = Counter()
        output = None
        arithmetic = self.arithmetic
        try:
            level = min(arithmetic._level(probabilities)-2, arithmetic._level(values[0])) - 1
            for tile_index, value in enumerate(values):
                for offset in range(ROWS):
                    self._check_inputs(originals, fingerprints)
                    router = BSGSScoreRouting(arithmetic, FactoredScoreLayout(tile_index, offset))
                    weights = router.inverse(probabilities)
                    for name, count in router.counts.items():
                        counts['route_' + name] += count
                    for bit in range(6):
                        weights = arithmetic.add(weights, arithmetic.rotate(weights, -(1 << bit) % SLOTS))
                        counts['head_rotations'] += 1
                        counts['head_additions'] += 1
                    aligned = arithmetic.rotate(value, offset*FEATURES) if offset else value
                    counts['row_alignment_rotations'] += bool(offset)
                    product = arithmetic.multiply(weights, aligned)
                    counts['ciphertext_products'] += 1
                    output = self._accumulate(output, product, level, counts)
                    self._note(f'value key_tile{tile_index} offset{offset}', output, counts, originals, fingerprints)
            self.last_counts = dict(counts)
            return output
        except BaseException as error:
            # Preserve every traceback-local temporary and borrowed backing owner
            # until dedicated-process exit after an entered arithmetic failure.
            _QUARANTINE.append((self, arithmetic, originals, original_payloads, output, error))
            raise
        finally:
            self._active = False

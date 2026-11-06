"""Factor full-slot score routing into a local key and a whole-key-tile rotation.

The two weighted sums retain the original two-level depth and rotation count.
Across all128 mappings, the route's public rotation-key union is66 rather
than276. This is a circuit/key-inventory property, not an encrypted noise proof.
Positive noncanonical input scales are aligned by the first real weighted sum.
"""
from encrypted_shallow_score_routing import ShallowScoreLayout, ShallowScoreRouting, SLOTS, ROWS, PAD_HEADS


class FactoredScoreLayout(ShallowScoreLayout):
    """The local key permutation is independent of the eight global key tiles."""
    def key_diagonals(self, *, inverse=False):
        yield from ShallowScoreLayout(0, self.offset).key_diagonals(inverse=inverse)

    @property
    def tile_shift(self):
        return self.key_tile * ROWS * PAD_HEADS

    def rotation_indices(self, *, include_inverse=True):
        steps = set(ShallowScoreLayout(0, self.offset).rotation_indices(include_inverse=include_inverse))
        if self.tile_shift:
            steps.add((-self.tile_shift) % SLOTS)
            if include_inverse:
                steps.add(self.tile_shift)
        return tuple(sorted(steps))


class FactoredScoreRouting(ShallowScoreRouting):
    """Borrow the hardened owner; preserve its two-level forward/inverse maps."""
    def __init__(self, arithmetic, layout, *, progress=None):
        if not isinstance(layout, FactoredScoreLayout):
            raise TypeError('factored score layout required')
        super().__init__(arithmetic, layout, progress=progress)

    def _tile_rotation(self, value, *, inverse, label):
        self._validate_ciphertext(value, minimum_levels=0, canonical=False)
        shift = self.layout.tile_shift * (1 if inverse else -1)
        level, scale = self.arithmetic._level(value), self._typed(value._ckks_scale)
        result = self.arithmetic.rotate(value, shift % SLOTS) if shift else value
        self.counts['rotations'] += bool(shift)
        self._validate_ciphertext(result, minimum_levels=0, canonical=False)
        if self.arithmetic._level(result) != level or self._typed(result._ckks_scale) != scale:
            raise RuntimeError('tile rotation changed ciphertext level or scale')
        # A zero tile rotation can return the original input. Its public tags
        # belong to the caller; later weighted sums clear their own result tags.
        if result is not self._active_input[0]:
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
            self._validate_ciphertext(result, minimum_levels=0, canonical=False)
        return result

    def _evaluate(self, value, *, inverse, gain=1.):
        if self._active_input is not None:
            raise ValueError('routing evaluation cannot be reentered')
        self._validate_ciphertext(value, canonical=False)
        self._active_input = (value, self._value_fingerprint(value))
        try:
            if inverse:
                local = self._tile_rotation(value, inverse=True, label='key tile extraction')
                result = self._stage(local, self.layout.key_diagonals(inverse=True), 'key extraction')
                return self._stage(result, self.layout.head_diagonals(inverse=True), 'head expansion')
            result = self._stage(value, self.layout.head_diagonals(gain=gain), 'head compaction')
            local = self._stage(result, self.layout.key_diagonals(), 'local key placement')
            return self._tile_rotation(local, inverse=False, label='key placement')
        finally:
            self._active_input = None

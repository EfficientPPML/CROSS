"""Bounded public controls/key caching for repeated WISE CKKS rotations.

Uses the same explicit CROSS rotation kernel as the shared low-memory helper.
The cache contains public evaluation keys and transform controls, never input
ciphertexts. Each call retains the original completion barrier.
"""
from collections import OrderedDict
from encrypted_rotation import CompiledRotation, rotation_state_without_controls


class CachedRotation(CompiledRotation):
    def __init__(self, context, cache_bytes):
        super().__init__(context)
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError("rotation cache budget must be a nonnegative integer")
        self.cache_bytes = cache_bytes
        self.entries = OrderedDict()
        self.cached_bytes = self.peak_cached_bytes = 0
        self.binding_hits = self.state_hits = 0

    def _get(self, key):
        entry = self.entries.get(key)
        if entry is not None:
            self.entries.move_to_end(key)
            return entry[0]
        return None

    def _retain(self, key, value, arrays):
        # Logical array sizes are charged separately even when buffers may be
        # shared. This bounds retained cache entries, not all runtime memory.
        size = sum(int(array.nbytes) for array in arrays)
        if size > self.cache_bytes:
            return
        old = self.entries.pop(key, None)
        if old is not None:
            self.cached_bytes -= old[1]
        while self.entries and self.cached_bytes + size > self.cache_bytes:
            _, (_, removed) = self.entries.popitem(last=False)
            self.cached_bytes -= removed
        self.entries[key] = (value, size)
        self.cached_bytes += size
        self.peak_cached_bytes = max(self.peak_cached_bytes, self.cached_bytes)

    def clear(self, *, executables=False):
        super().clear(executables=executables)
        self.entries.clear()
        self.cached_bytes = 0

    def rotate(self, value, index):
        import jax
        from support import require_secure_execution
        require_secure_execution(self, value)
        if type(index) is not int:
            raise ValueError("rotation index must be an integer")
        cache = self.ctx._param_cache
        level = cache.level_for_num_q(value.num_moduli)
        if tuple(value.shape) != (cache.batch, 2, *cache.degree_layout, value.num_moduli):
            raise ValueError("ciphertext layout differs from initialized rotation context")
        if len(value.polynomial.devices()) != 1:
            raise ValueError("cached rotation requires one physical device")
        device = next(iter(value.polynomial.devices()))
        signature = (level, tuple(value.shape), str(value.polynomial.dtype), device)
        binding_key, state_key = ("binding", signature), ("state", signature, index)
        binding, state = self._get(binding_key), self._get(state_key)
        if binding is None:
            operation = self.ctx.he_rot[level, index]
            if state is None:
                state = operation._rotation_state()
            rotate = self.ctx.he_rot.take_explicit_rotation_fn((level, index))
            binding = self._compiler.bind(rotate, (value.polynomial, *state), signature=signature, device=device)
            self._retain(binding_key, binding, binding.controls)
        else:
            self.binding_hits += 1
            if state is None:
                state = rotation_state_without_controls(cache, level, index)
            else:
                self.state_hits += 1
        self._retain(state_key, state, state)
        result = value._clone_with_payload(binding(value.polynomial, *state))
        jax.block_until_ready(result.polynomial)
        cache._formatted_rotation_keys.pop(index, None)
        self.ctx.he_rot._instances.pop((level, index), None)
        return result

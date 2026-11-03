"""Experimental compiled CKKS rotations with explicit public key operands.

Not yet used by SecureRefresh. This isolates compilation from rotation index so
one arithmetic executable can serve multiple keys at the same level. Controls
are lifted out of HLO literals, following Mapping's single-device compiler.
"""
from __future__ import annotations


class CompiledRotation:
    """Serial, single-device rotation executor; keeps at most one level bound."""

    def __init__(self, context):
        from encrypted_polynomial import EncryptedPolynomialEvaluator
        from encrypted_compilation import ExplicitArrayCompiler
        # Validate actual security and physically absent server secret keys.
        EncryptedPolynomialEvaluator(context)
        self.ctx = context
        self._binding = None
        self._compiler = ExplicitArrayCompiler()

    @property
    def compile_count(self):
        return self._compiler.compile_count

    def clear(self, *, executables=False):
        """Release active public arrays; optionally discard reusable code too."""
        self._binding = None
        if executables:
            self._compiler.clear_executables()

    def rotate(self, value, index):
        import jax
        from support import require_secure_execution

        require_secure_execution(self, value)
        if type(index) is not int:
            raise ValueError('rotation index must be an integer')
        cache = self.ctx._param_cache
        level = cache.level_for_num_q(value.num_moduli)
        expected_shape = (cache.batch, 2, *cache.degree_layout, value.num_moduli)
        if tuple(value.shape) != expected_shape:
            raise ValueError('ciphertext layout differs from the initialized rotation context')
        if len(value.polynomial.devices()) != 1:
            raise ValueError('compiled experimental rotation requires one physical device')
        device = next(iter(value.polynomial.devices()))
        signature = (level, tuple(value.shape), str(value.polynomial.dtype), device)
        if self._binding is None or self._binding[0] != signature:
            operation = self.ctx.he_rot[level, index]
            state = operation._rotation_state()
            # Only a new binding needs expensive NTT/control construction.
            # Detaching keys prevents compilation from capturing their arrays.
            rotate = self.ctx.he_rot.take_explicit_rotation_fn((level, index))
            binding = self._compiler.bind(rotate, (value.polynomial, *state),
                                          signature=signature, device=device)
            self._binding = (signature, binding)
        else:
            state = rotation_state_without_controls(cache, level, index)
        _, binding = self._binding
        result = value._clone_with_payload(binding(value.polynomial, *state))
        # Release transient formatted keys only after their execution finishes.
        # Raw supplied keys remain available for subsequent rotations.
        jax.block_until_ready(result.polynomial)
        cache._formatted_rotation_keys.pop(index, None)
        # Match take_explicit_rotation_fn's ownership behavior even if another
        # serial caller preconstructed this index before a binding hit.
        self.ctx.he_rot._instances.pop((level, index), None)
        return result


def rotation_state_without_controls(cache, level, index):
    """Match facade key slicing and setup_rotation without rebuilding NTTs.

    This is only used after an existing level/layout-specific binding owns its
    controls. Key arrays remain explicit operands, with the same conversion
    from standard residues to the configured computation representation.
    """
    import jax.numpy as jnp

    q_at_level = cache.q_moduli_at_level(level)
    keygen_size_q = len(q_at_level) if cache.per_level_rotation_keys else len(cache.q_towers)
    alpha = (keygen_size_q + cache.dnum - 1) // cache.dnum
    effective_dnum = (len(q_at_level) + alpha - 1) // alpha
    a, b, coef_map = cache.get_rot_key(index, level)
    field = cache.ff_context_cls(moduli=q_at_level + cache.p_towers)
    a = field.to_computation_format(jnp.asarray(a[:effective_dnum], dtype=jnp.uint64)).astype(jnp.uint32)
    b = field.to_computation_format(jnp.asarray(b[:effective_dnum], dtype=jnp.uint64)).astype(jnp.uint32)
    coef_map = jnp.asarray(coef_map, dtype=jnp.int32)
    expected = (effective_dnum, *cache.degree_layout, len(q_at_level) + len(cache.p_towers))
    if a.shape != expected or b.shape != expected:
        raise ValueError('rotation keys differ from initialized partition/layout contract')
    if coef_map.shape != (cache.degree,):
        raise ValueError('rotation map differs from initialized ring degree')
    return a, b, coef_map

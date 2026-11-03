"""Experimental CKKS primitive compilation with explicit public operands.

Each executor retains one active binding of public controls and a bounded
cache of compiled code. Clearing bindings releases controls without throwing
away executable code. Inputs must use an initialized, keyless, 128-bit-secure
single-device evaluator; arithmetic and metadata match the core facades.
"""
import math

from encrypted_compilation import ExplicitArrayCompiler
from encrypted_polynomial import EncryptedPolynomialEvaluator

class CompiledRelinearization:
    def __init__(self, context):
        EncryptedPolynomialEvaluator(context)
        self.ctx = context
        self.compiler = ExplicitArrayCompiler()
        self.binding = None

    def clear(self, *, executables=False):
        self.binding = None
        if executables:
            self.compiler.clear_executables()

    def relinearize(self, value):
        import jax
        import finite_field
        from support import require_secure_execution
        require_secure_execution(self, value)
        cache = self.ctx._param_cache
        level = cache.level_for_num_q(value.num_moduli)
        if (level < 1 or tuple(value.shape) != (cache.batch, 3, *cache.degree_layout, value.num_moduli)
                or tuple(value.moduli) != tuple(cache.q_moduli_at_level(level))
                or value.precision != 32 or str(value.polynomial.dtype) != 'uint32'):
            raise ValueError('invalid three-component relinearization input')
        finite_field.check_ct_operand(cache.ff_context_cls, value.num_moduli, value,
            'compiled relinearization', batch=cache.batch, num_elements=3,
            degree_layout=cache.degree_layout, moduli=cache.q_moduli_at_level(level))
        if len(value.polynomial.devices()) != 1:
            raise ValueError('one-device relinearization required')
        device = next(iter(value.polynomial.devices()))
        signature = ('relinearize-no-rescale', level, tuple(value.shape), device)
        if self.binding is None or self.binding[0] != signature:
            operation = self.ctx.he_mul[level - 1]
            kernel = operation._no_relin_operator
            if not kernel.skip_rescale:
                raise ValueError('no-rescale relinearization controls required')
            try:
                binding = self.compiler.bind(kernel._relinearize_array, (value.polynomial,),
                                             signature=signature, device=device)
                self.binding = (signature, binding)
            finally:
                self.ctx.he_mul._instances.pop(level - 1, None)
        result = value._clone_with_payload(self.binding[1](value.polynomial), num_elements=2)
        jax.block_until_ready(result.polynomial)
        return result


class CompiledRescale:
    def __init__(self, context):
        EncryptedPolynomialEvaluator(context)
        self.ctx = context
        self.compiler = ExplicitArrayCompiler()
        self.binding = None

    def clear(self, *, executables=False):
        self.binding = None
        if executables:
            self.compiler.clear_executables()

    def rescale(self, value, source, target):
        import jax
        import finite_field
        from support import require_secure_execution
        require_secure_execution(self, value)
        cache = self.ctx._param_cache
        if (type(source) is not int or type(target) is not int
                or cache.level_for_num_q(value.num_moduli) != source or not 0 <= target < source):
            raise ValueError('invalid rescale level contract')
        finite_field.check_ct_operand(cache.ff_context_cls, value.num_moduli, value,
            'compiled rescale', batch=cache.batch, num_elements=2,
            degree_layout=cache.degree_layout, moduli=cache.q_moduli_at_level(source))
        if len(value.polynomial.devices()) != 1:
            raise ValueError('one-device rescale required')
        device = next(iter(value.polynomial.devices()))
        signature = ('rescale', source, target, tuple(value.shape), device)
        if self.binding is None or self.binding[0] != signature:
            operation = self.ctx.he_rescale[source, target]
            try:
                binding = self.compiler.bind(operation._rescale_array, (value.polynomial,),
                                             signature=signature, device=device)
                self.binding = (signature, binding)
            finally:
                self.ctx.he_rescale._instances.pop((source, target), None)
        target_moduli = cache.q_moduli_at_level(target)
        result = value._clone_with_payload(self.binding[1](value.polynomial), moduli=target_moduli,
                                           ntt_ctx=cache.get_sliced_ntt_q(target))
        if hasattr(value, '_ckks_scale'):
            result._ckks_scale = value._ckks_scale / math.prod(value.moduli[len(target_moduli):])
        jax.block_until_ready(result.polynomial)
        return result


class CompiledPlaintextMultiply:
    def __init__(self, context):
        EncryptedPolynomialEvaluator(context)
        self.ctx = context
        self.compiler = ExplicitArrayCompiler()
        self.binding = None

    def clear(self, *, executables=False):
        self.binding = None
        if executables:
            self.compiler.clear_executables()

    def multiply_encoded(self, value, plaintext, level):
        """Match the private raw-plaintext facade, including broadcasting.

        Public callers should use ``multiply`` with a batch-one Polynomial.
        This raw path accepts the bootstrap engine's standard-residue arrays
        and may transfer public plaintext data onto the ciphertext device.
        """
        import jax
        import jax.numpy as jnp
        import finite_field
        from support import require_secure_execution
        require_secure_execution(self, value)
        if self.ctx._param_cache.level_for_num_q(value.num_moduli) != level:
            raise ValueError('plaintext multiply level mismatch')
        cache = self.ctx._param_cache
        finite_field.check_ct_operand(cache.ff_context_cls, value.num_moduli, value,
            'compiled plaintext multiply', batch=cache.batch, num_elements=2,
            degree_layout=cache.degree_layout, moduli=cache.q_moduli_at_level(level))
        if len(value.polynomial.devices()) != 1:
            raise ValueError('one-device plaintext multiply required')
        device = next(iter(value.polynomial.devices()))
        signature = ('plaintext-multiply', level, tuple(value.shape), tuple(plaintext.shape), str(plaintext.dtype), device)
        if self.binding is None or self.binding[0] != signature:
            operation = self.ctx.ptct_mul[level]
            def multiply(payload, encoded):
                prepared = operation._ptct.ff_ctx.to_computation_format(jnp.asarray(encoded, jnp.uint64))
                return operation._mul_array(payload, prepared)
            try:
                binding = self.compiler.bind(multiply, (value.polynomial, plaintext),
                                             signature=signature, device=device)
                self.binding = (signature, binding)
            finally:
                self.ctx.ptct_mul._instances.pop(level, None)
        result = value._clone_with_payload(self.binding[1](value.polynomial, plaintext))
        result.__dict__.pop('_ckks_scale', None)
        jax.block_until_ready(result.polynomial)
        return result

    def multiply(self, value, plaintext, level):
        """Match public multiplication by one canonical plaintext Polynomial."""
        import finite_field
        cache = self.ctx._param_cache
        finite_field.check_ct_operand(cache.ff_context_cls, cache.num_q_at_level(level), plaintext,
            'compiled plaintext operand', batch=1, num_elements=1,
            degree_layout=cache.degree_layout, moduli=cache.q_moduli_at_level(level))
        result = self.multiply_encoded(value, plaintext.polynomial, level)
        ciphertext_scale = getattr(value, '_ckks_scale', None)
        plaintext_scale = getattr(plaintext, '_ckks_scale', None)
        if ciphertext_scale is not None and plaintext_scale is not None:
            result._ckks_scale = float(ciphertext_scale) * float(plaintext_scale)
        result._ckks_nsd = getattr(value, '_ckks_nsd', 1) + 1
        return result


class CompiledTensorOnly:
    """Exact three-component tensor product, without key switching or rescale.

    Public finite-field controls are taken from the initialized parameter cache;
    no multiplication facade or evaluation-key controls are constructed here.
    The canonical core tensor kernel remains the arithmetic implementation.
    """
    def __init__(self, context):
        EncryptedPolynomialEvaluator(context)
        self.ctx = context
        self.compiler = ExplicitArrayCompiler()
        self.binding = None

    def clear(self, *, executables=False):
        self.binding = None
        if executables:
            self.compiler.clear_executables()

    def multiply(self, left, right):
        import jax
        import jax.numpy as jnp
        import finite_field
        from polynomial import Polynomial
        from types import SimpleNamespace
        from support import require_secure_execution

        require_secure_execution(self, (left, right))
        cache = self.ctx._param_cache
        level = cache.level_for_num_q(left.num_moduli)
        if level < 1:
            raise ValueError('tensor-only product requires at least level one')
        for value in (left, right):
            finite_field.check_ct_operand(cache.ff_context_cls, cache.num_q_at_level(level), value,
                'compiled tensor-only product', batch=cache.batch, num_elements=2,
                degree_layout=cache.degree_layout, moduli=cache.q_moduli_at_level(level))
        scales = [getattr(value, '_ckks_scale', None) for value in (left, right)]
        if any(scale is not None and (not math.isfinite(scale) or scale <= 0) for scale in scales):
            raise ValueError('tensor-only product requires finite positive tracked scales')
        scale = None if None in scales else scales[0] * scales[1]
        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise ValueError('tensor-only product scale is outside floating-point range')
        nsds = [getattr(value, '_ckks_nsd', 1) for value in (left, right)]
        if any(type(nsd) is not int or nsd < 1 for nsd in nsds):
            raise ValueError('tensor-only product requires positive integer noise-scale degrees')
        devices = left.polynomial.devices()
        if len(devices) != 1 or devices != right.polynomial.devices():
            raise ValueError('tensor-only operands must share one physical device')
        device = next(iter(devices))
        signature = ('tensor-only', level, tuple(left.shape), device)
        if self.binding is None or self.binding[0] != signature:
            field = cache.get_sliced_ff_q(level)
            def tensor(a, b):
                shell = SimpleNamespace(polynomial=jnp.concatenate((a, b), axis=1),
                    ntt_ctx=SimpleNamespace(ff_ctx=field), modulus_dtype=jnp.uint32)
                first, last = Polynomial._polynomial_mult_array(shell)
                return jnp.concatenate((first, last), axis=1).astype(jnp.uint32)
            binding = self.compiler.bind(tensor, (left.polynomial, right.polynomial),
                                         signature=signature, device=device)
            self.binding = (signature, binding)
        result = left._clone_with_payload(self.binding[1](left.polynomial, right.polynomial),
                                          num_elements=3)
        result.__dict__.pop('_ckks_scale', None)
        if scale is not None:
            result._ckks_scale = scale
        result._ckks_nsd = sum(nsds)
        jax.block_until_ready(result.polynomial)
        return result

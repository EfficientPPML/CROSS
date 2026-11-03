"""Experimental explicit-operand compilation for public HE primitive controls.

Callers bind only primitives from an initialized keyless evaluator. Executables
survive JAX's global cache clear; individual bindings own the public control
arrays and can be dropped independently to release device buffers.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math


@dataclass
class ArrayBinding:
    executable: object
    controls: tuple

    def __call__(self, *operands):
        return self.executable(*operands, self.controls)


class ExplicitArrayCompiler:
    def __init__(self, *, max_executables=128):
        if type(max_executables) is not int or max_executables < 1:
            raise ValueError('max_executables must be a positive integer')
        self.max_executables = max_executables
        self._executables = OrderedDict()
        self.compile_count = 0

    def bind(self, function, operands, *, signature, device):
        import jax
        structs = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in operands)
        closed = jax.make_jaxpr(function)(*structs)
        program = closed.jaxpr
        controls = tuple(jax.device_put(value, device) for value in closed.consts)
        # Different aliasing after table-cache reconstruction may change the
        # traced constant contract. Reuse only an identical public program and
        # abstract argument structure; values of lifted controls may differ.
        fingerprint = hashlib.sha256(str(program).encode()).hexdigest()
        control_specs = tuple((tuple(value.shape), str(value.dtype)) for value in controls)
        key = (signature, device, fingerprint, control_specs)
        executable = self._executables.get(key)
        if executable is None:
            def evaluate(*arguments):
                return jax.core.eval_jaxpr(program, arguments[-1], *arguments[:-1])[0]

            executable = jax.jit(evaluate).lower(*structs, controls).compile()
            self._executables[key] = executable
            self.compile_count += 1
            while len(self._executables) > self.max_executables:
                self._executables.popitem(last=False)
        else:
            self._executables.move_to_end(key)
        return ArrayBinding(executable, controls)

    def clear_executables(self):
        self._executables.clear()


class CompiledTensorProduct:
    """Compile bootstrap's tensor product plus relinearization, without rescale.

    Inputs are already level-aligned at nsd=1. The output preserves their Q
    prefix, has the product scale and nsd=2. This deliberately matches the
    split primitive contract used inside Bootstrap, not public ``he_mul.mul``.
    """

    def __init__(self, context):
        from encrypted_polynomial import EncryptedPolynomialEvaluator
        EncryptedPolynomialEvaluator(context)
        self.ctx = context
        self._compiler = ExplicitArrayCompiler()
        self._binding = None

    @property
    def compile_count(self):
        return self._compiler.compile_count

    def clear(self, *, executables=False):
        self._binding = None
        if executables:
            self._compiler.clear_executables()

    def multiply(self, left, right):
        import jax
        import jax.numpy as jnp
        from support import require_secure_execution

        require_secure_execution(self, (left, right))
        cache = self.ctx._param_cache
        if tuple(left.moduli) != tuple(right.moduli) or tuple(left.shape) != tuple(right.shape):
            raise ValueError('tensor-product operands must have identical levels and layouts')
        level = cache.level_for_num_q(left.num_moduli)
        if level < 1 or tuple(left.shape) != (cache.batch, 2, *cache.degree_layout, left.num_moduli):
            raise ValueError('tensor-product input differs from its initialized level/layout')
        scales = [getattr(value, '_ckks_scale', None) for value in (left, right)]
        if (any(scale is None or not math.isfinite(scale) or scale <= 0 for scale in scales)
                or any(getattr(value, '_ckks_nsd', 1) != 1 for value in (left, right))):
            raise ValueError('tensor-product operands require positive finite scales and nsd=1')
        scale = scales[0] * scales[1]
        if not math.isfinite(scale):
            raise ValueError('tensor-product scale overflows')
        devices = left.polynomial.devices()
        if len(devices) != 1 or devices != right.polynomial.devices():
            raise ValueError('tensor product requires both operands on one physical device')
        device = next(iter(devices))
        signature = ('tensor-no-rescale', level, tuple(left.shape), str(left.polynomial.dtype), device)
        if self._binding is None or self._binding[0] != signature:
            operation = self.ctx.he_mul[level - 1]
            kernel = operation._no_relin_operator
            if not kernel.skip_rescale:
                raise ValueError('tensor product requires explicit no-rescale controls')

            def tensor_product(a, b):
                tensor = kernel._hemul_no_relin_array(jnp.concatenate((a, b), axis=1))
                return kernel._relinearize_array(tensor)

            try:
                binding = self._compiler.bind(tensor_product, (left.polynomial, right.polynomial),
                                              signature=signature, device=device)
                self._binding = (signature, binding)
            finally:
                # Tracing temporarily populates the kernel's mutable working
                # Polynomial wrappers with tracers. Never retain that facade
                # in the evaluator's eager operator cache.
                self.ctx.he_mul._instances.pop(level - 1, None)
        result = left._clone_with_payload(self._binding[1](left.polynomial, right.polynomial))
        result._ckks_scale = scale
        result._ckks_nsd = 2
        jax.block_until_ready(result.polynomial)
        return result

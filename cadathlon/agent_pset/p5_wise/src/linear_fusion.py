"""Fuse small groups of WISE public diagonal products using CROSS kernels.

Coefficients and ciphertexts remain explicit runtime operands. The executable
does exactly the core plaintext products and modular additions, followed by
the caller's original composite rescale. No slot decoding is involved.
"""
from collections import OrderedDict
import math
import numpy as np


class FusedPlaintextSum:
    def __init__(self, context, chunk_size):
        from encrypted_compilation import ExplicitArrayCompiler
        if type(chunk_size) is not int or not 1 <= chunk_size <= 16:
            raise ValueError("linear fusion chunk must be an integer in 1..16")
        self.ctx = context
        self.chunk_size = chunk_size
        self.compiler = ExplicitArrayCompiler(max_executables=64)
        self.bindings = OrderedDict()

    def _sum(self, values, plaintexts, level):
        import jax.numpy as jnp
        import finite_field
        from support import require_secure_execution
        require_secure_execution(self, values)
        cache = self.ctx._param_cache
        moduli = cache.q_moduli_at_level(level)
        device = next(iter(values[0].polynomial.devices()))
        scales = []
        for value, plaintext in zip(values, plaintexts, strict=True):
            for item, batch, elements in ((value, cache.batch, 2), (plaintext, 1, 1)):
                finite_field.check_ct_operand(cache.ff_context_cls, len(moduli), item,
                    "WISE fused plaintext sum", batch=batch, num_elements=elements,
                    degree_layout=cache.degree_layout, moduli=moduli)
                if item.polynomial.devices() != {device}:
                    raise ValueError("fused sum operands must share one physical device")
            if getattr(value, "_ckks_nsd", 1) != 1:
                raise ValueError("fused sum requires ciphertext noise-scale degree one")
            scale = value._ckks_scale * plaintext._ckks_scale
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("invalid fused product scale")
            scales.append(scale)
        if any(not math.isclose(s, scales[0], rel_tol=1e-12) for s in scales):
            raise ValueError("fused products have different scales")
        count = len(values)
        key = (level, count, tuple(values[0].shape), device)
        binding = self.bindings.get(key)
        operands = tuple(v.polynomial for v in values) + tuple(p.polynomial for p in plaintexts)
        if binding is None:
            multiply, add = self.ctx.ptct_mul[level], self.ctx.he_add[level]
            def kernel(*arrays):
                result = None
                for i in range(count):
                    encoded = multiply._ptct.ff_ctx.to_computation_format(jnp.asarray(arrays[count+i], jnp.uint64))
                    term = multiply._mul_array(arrays[i], encoded)
                    result = term if result is None else add._add_array(result, term).astype(jnp.uint32)
                return result
            binding = self.compiler.bind(kernel, operands, signature=("wise-fused-sum", key), device=device)
            self.bindings[key] = binding
            while len(self.bindings) > 64:
                self.bindings.popitem(last=False)
        else:
            self.bindings.move_to_end(key)
        result = values[0]._clone_with_payload(binding(*operands))
        result._ckks_scale, result._ckks_nsd = scales[0], 2
        return result

    def weighted_sum(self, arithmetic, values, weights, *, bias=0.):
        """Preserve the ordinary evaluator's alignment and numerical guards."""
        values, weights = tuple(values), tuple(arithmetic._public_slots(w) for w in weights)
        bias = arithmetic._public_slots(bias)
        if not values or len(values) != len(weights):
            raise ValueError("nonempty values and equally sized public weights required")
        values = arithmetic._ensure_capacity(values, 1)
        level = min(arithmetic._level(value) for value in values)
        if level < 1:
            raise ValueError("insufficient levels for weighted sum")
        arithmetic._prepare(level)
        divisor = math.prod(self.ctx._param_cache.q_moduli_at_level(level)[-2:])
        target_scale = arithmetic.scale * divisor
        result = None
        for start in range(0, len(values), self.chunk_size):
            ciphertexts, plaintexts = [], []
            for value, weight in zip(values[start:start+self.chunk_size], weights[start:start+self.chunk_size], strict=True):
                value = arithmetic._lower(value, level)
                scale = target_scale / value._ckks_scale
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError("plaintext alignment scale is not finite and positive")
                nonzero = np.abs(weight[weight != 0])
                if nonzero.size and float(nonzero.min()) * scale < .5:
                    raise ValueError("plaintext coefficient would round away at this alignment scale")
                ciphertexts.append(value)
                plaintexts.append(arithmetic._encode_public(weight, level, scale))
            term = self._sum(ciphertexts, plaintexts, level)
            result = term if result is None else self.ctx.he_add[level].add(result, term)
        if np.any(bias):
            constant = arithmetic._encode_public(bias, level, result._ckks_scale)
            result = self.ctx.he_add[level].add_plain(result, constant)
        result = arithmetic._rescale.rescale(result, level, level-1)
        result._ckks_nsd = 1
        return arithmetic._finish(result)

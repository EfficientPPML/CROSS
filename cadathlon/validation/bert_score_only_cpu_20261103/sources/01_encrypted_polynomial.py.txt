"""Experimental ciphertext-only nonlinear building blocks for transformer work.

This is not a BERT/Qwen reference or a calibrated competition baseline.
Polynomial domains are public circuit contracts: ciphertext inputs cannot be
range-checked by the evaluator. Applications must establish their bounds and
budget the entire circuit, including approximation error and refresh.

The evaluator accepts an initialized, public/evaluation-key-only CKKSContext.
It never encrypts, decrypts, inspects slots, or invokes a client refresh. The
Chebyshev interpolation utility operates only on public function definitions.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Callable
import weakref

import numpy as np


@dataclass(frozen=True)
class PolynomialApproximation:
    name: str
    domain: tuple[float, float]
    coefficients: tuple[float, ...]
    input_power_basis: bool = False

    def __post_init__(self):
        lo, hi = self.domain
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise ValueError("approximation domain must be a finite increasing interval")
        if not self.coefficients or not all(math.isfinite(x) for x in self.coefficients):
            raise ValueError("finite Chebyshev coefficients required")
        # This opt-in removes a real domain-mapping multiplication only for
        # a centered quadratic. Its raw x and x^2 intermediates are bounded
        # by 2 and 4 on the declared domain. Shifted inverse-square-root
        # domains must retain their explicit affine transformation.
        if self.input_power_basis and (lo != -hi or hi > 2 or len(self.coefficients) != 3):
            raise ValueError("input power basis requires a centered quadratic domain bounded by [-2, 2]")

    @classmethod
    def interpolate(cls, name: str, function: Callable, domain, degree: int):
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
            raise ValueError("polynomial degree must be a positive integer")
        domain = tuple(float(x) for x in domain)
        # Validate before invoking the caller's public function.
        cls(name, domain, (0.,))
        series = np.polynomial.Chebyshev.interpolate(function, degree, domain=domain)
        # Remove interpolation roundoff (for example odd GELU coefficients
        # beyond its linear term), not meaningful tiny-amplitude functions.
        threshold = float(np.max(np.abs(series.coef))) * 32 * np.finfo(float).eps
        return cls(name, domain, tuple(0. if abs(c) < threshold else float(c) for c in series.coef))

    @property
    def depth(self):
        """Levels consumed by this explicit (deliberately conservative) plan."""
        def term_depth(n):
            if n < 2:
                return 0
            return 2 + max(term_depth(n // 2), term_depth((n + 1) // 2))
        return 2 + max(term_depth(n) for n in range(len(self.coefficients)))

    def evaluate_public(self, values):
        """Public numerical planning only; never used by the HE evaluator."""
        values = np.asarray(values, dtype=np.float64)
        lo, hi = self.domain
        if not np.isfinite(values).all() or np.any(values < lo) or np.any(values > hi):
            raise ValueError("public planning values outside approximation domain")
        return np.polynomial.chebyshev.chebval((2 * values - lo - hi) / (hi - lo), self.coefficients)


class EncryptedPolynomialEvaluator:
    """Compose real CKKS primitives with explicit scale/level alignment.

    A weighted sum aligns operands using real plaintext multiplication and
    rescaling, rather than rewriting scale metadata. This costs one level.
    Chebyshev products use a balanced tree, consuming logarithmic depth.
    """

    def __init__(self, context, *, strategy="chebyshev", low_memory=False, progress=None, refresher=None,
                 compiled_multiply=False, compiled_linear=False):
        if strategy not in ("chebyshev", "power"):
            raise ValueError("polynomial strategy must be chebyshev or power")
        self.strategy = strategy
        self.low_memory = bool(low_memory)
        self.progress = progress
        from jaxite_word import CKKSContext, he_params
        if not isinstance(context, CKKSContext):
            raise TypeError("expected CKKSContext")
        cache = getattr(context, "_param_cache", None)
        if cache is None:
            raise ValueError("evaluator context must be initialized")
        if (context.secret_key is not None or cache.secret_key is not None
                or context.parameters.get("secret_key") is not None):
            raise ValueError("evaluator must not hold a secret key")
        target = he_params.SecurityTarget(128, he_params.CostModel.CLASSICAL)
        if not he_params.qp_is_secure(context.q_towers, context.p_towers, context.degree, target):
            raise ValueError("polynomial evaluator requires a 128-bit-secure Q/P ring")
        if context.composite_degree != 2:
            raise ValueError("experimental evaluator currently requires composite_degree=2")
        if not cache.ff_q_max.computation_format_is_standard:
            raise ValueError("experimental scalar encoder requires standard computation format")
        self.ctx = context
        if refresher is not None:
            from encrypted_refresh import SecureRefresh
            if not isinstance(refresher, SecureRefresh) or refresher.ctx is not context:
                raise TypeError("refresh must be a SecureRefresh on the same public evaluator context")
        self.refresher = refresher
        self._constant_cache = OrderedDict()
        self._live_ntt_contexts = weakref.WeakValueDictionary()
        self._multiplication_bindings = {}
        self._multiplication_compiler = None
        if compiled_multiply:
            from encrypted_compilation import ExplicitArrayCompiler
            self._multiplication_compiler = ExplicitArrayCompiler()
        self._plaintext_multiply = self._rescale = None
        if compiled_linear:
            from encrypted_primitives import CompiledPlaintextMultiply, CompiledRescale
            self._plaintext_multiply = CompiledPlaintextMultiply(context)
            self._rescale = CompiledRescale(context)
        self._active_level = None
        if self.low_memory:
            # A sliced BAT NTT table is a materialized device allocation. The
            # dedicated reference process must not retain four inactive levels.
            cache._level_params_max = 1
        self.scale = float(context.scaling_factor)
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("finite positive CKKS scale required")

    def _finish(self, value):
        if self.low_memory:
            import jax
            # Flattening Polynomial itself registers its NTT context in a
            # process-global pytree registry. Synchronize the arrays directly
            # so a barrier does not accidentally keep every old level alive.
            values = value if isinstance(value, (list, tuple)) else (value,)
            jax.block_until_ready([item.polynomial for item in values])
            for item in values:
                # Different features at the same level must not retain separate
                # GiB-sized copies of identical public transform tables. Only
                # completed values share metadata; ciphertext payload/scale/
                # noise degree and modulus order are never changed here.
                context = item.ntt_ctx
                key = (tuple(int(q) for q in item.moduli), item.degree_layout,
                       item.precision, type(context), type(context.ff_ctx),
                       context.ff_ctx.computation_format_is_standard,
                       frozenset(item.polynomial.devices()))
                canonical = self._live_ntt_contexts.get(key)
                if canonical is None:
                    self._live_ntt_contexts[key] = context
                else:
                    item.ntt_ctx = canonical
        return value

    def _clear_working_caches(self):
        """Release reconstructible tables after all previous work has finished.

        Like CROSS Bootstrap's low-memory mode, this clears process-wide JAX
        compilation caches. It trades recompilation for a bounded live set and
        must only be used in a dedicated evaluator process.
        """
        import gc
        import importlib
        import jax
        from jaxite_word import polynomial
        # Internal primitives historically import flat ntt_mm. Its cache can
        # differ from jaxite_word.ntt_mm; resolve the module actually in use.
        ntt_module = importlib.import_module(type(self.ctx._param_cache.ntt_q_max).__module__)
        self._multiplication_bindings.clear()
        for helper in (self._plaintext_multiply, self._rescale):
            if helper is not None:
                helper.clear()
        for name in ('_he_add', '_he_sub', '_he_mul', '_he_rot', '_he_rescale',
                     '_he_level_reduce', '_ptct_mul'):
            accessor = getattr(self.ctx, name, None)
            instances = getattr(accessor, '_instances', None)
            if instances is not None:
                instances.clear()
        self._constant_cache.clear()
        self.ctx._param_cache._formatted_rotation_keys.clear()
        self.ctx._param_cache.level_params.clear()
        jax.clear_caches()
        # Compiled pytrees have been discarded. Live ciphertexts retain their
        # own context, and register again when passed to a new JAX transform.
        polynomial._POLYNOMIAL_REGISTRY.clear()
        ntt_module._barrett_ntt_context_cache.clear()
        gc.collect()

    def _prepare(self, level):
        if self.low_memory and self._active_level != level:
            self._clear_working_caches()
            self._active_level = level

    def checkpoint(self, value, label):
        """Synchronize and release caches between model phases."""
        self._finish(value)
        if self.low_memory:
            self._clear_working_caches()
            self._active_level = None
        if self.progress is not None:
            self.progress(label)
        return value

    def _level(self, value):
        from support import require_secure_execution
        require_secure_execution(self, value)
        scale = getattr(value, "_ckks_scale", None)
        if scale is None or not math.isfinite(scale) or scale <= 0:
            raise ValueError("ciphertext must carry a finite positive scale")
        if getattr(value, "_ckks_nsd", 1) != 1:
            raise ValueError("ciphertext must be rescaled to noise-scale degree 1")
        return self.ctx._param_cache.level_for_num_q(value.num_moduli)

    def _ensure_capacity(self, values, required):
        """Refresh only exhausted operands, preserving level one for bootstrap.

        The optional adapter owns a real evaluation-key-only Bootstrap engine;
        arbitrary callbacks and client refresh functions are not accepted.
        """
        values = tuple(values)
        if self.refresher is None:
            return values
        minimum = required + 1
        output = []
        refreshed = {}
        for value in values:
            if self._level(value) < minimum:
                if self.refresher.output_level < minimum:
                    raise ValueError(f"refresh supplies {self.refresher.output_level} levels; operation needs {minimum}")
                original = value
                if id(original) not in refreshed:
                    self.checkpoint(value, 'homomorphic refresh starting')
                    value = self.refresher.refresh(value)
                    self.checkpoint(value, 'homomorphic refresh complete')
                    refreshed[id(original)] = value
                else:
                    value = refreshed[id(original)]
                if self._level(value) < minimum:
                    raise ValueError('homomorphic refresh returned insufficient capacity')
            output.append(value)
        return tuple(output)

    def _lower(self, value, level):
        source = self._level(value)
        if level > source:
            raise ValueError("cannot raise a ciphertext level without homomorphic refresh")
        if level == source:
            return value
        return self.ctx.he_level_reduce[source, level].level_reduce(value)

    def _public_slots(self, value):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(self.ctx.num_slots, array, dtype=np.float64)
        if array.shape != (self.ctx.num_slots,) or not np.isfinite(array).all():
            raise ValueError("public coefficients must be finite scalars or one full slot vector")
        return array

    def add(self, left, right, *, subtract=False):
        level = min(self._level(left), self._level(right))
        self._prepare(level)
        left, right = self._lower(left, level), self._lower(right, level)
        return self._finish(self.ctx.he_sub[level].sub(left, right) if subtract
                            else self.ctx.he_add[level].add(left, right))

    def rotate(self, value, shift):
        if type(shift) is not int:
            raise ValueError("rotation shift must be an integer")
        if shift % self.ctx.num_slots == 0:
            return value
        level = self._level(value)
        self._prepare(level)
        result = self._finish(self.ctx.he_rot[level, shift].rotate(value))
        if self.low_memory:
            # Each rotation owns a distinct large switching key even at a
            # fixed level. Do not accumulate a whole transpose's key set.
            self._clear_working_caches()
            self._active_level = None
        return result

    def _encode_public(self, slots, level, scale):
        """Encode public constants; uniform slots have a constant NTT image."""
        import jax.numpy as jnp
        from jaxite_word import Polynomial, ckks_ctx
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("public encoding scale must be finite and positive")
        if np.all(slots == slots[0]):
            scalar = float(slots[0])
            cache_key = (level, scale, scalar)
            cached = self._constant_cache.get(cache_key)
            if cached is not None:
                # The scalar and PTCT-facade caches can outlive the separate
                # level-parameter LRU. Reinitialize that public level before
                # the eager facade validates the cached plaintext operand.
                self.ctx._param_cache.get_sliced_ff_q(level)
                self._constant_cache.move_to_end(cache_key)
                return cached
            scaled = scalar * scale
            if not math.isfinite(scaled):
                raise ValueError("public scalar encoding overflows the supported floating scale")
            # A constant in every CKKS slot is the constant polynomial.
            # Its NTT is that same rounded coefficient at every evaluation
            # point; no host FFT/NTT is needed for a dense scalar weight.
            coefficient = ckks_ctx.nearest_int(scaled)
            moduli = self.ctx._param_cache.q_moduli_at_level(level)
            residues = jnp.asarray([coefficient % modulus for modulus in moduli], dtype=jnp.uint32)
            payload = jnp.broadcast_to(residues, (1, 1, *self.ctx.degree_layout, len(moduli)))
            result = Polynomial.from_array(payload, {
                'batch': 1, 'num_elements': 1, 'degree': self.ctx.degree,
                'num_moduli': len(moduli), 'precision': 32, 'degree_layout': self.ctx.degree_layout,
            }, {'moduli': moduli, 'ntt_ctx': self.ctx._param_cache.get_sliced_ntt_q(level)})
            result._ckks_scale = float(scale)
            self._constant_cache[cache_key] = result
            # Large-ring plaintexts can occupy tens of MB each; never cache
            # an entire transformer model's unique weights on the device.
            if len(self._constant_cache) > 8:
                self._constant_cache.popitem(last=False)
            return result
        return self.ctx.encode_at_level(slots, level, scale=scale)

    def weighted_sum(self, values, weights, *, bias=0.):
        values, weights = tuple(values), tuple(self._public_slots(w) for w in weights)
        bias = self._public_slots(bias)
        if not values or len(values) != len(weights):
            raise ValueError("nonempty values and equally sized public weights required")
        values = self._ensure_capacity(values, 1)
        level = min(self._level(value) for value in values)
        if level < 1:
            raise ValueError("insufficient levels for weighted sum")
        self._prepare(level)
        divisor = math.prod(self.ctx._param_cache.q_moduli_at_level(level)[-2:])
        output_scale_before_rescale = self.scale * divisor
        result = None
        for value, weight in zip(values, weights):
            value = self._lower(value, level)
            encode_scale = output_scale_before_rescale / value._ckks_scale
            if not math.isfinite(encode_scale) or encode_scale <= 0:
                raise ValueError("plaintext alignment scale is not finite and positive")
            nonzero = np.abs(weight[weight != 0])
            if nonzero.size and float(nonzero.min()) * encode_scale < .5:
                raise ValueError("plaintext coefficient would round away at this alignment scale")
            plaintext = self._encode_public(weight, level, encode_scale)
            term = (self.ctx.ptct_mul[level].mul(value, plaintext) if self._plaintext_multiply is None
                    else self._plaintext_multiply.multiply(value, plaintext, level))
            result = term if result is None else self.ctx.he_add[level].add(result, term)
        if np.any(bias):
            constant = self._encode_public(bias, level, result._ckks_scale)
            result = self.ctx.he_add[level].add_plain(result, constant)
        result = (self.ctx.he_rescale[level, level - 1].rescale(result) if self._rescale is None
                  else self._rescale.rescale(result, level, level - 1))
        # CROSS's explicit rescale facade tracks scale but preserves nsd.
        # This schedule has exactly one plaintext multiply followed by one
        # composite rescale, so the output is again at noise-scale degree 1.
        result._ckks_nsd = 1
        return self._finish(result)

    def multiply(self, left, right):
        left, right = self._ensure_capacity((left, right), 1)
        level = min(self._level(left), self._level(right))
        if level < 1:
            raise ValueError("insufficient levels for ciphertext multiplication")
        self._prepare(level)
        left, right = self._lower(left, level), self._lower(right, level)
        operation = self.ctx.he_mul[level - 1]
        if self._multiplication_compiler is None:
            return self._finish(operation.mul(left, right))
        import finite_field
        cache = self.ctx._param_cache
        moduli = cache.q_moduli_at_level(level)
        for operand in (left, right):
            finite_field.check_ct_operand(cache.ff_context_cls, len(moduli), operand,
                'encrypted polynomial compiled multiply', batch=cache.batch,
                num_elements=2, degree_layout=cache.degree_layout, moduli=moduli)
        devices = left.polynomial.devices() | right.polynomial.devices()
        if len(devices) != 1:
            raise ValueError('experimental compiled multiply requires one execution device')
        binding = self._multiplication_bindings.get(level)
        if binding is None:
            binding = self._multiplication_compiler.bind(operation._mul_array,
                (left.polynomial, right.polynomial), signature=('leveled-multiply', level),
                device=next(iter(devices)))
            self._multiplication_bindings[level] = binding
        payload = binding(left.polynomial, right.polynomial)
        result = left._clone_with_payload(payload, moduli=cache.q_moduli_at_level(level - 1),
                                          ntt_ctx=cache.get_sliced_ntt_q(level - 1))
        # The compiled primitive performed an actual tensor multiplication,
        # relinearization and composite rescale, exactly as .mul does.
        result._ckks_scale = left._ckks_scale * right._ckks_scale / math.prod(moduli[-2:])
        result._ckks_nsd = 1
        return self._finish(result)

    def polynomial_depth(self, approximation):
        if self.strategy == "power":
            degree = len(approximation.coefficients) - 1
            return 2 - int(approximation.input_power_basis) + ((degree - 1).bit_length() if degree > 1 else 0)
        return approximation.depth

    def evaluate(self, value, approximation, *, output_weights=1.):
        required = self.polynomial_depth(approximation)
        (value,) = self._ensure_capacity((value,), required)
        level = self._level(value)
        if level < required:
            raise ValueError(f"{approximation.name} needs {required} levels; only {level} remain")
        output_weights = self._public_slots(output_weights)
        lo, hi = approximation.domain
        direct_input = self.strategy == 'power' and approximation.input_power_basis
        t = value if direct_input else self.weighted_sum([value], [2 / (hi - lo)], bias=-(hi + lo) / (hi - lo))
        del value
        terms = {1: t}
        if self.strategy == "power":
            # Balanced powers avoid scale-sensitive Chebyshev subtractions.
            # Restrict conditioning explicitly: large-degree monomial forms
            # can lose many bits through cancellation despite shallow depth.
            coefficients = np.polynomial.chebyshev.cheb2poly(approximation.coefficients)
            if direct_input:
                # For a centered domain t=x/hi, fold its public scaling into
                # the final coefficients without changing ciphertext metadata.
                coefficients = coefficients / hi ** np.arange(len(coefficients))
            if len(coefficients) > 33 or np.abs(coefficients).sum() > 1e8:
                raise ValueError("power strategy exceeds the supported coefficient-conditioning budget")

            def power(n):
                if n not in terms:
                    terms[n] = self.multiply(power(n // 2), power((n + 1) // 2))
                return terms[n]

            indices = [n for n in range(1, len(coefficients)) if coefficients[n] != 0]
            if not indices:
                return self.weighted_sum([t], [0.], bias=coefficients[0] * output_weights)
            return self.weighted_sum([power(n) for n in indices],
                                     [coefficients[n] * output_weights for n in indices],
                                     bias=coefficients[0] * output_weights)

        def chebyshev(n):
            if n in terms:
                return terms[n]
            left, right = n // 2, (n + 1) // 2
            product = self.multiply(chebyshev(left), chebyshev(right))
            # T_(2k)=2*T_k^2-1, T_(2k+1)=2*T_k*T_(k+1)-T_1.
            terms[n] = (self.weighted_sum([product], [2.], bias=-1.) if n % 2 == 0
                        else self.weighted_sum([product, t], [2., -1.]))
            return terms[n]

        # T_0 is the public constant one, added as a plaintext bias.
        if len(approximation.coefficients) == 1:
            return self.weighted_sum([t], [0.], bias=approximation.coefficients[0] * output_weights)
        values = [chebyshev(n) for n in range(1, len(approximation.coefficients))]
        return self.weighted_sum(values, [c * output_weights for c in approximation.coefficients[1:]],
                                 bias=approximation.coefficients[0] * output_weights)

    def normalize(self, features, inverse_sqrt, *, epsilon, center=False):
        """RMSNorm/LayerNorm core for one ciphertext per feature, SIMD rows.

        Each feature ciphertext packs independent rows in corresponding slots.
        Returns normalized features before the learned public affine transform.
        ``inverse_sqrt.domain`` must bound the public circuit's variance+epsilon.
        This reference layout prioritizes correctness; it is not yet efficient
        enough for full transformer workloads.
        """
        features = tuple(features)
        if not features or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("nonempty features and finite positive epsilon required")
        if inverse_sqrt.domain[0] <= 0:
            raise ValueError("inverse square root requires a strictly positive domain")
        # Centering can subtract at the mean's level without another scale
        # multiplication when its inputs use this evaluator's canonical scale.
        if center and any(not math.isclose(value._ckks_scale, self.scale, rel_tol=1e-12)
                          for value in features):
            raise ValueError("LayerNorm inputs need canonical scale; align with weighted_sum first")
        required = self.polynomial_depth(inverse_sqrt) + 3 + (1 if center else 0)
        if self.refresher is None and min(self._level(value) for value in features) < required:
            raise ValueError(f"normalization needs {required} levels before its learned affine transform")
        if center:
            features = self._ensure_capacity(features, 1)
            mean = self.weighted_sum(features, [1 / len(features)] * len(features))
            mean_level = self._level(mean)
            features = tuple(self.ctx.he_sub[mean_level].sub(self._lower(value, mean_level), mean)
                             for value in features)
        squares = [self.multiply(value, value) for value in features]
        variance = self.weighted_sum(squares, [1 / len(features)] * len(features), bias=epsilon)
        del squares
        inverse = self.evaluate(variance, inverse_sqrt)
        # Preserve room for the learned affine transform. Refreshing this one
        # shared inverse can avoid refreshing every normalized feature later.
        (inverse,) = self._ensure_capacity((inverse,), 2)
        return tuple(self.multiply(value, inverse) for value in features)


def nonlinear_approximation(name, domain, degree):
    """Interpolate a named public function on an explicit finite interval.

    Interpolation does not establish a certified global error bound. Validate
    approximation and HE error separately over the application's full domain.
    """
    functions = {
        "inverse_sqrt": lambda x: 1 / np.sqrt(x),
        "reciprocal": lambda x: 1 / x,
        "exp": np.exp,
        "tanh": np.tanh,
        "silu": lambda x: x * np.exp(-np.logaddexp(0, -x)),
        "gelu": lambda x: .5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2))),
    }
    if name not in functions:
        raise ValueError(f"unsupported nonlinear function: {name}")
    if name in ("inverse_sqrt", "reciprocal") and domain[0] <= 0:
        raise ValueError(f"{name} requires a strictly positive domain")
    return PolynomialApproximation.interpolate(name, functions[name], domain, degree)

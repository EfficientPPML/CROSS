"""Experimental full-width BERT LayerNorm on distinct row tiles.

The caller establishes the public inverse-square-root domain. This helper
cannot range-check ciphertext slots and never invokes a codec or refresh.
It borrows a keyless context; its cache policy needs a dedicated serial process.
"""
from dataclasses import dataclass
import math

import numpy as np

from encrypted_polynomial import PolynomialApproximation
from encrypted_streaming_linear import RowTileLayout


@dataclass(frozen=True)
class RowNormalizationResult:
    output: object
    checkpoints: dict


def _vector(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in 'fiu' or value.shape != (768,) or not np.isfinite(value).all():
        raise ValueError(name + ' must be a finite real vector of width768')
    result = value.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(name + ' exceeds float64 range')
    return result


class RowLayerNorm:
    """One secure batch-one ciphertext with 768 features at pitch4096.

    Nonzero input padding is ignored by the row moments. Variance padding is
    filled with valid-row variances or a public safe midpoint before evaluating
    the fixed inverse polynomial. Only the final affine guarantees zero ideal
    output padding. Captured intermediates are ciphertexts, never decoded data.
    """

    def __init__(self, context, layout, *, low_memory=True, preserve_dispatch_cache=False):
        if not isinstance(layout, RowTileLayout) or layout.features != 4096:
            raise ValueError('RowLayerNorm requires RowTileLayout with pitch4096')
        if type(low_memory) is not bool or type(preserve_dispatch_cache) is not bool:
            raise TypeError('memory options must be explicit booleans')
        if context.num_slots != layout.slots or context.degree != 2 * layout.slots:
            raise ValueError('row layout must cover every physical CKKS slot')
        cache = getattr(context, '_param_cache', None)
        if cache is not None and cache.batch != 1:
            raise ValueError('row normalization requires a batch-one context')
        from encrypted_packed_qwen import CompiledArithmetic
        self.ctx, self.layout = context, layout
        self.arithmetic = CompiledArithmetic(context, low_memory=low_memory,
            preserve_dispatch_cache=preserve_dispatch_cache)

    @staticmethod
    def required_rotation_indices(layout):
        if not isinstance(layout, RowTileLayout) or layout.features != 4096:
            raise ValueError('RowLayerNorm requires RowTileLayout with pitch4096')
        forward = {1 << bit for bit in range(10)}
        backward = {(-(1 << bit)) % layout.slots for bit in range(12)}
        return tuple(sorted(forward | backward))

    def _validate_ciphertext(self, value, required):
        from jaxite_word import Polynomial
        from support import require_secure_execution
        if (self.ctx.secret_key is not None or self.ctx._param_cache.secret_key is not None
                or self.ctx.parameters.get('secret_key') is not None):
            raise ValueError('row normalization context must remain keyless')
        if (not isinstance(value, Polynomial) or value.batch != 1 or value.num_elements != 2
                or value.degree != 2 * self.layout.slots):
            raise ValueError('one canonical ciphertext with the declared ring degree required')
        require_secure_execution(self.ctx, value)
        level = self.arithmetic._level(value)
        if level < required + 1:
            raise ValueError('row normalization needs its full circuit plus one reserved level')
        if not math.isclose(value._ckks_scale, self.arithmetic.scale, rel_tol=1e-12, abs_tol=0.):
            raise ValueError('row normalization input must have the canonical scale')

    def _prefix768(self, value):
        arithmetic = self.arithmetic
        block = value
        for bit in range(8):
            block = arithmetic.add(block, arithmetic.rotate(block, 1 << bit))
        first512 = arithmetic.add(block, arithmetic.rotate(block, 256))
        return arithmetic.add(first512, arithmetic.rotate(block, 512))

    def _broadcast(self, value, *, full):
        arithmetic = self.arithmetic
        block = value
        for bit in range(12 if full else 8):
            block = arithmetic.add(block, arithmetic.rotate(block, -(1 << bit)))
        if full:
            return block
        first512 = arithmetic.add(block, arithmetic.rotate(block, -256))
        return arithmetic.add(first512, arithmetic.rotate(block, -512))

    def normalize_tile(self, value, gamma, beta, approximation, *, epsilon, valid_rows, capture=False):
        self.layout._valid_rows(valid_rows)
        if type(capture) is not bool:
            raise TypeError('capture must be an explicit boolean')
        gamma, beta = _vector(gamma, 'gamma'), _vector(beta, 'beta')
        if (isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon) or epsilon <= 0):
            raise ValueError('epsilon must be finite and positive')
        if (not isinstance(approximation, PolynomialApproximation) or approximation.domain[0] <= 0
                or approximation.input_power_basis):
            raise ValueError('positive-domain inverse-square-root approximation required')
        if len(approximation.coefficients) > 33:
            raise ValueError('inverse polynomial exceeds the supported power conditioning budget')
        with np.errstate(over='ignore', invalid='ignore'):
            coefficients = np.polynomial.chebyshev.cheb2poly(approximation.coefficients)
        if not np.isfinite(coefficients).all() or np.abs(coefficients).sum() > 1e8:
            raise ValueError('inverse polynomial exceeds the supported power conditioning budget')
        arithmetic = self.arithmetic
        required = arithmetic.polynomial_depth(approximation) + 5
        self._validate_ciphertext(value, required)
        rows = self.layout.rows_per_tile
        anchors = np.zeros((rows, self.layout.features), dtype=np.float64)
        anchors[:valid_rows, 0] = 1 / 768
        mean = self._broadcast(arithmetic.weighted_sum([self._prefix768(value)], [anchors.ravel()]), full=False)
        centered = arithmetic.add(value, mean, subtract=True)
        squares = arithmetic.multiply(centered, centered)
        bias = np.zeros_like(anchors)
        bias[:valid_rows, 0] = epsilon
        bias[valid_rows:, 0] = sum(approximation.domain) / 2
        variance = self._broadcast(arithmetic.weighted_sum([self._prefix768(squares)],
            [anchors.ravel()], bias=bias.ravel()), full=True)
        inverse = arithmetic.evaluate(variance, approximation)
        normalized = arithmetic.multiply(centered, inverse)
        output = arithmetic.weighted_sum([normalized],
            [self.layout.feature(gamma, valid_rows=valid_rows)],
            bias=self.layout.feature(beta, valid_rows=valid_rows))
        checkpoints = {'mean': mean, 'variance': variance, 'inverse': inverse} if capture else {}
        # No interval bound or refresh contract is inferred for returned values.
        # Captured checkpoints also must not inherit an input's semantic bounds.
        for returned in (output, *checkpoints.values()):
            for name in tuple(vars(returned)):
                if name.startswith('_public_'):
                    delattr(returned, name)
        return RowNormalizationResult(output, checkpoints)

    def clear(self):
        """Clear shared working caches; ownership of the context stays external."""
        self.arithmetic._clear_working_caches()

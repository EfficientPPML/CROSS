"""Exact-real public-weight gate bound; no model execution or coefficient fit."""
from fractions import Fraction
import math
import numpy as np


def ceil_binary64(value):
    value = Fraction(value)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('finite representable public bound required')
    while Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def sqrt_upper(value):
    value = Fraction(value)
    if value < 0:
        raise ValueError('nonnegative square required')
    result = math.sqrt(ceil_binary64(value))
    while Fraction.from_float(result)**2 < value:
        result = math.nextafter(result, math.inf)
    return result


def weighted_rms_gate_bound(weight_row, gamma, *, relative_inverse_error=Fraction(0)):
    """Return an outward bound for W_j diag(gamma) x / sqrt(mean(x*x)+eps).

    The optional multiplier is conditional on a UNIFORM relative inverse-sqrt
    certificate over the actual variance domain. It does not certify floating
    execution, CKKS error, independently perturbed variance, or refresh.
    Public tensor values must be the exact pinned binary32 values.
    """
    row, gamma = np.asarray(weight_row), np.asarray(gamma)
    if (row.dtype != np.float32 or gamma.dtype != np.float32
            or row.ndim != 1 or row.shape != gamma.shape or not row.size
            or not np.isfinite(row).all() or not np.isfinite(gamma).all()):
        raise ValueError('matching finite public binary32 vectors required')
    if type(relative_inverse_error) is not Fraction or relative_inverse_error < 0:
        raise TypeError('explicit nonnegative rational uniform certificate required')
    # Every product/square/sum below is exact for the stored public bytes.
    squared = sum((Fraction.from_float(float(w))*Fraction.from_float(float(g)))**2
                  for w, g in zip(row, gamma))
    squared *= row.size
    base = sqrt_upper(squared)
    expanded = ceil_binary64(Fraction.from_float(base)*(1+relative_inverse_error))
    return {'width': row.size,
            'squared_native_bound_fraction': [str(squared.numerator), str(squared.denominator)],
            'native_exact_real_upper': base,
            'conditional_polynomial_upper': expanded,
            'relative_inverse_error_fraction': [str(relative_inverse_error.numerator), str(relative_inverse_error.denominator)]}

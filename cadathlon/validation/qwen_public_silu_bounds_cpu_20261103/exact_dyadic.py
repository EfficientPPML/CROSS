"""Bulk public binary32 products, followed by exact integer square/sums."""
from fractions import Fraction
import numpy as np
from public_bound import sqrt_upper


def rows(weight, gamma):
    weight, gamma = np.asarray(weight), np.asarray(gamma)
    if (weight.dtype != np.float32 or gamma.dtype != np.float32
            or weight.ndim != 2 or gamma.shape != (weight.shape[1],)
            or not weight.size or not np.isfinite(weight).all()
            or not np.isfinite(gamma).all()):
        raise ValueError('finite binary32 public matrix and gamma required')
    # Binary32 products have <=48 significand bits and exponent range safely
    # inside normal binary64 (smallest nonzero product is 2**-298). Thus this
    # conversion/multiplication is exact, even for binary32 subnormals.
    products = weight.astype(np.float64)*gamma.astype(np.float64)
    bits = np.ascontiguousarray(products, dtype=np.float64).view(np.uint64)
    exponents = ((bits >> np.uint64(52)) & np.uint64(2047)).astype(np.int64)
    mantissas = (bits & np.uint64((1 << 52)-1)) | np.uint64(1 << 52)
    zero = (bits & np.uint64((1 << 63)-1)) == 0
    if np.any((exponents == 0) & ~zero) or np.any(exponents == 2047):
        raise ValueError('unexpected non-normal/nonfinite binary64 product')
    mantissas[zero] = 0
    powers = 2*(exponents-1075)
    width = weight.shape[1]
    for mantissa, power in zip(mantissas, powers):
        active = mantissa != 0
        if not np.any(active):
            exact = Fraction(0)
        else:
            minimum = int(power[active].min())
            total = sum(m*m << (e-minimum) for m,e in
                        zip(mantissa[active].tolist(), power[active].tolist()))
            exact = (Fraction(width*total << minimum) if minimum >= 0
                     else Fraction(width*total, 1 << -minimum))
        yield exact, sqrt_upper(exact)

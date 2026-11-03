"""Isolated integer-correction Newton composition; no codec or refresh."""
import math
import numpy as np
from integer_affine import affine_at_scale,integer_affine


def integer_newton_updates(arithmetic, denominator, *, normalization_factor,
                           initial_estimate, updates, first_output_scale):
    """Yield every fixed public update; callers must exhaust for full coverage.

    The initial scale is realized by affine plaintext encoding and rescale.
    Later corrections retain their input scale and level. No convergence or
    encrypted-domain membership is inferred, and no plaintext is inspected.
    """
    if type(normalization_factor) is not int or not 1 <= normalization_factor < 2**64:
        raise ValueError('positive public integer normalization factor required')
    if type(updates) is not int or updates < 1:
        raise ValueError('positive public update count required')
    if isinstance(first_output_scale,bool) or not math.isfinite(first_output_scale) or first_output_scale <= 0:
        raise ValueError('finite positive planned first output scale required')
    initial=np.asarray(initial_estimate,dtype=np.float64)
    if initial.ndim not in (0,1) or (initial.ndim==1 and initial.shape!=(arithmetic.ctx.num_slots,)):
        raise ValueError('public estimate must be a scalar or one full slot vector')
    if not np.isfinite(initial).all() or not (initial>0).all():
        raise ValueError('public estimates must be positive and finite')
    with np.errstate(over='ignore',invalid='ignore'):
        weight=(-normalization_factor*initial)*initial
        bias=2.*initial
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('initial public affine coefficients overflow')
    weight,bias=(float(weight),float(bias)) if initial.ndim==0 else (weight.copy(),bias.copy())
    z=affine_at_scale(arithmetic,denominator,weight,bias,output_scale=first_output_scale)
    yield 1,z
    for update in range(2,updates+1):
        product=arithmetic.multiply(denominator,z)
        correction=integer_affine(arithmetic,product,-normalization_factor,bias=2.)
        z=arithmetic.multiply(z,correction)
        yield update,z

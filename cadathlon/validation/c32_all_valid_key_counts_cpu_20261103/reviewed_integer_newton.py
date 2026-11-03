"""Isolated static integer-correction Newton composition; no codec or refresh."""
import math
import numpy as np
from integer_affine import affine_at_scale,integer_affine,_positive_scale,_public_reals


def integer_newton_updates(arithmetic, denominator, *, normalization_factor,
                           initial_estimate, updates, first_output_scale):
    """Yield a fixed CD2/NSD1 leveled circuit; exhaust for complete coverage.

    The compiled arithmetic/context and denominator metadata must remain fixed
    while this generator is suspended. No refresher is permitted. Public scale
    recurrences are checked against real primitive outputs, never assigned to
    them. No encrypted-domain membership or convergence guarantee is inferred.
    """
    if type(normalization_factor) is not int or not 1 <= normalization_factor < 2**64:
        raise ValueError('positive public integer normalization factor required')
    if type(updates) is not int or updates < 1:
        raise ValueError('positive public update count required')
    first_output_scale=_positive_scale(first_output_scale,'planned first output scale')
    initial=_public_reals(initial_estimate,name='public initial estimate')
    if initial.ndim not in (0,1) or (initial.ndim==1 and initial.shape!=(arithmetic.ctx.num_slots,)):
        raise ValueError('public estimate must be a scalar or one full slot vector')
    if not (initial>0).all():raise ValueError('public estimates must be positive')
    with np.errstate(over='ignore',invalid='ignore',under='ignore'):
        weight=(-normalization_factor*initial)*initial
        bias=2.*initial
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError('initial public affine coefficients overflow')
    if np.any(weight==0):raise ValueError('nonzero initial coefficient underflowed to zero')
    weight,bias=(float(weight),float(bias)) if initial.ndim==0 else (weight.copy(),bias.copy())
    context=arithmetic.ctx;cache=context._param_cache
    if type(context.composite_degree) is not int or context.composite_degree!=2:
        raise ValueError('static Newton schedule requires composite degree2')
    if getattr(arithmetic,'refresher',None) is not None:
        raise ValueError('static Newton schedule forbids mid-circuit refresh')
    level=arithmetic._level(denominator)
    if type(getattr(denominator,'_ckks_nsd',None)) is not int or denominator._ckks_nsd!=1:
        raise ValueError('static Newton schedule requires explicit NSD1')
    source_scale=_positive_scale(denominator._ckks_scale,'denominator scale')
    if level<2*updates-1:raise ValueError('insufficient fixed levels for all Newton updates')
    q=tuple(cache.q_moduli_at_level(level))
    if len(q)!=2*(level+1):raise ValueError('noncanonical CD2 modulus schedule')
    def divisor(L):return math.prod(q[2*L:2*L+2])
    encode_scale=_positive_scale(first_output_scale*divisor(level)/source_scale,'initial plaintext scale')
    actual_scale=_positive_scale(source_scale*encode_scale/divisor(level),'first output scale')
    rows=[(level-1,actual_scale,None)]
    for _ in range(1,updates):
        L=rows[-1][0]
        product_scale=_positive_scale(source_scale*actual_scale/divisor(L),'planned product scale')
        actual_scale=_positive_scale(actual_scale*product_scale/divisor(L-1),'planned update scale')
        rows.append((L-2,actual_scale,product_scale))
    def static():
        if (arithmetic.ctx is not context or context._param_cache is not cache
                or context.composite_degree!=2 or getattr(arithmetic,'refresher',None) is not None
                or tuple(cache.q_moduli_at_level(level))!=q
                or arithmetic._level(denominator)!=level
                or denominator._ckks_scale!=source_scale
                or type(getattr(denominator,'_ckks_nsd',None)) is not int or denominator._ckks_nsd!=1):
            raise ValueError('static context, denominator or refresh assumptions changed')
    def check(value,L,scale):
        static()
        if (arithmetic._level(value)!=L or value._ckks_scale!=scale
                or type(getattr(value,'_ckks_nsd',None)) is not int or value._ckks_nsd!=1):
            raise ValueError('actual primitive metadata differs from static scale/level schedule')
    static()
    z=affine_at_scale(arithmetic,denominator,weight,bias,output_scale=first_output_scale)
    check(z,*rows[0][:2]);yield 1,z
    for update in range(2,updates+1):
        static();expected_level,expected_scale,product_scale=rows[update-1]
        check(z,*rows[update-2][:2])
        product=arithmetic.multiply(denominator,z)
        check(product,expected_level+1,product_scale)
        correction=integer_affine(arithmetic,product,-normalization_factor,bias=2.)
        check(correction,expected_level+1,product_scale)
        z=arithmetic.multiply(z,correction)
        check(z,expected_level,expected_scale)
        yield update,z

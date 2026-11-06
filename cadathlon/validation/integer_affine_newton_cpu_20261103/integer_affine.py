"""Isolated feasibility helpers; no refresh or inferred encrypted-value bounds."""
import copy
from decimal import Decimal, localcontext
import math


def integer_affine(arithmetic, value, multiplier, *, bias=0.):
    """Apply an integer addition chain and a normally encoded real bias.

    Level/scale/NSD stay unchanged. Integer residue arithmetic is exact; the
    plaintext bias still has ordinary CKKS encoding error. Existing semantic
    public contracts are removed from the returned wrapper, never the input.
    """
    if type(multiplier) is not int or abs(multiplier).bit_length() > 64:
        raise ValueError('multiplier must be a public signed integer of at most 64 bits')
    if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(bias):
        raise ValueError('bias must be a finite public real scalar')
    context = arithmetic.ctx
    if (context.secret_key is not None or context._param_cache.secret_key is not None
            or context.parameters.get('secret_key') is not None):
        raise ValueError('integer affine requires a keyless initialized server')
    level = arithmetic._level(value)
    arithmetic._prepare(level)
    # Even the identity and zero paths use the strict current-context facade
    # to validate layout, residue representation, batch and ordered moduli.
    zero = context.he_sub[level].sub(value, value)
    magnitude = abs(multiplier)
    result = zero if not magnitude else copy.copy(value)
    for bit in bin(magnitude)[3:]:
        result = context.he_add[level].add(result, result)
        if bit == '1':
            result = context.he_add[level].add(result, value)
    if multiplier < 0:
        result = context.he_sub[level].sub(zero, result)
    if bias:
        plaintext = arithmetic._encode_public(arithmetic._public_slots(bias), level, result._ckks_scale)
        result = context.he_add[level].add_plain(result, plaintext)
    for name in tuple(vars(result)):
        if name.startswith('_public_'):
            delattr(result, name)
    return arithmetic._finish(result)


def backward_scale_plan(q_moduli, *, input_scale, first_output_level, updates, final_scale):
    """Solve public two-level Newton scale recurrence, then simulate floats.

    The initial affine consumes one level before first_output_level. This
    returns a target for a REAL affine encoding/rescale, never a scale edit.
    All later metadata entries are the exact binary64 arithmetic recurrence.
    """
    q = tuple(q_moduli)
    if (not q or len(q) % 2 or any(type(x) is not int or x <= 1 for x in q)
            or type(first_output_level) is not int
            or type(updates) is not int or updates < 1
            or first_output_level < 2*(updates-1)
            or first_output_level+1 >= len(q)//2):
        raise ValueError('invalid public composite-degree-two level schedule')
    for scale in (input_scale, final_scale):
        if isinstance(scale, bool) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('scales must be finite positive real values')
    def divisor(level):
        return q[2*level]*q[2*level+1]
    levels = tuple(first_output_level-2*index for index in range(updates))
    with localcontext() as context:
        context.prec = 100
        sigma = Decimal.from_float(float(final_scale))
        delta = Decimal.from_float(float(input_scale))
        for level in reversed(levels[:-1]):
            sigma = (sigma*Decimal(divisor(level))*Decimal(divisor(level-1))/delta).sqrt()
        initial_decimal = str(sigma)
        initial = float(sigma)
    # The initial PTCT and rescale operate in binary64 metadata as well.
    first_divisor = divisor(first_output_level+1)
    encode_scale = initial*first_divisor/float(input_scale)
    actual = float(input_scale)*encode_scale/first_divisor
    rows = [dict(update=1,level=levels[0],scale=actual)]
    for index, level in enumerate(levels[:-1],2):
        product = float(input_scale)*actual/divisor(level)
        actual = actual*product/divisor(level-1)
        rows.append(dict(update=index,level=level-2,product_scale=product,scale=actual))
    if not all(math.isfinite(row['scale']) and row['scale'] > 0 for row in rows):
        raise ValueError('public scale recurrence is not representable')
    return dict(initial_scale_decimal=initial_decimal,initial_target_scale=initial,
                initial_plaintext_scale=encode_scale,rows=rows,
                requested_final_scale=float(final_scale),
                final_relative_difference=actual/float(final_scale)-1)


def affine_at_scale(arithmetic, value, weight, bias, *, output_scale):
    """One actual PTCT affine/rescale with a public planned output scale."""
    context = arithmetic.ctx
    if (context.secret_key is not None or context._param_cache.secret_key is not None
            or context.parameters.get('secret_key') is not None):
        raise ValueError('initial affine requires a keyless initialized server')
    level = arithmetic._level(value)
    if level < 1 or isinstance(output_scale, bool) or not math.isfinite(output_scale) or output_scale <= 0:
        raise ValueError('one level and finite positive output scale required')
    arithmetic._prepare(level)
    divisor = math.prod(arithmetic.ctx._param_cache.q_moduli_at_level(level)[-2:])
    encode_scale = output_scale*divisor/value._ckks_scale
    plaintext = arithmetic._encode_public(arithmetic._public_slots(weight), level, encode_scale)
    product = arithmetic._plaintext_multiply.multiply(value, plaintext, level)
    constant = arithmetic._encode_public(arithmetic._public_slots(bias), level, product._ckks_scale)
    product = arithmetic.ctx.he_add[level].add_plain(product, constant)
    result = arithmetic._rescale.rescale(product, level, level-1)
    # One actual PTCT followed by one actual composite rescale.
    result._ckks_nsd = 1
    for name in tuple(vars(result)):
        if name.startswith('_public_'):
            delattr(result, name)
    return arithmetic._finish(result)

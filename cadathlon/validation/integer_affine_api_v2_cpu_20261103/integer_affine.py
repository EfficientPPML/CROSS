"""Isolated feasibility helpers; no refresh or inferred encrypted-value bounds."""
import copy
from decimal import Decimal, localcontext
import math
import numpy as np


def _positive_scale(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(name + " must be a finite positive real scale")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(name + " is not representable as binary64")
    return result


def _public_reals(value, *, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(name + " must be real, excluding bool/complex/object values")
    if isinstance(value, (list, tuple)) and any(isinstance(v, (bool, np.bool_, complex, np.complexfloating)) for v in np.asarray(value, dtype=object).flat):
        raise ValueError(name + " must exclude bool/complex values")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError(name + " must be finite and representable as float64")
    return result


def _encode_checked(arithmetic, slots, level, scale):
    scale = _positive_scale(scale, "public encoding scale")
    nonzero = np.abs(slots[slots != 0])
    if nonzero.size:
        if float(nonzero.min()) * scale < .5:
            raise ValueError("nonzero public coefficient would round away at this encoding scale")
        if not math.isfinite(float(nonzero.max()) * scale):
            raise ValueError("scaled public coefficient is not representable")
    plaintext = arithmetic._encode_public(slots, level, scale)
    # This is public data, never a ciphertext inspection. The slot guard alone
    # cannot exclude a whole nonuniform FFT image quantizing to zero.
    if nonzero.size and not np.any(np.asarray(plaintext.polynomial)):
        raise ValueError("nonzero public coefficients encoded to the zero plaintext")
    return plaintext


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
    if type(getattr(value, "_ckks_nsd", None)) is not int or value._ckks_nsd != 1:
        raise ValueError("explicit noise-scale degree1 required")
    _positive_scale(getattr(value, "_ckks_scale", None), "ciphertext scale")
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
        plaintext = _encode_checked(arithmetic, arithmetic._public_slots(bias), level, result._ckks_scale)
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
    input_scale = _positive_scale(input_scale, "input scale")
    final_scale = _positive_scale(final_scale, "final scale")
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
    _positive_scale(initial, "planned initial scale")
    _positive_scale(encode_scale, "planned plaintext scale")
    actual = float(input_scale)*encode_scale/first_divisor
    rows = [dict(update=1,level=levels[0],scale=actual)]
    for index, level in enumerate(levels[:-1],2):
        product = float(input_scale)*actual/divisor(level)
        actual = actual*product/divisor(level-1)
        rows.append(dict(update=index,level=level-2,product_scale=product,scale=actual))
    if not all(math.isfinite(value) and value > 0 for row in rows for name, value in row.items() if name in ("scale", "product_scale")):
        raise ValueError('public scale recurrence is not representable')
    return dict(initial_scale_decimal=initial_decimal,initial_target_scale=initial,
                initial_plaintext_scale=encode_scale,rows=rows,
                requested_final_scale=float(final_scale),
                final_relative_difference=actual/float(final_scale)-1)


def affine_at_scale(arithmetic, value, weight, bias, *, output_scale):
    """One actual PTCT affine/rescale, requiring compiled CD2/NSD1 helpers."""
    context = arithmetic.ctx
    if (context.secret_key is not None or context._param_cache.secret_key is not None
            or context.parameters.get('secret_key') is not None):
        raise ValueError('initial affine requires a keyless initialized server')
    if type(getattr(value, "_ckks_nsd", None)) is not int or value._ckks_nsd != 1:
        raise ValueError("explicit noise-scale degree1 required")
    _positive_scale(getattr(value, "_ckks_scale", None), "ciphertext scale")
    level = arithmetic._level(value)
    output_scale = _positive_scale(output_scale, "planned output scale")
    if level < 1:
        raise ValueError("one level required for initial affine")
    if getattr(context, "composite_degree", None) != 2:
        raise ValueError("initial affine requires composite degree2")
    if (getattr(arithmetic, "_plaintext_multiply", None) is None or getattr(arithmetic, "_rescale", None) is None):
        raise ValueError("initial affine requires compiled plaintext/rescale helpers")
    weight = arithmetic._public_slots(_public_reals(weight, name="affine weight"))
    bias = arithmetic._public_slots(_public_reals(bias, name="affine bias"))
    arithmetic._prepare(level)
    divisor = math.prod(arithmetic.ctx._param_cache.q_moduli_at_level(level)[-2:])
    encode_scale = output_scale*divisor/value._ckks_scale
    plaintext = _encode_checked(arithmetic, weight, level, encode_scale)
    product = arithmetic._plaintext_multiply.multiply(value, plaintext, level)
    constant = _encode_checked(arithmetic, bias, level, product._ckks_scale)
    product = arithmetic.ctx.he_add[level].add_plain(product, constant)
    result = arithmetic._rescale.rescale(product, level, level-1)
    # One actual PTCT followed by one actual composite rescale.
    result._ckks_nsd = 1
    for name in tuple(vars(result)):
        if name.startswith('_public_'):
            delattr(result, name)
    return arithmetic._finish(result)

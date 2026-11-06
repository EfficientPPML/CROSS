"""Experimental fixed reciprocal polynomial with a separately tracked residual.

The exact-real recurrence matches Newton from the supplied rounded first affine.
CKKS errors in the two branches do not obey that identity automatically. This
module establishes no convergence, input-domain, residual or error guarantee.
No refresh, client codec, coefficient selection or metadata relabeling is used.
"""
from decimal import Decimal, localcontext
import math
import numpy as np
from encrypted_integer_affine import (
    affine_at_scale, integer_affine, _positive_scale, _public_reals,
    _require_unbounded_keyless_owner,
)


def _schedule(q_moduli, input_level, updates):
    q = tuple(q_moduli)
    if (type(input_level) is not int or type(updates) is not int or updates < 1
            or input_level < (1 if updates == 1 else updates + 1)
            or len(q) != 2 * (input_level + 1)
            or any(type(x) is not int or x <= 1 for x in q)
            or len(set(q)) != len(q)):
        raise ValueError('invalid fixed composite-degree-two level schedule')
    return q


def _forward(q, *, input_scale, input_level, updates, first_output_scale):
    def divisor(level):
        return q[2 * level] * q[2 * level + 1]
    encode = _positive_scale(first_output_scale * divisor(input_level) / input_scale,
                             'initial plaintext scale')
    y_scale = _positive_scale(input_scale * encode / divisor(input_level), 'actual first scale')
    y_level = input_level - 1
    rows = [dict(update=1, level=y_level, scale=y_scale)]
    residual = None
    if updates > 1:
        e_level = y_level - 1
        e_scale = _positive_scale(input_scale * y_scale / divisor(y_level), 'initial residual scale')
        residual = dict(level=e_level, scale=e_scale)
        rows[0].update(residual_level=e_level, residual_scale=e_scale)
        for update in range(2, updates + 1):
            y_scale = _positive_scale(y_scale * e_scale / divisor(e_level), 'updated estimate scale')
            y_level = e_level - 1
            row = dict(update=update, level=y_level, scale=y_scale)
            if update < updates:
                e_scale = _positive_scale(e_scale * e_scale / divisor(e_level), 'updated residual scale')
                e_level -= 1
                row.update(residual_level=e_level, residual_scale=e_scale)
            rows.append(row)
    return dict(first_output_scale=first_output_scale, initial_plaintext_scale=encode,
                initial_residual=residual, rows=rows)


def tracked_scale_plan(q_moduli, *, input_scale, input_level, updates, final_scale):
    """Solve both branch scales, then simulate the actual binary64 operations.

    This returns an encoding scale for a real initial affine. The final target
    is not assigned to any ciphertext; rows contain the actual floating scale
    recurrence and may differ slightly from that requested target.
    """
    q = _schedule(q_moduli, input_level, updates)
    delta = _positive_scale(input_scale, 'input scale')
    target = _positive_scale(final_scale, 'final scale')
    if updates == 1:
        first = target
        first_decimal = str(Decimal.from_float(target))
    else:
        with localcontext() as context:
            context.prec = 100
            # After the first parallel update, scale(y)/scale(e) is constant.
            ratio = Decimal(q[2*(input_level-1)] * q[2*(input_level-1)+1]) / Decimal.from_float(delta)
            error = Decimal.from_float(target) / ratio
            input_levels = range(input_level - 2, input_level - updates - 1, -1)
            for level in reversed(tuple(input_levels)):
                error = (error * Decimal(q[2*level] * q[2*level+1])).sqrt()
            first_decimal = str(error * ratio)
            first = _positive_scale(float(first_decimal), 'planned first scale')
    plan = _forward(q, input_scale=delta, input_level=input_level,
                    updates=updates, first_output_scale=first)
    plan.update(first_output_scale_decimal=first_decimal,
                requested_final_scale=target,
                final_relative_difference=plan['rows'][-1]['scale']/target-1,
                consumed_levels=1 if updates == 1 else updates+1)
    return plan


def tracked_reciprocal_updates(arithmetic, denominator, *, normalization_factor,
                               first_weight, first_bias, updates, first_output_scale):
    """Yield (update, y, e), with no unused final residual square.

    The supplied first affine is evaluated exactly as specified, including its
    rounded coefficients. For updates>1 the first yield includes e1=1-C*x*y1.
    Subsequent branches use the same old e: y'=y*(1+e), e'=e*e. Yielded wrappers
    and the context/input must not be changed while the generator is suspended.
    Bounds/relation tags on outputs are not retained or inferred.
    """
    if type(normalization_factor) is not int or not 1 <= normalization_factor < 2**64:
        raise ValueError('positive public integer normalization factor required')
    first_output_scale = _positive_scale(first_output_scale, 'planned first output scale')
    coefficients = []
    for name, supplied in (('first weight', first_weight), ('first bias', first_bias)):
        coefficient = _public_reals(supplied, name=name)
        if coefficient.ndim not in (0, 1) or (coefficient.ndim == 1 and coefficient.shape != (arithmetic.ctx.num_slots,)):
            raise ValueError('first coefficients must be scalar or full physical slot vectors')
        coefficients.append(float(coefficient) if coefficient.ndim == 0 else coefficient.copy())
    context = _require_unbounded_keyless_owner(arithmetic)
    cache = context._param_cache
    if type(context.composite_degree) is not int or context.composite_degree != 2:
        raise ValueError('tracked reciprocal requires composite degree two')
    if getattr(arithmetic, 'refresher', None) is not None:
        raise ValueError('tracked reciprocal forbids implicit refresh')
    if type(getattr(denominator, '_ckks_nsd', None)) is not int or denominator._ckks_nsd != 1:
        raise ValueError('tracked reciprocal requires explicit integer NSD1')
    source_scale = _positive_scale(getattr(denominator, '_ckks_scale', None), 'denominator scale')
    level = arithmetic._level(denominator)
    q = _schedule(cache.q_moduli_at_level(level), level, updates)
    plan = _forward(q, input_scale=source_scale, input_level=level,
                    updates=updates, first_output_scale=first_output_scale)

    def static():
        _require_unbounded_keyless_owner(arithmetic)
        if (arithmetic.ctx is not context or context._param_cache is not cache
                or type(context.composite_degree) is not int or context.composite_degree != 2
                or getattr(arithmetic, 'refresher', None) is not None
                or tuple(cache.q_moduli_at_level(level)) != q
                or arithmetic._level(denominator) != level
                or denominator._ckks_scale != source_scale
                or type(getattr(denominator, '_ckks_nsd', None)) is not int or denominator._ckks_nsd != 1):
            raise ValueError('static context, denominator or refresh assumptions changed')

    def check(value, expected_level, expected_scale):
        static()
        if (arithmetic._level(value) != expected_level
                or value._ckks_scale != expected_scale
                or type(getattr(value, '_ckks_nsd', None)) is not int or value._ckks_nsd != 1):
            raise ValueError('actual primitive metadata differs from the public schedule')

    def check_row(y, e, row):
        check(y, row['level'], row['scale'])
        if e is not None:
            check(e, row['residual_level'], row['residual_scale'])

    static()
    y = affine_at_scale(arithmetic, denominator, *coefficients, output_scale=first_output_scale)
    e = None
    if updates > 1:
        product = arithmetic.multiply(denominator, y)
        e = integer_affine(arithmetic, product, -normalization_factor, bias=1.)
        del product
    check_row(y, e, plan['rows'][0])
    yield 1, y, e
    for update in range(2, updates + 1):
        static()
        check_row(y, e, plan['rows'][update-2])
        correction = integer_affine(arithmetic, e, 1, bias=1.)
        next_y = arithmetic.multiply(y, correction)
        next_e = arithmetic.multiply(e, e) if update < updates else None
        y, e = next_y, next_e
        del next_y, next_e, correction
        check_row(y, e, plan['rows'][update-1])
        yield update, y, e

"""Public binary64 scale metadata, in the exact compiled primitive order.

These helpers perform no encryption, encode no plaintext and never relabel a
ciphertext. They predict existing metadata; they do not bound CKKS error.
"""
import math


def positive_scale(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
        raise ValueError('finite positive binary64 scale required')
    value=float(value)
    if not math.isfinite(value) or value<=0:raise ValueError('unrepresentable scale')
    return value


def drop_divisor(q_moduli,level):
    q=tuple(q_moduli)
    if type(level)is not int or level<1 or len(q)<2*(level+1) or any(type(x)is not int or x<=1 for x in q):
        raise ValueError('ordered CD2 modulus prefix and positive source level required')
    return q[2*level]*q[2*level+1]


def affine_scale(input_scale,target_scale,divisor):
    """Actual PTCT then composite-rescale metadata for a target encoding."""
    source=positive_scale(input_scale);target=positive_scale(target_scale)
    if type(divisor)is not int or divisor<=1:raise ValueError('integer divisor required')
    before=positive_scale(target*divisor)
    encoded=positive_scale(before/source)
    product=positive_scale(float(source)*float(encoded))
    return dict(input_scale=source,target_scale=target,divisor=divisor,
        target_before_rescale=before,plaintext_scale=encoded,
        product_scale=product,output_scale=positive_scale(product/divisor))


def weighted_sum_scale(input_scales,canonical_scale,divisor):
    """Ordered first term owns scale after HEAdd, even for a zero weight.

    Public bias addition preserves that scale. Every term uses the same target
    but its independently rounded encoding/product can have a different scale.
    The ordinary facade's compatible-scale checks still apply at execution.
    """
    scales=tuple(input_scales)
    if not scales:raise ValueError('nonempty ordered input scales required')
    terms=[affine_scale(s,canonical_scale,divisor) for s in scales]
    return dict(terms=terms,output_scale=terms[0]['output_scale'],
        first_term_owns_scale=True)


def multiply_scale(left,right,divisor):
    left=positive_scale(left);right=positive_scale(right)
    if type(divisor)is not int or divisor<=1:raise ValueError('integer divisor required')
    return positive_scale(positive_scale(left*right)/divisor)


def static_newton_forward(q_moduli,*,input_scale,first_output_level,updates,first_target_scale):
    """Replay the existing two-level Newton schedule, preserving its target."""
    q=tuple(q_moduli);source=positive_scale(input_scale)
    if type(updates)is not int or updates<1:raise ValueError('positive update count required')
    initial=affine_scale(source,first_target_scale,drop_divisor(q,first_output_level+1))
    actual=initial['output_scale'];rows=[dict(update=1,level=first_output_level,scale=actual)]
    for index in range(2,updates+1):
        level=first_output_level-2*(index-2)
        product=multiply_scale(source,actual,drop_divisor(q,level))
        actual=multiply_scale(actual,product,drop_divisor(q,level-1))
        rows.append(dict(update=index,level=level-2,product_scale=product,scale=actual))
    return dict(initial_target_scale=first_target_scale,initial_affine=initial,rows=rows)

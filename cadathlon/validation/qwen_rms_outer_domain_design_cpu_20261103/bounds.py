"""Exact-real public envelope reference; no model loader or coefficient selection."""
from fractions import Fraction
from decimal import Decimal,localcontext,ROUND_CEILING
import math
import numbers

def rational(value):
    if isinstance(value,bool):raise TypeError('boolean is not a public real scalar')
    if isinstance(value,numbers.Integral):return Fraction(int(value))
    if type(value)is Fraction:return value
    if isinstance(value,numbers.Real) and math.isfinite(value) and hasattr(value,'as_integer_ratio'):
        numerator,denominator=value.as_integer_ratio();return Fraction(int(numerator),int(denominator))
    raise TypeError('finite public real scalar required')

def sqrt_upper(value):
    value=rational(value)
    if value<0:raise ValueError('negative squared bound')
    if not value:return 0.
    with localcontext() as ctx:
        ctx.prec=100
        approximate=(Decimal(value.numerator)/Decimal(value.denominator)).sqrt()
    result=float(approximate)
    if not math.isfinite(result):raise ValueError('outer bound is not representable as binary64')
    while rational(result)**2<value:result=math.nextafter(result,math.inf)
    return result

def rms_variance_upper(envelope,epsilon):
    values=tuple(rational(x) for x in envelope);eps=rational(epsilon)
    if not values or any(x<0 for x in values) or eps<=0:raise ValueError('nonnegative envelope and positive epsilon required')
    return sum((x*x for x in values),Fraction())/len(values)+eps

def normalized_linear_bounds(matrix,gamma,bias=None):
    gamma=tuple(rational(x) for x in gamma);D=len(gamma)
    if not D:raise ValueError('nonempty norm features required')
    rows=tuple(tuple(rational(x) for x in row) for row in matrix)
    if any(len(row)!=D for row in rows):raise ValueError('matrix shape differs from gamma')
    bias=tuple(Fraction() for _ in rows) if bias is None else tuple(rational(x) for x in bias)
    if len(bias)!=len(rows):raise ValueError('bias shape differs')
    return tuple(rational(sqrt_upper(D*sum(((w*g)**2 for w,g in zip(row,gamma)),Fraction())))+abs(b) for row,b in zip(rows,bias))

def head_variance_bounds(matrix,gamma,head_dim,epsilon):
    """Bias-free Q/K heads: D||W_head diag(gamma)||F²/head_dim+eps."""
    gamma=tuple(rational(x) for x in gamma);rows=tuple(tuple(rational(x) for x in row) for row in matrix)
    if type(head_dim)is not int or head_dim<1 or len(rows)%head_dim or not rows or any(len(row)!=len(gamma) for row in rows):raise ValueError('complete heads required')
    eps=rational(epsilon)
    if eps<=0:raise ValueError('positive epsilon required')
    row_square=[sum(((w*g)**2 for w,g in zip(row,gamma)),Fraction()) for row in rows]
    return tuple(Fraction(len(gamma),head_dim)*sum(row_square[i:i+head_dim],Fraction())+eps for i in range(0,len(rows),head_dim))

def linear_envelope(matrix,envelope,bias=None):
    envelope=tuple(rational(x) for x in envelope)
    if any(x<0 for x in envelope):raise ValueError('negative envelope')
    rows=tuple(tuple(rational(x) for x in row) for row in matrix)
    if any(len(row)!=len(envelope) for row in rows):raise ValueError('matrix/envelope shape mismatch')
    bias=tuple(Fraction() for _ in rows) if bias is None else tuple(rational(x) for x in bias)
    if len(bias)!=len(rows):raise ValueError('bias shape mismatch')
    return tuple(sum((abs(w)*u for w,u in zip(row,envelope)),Fraction())+abs(b) for row,b in zip(rows,bias))

def repeat_kv_bounds(values,head_dim,repeats):
    values=tuple(rational(x) for x in values)
    if type(head_dim)is not int or type(repeats)is not int or head_dim<1 or repeats<1 or len(values)%head_dim:raise ValueError('valid GQA head shape required')
    return tuple(x for i in range(0,len(values),head_dim) for _ in range(repeats) for x in values[i:i+head_dim])

def residual_envelope(left,right):
    left,right=tuple(map(rational,left)),tuple(map(rational,right))
    if len(left)!=len(right) or any(x<0 for x in left+right):raise ValueError('matching nonnegative envelopes required')
    return tuple(a+b for a,b in zip(left,right))

def ideal_newton_cost(condition,target='1e-6'):
    """Illustrative ideal seed y0=1/sqrt(b), not selected site coefficients.

    r=1-v*y² obeys r'=r²(3+r)/4. This directed Decimal upper recurrence
    excludes rounded seed/affine/update and CKKS error; it selects no polynomial.
    """
    condition=rational(condition)
    if condition<1:raise ValueError('condition ratio must be at least one')
    with localcontext() as ctx:
        ctx.prec=120;ctx.rounding=ROUND_CEILING
        k=Decimal(condition.numerator)/Decimal(condition.denominator)
        # Lower 1/k ensures r begins at an upper endpoint.
        inverse=(Decimal(1)/k).next_minus();r=Decimal(1)-inverse;limit=Decimal(target)
        if limit<=0:raise ValueError('positive target required')
        for updates in range(1,1000):
            r=r*r*(Decimal(3)+r)/Decimal(4)
            if r>=1:continue
            upper=r/(Decimal(2)-r)
            if upper<=limit:return dict(condition=str(condition),updates=updates,depth=4*updates-3,ideal_relative_error_upper=str(upper),coefficient_selected=False)
    raise ValueError('illustrative schedule unsupported')

"""Exact copied outward/check_certificate definitions; no historical plan path."""
from fractions import Fraction as F
from decimal import Decimal,localcontext,ROUND_CEILING,ROUND_FLOOR
import math

def outward(value,up):
    y=float(value)
    if (up and F(y)<value) or (not up and F(y)>value):y=math.nextafter(y,math.inf if up else -math.inf)
    if not math.isfinite(y) or y<=0:raise ValueError('positive finite expanded endpoints required')
    return y

def check_certificate(rule):
    a,b=map(F,rule['domain']);w=F(rule['rounded_first_weight']);bias=F(rule['rounded_first_bias'])
    assert min(w*a+bias,w*b+bias)>0
    points=[a,b]+[x for x in (-bias/w,-bias/(3*w)) if a<=x<=b]
    first=max(abs(1-x*(w*x+bias)**2) for x in points)
    given=rule['first_exact_residual_abs_bound'];assert first==F(int(given['numerator']),int(given['denominator'])) and first<1
    with localcontext() as c:
        c.prec=160;c.rounding=ROUND_CEILING
        R=Decimal(first.numerator)/Decimal(first.denominator)
        for _ in range(rule['iterations']-1):R=R*R*(3+R)/4
    with localcontext() as c:
        c.prec=160;c.rounding=ROUND_FLOOR
        lower=(1-R).sqrt().next_minus();denom=1+lower
    with localcontext() as c:
        c.prec=160;c.rounding=ROUND_CEILING;relative=R/denom
    assert R<=Decimal(rule['final_residual_bound']) and relative<=Decimal(rule['final_relative_error_bound'])<=Decimal('1e-6')
    assert rule['depth']==4*rule['iterations']-3
    return dict(first_residual_gap=outward(1-first,False),independent_relative_upper=str(relative),verified=True)

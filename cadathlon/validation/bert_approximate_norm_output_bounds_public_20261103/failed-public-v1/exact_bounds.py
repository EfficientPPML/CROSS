"""Exact public Chebyshev envelopes; no encoding, ciphertext or error oracle."""
from fractions import Fraction as F
import math
import numpy as np

def dyadic(e):return F(1<<e) if e>=0 else F(1,1<<(-e))
def trim(c):
    c=list(map(F,c))
    while len(c)>1 and c[-1]==0:c.pop()
    return c

def cheb_add(a,b):
    out=[F()]*max(len(a),len(b))
    for i,v in enumerate(a):out[i]+=v
    for i,v in enumerate(b):out[i]+=v
    return trim(out)

def cheb_scale(c,s):return trim([v*s for v in c])
def cheb_multiply(a,b):
    out=[F()]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            term=x*y/2
            out[i+j]+=term;out[abs(i-j)]+=term
    return trim(out)

def power_to_cheb(coefficients):
    # Build each exact t^k in the Chebyshev basis. Multiplication by T1
    #uses T_i*T_1=(T_(i+1)+T_|i-1|)/2, including T0 correctly.
    out=[F()];power=[F(1)]
    for coefficient in coefficients:
        out=cheb_add(out,cheb_scale(power,F(coefficient)))
        power=cheb_multiply(power,[F(),F(1)])
    return trim(out)

def eval_cheb(c,x):
    a,b=F(1),F(x);out=c[0]
    if len(c)>1:out+=c[1]*b
    for coefficient in c[2:]:a,b=b,2*x*b-a;out+=coefficient*b
    return out

def cheb_abs_envelope(c,M):
    M=F(M)
    if M<1:raise ValueError('extension radius must be at least1')
    return eval_cheb([abs(v) for v in c],M)

def ceil_float(v):
    v=F(v)
    try:out=float(v)
    except OverflowError:raise ValueError('bound exceeds finite float range') from None
    if not math.isfinite(out):raise ValueError('bound exceeds finite float range')
    if F(out)<v:out=math.nextafter(out,math.inf)
    if not math.isfinite(out):raise ValueError('bound exceeds finite float range')
    return out

def sqrt_upper(v):
    v=F(v)
    if v<0:raise ValueError('nonnegative radicand required')
    if v==0:return F()
    exponent=v.numerator.bit_length()-v.denominator.bit_length()
    if v<dyadic(exponent):exponent-=1
    shift=exponent//2;scaled=v/dyadic(2*shift)
    assert 1<=scaled<4
    try:out=math.ldexp(math.sqrt(float(scaled)),shift)
    except OverflowError:raise ValueError('sqrt bound exceeds finite float range') from None
    if not math.isfinite(out):raise ValueError('sqrt bound exceeds finite float range')
    while F(out)**2<v:
        out=math.nextafter(out,math.inf)
        if not math.isfinite(out):raise ValueError('sqrt bound exceeds finite float range')
    return F(out)

def polynomial_contract(rule):
    p=rule['polynomial'];lo,hi=map(float,p['domain']);c=p['coefficients']
    if not 0<lo<hi or rule['method'] not in ('power','balanced_chebyshev'):raise ValueError('fixed positive polynomial contract required')
    A=F(2/(hi-lo));B=F(-(hi+lo)/(hi-lo))
    if rule['method']=='power':
        evaluated=np.polynomial.chebyshev.cheb2poly(np.asarray(c,dtype=np.float64))
        P=power_to_cheb([F(float(v)) for v in evaluated])
    else:
        evaluated=np.asarray(c,dtype=np.float64);P=list(map(F,map(float,c)))
    tlo=A*F(lo)+B;thi=A*F(hi)+B;M=max(F(1),abs(tlo),abs(thi))
    # Actual v=(t-B)/A, including public affine coefficient rounding.
    H=cheb_add(cheb_multiply([-B/A,1/A],cheb_multiply(P,P)),[F(-1)])
    error=cheb_abs_envelope(H,M)
    assert error>=0
    return dict(A=A,B=B,tlo=tlo,thi=thi,M=M,P=P,H=H,squared_relative_envelope=error,
        normalization_factor=sqrt_upper(1+error),evaluated_coefficients=evaluated)

def affine_bound(contract,gamma,beta):
    gamma=np.asarray(gamma);beta=np.asarray(beta)
    if gamma.ndim!=1 or gamma.shape!=beta.shape or len(gamma)!=768 or not np.isfinite(gamma).all() or not np.isfinite(beta).all():raise ValueError('finite width768 public affine required')
    alpha=F(1./768)
    # v=alpha*sum(u_i²)+epsilon and positive epsilon imply
    # |u_i*P(v)|² <= (1+error)/alpha. No zero-mean identity is assumed.
    factor=sqrt_upper((1+contract['squared_relative_envelope'])/alpha)
    terms=[factor*abs(F(float(g)))+abs(F(float(b))) for g,b in zip(gamma,beta)]
    bound=max(terms);guarded=F(9,8)*bound;power=1
    while power<guarded:power*=2
    return dict(alpha=alpha,coordinate_factor=factor,output_bound=bound,output_bound_f64=ceil_float(bound),
        maximum_coordinate=terms.index(bound),guarded_bound=guarded,refresh_bound=power)

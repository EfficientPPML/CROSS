"""Public NumPy polynomial coefficients and explicit arithmetic; no HE imports."""
from decimal import Decimal,localcontext,ROUND_CEILING,ROUND_FLOOR
from fractions import Fraction as F
import math
import numpy as np

def coefficients(kind,domain,degree):
    funcs={'inverse_sqrt':lambda x:1/np.sqrt(x),'silu':lambda x:x*np.exp(-np.logaddexp(0,-x)),'tanh':np.tanh,'exp':np.exp}
    c=np.polynomial.Chebyshev.interpolate(funcs[kind],degree,domain=domain).coef
    threshold=float(np.max(np.abs(c)))*32*np.finfo(float).eps
    return np.where(np.abs(c)<threshold,0.,c)

def basis_coefficients(c,strategy):
    return np.polynomial.chebyshev.cheb2poly(c) if strategy=='power' else np.asarray(c)

def polynomial(values,rule,input_scale=1.):
    lo,hi=rule['domain'];t=np.asarray(values)*(2/(hi-lo)*input_scale)-(hi+lo)/(hi-lo)
    c=np.asarray(rule['evaluation_coefficients']);strategy=rule['strategy'];terms={1:t}
    def term(n):
        if n not in terms:
            product=term(n//2)*term((n+1)//2)
            terms[n]=product if strategy=='power' else (2*product-1 if n%2==0 else 2*product-t)
        return terms[n]
    indices=[n for n in range(1,len(c)) if strategy!='power' or c[n]!=0]
    if not indices:return np.full_like(t,c[0])
    result=term(indices[0])*c[indices[0]]
    for n in indices[1:]:result=result+term(n)*c[n]
    return result+c[0]

def polynomial_depth(rule):
    c=rule['evaluation_coefficients'];strategy=rule['strategy'];memo={0:0,1:0}
    def depth(n):
        if n not in memo:memo[n]=(1 if strategy=='power' else 2)+max(depth(n//2),depth((n+1)//2))
        return memo[n]
    return 2+max(depth(n) for n in range(len(c)) if strategy!='power' or c[n]!=0)

def sqrt_lower(value):
    with localcontext() as c:
        c.prec=100;c.rounding=ROUND_FLOOR
        return value.sqrt().next_minus()

def rsqrt_contract(domain,target=1e-6):
    a,b=map(float,domain);initial=1/math.sqrt(b);w=-.5*initial**3;bias=1.5*initial
    af,bf,wf,biasf=map(F.from_float,(a,b,w,bias))
    if min(wf*af+biasf,wf*bf+biasf)<=0:raise ValueError('Nonpositive rounded initial inverse-sqrt affine')
    points=[af,bf]
    for x in (-biasf/wf,-biasf/(3*wf)):
        if af<=x<=bf:points.append(x)
    residual=lambda x:1-x*(wf*x+biasf)**2
    values=[residual(x) for x in points];worst=max(map(abs,values))
    if worst>=1:raise ValueError('Rounded initial inverse-sqrt residual not contractive')
    with localcontext() as c:c.prec=100;c.rounding=ROUND_CEILING;R=Decimal(worst.numerator)/Decimal(worst.denominator)
    def relative_upper(R):
        with localcontext() as c:c.prec=100;c.rounding=ROUND_FLOOR;denominator=1+sqrt_lower(1-R)
        with localcontext() as c:c.prec=100;c.rounding=ROUND_CEILING;return R/denominator
    n=1
    while relative_upper(R)>Decimal(str(target)):
        with localcontext() as c:c.prec=100;c.rounding=ROUND_CEILING;R=R*R*(3+R)/4
        n+=1
        if n>64:raise ValueError('Unsupported inverse-sqrt iteration count')
    return {'domain':[a,b],'initial_estimate':initial,'rounded_first_weight':w,'rounded_first_bias':bias,'iterations':n,'depth':4*n-3,
        'first_exact_residual_abs_bound':{'numerator':str(worst.numerator),'denominator':str(worst.denominator)},'final_residual_bound':str(R),
        'final_relative_error_bound':str(relative_upper(R)),'certificate_scope':'Actual rounded initial affine, followed by exact-real Newton updates. No later binary64 or CKKS error bound.'}

def reciprocal_contract(domain,target=1e-12):
    a,b=map(float,domain);initial=2/(a+b);w=(-initial)*initial;bias=2*initial
    af,bf,wf,biasf=map(F.from_float,(a,b,w,bias));points=[af,bf];vertex=-biasf/(2*wf)
    if af<=vertex<=bf:points.append(vertex)
    worst=max(abs(1-x*(wf*x+biasf)) for x in points)
    if worst>=1:raise ValueError('Rounded initial reciprocal not contractive')
    with localcontext() as c:c.prec=100;c.rounding=ROUND_CEILING;R=Decimal(worst.numerator)/Decimal(worst.denominator)
    n=1
    while R>Decimal(str(target)):
        with localcontext() as c:c.prec=100;c.rounding=ROUND_CEILING;R=R*R
        n+=1
    return {'domain':[a,b],'initial_estimate':initial,'rounded_first_weight':w,'rounded_first_bias':bias,'iterations':n,'depth':3*n-2,'final_residual_bound':str(R),
        'certificate_scope':'Actual rounded initial affine followed by exact-real Newton updates; later operation rounding excluded.'}

def reciprocal(values,rule):
    y=np.asarray(values)*rule['rounded_first_weight']+rule['rounded_first_bias']
    for _ in range(1,rule['iterations']):y=y*(2-values*y)
    return y

def rsqrt(values,rule):
    y=np.asarray(values)*rule['rounded_first_weight']+rule['rounded_first_bias']
    for _ in range(1,rule['iterations']):y=y*(1.5-.5*(values*(y*y)))
    return y

def rational_silu(values,rule):
    t=polynomial(values,rule['base'],2.**-(rule['doublings']+1))
    for _ in range(rule['doublings']):t=2*(t*reciprocal(1+t*t,rule['reciprocal']))
    return values*(.5*t+.5)

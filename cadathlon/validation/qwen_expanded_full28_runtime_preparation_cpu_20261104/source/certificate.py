"""Exact Chebyshev residual envelope including rounded mapper endpoint extension."""
from fractions import Fraction as F
import math,hashlib,struct

def upper(x):
    try:y=float(x)
    except OverflowError:return None
    if not math.isfinite(y):return None
    return math.nextafter(y,math.inf) if F(y)<x else y

def packed(x):return dict(numerator_hex=hex(x.numerator),denominator_hex=hex(x.denominator))
def unpack(x):return F(int(x['numerator_hex'],16),int(x['denominator_hex'],16))

def square(c):
    D=max(x.denominator for x in c);assert D&(D-1)==0
    a=[x.numerator*(D//x.denominator) for x in c];out=[0]*(2*len(c)-1)
    for i,x in enumerate(a):
        for j in range(i,len(a)):
            value=x*a[j]*(1 if i==j else 2);out[i+j]+=value;out[j-i]+=value
    return [F(x,2*D*D) for x in out]

def residual(c,A,B):
    sq=square(c);alpha=1/A;beta=-B/A;h=[F(0)]*(len(sq)+1)
    for n,value in enumerate(sq):
        h[n]+=beta*value
        if n==0:h[1]+=alpha*value
        else:h[n-1]+=alpha*value/2;h[n+1]+=alpha*value/2
    h[0]-=1
    while len(h)>1 and h[-1]==0:h.pop()
    return h

def T(n,x):
    if n==0:return F(1)
    # Keep a common denominator D**n; avoid repeated huge Fraction gcds.
    m,D=x.numerator,x.denominator;a,b=1,m;D2=D*D
    for _ in range(1,n):a,b=b,2*m*b-D2*a
    return F(b,D**n)

def certify(rule,target=F(1,10**6),details=True):
    lo,hi=rule['domain'];a,b=F(lo),F(hi);assert 0<a<b and rule['strategy']=='chebyshev'
    A=F(2/(hi-lo));B=F(-(hi+lo)/(hi-lo));left,right=A*a+B,A*b+B
    assert A>0 and left<=right
    c=list(map(F,rule['evaluation_coefficients']));h=residual(c,A,B)
    M=max(F(1),abs(left),abs(right));envelope=T(len(h)-1,M);l1=sum(map(abs,h),F(0));delta=l1*envelope
    positive=sum((v*((-1)**(i//2)) for i,v in enumerate(c) if i%2==0),F(0))>0
    relative=delta/(2-delta) if delta<1 else None
    passed=bool(left<=0<=right and positive and relative is not None and relative<=target)
    result=dict(certified=passed,positive_at_mapped_zero=positive,zero_inside_mapped_domain=left<=0<=right,
      H_degree=len(h)-1,delta_upper_binary64=upper(delta),relative_upper_binary64=upper(relative) if relative is not None else None,
      coefficient_sha256=hashlib.sha256(struct.pack('<'+str(len(c))+'d',*rule['evaluation_coefficients'])).hexdigest(),
      actual_A_hex=float(A).hex(),actual_B_hex=float(B).hex(),endpoint_extension=M>1,
      scope='Exact stored coefficient/map polynomial only; no subsequent binary64 or CKKS/refresh error bound.')
    if details:result.update(delta=packed(delta),relative=packed(relative) if relative is not None else None,extent=packed(M),H_L1=packed(l1),chebyshev_extent_envelope=packed(envelope))
    return result

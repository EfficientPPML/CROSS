"""Public binary64 metadata equations, not payload/noise/refresh guarantees."""
from fractions import Fraction
import math,sys
from component_contract import numerical_plan

def positive(value):
    if type(value)is not float or not math.isfinite(value)or value<=0:raise ValueError('finite positive binary64 scale required')
    return value

def divisor(level):
    if type(level)is not int or not 1<=level<=49:raise ValueError('actual source level1..49 required')
    return math.prod(numerical_plan()['profile']['q_towers'][2*level:2*(level+1)])

def weighted_scale(scale,level):
    scale=positive(scale);D=divisor(level);delta=numerical_plan()['profile']['scale']
    encoded=positive((delta*D)/scale)
    return positive(positive(scale*encoded)/D)

def product_scale(left,right,level):return positive(positive(positive(left)*positive(right))/divisor(level))

def refresh_scale_certificate():
    if(sys.float_info.radix,sys.float_info.mant_dig,sys.float_info.rounds)!=(2,53,1):raise ValueError('binary64 nearest rounding required')
    D=divisor(26);delta=numerical_plan()['profile']['scale'];target=delta*D
    T=Fraction(target);u=Fraction(1,2**53);lo=T*(1-u)**2;hi=T*(1+u)**2
    minimum=Fraction(sys.float_info.min);maximum=Fraction(sys.float_info.max)
    if not(T/maximum>minimum and T*(1-u)>minimum and T*(1+u)<maximum
           and lo/Fraction(float(D))>minimum and hi/Fraction(float(D))<maximum):raise ValueError('normal finite intermediate proof failed')
    first=float(lo)
    while Fraction(first)<lo:first=math.nextafter(first,math.inf)
    products=[];value=first
    while Fraction(value)<=hi:
        products.append(value)
        if len(products)>16:raise ValueError('unexpectedly broad enclosure')
        value=math.nextafter(value,math.inf)
    return dict(input_level=26,output_level=25,divisor=D,target=target,
        product_candidates=products,accepted_scales=sorted(set(x/D for x in products)),
        normal_intermediates_proved=True,scope='Enclosure only; no reachability, CKKS or refresh error claim.')

def check_refresh_scale(scale):
    positive(scale)
    if scale not in refresh_scale_certificate()['accepted_scales']:raise ValueError('refresh final weighted metadata outside production enclosure')
    return scale

def client_replay(refresh_scales):
    """Client-only full graph replay after all server artifacts are terminal.

    This imports NumPy clear planning doubles. Server instrumentation must never
    call this function or import public_graph; its primitive checks are local.
    """
    if type(refresh_scales)is not tuple or len(refresh_scales)!=5:raise ValueError('five ordered actual refresh scales required')
    for value in refresh_scales:check_refresh_scale(value)
    from public_plan.public_graph import run
    result=run(refresh_scale=refresh_scales);plan=numerical_plan()
    if len(result['trace'])!=96:raise ValueError('complete96 metadata replay required')
    for actual,expected in zip(result['trace'],plan['checkpoints']):
        for key in('index','kind','label','stack','level','nsd','expected_array_sha256'):
            if actual[key]!=expected[key]:raise ValueError('fixed clear graph/checkpoint drift: '+key)
    return result

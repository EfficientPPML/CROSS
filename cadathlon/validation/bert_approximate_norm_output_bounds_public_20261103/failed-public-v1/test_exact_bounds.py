from fractions import Fraction as F
import json,math
from pathlib import Path
import numpy as np
import pytest
from exact_bounds import *

@pytest.mark.parametrize('degree',[0,1,2,3,4,8,16])
def test_power_conversion_independent_horner(degree):
    c=[F((-1)**i*(i+1),2**(i+2)) for i in range(degree+1)];cheb=power_to_cheb(c)
    for x in (F(-9,8),F(-1),F(-2,7),F(),F(1,3),F(1),F(9,8)):
        expected=F()
        for v in reversed(c):expected=expected*x+v
        assert eval_cheb(cheb,x)==expected

@pytest.mark.parametrize('x',[F(-7,4),F(-1),F(-1,8),F(),F(1,3),F(1),F(5,4)])
def test_product_identity_independent_values(x):
    a=[F(1,3),F(-4,7),F(5,9)];b=[F(7,11),F(-5,9),F(),F(3,13)]
    assert eval_cheb(cheb_multiply(a,b),x)==eval_cheb(a,x)*eval_cheb(b,x)

@pytest.mark.parametrize('x',[F(),F(1),F(2),F(3),F(2)**-1500,F(2)**-2150,F(2)**1000])
def test_sqrt_outward_including_underflow(x):
    bound=sqrt_upper(x);assert bound>=0 and bound*bound>=x
    if bound:assert F(math.nextafter(float(bound),0))**2<x

@pytest.mark.parametrize('x',[F(-1),F(2)**3000])
def test_sqrt_invalid_or_unrepresentable(x):
    with pytest.raises(ValueError):sqrt_upper(x)

def test_extension_envelope_does_not_assume_exact_unit_domain():
    M=F(math.nextafter(1.,math.inf));c=[F(1,7),F(-1,3)]+[F()]*126+[F(1)]
    bound=cheb_abs_envelope(c,M)
    assert bound>sum(abs(v) for v in c)
    for x in (-M,F(-1),F(0),F(1),M):assert abs(eval_cheb(c,x))<=bound

@pytest.mark.parametrize('site',['0.attention','0.mlp','3.mlp','8.mlp','9.mlp','11.attention'])
def test_actual_rounded_polynomial_identity(site):
    rule=json.loads((Path(__file__).parent/'candidate_manifest.json').read_text())['norms'][site];p=polynomial_contract(rule)
    for v in map(F,rule['polynomial']['domain']):
        t=p['A']*v+p['B'];expected=v*eval_cheb(p['P'],t)**2-1
        assert eval_cheb(p['H'],t)==expected
        assert abs(expected)<=p['squared_relative_envelope']
    assert p['normalization_factor']**2>=1+p['squared_relative_envelope']

def test_affine_bound_uses_actual_positive_variance_coefficient():
    gamma=np.zeros(768);beta=np.zeros(768);gamma[12]=-2.;beta[12]=3.
    contract={'squared_relative_envelope':F(1,1000)};r=affine_bound(contract,gamma,beta)
    assert r['alpha']==F(1./768) and r['alpha']!=F(1,768)
    assert r['coordinate_factor']**2*r['alpha']>=F(1001,1000)
    assert r['output_bound']==2*r['coordinate_factor']+3
    assert r['refresh_bound']>=r['output_bound']*F(9,8)

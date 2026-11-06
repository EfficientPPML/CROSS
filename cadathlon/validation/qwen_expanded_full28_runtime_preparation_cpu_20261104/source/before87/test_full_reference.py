"""Tiny independent-reference tests; no actual full trajectory/assets read."""
from pathlib import Path
import importlib.util,sys
import numpy as np
import pytest
import full_reference as ref
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_test_arithmetic',P/'inputs/frozen_arithmetic.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)

def poly(strategy):
    c=[.8,.07,-.006,.0002]
    return dict(domain=[-2.,3.],strategy=strategy,evaluation_coefficients=c,chebyshev_coefficients=c)
@pytest.mark.parametrize('strategy',['power','chebyshev'])
def test_independent_polynomial_against_frozen_tiny(strategy):
    x=np.array([-2.,-.3,0.,2.4,3.]);r=poly(strategy);engine=old.Arithmetic();a=engine.polynomial(old.Value(x),old.FrozenPolynomial(r),strategy,'p').data
    b=ref.polynomial(x,r,'p',{})
    np.testing.assert_allclose(a,b,rtol=0,atol=3e-16)

def test_actual_power_coefficients_used():
    r=poly('power');r['chebyshev_coefficients']=[900.,800.]
    x=np.array([-.7,.4]);np.testing.assert_array_equal(ref.polynomial(x,r,'p',{}),np.polynomial.polynomial.polyval(x*.4-.2,r['evaluation_coefficients']))

@pytest.mark.parametrize('iterations',[1,2,6,11])
def test_original_newton_recurrence(iterations):
    x=np.array([.5,.9,1.2,1.9]);r=dict(domain=[.5,2.],rounded_first_weight=-.64,rounded_first_bias=1.6,iterations=iterations)
    engine=old.Arithmetic();expected=engine.inverse(old.Value(x),r,'r').data
    np.testing.assert_array_equal(ref.reciprocal(x,r,'r',{}),expected)

@pytest.mark.parametrize('method',['polynomial','newton'])
def test_rms_same_source_and_sum_alpha(method):
    x=np.array([[.4,-.2,.3],[.1,.3,-.2]]);g=np.array([2.,.5,1.2]);r=dict(method=method,domain=[.01,2.],strategy='chebyshev',evaluation_coefficients=[1.,-.01,.003],chebyshev_coefficients=[1.,-.01,.003],rounded_first_weight=-.05,rounded_first_bias=1.2,iterations=5)
    engine=old.QwenArithmetic({});expected=engine.norm(old.Value(x),g,1e-6,r,'n').data
    np.testing.assert_allclose(ref.norm(x,g,1e-6,r,'n',{}),expected,rtol=0,atol=3e-16)

def test_original_positions_rope_independent_scalar():
    x=np.arange(2*2*128,dtype=float).reshape(2,2,128)/100.;positions=(3,19)
    actual=ref.rope(x,positions,1e6);expected=x.copy()
    import math
    frequencies=1e6**(-np.arange(0,128,2)/128)
    for row,pos in enumerate(positions):
        for h in range(2):
            for d in range(64):
                angle=pos*frequencies[d];c=math.cos(angle);s=math.sin(angle)
                expected[row,h,d]=x[row,h,d]*c-x[row,h,d+64]*s
                expected[row,h,d+64]=x[row,h,d+64]*c+x[row,h,d]*s
    np.testing.assert_allclose(actual,expected,rtol=0,atol=4e-15)

def test_causal_k1_direct_and_same_alpha():
    exp=dict(domain=[-.5,.5],strategy='power',evaluation_coefficients=[1.,.5,.1],chebyshev_coefficients=[1.,.5,.1])
    reciprocal=dict(domain=[.1,2.],rounded_first_weight=-.64,rounded_first_bias=1.6,iterations=8)
    manifest=dict(exp=exp,initial_reciprocal=reciprocal,reciprocal_by_keys={'2':reciprocal,'3':reciprocal})
    q=np.arange(3*2*4).reshape(3,2,4)*.001;k=q*.8;v=q*2.
    rule=dict(centered_domain=[-1.,1.],squarings=2)
    engine=old.QwenArithmetic(manifest)
    expected=engine.causal_attention(*(old.Value(t.transpose(1,0,2)[None]) for t in (q,k,v)),rule,'a').data[0].transpose(1,0,2)
    actual=ref.causal(q,k,v,rule,manifest,'a',{})
    np.testing.assert_array_equal(actual[0],v[0]);np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-17)

def test_rational_silu_original_x():
    r=dict(method='rational_tanh',domain=[-2.,2.],doublings=3,base=poly('power'),reciprocal=dict(domain=[1.,2.1],rounded_first_weight=-.4,rounded_first_bias=1.3,iterations=5))
    x=np.array([-.8,0.,.8]);engine=old.QwenArithmetic({});expected=engine.silu(old.Value(x),r,'s').data
    np.testing.assert_allclose(ref.silu(x,r,'s',{}),expected,rtol=0,atol=3e-16)

@pytest.mark.parametrize('value',[np.array([float('nan')]),np.array([float('inf')]),np.array([5.])])
def test_domain_rejects_without_clip(value):
    records={}
    with pytest.raises(ref.DomainFailure):ref.polynomial(value,poly('power'),'p',records)
    assert records['p']['above']+records['p']['nonfinite']>0

def test_nonfinite_internal_stage_rejected():
    rule=dict(domain=[1.,2.],rounded_first_weight=1e308,rounded_first_bias=1e308,iterations=3)
    with np.errstate(over='ignore'):
        with pytest.raises(FloatingPointError):ref.reciprocal(np.array([1.5]),rule,'r',{})

def test_forward_rejects_identity_before_weights():
    called=[]
    with pytest.raises(ValueError,match='frozen'):
        ref.forward(np.zeros((1,1024)),(0,),manifest_bytes=b'{}',config_bytes=b'{}',matrix=lambda n:called.append(n),gamma=lambda n:called.append(n))
    assert not called

from fractions import Fraction
import math
import numpy as np
import pytest
from bounds import *

def test_exact_dyadic_and_outward_sqrt_boundaries():
    for q in [Fraction(),Fraction(1,2**300),Fraction(2),Fraction(2**100+1),Fraction.from_float(math.nextafter(1.,0.))]:
        out=sqrt_upper(q);assert Fraction.from_float(out)**2>=q
    with pytest.raises(TypeError):rational(True)
    with pytest.raises(TypeError):rational(1+0j)
    with pytest.raises(TypeError):rational(np.bool_(True))
    assert rational(np.float32(.1))==Fraction.from_float(float(np.float32(.1)))
    assert rational(np.float64(.1))==Fraction.from_float(.1)


def test_head_bound_has_final_epsilon_and_contains_zero_and_extreme_inputs():
    W=np.array([[1.,-2.,.5],[2.,1.,-1.],[.25,2.,1.],[-3.,.5,.75]]);g=np.array([2.,.5,-1.]);eps=1e-6
    limits=head_variance_bounds(W,g,2,eps)
    for x in [np.zeros(3),np.array([1e6,-2e6,3e6]),np.array([1e-20,2e-20,0.])]:
        y=x/np.sqrt(np.mean(x*x)+eps)*g;heads=(W@y).reshape(2,2);v=np.mean(heads*heads,axis=1)+eps
        assert all(eps<=actual<=float(bound) for actual,bound in zip(v,limits))
    assert limits[0]==Fraction(3,2)*sum(((Fraction.from_float(float(w))*Fraction.from_float(float(gg)))**2 for row in W[:2] for w,gg in zip(row,g)),Fraction())+Fraction.from_float(eps)


def test_linear_bias_and_gqa_repeat_order_are_explicit():
    assert repeat_kv_bounds([1,2,3,4],2,2)==(1,2,1,2,3,4,3,4)
    assert linear_envelope([[1,-2]], [3,4],bias=[-5])==(Fraction(16),)
    assert normalized_linear_bounds([[1,0]],[1,1],bias=[-3])[0]>=3+math.sqrt(2)


def test_attention_convex_mix_and_mlp_envelopes_cover_independent_small_circuit():
    rng=np.random.default_rng(19);D=4;S=7;hd=2;nkv=2;heads=4;eps=1e-6
    embedding=rng.normal(size=(S,D));gamma=rng.normal(size=D);V=rng.normal(size=(nkv*hd,D));O=rng.normal(size=(D,heads*hd))
    z=embedding/np.sqrt(np.mean(embedding**2,axis=-1,keepdims=True)+eps)*gamma
    values=(z@V.T).reshape(S,nkv,hd);values=np.repeat(values,2,axis=1)
    probability=rng.uniform(size=(S,heads,S));probability/=probability.sum(axis=-1,keepdims=True)
    context=np.einsum('qhk,khd->qhd',probability,values).reshape(S,heads*hd)
    attention=context@O.T
    v_bound=normalized_linear_bounds(V,gamma);attention_bound=linear_envelope(O,repeat_kv_bounds(v_bound,hd,2))
    assert np.all(np.abs(attention)<=np.asarray(attention_bound,float))
    residual=np.max(np.abs(embedding),axis=0);post=residual_envelope(residual,attention_bound);after=embedding+attention
    assert np.all(np.mean(after**2,axis=-1)+eps<=float(rms_variance_upper(post,eps)))
    gp=rng.normal(size=D);gate=rng.normal(size=(6,D));up=rng.normal(size=(6,D));down=rng.normal(size=(D,6))
    z=after/np.sqrt(np.mean(after**2,axis=-1,keepdims=True)+eps)*gp
    g=z@gate.T;u=z@up.T;mlp=(g/(1+np.exp(-g))*u)@down.T
    gb=normalized_linear_bounds(gate,gp);ub=normalized_linear_bounds(up,gp)
    mb=linear_envelope(down,[a*b for a,b in zip(gb,ub)])
    assert np.all(np.abs(mlp)<=np.asarray(mb,float))
    final=residual_envelope(post,mb)
    assert np.all(np.mean((after+mlp)**2,axis=-1)+eps<=float(rms_variance_upper(final,eps)))

@pytest.mark.parametrize('condition',[1,10**2,10**4,10**6,10**12,10**18,10**24])
def test_illustrative_ideal_newton_cost_is_scoped_and_not_a_coefficient_selection(condition):
    result=ideal_newton_cost(condition)
    assert Decimal(result['ideal_relative_error_upper'])<=Decimal('1e-6')
    assert result['depth']==4*result['updates']-3 and result['coefficient_selected'] is False

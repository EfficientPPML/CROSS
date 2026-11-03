from types import SimpleNamespace
import numpy as np
import pytest
import integer_affine as affine
import integer_newton as newton
from test_contract import arithmetic,Value


def options(**kw):
    return dict(normalization_factor=32,initial_estimate=.1,updates=3,first_output_scale=123.)|kw


@pytest.mark.parametrize('value',[True,np.bool_(True),[True]+[.1]*7,1+.0j,np.ones(8,dtype=complex),np.array([.1]*8,dtype=object),1e-300])
def test_estimate_rejected_before_any_operation(arithmetic,value):
    with pytest.raises(ValueError):list(newton.integer_newton_updates(arithmetic,Value(np.ones(8)),**options(initial_estimate=value)))
    assert arithmetic.calls==[]


@pytest.mark.parametrize('change',['refresh','level','scale','nsd','q','context','yielded_scale','yielded_level'])
def test_static_schedule_rejects_generator_resume_drift(arithmetic,change):
    denominator=Value(np.ones(8));g=newton.integer_newton_updates(arithmetic,denominator,**options())
    _,first=next(g);calls=list(arithmetic.calls)
    if change=='refresh':arithmetic.refresher=object()
    elif change=='level':denominator.level-=1
    elif change=='scale':denominator._ckks_scale*=2
    elif change=='nsd':denominator._ckks_nsd=True
    elif change=='q':arithmetic.q[-1]+=2
    elif change=='context':arithmetic.ctx=SimpleNamespace(**vars(arithmetic.ctx))
    elif change=='yielded_scale':first._ckks_scale*=2
    elif change=='yielded_level':first.level-=1
    with pytest.raises(ValueError):next(g)
    assert arithmetic.calls==calls


@pytest.mark.parametrize('change',['short','refresh','cd1','nsd2','nsdbool','overflow'])
def test_fixed_plan_rejected_before_operations(arithmetic,change):
    value=Value(np.ones(8));kw=options()
    if change=='short':value.level=4
    elif change=='refresh':arithmetic.refresher=object()
    elif change=='cd1':arithmetic.ctx.composite_degree=1
    elif change=='nsd2':value._ckks_nsd=2
    elif change=='nsdbool':value._ckks_nsd=True
    elif change=='overflow':kw['first_output_scale']=1e308
    with pytest.raises(ValueError):list(newton.integer_newton_updates(arithmetic,value,**kw))
    assert arithmetic.calls==[]


def test_actual_affine_scale_mismatch_not_relabelled(arithmetic,monkeypatch):
    original=newton.affine_at_scale
    def wrong(*a,**kw):
        value=original(*a,**kw);value._ckks_scale*=2;return value
    monkeypatch.setattr(newton,'affine_at_scale',wrong)
    with pytest.raises(ValueError,match='actual primitive metadata'):
        next(newton.integer_newton_updates(arithmetic,Value(np.ones(8)),**options()))


@pytest.mark.parametrize('value',[True,1+0j,np.bool_(False),float('inf')])
def test_planner_scale_type_rejected(value):
    with pytest.raises(ValueError):affine.backward_scale_plan(range(101,109),input_scale=value,first_output_level=2,updates=2,final_scale=123.)


def test_subresolution_and_zero_public_encoding_fail_closed():
    calls=[]
    def encode(slots,level,scale):calls.append((slots,level,scale));return SimpleNamespace(polynomial=np.zeros(4,dtype=np.uint32))
    a=SimpleNamespace(_encode_public=encode)
    with pytest.raises(ValueError,match='round away'):affine._encode_checked(a,np.array([0.,1e-100]),1,100.)
    assert not calls
    with pytest.raises(ValueError,match='zero plaintext'):affine._encode_checked(a,np.array([0.,1.]),1,100.)
    assert len(calls)==1
    # An intentionally zero coefficient is allowed.
    assert np.all(affine._encode_checked(a,np.zeros(2),1,100.).polynomial==0)


@pytest.mark.parametrize('value',[True,np.array([1+0j]),[.1,True]])
def test_public_affine_coefficients_reject_coercive_types(value):
    with pytest.raises(ValueError):affine._public_reals(value,name='coefficient')


def test_full_static_scale_underflow_rejected_before_operations(arithmetic):
    with pytest.raises(ValueError,match='planned update scale'):
        list(newton.integer_newton_updates(arithmetic,Value(np.ones(8),scale=123.),**options(updates=13)))
    assert arithmetic.calls==[]

"""Public/static API regression; actual encrypted two-update proof is separate."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from types import SimpleNamespace
import numpy as np
import pytest
import encrypted_scaled_newton as module
from encrypted_integer_affine import backward_scale_plan


class Value:
    def __init__(self,data,level=26,scale=23562.,nsd=1):
        self.data=np.asarray(data);self.level=level;self._ckks_scale=scale;self._ckks_nsd=nsd


class Arithmetic:
    def __init__(self,slots):
        self.q=list(range(101,155));self.calls=[];self.refresher=None
        self.ctx=SimpleNamespace(num_slots=slots,composite_degree=2,secret_key=None,parameters={},
            _param_cache=SimpleNamespace(secret_key=None,q_moduli_at_level=lambda level:tuple(self.q[:2*(level+1)])))
    def _level(self,value):return value.level
    def multiply(self,left,right):
        self.calls.append('multiply');level=min(left.level,right.level)
        divisor=self.q[2*level]*self.q[2*level+1]
        return Value(left.data*right.data,level-1,left._ckks_scale*right._ckks_scale/divisor)


@pytest.fixture
def arithmetic(monkeypatch):
    a=Arithmetic(8)
    def affine(arithmetic,value,weight,bias,*,output_scale):
        assert arithmetic is a;arithmetic.calls.append(('first_affine',output_scale))
        divisor=a.q[2*value.level]*a.q[2*value.level+1]
        scale=value._ckks_scale*(output_scale*divisor/value._ckks_scale)/divisor
        return Value(value.data*weight+bias,value.level-1,scale)
    def integer(arithmetic,value,multiplier,*,bias):
        assert arithmetic is a;arithmetic.calls.append(('integer',multiplier,bias))
        return Value(multiplier*value.data+bias,value.level,value._ckks_scale)
    monkeypatch.setattr(module,'affine_at_scale',affine);monkeypatch.setattr(module,'integer_affine',integer)
    return a


def test_all_thirteen_updates_and_declared_checkpoints(arithmetic):
    x=np.array([.00390625,.04,.2,.4,1.,1.5,1.9,2.])
    initial=np.array([.2,.3,.2,.1,.03,.02,.02,.01]);original=initial.copy();input_copy=x.copy()
    first_scale=backward_scale_plan(arithmetic.q,input_scale=23562.,first_output_level=25,updates=13,final_scale=23562.)['initial_target_scale']
    outputs=dict(module.integer_newton_updates(arithmetic,Value(x),normalization_factor=32,initial_estimate=initial,updates=13,first_output_scale=first_scale))
    expected=x*((-32*initial)*initial)+2.*initial
    for update in range(1,14):
        if update>1:expected=expected*(2.-32*(x*expected))
        assert np.array_equal(outputs[update].data,expected)
    assert list(outputs)==list(range(1,14))
    assert arithmetic.calls==[('first_affine',first_scale)]+['multiply',('integer',-32,2.),'multiply']*12
    assert np.array_equal(initial,original) and np.array_equal(x,input_copy)
    assert outputs[9] is not outputs[13] and outputs[12] is not outputs[13]


@pytest.mark.parametrize('field,value',[
    ('normalization_factor',True),('normalization_factor',32.),('normalization_factor',0),
    ('updates',0),('updates',True),('first_output_scale',float('inf')),
    ('first_output_scale',True),('initial_estimate',0.),('initial_estimate',float('nan')),
    ('initial_estimate',[1.,2.]),('initial_estimate',np.ones((2,4))),('initial_estimate',1e308)])
def test_invalid_contract_before_server_operations(arithmetic,field,value):
    kwargs=dict(normalization_factor=32,initial_estimate=.1,updates=13,first_output_scale=123.)
    kwargs[field]=value
    with pytest.raises(ValueError):list(module.integer_newton_updates(arithmetic,Value(np.ones(8)),**kwargs))
    assert arithmetic.calls==[]


def test_thirteen_update_real_affine_scale_recurrence():
    q=tuple(range(101,155));delta=float(q[-2]*q[-1])
    plan=backward_scale_plan(q,input_scale=delta,first_output_level=25,updates=13,final_scale=delta)
    scale=delta*plan['initial_plaintext_scale']/(q[52]*q[53])
    assert plan['rows'][0]['scale']==scale
    for row in plan['rows'][1:]:
        L=row['level']+2
        product=delta*scale/(q[2*L]*q[2*L+1])
        scale=scale*product/(q[2*(L-1)]*q[2*(L-1)+1])
        assert row['product_scale']==product and row['scale']==scale
    assert [r['level'] for r in plan['rows']]==list(range(25,0,-2))
    assert abs(plan['final_relative_difference'])<2e-12

from types import SimpleNamespace
import numpy as np
import pytest
import encrypted_integer_affine as affine
import encrypted_scaled_newton as newton


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

import math
import pytest
from encrypted_integer_affine import backward_scale_plan

Q=(101,103,107,109,113,127,131,137)


def test_real_initial_affine_and_two_product_metadata():
    delta=float(Q[-2]*Q[-1])
    p=backward_scale_plan(Q,input_scale=delta,first_output_level=2,updates=2,final_scale=delta)
    first=delta*p['initial_plaintext_scale']/(Q[6]*Q[7])
    product=delta*first/(Q[4]*Q[5])
    final=first*product/(Q[2]*Q[3])
    assert p['rows']==[dict(update=1,level=2,scale=first),dict(update=2,level=0,product_scale=product,scale=final)]
    assert abs(p['final_relative_difference'])<1e-14


@pytest.mark.parametrize('field,value',[
    ('updates',0),('updates',True),('first_output_level',1),
    ('first_output_level',3),('input_scale',0.),('input_scale',float('nan')),
    ('final_scale',float('inf')),('final_scale',True)])
def test_invalid_schedule_rejected(field,value):
    args=dict(input_scale=100.,first_output_level=2,updates=2,final_scale=100.)
    args[field]=value
    with pytest.raises(ValueError):backward_scale_plan(Q,**args)


def test_reject_invalid_composite_moduli():
    with pytest.raises(ValueError):
        backward_scale_plan(Q[:-1],input_scale=100.,first_output_level=2,updates=2,final_scale=100.)


@pytest.mark.parametrize('location',['context','cache','parameters'])
def test_key_injection_while_suspended_rejected_before_multiply(arithmetic,location):
    denominator=Value(np.ones(8));g=newton.integer_newton_updates(arithmetic,denominator,**options())
    next(g);calls=list(arithmetic.calls)
    if location=='context':arithmetic.ctx.secret_key='injected sentinel'
    elif location=='cache':arithmetic.ctx._param_cache.secret_key='injected sentinel'
    else:arithmetic.ctx.parameters['secret_key']='injected sentinel'
    with pytest.raises(ValueError,match='keyless'):next(g)
    assert arithmetic.calls==calls


@pytest.mark.parametrize('operation',['integer','affine','newton'])
def test_bound_aware_owner_rejected_before_arithmetic(arithmetic,operation):
    arithmetic.declare_bound=lambda *a,**kw:None
    value=Value(np.ones(8))
    with pytest.raises(ValueError,match='bound-aware'):
        if operation=='integer':affine.integer_affine(arithmetic,value,32)
        elif operation=='affine':affine.affine_at_scale(arithmetic,value,1.,0.,output_scale=123.)
        else:next(newton.integer_newton_updates(arithmetic,value,**options()))
    assert arithmetic.calls==[]


def test_bound_owner_added_during_yield_rejected_before_multiply(arithmetic):
    g=newton.integer_newton_updates(arithmetic,Value(np.ones(8)),**options())
    next(g);calls=list(arithmetic.calls);arithmetic.declare_bound=lambda *a,**kw:None
    with pytest.raises(ValueError,match='bound-aware'):next(g)
    assert arithmetic.calls==calls

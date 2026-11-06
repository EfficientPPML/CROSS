from types import SimpleNamespace
import numpy as np
import pytest
import integer_newton as module
from integer_affine import backward_scale_plan


class Value:
    def __init__(self,data,level=26,scale=23562.,nsd=1):
        self.data=np.asarray(data);self.level=level;self._ckks_scale=scale;self._ckks_nsd=nsd


class Arithmetic:
    def __init__(self,slots):
        self.q=list(range(101,155));self.calls=[];self.refresher=None
        self.ctx=SimpleNamespace(num_slots=slots,composite_degree=2,
            _param_cache=SimpleNamespace(q_moduli_at_level=lambda level:tuple(self.q[:2*(level+1)])))
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

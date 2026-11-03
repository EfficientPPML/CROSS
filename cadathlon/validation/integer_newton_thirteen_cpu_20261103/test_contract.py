from types import SimpleNamespace
import numpy as np
import pytest
import integer_newton as module
from integer_affine import backward_scale_plan


class Arithmetic:
    def __init__(self,slots):self.ctx=SimpleNamespace(num_slots=slots);self.calls=[]
    def multiply(self,left,right):
        self.calls.append('multiply');return left*right


@pytest.fixture
def arithmetic(monkeypatch):
    a=Arithmetic(8)
    def affine(arithmetic,value,weight,bias,*,output_scale):
        assert arithmetic is a;arithmetic.calls.append(('first_affine',output_scale))
        return value*weight+bias
    def integer(arithmetic,value,multiplier,*,bias):
        assert arithmetic is a;arithmetic.calls.append(('integer',multiplier,bias))
        return multiplier*value+bias
    monkeypatch.setattr(module,'affine_at_scale',affine);monkeypatch.setattr(module,'integer_affine',integer)
    return a


def test_all_thirteen_updates_and_declared_checkpoints(arithmetic):
    x=np.array([.00390625,.04,.2,.4,1.,1.5,1.9,2.])
    initial=np.array([.2,.3,.2,.1,.03,.02,.02,.01]);original=initial.copy();input_copy=x.copy()
    outputs=dict(module.integer_newton_updates(arithmetic,x,normalization_factor=32,initial_estimate=initial,updates=13,first_output_scale=123.))
    expected=x*((-32*initial)*initial)+2.*initial
    for update in range(1,14):
        if update>1:expected=expected*(2.-32*(x*expected))
        assert np.array_equal(outputs[update],expected)
    assert list(outputs)==list(range(1,14))
    assert arithmetic.calls==[('first_affine',123.)]+['multiply',('integer',-32,2.),'multiply']*12
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
    with pytest.raises(ValueError):list(module.integer_newton_updates(arithmetic,np.ones(8),**kwargs))
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

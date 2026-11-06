"""Synchronous-mode contract and exact production _finish AST ordering.

All barrier behavior here is mocked. No JAX backend, keys or HE executes.
"""
from pathlib import Path
from types import MethodType,ModuleType,SimpleNamespace
import ast,sys,weakref
import numpy as np
import pytest
from test_causal_softmax_guards import setup,pv_setup,RAW,CERT,CONFIG
from ideal_primitives import Arithmetic,Value
from ideal_refresh import IdentityRefresh
from encrypted_qwen_causal_softmax import QwenCausalSoftmax,QwenCausalSoftmaxContract,QwenProbabilityTile

@pytest.mark.parametrize('mode',[False,None,1,'True'])
@pytest.mark.parametrize('target',['arithmetic','refresh_evaluator','refresh_engine','PV'])
def test_all_modes_must_be_exact_true_before_work(target,mode):
    a=Arithmetic();r=IdentityRefresh(a);c=QwenCausalSoftmaxContract(RAW,CERT,3)
    if target in ('arithmetic','PV'):a.low_memory=mode
    elif target=='refresh_evaluator':r.evaluator.low_memory=mode
    else:r.engine._low_mem=mode
    with pytest.raises(ValueError):
        if target=='PV':pv_setup(a,(0,2))
        else:QwenCausalSoftmax(a,c,CONFIG,positions=(0,2),refresher=r)
    assert not a.calls and not r.calls

@pytest.mark.parametrize('target',['arithmetic','refresh_evaluator','refresh_engine','arithmetic_finish','refresh_finish'])
@pytest.mark.parametrize('phase',['K2.source_branch_mask','output'])
def test_synchronous_flags_and_accessors_bound_through_callbacks(target,phase):
    box={};stopped=[]
    def callback(label,value,meta):
        if label!=phase:return
        o,a,r,x=box['state'];stopped.append(dict(a.calls))
        if target=='arithmetic':a.low_memory=False
        elif target=='refresh_evaluator':r.evaluator.low_memory=1
        elif target=='refresh_engine':r.engine._low_mem=False
        elif target=='arithmetic_finish':a._finish=MethodType(lambda self,value:value,a)
        else:r.evaluator._finish=MethodType(lambda self,value:value,r.evaluator)
    box['state']=setup(callback);o,a,r,x=box['state']
    with pytest.raises(ValueError):o.evaluate_tile(x,query_tile=0)
    assert stopped and dict(a.calls)==stopped[0] and o.last_run is None

@pytest.mark.parametrize('target',['arithmetic','refresh'])
def test_finish_must_bind_original_owner_and_function_before_work(target):
    a=Arithmetic();r=IdentityRefresh(a);other=Arithmetic()
    if target=='arithmetic':a._finish=other._finish
    else:r.evaluator._finish=a._finish
    with pytest.raises(ValueError):QwenCausalSoftmax(a,QwenCausalSoftmaxContract(RAW,CERT,3),CONFIG,positions=(0,2),refresher=r)
    assert not a.calls and not r.calls

def test_custom_finish_subclass_cannot_claim_synchronous_contract():
    class Custom(Arithmetic):
        def _finish(self,value):return value
    a=Custom();r=IdentityRefresh(a)
    with pytest.raises(ValueError):QwenCausalSoftmax(a,QwenCausalSoftmaxContract(RAW,CERT,3),CONFIG,positions=(0,2),refresher=r)
    assert not a.calls

@pytest.mark.parametrize('target',['flag','finish'])
@pytest.mark.parametrize('phase',['key0_probability_anchors','output'])
def test_PV_synchronous_contract_is_bound_after_callbacks(target,phase):
    a=Arithmetic();stopped=[]
    def callback(label,value,meta):
        if label!=phase:return
        stopped.append(dict(a.calls))
        if target=='flag':a.low_memory=False
        else:a._finish=MethodType(lambda self,value:value,a)
    o=pv_setup(a,(0,2),callback)
    result=QwenProbabilityTile(Value(np.zeros(65536),24),(0,2),(0,2),(0,),0,3,{}, {},{})
    with pytest.raises(ValueError):o.attend_softmax(result,(Value(np.zeros(65536),24),),query_tile=0)
    assert stopped and dict(a.calls)==stopped[0] and o.last_run is None

def production_finish():
    tree=ast.parse((Path(__file__).parent/'execution_references/encrypted_polynomial.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EncryptedPolynomialEvaluator')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_finish')
    namespace={};exec(compile(ast.Module(body=[method],type_ignores=[]),'exact_production_finish_AST','exec'),namespace)
    return namespace['_finish']

@pytest.mark.parametrize('failure',[None,RuntimeError('barrier failed'),KeyboardInterrupt('barrier cancelled')])
def test_production_barrier_precedes_owner_callback_and_failure_never_reaches_callback(monkeypatch,failure):
    events=[];finish=production_finish();old_weighted=Arithmetic.weighted_sum
    class Payload(np.ndarray):
        def devices(self):return frozenset(('mock-device',))
    class PublicContext:
        def __init__(self):self.ff_ctx=SimpleNamespace(computation_format_is_standard=True)
    def weighted(self,*args,**kwargs):
        value=old_weighted(self,*args,**kwargs);value.polynomial=value.polynomial.view(Payload)
        value.ntt_ctx=PublicContext()
        return self._finish(value)
    def barrier(payloads):
        events.append('barrier')
        assert isinstance(payloads,list) and all(isinstance(x,Payload) for x in payloads)
        if failure is not None:raise failure
        return payloads
    fake_jax=ModuleType('jax');fake_jax.block_until_ready=barrier
    monkeypatch.setitem(sys.modules,'jax',fake_jax)
    monkeypatch.setattr(Arithmetic,'_finish',finish)
    monkeypatch.setattr(Arithmetic,'weighted_sum',weighted)
    cancel=KeyboardInterrupt('callback cancelled after successful barrier')
    def callback(*unused):events.append('callback');raise cancel
    o,a,r,x=setup(callback);a._live_ntt_contexts=weakref.WeakValueDictionary()
    expected=failure if failure is not None else cancel
    with pytest.raises(type(expected)) as caught:o.evaluate_tile(x,query_tile=0)
    assert caught.value is expected
    assert events==(['barrier'] if failure is not None else ['barrier','callback'])
    assert o.last_run is None and not o._active
    if failure is not None:assert len(a._live_ntt_contexts)==0

def test_actual_file_engine_explicitly_constructs_three_synchronous_facades():
    tree=ast.parse((Path(__file__).parent/'execution_references/full_model_engine.py').read_text())
    selected=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('Arithmetic','Linear','Refresh'):
            keyword=next(k for k in node.keywords if k.arg=='low_memory');assert isinstance(keyword.value,ast.Constant) and keyword.value.value is True;selected.append(node.func.id)
    assert sorted(selected)==['Arithmetic','Linear','Refresh']

def test_backend_not_imported_outside_explicit_mock_scope():
    assert not any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors'))

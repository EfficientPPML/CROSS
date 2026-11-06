"""Focused fail-before-next-primitive and retained-original regressions."""
import math
from types import SimpleNamespace
import numpy as np
import pytest
from tail_ideal import build,pack,synthetic_inputs,Arithmetic,Linear,DELTA
import encrypted_qwen_decoder_tail as runtime
import encrypted_qwen_silu as silu_runtime

@pytest.fixture(autouse=True)
def clear_fake_quarantine():
    runtime._QUARANTINE.clear();silu_runtime._QUARANTINE.clear()
    yield
    runtime._QUARANTINE.clear();silu_runtime._QUARANTINE.clear()

def distinct(monkeypatch,positions=(0,),progress=None):
    a=Arithmetic();b=Arithmetic();b.ctx=a.ctx
    for name in ('rotation','_plaintext_multiply','_rescale'):getattr(b,name).ctx=a.ctx
    o,a,r,w,g=build(monkeypatch,0,positions,progress,arithmetic=a,linear=Linear(b))
    source,attended=synthetic_inputs(0,positions);inputs,attention=pack(source,attended,positions,0)
    return o,a,b,r,inputs,attention

def mutate(o,what):
    a=o.linear.arithmetic
    if what=='linear_scale':a.scale=math.nextafter(a.scale,math.inf)
    elif what=='linear_strategy':a.strategy='chebyshev'
    elif what=='linear_progress':a.progress=lambda *args:None
    elif what=='linear_context':a.rotation.ctx=object()
    elif what=='linear_finish':a._finish=lambda x:x
    elif what=='linear_low_memory':a.low_memory=1
    elif what=='compiler_capacity':a._plaintext_multiply.compiler.max_executables=True
    elif what=='compiler_cache':a._plaintext_multiply.compiler._executables={}
    elif what=='compiler_method':a._plaintext_multiply.compiler.bind=lambda *args:None
    elif what=='matrix_map':o.ctx._param_cache.coef_maps[o._linear_keys[0]]=object()
    elif what=='matrix_pair':o.ctx._param_cache.raw_rotation_keys[o._linear_keys[0]]['a']=object()
    elif what=='outer_refresh':o.outer.refresh_event=lambda *args,**kwargs:None
    elif what=='outer_check':o.outer.check=lambda:None
    elif what=='norm':o.norm.evaluate_tile=lambda *args,**kwargs:None
    elif what=='silu':o.silu.evaluate_tile=lambda *args,**kwargs:None
    elif what=='weights':o.weights=tuple(np.array(w) for w in o.weights)
    else:raise AssertionError(what)

CASES=('linear_scale','linear_strategy','linear_progress','linear_context','linear_finish','linear_low_memory',
       'compiler_capacity','compiler_cache','compiler_method','matrix_map','matrix_pair','outer_refresh','outer_check','norm','silu','weights')
@pytest.mark.parametrize('what',CASES)
def test_before_work_bindings(monkeypatch,what):
    o,a,b,r,inputs,attention=distinct(monkeypatch);mutate(o,what)
    with pytest.raises((ValueError,TypeError)):o.evaluate(attention,inputs)
    assert not a.calls and not b.calls and not r.calls and not runtime._QUARANTINE

@pytest.mark.parametrize('what',CASES)
def test_first_callback_bindings(monkeypatch,what):
    held={};counts=[]
    def progress(*unused):
        counts.append((dict(held['a'].calls),dict(held['b'].calls)));mutate(held['o'],what)
    o,a,b,r,inputs,attention=distinct(monkeypatch,progress=progress);held.update(o=o,a=a,b=b)
    with pytest.raises((ValueError,TypeError)):o.evaluate(attention,inputs)
    assert len(counts)==1 and (dict(a.calls),dict(b.calls))==counts[0]
    assert not b.calls and not r.calls and runtime._QUARANTINE and o.last_run is None

@pytest.mark.parametrize('field,value',[('layer',False),('layer',0.),('sequence_length',True),('sequence_length',1.),('outer_events',[])])
def test_typed_contract_before_work(monkeypatch,field,value):
    o,a,b,r,inputs,attention=distinct(monkeypatch);object.__setattr__(o.contract,field,value)
    with pytest.raises(ValueError):o.evaluate(attention,inputs)
    assert not a.calls and not b.calls

@pytest.mark.parametrize('field,value',[('public_bound',32),('input_level',21.),('output_level',True),('layer',False),('stage','normalized_post')])
def test_typed_outer_event_before_work(monkeypatch,field,value):
    o,a,b,r,inputs,attention=distinct(monkeypatch);object.__setattr__(o.contract.outer_events[0],field,value)
    with pytest.raises(ValueError):o.evaluate(attention,inputs)
    assert not a.calls and not b.calls

@pytest.mark.parametrize('which',['pending_original','pending_attention','attention_metadata','attention_positions','current_moduli','current_nsd'])
def test_first_callback_retains_pending_sibling_inputs(monkeypatch,which):
    held={};counts=[]
    def progress(label,value,metadata):
        counts.append(dict(held['a'].calls))
        if which=='pending_original':held['inputs'][1].polynomial=np.zeros(65536)
        elif which=='pending_attention':held['attention'].outputs[1].num_moduli=1
        elif which=='attention_metadata':held['attention'].metadata[1]['scale']*=2
        elif which=='attention_positions':object.__setattr__(held['attention'],'original_positions',(0,))
        elif which=='current_moduli':value.num_moduli=1
        else:value._ckks_nsd=True
    positions=tuple(range(19));o,a,b,r,inputs,attention=distinct(monkeypatch,positions,progress)
    original=inputs[1].polynomial;held.update(a=a,inputs=inputs,attention=attention)
    with pytest.raises(ValueError):o.evaluate(attention,inputs)
    assert len(counts)==1 and dict(a.calls)==counts[0] and not b.calls and runtime._QUARANTINE
    assert any(state[1][0]is original for state in runtime._QUARANTINE[0][3])

@pytest.mark.parametrize('which',['original','raw_residual','up','output'])
def test_final_callback_retention(monkeypatch,which):
    held={};counts=[]
    def progress(label,value,metadata):
        if label=='tile0.attention_residual':held['residual']=value
        if label=='tile0.up.linear':held['up']=value
        if label=='tile0.output':
            counts.append((dict(held['a'].calls),dict(held['b'].calls)))
            victim={'original':held['inputs'][0],'raw_residual':held['residual'],'up':held['up'],'output':value}[which]
            victim.num_moduli=1
    o,a,b,r,inputs,attention=distinct(monkeypatch,progress=progress);held.update(a=a,b=b,inputs=inputs)
    with pytest.raises(ValueError):o.evaluate(attention,inputs)
    assert counts and (dict(a.calls),dict(b.calls))==counts[0] and runtime._QUARANTINE and o.last_run is None

@pytest.mark.parametrize('error',[RuntimeError,KeyboardInterrupt,SystemExit])
def test_callback_primary_and_reuse(monkeypatch,error):
    def progress(*args):raise error('chosen')
    o,a,b,r,inputs,attention=distinct(monkeypatch,progress=progress)
    with pytest.raises(error,match='chosen'):o.evaluate(attention,inputs)
    assert runtime._QUARANTINE and not o._active and o.last_run is None
    with pytest.raises(RuntimeError,match='dedicated-process'):o.evaluate(attention,inputs)

def test_callback_reentry_rejects_before_next_primitive(monkeypatch):
    held={}
    def progress(*args):held['o'].evaluate(held['attention'],held['inputs'])
    o,a,b,r,inputs,attention=distinct(monkeypatch,progress=progress);held.update(o=o,attention=attention,inputs=inputs)
    with pytest.raises(ValueError,match='reentered'):o.evaluate(attention,inputs)
    assert a.calls=={'weighted':1} and not b.calls and runtime._QUARANTINE

@pytest.mark.parametrize('field,value',[('level',3),('_ckks_scale',1.),('_ckks_nsd',True),('num_moduli',1)])
def test_add_metadata_independent_expectation(monkeypatch,field,value):
    a=Arithmetic();real=a.add
    # Install before construction so this tests output validation rather than
    # merely the accessor mutation guard.
    def bad_add(x,y,**kwargs):
        result=real(x,y,**kwargs)
        if kwargs.get('subtract',False)is False and result.level in (21,19,4):setattr(result,field,value)
        return result
    a.add=bad_add
    b=Arithmetic();b.ctx=a.ctx
    for name in ('rotation','_plaintext_multiply','_rescale'):getattr(b,name).ctx=a.ctx
    o,a,r,w,g=build(monkeypatch,arithmetic=a,linear=Linear(b));source,attended=synthetic_inputs(0,(0,));inputs,attention=pack(source,attended,(0,),0)
    with pytest.raises(ValueError):o.evaluate(attention,inputs)
    assert not r.calls and runtime._QUARANTINE

def test_callback_compiler_cache_and_original_zero_array(monkeypatch):
    held={}
    def progress(*args):
        held['o'].linear.arithmetic._constant_cache['injected']=object()
    o,a,b,r,inputs,attention=distinct(monkeypatch,progress=progress);inputs[0].polynomial[:]=0.;old=inputs[0].polynomial;held['o']=o
    with pytest.raises(ValueError,match='cache'):o.evaluate(attention,inputs)
    assert any(state[1][0]is old for state in runtime._QUARANTINE[0][3])

def test_annotations_stripped_and_original_untouched(monkeypatch):
    def progress(label,value,metadata):value._public_abs_bound=0.
    o,a,b,r,inputs,attention=distinct(monkeypatch,progress=progress);inputs[0]._public_abs_bound=7.
    result=o.evaluate(attention,inputs)
    assert inputs[0]._public_abs_bound==7. and not any(k.startswith('_public_') for k in vars(result.outputs[0]))

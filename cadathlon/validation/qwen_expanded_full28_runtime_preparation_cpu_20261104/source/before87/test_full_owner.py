"""Synthetic component tests only; no parameter body or fullmodel execution."""
import math
import numpy as np
import pytest
from full_synthetic import build,Attention,Tail,Final,DELTA
import encrypted_qwen_full_model as runtime
from encrypted_scale_planning import weighted_sum_scale,drop_divisor

@pytest.fixture(autouse=True)
def clear():
    runtime._QUARANTINE.clear();yield;runtime._QUARANTINE.clear()

@pytest.mark.parametrize('sequence',[1,8,16,19,32])
def test_actual_objects_flow_and_last_row_only(monkeypatch,sequence):
    positions=tuple(2*i+3 for i in range(sequence));o,a,b,r,p,inputs=build(monkeypatch,positions)
    before=[(v.polynomial,v.polynomial.copy()) for v in inputs]
    refreshed=[];r.inject=lambda source,result:refreshed.append((source,result))
    result=o.evaluate(inputs)
    assert len(Attention.records)==len(Tail.records)==28
    for layer,(ar,tr) in enumerate(zip(Attention.records,Tail.records)):
        assert ar[0]==tr[0]==layer and ar[1]is tr[2]
        if not layer:assert ar[1]is inputs
        assert tr[1].layer==layer and tr[1].outputs
        if layer:
            for tile,value in enumerate(ar[1]):assert value is refreshed[(layer-1)*len(inputs)+tile][1]
    assert len(r.calls)==27*len(inputs)+1
    assert [bound for _,bound in r.calls]==[e.bound for e in o.contract.entry_events[:-1] for _ in inputs]+[2048.]
    normalized,original,tile=Final.records[0]
    assert original==(positions[-1],) and tile==0 and normalized.level==25
    expected=.01*sequence+.001*sum(range(1,29))
    np.testing.assert_allclose(normalized.polynomial[:1024],expected,rtol=0,atol=1e-14)
    assert np.count_nonzero(normalized.polynomial[1024:])==0
    assert len(result.logit_blocks)==38 and sum(d.stop-d.start for d in result.blocks)==151936
    assert result.last_position==positions[-1] and result.original_positions==positions
    for v,d in zip(result.logit_blocks,result.blocks):
        assert v.level==6 and np.isfinite(v.polynomial).all()
        np.testing.assert_allclose(v.polynomial[:d.stop-d.start],expected,rtol=0,atol=1e-14)
        assert not np.count_nonzero(v.polynomial[d.stop-d.start:])
    assert len(o.linear.calls)==38 and o.linear.calls[-1][1]==(384,1024)
    for v,(original,payload) in zip(inputs,before):assert v.polynomial is original and np.array_equal(v.polynomial,payload)
    assert o.last_run is not None and not o._refs and not runtime._QUARANTINE

CASES=('linear_scale','linear_strategy','linear_ctx','linear_finish','linear_low_memory','compiler','keys','provider','outer','contract','factory')
def mutate(o,what):
    a=o.linear.arithmetic
    if what=='linear_scale':a.scale=math.nextafter(a.scale,math.inf)
    elif what=='linear_strategy':a.strategy='chebyshev'
    elif what=='linear_ctx':a.rotation.ctx=object()
    elif what=='linear_finish':a._finish=lambda x:x
    elif what=='linear_low_memory':a.low_memory=1
    elif what=='compiler':a.rotation._compiler._executables={}
    elif what=='keys':o.ctx._param_cache.raw_rotation_keys[o._linear_keys[0]]['a']=object()
    elif what=='provider':o.parameters.matrix=lambda n:None
    elif what=='outer':o.outer.refresh_event=lambda *a,**k:None
    elif what=='contract':object.__setattr__(o.contract,'last_tile',False)
    elif what=='factory':o._make_tail=lambda n:None

@pytest.mark.parametrize('what',CASES)
@pytest.mark.parametrize('when',['before','first'])
def test_construction_bindings(monkeypatch,what,when):
    held={};calls=[]
    def callback(*args):
        calls.append((dict(held['a'].calls),dict(held['b'].calls)));mutate(held['o'],what)
    o,a,b,r,p,inputs=build(monkeypatch,progress=callback if when=='first' else None,distinct=True);held.update(o=o,a=a,b=b)
    if when=='before':mutate(o,what)
    with pytest.raises((ValueError,TypeError)):o.evaluate(inputs)
    if when=='before':assert not a.calls and not b.calls and not p.calls and not r.calls and not runtime._QUARANTINE
    else:assert len(calls)==1 and (dict(a.calls),dict(b.calls))==calls[0] and runtime._QUARANTINE
    assert o.last_run is None

@pytest.mark.parametrize('victim',['original','current','pending','refs','child_method','child_progress','result_metadata'])
def test_callback_retains_values_and_children(monkeypatch,victim):
    held={};counts=[]
    def callback(label,value,metadata):
        counts.append(dict(held['a'].calls));o=held['o']
        if victim=='original':held['inputs'][0].num_moduli=1
        elif victim=='pending':held['inputs'][-1].polynomial=np.zeros(65536)
        elif victim=='current':value._ckks_nsd=True
        elif victim=='refs':o._refs.clear()
        elif victim=='child_method':o._child.evaluate=lambda *args:None
        elif victim=='child_progress':o._child.progress=lambda *args:None
        elif victim=='result_metadata':
            if label=='layer0.attention.output0':o._result.metadata[0]['tile']=False
            else:counts.pop();return
    o,a,b,r,p,inputs=build(monkeypatch,tuple(range(19)),callback);original=inputs[-1].polynomial;held.update(o=o,a=a,inputs=inputs)
    with pytest.raises(ValueError):o.evaluate(inputs)
    assert len(counts)==1 and dict(a.calls)==counts[0] and runtime._QUARANTINE
    assert any(s[1][0]is original for s in runtime._QUARANTINE[0][2])

@pytest.mark.parametrize('error',[RuntimeError,KeyboardInterrupt,SystemExit])
def test_primary_and_failure_quarantine(monkeypatch,error):
    def callback(*args):raise error('chosen')
    o,a,b,r,p,inputs=build(monkeypatch,progress=callback)
    with pytest.raises(error,match='chosen'):o.evaluate(inputs)
    assert runtime._QUARANTINE and not o._active and o.last_run is None
    with pytest.raises(RuntimeError,match='dedicated-process'):o.evaluate(inputs)

def test_reentry(monkeypatch):
    held={}
    def callback(*args):held['o'].evaluate(held['x'])
    o,a,b,r,p,x=build(monkeypatch,progress=callback);held.update(o=o,x=x)
    with pytest.raises(ValueError,match='reentered'):o.evaluate(x)
    assert a.calls=={'synthetic_attention':1}

@pytest.mark.parametrize('label,victim',[(label,victim) for label in ('final.select_last','final.row_to_zero','final.input.refresh0','final.norm.output','logits.37') for victim in ('current','original')]+[('logits.37','retained_logit')])
def test_final_phases_retained_metadata(monkeypatch,label,victim):
    held={};counts=[]
    def callback(actual,value,metadata):
        if actual!=label:return
        counts.append(dict(held['a'].calls))
        x=value if victim=='current' else held['x'][0] if victim=='original' else held['o']._refs['logits.0'][0]
        x.num_moduli=1
    o,a,b,r,p,x=build(monkeypatch,(3,7),callback);held.update(o=o,a=a,x=x)
    with pytest.raises(ValueError):o.evaluate(x)
    assert len(counts)==1 and dict(a.calls)==counts[0] and runtime._QUARANTINE and o.last_run is None

def test_literal_refresh_scales_and_output_metadata(monkeypatch):
    observed=[]
    def callback(label,value,metadata):
        if '.refresh' in label:observed.append(value._ckks_scale)
    o,a,b,r,p,x=build(monkeypatch,(3,),callback);out=o.evaluate(x)
    assert observed==[math.nextafter(DELTA,math.inf if i%2==0 else -math.inf) for i in range(28)]
    for v,row in zip(out.logit_blocks,out.metadata):assert row['scale']==v._ckks_scale and type(row['scale'])is float

@pytest.mark.parametrize('field,value',[('level',24),('_ckks_nsd',True),('num_moduli',1),('_ckks_scale',float('nan'))])
def test_bad_entry_before_load(monkeypatch,field,value):
    o,a,b,r,p,x=build(monkeypatch);setattr(x[0],field,value)
    with pytest.raises(ValueError):o.evaluate(x)
    assert not a.calls and not p.calls and not r.calls

def test_borrowed_tags_and_owned_cleanup(monkeypatch):
    def callback(label,value,metadata):value._public_abs_bound=0.
    o,a,b,r,p,x=build(monkeypatch,progress=callback);x[0]._public_abs_bound=9.
    out=o.evaluate(x)
    assert x[0]._public_abs_bound==9.
    assert not any(n.startswith('_public_') for v in out.logit_blocks for n in vars(v))

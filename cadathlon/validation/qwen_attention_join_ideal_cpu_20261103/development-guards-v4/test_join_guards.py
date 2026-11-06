"""Distinct borrowed facade and retained-object guard regressions."""
import math
import numpy as np
import pytest
from test_join import P,make_contract,inputs
from join_ideal import Arithmetic,Linear,OuterIdentityRefresh,install_matrix,gamma,weights,DELTA
import encrypted_qwen_attention_join as runtime

@pytest.fixture(autouse=True)
def clean_fake_quarantine():
    runtime._QUARANTINE.clear()
    yield
    runtime._QUARANTINE.clear()

def build_distinct(monkeypatch,progress=None):
    install_matrix(monkeypatch);a=Arithmetic();b=Arithmetic();b.ctx=a.ctx
    for name in ('rotation','_plaintext_multiply','_rescale'):getattr(b,name).ctx=a.ctx
    linear=Linear(b);refresh=OuterIdentityRefresh(a);raw,w=weights(0)
    owner=runtime.QwenAttentionJoin(a,linear,make_contract(),(P/'inputs/config.json').read_bytes(),
        parameter_manifest_bytes=raw,weights=w,gammas=gamma(0),positions=(0,),refresher=refresh,progress=progress)
    return owner,a,b,refresh

def mutate(owner,which):
    a=owner.linear.arithmetic
    if which=='scale':a.scale=math.nextafter(a.scale,math.inf)
    elif which=='strategy':a.strategy='chebyshev'
    elif which=='progress':a.progress=lambda *args:None
    elif which in ('rotation','_plaintext_multiply','_rescale'):getattr(a,which).ctx=object()
    elif which=='capacity':a._ensure_capacity=lambda values,*unused:values
    elif which=='finish':a._finish=lambda value:value
    elif which=='layout':object.__setattr__(owner.linear.layout,'features',4096.)
    elif which=='key':
        key=owner._linear_keys[0];owner.ctx._param_cache.raw_rotation_keys[key]={'a':object(),'b':object()}

@pytest.mark.parametrize('which',['scale','strategy','progress','rotation','_plaintext_multiply','_rescale','capacity','finish','layout','key'])
def test_distinct_facade_before_entry(monkeypatch,which):
    owner,a,b,_=build_distinct(monkeypatch);mutate(owner,which);_,tiles=inputs((0,))
    with pytest.raises(ValueError):owner.evaluate(tiles)
    assert not a.calls and not b.calls and not runtime._QUARANTINE

@pytest.mark.parametrize('which',['scale','strategy','progress','rotation','_plaintext_multiply','_rescale','capacity','finish','layout','key'])
def test_distinct_facade_at_first_callback(monkeypatch,which):
    holder={}
    def progress(*args):mutate(holder['owner'],which)
    owner,a,b,_=build_distinct(monkeypatch,progress);holder['owner']=owner;_,tiles=inputs((0,))
    with pytest.raises(ValueError):owner.evaluate(tiles)
    assert a.calls and not b.calls and runtime._QUARANTINE

@pytest.mark.parametrize('field,value',[('input_level',7.),('input_level',True),('public_bound',64),('output_level',25.),('layer',False),('stage','q_raw')])
def test_complete_typed_outer_event_before_work(monkeypatch,field,value):
    owner,a,b,_=build_distinct(monkeypatch);object.__setattr__(owner.contract.outer_events[0],field,value)
    _,tiles=inputs((0,))
    with pytest.raises(ValueError):owner.evaluate(tiles)
    assert not a.calls and not b.calls

def test_outer_event_clone_not_authorized(monkeypatch):
    owner,a,b,_=build_distinct(monkeypatch);event=owner.contract.outer_events[0]
    clone=runtime.QwenOuterRefreshEvent(event.stage,event.layer,event.input_level,event.public_bound)
    _,tiles=inputs((0,))
    with pytest.raises(ValueError,match='construction-bound outer event'):owner.outer.refresh_event(tiles[0],event=clone)
    assert not a.calls and not b.calls

def test_all_zero_input_callback_keeps_original_arrays(monkeypatch):
    holder={}
    def progress(*args):holder['value'].polynomial=np.zeros(65536)
    owner,a,b,_=build_distinct(monkeypatch,progress);_,tiles=inputs((0,));tiles[0].polynomial[:]=0.;holder['value']=tiles[0]
    original=tiles[0].polynomial
    with pytest.raises(ValueError):owner.evaluate(tiles)
    retained=runtime._QUARANTINE[0][2]
    assert any(state[1][0]is original for state in retained)
    assert not b.calls

"""Malformed returned wrappers retain raw arrays before validation failure."""
import pytest
from full_synthetic import build,Attention,Tail,Final
import encrypted_qwen_full_model as runtime
@pytest.fixture(autouse=True)
def clear():
    runtime._QUARANTINE.clear();yield;runtime._QUARANTINE.clear()

@pytest.mark.parametrize('child',[Attention,Tail,Final])
def test_returned_arrays_survive_bad_metadata(monkeypatch,child):
    method='evaluate_tile' if child is Final else 'evaluate';real=getattr(child,method);seen=[]
    def broken(self,*args,**kwargs):
        result=real(self,*args,**kwargs)
        values=(result.output,) if child is Final else result.outputs
        seen.extend((v,v.polynomial,v.moduli_array) for v in values);values[-1].num_moduli=1
        return result
    monkeypatch.setattr(child,method,broken)
    o,a,b,r,p,x=build(monkeypatch)
    with pytest.raises(ValueError):o.evaluate(x)
    assert seen and runtime._QUARANTINE and o.last_run is None
    pending=runtime._QUARANTINE[0][-2]
    assert all(any(v is old and data is payload and mods is rawmods for old,payload,rawmods in pending[1]) for v,data,mods in seen)

@pytest.mark.parametrize('error',[RuntimeError,KeyboardInterrupt,SystemExit])
def test_pre_return_failure_keeps_whole_owner(monkeypatch,error):
    def failed(self,*args):raise error('before returned owner')
    monkeypatch.setattr(Attention,'evaluate',failed)
    o,a,b,r,p,x=build(monkeypatch)
    with pytest.raises(error,match='before returned'):o.evaluate(x)
    row=runtime._QUARANTINE[0]
    assert row[0]is o and o.arithmetic is a and o.refresher is r and row[1]is x and all(s[1][0]is v.polynomial for s,v in zip(row[2],x))
    assert not o._active and o.last_run is None

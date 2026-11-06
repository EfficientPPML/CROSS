"""Borrowed caller annotations are preserved; owned results stay stripped."""
import numpy as np
import pytest
from tail_ideal import build,pack,synthetic_inputs
import encrypted_qwen_decoder_tail as runtime

@pytest.fixture(autouse=True)
def clear_fake_quarantine():
    runtime._QUARANTINE.clear();yield;runtime._QUARANTINE.clear()

@pytest.mark.parametrize('failure',[False,True])
def test_supplied_attention_annotations_survive(monkeypatch,failure):
    def progress(label,value,metadata):
        if failure and label=='tile0.o.linear':raise RuntimeError('chosen body failure')
        value._public_interval=(-1.,1.)
    o,a,r,w,g=build(monkeypatch,progress=progress);source,attended=synthetic_inputs(0,(0,));inputs,attention=pack(source,attended,(0,),0)
    original=inputs[0];borrowed=attention.outputs[0]
    original._public_abs_bound=7.;borrowed._public_abs_bound=9.;borrowed._public_interval=(-9.,9.)
    if failure:
        with pytest.raises(RuntimeError,match='chosen body failure'):o.evaluate(attention,inputs)
    else:
        result=o.evaluate(attention,inputs)
        assert not any(name.startswith('_public_') for name in vars(result.outputs[0]))
    assert original._public_abs_bound==7.
    assert borrowed._public_abs_bound==9. and borrowed._public_interval==(-9.,9.)

@pytest.mark.parametrize('which',['replace','clear'])
def test_callback_cannot_discard_retained_owner_map(monkeypatch,which):
    held={};counts=[]
    def progress(*unused):
        counts.append(dict(held['a'].calls))
        if which=='replace':held['o']._refs={}
        else:held['o']._refs.clear()
        held['inputs'][0].polynomial=np.zeros(65536)
    o,a,r,w,g=build(monkeypatch,progress=progress);s,v=synthetic_inputs(0,(0,));inputs,attention=pack(s,v,(0,),0)
    original_array=inputs[0].polynomial;held.update(o=o,a=a,inputs=inputs)
    with pytest.raises(ValueError,match='retained'):o.evaluate(attention,inputs)
    assert counts==[dict(a.calls)] and runtime._QUARANTINE and o.last_run is None
    assert any(state[1][0]is original_array for state in runtime._QUARANTINE[0][3])

@pytest.mark.parametrize('field,value',[('tile',False),('level',22.),('scale',np.float64(1.1467305003699404e18))])
def test_attention_public_metadata_requires_exact_types(monkeypatch,field,value):
    o,a,r,w,g=build(monkeypatch);s,v=synthetic_inputs(0,(0,));inputs,attention=pack(s,v,(0,),0)
    attention.metadata[0][field]=value
    with pytest.raises(ValueError,match='metadata'):o.evaluate(attention,inputs)
    assert not a.calls and not r.calls and not runtime._QUARANTINE

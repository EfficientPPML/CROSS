"""Full-slot composition tests with explicit primitive/identity-refresh doubles."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
import pytest
from collections import Counter
from join_ideal import Arithmetic,Linear,OuterIdentityRefresh,install_matrix,gamma,weights,dense_reference,Value,DELTA
import encrypted_qwen_attention_join as runtime
from ideal_metrics import record
P=Path(__file__).resolve().parent

def make_contract(layer=0):
    return runtime.QwenAttentionJoinContract((P/'inputs/candidate_manifest.json').read_bytes(),
        (P/'inputs/site_metadata.json').read_bytes(),(P/'inputs/certificates.json').read_bytes(),
        (P/'inputs/join_plan.json').read_bytes(),layer)

@pytest.fixture(autouse=True)
def isolated_quarantine():
    # Only test doubles are ever held here. Production exposes no reset API.
    runtime._QUARANTINE.clear()
    yield
    runtime._QUARANTINE.clear()

def build(monkeypatch,positions=(0,),layer=0,progress=None):
    install_matrix(monkeypatch);a=Arithmetic();linear=Linear(a);refresh=OuterIdentityRefresh(a)
    raw,w=weights(layer);g=gamma(layer)
    owner=runtime.QwenAttentionJoin(a,linear,make_contract(layer),(P/'inputs/config.json').read_bytes(),
        parameter_manifest_bytes=raw,weights=w,gammas=g,positions=positions,refresher=refresh,progress=progress)
    return owner,a,refresh,w,g

def inputs(positions,layer=0):
    # Predeclared public synthetic vectors, never model/validation trajectories.
    scale=.035 if layer==0 else 10.
    rows=np.stack([scale*(np.sin(np.arange(1024)*.137+i*.053)+.1*np.cos(np.arange(1024)*.061+i*.101)) for i in range(len(positions))])
    values=[]
    for start in range(0,len(rows),16):
        data=np.full((16,4096),37.);part=rows[start:start+16];data[:len(part),:1024]=part
        values.append(Value(data.ravel(),25,math.nextafter(DELTA,math.inf)))
    return rows,tuple(values)

def expected_counts(owner):
    """Independent phase count formulas plus actual public sparse plans."""
    from encrypted_packed_qwen import MatrixDiagonals
    from encrypted_chained_streaming_linear import linear_rotation_plan
    result=Counter();tiles=owner.layout.tile_count;sequence=len(owner.layout.positions)
    for norm in owner.norms:
        c=norm.contract;bits=c.width.bit_length()-1
        result.update(rotate=tiles*2*bits,add=tiles*2*bits,
            weighted=tiles*(c.degree+3 if c.method=='polynomial' else c.iterations+2),
            multiply=tiles*(c.degree+1 if c.method=='polynomial' else 3*c.iterations-1))
    for weight in owner.weights:
        p=linear_rotation_plan(MatrixDiagonals(weight,65536,4096),strategy='chained')
        result.update(rotate=tiles*p.nonzero_rotation_calls,weighted=tiles*len(p.giant_indices),add=tiles*(len(p.giant_indices)-1))
    result.update(rotate=tiles*20,weighted=tiles*4) # two RoPE and two GQA maps
    if sequence==1:result.update(weighted=1);return result
    s=owner.softmax.contract.squarings
    for tile in range(tiles):
        rows=owner.layout.tile_rows(tile)
        broadcasts=sum(rows-int(key%16<rows) for key in range(sequence))
        # QK and PV each perform seven head reductions/broadcasts, fifteen
        # head-offset rotations and the nonzero global key shifts.
        result.update(rotate=2*(broadcasts+22*sequence+sequence-1),
            add=2*(8*sequence-1),weighted=4*sequence+1,multiply=2*sequence)
        branches=0
        for row in range(rows):
            k=16*tile+row+1
            if k==1:continue
            branches+=1;n=owner.softmax.contract.key_rules[k-1][3]
            result.update(rotate=20+10*s,add=21+10*s,weighted=14+s*(n+2),multiply=18+2*s*n)
        result.update(add=max(0,branches-1))
    return result

@pytest.mark.parametrize('layer',range(28))
def test_all_typed_contracts(layer):
    c=make_contract(layer);runtime.verify_contract(c)
    assert len(c.outer_events)==5+sum(e.stage.endswith('_reserve') for e in c.outer_events)
    assert c.outer_events[0].stage=='normalized_input'
    assert all(type(e.public_bound)is float and e.output_level==25 for e in c.outer_events)

def test_genuine_pinned_qkv_one_row(monkeypatch):
    owner,a,refresh,w,g=build(monkeypatch);rows,tiles=inputs((0,))
    expected,domains=dense_reference(rows,w,g,0,(0,));assert all(r['passed'] for r in domains),domains
    before=[v.polynomial.copy() for v in tiles]
    result=owner.evaluate(tiles)
    want=np.zeros((16,4096));want[0,:2048]=expected[0]
    np.testing.assert_allclose(result.outputs[0].polynomial,want.ravel(),atol=1e-8,rtol=0.)
    assert result.metadata[0]['level']==22
    assert len(result.refresh_events)==5
    assert len(refresh.calls)==5
    assert a.calls==expected_counts(owner)
    for old,value in zip(before,tiles):np.testing.assert_array_equal(old,value.polynomial)
    assert owner.last_run is not None
    record(layer=0,sequence=1,positions=(0,),domains=domains,max_abs_error=float(np.max(np.abs(result.outputs[0].polynomial-want.ravel()))),
        counts=a.calls,metadata=result.metadata,refreshes=result.refresh_events)

@pytest.mark.parametrize('layer,positions',[(0,tuple(range(0,16,2))),(0,tuple(range(0,57,3))),(5,(7,)),(9,(11,))])
def test_genuine_pinned_distinct_rows(monkeypatch,layer,positions):
    owner,a,refresh,w,g=build(monkeypatch,positions,layer);rows,tiles=inputs(positions,layer)
    expected,domains=dense_reference(rows,w,g,layer,positions)
    assert all(r['passed'] for r in domains),domains
    old=tuple(v.polynomial.copy() for v in tiles);result=owner.evaluate(tiles)
    worst=0.
    for tile,value in enumerate(result.outputs):
        n=min(16,len(rows)-tile*16);want=np.zeros((16,4096));want[:n,:2048]=expected[tile*16:tile*16+n]
        np.testing.assert_allclose(value.polynomial,want.ravel(),atol=1e-8,rtol=0.)
        worst=max(worst,float(np.max(np.abs(value.polynomial-want.ravel()))))
        assert value.level==(22 if len(rows)==1 else 20)
    assert len(result.refresh_events)==len(owner.contract.outer_events)*len(tiles)
    assert a.calls==expected_counts(owner)
    expected_refreshes=len(result.refresh_events)+len(tiles)*sum(len(n.contract.events) for n in owner.norms)
    expected_refreshes+=(len(rows)-1)*(3+2*owner.softmax.contract.squarings)
    assert len(refresh.calls)==expected_refreshes
    for before,value in zip(old,tiles):np.testing.assert_array_equal(before,value.polynomial)
    record(layer=layer,sequence=len(rows),positions=positions,domains=domains,max_abs_error=worst,
        counts=a.calls,metadata=result.metadata,refreshes=result.refresh_events)

def test_final_output_callback_quarantines(monkeypatch):
    def callback(label,value,metadata):
        if label=='tile0.output':value.num_moduli=1
    owner,_,_,_,_=build(monkeypatch,progress=callback);_,tiles=inputs((0,))
    with pytest.raises(ValueError):owner.evaluate(tiles)
    assert runtime._QUARANTINE and owner.last_run is None

@pytest.mark.parametrize('target,value',[('layer',True),('layer',0.),('outer_events',[])])
def test_contract_alias_before_work(monkeypatch,target,value):
    owner,a,_,_,_=build(monkeypatch);setattr_value=lambda:object.__setattr__(owner.contract,target,value)
    setattr_value();_,tiles=inputs((0,))
    with pytest.raises((TypeError,ValueError)):owner.evaluate(tiles)
    assert not a.calls and not runtime._QUARANTINE

@pytest.mark.parametrize('flag',[False,1,None])
def test_sync_before_work(monkeypatch,flag):
    owner,a,_,_,_=build(monkeypatch);a.low_memory=flag;_,tiles=inputs((0,))
    with pytest.raises(ValueError):owner.evaluate(tiles)
    assert not a.calls

@pytest.mark.parametrize('which',['refresh','linear','norm','pending_input','result_moduli'])
def test_first_callback_quarantines(monkeypatch,which):
    holder={};called=[]
    def callback(label,value,metadata):
        if called:return
        called.append(label);owner=holder['owner']
        if which=='refresh':owner.refresher.refresh=lambda *a,**kw:None
        elif which=='linear':owner.linear.linear_tile=lambda *a,**kw:None
        elif which=='norm':owner.norms[1].evaluate_tile=lambda *a,**kw:None
        elif which=='pending_input':holder['tiles'][1].num_moduli=1
        else:value.num_moduli=1
    owner,a,_,_,_=build(monkeypatch,tuple(range(19)),progress=callback);_,tiles=inputs(tuple(range(19)));holder.update(owner=owner,tiles=tiles)
    with pytest.raises(ValueError):owner.evaluate(tiles)
    assert called and runtime._QUARANTINE and owner.last_run is None
    with pytest.raises(RuntimeError):owner.evaluate(tiles)

@pytest.mark.parametrize('error',[RuntimeError,KeyboardInterrupt,SystemExit])
def test_entered_failure_quarantine(monkeypatch,error):
    def callback(*args):raise error('chosen')
    owner,a,refresh,_,_=build(monkeypatch,progress=callback);_,tiles=inputs((0,))
    with pytest.raises(error,match='chosen'):owner.evaluate(tiles)
    assert runtime._QUARANTINE and runtime._QUARANTINE[0][0]is owner
    assert owner.last_run is None and owner._active is False

def test_no_backend_imports():
    assert not any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors'))

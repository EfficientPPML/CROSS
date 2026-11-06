"""Actual learned maps/owner loops with explicitly ideal primitive payloads."""
import json,math,os,sys
import numpy as np
import pytest
from tail_ideal import P,build,contract,reference,pack,synthetic_inputs,expected_counts,DELTA
import encrypted_qwen_decoder_tail as runtime
import encrypted_qwen_silu as silu_runtime
import encrypted_qwen_attention_join as attention_runtime
import encrypted_qwen_rmsnorm as rms_runtime

@pytest.fixture(autouse=True)
def clear_fake_quarantine():
    for m in (runtime,silu_runtime,attention_runtime):m._QUARANTINE.clear()
    yield
    for m in (runtime,silu_runtime,attention_runtime):m._QUARANTINE.clear()

@pytest.mark.parametrize('layer',range(28))
@pytest.mark.parametrize('sequence',[1,19])
def test_all_typed_contracts(layer,sequence):
    c=contract(layer,sequence);runtime.verify_contract(c)
    assert c.outer_events[0].input_level==(21 if sequence==1 else 19)
    assert c.outer_events[0].stage=='raw_post' and c.outer_events[1].stage=='normalized_post'
    assert all(type(e.public_bound)is float and e.output_level==25 for e in c.outer_events)

def assert_result(o,a,r,result,expected,inputs,old,domains,key):
    assert a.calls==expected_counts(o)
    assert len(r.calls)==o.layout.tile_count*(2+len(o.norm.contract.events)+o.silu.contract.doublings)
    worst=0.
    for tile,value in enumerate(result.outputs):
        n=o.layout.tile_rows(tile);want=np.zeros((16,4096));want[:n,:1024]=expected[16*tile:16*tile+n]
        assert value.polynomial.shape==(65536,) and np.isfinite(value.polynomial).all()
        error=float(np.max(np.abs(value.polynomial-want.ravel())))
        assert math.isfinite(error) and error<=1e-8
        worst=max(worst,error)
        assert value.level==4
        mask_scale=next(x['scale'] for x in o.last_run['stages'] if x['label']==f'tile{tile}.original_residual_mask')
        assert value._ckks_scale==mask_scale
        assert not np.any(value.polynomial.reshape(16,4096)[:n,1024:]) and not np.any(value.polynomial.reshape(16,4096)[n:])
    for value,before in zip(inputs,old):np.testing.assert_array_equal(value.polynomial,before)
    assert o.last_run and not o._refs and not o._phase_refs
    path=os.environ.get('QWEN_TAIL_METRICS_PATH')
    if path:
        from pathlib import Path
        p=Path(path);rows=json.loads(p.read_bytes()) if p.exists() else {};assert key not in rows
        rows[key]=dict(layer=o.contract.layer,sequence=o.contract.sequence_length,max_abs_error=worst,gate=1e-8,physical_slots=65536*len(inputs),counts=dict(a.calls),domains=domains,metadata=result.metadata,outer_refreshes=result.refresh_events)
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n');tmp.replace(p)

@pytest.mark.parametrize('layer,sequence',[(3,1),(3,19),(15,1),(27,1)])
def test_learned_tail_standins(monkeypatch,layer,sequence):
    positions=tuple(3*i+1 for i in range(sequence));o,a,r,w,g=build(monkeypatch,layer,positions)
    source,attended=synthetic_inputs(layer,positions);inputs,attention=pack(source,attended,positions,layer)
    expected,domains=reference(source,attended,w,g,layer);old=[x.polynomial.copy() for x in inputs]
    # Real downstream masks must reject inactive noise in refresh stand-ins.
    def inject(old,new):new.polynomial.reshape(16,4096)[:,3072:]+=1e-12
    r.inject=inject
    result=o.evaluate(attention,inputs)
    assert_result(o,a,r,result,expected,inputs,old,domains,f'layer{layer}.S{sequence}.standin')

@pytest.mark.parametrize('sequence',[1,19])
def test_actual_preceding_attention_objects(monkeypatch,sequence):
    from test_join import make_contract,inputs as attention_inputs
    from join_ideal import dense_reference,weights,gamma,Linear,install_matrix
    from tail_ideal import Arithmetic,OuterIdentityRefresh
    positions=tuple(2*i for i in range(sequence));install_matrix(monkeypatch)
    a=Arithmetic();linear=Linear(a);r=OuterIdentityRefresh(a);raw,wq=weights(0);gq=gamma(0)
    head=attention_runtime.QwenAttentionJoin(a,linear,make_contract(),(P/'inputs/config.json').read_bytes(),
        parameter_manifest_bytes=raw,weights=wq,gammas=gq,positions=positions,refresher=r)
    source,inputs=attention_inputs(positions);attended,attention_domains=dense_reference(source,wq,gq,0,positions)
    attention=head.evaluate(inputs)
    o,a,r,w,g=build(monkeypatch,0,positions,arithmetic=a,linear=head.linear,refresher=r)
    expected,domains=reference(source,attended,w,g,0);old=[x.polynomial.copy() for x in inputs]
    a.calls.clear();r.calls.clear()
    result=o.evaluate(attention,inputs)
    assert_result(o,a,r,result,expected,inputs,old,attention_domains+domains,f'layer0.S{sequence}.joined')

def test_no_backend_model_imports():
    assert not any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors'))

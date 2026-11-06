"""Full128-key attention loops with explicit ideal payloads; no HE/model claim."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,math
import numpy as np
import pytest
from test_bsgs import doubles, Arithmetic as BaseArithmetic, Value, Q, DELTA
import encrypted_bert_bsgs_attention as module
from encrypted_bert_bsgs_attention import BertBSGSAttentionTile

class Arithmetic(BaseArithmetic):
    def __init__(self):
        super().__init__()
        cache=self.ctx._param_cache
        for k in BertBSGSAttentionTile.required_rotation_indices():
            cache.raw_rotation_keys.setdefault(k,{n:SimpleNamespace(shape=(8,114,131072),dtype=np.dtype('uint32')) for n in ('a','b')})
            cache.coef_maps.setdefault(k,SimpleNamespace(shape=(131072,),dtype=np.dtype('int32')))
    def multiply(self,left,right):
        self.calls.append(('multiply',));level=min(left.level,right.level)
        return Value(left.polynomial.data*right.polynomial.data,level-1,
            left._ckks_scale*right._ckks_scale/math.prod(Q[2*level:2*level+2]))

@pytest.fixture(autouse=True)
def retirement(doubles):
    assert not module._QUARANTINE
    yield
    # Synthetic objects only; real execution must retire its dedicated process.
    module._QUARANTINE.clear()

def sources(level,scale):
    rng=np.random.default_rng(20261103)
    query=rng.uniform(-.25,.25,(16,4096));keys=rng.uniform(-.25,.25,(128,4096));values=rng.uniform(-.25,.25,(128,4096))
    # Arbitrary finite unused features must not contaminate the active768 outputs.
    query[:,768:]=17.;keys[:,768:]=-19.;values[:,768:]=23.
    probabilities=rng.uniform(0.,1.,(16,128,12));probabilities/=probabilities.sum(axis=1,keepdims=True)
    p=np.full((16,4096),-13.);p[:,:2048].reshape(16,128,16)[:,:,:12]=probabilities
    wrap=lambda a:Value(a.reshape(-1),level,scale)
    return query,keys,values,probabilities,wrap(query),tuple(wrap(x) for x in keys.reshape(8,16,4096)),tuple(wrap(x) for x in values.reshape(8,16,4096)),wrap(p)

@pytest.mark.parametrize('level,scale',[(49,DELTA),(25,math.nextafter(DELTA,math.inf)),(25,DELTA*1.21)])
@pytest.mark.parametrize('operation',['scores','values'])
def test_all_keys_all_physical_slots(level,scale,operation):
    ops=Arithmetic();owner=BertBSGSAttentionTile(ops)
    q,k,v,p,query,keys,values,probabilities=sources(level,scale)
    originals=(query,*keys,*values,probabilities)
    states=[(x.polynomial,x._ckks_scale,x.level,x.polynomial.data.copy(),x._public_bound) for x in originals]
    expected=np.zeros((16,4096))
    if operation=='scores':
        actual=owner.scores(query,keys)
        exact=np.einsum('rhd,khd->rkh',q[:,:768].reshape(16,12,64),k[:,:768].reshape(128,12,64))/8
        expected[:,:2048].reshape(16,128,16)[:,:,:12]=exact
    else:
        actual=owner.mix_values(probabilities,values)
        expected[:,:768]=np.einsum('rkh,khd->rhd',p,v[:,:768].reshape(128,12,64)).reshape(16,768)
    data=actual.polynomial.data
    assert data.shape==(65536,) and np.isfinite(data).all() and np.isfinite(expected).all()
    error=float(np.max(np.abs(data-expected.ravel())));assert math.isfinite(error) and error<=1e-8
    assert actual.level==level-3 and actual._ckks_nsd==1 and not hasattr(actual,'_public_bound')
    for x,(payload,s,L,raw,annotation) in zip(originals,states):
        assert x.polynomial is payload and x._ckks_scale==s and x.level==L and x._public_bound==annotation
        np.testing.assert_array_equal(x.polynomial.data,raw)
    c=owner.last_counts
    assert c['ciphertext_products']==128 and c['head_rotations']==768 and c['row_alignment_rotations']==120 and c['accumulator_additions']==127
    assert c['route_rotations']==1520 and c['route_public_multiplications']==3584 and c['route_rescales']==896
    record=dict(operation=operation,input_level=level,input_scale=scale,output_level=actual.level,output_scale=actual._ckks_scale,physical_slots=65536,max_abs_error=error,gate=1e-8,counts=c)
    with (Path(__file__).parent/'ideal_results.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')

@pytest.mark.parametrize('mutation',['callback','multiply','row_key','head_map'])
def test_construction_bindings_before_work(mutation):
    a=Arithmetic();owner=BertBSGSAttentionTile(a)
    if mutation=='callback':owner.progress=lambda *args:None
    elif mutation=='multiply':a.multiply=lambda *args:None
    elif mutation=='row_key':a.ctx._param_cache.raw_rotation_keys[4096]['a']=SimpleNamespace(shape=(8,114,131072),dtype=np.dtype('uint32'))
    else:a.ctx._param_cache.coef_maps[32]=SimpleNamespace(shape=(131072,),dtype=np.dtype('int32'))
    with pytest.raises(ValueError):owner._guard()
    assert not a.calls

@pytest.mark.parametrize('operation',['scores','values'])
@pytest.mark.parametrize('fault',['source','result','callback','cancel'])
def test_callback_and_failure_retention(operation,fault):
    a=Arithmetic();primary=KeyboardInterrupt('same cancellation');box={}
    def callback(label,value,counts):
        if fault=='source':box['query']._ckks_nsd=True
        elif fault=='result':value._ckks_nsd=True
        elif fault=='callback':box['owner'].progress=lambda *args:None
        else:raise primary
    owner=BertBSGSAttentionTile(a,progress=callback);q,k,v,p,query,keys,values,probabilities=sources(25,DELTA)
    first=query if operation=='scores' else probabilities;box.update(query=first,owner=owner)
    with pytest.raises((ValueError,KeyboardInterrupt)) as caught:
        (owner.scores if operation=='scores' else owner.mix_values)(first,keys if operation=='scores' else values)
    if fault=='cancel':assert caught.value is primary
    assert module._QUARANTINE and module._QUARANTINE[-1][-1] is caught.value and not owner._active
    with pytest.raises(RuntimeError):owner._guard()

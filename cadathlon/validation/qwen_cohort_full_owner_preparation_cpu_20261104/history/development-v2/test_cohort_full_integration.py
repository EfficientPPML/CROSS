"""Actual join/factory plus real cohort/PV; explicit synthetic upstream seams.

No historical gamma()/weights()/parameters() helper is invoked. Parameter
verification, RMS, QK/head maps and linear values are labeled synthetic below;
this is wiring/lifetime and cohort/PV arithmetic, not whole-model numerics.
"""
from pathlib import Path
from types import SimpleNamespace
import gc,hashlib,json,math,sys,weakref
import numpy as np
import pytest
P=Path(__file__).resolve().parent
sys.path[:0]=[str(P),str(P/'checkout/cadathlon/common')]
import test_softmax_cohorts as leaf
from join_ideal import Arithmetic,OuterIdentityRefresh,Value,DELTA
from encrypted_streaming_linear import RowTileLayout
from encrypted_scale_planning import drop_divisor,weighted_sum_scale
import encrypted_qwen_attention_join as join
import encrypted_qwen_full_model as full
import encrypted_qwen_causal_softmax_batched as cohort
from plan_full import contract

class SyntheticNorm:
    def __init__(self,a,c,config,*,positions,progress,**kwargs):
        self.arithmetic=a;self.ctx=a.ctx;self.contract=c;self.positions=positions;self.progress=progress
    def _rms_guard(self):pass
    def _fingerprint(self,v):return id(v.polynomial),v.level,v._ckks_scale
    def _validate(self,v,depth):pass
    def evaluate_tile(self,value,*,tile):
        out=Value(np.zeros(65536),self.contract.output_level,DELTA)
        self.progress('synthetic.norm',out,dict(level=out.level,scale=out._ckks_scale))
        return SimpleNamespace(output=out)

class SyntheticLinear:
    def __init__(self,a):self.arithmetic=a;self.ctx=a.ctx;self.layout=RowTileLayout(65536)
    def required_rotation_indices(self,*args):return ()
    def _validate_ciphertext(self,*args):pass
    def linear_tile(self,value,weight,*,valid_rows):
        self.arithmetic.calls['synthetic_linear']+=1
        scale=weighted_sum_scale([value._ckks_scale],DELTA,drop_divisor(self.ctx._param_cache.q_towers,value.level))['output_scale']
        return Value(np.zeros(65536),value.level-1,scale)

def native_v(n):return np.sin(np.arange(n*16*128,dtype=np.float64).reshape(n,16,128)*.013)

class SyntheticHead(join._CompleteHead):
    def delivered(self,label,data,level,scale):
        value=Value(data,level,scale);self.progress('synthetic.'+label,value,dict(level=level,scale=scale));return value
    def rope(self,value,*,tile,heads):
        self._guard();scale=weighted_sum_scale([value._ckks_scale],DELTA,drop_divisor(self.ctx._param_cache.q_towers,value.level))['output_scale']
        return self.delivered('rope',np.zeros(65536),value.level-1,scale)
    def expand_kv(self,value,*,tile):
        self._guard();data=np.zeros((16,4096));n=len(self.layout.positions);count=self.layout.tile_rows(tile)
        if value.level==24:data[:count,:2048]=native_v(n)[tile*16:tile*16+count].reshape(count,2048)
        scale=weighted_sum_scale([value._ckks_scale],DELTA,drop_divisor(self.ctx._param_cache.q_towers,value.level))['output_scale']
        return self.delivered('expand',data.ravel(),value.level-1,scale)
    def scores(self,query,keys,*,query_tile):
        self._guard();assert all(k.level==24 for k in keys)
        value=self.delivered('scores',leaf.scores(len(self.layout.positions),query_tile),21,DELTA)
        return SimpleNamespace(output=value)

class SyntheticParameters:
    def __init__(self,c):self.contract=c
    def check(self):pass
    def matrix(self,name):
        assert '.self_attn.'in name
        return np.zeros((2048 if '.q_proj.'in name else 1024,1024),dtype=np.float64)
    def gamma(self,name):return np.ones(128 if '.q.'in name or '.k.'in name else 1024)

def setup(monkeypatch,positions,progress=None):
    # Only the public-parameter authentication seam is replaced. The actual
    # join still checks every synthetic array shape/dtype/hash and freezes it.
    def synthetic_parameters(manifest,raw,role):
        assert role=='qkv'and hashlib.sha256(raw).hexdigest()==join.PARAMETERS
        data=json.loads(raw)
        hashes={}
        for row in data['matrices'].values():
            shape=tuple(row['shape'])
            if shape not in hashes:hashes[shape]=hashlib.sha256(np.zeros(shape,dtype='<f8').tobytes()).hexdigest()
            row['sha256']=hashes[shape]
        return data
    monkeypatch.setattr(join,'verify_parameters',synthetic_parameters)
    monkeypatch.setattr(join,'_CompleteRMS',SyntheticNorm);monkeypatch.setattr(join,'_CompleteHead',SyntheticHead)
    monkeypatch.setattr(join,'ChainedStreamingLinear',SyntheticLinear)
    monkeypatch.setattr(full,'ChainedStreamingLinear',SyntheticLinear)
    monkeypatch.setattr(full,'QwenParameterSource',SyntheticParameters)
    a=Arithmetic();linear=SyntheticLinear(a);r=OuterIdentityRefresh(a);c=contract(positions);parameters=SyntheticParameters(c)
    owner=full.QwenFullModel(a,linear,c,parameters,refresher=r,progress=progress)
    return owner,a,r

def inputs(n):
    rows=[]
    for start in range(0,n,16):
        data=np.full((16,4096),27.);data[:min(16,n-start),:1024]=.125
        rows.append(Value(data.ravel(),25,math.nextafter(DELTA,math.inf)))
    return tuple(rows)

@pytest.fixture(autouse=True)
def synthetic_process_isolation(monkeypatch):
    assert not any((cohort._QUARANTINE,join._QUARANTINE,full._QUARANTINE))
    import join_ideal
    def denied(*args,**kwargs):raise AssertionError('historical parameter/gamma body helper forbidden')
    monkeypatch.setattr(join_ideal,'gamma',denied);monkeypatch.setattr(join_ideal,'weights',denied)
    yield
    cohort._QUARANTINE.clear();join._QUARANTINE.clear();full._QUARANTINE.clear();gc.collect()

@pytest.mark.parametrize('n,layer',[(1,0),(8,0),(19,0),(32,3)])
def test_actual_full_factory_and_join_real_v23_pv(monkeypatch,n,layer):
    positions=leaf.positions(n);owner,a,r=setup(monkeypatch,positions)
    child=owner._make_attention(layer)
    assert type(child)is join.QwenAttentionJoin and type(child.softmax)is cohort.QwenCausalSoftmaxBatched
    assert child.softmax in child._children and child.softmax.layout.positions==positions
    values=inputs(n);original=tuple(v.polynomial for v in values)
    result=owner._child_call('layer%d.attention'%layer,child,lambda:child.evaluate(values))
    owner._bind_result(result,values,layer,'attention')
    assert result.original_positions==positions and len(result.outputs)==len(values)
    native=native_v(n)
    for tile,value in enumerate(result.outputs):
        want=np.zeros((16,4096));data=leaf.scores(n,tile).reshape(16,4096)
        for row in range(min(16,n-16*tile)):
            k=16*tile+row+1
            if k==1:want[row,:2048]=native[0].ravel()
            else:
                probability=leaf.dense(data[row,:512].reshape(16,32)[:,:k],k,layer)
                want[row,:2048]=np.einsum('hk,khd->hd',probability,native[:k]).ravel()
        np.testing.assert_allclose(value.polynomial,want.ravel(),rtol=0,atol=5e-12)
        assert value.level==(22 if n==1 else 20)and result.metadata[tile]['scale']==value._ckks_scale
    assert all(v.polynomial is p for v,p in zip(values,original))
    softmax_refreshes=sum(len(t)*(3+2*child.softmax.contract.squarings)for t in child.softmax.cohorts)
    assert len(r.calls)==len(values)*len(child.contract.outer_events)+softmax_refreshes
    assert not any((cohort._QUARANTINE,join._QUARANTINE,full._QUARANTINE))
    assert len(a.ctx._param_cache.raw_rotation_keys)==166

@pytest.mark.parametrize('layer',[0,13,27])
def test_real_full_factory_selects_bound_cohort_for_arbitrary_layer(monkeypatch,layer):
    owner,a,r=setup(monkeypatch,(3,7,10));child=owner._make_attention(layer)
    assert type(child)is join.QwenAttentionJoin and type(child.softmax)is cohort.QwenCausalSoftmaxBatched
    assert child.softmax.contract.layer==layer and child.softmax.progress is child._child_progress
    assert child.progress is owner._child_progress and child.softmax.cohorts[0][0].queries[0].position==7
    assert not a.calls and not r.calls

@pytest.mark.parametrize('mutation',['refresh','check','descriptor','body'])
def test_parent_callback_replacement_retains_entered_child(monkeypatch,mutation):
    held={};originals=[];error=KeyboardInterrupt('synthetic public primary')
    def progress(label,value,metadata):
        if '.softmax.C0.'not in label or not label.endswith('normalized_t.refresh'):return
        child=held['owner']._child;held['child']=child;originals.append(weakref.ref(value.polynomial))
        if mutation=='body':value.polynomial=np.zeros(65536);raise error
        if mutation=='descriptor':object.__setattr__(child.softmax.cohorts[0][0],'refresh_bound',16.)
        else:setattr(child.softmax.refresh_binding,mutation,lambda *args,**kwargs:value)
    owner,a,r=setup(monkeypatch,leaf.positions(8),progress);held['owner']=owner
    values=inputs(8)
    with pytest.raises(KeyboardInterrupt if mutation=='body'else ValueError)as caught:
        owner.evaluate(values)
    if mutation=='body':assert caught.value is error
    child=held['child']
    gc.collect();assert originals[0]()is not None and join._QUARANTINE and cohort._QUARANTINE and full._QUARANTINE
    assert owner._child is child and child.softmax._failure is not None
    assert child.last_run is None and child.softmax.last_run is None
    with pytest.raises(RuntimeError):owner._full_guard()

def test_no_model_backend_imported_or_parameter_helper_called():
    assert not any(name in sys.modules for name in('jax','jaxlib','torch','transformers','safetensors'))

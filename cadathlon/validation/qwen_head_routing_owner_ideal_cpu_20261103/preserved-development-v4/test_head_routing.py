"""Ideal fullslot payload tests of actual owner loops; no encrypted claim."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
from collections import Counter
import json,sys,math
import numpy as np
import pytest
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'checkout/cadathlon/common'))
from encrypted_scale_planning import weighted_sum_scale,multiply_scale,drop_divisor

class Value:
    def __init__(self,data,level=25,scale=None):
        self.polynomial=np.array(data,dtype=np.float64,copy=True);self.level=level
        self._ckks_scale=DELTA if scale is None else scale;self._ckks_nsd=1
        self.batch=1;self.num_elements=2;self.degree=131072;self.degree_layout=('ideal',)
        self.moduli=tuple(Q[:2*(level+1)]);self.ntt_ctx=None

class Arithmetic:
    def __init__(self):
        self.scale=DELTA;self.strategy='power';self.refresher=None;self.progress=None;self.calls=Counter()
        # Supply every possible public rotation in the double. Tests separately
        # remove/replace entries from the owner's exact inventory.
        keys={i:{'a':object(),'b':object()} for i in range(1,65536)};maps={i:object() for i in keys}
        k=SimpleNamespace(degree=131072,batch=1,composite_degree=2,dnum=8,degree_layout=('ideal',),q_towers=Q,p_towers=PP,secret_key=None,raw_rotation_keys=keys,coef_maps=maps,per_level_rotation_keys=False)
        self.ctx=SimpleNamespace(degree=131072,num_slots=65536,composite_degree=2,dnum=8,degree_layout=('ideal',),secret_key=None,parameters={},_param_cache=k,max_level=49,scaling_factor=DELTA,q_towers=Q,p_towers=PP)
        self.rotation=SimpleNamespace(ctx=self.ctx,rotate=lambda *a:None)
        self._plaintext_multiply=SimpleNamespace(ctx=self.ctx,multiply=lambda *a:None)
        self._rescale=SimpleNamespace(ctx=self.ctx,rescale=lambda *a:None)
        self._multiplication_compiler=object()
    def _level(self,x):return x.level
    def rotate(self,x,s):self.calls['rotate']+=1;return Value(np.roll(x.polynomial,-s),x.level,x._ckks_scale)
    def weighted_sum(self,values,weights):
        self.calls['weighted']+=1;l=min(x.level for x in values);d=drop_divisor(Q,l)
        data=values[0].polynomial*weights[0]
        for x,w in zip(values[1:],weights[1:]):data=data+x.polynomial*w
        return Value(data,l-1,weighted_sum_scale([x._ckks_scale for x in values],DELTA,d)['output_scale'])
    def multiply(self,x,y):
        self.calls['multiply']+=1;l=min(x.level,y.level)
        return Value(x.polynomial*y.polynomial,l-1,multiply_scale(x._ckks_scale,y._ckks_scale,drop_divisor(Q,l)))
    def add(self,x,y):
        self.calls['add']+=1
        assert abs(x._ckks_scale-y._ckks_scale)<=max(x._ckks_scale,y._ckks_scale)*1e-12
        return Value(x.polynomial+y.polynomial,min(x.level,y.level),x._ckks_scale)

profile=next(r for r in json.loads((P/'profile_audit.json').read_bytes())['profiles'] if r['name']=='depth49_p60_d8')
Q=tuple(profile['q']);PP=tuple(profile['p']);DELTA=profile['canonical_scale']
for name,attrs in [('encrypted_packed_qwen',{'CompiledArithmetic':Arithmetic}),('jaxite_word',{'Polynomial':Value}),('support',{'require_secure_execution':lambda c,v:None})]:
    m=ModuleType(name);m.__dict__.update(attrs);sys.modules[name]=m
from encrypted_qwen_head_routing import QwenHeadRouting,QwenHeadLayout

def owner(positions,progress=None):
    return QwenHeadRouting(Arithmetic(),(P/'candidate_manifest.json').read_bytes(),(P/'config.json').read_bytes(),positions=positions,progress=progress)
def tile(rows,width,seed=1,scale=None,level=25):
    rng=np.random.default_rng(seed);x=rng.normal(size=(16,4096));x[:,width:]=1e6
    x[rows:]=np.arange((16-rows)*4096).reshape(16-rows,4096)-1e6
    return Value(x.ravel(),level,scale)
def expected_rope(x,positions,heads):
    out=np.zeros((16,4096));theta=json.loads((P/'config.json').read_bytes())['rope_theta']
    for r,pos in enumerate(positions):
        for h in range(heads):
            v=x[r,h*128:(h+1)*128];result=[]
            for d in range(128):
                angle=pos*(theta**(-np.arange(0,128,2)/128))[d%64]
                shifted=-v[d+64] if d<64 else v[d-64]
                result.append(v[d]*math.cos(angle)+shifted*math.sin(angle))
            out[r,h*128:(h+1)*128]=result
    return out

@pytest.mark.parametrize('rows,tile_index',[(1,0),(7,0),(16,0),(19,0),(19,1),(32,0),(32,1)])
def test_gqa_distinct_heads_rows_and_hostile_padding(rows,tile_index):
    o=owner(tuple(range(rows)));r=o.layout.tile_rows(tile_index);x=tile(r,1024,seed=rows)
    source=x.polynomial.copy();out=o.expand_kv(x,tile=tile_index)
    expected=np.zeros((16,4096));expected[:r,:2048]=np.repeat(source.reshape(16,4096)[:r,:1024].reshape(r,8,128),2,axis=1).reshape(r,2048)
    np.testing.assert_array_equal(out.polynomial,expected.ravel());np.testing.assert_array_equal(x.polynomial,source)
    assert o.last_run['counts']==dict(rotations=8,weighted_sums=1,ciphertext_products=0,additions=0)

@pytest.mark.parametrize('heads',[8,16])
@pytest.mark.parametrize('positions',[(0,),(0,3,7,21),tuple(range(16)),tuple(range(0,64,2))])
def test_rope_original_positions_halfsplit(heads,positions):
    o=owner(positions)
    for t in range(o.layout.tile_count):
        n=o.layout.tile_rows(t);x=tile(n,heads*128,seed=t+heads,scale=math.nextafter(DELTA,math.inf))
        out=o.rope(x,tile=t,heads=heads)
        expected=expected_rope(x.polynomial.reshape(16,4096),o.layout.tile_positions(t),heads)
        np.testing.assert_allclose(out.polynomial,expected.ravel(),rtol=0,atol=3e-15)
        assert out.level==24 and o.last_run['counts']['rotations']==2

@pytest.mark.parametrize('count',[1,8,16,19,32])
def test_all_causal_scores_against_independent_dense(count):
    positions=tuple(2*k for k in range(count));o=owner(positions)
    q=np.random.default_rng(count).normal(size=(count,16,128));k=np.random.default_rng(count+1).normal(size=(count,8,128))
    keys=[]
    for t in range(o.layout.tile_count):
        n=o.layout.tile_rows(t);x=tile(n,1024,seed=77,level=26)
        x.polynomial.reshape(16,4096)[:n,:1024]=k[t*16:t*16+n].reshape(n,1024)
        keys.append(o.expand_kv(o.rope(x,tile=t,heads=8),tile=t))
    # Independent per-token native array expression, not routes or head anchors.
    all_k=np.concatenate([expected_rope(np.pad(k[t*16:t*16+o.layout.tile_rows(t)].reshape(o.layout.tile_rows(t),1024),((0,16-o.layout.tile_rows(t)),(0,3072))),o.layout.tile_positions(t),8)[:o.layout.tile_rows(t),:1024].reshape(-1,8,128) for t in range(o.layout.tile_count)])
    for t in range(o.layout.tile_count):
        n=o.layout.tile_rows(t);x=tile(n,2048,seed=91,level=26)
        x.polynomial.reshape(16,4096)[:n,:2048]=q[t*16:t*16+n].reshape(n,2048)
        actual_query=o.rope(x,tile=t,heads=16);result=o.scores(actual_query,tuple(keys),query_tile=t)
        dense_q=expected_rope(x.polynomial.reshape(16,4096),o.layout.tile_positions(t),16)[:n,:2048].reshape(n,16,128)
        reference=np.einsum('rhd,khd->rhk',dense_q,np.repeat(all_k,2,axis=1))/math.sqrt(128)
        expected=np.zeros((16,4096))
        for row,pos in enumerate(o.layout.tile_positions(t)):
            for h in range(16):
                for key,kpos in enumerate(positions):expected[row,h*32+key]=reference[row,h,key] if kpos<=pos else 0
        np.testing.assert_allclose(result.output.polynomial,expected.ravel(),rtol=0,atol=5e-14)
        assert result.level==21 # original26 -> keyRoPE25 -> GQA24 -> score21
        assert result.counts['weighted_sums']==2*count and result.counts['ciphertext_products']==count
        expected_rot=sum(n-int(j%16<n)+7+15+int(j!=0) for j in range(count))
        assert result.counts['rotations']==expected_rot
        assert result.counts['additions']==7*count+count-1

@pytest.mark.parametrize('mutation',['arithmetic','layout','context','keys','maps','mode','theta','boundaware'])
def test_before_entry_mutations_do_no_work(mutation):
    o=owner((0,1));x=tile(2,1024);a=o.arithmetic
    if mutation=='arithmetic':o.arithmetic=Arithmetic()
    elif mutation=='layout':object.__setattr__(o.layout,'head_dim',128.0)
    elif mutation=='context':o.ctx._param_cache.composite_degree=True
    elif mutation=='keys':o.ctx._param_cache.raw_rotation_keys[o.required_rotation_indices()[0]]={'a':object(),'b':object()}
    elif mutation=='maps':o.ctx._param_cache.coef_maps[o.required_rotation_indices()[0]]=object()
    elif mutation=='mode':o.ctx._param_cache.per_level_rotation_keys=0
    elif mutation=='theta':o.theta=1.0
    else:a.declare_bound=lambda *x:None
    with pytest.raises((ValueError,TypeError)):o.expand_kv(x,tile=0)
    assert not a.calls

@pytest.mark.parametrize('phase',['first','final'])
@pytest.mark.parametrize('alias',[True,1.0])
def test_callback_mutation_rejects_and_clears_state(phase,alias):
    held={}
    def callback(label,value,meta):
        if (phase=='first' and label=='gqa_expand') or (phase=='final' and label=='output'):held['input']._ckks_nsd=alias
    o=owner((0,),callback);held['input']=tile(1,1024)
    with pytest.raises(ValueError):o.expand_kv(held['input'],tile=0)
    assert o.last_run is None and not o._active and o.arithmetic.calls['weighted']==1

def test_prior_score_callback_result_is_retained():
    seen={}
    def cb(label,value,meta):
        if label=='key0_scores':seen['prior']=value
        if label=='key1_broadcast':seen['prior']._ckks_nsd=True
    o=owner((0,1),cb)
    with pytest.raises(ValueError):o.scores(tile(2,2048),(tile(2,2048),),query_tile=0)
    assert o.last_run is None and not o._active

def test_callback_semantic_annotations_are_not_exported():
    def cb(label,value,meta):value._public_abs_bound=0
    o=owner((0,),cb);x=tile(1,1024);x._public_abs_bound=9
    out=o.expand_kv(x,tile=0)
    assert not hasattr(out,'_public_abs_bound') and x._public_abs_bound==9

def test_bad_positions_and_manifest_are_rejected():
    for positions in ([0],(True,),(0.0,),(0,0),(2,1),tuple(range(33))):
        with pytest.raises(ValueError):owner(positions)
    with pytest.raises(ValueError):QwenHeadRouting(Arithmetic(),b'{}',(P/'config.json').read_bytes(),positions=(0,))

@pytest.mark.parametrize('field,bad',[('batch',True),('degree',131072.0),('num_elements',2.0),('_ckks_nsd',True),('_ckks_nsd',1.0),('_ckks_scale',True),('_ckks_scale',float('inf')),('level',2)])
def test_invalid_ciphertext_fails_before_primitives(field,bad):
    o=owner((0,));x=tile(1,1024);setattr(x,field,bad)
    with pytest.raises((ValueError,TypeError)):o.expand_kv(x,tile=0)
    assert not o.arithmetic.calls

def test_cancel_and_reentry_leave_borrowed_context_open():
    held={}
    def cb(label,value,meta):
        with pytest.raises(ValueError):held['owner'].expand_kv(held['input'],tile=0)
        raise KeyboardInterrupt('fixed cancellation')
    o=owner((0,),cb);held.update(owner=o,input=tile(1,1024));cache=o.ctx._param_cache
    with pytest.raises(KeyboardInterrupt):o.expand_kv(held['input'],tile=0)
    assert not o._active and o.last_run is None and o.ctx._param_cache is cache

def test_pinned_max_position_and_no_padding_rotation_leak():
    o=owner((0,40959));x=tile(2,2048)
    out=o.rope(x,tile=0,heads=16)
    expected=expected_rope(x.polynomial.reshape(16,4096),(0,40959),16)
    np.testing.assert_allclose(out.polynomial,expected.ravel(),rtol=0,atol=3e-15)
    with pytest.raises(ValueError):owner((0,40960))

@pytest.mark.parametrize('field',['rotation','_plaintext_multiply','_rescale','_multiplication_compiler'])
@pytest.mark.parametrize('phase',['before','callback'])
def test_compiled_helper_replacement_stops_next_work(field,phase):
    held={}
    def corrupt():setattr(held['owner'].arithmetic,field,object())
    def cb(label,v,m):
        if phase=='callback':corrupt()
    o=owner((0,),cb);held['owner']=o;x=tile(1,1024)
    if phase=='before':corrupt()
    with pytest.raises(ValueError):o.expand_kv(x,tile=0)
    assert o.arithmetic.calls['weighted']==(0 if phase=='before' else 1)
    assert o.last_run is None and not o._active

@pytest.mark.parametrize('field',['rotation','_plaintext_multiply','_rescale'])
def test_helper_context_replacement_before_entry(field):
    o=owner((0,));getattr(o.arithmetic,field).ctx=object()
    with pytest.raises(ValueError):o.expand_kv(tile(1,1024),tile=0)
    assert not o.arithmetic.calls

"""Synthetic full-slot cohorts vs dense frozen-rule oracle and literal owner.

No learned parameter, token, model, backend, key or ciphertext body is loaded.
The explicit NumPy doubles model plaintext ideal operations and actual scales.
"""
from pathlib import Path
from collections import Counter
import gc,hashlib,json,math,sys,weakref
import numpy as np
import pytest

P=Path(__file__).resolve().parent
sys.path[:0]=[str(P),str(P/'checkout/cadathlon/common')]
from ideal_primitives import Arithmetic,Value,DELTA,Q
from ideal_refresh import IdentityRefresh
from encrypted_qwen_causal_softmax import QwenCausalSoftmax,QwenCausalSoftmaxContract
import encrypted_qwen_causal_softmax_batched as batched
from encrypted_qwen_attention_bypass import QwenAttentionRoutingWithBypass

RAW=(P/'inputs/candidate_manifest.json').read_bytes()
CERT=(P/'inputs/certificates.json').read_bytes()
CONFIG=(P/'inputs/config.json').read_bytes()
M=json.loads(RAW);C=json.loads(CERT)
KEYS=tuple(json.loads((P/'inputs/actual_key_plan.json').read_bytes())['combined_keys'])

@pytest.fixture(autouse=True)
def isolated_ideal_process():
    # Test isolation only. Production has no quarantine-reset API.
    assert not batched._QUARANTINE
    yield
    batched._QUARANTINE.clear();gc.collect()

def positions(n):return tuple(3*i+i//5 for i in range(n))

def make(n,layer=3,callback=None,*,literal=False,arithmetic_type=Arithmetic):
    a=arithmetic_type();k=a.ctx._param_cache
    k.raw_rotation_keys={key:{'a':object(),'b':object()}for key in KEYS}
    k.coef_maps={key:object()for key in KEYS}
    r=IdentityRefresh(a)if n>1 else None
    cls=QwenCausalSoftmax if literal else batched.QwenCausalSoftmaxBatched
    o=cls(a,QwenCausalSoftmaxContract(RAW,CERT,layer),CONFIG,positions=positions(n),refresher=r,progress=callback)
    assert set(k.raw_rotation_keys)==set(KEYS)and len(KEYS)==166
    return a,r,o

def scores(n,tile):
    # Poison all invalid lanes; valid data is heterogeneous across heads/rows/K.
    out=(27.+np.arange(65536)%29).reshape(16,4096)
    for row in range(min(16,n-16*tile)):
        k=16*tile+row+1;h=np.arange(16)[:,None];j=np.arange(k)[None,:]
        value=.73*np.sin((h+1)*(j+1)*.413)+.17*np.cos((j+2)*.271)+.031*row+.019*h
        out[row,:512].reshape(16,32)[:,:k]=value
    return out.ravel()

def inverse(x,rule):
    value=x*rule['rounded_first_weight']+rule['rounded_first_bias']
    for _ in range(1,rule['iterations']):value=value*(2.-x*value)
    return value

def dense(data,k,layer):
    alpha=float(1/k);site=M['softmax'][str(layer)]
    centered=data-np.sum(data,axis=-1,keepdims=True)*alpha
    assert centered.min()>=site['centered_domain'][0]and centered.max()<=site['centered_domain'][1]
    t=centered*2.**(1-site['squarings'])
    base=np.polynomial.polynomial.polyval(t,M['exp']['evaluation_coefficients'])
    denominator=np.sum(base,axis=-1,keepdims=True)*alpha
    value=base*inverse(denominator,M['initial_reciprocal'])*alpha
    for _ in range(site['squarings']):
        square=value*value
        value=square*inverse(np.sum(square,axis=-1,keepdims=True),M['reciprocal_by_keys'][str(k)])
    return value

def expected(data,n,tile,layer):
    result=np.zeros((16,4096));source=data.reshape(16,4096)
    for row in range(min(16,n-tile*16)):
        k=tile*16+row+1
        if k>1:result[row,:512].reshape(16,32)[:,:k]=dense(source[row,:512].reshape(16,32)[:,:k],k,layer)
    return result.ravel()

def independent_cohorts(n,tile):
    groups={}
    for k in range(max(2,tile*16+1),min(n,(tile+1)*16)+1):
        rule=M['reciprocal_by_keys'][str(k)]
        groups.setdefault((rule['iterations'],float(C['softmax_K'][k-1]['existing_inverse_refresh_bound'])),[]).append(k)
    return tuple((number,bound,tuple(keys))for(number,bound),keys in groups.items())

def counts(n,tile,layer):
    c=Counter();s=M['softmax'][str(layer)]['squarings'];groups=independent_cohorts(n,tile)
    for number,bound,keys in groups:
        c.update(weighted_sums=14+s*(number+2),ciphertext_products=18+2*s*number,
            rotations=20+10*s,ordinary_additions=21+10*s,refreshes=3+2*s)
    c['ordinary_additions']+=max(0,len(groups)-1)
    return dict(c)

def assert_result(result,data,n,tile,layer):
    np.testing.assert_allclose(result.output.polynomial,expected(data,n,tile,layer),rtol=0,atol=3e-12)
    assert result.query_positions==positions(n)[16*tile:16*(tile+1)]
    assert result.key_positions==positions(n)and result.direct_value_queries==((0,)if tile==0 else())
    assert result.output.level==24 and result.output._ckks_nsd==1 and result.output.moduli==Q[:50]
    assert result.counts==counts(n,tile,layer)
    assert np.isfinite(result.output.polynomial).all()
    valid=expected(np.zeros(65536),n,tile,layer)!=0
    assert np.count_nonzero(result.output.polynomial[~valid])==0

@pytest.mark.parametrize('layer',[0,3])
@pytest.mark.parametrize('n',range(1,33))
def test_all_lengths_full_slots_both_squaring_families(n,layer):
    a,r,o=make(n,layer)
    if n==1:
        result=o.evaluate_tile(None,query_tile=0)
        assert result.output is None and result.direct_value_queries==(0,)and result.counts=={}and r is None
        return
    for tile in range((n+15)//16):
        data=scores(n,tile);source=Value(data,21,math.nextafter(DELTA,math.inf))
        before=source.polynomial.copy();calls=len(r.calls);operations=Counter(a.calls)
        result=o.evaluate_tile(source,query_tile=tile);assert_result(result,data,n,tile,layer)
        np.testing.assert_array_equal(source.polynomial,before)
        assert len(r.calls)-calls==result.counts['refreshes']
        got=Counter(a.calls);got.subtract(operations)
        assert dict(got)==dict(rotate=result.counts['rotations'],weighted=result.counts['weighted_sums'],multiply=result.counts['ciphertext_products'],add=result.counts['ordinary_additions'])
        assert all(level>=2 for level,bound in r.calls[calls:])
        for cohort in o.cohorts[tile]:
            found=[x for x in result.plan['refreshes']if x['label'].startswith('C'+str(cohort.index)+'.')]
            assert len(found)==3+2*M['softmax'][str(layer)]['squarings']
            assert sum(x['public_bound']==4. for x in found)==1
            inverses=[x for x in found if 'before7'in x['label']]
            assert len(inverses)==M['softmax'][str(layer)]['squarings']
            assert all(x['public_bound']==cohort.refresh_bound and x['input_level']==6 for x in inverses)

@pytest.mark.parametrize('layer',[0,3])
@pytest.mark.parametrize('tile',[0,1])
def test_literal_predecessor_equivalence_same_fullslot_source(layer,tile):
    data=scores(32,tile)
    _,_,old=make(32,layer,literal=True);_,_,new=make(32,layer)
    before=old.evaluate_tile(Value(data,21),query_tile=tile)
    after=new.evaluate_tile(Value(data,21),query_tile=tile)
    np.testing.assert_allclose(after.output.polynomial,before.output.polynomial,rtol=0,atol=3e-12)
    assert after.counts['refreshes']<before.counts['refreshes']
    assert after.query_positions==before.query_positions and after.key_positions==before.key_positions

@pytest.mark.parametrize('layer',range(28))
def test_all_layers_original_rule_and_exact_key_inventory(layer):
    a,_,o=make(32,layer)
    for tile,cohorts in enumerate(o.cohorts):
        assert [(c.iterations,c.refresh_bound,tuple(q.k for q in c.queries))for c in cohorts]==list(independent_cohorts(32,tile))
        for c in cohorts:
            for q in c.queries:
                rule=M['reciprocal_by_keys'][str(q.k)]
                assert (q.alpha,q.domain,q.first_weight,q.first_bias,q.iterations)==(float(1/q.k),tuple(rule['domain']),rule['rounded_first_weight'],rule['rounded_first_bias'],rule['iterations'])
                assert q.position==positions(32)[q.k-1]and q.row==(q.k-1)%16
    assert set(o.required_rotation_indices())<=set(KEYS)
    assert set(o.refresh_binding.refresher.plan['rotations'])<=set(KEYS)

def test_exact_public_operation_projection_not_measured_speedup():
    literal=cohort=rotations=products=0
    for layer in range(28):
        s=M['softmax'][str(layer)]['squarings'];literal+=31*(3+2*s)
        for tile in (0,1):
            row=counts(32,tile,layer);cohort+=row['refreshes'];rotations+=row['rotations'];products+=row['ciphertext_products']
    assert(literal,cohort,rotations,products)==(11656,3760,20200,32196)

@pytest.mark.parametrize('tile',[0,1])
def test_mixed_affine_arrays_and_all_dummy_lanes(tile):
    _,_,o=make(32)
    for cohort in o.cohorts[tile]:
        mask,anchors,dummy,alpha,aa,mapped,w,b=batched._vectors(cohort,o.contract.squarings)
        active=np.zeros((16,4096),dtype=bool)
        for q in cohort.queries:
            active[q.row,:512]=True
            assert np.all(w.reshape(16,4096)[q.row,:512]==q.first_weight)
            assert np.all(b.reshape(16,4096)[q.row,:512]==q.first_bias)
            for h in range(16):
                s=q.row*4096+h*32
                assert np.all(alpha[s:s+q.k]==float(1/q.k))and alpha[s+q.k:s+32].sum()==0
                assert aa[s]==float(1/q.k)and dummy[s]==0
        first=cohort.queries[0]
        assert np.all(w[~active.ravel()]==first.first_weight)and np.all(b[~active.ravel()]==first.first_bias)
        # Reproduce every frozen inactive Newton update on denominator1.
        for q in cohort.queries:
            y=q.first_weight+q.first_bias
            for update in range(2,q.iterations+1):
                if update==7:assert abs(y)<=q.refresh_bound
                y=y*(2.-y)
            assert math.isfinite(y)and abs(y-1.)<1e-12
        assert all(not a.flags.writeable and type(a.base)is bytes for a in(mask,anchors,dummy,alpha,aa,mapped,w,b))

@pytest.mark.parametrize('layer,tile',[(0,0),(3,1)])
def test_bounded_inactive_refresh_perturbations_are_masked(layer,tile):
    a,r,o=make(32,layer);per=3+2*o.contract.squarings
    def perturb(source,out):
        cohort=o.cohorts[tile][(len(r.calls)-1)//per];valid=np.zeros((16,4096),bool)
        for q in cohort.queries:valid[q.row,:512].reshape(16,32)[:,:q.k]=True
        out.polynomial[~valid.ravel()]=.03125
    r.inject=perturb;data=scores(32,tile)
    assert_result(o.evaluate_tile(Value(data,21),query_tile=tile),data,32,tile,layer)

@pytest.mark.parametrize('field,value',[('final_dummy','bad'),('iterations',True),('iterations',9.),('refresh_bound',16),('refresh_bound',32.),('queries',())])
def test_mutated_cohort_fails_before_arithmetic(field,value):
    a,_,o=make(16)
    if field=='final_dummy':o.cohorts=tuple(list(o.cohorts))
    else:object.__setattr__(o.cohorts[0][3],field,value)
    with pytest.raises(ValueError):o.evaluate_tile(Value(scores(16,0),21),query_tile=0)
    assert not a.calls and not batched._QUARANTINE

@pytest.mark.parametrize('field,value',[('k',False),('position',0.),('alpha',.5),('first_weight',0.),('domain',(0.,2.))])
def test_mutated_query_rule_fails_before_arithmetic(field,value):
    a,_,o=make(16);object.__setattr__(o.cohorts[0][3].queries[0],field,value)
    with pytest.raises(ValueError):o.evaluate_tile(Value(scores(16,0),21),query_tile=0)
    assert not a.calls

@pytest.mark.parametrize('error_type',[ValueError,KeyboardInterrupt,SystemExit])
def test_callback_primary_and_replaced_payload_remain_owned(error_type):
    error=error_type('synthetic callback');refs=[]
    def callback(label,value,metadata):
        if label.endswith('.base_exp'):
            refs.append(weakref.ref(value.polynomial));value.polynomial=np.zeros(65536)
            raise error
    _,_,o=make(8,callback=callback)
    with pytest.raises(error_type)as caught:o.evaluate_tile(Value(scores(8,0),21),query_tile=0)
    assert caught.value is error and o.last_run is None and not o._active
    gc.collect();assert refs[0]()is not None and batched._QUARANTINE[0][0]is o
    with pytest.raises(RuntimeError):o.evaluate_tile(Value(scores(8,0),21),query_tile=0)

@pytest.mark.parametrize('mutation',['pending','arrays','shape','descriptor','previous'])
def test_callback_owned_state_substitution_is_rejected_and_retained(mutation):
    holder={};refs=[]
    def callback(label,value,metadata):
        o=holder['owner']
        if mutation=='previous':
            if label.endswith('final_branch_mask')and label.startswith('C0.'):
                holder['previous']=value;refs.append(weakref.ref(value.polynomial))
            elif label.startswith('C1.')and label.endswith('source_branch_mask'):
                holder['previous'].polynomial=np.zeros(65536)
            return
        if label.endswith('.base_exp'):
            refs.append(weakref.ref(value.polynomial))
            if mutation=='pending':o._pending_return=()
            elif mutation=='arrays':o._active_arrays=()
            elif mutation=='shape':o._active_arrays[0].shape=(16,4096)
            else:object.__setattr__(o.cohorts[0][0],'refresh_bound',16.)
    _,_,o=make(8,callback=callback);holder['owner']=o
    with pytest.raises(ValueError):o.evaluate_tile(Value(scores(8,0),21),query_tile=0)
    gc.collect();assert refs and all(r()is not None for r in refs)and batched._QUARANTINE

@pytest.mark.parametrize('n,tile',[(1,0),(8,0),(19,1),(32,1)])
def test_unchanged_pv_layout_and_real_k1_value_bypass(n,tile):
    a,_,o=make(n);data=None if n==1 else scores(n,tile)
    result=o.evaluate_tile(None if data is None else Value(data,21),query_tile=tile)
    pv=QwenAttentionRoutingWithBypass(a,RAW,CONFIG,positions=positions(n))
    native=np.sin(np.arange(n*16*128,dtype=np.float64).reshape(n,16,128)*.013)
    values=[]
    for t in range((n+15)//16):
        array=np.full((16,4096),31.);length=min(16,n-t*16)
        array[:length,:2048]=native[16*t:16*t+length].reshape(length,2048)
        values.append(Value(array.ravel(),24))
    got=pv.attend_softmax(result,tuple(values),query_tile=tile);want=np.zeros((16,4096))
    for row in range(min(16,n-16*tile)):
        k=16*tile+row+1
        if k==1:want[row,:2048]=native[0].ravel()
        else:
            p=dense(data.reshape(16,4096)[row,:512].reshape(16,32)[:,:k],k,3)
            want[row,:2048]=np.einsum('hk,khd->hd',p,native[:k]).ravel()
    np.testing.assert_allclose(got.output.polynomial,want.ravel(),rtol=0,atol=5e-12)
    assert got.level==(23 if n==1 else 21)

def test_parent199_scoped_overlays_and_no_backend_model_imports():
    parent=json.loads((P/'cohort_parent199_sources.json').read_bytes())
    assert len(parent)==199
    overlays={'checkout/cadathlon/common/encrypted_qwen_attention_join.py','plan_full.py','test_full_plan.py'}
    assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==pin for n,pin in parent.items()if n not in overlays)
    assert not any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors'))

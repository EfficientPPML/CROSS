"""Independent full-slot causal probabilities/PV; ideal identity refresh only.

No model, private prompt, keys, backend or encrypted/noise claim. The oracle is
small dense head/key arrays with Horner exp and original rounded Newton rules.
"""
from pathlib import Path
from collections import Counter
import hashlib,json,math,sys
import numpy as np
import pytest
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));sys.path.insert(0,str(P/'checkout/cadathlon/common'))
from ideal_primitives import Arithmetic,Value,DELTA,Q
from ideal_refresh import IdentityRefresh
from encrypted_qwen_causal_softmax import QwenCausalSoftmax,QwenCausalSoftmaxContract
from encrypted_qwen_attention_bypass import QwenAttentionRoutingWithBypass
from encrypted_qwen_head_routing import QwenHeadLayout
from plan_public import branch_plan,inputs
RAW=(P/'inputs/candidate_manifest.json').read_bytes();CERT=(P/'inputs/certificates.json').read_bytes();CONFIG=(P/'inputs/config.json').read_bytes()
M,C,_,PROFILE=inputs()

def original_inverse(d,rule):
    # Stored rounded affine is the actual frozen first polynomial. Subsequent
    # updates are evaluated directly, independently of the server's primitives.
    y=np.asarray(d,dtype=np.float64)*rule['rounded_first_weight']+rule['rounded_first_bias']
    for _ in range(1,rule['iterations']):y=y*(2.-d*y)
    return y

def dense_probability(scores,layer):
    k=scores.shape[-1];alpha=float(1/k)
    centered=scores-np.sum(scores,axis=-1,keepdims=True)*alpha
    lo,hi=M['softmax'][str(layer)]['centered_domain'];assert np.min(centered)>=lo and np.max(centered)<=hi
    t=centered*2.**(1-M['softmax'][str(layer)]['squarings'])
    # No second affine: these actual power coefficients already encode the
    # frozen exp input convention on the normalized argument.
    base=np.polynomial.polynomial.polyval(t,np.asarray(M['exp']['evaluation_coefficients']))
    denominator=np.sum(base*alpha,axis=-1,keepdims=True)
    p=base*original_inverse(denominator,M['initial_reciprocal'])*alpha
    for _ in range(M['softmax'][str(layer)]['squarings']):
        square=p*p;denominator=np.sum(square,axis=-1,keepdims=True)
        p=square*original_inverse(denominator,M['reciprocal_by_keys'][str(k)])
    return p

def positions(n):return tuple(3*i+(i//5) for i in range(n))

def scores_for(n,tile):
    # Deliberately nonzero all physical padding, future keys and unused rows.
    data=(27.+np.arange(65536)%29).reshape(16,4096)
    for row in range(min(16,n-16*tile)):
        k=16*tile+row+1
        h=np.arange(16)[:,None];key=np.arange(k)[None,:]
        values=.73*np.sin((h+1)*(key+1)*.413)+.17*np.cos((key+2)*.271)+.031*row+.019*h
        data[row,:512].reshape(16,32)[:,:k]=values
    return data.ravel()

def expected_probability(data,n,tile,layer):
    out=np.zeros((16,4096));view=np.asarray(data).reshape(16,4096)
    for row in range(min(16,n-16*tile)):
        k=16*tile+row+1
        if k==1:continue
        out[row,:512].reshape(16,32)[:,:k]=dense_probability(view[row,:512].reshape(16,32)[:,:k],layer)
    return out.ravel()

def expected_counts(n,tile,layer):
    total=Counter();branches=[];s=M['softmax'][str(layer)]['squarings']
    for k in range(max(2,16*tile+1),min(n,16*(tile+1))+1):
        updates=M['reciprocal_by_keys'][str(k)]['iterations'];branches.append(k)
        total.update(dict(weighted_sums=14+s*(updates+2),ciphertext_products=18+2*s*updates,
            rotations=20+10*s,ordinary_additions=21+10*s,refreshes=3+2*s))
    total['ordinary_additions']+=max(0,len(branches)-1)
    return dict(total)

def construct(n,layer,progress=None):
    a=Arithmetic();refresh=IdentityRefresh(a) if n>1 else None
    owner=QwenCausalSoftmax(a,QwenCausalSoftmaxContract(RAW,CERT,layer),CONFIG,positions=positions(n),refresher=refresh,progress=progress)
    return a,refresh,owner

def signature(value):
    return (id(value.polynomial),value.polynomial.tobytes(),value.level,value._ckks_scale,value._ckks_nsd,value.moduli,dict((k,v) for k,v in vars(value).items() if k.startswith('_public_')))

def assert_result(result,source,n,tile,layer):
    expected=expected_probability(source.polynomial,n,tile,layer)
    np.testing.assert_allclose(result.output.polynomial,expected,rtol=0,atol=3e-12)
    assert result.output.polynomial.shape==(65536,)
    assert result.query_positions==positions(n)[16*tile:16*(tile+1)] and result.key_positions==positions(n)
    assert result.direct_value_queries==((0,) if tile==0 else ())
    assert result.output.level==24 and result.output._ckks_nsd==1 and result.output.moduli==tuple(Q[:50])
    assert result.counts==expected_counts(n,tile,layer)
    # Physical zero checks independently cover all invalid regions.
    mask=np.zeros((16,4096),dtype=bool)
    for row in range(len(result.query_positions)):
        k=16*tile+row+1
        if k>1:mask[row,:512].reshape(16,32)[:,:k]=True
    assert np.count_nonzero(result.output.polynomial[~mask.ravel()])==0
    assert not any(k.startswith('_public_') for k in vars(result.output))

@pytest.mark.parametrize('layer',[0,3])
@pytest.mark.parametrize('tile',[0,1])
def test_allK2through32_two_squaring_families_distinct_heads_rows(layer,tile):
    a,r,o=construct(32,layer);source=Value(scores_for(32,tile),21,math.nextafter(DELTA,math.inf));source._public_bound=123.
    old=signature(source);result=o.evaluate_tile(source,query_tile=tile)
    assert_result(result,source,32,tile,layer);assert signature(source)==old
    assert len(r.calls)==result.counts['refreshes']
    assert dict(a.calls)==dict(rotate=result.counts['rotations'],weighted=result.counts['weighted_sums'],multiply=result.counts['ciphertext_products'],add=result.counts['ordinary_additions'])
    assert all(level>=2 for level,bound in r.calls)

@pytest.mark.parametrize('n,tile',[(2,0),(7,0),(16,0),(17,1),(19,1),(31,1)])
def test_partial_tiles_noncontiguous_positions_and_hostile_padding(n,tile):
    a,r,o=construct(n,3);source=Value(scores_for(n,tile),21);result=o.evaluate_tile(source,query_tile=tile)
    assert_result(result,source,n,tile,3)

@pytest.mark.parametrize('layer',range(28))
def test_every_layer_contract_uses_frozen_rules_without_selection(layer):
    c=QwenCausalSoftmaxContract(RAW,CERT,layer)
    assert c.squarings==M['softmax'][str(layer)]['squarings'] and c.centered_domain==tuple(M['softmax'][str(layer)]['centered_domain'])
    assert c.coefficients==tuple(M['exp']['evaluation_coefficients']) and c.initial_rule[-1]==6
    for k,rule in enumerate(c.key_rules,1):
        raw=M['reciprocal_by_keys'][str(k)]
        assert rule[:4]==(tuple(raw['domain']),raw['rounded_first_weight'],raw['rounded_first_bias'],raw['iterations'])

@pytest.mark.parametrize('n,tile,layer',[(2,0,0),(2,0,3),(17,1,0),(17,1,3)])
def test_single_branch_metadata_against_independent_public_schedule(n,tile,layer):
    a,r,o=construct(n,layer);source=Value(scores_for(n,tile),21,math.nextafter(DELTA,-math.inf))
    result=o.evaluate_tile(source,query_tile=tile)
    k=2 if n==2 else 17;expected=branch_plan(M,C,PROFILE,layer,k,input_scale=source._ckks_scale,refresh_mode='alternating_ulp')
    stages={s['label']:s for s in result.plan['stages']}
    for row in expected['stages']:
        if row['kind'] in ('input','add_or_subtract'):continue
        got=stages['K'+str(k)+'.'+row['label']]
        assert (got['kind'],got['level'],got['scale'])==(row['kind'],row['level'],row['scale'])
    assert result.plan['output_scale']==expected['output_scale'] and result.counts==expected['counts']


def test_nonzero_invalid_refresh_lanes_do_not_become_causal_mass():
    a,r,o=construct(7,0);current=[]
    # All nonlinear branches have one active public row. Add bounded values
    # only outside it at every ideal refresh; valid lanes stay exact.
    def inject(source,out):
        view=out.polynomial.reshape(16,4096)
        active=np.zeros_like(view,dtype=bool)
        # Sources before a normalized-t refresh have their live row values;
        # this test identifies the row from the already-known public branch
        # rather than inspecting numerical data to choose an HE path.
        branch=(len(r.calls)-1)//15+2;row=branch-1
        active[row,:512].reshape(16,32)[:,:branch]=True
        view[~active]=.03125
    r.inject=inject
    source=Value(scores_for(7,0),21);result=o.evaluate_tile(source,query_tile=0)
    assert_result(result,source,7,0,0)


def make_pv(a,n):
    prototype=QwenAttentionRoutingWithBypass.__new__(QwenAttentionRoutingWithBypass);prototype.layout=QwenHeadLayout(positions(n))
    for index in prototype.required_rotation_indices():
        a.ctx._param_cache.raw_rotation_keys.setdefault(index,{'a':object(),'b':object()});a.ctx._param_cache.coef_maps.setdefault(index,object())
    return QwenAttentionRoutingWithBypass(a,RAW,CONFIG,positions=positions(n))

def expanded_values(n):
    native=np.sin(np.arange(n*16*128,dtype=np.float64).reshape(n,16,128)*.013)+np.arange(n)[:,None,None]/17
    tiles=[]
    for tile in range((n+15)//16):
        data=np.full((16,4096),37.25);length=min(16,n-16*tile);data[:length,:2048]=native[16*tile:16*tile+length].reshape(length,2048)
        tiles.append(Value(data.ravel(),24,math.nextafter(DELTA,-math.inf)))
    return native,tuple(tiles)

@pytest.mark.parametrize('n,tile',[(1,0),(7,0),(19,1)])
def test_actual_softmax_result_to_realV_bypass_or_densePV(n,tile):
    a,r,o=construct(n,3);source=None if n==1 else Value(scores_for(n,tile),21)
    probability=o.evaluate_tile(source,query_tile=tile);pv=make_pv(a,n);native,values=expanded_values(n);old=tuple(signature(v) for v in values)
    got=pv.attend_softmax(probability,values,query_tile=tile);expected=np.zeros((16,4096))
    for row in range(len(probability.query_positions)):
        k=16*tile+row+1
        if k==1:expected[row,:2048]=native[0].ravel()
        else:
            p=dense_probability(source.polynomial.reshape(16,4096)[row,:512].reshape(16,32)[:,:k],3)
            expected[row,:2048]=np.einsum('hk,khd->hd',p,native[:k]).ravel()
    np.testing.assert_allclose(got.output.polynomial,expected.ravel(),rtol=0,atol=5e-12)
    assert tuple(signature(v) for v in values)==old and got.level==(23 if n==1 else 21)
    if n==1:
        assert probability.output is None and probability.counts=={} and probability.direct_value_queries==(0,)
        assert got.counts==dict(rotations=0,weighted_sums=1,ciphertext_products=0,additions=0)
    assert np.count_nonzero(got.output.polynomial.reshape(16,4096)[:,2048:])==0


def test_fullslot_oracles_do_not_load_model_or_backend():
    assert not any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors'))


@pytest.mark.parametrize('layer,endpoint',[(0,26.),(3,16.),(5,16.),(12,16.),(15,-16.)])
def test_declared_centered_endpoint_and_zero_margin_exp_arguments(layer,endpoint):
    a,r,o=construct(3,layer);data=scores_for(3,0).reshape(16,4096)
    head=data[2,:512].reshape(16,32)
    for h in range(16):head[h,:3]=np.roll(np.array([endpoint,-endpoint/2,-endpoint/2]),h%3)
    source=Value(data.ravel(),21);result=o.evaluate_tile(source,query_tile=0)
    assert_result(result,source,3,0,layer)
    mapped=head[:,:3]*2.**(1-M['softmax'][str(layer)]['squarings'])
    if layer!=0:assert np.max(np.abs(mapped))==1.
    # This exact-function comparison is an oracle diagnostic only. The owner
    # used its unchanged degree8 polynomial/original Newton graph throughout.
    exact=np.exp(head[:,:3]-np.max(head[:,:3],axis=-1,keepdims=True));exact/=np.sum(exact,axis=-1,keepdims=True)
    actual=result.output.polynomial.reshape(16,4096)[2,:512].reshape(16,32)[:,:3]
    np.testing.assert_allclose(actual,exact,rtol=0,atol=1e-6)

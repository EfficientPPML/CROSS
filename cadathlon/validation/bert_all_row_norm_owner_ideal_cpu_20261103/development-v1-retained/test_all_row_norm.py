"""All24 selected norm circuits on ideal slots; no encrypted-model claim."""
from pathlib import Path
from collections import Counter
import copy,hashlib,json,math,os
import numpy as np
import pytest
from test_initial_ideal import ideal_primitives,Q,DELTA
from ideal_primitives import Arithmetic,Value
from encrypted_bert_all_row_norm import BertRowLayerNorm,BertRowNormContract,AFFINE_SHA256,row_norm_scale_plan
from encrypted_bert_v2_row_norm import V2RowLayerNorm

HERE=Path(__file__).resolve().parent;RAW=(HERE/'candidate_manifest.json').read_bytes()
SITES=tuple(AFFINE_SHA256)
MODEL=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c/model.safetensors')
MODEL_SHA='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
def file_sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@pytest.fixture(scope='module')
def affines():
    assert file_sha(MODEL)==MODEL_SHA
    pins=json.loads((HERE/'affine_pins.json').read_text());assert pins['checkpoint_sha256']==MODEL_SHA
    result={}
    with MODEL.open('rb') as f:
        length=int.from_bytes(f.read(8),'little');header=json.loads(f.read(length));base=8+length
        for site,pair in pins['sites'].items():
            vectors=[]
            for part in ('weight','bias'):
                desc=header[pair[part]['tensor']];assert desc['dtype']=='F32' and desc['shape']==[768]
                start,end=desc['data_offsets'];f.seek(base+start);raw=f.read(end-start)
                assert hashlib.sha256(raw).hexdigest()==pair[part]['f32le_sha256']
                vectors.append(np.frombuffer(raw,dtype='<f4').astype(np.float64))
            result[site]=tuple(vectors)
    yield result
    assert file_sha(MODEL)==MODEL_SHA

def setup(affines,site,*,progress=None):
    ops=Arithmetic();cache=ops.ctx._param_cache
    cache.per_level_rotation_keys=False
    cache.raw_rotation_keys={key:{'a':object(),'b':object()} for key in BertRowLayerNorm.required_rotation_indices()}
    cache.coef_maps={key:object() for key in cache.raw_rotation_keys}
    owner=BertRowLayerNorm(ops,RAW,*affines[site],site=site,progress=progress)
    return owner,ops

def fixture(site,valid):
    contract=BertRowNormContract(RAW,site);lo,hi=contract.domain
    pattern=np.cos(np.arange(768)*2*np.pi/768)
    pattern-=pattern.mean();pattern/=np.sqrt(np.mean(pattern*pattern))
    fractions=np.linspace(.03,.97,valid)
    variance=lo+(hi-lo)*fractions
    active=pattern[None,:]*np.sqrt(variance[:,None]-1e-12)+np.arange(valid)[:,None]*.25-2.
    data=np.full((16,4096),10000.);data[::2]*=-1;data[:valid,:768]=active
    return data,active

def independent(active,contract,gamma,beta):
    mean=active.mean(axis=-1,keepdims=True);centered=active-mean
    variance=np.mean(centered*centered,axis=-1,keepdims=True)+1e-12
    lo,hi=contract.domain
    assert np.all((variance>=lo)&(variance<=hi))
    inverse=np.polynomial.chebyshev.chebval((2/(hi-lo))*variance-(hi+lo)/(hi-lo),contract.coefficients)
    return centered*inverse*gamma+beta,mean,variance,inverse

@pytest.mark.parametrize('site',SITES)
@pytest.mark.parametrize('valid',(1,16))
@pytest.mark.parametrize('scale',(math.nextafter(DELTA,-math.inf),DELTA,math.nextafter(DELTA,math.inf)))
@pytest.mark.parametrize('level_choice',('minimum','normal'))
def test_every_site_full_slots_selected_method_and_metadata(affines,site,valid,scale,level_choice):
    owner,ops=setup(affines,site);depth=owner.inverse_evaluator.polynomial_depth(owner.approximation)+5
    level=depth+2 if level_choice=='minimum' else (20 if site.endswith('attention') else 24)
    data,active=fixture(site,valid);source=Value(data.ravel(),level,scale);saved=source.data.copy()
    positions=tuple(range(0,2*valid,2));result=owner.normalize_tile(source,valid_rows=valid,original_positions=positions,capture_checkpoints=True)
    reference,mean,variance,inverse=independent(active,owner.contract,*affines[site])
    expected=np.zeros((16,4096));expected[:valid,:768]=reference
    error=float(np.max(np.abs(result.output.data-expected.ravel())))
    assert np.isfinite(result.output.data).all() and error<1e-8
    for key,ref in [('mean',mean),('variance',variance),('inverse',inverse)]:
        actual=result.checkpoints[key].data.reshape(16,4096)[:valid,:768]
        np.testing.assert_allclose(actual,np.broadcast_to(ref,(valid,768)),rtol=0,atol=1e-8)
    assert result.output.level==level-depth>=2
    assert result.output._ckks_scale==result.plan['scales']['output'] and result.original_positions==positions
    assert ops.strategy=='power' and owner.inverse_evaluator.strategy==('power' if owner.contract.method=='power' else 'chebyshev')
    counts=Counter(ops.calls);d=owner.contract.degree
    assert counts['rotate']==42 and counts['multiply']==d+1
    assert counts['weighted_sum']==(5 if owner.contract.method=='power' else d+4)
    assert not owner._active
    np.testing.assert_array_equal(source.data,saved)
    for v in (result.output,*result.checkpoints.values()):assert not any(n.startswith('_public_') for n in vars(v))
    destination=os.environ.get('NORM_CASE_REPORT_DIR')
    if destination:
        p=Path(destination);p.mkdir(exist_ok=True)
        name=f'{site}-{valid}-{level_choice}-{scale.hex()}.json'
        record=dict(site=site,method=owner.contract.method,degree=d,valid_rows=valid,level_choice=level_choice,
            source_level=level,source_scale=scale,slots_checked=65536,gate=1e-8,max_abs_vs_independent_chebval=error,
            variance_observed=[float(variance.min()),float(variance.max())],actual_counts=dict(counts),plan=result.plan)
        (p/name).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')

@pytest.mark.parametrize('site',('0.attention','0.mlp'))
def test_layer0_arithmetic_identical_to_frozen_specialized_owner(affines,site):
    current,ops=setup(affines,site);oldops=Arithmetic();oldops.ctx._param_cache.raw_rotation_keys=dict(ops.ctx._param_cache.raw_rotation_keys)
    oldops.ctx._param_cache.coef_maps=dict(ops.ctx._param_cache.coef_maps);oldops.ctx._param_cache.per_level_rotation_keys=False
    old=V2RowLayerNorm(oldops,RAW,*affines[site],site=site)
    data,_=fixture(site,16);level=20 if site.endswith('attention') else 24
    a=old.normalize_tile(Value(data.ravel(),level),valid_rows=16,original_positions=tuple(range(16)))
    b=current.normalize_tile(Value(data.ravel(),level),valid_rows=16,original_positions=tuple(range(16)))
    np.testing.assert_array_equal(a.output.data,b.output.data)
    assert a.plan['levels']==b.plan['levels'] and a.plan['scales']==b.plan['scales'] and oldops.calls==ops.calls

@pytest.mark.parametrize('site',SITES)
def test_every_supported_level_and_conditioning_contract(site):
    c=BertRowNormContract(RAW,site);depth=(2 if c.method=='power' else 2)*(c.degree-1).bit_length()
    depth=(c.degree-1).bit_length()*(1 if c.method=='power' else 2)+7
    for L in range(depth+2,50):
        p=row_norm_scale_plan(Q[:2*(L+1)],input_level=L,input_scale=math.nextafter(DELTA,math.inf),canonical_scale=DELTA,contract=c)
        assert p['consumed_levels']==depth and p['levels']['output']==L-depth>=2
    with pytest.raises(ValueError):row_norm_scale_plan(Q[:2*(depth+2)],input_level=depth+1,input_scale=DELTA,canonical_scale=DELTA,contract=c)

@pytest.mark.parametrize('mutation',('view','method','view_arithmetic','method_type','site','power_strategy','bound_aware','layout'))
def test_selected_view_and_contract_changes_fail_before_work(affines,mutation):
    owner,ops=setup(affines,'9.mlp');source=Value(np.zeros(65536),24)
    if mutation=='view':owner.inverse_evaluator=copy.copy(owner.inverse_evaluator)
    elif mutation=='method':object.__setattr__(owner.inverse_evaluator,'method','power')
    elif mutation=='view_arithmetic':object.__setattr__(owner.inverse_evaluator,'arithmetic',Arithmetic())
    elif mutation=='method_type':object.__setattr__(owner.inverse_evaluator,'method',None)
    elif mutation=='site':object.__setattr__(owner.contract,'site','10.mlp')
    elif mutation=='power_strategy':ops.strategy='chebyshev'
    elif mutation=='bound_aware':ops.declare_bound=lambda *args:None
    else:source.degree_layout=(512,256)
    with pytest.raises((ValueError,TypeError)):owner.normalize_tile(source,valid_rows=1,original_positions=(0,))
    assert not ops.calls and not owner._active

@pytest.mark.parametrize('event',('mean','inverse','output'))
@pytest.mark.parametrize('mutation',('source','result','captured','selected_method','context'))
def test_balanced_callback_guards_retained_and_captured_values(affines,event,mutation):
    source=Value(fixture('9.mlp',1)[0].ravel(),24);saved=[];calls=[]
    def progress(label,result,counts):
        if label.endswith(' mean'):saved.append(result)
        if not label.endswith(' '+event):return
        calls.append(len(ops.calls))
        if mutation=='source':source._ckks_nsd=True
        elif mutation=='result':result._ckks_nsd=True
        elif mutation=='captured':saved[0]._ckks_nsd=True
        elif mutation=='selected_method':object.__setattr__(owner.inverse_evaluator,'method','power')
        else:ops.ctx.secret_key=object()
    owner,ops=setup(affines,'9.mlp',progress=progress)
    with pytest.raises(ValueError):owner.normalize_tile(source,valid_rows=1,original_positions=(0,),capture_checkpoints=True)
    assert calls==[len(ops.calls)] and not owner._active and owner.last_run is None

def test_cancellation_reentry_and_annotation_removal(affines):
    source=Value(fixture('9.mlp',1)[0].ravel(),24);seen=[]
    def annotate(label,result,counts):
        seen.append(result)
        for v in seen:v._public_abs_bound=0.
        with pytest.raises(ValueError,match='reentered'):owner.normalize_tile(source,valid_rows=1,original_positions=(0,))
    owner,ops=setup(affines,'9.mlp',progress=annotate)
    result=owner.normalize_tile(source,valid_rows=1,original_positions=(0,),capture_checkpoints=True)
    for v in (result.output,*result.checkpoints.values()):assert not any(n.startswith('_public_') for n in vars(v))
    error=KeyboardInterrupt('retain primary')
    def cancel(*args):raise error
    owner,ops=setup(affines,'9.mlp',progress=cancel)
    with pytest.raises(KeyboardInterrupt) as caught:owner.normalize_tile(source,valid_rows=1,original_positions=(0,))
    assert caught.value is error and not owner._active

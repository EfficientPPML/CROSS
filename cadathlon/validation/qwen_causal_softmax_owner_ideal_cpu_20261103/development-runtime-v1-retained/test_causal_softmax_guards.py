"""Mutation and borrowed-owner boundaries; no model, encryption or refresh."""
from pathlib import Path
import json,math
import numpy as np
import pytest
from ideal_primitives import Arithmetic,Value,DELTA
from ideal_refresh import IdentityRefresh
from encrypted_qwen_causal_softmax import QwenCausalSoftmax,QwenCausalSoftmaxContract,QwenProbabilityTile
from encrypted_qwen_attention_bypass import QwenAttentionRoutingWithBypass
P=Path(__file__).resolve().parent
RAW=(P/'inputs/candidate_manifest.json').read_bytes();CERT=(P/'inputs/certificates.json').read_bytes();CONFIG=(P/'inputs/config.json').read_bytes()

def setup(callback=None,positions=(0,2)):
    a=Arithmetic();r=IdentityRefresh(a) if len(positions)>1 else None
    c=QwenCausalSoftmaxContract(RAW,CERT,3);o=QwenCausalSoftmax(a,c,CONFIG,positions=positions,refresher=r,progress=callback)
    x=np.full((16,4096),111.)
    for row in range(len(positions)):
        for h in range(16):x[row,32*h:32*h+row+1]=np.linspace(-.4,.6,row+1)
    return o,a,r,Value(x.ravel(),21)

@pytest.mark.parametrize('field,value',[('squarings',True),('squarings',5.),('coefficients',[1.]*9),('centered_domain',(-13,16)),('layer',True),('initial_rule',((.5,2.),-.64,1.6,6.))])
def test_contract_replacement_before_work(field,value):
    o,a,r,x=setup();object.__setattr__(o.contract,field,value)
    with pytest.raises(ValueError):o.evaluate_tile(x,query_tile=0)
    assert not a.calls and not r.calls

@pytest.mark.parametrize('what',['arithmetic','context','layout','contract','refresh_binding','refresher','refresh_method','helper_method','arithmetic_method'])
def test_constructor_bound_identities_before_work(what):
    o,a,r,x=setup()
    if what=='arithmetic':o.arithmetic=Arithmetic()
    elif what=='context':o.ctx=Arithmetic().ctx
    elif what=='layout':o.layout=type(o.layout)(o.layout.positions)
    elif what=='contract':o.contract=QwenCausalSoftmaxContract(RAW,CERT,3)
    elif what=='refresh_binding':o.refresh_binding=object()
    elif what=='refresher':o.refresher=IdentityRefresh(a)
    elif what=='refresh_method':r.refresh=lambda *args,**kw:None
    elif what=='helper_method':r.compiled_primitives[3].multiply_encoded=lambda *args:None
    else:a.multiply=lambda *args:None
    with pytest.raises((ValueError,AttributeError)):o.evaluate_tile(x,query_tile=0)
    assert not a.calls and not r.calls

@pytest.mark.parametrize('phase',['K2.source_branch_mask','K2.base_exp','K2.initial_inverse.first','K2.round0.inverse.first','K2.final_branch_mask','output'])
@pytest.mark.parametrize('target',['source_payload','current_nsd','refresh_accessor'])
def test_callbacks_abort_before_next_primitive(phase,target):
    stopped=[];box={}
    def callback(label,value,meta):
        if label!=phase:return
        o,a,r,x=box['state'];stopped.append(dict(a.calls))
        if target=='source_payload':x.polynomial=np.zeros(65536)
        elif target=='current_nsd':value._ckks_nsd=True
        else:r.refresh=lambda *args,**kw:None
    box['state']=setup(callback);o,a,r,x=box['state']
    with pytest.raises(ValueError):o.evaluate_tile(x,query_tile=0)
    assert stopped and dict(a.calls)==stopped[0] and o.last_run is None and not o._active

@pytest.mark.parametrize('earlier,later',[('K2.power2','K2.base_exp'),('K2.base_exp','K2.initial_inverse.first'),('K2.initial_mean_anchor_dummy1','K2.initial_inverse.3.update'),('K2.round0.square','K2.round0.inverse.6.update')])
def test_retained_intermediates_protected(earlier,later):
    seen={};stopped=[];box={}
    def callback(label,value,meta):
        seen[label]=value
        if label==later:
            stopped.append(dict(box['a'].calls));seen[earlier]._ckks_nsd=1.
    o,a,r,x=setup(callback);box['a']=a
    with pytest.raises(ValueError):o.evaluate_tile(x,query_tile=0,capture_checkpoints=True)
    assert stopped and dict(a.calls)==stopped[0] and o.last_run is None

def test_all_captured_annotations_stripped_after_late_callback():
    seen=[]
    def callback(label,value,meta):
        seen.append(value)
        for prior in seen:prior._public_abs_bound=0.
    o,a,r,x=setup(callback);out=o.evaluate_tile(x,query_tile=0,capture_checkpoints=True)
    assert all(not any(n.startswith('_public_') for n in vars(v)) for v in (*out.checkpoints.values(),out.output))

def test_primary_cancellation_and_reentry():
    box={}
    def callback(*unused):
        o,a,r,x=box['state']
        with pytest.raises(ValueError,match='reentered'):o.evaluate_tile(x,query_tile=0)
        raise KeyboardInterrupt('cancel')
    box['state']=setup(callback);o,a,r,x=box['state']
    with pytest.raises(KeyboardInterrupt,match='cancel'):o.evaluate_tile(x,query_tile=0)
    assert not o._active and o.last_run is None

@pytest.mark.parametrize('mutation',['nsd','source','engine','output_layout'])
def test_refresh_callbacks_retained_sources_and_context(mutation):
    o,a,r,x=setup();stopped=[]
    def inject(source,result):
        stopped.append(dict(a.calls))
        if mutation=='nsd':source._ckks_nsd=True
        elif mutation=='source':x.polynomial=np.zeros(65536)
        elif mutation=='engine':r.engine._gap=True
        else:result.degree_layout=('other',)
    r.inject=inject
    with pytest.raises(ValueError):o.evaluate_tile(x,query_tile=0)
    assert stopped and dict(a.calls)==stopped[0] and o.last_run is None

def pv_setup(a,positions,callback=None):
    # The public inventory is supplied before construction; no keygen happens.
    from encrypted_qwen_head_routing import QwenHeadLayout
    view=object.__new__(QwenAttentionRoutingWithBypass);view.layout=QwenHeadLayout(positions)
    for key in view.required_rotation_indices():
        a.ctx._param_cache.raw_rotation_keys.setdefault(key,dict(a=object(),b=object()));a.ctx._param_cache.coef_maps.setdefault(key,object())
    return QwenAttentionRoutingWithBypass(a,RAW,CONFIG,positions=positions,progress=callback)

def test_K1_uses_actual_V_with_real_mask_no_softmax():
    o,a,r,x=setup(positions=(7,));result=o.evaluate_tile(None,query_tile=0);assert not a.calls
    pv=pv_setup(a,(7,));data=np.arange(65536,dtype=float);v=Value(data,24,math.nextafter(DELTA,math.inf))
    out=pv.attend_softmax(result,(v,),query_tile=0)
    expected=np.zeros(65536);expected[:2048]=data[:2048]
    np.testing.assert_array_equal(out.output.polynomial,expected)
    assert out.output.level==23 and a.calls==dict(weighted=1) and result.output is None

@pytest.mark.parametrize('field,value',[('direct_value_queries',()),('query_tile',True),('query_positions',(2,0)),('layer',True)])
def test_PV_rejects_typed_result_layout_before_work(field,value):
    a=Arithmetic();result=QwenProbabilityTile(Value(np.zeros(65536),24),(0,2),(0,2),(0,),0,3,{}, {},{})
    object.__setattr__(result,field,value);pv=pv_setup(a,(0,2))
    with pytest.raises(ValueError):pv.attend_softmax(result,(Value(np.zeros(65536),24),),query_tile=0)
    assert not a.calls

@pytest.mark.parametrize('target',['source_payload','header','current_nsd','accessor'])
def test_PV_callback_retains_all_originals_and_header(target):
    a=Arithmetic();v=Value(np.zeros(65536),24);result=QwenProbabilityTile(Value(np.zeros(65536),24),(0,2),(0,2),(0,),0,3,{}, {},{});stopped=[]
    def callback(label,current,meta):
        stopped.append(dict(a.calls))
        if target=='source_payload':v.polynomial=np.ones(65536)
        elif target=='header':object.__setattr__(result,'direct_value_queries',())
        elif target=='current_nsd':current._ckks_nsd=True
        else:a.add=lambda *args:None
    pv=pv_setup(a,(0,2),callback)
    with pytest.raises(ValueError):pv.attend_softmax(result,(v,),query_tile=0)
    assert stopped and dict(a.calls)==stopped[0] and pv.last_run is None and not pv._active

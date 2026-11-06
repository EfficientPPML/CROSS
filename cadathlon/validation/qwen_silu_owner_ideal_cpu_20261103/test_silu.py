from pathlib import Path
import math,os,subprocess,sys,json
import numpy as np
import pytest
from silu_ideal import owner,contract,reference,Value,DELTA,Q,P
from qwen_silu_contract import silu_scale_plan,QwenSiLUContract,contract_state
import encrypted_qwen_silu as runtime
from silu_metrics import record

@pytest.fixture(autouse=True)
def isolated_quarantine():
    runtime._QUARANTINE.clear();yield;runtime._QUARANTINE.clear()

@pytest.mark.parametrize('layer',range(28))
@pytest.mark.parametrize('rows',[1,16])
def test_all_sites_full_slots(layer,rows):
    o,a,r=owner(layer,positions=tuple(3*i for i in range(rows)));c=o.contract
    lo,hi=c.domain;x=np.full((16,4096),-321.25)
    active=np.linspace(lo,hi,rows*3072).reshape(rows,3072)
    active[:,0]=lo;active[:,1]=hi;active[:,2]=0.;x[:rows,:3072]=active
    old=x.copy();expected=np.zeros_like(x);expected[:rows,:3072],domains=reference(active,layer)
    source=Value(x.ravel(),24,math.nextafter(DELTA,math.inf));before=source.polynomial.copy()
    result=o.evaluate_tile(source,tile=0)
    assert result.output.level==6 and result.original_positions==o.layout.positions
    assert np.max(np.abs(result.output.polynomial-expected.ravel()))<=1e-8
    assert np.array_equal(source.polynomial,before) and np.array_equal(x,old)
    inactive=np.ones_like(x,dtype=bool);inactive[:rows,:3072]=False
    assert not np.any(result.output.polynomial.reshape(16,4096)[inactive])
    assert a.calls['weighted']==result.plan['counts']['weighted_sums']
    assert a.calls['multiply']==result.plan['counts']['ciphertext_products']
    assert a.calls['rotate']==a.calls['add']==0
    if r:
        assert r.calls==[(18 if i==0 else 8,2.) for i in range(c.doublings)]
        assert result.plan['counts']==dict(weighted_sums=3+7*c.doublings,ciphertext_products=9+10*c.doublings,refreshes=c.doublings)
        assert len(domains)==1+c.doublings
    else:assert result.plan['counts']==dict(weighted_sums=c.degree+1,ciphertext_products=c.degree-1,refreshes=0)
    record(f'layer{layer}.rows{rows}',error=np.max(np.abs(result.output.polynomial-expected.ravel())),domains=domains,result=result,scope='Fixed declared endpoints/interior/zero, full physical slots, explicit ideal primitives/identity refresh only.')

@pytest.mark.parametrize('layer',[0,3,26,27])
def test_partial_second_tile_original_positions_and_invalid_refresh_lanes(layer):
    positions=tuple(2*i+1 for i in range(19));o,a,r=owner(layer,positions=positions)
    x=np.full((16,4096),1000.);active=np.linspace(*o.contract.domain,3*3072).reshape(3,3072);x[:3,:3072]=active
    if r:
        def inject(old,new):new.polynomial.reshape(16,4096)[3:,:]=.125;new.polynomial.reshape(16,4096)[:3,3072:]=-.125
        r.inject=inject
    result=o.evaluate_tile(Value(x.ravel(),24),tile=1);expected=np.zeros_like(x);expected[:3,:3072]=reference(active,layer)[0]
    assert np.max(np.abs(result.output.polynomial-expected.ravel()))<=1e-8
    assert result.original_positions==positions[16:]
    record(f'layer{layer}.partial3',error=np.max(np.abs(result.output.polynomial-expected.ravel())),domains=reference(active,layer)[1],result=result,scope='Partial second tile/original positions; small nonzero inactive refresh lanes, ideal primitives only.')

@pytest.mark.parametrize('layer',[0,3])
def test_captured_output_has_no_semantic_tags_input_untouched(layer):
    o,a,r=owner(layer);x=Value(np.zeros(65536),24);x._public_abs_bound=7.
    def progress(label,v,meta):v._public_abs_bound=0.;v._public_interval=(0.,0.)
    o.progress=progress # constructor binding must reject replacement; reconstruct instead
    o,a,r=owner(layer,progress=progress)
    result=o.evaluate_tile(x,tile=0,capture_checkpoints=True)
    assert x._public_abs_bound==7.
    assert not any(n.startswith('_public_') for v in (result.output,*result.checkpoints.values()) for n in vars(v))
    assert result.checkpoints['output']is result.output

@pytest.mark.parametrize('layer',range(28))
def test_actual_metadata_plan_has_no_nominal_scale_relabel(layer):
    c=contract(layer);scales=tuple(math.nextafter(DELTA,math.inf if i%2 else -math.inf) for i in range(c.doublings))
    p=silu_scale_plan(Q,c,input_scale=math.nextafter(DELTA,math.inf),refresh_scales=scales)
    assert p['output_level']==6
    if c.doublings:
        assert p['output_scale']!=DELTA
        for j in range(1,c.doublings+1):
            levels={s['label']:s['level'] for s in p['stages']}
            assert [levels[f'double{j}.'+n] for n in ['refresh','square','denominator','inverse.first','inverse5','quotient','output']]==[25,24,23,22,10,9,8]

@pytest.mark.parametrize('field,value', [('layer',True),('degree',True),('doublings',6.),('coefficients',[]),('domain',[-91,91]),('input_factor',True),('events',()),('inverse_rule',())])
def test_typed_contract_mutation_rejected_before_primitive(field,value):
    o,a,r=owner(3);object.__setattr__(o.contract,field,value)
    with pytest.raises((ValueError,TypeError)):o.evaluate_tile(Value(np.zeros(65536),24),tile=0)
    assert not a.calls and o.last_run is None

@pytest.mark.parametrize('label',['map','double1.refresh','output'])
@pytest.mark.parametrize('field,value',[('num_moduli',1),('_ckks_nsd',True),('_ckks_nsd',1.),('precision',64),('r',512),('c',256),('moduli_array',None)])
def test_current_callback_complete_metadata(label,field,value):
    marks=[]
    def progress(name,v,meta):
        if name==label:marks.append(dict(a.calls));setattr(v,field,value)
    o,a,r=owner(3,progress=progress)
    with pytest.raises(ValueError):o.evaluate_tile(Value(np.zeros(65536),24),tile=0)
    assert marks and dict(a.calls)==marks[0] and o.last_run is None
    assert runtime._QUARANTINE

@pytest.mark.parametrize('retained_label',['map','basis2','double1.denominator','double1.inverse.first'])
def test_earlier_captured_wrapper_and_array_retained_through_final_callback(retained_label):
    saved={};marks=[]
    layer=0 if retained_label.startswith(('map','basis')) else 3
    def progress(label,v,meta):
        if label==retained_label:saved['value']=v;saved['array']=v.polynomial
        if label=='output':
            assert saved['value'].polynomial is saved['array'];marks.append(dict(a.calls));saved['value'].num_moduli=1
    o,a,r=owner(layer,progress=progress)
    with pytest.raises(ValueError):o.evaluate_tile(Value(np.zeros(65536),24),tile=0,capture_checkpoints=True)
    assert marks and dict(a.calls)==marks[0] and o.last_run is None

@pytest.mark.parametrize('layer',[0,3])
def test_original_source_mutation_even_all_zero_path(layer):
    source=Value(np.zeros(65536),24);marks=[]
    def progress(label,v,meta):
        if label=='output':marks.append(dict(a.calls));source.polynomial=np.zeros(65536)
    o,a,r=owner(layer,progress=progress)
    with pytest.raises(ValueError):o.evaluate_tile(source,tile=0)
    assert marks and dict(a.calls)==marks[0] and runtime._QUARANTINE

@pytest.mark.parametrize('mode',['arithmetic','evaluator','engine'])
@pytest.mark.parametrize('value',[False,1,1.])
def test_synchronous_exact_true(mode,value):
    o,a,r=owner(3)
    if mode=='arithmetic':a.low_memory=value
    elif mode=='evaluator':r.evaluator.low_memory=value
    else:r.engine._low_mem=value
    with pytest.raises(ValueError):o.evaluate_tile(Value(np.zeros(65536),24),tile=0)
    assert not a.calls

@pytest.mark.parametrize('change',['refresh','finish','helper','compiler','cache'])
def test_callback_bound_accessor_compiler_cache(change):
    marks=[]
    def progress(label,v,meta):
        if label!='map':return
        marks.append(dict(a.calls))
        if change=='refresh':r.refresh=lambda *args,**kwargs:None
        elif change=='finish':a._finish=lambda x:x
        elif change=='helper':a._plaintext_multiply.ctx=object()
        elif change=='compiler':a._multiplication_compiler.max_executables+=1
        else:a._constant_cache[('illegal',)]=Value(np.zeros(65536),24)
    o,a,r=owner(3,progress=progress)
    with pytest.raises(ValueError):o.evaluate_tile(Value(np.zeros(65536),24),tile=0)
    assert marks and dict(a.calls)==marks[0]

@pytest.mark.parametrize('error',[RuntimeError,KeyboardInterrupt,SystemExit])
def test_primitive_pre_return_failure_quarantines_owners(error):
    o,a,r=owner(3);source=Value(np.zeros(65536),24)
    # Simulate an actual readiness failure after a primitive entered. Bind the
    # injected method at construction so the test reaches the body failure.
    from types import MethodType
    def fail(self,*args,**kwargs):raise error('failed array readiness')
    a.weighted_sum=MethodType(fail,a);o._methods=o._accessors()
    with pytest.raises(error):o.evaluate_tile(source,tile=0)
    assert runtime._QUARANTINE and runtime._QUARANTINE[0][2]is a
    with pytest.raises(RuntimeError,match='dedicated-process'):o.evaluate_tile(source,tile=0)


def test_import_no_backend_or_model():
    command=[sys.executable,'-c',"import sys;sys.path.insert(0,'checkout/cadathlon/common');import encrypted_qwen_silu;assert not any(n.split('.')[0] in ('jax','jaxlib','torch','transformers','safetensors') for n in sys.modules)"]
    result=subprocess.run(command,cwd=P,capture_output=True,text=True,check=False)
    assert result.returncode==0,result.stderr

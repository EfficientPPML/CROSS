"""All frozen GELU rules on ideal full slots and actual scale equations."""
from pathlib import Path
import hashlib,json,math,os
import numpy as np
import pytest
from test_routing import backend_free,SLOTS
from test_gelu64 import Arithmetic,value,Q,SCALE,MANIFEST
from encrypted_bert_v2_gelu import BertV2Gelu,Gelu64Contract
from encrypted_bert_all_gelu import BertGelu,BertGeluContract,gelu_plan

@pytest.mark.parametrize('layer',range(12))
@pytest.mark.parametrize('level_choice',('minimum',24,49))
@pytest.mark.parametrize('source_scale',(SCALE,math.nextafter(SCALE,math.inf)))
def test_every_layer_and_physical_slot(layer,level_choice,source_scale):
    contract=BertGeluContract(MANIFEST,layer);depth=2*math.ceil(math.log2(contract.degree))+2
    level=depth+2 if level_choice=='minimum' else level_choice
    lo,hi=contract.domain;data=np.linspace(lo,hi,SLOTS);data[::11]=0.
    data[:5]=[lo,hi,0.,-1.,1.];source=value(data,level,source_scale)
    ops=Arithmetic();owner=BertGelu(ops,contract);output=owner.evaluate_tile(source)
    reference=np.polynomial.chebyshev.chebval((2/(hi-lo))*data-(hi+lo)/(hi-lo),contract.coefficients)
    error=float(np.max(np.abs(output.data-reference)))
    assert np.isfinite(output.data).all() and error<1e-10
    assert output.level==level-depth>=2 and output._ckks_scale==owner.last_plan['output_scale']
    assert sum(c[0]=='multiply' for c in ops.calls)==contract.degree-1
    assert sum(c[0]=='weighted_sum' for c in ops.calls)==contract.degree+1
    assert not any(c[0]=='rotate' for c in ops.calls)
    np.testing.assert_array_equal(source.data,data)
    assert source._public_bound=='deliberately stale' and not any(n.startswith('_public_') for n in vars(output))
    assert not owner._active
    destination=os.environ.get('GELU_CASE_REPORT_DIR')
    if destination:
        p=Path(destination);p.mkdir(exist_ok=True)
        name=f'{layer}-{level_choice}-{source_scale.hex()}.json'
        row=dict(layer=layer,degree=contract.degree,domain=contract.domain,source_level=level,
            source_scale=source_scale,level_choice=level_choice,coefficient_sha256=hashlib.sha256(np.asarray(contract.coefficients,dtype='<f8').tobytes()).hexdigest(),
            physical_slots_checked=SLOTS,max_abs_vs_independent_chebval=error,gate=1e-10,actual_plan=owner.last_plan)
        (p/name).write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')

def test_layer0_is_identical_to_frozen_specialized_owner():
    x=np.linspace(-21.,14.,SLOTS);old_ops,new_ops=Arithmetic(),Arithmetic()
    old=BertV2Gelu(old_ops,Gelu64Contract(MANIFEST));new=BertGelu(new_ops,BertGeluContract(MANIFEST,0))
    before=old.evaluate_tile(value(x,24,SCALE*.93));after=new.evaluate_tile(value(x,24,SCALE*.93))
    np.testing.assert_array_equal(before.data,after.data)
    assert old.last_plan==new.last_plan and old_ops.calls==new_ops.calls

@pytest.mark.parametrize('layer',(-1,12,True,0.,'0',None))
def test_typed_layer_rejected_without_work(layer):
    ops=Arithmetic()
    with pytest.raises(ValueError):BertGelu(ops,BertGeluContract(MANIFEST,layer))
    assert not ops.calls

@pytest.mark.parametrize('field',('layer','degree','domain','coefficients','manifest_bytes'))
def test_contract_change_fails_before_first_primitive(field):
    ops=Arithmetic();contract=BertGeluContract(MANIFEST,9);owner=BertGelu(ops,contract)
    change={'layer':9.,'degree':192.,'domain':list(contract.domain),'coefficients':tuple([0.]*193),'manifest_bytes':MANIFEST+b' '}
    object.__setattr__(contract,field,change[field])
    with pytest.raises(ValueError):owner.evaluate_tile(value(np.zeros(SLOTS),24,SCALE))
    assert not ops.calls and not owner._active

@pytest.mark.parametrize('event',('domain map','T192','output'))
@pytest.mark.parametrize('mutation',('source','current','retained','context'))
def test_largest_graph_callback_continuation_is_guarded(event,mutation):
    ops=Arithmetic();source=value(np.zeros(SLOTS),24,SCALE);saved=[];calls=[]
    def progress(label,result,counts):
        if label=='domain map':saved.append(result)
        if label!=event:return
        calls.append(len(ops.calls))
        if mutation=='source':source._ckks_nsd=True
        elif mutation=='current':result._ckks_nsd=True
        elif mutation=='retained':saved[0]._ckks_nsd=True
        else:ops.ctx.secret_key=object()
    owner=BertGelu(ops,BertGeluContract(MANIFEST,9),progress=progress)
    with pytest.raises(ValueError):owner.evaluate_tile(source)
    assert calls==[len(ops.calls)] and owner.last_plan is None and not owner._active

def test_annotations_reentry_and_primary_exception():
    ops=Arithmetic();source=value(np.zeros(SLOTS),24,SCALE);saved=[]
    def progress(label,result,counts):
        saved.append(result)
        for v in saved:v._public_abs_bound=0.
        with pytest.raises(ValueError,match='reentered'):owner.evaluate_tile(source)
    owner=BertGelu(ops,BertGeluContract(MANIFEST,9),progress=progress)
    result=owner.evaluate_tile(source)
    assert all(not any(n.startswith('_public_') for n in vars(v)) for v in saved)
    assert not owner._active and result.level==6
    error=KeyboardInterrupt('keep primary');ops=Arithmetic()
    def cancel(*args):raise error
    owner=BertGelu(ops,BertGeluContract(MANIFEST,11),progress=cancel)
    with pytest.raises(KeyboardInterrupt) as caught:owner.evaluate_tile(source)
    assert caught.value is error and len(ops.calls)==1 and not owner._active

@pytest.mark.parametrize('layer',range(12))
def test_all_supported_input_levels_have_two_level_reserve(layer):
    c=BertGeluContract(MANIFEST,layer);depth=2*math.ceil(math.log2(c.degree))+2
    for level in range(depth+2,50):
        plan=gelu_plan(Q[:2*(level+1)],degree=c.degree,input_level=level,canonical_scale=SCALE,input_scale=math.nextafter(SCALE,math.inf))
        assert plan['output_level']==level-depth>=2 and plan['consumed_levels']==depth
        assert plan['ciphertext_products']==c.degree-1 and plan['weighted_operations']==c.degree+1
        assert plan['actual_composite_rescales']==2*c.degree
    with pytest.raises(ValueError):gelu_plan(Q[:2*(depth+2)],degree=c.degree,input_level=depth+1,canonical_scale=SCALE,input_scale=SCALE)

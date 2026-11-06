import copy
import pytest
from qwen_file_contract import *
from qwen_public_parameters import ParameterCatalog

@pytest.mark.parametrize('length',[1,8,16,17,19,32])
def test_real_public_profile_and_variable_tile_geometry(public,length):
    identity=public.identity(context_id='1'*64,model_id='2'*64,producer_sources='3'*64,consumer_sources='4'*64)
    c=request(public,identity,'r',[3*i for i in range(length)])
    assert c['input']['input_tile_count']==(length+15)//16
    assert sum(c['input']['valid_rows'])==length
    assert c['response']['last_original_position']==3*(length-1)
    assert c['transport_input_level']==49 and c['owner_input_level']==25
    b=ciphertext_binding(public,c,'request',0)
    assert b['shape']==[1,2,256,512,100] and b['scale']==DELTA
    outputs=[artifact(public,c,'response',i,payload_sha256='5'*64,actual_scale=float(i+1),input_digest='6'*64)for i in range(38)]
    out=response(public,c,'6'*64,outputs)
    assert sum(x['binding']['logical']['stop']-x['binding']['logical']['start']for x in out['artifacts'])==151936
    assert outputs[-1]['binding']['logical']['stop']-outputs[-1]['binding']['logical']['start']==384
    assert all(x['binding']['shape']==[1,2,256,512,6]and x['binding']['level']==2 for x in outputs)

@pytest.mark.parametrize('positions',[[True],[0.0],[],list(range(33)),[1,1],[3,1],[-1],[40960]])
def test_positions_reject_aliases_missing_and_outside(positions):
    with pytest.raises(ValueError):packing(positions)

@pytest.mark.parametrize('change',[lambda c:c['input'].update(input_tile_count=8),lambda c:c['input']['packing'].update(active_features=768),
    lambda c:c.update(owner_input_level=25.0),lambda c:c['response'].update(ciphertext_count=1),
    lambda c:c['public'].update(producer_source_manifest_sha256='4'*64),lambda c:c['public']['parameters'].update(dnum=True)])
def test_request_exact_schema(public,request_case,change):
    original=copy.deepcopy(request_case);change(request_case)
    # A different legitimate producer hash is a different request identity,
    # not intrinsically invalid. It must not equal the client-retained binding.
    if request_case['public']['producer_source_manifest_sha256']!=original['public']['producer_source_manifest_sha256']:
        assert digest(request_case)!=digest(original)
    else:
        with pytest.raises((ValueError,TypeError)):validate_request(public,request_case)

def test_ordered_variable_input_digest_and_output_scope(public,request_case):
    inputs=[artifact(public,request_case,'request',i,payload_sha256=str(i+1)*64)for i in range(2)]
    digest0=request_digest(public,request_case,inputs)
    with pytest.raises(ValueError):request_digest(public,request_case,inputs[::-1])
    outputs=[artifact(public,request_case,'response',i,payload_sha256='7'*64,actual_scale=DELTA,input_digest=digest0)for i in range(38)]
    good=response(public,request_case,digest0,outputs)
    assert validate_response(public,request_case,digest0,good)==good
    for changed in (outputs[:-1],outputs[::-1],outputs+[outputs[-1]]):
        with pytest.raises(ValueError):response(public,request_case,digest0,changed)
    with pytest.raises(ValueError):response(public,request_case,'8'*64,outputs)

@pytest.mark.parametrize('value',[1,True,0.,-1.,float('inf'),float('nan')])
def test_actual_binary64_scales_required(public,request_case,value):
    with pytest.raises(ValueError):ciphertext_binding(public,request_case,'response',0,actual_scale=value,input_digest='8'*64)

def test_actual_public_parameter_inventory_alias_and_provenance(public):
    catalog=ParameterCatalog.from_public(public)
    matrices=parse(catalog.matrices);gammas=parse(catalog.gammas)
    assert len(matrices)==198 and len(gammas)==113
    assert sum(r['family']=='qkv'for r in matrices.values())==84
    assert len({(r['family'],r['descriptor']['file'])for r in matrices.values()})==197
    head=matrices['lm_head.weight']['descriptor'];embedding=matrices['model.embed_tokens.weight']['descriptor']
    assert embedding['storage_alias_of']=='lm_head.weight' and embedding['sha256']==head['sha256']
    assert parse(catalog.provenance)['historical_decoder_identity']['candidate_sha256'].startswith('7f6478')

def test_public_source_corruption_rejected_before_catalog(public):
    blobs=public.check();blobs['inputs/config.json']+=b' '
    with pytest.raises(ValueError):PublicContract.from_blobs(blobs)

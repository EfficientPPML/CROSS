"""Public schema/actual-weight checks only; no backend, encryption or model forward."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import layer_attention_contract as c

CTX = 'a' * 64
REQ = 'b' * 32
MODEL = Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c/model.safetensors')


def contract(k=15, layer=0):
    return c.make_contract(tuple(range(k)), CTX, REQ, layer=layer)


def test_all1536_layer_counts_and_noncontiguous_positions():
    for layer in range(12):
        for k in range(1, 129):
            value = contract(k, layer)
            assert c.validate_contract(value) == value
            assert value['query_tiles'] == (k+15)//16
            assert sum(value['output_valid_rows']) == k
            assert value['input_tile_count'] == (8 if layer == 0 else (k+15)//16)
            assert value['input_level'] == (49 if layer == 0 else None)
            assert value['input_level_range'] == ([49,49] if layer == 0 else [2,49])
            assert value['output_level'] == (9 if layer < 6 else 8)
            for role, fields in value['model']['tensors'].items():
                for kind, item in fields.items():
                    assert item['name'] == f'bert.encoder.layer.{layer}.{c.PREFIX[role]}.{kind}'
                    assert item['sha256'] == c.TENSOR_SHA256[layer][role][kind]
        value = c.make_contract((0,2,7,127),CTX,REQ,layer=layer)
        assert value['public_key_mask'] == [True]*4+[False]*124
        assert value['original_positions'] == [0,2,7,127]


@pytest.mark.parametrize('positions',[(),tuple(range(129)),[0],(True,),(1.,),(np.int64(1),),(-1,),(128,),(2,1),(1,1)])
def test_positions(positions):
    with pytest.raises(ValueError): c.make_contract(positions,CTX,REQ,layer=1)


@pytest.mark.parametrize('layer',[-1,12,True,1.,np.int64(1),'1',None])
def test_layer_types(layer):
    with pytest.raises(ValueError): contract(layer=layer)
    with pytest.raises(ValueError): c.validate_projection_arrays({},layer)


@pytest.mark.parametrize('name,value',[('context_id','A'*64),('context_id','a'*63),('request_id','b'*64),('request_id',None)])
def test_ids(name,value):
    args=dict(original_positions=(0,),context_id=CTX,request_id=REQ,layer=1);args[name]=value
    with pytest.raises(ValueError): c.make_contract(**args)


@pytest.mark.parametrize('field,value',[('layer',1.),('key_count',15.),('input_tile_count',8),('input_level',49),('input_level_range',[1,49]),('output_level',9),('output_valid_rows',[16]),('extra',1)])
def test_contract_substitution(field,value):
    v=contract(layer=6);v[field]=value
    with pytest.raises(ValueError): c.validate_contract(v)


@pytest.mark.parametrize('field',['profile','packing','model','source_input_sha256','public_key_mask'])
def test_nested_substitution(field):
    v=contract(layer=2)
    if field=='profile':v[field]['q_towers'].reverse()
    elif field=='packing':v[field]['learned_position_order']='after-compaction'
    elif field=='model':v[field]['tensors']['key']['bias']['sha256']='0'*64
    elif field=='source_input_sha256':v[field]['candidate_manifest.json']='0'*64
    else:v[field][127]=True
    with pytest.raises(ValueError):c.validate_contract(v)


@pytest.mark.parametrize('name',list(c.INPUT_SHA256))
def test_each_pinned_input_required_before_contract(tmp_path,monkeypatch,name):
    for filename in c.INPUT_SHA256:(tmp_path/filename).write_bytes((c.ROOT/filename).read_bytes())
    p=tmp_path/name;p.write_bytes(p.read_bytes()+b' ')
    monkeypatch.setattr(c,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='changed pinned input'):contract(layer=11)


def test_parameter_copy_is_independent():
    p=c.parameters();p['q_towers'].reverse();p['degree_layout'][0]=1
    q=c.parameters();assert q['degree_layout']==[256,512] and p['q_towers']!=q['q_towers']


@pytest.fixture(scope='module')
def projections():
    with MODEL.open('rb') as f:before=hashlib.file_digest(f,'sha256').hexdigest()
    assert before==c._pinned('verified_model_manifest.json')['files']['model.safetensors']['sha256']
    result={}
    with MODEL.open('rb') as f:
        n=int.from_bytes(f.read(8),'little');header=json.loads(f.read(n));base=n+8
        for layer in range(12):
            result[layer]={}
            for role,prefix in c.PREFIX.items():
                result[layer][role]={}
                for kind in ('weight','bias'):
                    desc=header[f'bert.encoder.layer.{layer}.{prefix}.{kind}']
                    shape=[768,768] if kind=='weight' else [768]
                    assert desc['dtype']=='F32' and desc['shape']==shape
                    lo,hi=desc['data_offsets'];f.seek(base+lo);raw=f.read(hi-lo)
                    assert len(raw)==4*np.prod(shape)
                    assert hashlib.sha256(raw).hexdigest()==c.TENSOR_SHA256[layer][role][kind]
                    result[layer][role][kind]=np.frombuffer(raw,dtype='<f4').reshape(shape)
    yield result
    with MODEL.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==before


@pytest.mark.parametrize('layer',range(12))
def test_all96_actual_pins_exact_lift_and_immutability(projections,layer):
    result=c.validate_projection_arrays(projections[layer],layer)
    for role in c.PREFIX:
        for kind in ('weight','bias'):
            a=result[role][kind]
            assert a.dtype==np.float64 and not a.flags.writeable
            assert np.array_equal(a,projections[layer][role][kind].astype(np.float64))
            with pytest.raises(ValueError):a.setflags(write=True)
    again=c.validate_projection_arrays(result,layer)
    assert np.array_equal(again['query']['bias'],result['query']['bias'])
    with pytest.raises(ValueError):c.validate_projection_arrays(projections[(layer+1)%12],layer)


@pytest.mark.parametrize('kind',['complex','object','string','integer','nan','wrong_shape','changed','rounded_away','extra','missing'])
def test_projection_rejection(projections,kind):
    d={role:dict(arrays) for role,arrays in projections[7].items()};a=d['query']['bias'].astype(np.float64)
    if kind=='complex':a=a.astype(complex)+1j
    elif kind=='object':a=a.astype(object)
    elif kind=='string':a=a.astype(str)
    elif kind=='integer':a=a.astype(np.int64)
    elif kind=='nan':a[0]=np.nan
    elif kind=='wrong_shape':a=a[:,None]
    elif kind=='changed':a[0]+=.1
    elif kind=='rounded_away':a[0]=np.nextafter(a[0],np.inf)
    elif kind=='extra':d['extra']=d['query']
    elif kind=='missing':del d['key']
    d['query']['bias']=a
    with pytest.raises(ValueError):c.validate_projection_arrays(d,7)


def inputs(v):
    n=v['input_tile_count'];canonical=c.parameters()['scaling_factor']
    levels=[49]*n if v['layer']==0 else [2+i for i in range(n)]
    scales=[canonical]*n if v['layer']==0 else [float(canonical*(i+1)/2) for i in range(n)]
    records=[c.make_descriptor(v,role='attention_input',tile_index=i,payload_sha256=f'{i:064x}',level=l,scale=s) for i,(l,s) in enumerate(zip(levels,scales))]
    return records,levels,scales


@pytest.mark.parametrize('layer',[0,1,5,6,11])
def test_actual_level_scale_binding_and_request_identity(layer):
    v=contract(17,layer);records,levels,scales=inputs(v)
    binding=c.request_digest(records,v,expected_levels=levels,expected_scales=scales)
    assert c.validate_descriptors(records,v,role='attention_input',expected_levels=levels,expected_scales=scales)==records
    outscales=[np.nextafter(scales[0],np.inf).item(),scales[-1]]
    outlevels=[v['output_level']]*2
    out=[c.make_descriptor(v,role='attention_output',tile_index=i,payload_sha256='c'*64,level=l,scale=s,input_digest=binding) for i,(l,s) in enumerate(zip(outlevels,outscales))]
    assert [x['valid_rows'] for x in out]==[16,1]
    assert c.validate_descriptors(out,v,role='attention_output',expected_levels=outlevels,expected_scales=outscales,input_digest=binding)==out
    for key,replacement in [('expected_scales',[scales[0]]*2),('expected_levels',[v['output_level']-1]*2),('input_digest','0'*64)]:
        kwargs=dict(expected_scales=outscales,expected_levels=outlevels,input_digest=binding);kwargs[key]=replacement
        with pytest.raises(ValueError):c.validate_descriptors(out,v,role='attention_output',**kwargs)
    records[-1]['payload_sha256']='e'*64
    assert c.request_digest(records,v,expected_levels=levels,expected_scales=scales)!=binding


@pytest.mark.parametrize('mutation',['missing','extra','duplicate','reverse','role','ctx','request','nsd_bool','nsd_float','nsd_numpy','scale_int','scale_numpy','level','moduli','rows','contract','layer'])
def test_descriptor_rejection(mutation):
    v=contract(17,3);d,levels,scales=inputs(v)
    if mutation=='missing':d.pop()
    elif mutation=='extra':d.append(d[-1])
    elif mutation=='duplicate':d[1]=copy.deepcopy(d[0])
    elif mutation=='reverse':d.reverse()
    else:
        fields={'role':('role','attention_output'),'ctx':('context_id','0'*64),'request':('request_id','0'*32),
                'nsd_bool':('nsd',True),'nsd_float':('nsd',1.),'nsd_numpy':('nsd',np.int64(1)),
                'scale_int':('scale',int(d[0]['scale'])),'scale_numpy':('scale',np.float64(d[0]['scale'])),
                'level':('level',49),'moduli':('moduli',d[0]['moduli'][::-1]),'rows':('valid_rows',15),
                'contract':('contract_sha256','0'*64),'layer':('layer',2)}
        key,value=fields[mutation];d[0][key]=value
    with pytest.raises((ValueError,TypeError)):c.validate_descriptors(d,v,role='attention_input',expected_levels=levels,expected_scales=scales)


@pytest.mark.parametrize('role',['attention_input','attention_output'])
@pytest.mark.parametrize('name',['expected_levels','expected_scales'])
@pytest.mark.parametrize('bad',[None,[],[True],[1.]])
def test_missing_or_typed_metadata_plan(role,name,bad):
    v=contract(1,11);d,levels,scales=inputs(v)
    kwargs=dict(expected_levels=levels,expected_scales=scales);kwargs[name]=bad
    with pytest.raises(ValueError):c.validate_descriptors(d,v,role=role,**kwargs)


@pytest.mark.parametrize('scale',[None,True,1,np.float64(1),float('inf'),float('nan'),-1.,0.,complex(1)])
@pytest.mark.parametrize('role',['attention_input','attention_output'])
def test_invalid_actual_scale(scale,role):
    with pytest.raises(ValueError):c.make_descriptor(contract(layer=1),role=role,tile_index=0,payload_sha256='a'*64,level=2 if role=='attention_input' else9,scale=scale,input_digest=None if role=='attention_input' else '0'*64)


@pytest.mark.parametrize('level',[None,True,1.,np.int64(2),-1,0,1,50])
def test_invalid_later_level(level):
    with pytest.raises(ValueError):c.make_descriptor(contract(layer=1),role='attention_input',tile_index=0,payload_sha256='a'*64,level=level,scale=1.)


def test_later_boundary_levels_and_layer0_canonical_rules():
    v=contract(17,11)
    for i,level in enumerate((2,49)):
        d=c.make_descriptor(v,role='attention_input',tile_index=i,payload_sha256='a'*64,level=level,scale=float(i+1))
        assert len(d['moduli'])==2*(level+1)
    v=contract(1,0)
    with pytest.raises(ValueError):c.make_descriptor(v,role='attention_input',tile_index=0,payload_sha256='a'*64,level=48)
    with pytest.raises(ValueError):c.make_descriptor(v,role='attention_input',tile_index=0,payload_sha256='a'*64,scale=np.nextafter(c.parameters()['scaling_factor'],np.inf).item())
    with pytest.raises(ValueError):c.make_descriptor(v,role='attention_input',tile_index=0,payload_sha256='a'*64,input_digest='0'*64)


def test_import_no_crypto_backend():
    assert 'jax' not in sys.modules and 'torch' not in sys.modules

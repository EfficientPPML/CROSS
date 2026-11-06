import copy
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
import first_attention_contract as c

CTX = 'a' * 64
REQ = 'b' * 32


def contract(k=15):
    return c.make_contract(tuple(range(k)), CTX, REQ)


def test_all_key_counts_and_noncontiguous_positions():
    for k in range(1, 129):
        v = contract(k)
        assert c.validate_contract(v) == v
        assert v['query_tiles'] == (k + 15) // 16
        assert sum(v['output_valid_rows']) == k
        assert v['input_tile_count'] == 8
    v = c.make_contract((0, 2, 7, 127), CTX, REQ)
    assert v['original_positions'] == [0, 2, 7, 127]
    assert v['public_key_mask'] == [True]*4 + [False]*124
    assert c.parameters()['q_towers'] == v['profile']['q_towers']


@pytest.mark.parametrize('positions', [(), tuple(range(129)), [0], (True,), (1.,), (np.int64(1),), (-1,), (128,), (2, 1), (1, 1)])
def test_bad_positions(positions):
    with pytest.raises(ValueError): c.make_contract(positions, CTX, REQ)


@pytest.mark.parametrize('name,value', [('context_id','A'*64), ('context_id','a'*63), ('request_id','b'*64), ('request_id',None)])
def test_bad_ids(name, value):
    args = dict(original_positions=(0,), context_id=CTX, request_id=REQ); args[name] = value
    with pytest.raises(ValueError): c.make_contract(**args)


@pytest.mark.parametrize('field,value', [('key_count',16), ('key_count',15.), ('input_tile_count',7), ('input_level',48), ('output_level',10), ('output_valid_rows',[16]), ('extra',1)])
def test_contract_substitution(field, value):
    v = contract(); v[field] = value
    with pytest.raises(ValueError): c.validate_contract(v)


@pytest.mark.parametrize('field', ['profile','packing','model','source_input_sha256','public_key_mask'])
def test_nested_substitution(field):
    v = contract()
    if field == 'profile': v[field]['q_towers'].reverse()
    elif field == 'packing': v[field]['learned_position_order'] = 'after-compaction'
    elif field == 'model': v[field]['tensors']['key']['bias']['sha256'] = '0'*64
    elif field == 'source_input_sha256': v[field]['candidate_manifest.json'] = '0'*64
    else: v[field][127] = True
    with pytest.raises(ValueError): c.validate_contract(v)


def test_pins_fail_before_contract(tmp_path, monkeypatch):
    for name in c.INPUT_SHA256:
        (tmp_path/name).write_bytes((c.ROOT/name).read_bytes())
    d = json.loads((tmp_path/'p60_profile_and_refresh.json').read_bytes())
    d['profile']['q'].reverse(); (tmp_path/'p60_profile_and_refresh.json').write_text(json.dumps(d))
    monkeypatch.setattr(c, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='changed pinned input'): c.parameters()


@pytest.fixture(scope='module')
def projections():
    p = Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c/model.safetensors')
    with p.open('rb') as f:
        hsh = hashlib.file_digest(f, 'sha256').hexdigest()
    assert hsh == c._pinned('verified_model_manifest.json')['files']['model.safetensors']['sha256']
    result = {}
    with p.open('rb') as f:
        length = struct.unpack('<Q', f.read(8))[0]; header = json.loads(f.read(length)); start = length+8
        for role, prefix in c.PREFIX.items():
            result[role] = {}
            for kind in ('weight','bias'):
                desc = header['bert.encoder.layer.0.' + prefix + '.' + kind]
                assert desc['dtype'] == 'F32'
                lo, hi = desc['data_offsets']; f.seek(start+lo)
                result[role][kind] = np.frombuffer(f.read(hi-lo),dtype='<f4').reshape(desc['shape'])
    return result


def test_pinned_projection_promotion_immutable(projections):
    result = c.validate_projection_arrays(projections)
    for role in projections:
        for kind in projections[role]:
            a = result[role][kind]
            assert a.dtype == np.float64 and not a.flags.writeable
            assert np.array_equal(a, projections[role][kind].astype(np.float64))
            with pytest.raises(ValueError): a.setflags(write=True)
    again = c.validate_projection_arrays(result)
    assert np.array_equal(again['query']['bias'], result['query']['bias'])


@pytest.mark.parametrize('kind', ['complex','object','string','integer','nan','wrong_shape','changed','rounded_away','extra','missing'])
def test_projection_rejection(projections, kind):
    d = {role: dict(arrays) for role, arrays in projections.items()}
    a = d['query']['bias'].astype(np.float64)
    if kind == 'complex': a = a.astype(complex) + 1j
    elif kind == 'object': a = a.astype(object)
    elif kind == 'string': a = a.astype(str)
    elif kind == 'integer': a = a.astype(np.int64)
    elif kind == 'nan': a[0] = np.nan
    elif kind == 'wrong_shape': a = a[:,None]
    elif kind == 'changed': a[0] += .1
    elif kind == 'rounded_away': a[0] = np.nextafter(a[0], np.inf)
    elif kind == 'extra': d['extra'] = d['query']
    elif kind == 'missing': del d['key']
    d['query']['bias'] = a
    with pytest.raises(ValueError): c.validate_projection_arrays(d)


def requests(v):
    return [c.make_descriptor(v,role='attention_input',tile_index=i,payload_sha256=f'{i:064x}') for i in range(8)]


def test_request_response_actual_scale_and_digest():
    v = contract(17); inp = requests(v); binding = c.request_digest(inp,v)
    scales = [np.nextafter(c.parameters()['scaling_factor'],np.inf).item(), c.parameters()['scaling_factor']]
    out = [c.make_descriptor(v,role='attention_output',tile_index=i,payload_sha256='c'*64,scale=s,input_digest=binding) for i,s in enumerate(scales)]
    assert [x['valid_rows'] for x in out] == [16,1]
    assert c.validate_descriptors(out,v,role='attention_output',expected_scales=scales,input_digest=binding) == out
    with pytest.raises(ValueError): c.validate_descriptors(out,v,role='attention_output',expected_scales=[c.parameters()['scaling_factor']]*2,input_digest=binding)
    with pytest.raises(ValueError): c.validate_descriptors(out,v,role='attention_output',expected_scales=scales,input_digest='0'*64)
    inp[7]['payload_sha256'] = 'e'*64
    assert c.request_digest(inp,v) != binding


@pytest.mark.parametrize('mutation', ['missing','extra','duplicate','reverse','role','ctx','request','nsd_bool','nsd_float','nsd_numpy','scale_int','level','moduli','rows','contract'])
def test_descriptor_rejection(mutation):
    v=contract(); d=requests(v)
    if mutation=='missing': d.pop()
    elif mutation=='extra': d.append(d[-1])
    elif mutation=='duplicate': d[1]=copy.deepcopy(d[0])
    elif mutation=='reverse': d.reverse()
    else:
        fields = {'role':('role','attention_output'),'ctx':('context_id','0'*64),'request':('request_id','0'*32),
                  'nsd_bool':('nsd',True),'nsd_float':('nsd',1.),'nsd_numpy':('nsd',np.int64(1)),
                  'scale_int':('scale',int(d[0]['scale'])),'level':('level',48),'moduli':('moduli',d[0]['moduli'][::-1]),
                  'rows':('valid_rows',16),'contract':('contract_sha256','0'*64)}
        key,value=fields[mutation]; d[0][key]=value
    with pytest.raises((ValueError,TypeError)): c.validate_descriptors(d,v,role='attention_input')


@pytest.mark.parametrize('scale', [None,True,1,np.float64(1),float('inf'),float('nan'),-1.,0.])
def test_invalid_response_scale(scale):
    with pytest.raises(ValueError): c.make_descriptor(contract(),role='attention_output',tile_index=0,payload_sha256='a'*64,scale=scale,input_digest='0'*64)


def test_import_no_crypto_backend():
    assert 'jax' not in sys.modules
    assert 'torch' not in sys.modules

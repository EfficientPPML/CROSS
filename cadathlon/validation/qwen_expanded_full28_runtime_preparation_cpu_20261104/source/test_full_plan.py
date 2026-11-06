import ast,json,math
from pathlib import Path
import numpy as np
import pytest
from plan_full import derive,contract,P
from full_contract import verify

@pytest.fixture(scope='module')
def report():return derive()

@pytest.mark.parametrize('sequence',range(1,33))
def test_all_workloads_and_last_row_extraction(report,sequence):
    row=report['workloads'][sequence-1];tiles=(sequence+15)//16
    assert row['tiles']==tiles and len(row['entry_events'])==28
    assert [e['input_level'] for e in row['entry_events']]==[4]*27+[3]
    assert row['entry_events'][-1]['bound']==8192. and row['output_ciphertexts']==38
    assert row['refresh_counts']['rms_inverse']==260*tiles and row['refresh_counts']['silu']==145*tiles
    assert row['refresh_counts']['interlayer']==27*tiles
    data=np.arange(65536,dtype=float).reshape(16,4096)+12345.
    target=np.linspace(-2,2,1024);data[row['last_local_row'],:1024]=target
    mask=np.zeros((16,4096));mask[row['last_local_row'],:1024]=1.
    shifted=np.roll((data*mask).ravel(),-row['last_row_rotation']).reshape(16,4096)
    want=np.zeros_like(data);want[0,:1024]=target
    np.testing.assert_array_equal(shifted,want)
    assert row['last_position']==3*(sequence-1)
    assert not row['last_row_rotation'] or row['last_row_rotation'] in report['resources']['rotation_indices']

@pytest.mark.parametrize('layer',range(28))
def test_each_original_layer_schedule(report,layer):
    r=report['layers'][layer]
    assert r['input_level']==25 and r['tail_output_level']==4 and r['original_residual_mask']==[25,24]
    assert len(r['tail_outer_events'])==2
    assert r['attention_output_level_single']==22 and r['attention_output_level_multi']==20

def test_complete_head_and_no_added_keys(report):
    b=report['vocabulary']['blocks'];assert len(b)==38
    assert [(x['start'],x['stop']) for x in b]==[(s,min(s+4096,151936)) for s in range(0,151936,4096)]
    assert sum(x['stop']-x['start'] for x in b)==151936 and b[-1]['stop']-b[-1]['start']==384
    assert all(x['physical_row']==0 and x['input_level']==3 and x['output_level']==2 for x in b)
    assert report['resources']['rotation_keys']==166 and 262143 in report['resources']['rotation_indices']
    assert report['resources']['all38_final_ciphertext_bytes']==228*1024**2
    assert report['workloads'][-1]['refresh_counts']['total']==12931
    assert report['reference']['executed'] is False

def test_actual_final_scale_examples(report):
    assert len(report['final_scale_examples'])==9
    for row in report['final_scale_examples']:
        assert row['masked_L3_scale']==row['rotation_preserves_scale']
        assert row['actual_final_norm_scale']==row['final_norm_plan']['output_scale']
        assert row['final_norm_plan']['output_level']==3
        assert all(math.isfinite(row[k]) and row[k]>0 for k in ('source_L4_scale','masked_L3_scale','actual_refresh_L25_scale','actual_final_norm_scale','actual_all38_scale'))

@pytest.mark.parametrize('field,value',[('last_tile',False),('last_local_row',0.),('last_position',False),('positions',[0]),('required_keys',[]),('entry_events',()),('vocabulary_blocks',()),('final_gamma_sha256','0'*64),('vocabulary_sha256','0'*64)])
def test_immutable_complete_typed_contract(field,value):
    c=contract((0,));object.__setattr__(c,field,value)
    with pytest.raises((ValueError,TypeError)):verify(c)

@pytest.mark.parametrize('field,value',[('input_level',3.),('bound',8192),('layer',True)])
def test_typed_final_event(field,value):
    c=contract((0,));object.__setattr__(c.entry_events[-1],field,value)
    with pytest.raises(ValueError):verify(c)

@pytest.mark.parametrize('field,value',[('start',False),('stop',4096.),('physical_row',False)])
def test_typed_vocabulary_descriptor(field,value):
    c=contract((0,));object.__setattr__(c.vocabulary_blocks[0],field,value)
    with pytest.raises(ValueError):verify(c)

def test_frozen_reference_normalizes_last_token_before_full_vocabulary():
    import hashlib
    raw=(P/'inputs/frozen_arithmetic.py').read_bytes();assert hashlib.sha256(raw).hexdigest()=='9689b843b4ec2d0a0ef724dbdca7a516e9a4f23da83f83af5571da13380e9a6f'
    tree=ast.parse(raw);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='approximate_forward')
    source=ast.unparse(fn)
    assert "value.data[:, -1]" in source and "weights['model.embed_tokens.weight']" in source
    assert "nonlinear['final.inverse_sqrt']" in source

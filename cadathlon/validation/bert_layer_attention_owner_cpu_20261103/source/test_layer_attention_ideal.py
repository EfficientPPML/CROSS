"""Pinned all-layer component inputs, actual graph loops, identity refresh only."""
from pathlib import Path
from collections import Counter
import json,math,time
import numpy as np
import pytest
from test_initial_ideal import ideal_primitives
from test_softmax_tile import refresh_and_tracked
from ideal_refresh import IdentityRefresh,install_public_bootstrap_plan
from ideal_primitives import Arithmetic,Value,IdealLinear,install_matrix_class,DELTA
from encrypted_bert_attention_tile import BertAttentionTile
from encrypted_bert_all_row_norm import BertRowLayerNorm
from encrypted_bert_layer_attention import BertLayerAttention
from layer_attention_contract import make_contract
from attention_public_fixtures import load
import layer_attention_reference as reference
HERE=Path(__file__).resolve().parent;RAW=(HERE/'candidate_manifest.json').read_bytes();MANIFEST=json.loads(RAW)

@pytest.fixture(autouse=True)
def primitives(refresh_and_tracked,monkeypatch):
    install_matrix_class(monkeypatch);install_public_bootstrap_plan(monkeypatch)
    from test_initial_ideal import Value as BaseValue
    BaseValue.degree_layout=(256,512)

@pytest.fixture(scope='module')
def arrays():
    result,models,report=load();(HERE/'fixture_pins.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return result,models

def setup(arrays,layer,K,progress=None):
    data,models=arrays;ops=Arithmetic();refresh=IdentityRefresh(ops);linear=IdealLinear(ops)
    keys=set(BertAttentionTile.required_rotation_indices())|set(BertRowLayerNorm.required_rotation_indices())|{1,64,64064}
    for key in keys:ops.ctx._param_cache.raw_rotation_keys[key]={'a':object(),'b':object()};ops.ctx._param_cache.coef_maps[key]=object()
    positions=tuple(map(int,data[f'K{K}_positions']));contract=make_contract(positions,'c'*64,'d'*32,layer=layer)
    owner=BertLayerAttention(ops,linear,refresh,models[layer],contract,RAW,data[f'layer{layer}_gamma'],data[f'layer{layer}_beta'],progress=progress)
    count=8 if layer==0 else math.ceil(K/16);packed=np.zeros((count*16,4096));packed[:K,:768]=data[f'layer{layer}_K{K}_input']
    # Later-layer entry retains arbitrary positive actual scale and low levels.
    # These are component fixtures, not substituted outputs in a joined graph.
    levels=[49]*count if layer==0 else [2+(i%4) for i in range(count)]
    sources=[Value(row.reshape(-1),level,DELTA if layer==0 else math.nextafter(DELTA,math.inf if i%2 else -math.inf))
             for i,(row,level) in enumerate(zip(packed.reshape(count,16,4096),levels))]
    return owner,ops,refresh,sources

def power_products(rule):
    assert rule['method']=='power'
    coefficients=np.polynomial.chebyshev.cheb2poly(rule['polynomial']['coefficients'])
    exponents=[i for i,c in enumerate(coefficients) if i and c!=0];powers={1}
    def visit(n):
        if n not in powers:visit(n//2);visit((n+1)//2);powers.add(n)
    for n in exponents:visit(n)
    return len(powers)-1,len(exponents)

@pytest.mark.parametrize('layer',range(12))
@pytest.mark.parametrize('K',(15,128))
def test_full_pinned_layer_attention_component_all_slots(arrays,layer,K):
    data,models=arrays;returned=[]
    def progress(label,value,counts):
        value._public_unjustified=0.
        for earlier in returned:earlier._public_callback_relation=0.
        if label=='attention normalized output':returned.append(value)
    started=time.monotonic();owner,ops,refresh,sources=setup(arrays,layer,K,progress)
    original=[v.data.copy() for v in sources]
    for v in sources:v._public_borrowed='preserved'
    x=data[f'layer{layer}_K{K}_input'];gamma=data[f'layer{layer}_gamma'];beta=data[f'layer{layer}_beta']
    native=reference.compute(x,models[layer],gamma,beta,MANIFEST,approximate=False,layer=layer)
    native_error=float(np.max(np.abs(native['output']-data[f'layer{layer}_K{K}_native'])))
    assert native_error<1e-11
    oracle=reference.compute(x,models[layer],gamma,beta,MANIFEST,approximate=True,layer=layer)
    result=owner.evaluate(sources);T=math.ceil(K/16);assert len(result.outputs)==T and tuple(returned)==result.outputs
    maximum=0.;expected_level=9 if layer<6 else 8
    for index,output in enumerate(result.outputs):
        start=index*16;valid=min(16,K-start);expected=np.zeros((16,4096));expected[:valid,:768]=oracle['output'][start:start+valid]
        error=float(np.max(np.abs(output.data-expected.reshape(-1))));maximum=max(maximum,error)
        assert np.isfinite(output.data).all() and error<1e-8,(layer,K,error)
        assert output.level==expected_level and type(output._ckks_nsd)is int and output._ckks_nsd==1
        assert result.original_positions[index]==tuple(map(int,data[f'K{K}_positions'][start:start+valid]))
        assert not any(name.startswith('_public_') for name in vars(output))
    for source,saved in zip(sources,original):np.testing.assert_array_equal(source.data,saved);assert source._public_borrowed=='preserved'
    assert not owner._active and not owner._sources and not owner._retained
    expected_B2=sum(row['softmax']['plan']['refresh_count'] for row in result.metadata['outputs'])
    assert sum(bound==2. for _,_,bound in refresh.calls)==expected_B2
    assert sum(bound==128. for _,_,bound in refresh.calls)==(0 if layer==0 else T)
    assert result.metadata['internal_frame_tiles']==8
    assert len([x for x in result.metadata['entry_metadata'] if x['kind']=='subtract_zero'])==(0 if layer==0 else 8-T)
    assert not result.metadata['range_verified'] and not result.metadata['repeated_refresh_error_verified']
    counts=Counter(ops.calls);counts['pt_products']=ops.public_multiplies+counts['initial_affine']
    counts['rescales']=counts['weighted_sum']+counts['multiply']+counts['initial_affine']
    s=MANIFEST['softmax'][str(layer)]['squarings'];n=MANIFEST['normalization_reciprocal_by_public_K'][str(K)]['iterations']
    powers,terms=power_products(MANIFEST['norms'][str(layer)+'.attention']);maps=16+2*T
    expected_counts=dict(rotate=86*maps+T*(8656+28+14*s+42),
        multiply=T*(256+18+2*n*s+powers+2),
        pt_products=1535*maps+T*(7168+17+3*s+4+terms),
        rescales=24*maps+T*(768+27+(2*n+3)*s+powers+7))
    assert all(counts[k]==v for k,v in expected_counts.items()),(dict(counts),expected_counts)
    report=dict(passed=True,scope='Independent per-layer native-input component; actual module loops and identity refresh; no HE/full12 composition.',
        layer=layer,K=K,physical_slots_checked=T*65536,ideal_gate=1e-8,native_oracle_gate=1e-11,
        ideal_vs_dense_polynomial_max_abs=maximum,numpy_exact_vs_pinned_native_max_abs=native_error,
        polynomial_vs_native_diagnostic=float(np.max(np.abs(oracle['output']-data[f'layer{layer}_K{K}_native']))),
        domain_observations=oracle['domain_observations'],primitive_counts=dict(counts),identity_refresh_calls=len(refresh.calls),
        actual_metadata=result.metadata,elapsed_s=time.monotonic()-started)
    (HERE/f'development-layer{layer}-K{K}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

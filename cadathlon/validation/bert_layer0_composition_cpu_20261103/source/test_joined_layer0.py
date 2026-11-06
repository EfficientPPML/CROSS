"""Actual owner-loop ideal layer0 composition; no ciphertext or refresh execution."""
from collections import Counter
from pathlib import Path
import hashlib,json,math,os,time
import numpy as np
import pytest
from test_initial_ideal import ideal_primitives
from test_softmax_tile import refresh_and_tracked
from test_first_attention_ideal import primitives
from ideal_primitives import Arithmetic,Value,IdealLinear,DELTA
from ideal_refresh import IdentityRefresh as BaseRefresh
from encrypted_bert_first_attention import BertFirstAttention
from encrypted_bert_attention_tile import BertAttentionTile
from encrypted_bert_v2_ffn import BertV2FFNTile,TENSORS
from encrypted_bert_v2_row_norm import V2RowLayerNorm
from first_attention_contract import make_contract
import ffn_public_fixtures,joined_reference
HERE=Path(__file__).resolve().parent;RAW=(HERE/'candidate_manifest.json').read_bytes();MANIFEST=json.loads(RAW)

class IdentityRefresh(BaseRefresh):
    """One fixed public engine contract; each owner explicitly supplies its bound."""
    def refresh(self,value,*,public_bound):
        if type(public_bound)is not float or public_bound not in (2.,16.,128.) or value.level<2:
            raise ValueError('fixed B2/B16/B128 contracts and two input levels required')
        self.calls.append((value.level,value._ckks_scale,public_bound))
        output=Value(value.data,self.output_level,math.nextafter(DELTA,math.inf if len(self.calls)%2 else -math.inf))
        if self.inject is not None:self.inject(value,output)
        return output

@pytest.fixture(scope='module')
def joined_arrays():
    first_path=Path(os.environ.get('BERT_FIRST_ATTENTION_FIXTURE','/tmp/bert-first-attention-owner-20261103/client_fixtures.npz'))
    prep=json.loads((HERE/'fixture_preparation.json').read_text())
    assert hashlib.sha256(first_path.read_bytes()).hexdigest()==prep['fixture_npz_sha256']
    with np.load(first_path,allow_pickle=False) as source:first={k:source[k].copy() for k in source.files}
    ffn,pins=ffn_public_fixtures.load()
    native_path=Path(os.environ.get('BERT_NATIVE_BLOCK_FIXTURE',HERE/'native_block_fixtures.npz'))
    native_prep=json.loads((HERE/'native_block_preparation.json').read_text())
    assert native_prep['passed'] and native_prep['source_files_unchanged']
    assert hashlib.sha256(native_path.read_bytes()).hexdigest()==native_prep['output_fixture_sha256']
    with np.load(native_path,allow_pickle=False) as source:native={k:source[k].copy() for k in source.files}
    (HERE/'joined_fixture_pins.json').write_text(json.dumps(pins,indent=2,allow_nan=False)+'\n')
    return first,ffn,native

def build(first,ffn,K):
    ops=Arithmetic();linear=IdealLinear(ops);refresh=IdentityRefresh(ops)
    model={kind:{name:first[kind+'_'+name] for name in ('weight','bias')} for kind in ('query','key','value','output')}
    graph=set(BertAttentionTile.required_rotation_indices())|set(V2RowLayerNorm.required_rotation_indices())
    dense=set()
    for row in model.values():dense.update(linear.required_rotation_indices(row['weight'],linear.layout))
    for name in ('intermediate','output'):dense.update(linear.required_rotation_indices(ffn[name+'.weight'],linear.layout))
    graph.update(dense);all_keys=graph|set(refresh.plan['rotations'])
    cache=ops.ctx._param_cache
    for key in all_keys:cache.raw_rotation_keys[key]={'a':object(),'b':object()};cache.coef_maps[key]=object()
    positions=tuple(map(int,first[f'K{K}_original_positions']))
    attention=BertFirstAttention(ops,linear,refresh,model,make_contract(positions,'c'*64,'d'*32),
                                 RAW,first['gamma'],first['beta'])
    tiles=tuple(tuple(positions[start:start+16]) for start in range(0,K,16))
    tails=tuple(BertV2FFNTile(ops,linear,refresh,{n:ffn[n] for n in TENSORS},RAW,ffn['gamma'],ffn['beta'],
                             original_positions=rows) for rows in tiles)
    frame=np.zeros((128,4096));frame[:,:768]=first[f'K{K}_embeddings']
    sources=tuple(Value(row.reshape(-1)) for row in frame.reshape(8,16,4096))
    return attention,tails,ops,refresh,sources,model,dict(graph=sorted(graph),dense=sorted(dense),
        bootstrap=list(refresh.plan['rotations']),union=sorted(all_keys))

@pytest.mark.parametrize('K',(15,128))
def test_complete_layer0_uses_actual_attention_outputs_and_full_physical_slots(joined_arrays,K):
    start=time.monotonic();first,ffn,native=joined_arrays
    attention,tails,ops,refresh,sources,model,inventory=build(first,ffn,K)
    assert inventory['dense']==[1,64,61760,64064]
    assert len(inventory['union'])==130
    assert len(inventory['graph'])==92 and len(inventory['bootstrap'])==106
    before=[v.data.copy() for v in sources]
    embeddings=first[f'K{K}_embeddings'][:K]
    expected=joined_reference.compute(embeddings,model,first['gamma'],first['beta'],ffn,MANIFEST,approximate=True)
    exact=joined_reference.compute(embeddings,model,first['gamma'],first['beta'],ffn,MANIFEST,approximate=False)
    native_error=float(np.max(np.abs(exact['output']-native[f'K{K}_native_block'])))
    assert native_error<1e-11
    result=attention.evaluate(sources)
    actual_inputs=tuple(result.outputs)
    assert len(actual_inputs)==len(tails)==math.ceil(K/16)
    attention_error=0.;final_error=0.;metadata=[]
    for index,(value,tail) in enumerate(zip(actual_inputs,tails)):
        begin=index*16;valid=min(16,K-begin)
        first_oracle=np.zeros((16,4096));first_oracle[:valid,:768]=expected['attention'][begin:begin+valid]
        attention_error=max(attention_error,float(np.max(np.abs(value.data-first_oracle.reshape(-1)))))
        assert value is result.outputs[index] and value.level==9
        original=value.data.copy();fingerprint=attention._fingerprint(value)
        # This is the actual returned object, with its real recorded level and
        # scale. Native or independent polynomial activation arrays never enter.
        final=tail.evaluate_tile(value)
        assert attention._fingerprint(value)==fingerprint
        np.testing.assert_array_equal(value.data,original)
        oracle=np.zeros((16,4096));oracle[:valid,:768]=expected['output'][begin:begin+valid]
        difference=float(np.max(np.abs(final.output.data-oracle.reshape(-1))));final_error=max(final_error,difference)
        assert np.isfinite(final.output.data).all() and difference<1e-8
        assert final.output.level==13 and type(final.output._ckks_nsd)is int and final.output._ckks_nsd==1
        assert final.original_positions==result.original_positions[index]
        assert np.all(final.output.data.reshape(16,4096)[:valid,768:]==0)
        assert np.all(final.output.data.reshape(16,4096)[valid:]==0)
        assert not any(name.startswith('_public_') for name in vars(final.output))
        assert not tail._active and tail._source is None and not tail._retained
        metadata.append(tail.last_run)
    assert attention_error<1e-8
    for source,saved in zip(sources,before):np.testing.assert_array_equal(source.data,saved)
    assert not attention._active and not attention._sources and not attention._retained
    counts=Counter(ops.calls);counts['pt_products']=ops.public_multiplies+counts['initial_affine']
    counts['rescales']=counts['weighted_sum']+counts['multiply']+counts['initial_affine']
    expected_counts=({'rotate':75408,'multiply':3976,'pt_products':169704,'rescales':10592} if K==128
        else {'rotate':10630,'multiply':467,'pt_products':42703,'rescales':1630})
    assert all(counts[key]==count for key,count in expected_counts.items()),dict(counts)
    bounds=Counter(bound for _,_,bound in refresh.calls)
    assert bounds=={2.:5*len(tails),128.:len(tails),16.:len(tails)}
    pair_bytes=912*1024**2
    report=dict(passed=True,scope='Ideal complete pinned layer0 using actual owner outputs; no HE, real refresh, or12-layer readiness.',
        K=K,physical_slots_checked=65536*len(tails),attention_output_objects_passed_directly=True,
        native_attention_substitution=False,ideal_absolute_gate=1e-8,native_reference_gate=1e-11,
        ideal_attention_max_abs=attention_error,ideal_complete_block_max_abs=final_error,
        exact_dense_vs_native_full_block_max_abs=native_error,
        polynomial_vs_native_full_block_max_abs=float(np.max(np.abs(expected['output']-native[f'K{K}_native_block']))),
        native_block_accuracy_gate_invented=False,domain_observations=expected['domain_observations'],
        fixture_attention_abs_max=float(np.max(np.abs(expected['attention']))),
        fixture_GELU_abs_max=float(np.max(np.abs(expected['activated']))),
        primitive_counts=dict(counts),identity_refresh_counts={str(int(k)):v for k,v in bounds.items()},
        public_key_inventory=inventory,raw_public_rotation_key_bytes=len(inventory['union'])*pair_bytes,
        whole_process_peak_memory_or_latency_inferred=False,attention_metadata=result.metadata,ffn_metadata=metadata,
        elapsed_s=time.monotonic()-start)
    (HERE/f'layer0-K{K}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

@pytest.mark.parametrize('bound',(True,2,16,128,0.,1.,4.,float('nan'),float('inf')))
def test_identity_refresh_never_selects_an_undeclared_bound(bound):
    ops=Arithmetic();refresh=IdentityRefresh(ops);source=Value(np.ones(65536),2)
    with pytest.raises(ValueError):refresh.refresh(source,public_bound=bound)
    assert not refresh.calls and not ops.calls

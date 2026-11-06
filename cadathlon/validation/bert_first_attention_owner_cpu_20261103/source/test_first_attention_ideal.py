"""Full pinned-weight graph with ideal slots/identity refresh; no HE evidence."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import hashlib,json,math,os,sys,time
import numpy as np
import pytest
from test_initial_ideal import ideal_primitives
from test_softmax_tile import refresh_and_tracked
from ideal_refresh import IdentityRefresh,install_public_bootstrap_plan
from ideal_primitives import Arithmetic,Value,IdealLinear,install_matrix_class,DELTA
from encrypted_bert_attention_tile import BertAttentionTile
from encrypted_bert_v2_row_norm import V2RowLayerNorm
from encrypted_chained_streaming_linear import ChainedStreamingLinear
from encrypted_streaming_linear import RowTileLayout
from encrypted_bert_first_attention import BertFirstAttention
from first_attention_contract import make_contract
import reference
HERE=Path(__file__).resolve().parent
RAW=(HERE/'candidate_manifest.json').read_bytes();MANIFEST=json.loads(RAW)

@pytest.fixture(autouse=True)
def primitives(refresh_and_tracked,monkeypatch):
    install_matrix_class(monkeypatch)
    install_public_bootstrap_plan(monkeypatch)

@pytest.fixture(scope='module')
def arrays():
    fixture=Path(os.environ.get('BERT_FIRST_ATTENTION_FIXTURE',HERE/'client_fixtures.npz'))
    prep=json.loads((HERE/'fixture_preparation.json').read_text());raw=fixture.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==prep['fixture_npz_sha256']
    with np.load(fixture,allow_pickle=False) as data:result={k:data[k] for k in data.files}
    for k,v in result.items():assert hashlib.sha256(v.tobytes()).hexdigest()==prep['arrays'][k]['sha256']
    return result

def setup(arrays,K,*,progress=None):
    ops=Arithmetic();refresher=IdentityRefresh(ops)
    keys=set(BertAttentionTile.required_rotation_indices())|set(V2RowLayerNorm.required_rotation_indices())|{1,64,64064}
    cache=ops.ctx._param_cache
    for key in keys:cache.raw_rotation_keys[key]={'a':object(),'b':object()};cache.coef_maps[key]=object()
    model={kind:{name:arrays[kind+'_'+name] for name in ('weight','bias')} for kind in ('query','key','value','output')}
    positions=tuple(map(int,arrays[f'K{K}_original_positions']))
    contract=make_contract(positions,'c'*64,'d'*32)
    linear=IdealLinear(ops)
    owner=BertFirstAttention(ops,linear,refresher,model,contract,RAW,arrays['gamma'],arrays['beta'],progress=progress)
    data=np.zeros((128,4096));data[:,:768]=arrays[f'K{K}_embeddings']
    sources=[Value(row.reshape(-1)) for row in data.reshape(8,16,4096)]
    return owner,ops,refresher,sources,model

@pytest.mark.parametrize('K',(15,128))
def test_complete_pinned_first_attention_matches_independent_dense_polynomial(arrays,K):
    returned=[]
    def progress(label,value,counts):
        value._public_callback_bound='not justified'
        for earlier in returned:earlier._public_callback_relation='not justified'
        if label=='attention normalized output':returned.append(value)
    started=time.monotonic();owner,ops,refresh,sources,model=setup(arrays,K,progress=progress)
    before=[v.data.copy() for v in sources]
    x=arrays[f'K{K}_embeddings'][:K]
    native=reference.compute(x,model,arrays['gamma'],arrays['beta'],MANIFEST,approximate=False)
    native_error=float(np.max(np.abs(native['output']-arrays[f'K{K}_native'])))
    assert native_error<1e-11
    oracle=reference.compute(x,model,arrays['gamma'],arrays['beta'],MANIFEST,approximate=True)
    result=owner.evaluate(sources);assert len(result.outputs)==math.ceil(K/16)
    assert tuple(returned)==result.outputs
    maximum=0.
    for index,output in enumerate(result.outputs):
        start=index*16;valid=min(16,K-start);expected=np.zeros((16,4096));expected[:valid,:768]=oracle['output'][start:start+valid]
        error=float(np.max(np.abs(output.data-expected.reshape(-1))));maximum=max(maximum,error)
        assert np.isfinite(output.data).all() and error<1e-8
        assert output.level==9 and type(output._ckks_nsd)is int and output._ckks_nsd==1
        assert result.original_positions[index]==tuple(map(int,arrays[f'K{K}_original_positions'][start:start+valid]))
        assert np.all(output.data.reshape(16,4096)[:valid,768:]==0)
        assert np.all(output.data.reshape(16,4096)[valid:]==0)
        assert not any(name.startswith('_public_') for name in vars(output))
    for value,saved in zip(sources,before):np.testing.assert_array_equal(value.data,saved)
    assert not owner._active and not owner._sources and not owner._retained
    assert len(refresh.calls)==5*math.ceil(K/16)
    assert not result.metadata['range_verified'] and not result.metadata['repeated_refresh_error_verified']
    counts=Counter(ops.calls);counts['pt_products']=ops.public_multiplies+counts['initial_affine']
    counts['rescales']=counts['weighted_sum']+counts['multiply']+counts['initial_affine']
    expected_counts=({'rotate':73120,'multiply':3336,'pt_products':106848,'rescales':8432} if K==128
        else {'rotate':10344,'multiply':387,'pt_products':34846,'rescales':1360})
    assert all(counts[key]==count for key,count in expected_counts.items())
    report=dict(passed=True,scope='Ideal full-slot actual module composition with identity refresh; no ciphertext/HE precision claim.',
        K=K,tiles=len(result.outputs),physical_slots_checked=65536*len(result.outputs),ideal_polynomial_absolute_gate=1e-8,
        ideal_vs_independent_polynomial_max_abs=maximum,numpy_exact_vs_pinned_native_max_abs=native_error,
        polynomial_vs_native_max_abs=float(np.max(np.abs(oracle['output']-arrays[f'K{K}_native']))),
        native_block_accuracy_gate_invented=False,domain_observations=oracle['domain_observations'],
        primitive_counts=dict(counts),identity_refresh_calls=len(refresh.calls),actual_metadata=result.metadata,elapsed_s=time.monotonic()-started)
    (HERE/f'development-K{K}.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

@pytest.mark.parametrize('mutation',('arithmetic','linear_arithmetic','norm','refresher','model','contract','positions','context_secret','scale'))
def test_before_entry_owner_changes_reject_without_primitive(arrays,mutation):
    owner,ops,refresh,sources,model=setup(arrays,15)
    if mutation=='arithmetic':owner.arithmetic=Arithmetic()
    elif mutation=='linear_arithmetic':owner.linear.arithmetic=Arithmetic()
    elif mutation=='norm':owner.norm=object()
    elif mutation=='refresher':owner.refresher=object()
    elif mutation=='model':owner.model=dict(owner.model)
    elif mutation=='contract':owner.contract['key_count']=16
    elif mutation=='positions':owner.positions=((1,)*15,)
    elif mutation=='context_secret':ops.ctx.secret_key=object()
    else:ops.scale*=2
    with pytest.raises((ValueError,TypeError,AttributeError)):owner.evaluate(sources)
    assert not ops.calls and not refresh.calls

@pytest.mark.parametrize('count',(0,7,9))
def test_wrong_input_count_fails_before_arithmetic(arrays,count):
    owner,ops,refresh,sources,model=setup(arrays,15)
    with pytest.raises(ValueError):owner.evaluate((sources*2)[:count])
    assert not ops.calls

def test_constructor_copies_coefficients_without_input_mutation(arrays):
    owner,ops,refresh,sources,model=setup(arrays,15)
    for name,row in owner.model.items():
        for key,array in row.items():
            np.testing.assert_array_equal(array,model[name][key]);assert not array.flags.writeable
            with pytest.raises(ValueError):array.flags.writeable=True

def test_callback_cancellation_releases_state_and_preserves_borrowed_objects(arrays,monkeypatch):
    error=KeyboardInterrupt('stop after first map')
    def progress(*args):raise error
    owner,ops,refresh,sources,model=setup(arrays,15,progress=progress)
    # A cheap arithmetic map stand-in isolates cancellation ownership; full
    # real chained-loop composition is exercised in the two tests above.
    monkeypatch.setattr(owner.linear,'linear_tile',lambda value,*a,**k:Value(np.zeros(65536),48,DELTA))
    with pytest.raises(KeyboardInterrupt) as caught:owner.evaluate(sources)
    assert caught.value is error and owner.ctx is ops.ctx
    assert not owner._active and not owner._sources and not owner._retained and owner.last_run is None
    assert not refresh.calls

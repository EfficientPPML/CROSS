"""All-slot ideal complete softmax with frozen per-layer rules; no HE execution."""
from pathlib import Path
from collections import Counter
import json,math
import numpy as np
import pytest
from test_split_initial import primitives,ideal_primitives,refresh_and_tracked,RAW,fixture
from ideal_primitives import Arithmetic,Value,DELTA
from ideal_refresh import IdentityRefresh
from encrypted_score_reduction import ScoreReductionLayout
from encrypted_bert_layer_softmax import BertLayerSoftmax
MANIFEST=json.loads(RAW)

def setup(layer,K,valid=16,progress=None):
    ops=Arithmetic();refresh=IdentityRefresh(ops);keys=np.arange(128)
    mask=np.array([((keys+13*row+17)%128)<K for row in range(16)])
    layout=ScoreReductionLayout(mask,valid_rows=valid)
    owner=BertLayerSoftmax(ops,layout,RAW,refresh,layer=layer,
        original_positions=tuple(range(128-valid,128)),progress=progress)
    return owner,ops,refresh,layout

def independent(data,owner,layout):
    grid=data.reshape(16,4096)[:,:2048].reshape(16,128,16)[:,:,:12]
    mask=layout.key_mask[:,:,None];count=np.maximum(layout.counts,1)[:,None,None]
    mean=np.sum(np.where(mask,grid,0.),axis=1,keepdims=True)/count
    centered=grid-mean;valid=centered[np.broadcast_to(mask,centered.shape)]
    lo,hi=owner.contract.centered_domain
    assert valid.min()>=lo and valid.max()<=hi
    t=np.where(mask,centered/owner.contract.normalized_divisor,0.)
    coeff=np.polynomial.chebyshev.cheb2poly(MANIFEST['base_exp']['coefficients'])
    base=np.polynomial.polynomial.polyval(t,coeff)*mask
    denominator=np.sum(base,axis=1,keepdims=True)/count
    denominator[layout.valid_rows:]=1.
    assert np.all((denominator>=.5)&(denominator<=2.))
    seed=MANIFEST['initial_reciprocal']['seed'];inverse=denominator*(-seed*seed)+2*seed
    for _ in range(1,6):inverse*=2-denominator*inverse
    probability=base*inverse/count
    rule=MANIFEST['normalization_reciprocal_by_public_K'][str(int(layout.counts[0]))]
    observations=[]
    for index in range(owner.contract.squarings):
        square=probability*probability;denominator=square.sum(axis=1,keepdims=True)
        denominator[layout.valid_rows:]=1.
        lo,hi=rule['domain'];assert np.all((denominator>=lo)&(denominator<=hi))
        observations.append((float(denominator.min()),float(denominator.max())))
        seed=rule['seed'];inverse=denominator*(-seed*seed)+2*seed
        for _ in range(1,rule['iterations']):inverse*=2-denominator*inverse
        probability=square*inverse
    output=np.zeros((16,4096));output[:,:2048].reshape(16,128,16)[:,:,:12]=probability
    return output.reshape(-1),observations

@pytest.mark.parametrize('layer',(0,1,4))
@pytest.mark.parametrize('K',range(1,129))
def test_every_public_K_for_all_selected_squaring_counts(layer,K):
    valid=(1,7,16)[K%3];owner,ops,refresh,layout=setup(layer,K,valid)
    data=fixture(owner,layout,K);source=Value(data,45 if layer==0 else 21,math.nextafter(DELTA,math.inf))
    saved=data.copy();source._public_borrowed='keep'
    # Nonzero invalid refreshed slots are not assumed zero. Every subsequent
    # round's input mask and the mandatory final refresh mask remain real ops.
    refresh.inject=lambda original,result:result.data.__setitem__(layout.input_mask==0,.125)
    expected,domains=independent(data,owner,layout)
    output,stages=owner.evaluate_tile(source,capture_checkpoints=True)
    np.testing.assert_allclose(output.data,expected,atol=3e-13,rtol=0)
    assert np.all(output.data[layout.input_mask==0]==0) and output.level==24
    assert not any(n.startswith('_public_') for value in stages.values() for n in vars(value))
    np.testing.assert_array_equal(source.data,saved);assert source._public_borrowed=='keep'
    plan=owner.last_run['plan'];assert len(refresh.calls)==plan['refresh_count']
    assert all(bound==2. for _,_,bound in refresh.calls)
    assert plan['selected_squarings']==MANIFEST['softmax'][str(layer)]['squarings']
    assert plan['initial_normalized_t_refresh']==(layer!=0)
    assert plan['initial_output_level']==(23 if layer==0 else 6)
    assert [row['level'] for row in owner.last_run['actual_metadata'][-2:]]==[25,24]
    n=owner.round.contract.updates;s=owner.contract.squarings;counts=Counter(ops.calls)
    assert counts['rotate']==28+14*s and counts['multiply']==18+2*n*s
    assert ops.public_multiplies+counts['initial_affine']==17+3*s
    assert counts['weighted_sum']+counts['multiply']+counts['initial_affine']==27+(2*n+3)*s
    assert not owner._active and owner._source is None and not owner._retained and not owner._captured

@pytest.mark.parametrize('layer',range(12))
def test_every_layer_actual_endpoint_contract_and_continuation_depth(layer):
    owner,ops,refresh,layout=setup(layer,3,7);data=np.full(65536,-2048.)
    lo,hi=owner.contract.centered_domain
    for row in range(7):
        keys=np.flatnonzero(layout.key_mask[row])
        for head in range(12):
            endpoint=lo if head%2 else hi
            data[row*4096+keys*16+head]=[endpoint,-endpoint/2,-endpoint/2]
    expected,_=independent(data,owner,layout)
    output=owner.evaluate_tile(Value(data,45 if layer==0 else 21))
    np.testing.assert_allclose(output.data,expected,atol=3e-13,rtol=0)
    assert owner.continuation_levels==(17 if layer<6 else 18)
    assert output.level==24 and owner.last_run['plan']['final_refresh_mask']

@pytest.mark.parametrize('layer',range(12))
def test_all1536_public_level_paths_keep_capacity(layer):
    for K in range(1,129):
        owner,ops,refresh,layout=setup(layer,K,1)
        plan=owner.level_plan(45 if layer==0 else 21)
        assert plan['output_level']==24 and plan['continuation_levels_after_mask'] in (17,18)
        assert all(row['output_level']>=2 for row in plan['normalized_rounds'])
        assert not ops.calls and not refresh.calls

@pytest.mark.parametrize('layer,level',((0,21),(0,44),(1,45),(1,20),(11,22)))
def test_unpublished_entry_level_rejects_before_work(layer,level):
    owner,ops,refresh,layout=setup(layer,15,7)
    with pytest.raises(ValueError):owner.evaluate_tile(Value(np.zeros(65536),level))
    assert not ops.calls and not refresh.calls
